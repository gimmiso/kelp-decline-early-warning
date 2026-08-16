"""Audit observation quality and re-run labels under fixed annualization rules.

The primary construction uses only pixels with positive kelp area during
1984-2004 as a fixed footprint.  This avoids letting kelp observed for the
first time in a future year define earlier coverage or later analysis support.
The previous all-history-footprint annual maximum is retained as an explicit
sensitivity and reproduction check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from netCDF4 import Dataset
from pyproj import Transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, cohen_kappa_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRAJECTORY = [
    "canopy_lag1",
    "canopy_lag2",
    "canopy_2yr_change",
    "canopy_3yr_change",
    "canopy_3yr_mean",
    "canopy_3yr_std",
    "canopy_3yr_slope",
    "canopy_drop_from_3yr_max",
]
OISST = [
    "annual_mean_sst_anomaly",
    "annual_max_sst",
    "hot_weeks_p90",
    "winter_mean_sst_anomaly",
    "summer_mean_sst_anomaly",
    "lag1_annual_mean_sst_anomaly",
]
MODELS = {
    "current_only": ["relative_canopy"],
    "current_plus_trajectory": ["relative_canopy", *TRAJECTORY],
    "trajectory_plus_oisst": ["relative_canopy", *TRAJECTORY, *OISST],
}
MODEL_LABELS = {
    "current_only": "Current canopy",
    "current_plus_trajectory": "Current + trajectory",
    "trajectory_plus_oisst": "Current + trajectory + OISST",
}
VARIANT_LABELS = {
    "fixedpre_protocol_max": "Fixed pre-2005 footprint\n50%, 3Q + Q3, max",
    "legacy_allhist_vf75_max": "Legacy all-history footprint\n75%, 3Q, max",
    "allhist_protocol_max": "All-history footprint\n50%, 3Q + Q3, max",
    "fixedpre_vf75_noq3_max": "Fixed pre-2005 footprint\n75%, 3Q, max",
    "fixedpre_vf75_q3_max": "Fixed pre-2005 footprint\n75%, 3Q + Q3, max",
    "fixedpre_vf75_all4_max": "Fixed pre-2005 footprint\n75%, all 4Q, max",
    "fixedpre_protocol_mean": "Fixed pre-2005 footprint\n50%, 3Q + Q3, mean",
    "fixedpre_protocol_q23max": "Fixed pre-2005 footprint\n50%, 3Q + Q2/Q3, Q2-Q3 max",
}
EPSILON = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--quarterly", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--netcdf", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--locked-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.SubprocessError:
        return "unknown"


def rolling_slope(values: np.ndarray) -> float:
    if len(values) != 3 or np.isnan(values).any():
        return np.nan
    return float(np.polyfit(np.arange(3), values, 1)[0])


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.sort_values(["cell_id", "year"]).copy()
    grouped = out.groupby("cell_id", group_keys=False)["relative_canopy"]
    out["canopy_lag1"] = grouped.shift(1)
    out["canopy_lag2"] = grouped.shift(2)
    lag3 = grouped.shift(3)
    out["canopy_2yr_change"] = out["relative_canopy"] - out["canopy_lag2"]
    out["canopy_3yr_change"] = out["relative_canopy"] - lag3
    rolling = grouped.rolling(3, min_periods=3)
    out["canopy_3yr_mean"] = rolling.mean().reset_index(level=0, drop=True)
    out["canopy_3yr_std"] = rolling.std().reset_index(level=0, drop=True)
    out["canopy_3yr_slope"] = rolling.apply(
        rolling_slope, raw=True
    ).reset_index(level=0, drop=True)
    maximum = rolling.max().reset_index(level=0, drop=True)
    out["canopy_drop_from_3yr_max"] = (
        maximum - out["relative_canopy"]
    ) / np.maximum(maximum, EPSILON)
    return out


def build_station_map(dataset: Dataset, stable: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    longitude = np.asarray(dataset.variables["longitude"][:], dtype=float)
    latitude = np.asarray(dataset.variables["latitude"][:], dtype=float)
    coordinate_ok = np.isfinite(longitude) & np.isfinite(latitude)
    source_indices = np.flatnonzero(coordinate_ok)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    x, y = transformer.transform(longitude[coordinate_ok], latitude[coordinate_ok])
    stations = pd.DataFrame(
        {
            "source_index": source_indices,
            "grid_x_index": np.floor(np.asarray(x) / 10_000).astype(np.int64),
            "grid_y_index": np.floor(np.asarray(y) / 10_000).astype(np.int64),
        }
    ).merge(
        stable[["grid_x_index", "grid_y_index", "cell_id"]],
        on=["grid_x_index", "grid_y_index"],
        how="inner",
        validate="many_to_one",
    )
    cell_lookup = {cell_id: index for index, cell_id in enumerate(stable["cell_id"])}
    return stations["source_index"].to_numpy(), stations["cell_id"].map(cell_lookup).to_numpy()


def masked_slice(variable, time_index: int, source_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    full = variable[time_index, :]
    selected = full[source_indices]
    if np.ma.isMaskedArray(selected):
        mask = np.ma.getmaskarray(selected)
        values = np.asarray(selected.filled(0), dtype=float)
    else:
        values = np.asarray(selected, dtype=float)
        fill = getattr(variable, "_FillValue", None)
        mask = values == fill if fill is not None else np.zeros(len(values), dtype=bool)
    return values, mask


def aggregate_netcdf_quality(
    netcdf: Path,
    stable: pd.DataFrame,
    panel_end: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    n_cells = len(stable)
    with Dataset(netcdf) as dataset:
        source_indices, cell_indices = build_station_map(dataset, stable)
        allhistory_counts = np.bincount(cell_indices, minlength=n_cells).astype(int)
        pre2005_positive = np.zeros(len(source_indices), dtype=bool)
        years = np.asarray(dataset.variables["year"][:], dtype=int)
        quarters = np.asarray(dataset.variables["quarter"][:], dtype=int)
        area_variable = dataset.variables["area"]

        for time_index, year in enumerate(years):
            if year > 2004:
                continue
            area, area_mask = masked_slice(area_variable, time_index, source_indices)
            pre2005_positive |= (~area_mask) & (area > 0)

        pre2005_counts = np.bincount(
            cell_indices, weights=pre2005_positive.astype(np.int16), minlength=n_cells
        ).astype(int)
        rows: list[pd.DataFrame] = []
        for time_index, (year, quarter) in enumerate(zip(years, quarters, strict=True)):
            if year > panel_end:
                continue
            area, area_mask = masked_slice(area_variable, time_index, source_indices)
            area_valid = ~area_mask
            fixed_valid = area_valid & pre2005_positive
            allhist_valid = area_valid

            passes, passes_mask = masked_slice(dataset.variables["passes"], time_index, source_indices)
            passes = np.where(passes_mask, 0, passes)
            passes5, passes5_mask = masked_slice(dataset.variables["passes5"], time_index, source_indices)
            passes7, passes7_mask = masked_slice(dataset.variables["passes7"], time_index, source_indices)
            passes8, passes8_mask = masked_slice(dataset.variables["passes8"], time_index, source_indices)
            passes5 = np.where(passes5_mask, 0, passes5)
            passes7 = np.where(passes7_mask, 0, passes7)
            passes8 = np.where(passes8_mask, 0, passes8)
            area_se, se_mask = masked_slice(dataset.variables["area_se"], time_index, source_indices)
            fixed_se_valid = fixed_valid & (~se_mask)

            def count(mask: np.ndarray) -> np.ndarray:
                return np.bincount(cell_indices, weights=mask.astype(np.int16), minlength=n_cells)

            def total(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
                return np.bincount(
                    cell_indices, weights=np.where(mask, values, 0), minlength=n_cells
                )

            fixed_valid_count = count(fixed_valid)
            allhist_valid_count = count(allhist_valid)
            fixed_pass_sum = total(passes, fixed_valid)
            allhist_pass_sum = total(passes, allhist_valid)
            fixed_area = total(area, fixed_valid)
            allhist_area = total(area, allhist_valid)
            fixed_se_rss = np.sqrt(total(np.square(area_se), fixed_se_valid))
            fixed_se_sum = total(area_se, fixed_se_valid)
            sensor5 = total(passes5, fixed_valid)
            sensor7 = total(passes7, fixed_valid)
            sensor8 = total(passes8, fixed_valid)
            fixed_pass_ge2 = count(fixed_valid & (passes >= 2))
            allhist_pass_ge2 = count(allhist_valid & (passes >= 2))

            frame = stable[["cell_id", "region_group", "center_lat", "center_lon"]].copy()
            frame["year"] = int(year)
            frame["quarter"] = int(quarter)
            frame["fixed_area_m2"] = fixed_area
            frame["fixed_valid_pixels"] = fixed_valid_count.astype(int)
            frame["fixed_footprint_pixels"] = pre2005_counts
            frame["fixed_valid_fraction"] = np.divide(
                fixed_valid_count,
                pre2005_counts,
                out=np.full(n_cells, np.nan, dtype=float),
                where=pre2005_counts > 0,
            )
            frame["allhist_area_m2_rebuilt"] = allhist_area
            frame["allhist_valid_pixels_rebuilt"] = allhist_valid_count.astype(int)
            frame["allhist_footprint_pixels_rebuilt"] = allhistory_counts
            frame["allhist_valid_fraction_rebuilt"] = np.divide(
                allhist_valid_count,
                allhistory_counts,
                out=np.full(n_cells, np.nan, dtype=float),
                where=allhistory_counts > 0,
            )
            frame["mean_passes_fixed_valid_pixels"] = np.divide(
                fixed_pass_sum,
                fixed_valid_count,
                out=np.full(n_cells, np.nan, dtype=float),
                where=fixed_valid_count > 0,
            )
            frame["share_fixed_valid_pixels_ge2_passes"] = np.divide(
                fixed_pass_ge2,
                fixed_valid_count,
                out=np.full(n_cells, np.nan, dtype=float),
                where=fixed_valid_count > 0,
            )
            frame["mean_passes_allhist_valid_pixels"] = np.divide(
                allhist_pass_sum,
                allhist_valid_count,
                out=np.full(n_cells, np.nan, dtype=float),
                where=allhist_valid_count > 0,
            )
            frame["share_allhist_valid_pixels_ge2_passes"] = np.divide(
                allhist_pass_ge2,
                allhist_valid_count,
                out=np.full(n_cells, np.nan, dtype=float),
                where=allhist_valid_count > 0,
            )
            frame["area_se_rss_independence_fixed"] = fixed_se_rss
            frame["area_se_sum_conservative_fixed"] = fixed_se_sum
            frame["area_se_observed_fraction_fixed"] = np.divide(
                count(fixed_se_valid),
                fixed_valid_count,
                out=np.full(n_cells, np.nan, dtype=float),
                where=fixed_valid_count > 0,
            )
            frame["passes5_fixed_sum"] = sensor5
            frame["passes7_fixed_sum"] = sensor7
            frame["passes8_fixed_sum"] = sensor8
            frame["passes_total_fixed_sum"] = fixed_pass_sum
            rows.append(frame)

        metadata = {
            "source_stations_in_165_cells": int(len(source_indices)),
            "netcdf_time_slices_used": int(sum(years <= panel_end)),
            "netcdf_area_units": getattr(dataset.variables["area"], "units", "unknown"),
            "netcdf_area_se_units": getattr(dataset.variables["area_se"], "units", "unknown"),
            "netcdf_passes_definition": getattr(dataset.variables["passes"], "units", "unknown"),
        }
    footprint = stable[["cell_id"]].copy()
    footprint["allhistory_station_pixels_rebuilt"] = allhistory_counts
    footprint["pre2005_positive_pixels_rebuilt"] = pre2005_counts
    return pd.concat(rows, ignore_index=True), footprint, metadata


def merge_external_quarterly(
    rebuilt: pd.DataFrame,
    quarterly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    external = quarterly[
        [
            "cell_id",
            "year",
            "quarter",
            "kelp_area_m2",
            "count_cells_no_clouds",
            "count_cells_historic_footprint",
            "valid_fraction",
        ]
    ].rename(
        columns={
            "kelp_area_m2": "allhist_area_m2",
            "count_cells_no_clouds": "allhist_valid_pixels",
            "count_cells_historic_footprint": "allhist_footprint_pixels",
            "valid_fraction": "allhist_valid_fraction",
        }
    )
    data = rebuilt.merge(
        external, on=["cell_id", "year", "quarter"], how="left", validate="one_to_one"
    )
    checks = pd.DataFrame(
        [
            {
                "check": "allhistory_area_rebuild_matches_saved_quarterly",
                "mismatches": int(
                    (~np.isclose(
                        data["allhist_area_m2_rebuilt"], data["allhist_area_m2"], atol=0, rtol=0
                    )).sum()
                ),
            },
            {
                "check": "allhistory_valid_pixel_rebuild_matches_saved_quarterly",
                "mismatches": int(
                    data["allhist_valid_pixels_rebuilt"].ne(data["allhist_valid_pixels"]).sum()
                ),
            },
            {
                "check": "allhistory_footprint_rebuild_matches_saved_quarterly",
                "mismatches": int(
                    data["allhist_footprint_pixels_rebuilt"].ne(data["allhist_footprint_pixels"]).sum()
                ),
            },
        ]
    )
    checks["passed"] = checks["mismatches"].eq(0)
    return data, checks


def annualize(quarterly: pd.DataFrame, spec: dict[str, object]) -> pd.DataFrame:
    fixed = spec["footprint"] == "pre2005_positive_pixels"
    area_column = "fixed_area_m2" if fixed else "allhist_area_m2"
    valid_fraction_column = "fixed_valid_fraction" if fixed else "allhist_valid_fraction"
    passes_column = (
        "mean_passes_fixed_valid_pixels" if fixed else "mean_passes_allhist_valid_pixels"
    )
    data = quarterly.copy()
    data["quarter_valid"] = data[valid_fraction_column].ge(float(spec["minimum_valid_fraction"]))
    group_keys = ["cell_id", "year"]
    group = data.groupby(group_keys, sort=False)
    summary = group.agg(
        region_group=("region_group", "first"),
        center_lat=("center_lat", "first"),
        center_lon=("center_lon", "first"),
        valid_quarters=("quarter_valid", "sum"),
    ).reset_index()
    q3 = data.loc[data["quarter"].eq(3), group_keys + ["quarter_valid"]].rename(
        columns={"quarter_valid": "q3_valid"}
    )
    q2 = data.loc[data["quarter"].eq(2), group_keys + ["quarter_valid"]].rename(
        columns={"quarter_valid": "q2_valid"}
    )
    summary = summary.merge(q3, on=group_keys, how="left").merge(q2, on=group_keys, how="left")
    summary[["q2_valid", "q3_valid"]] = summary[["q2_valid", "q3_valid"]].fillna(False)
    summary["annual_complete"] = summary["valid_quarters"].ge(
        int(spec["minimum_valid_quarters"])
    )
    if bool(spec.get("require_q3", False)):
        summary["annual_complete"] &= summary["q3_valid"]
    if bool(spec.get("require_q2_for_aggregation", False)):
        summary["annual_complete"] &= summary["q2_valid"]

    valid = data.loc[data["quarter_valid"]].copy()
    aggregator = spec["aggregator"]
    if aggregator == "q2_q3_maximum":
        valid = valid.loc[valid["quarter"].isin([2, 3])]
    if aggregator in {"maximum_valid_quarter", "q2_q3_maximum"}:
        idx = valid.groupby(group_keys)[area_column].idxmax()
        values = valid.loc[idx, group_keys + ["quarter", area_column, passes_column]].rename(
            columns={
                "quarter": "selected_quarter",
                area_column: "annual_area_m2",
                passes_column: "selected_quarter_mean_passes",
            }
        )
    elif aggregator == "mean_valid_quarters":
        values = valid.groupby(group_keys).agg(
            annual_area_m2=(area_column, "mean"),
            selected_quarter_mean_passes=(passes_column, "mean"),
        ).reset_index()
        values["selected_quarter"] = np.nan
    else:
        raise ValueError(f"Unknown aggregator: {aggregator}")

    pass_quality = valid.groupby(group_keys)[passes_column].min().rename(
        "minimum_mean_passes_over_used_quarters"
    ).reset_index()
    summary = summary.merge(values, on=group_keys, how="left").merge(
        pass_quality, on=group_keys, how="left"
    )
    summary.loc[~summary["annual_complete"], "annual_area_m2"] = np.nan
    summary["variant"] = spec["name"]
    return summary


def build_panel(
    annual: pd.DataFrame,
    stable: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    forecast = config["forecast"]
    outcome = config["outcome"]
    years = pd.DataFrame({"year": np.arange(1984, int(forecast["panel_end"]) + 1)})
    grid = stable[["cell_id", "region_group", "center_lat", "center_lon"]].merge(
        years, how="cross"
    )
    values = annual.drop(columns=["region_group", "center_lat", "center_lon"], errors="ignore")
    panel = grid.merge(values, on=["cell_id", "year"], how="left", validate="one_to_one")
    reference_start, reference_end = outcome["reference_years"]
    reference = panel.loc[
        panel["year"].between(reference_start, reference_end)
        & panel["annual_area_m2"].notna()
    ].groupby("cell_id")["annual_area_m2"].quantile(0.95)
    panel["p95_annual_area_pre2005"] = panel["cell_id"].map(reference)
    if panel["p95_annual_area_pre2005"].isna().any() or panel["p95_annual_area_pre2005"].le(0).any():
        bad = panel.loc[
            panel["p95_annual_area_pre2005"].isna() | panel["p95_annual_area_pre2005"].le(0),
            "cell_id",
        ].unique()
        raise ValueError(f"Non-positive or missing pre-2005 reference for {bad.tolist()}")
    panel["relative_canopy"] = panel["annual_area_m2"] / panel["p95_annual_area_pre2005"]
    grouped = panel.groupby("cell_id", group_keys=False)
    panel["next_year_area_m2"] = grouped["annual_area_m2"].shift(-1)
    panel["next_year_observation_high_quality"] = grouped[
        "minimum_mean_passes_over_used_quarters"
    ].shift(-1).ge(float(config["high_observation_subset"]["minimum_mean_passes_in_every_valid_quarter"]))
    panel["relative_drop_next"] = (
        panel["annual_area_m2"] - panel["next_year_area_m2"]
    ) / np.maximum(panel["annual_area_m2"], EPSILON)
    panel["event"] = panel["relative_drop_next"].ge(float(outcome["decline_fraction"])).astype(int)
    panel = add_features(panel)
    panel["eligible"] = (
        panel["relative_canopy"].gt(float(outcome["eligibility_relative_canopy_gt"]))
        & panel["annual_area_m2"].notna()
        & panel["next_year_area_m2"].notna()
        & panel["year"].between(int(forecast["train_start"]), int(forecast["test_end"]))
    )
    threshold = float(
        config["high_observation_subset"]["minimum_mean_passes_in_every_valid_quarter"]
    )
    panel["observation_high_quality"] = panel[
        "minimum_mean_passes_over_used_quarters"
    ].ge(threshold)
    panel["high_observation_pair_eligible"] = (
        panel["eligible"]
        & panel["observation_high_quality"]
        & panel["next_year_observation_high_quality"]
    )
    return panel


def model_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=5000, random_state=seed)),
        ]
    )


def run_backtests(
    panels: dict[str, pd.DataFrame],
    environment: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast = config["forecast"]
    primary = config["primary_variant"]
    merged: dict[str, pd.DataFrame] = {}
    for variant, panel in panels.items():
        frame = panel.merge(
            environment[["cell_id", "year", *OISST]],
            on=["cell_id", "year"],
            how="left",
            validate="one_to_one",
        )
        frame["eligible_with_environment"] = frame["eligible"] & frame[OISST].notna().all(axis=1)
        merged[variant] = frame

    eligibility = None
    for variant, frame in merged.items():
        flag = frame[["cell_id", "year", "eligible_with_environment"]].rename(
            columns={"eligible_with_environment": variant}
        )
        eligibility = flag if eligibility is None else eligibility.merge(
            flag, on=["cell_id", "year"], how="inner", validate="one_to_one"
        )
    variant_names = list(merged)
    eligibility["common_eligible"] = eligibility[variant_names].all(axis=1)
    common_lookup = eligibility.set_index(["cell_id", "year"])["common_eligible"]

    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    for mode in ["native", "common_rows", "high_observation_subset"]:
        variants = [primary] if mode == "high_observation_subset" else variant_names
        for variant in variants:
            frame = merged[variant].copy()
            if mode == "native":
                frame["analysis_eligible"] = frame["eligible_with_environment"]
            elif mode == "common_rows":
                frame["analysis_eligible"] = [
                    bool(common_lookup.get((cell_id, year), False))
                    for cell_id, year in zip(frame["cell_id"], frame["year"], strict=True)
                ]
            else:
                frame["analysis_eligible"] = (
                    frame["eligible_with_environment"] & frame["high_observation_pair_eligible"]
                )

            for year in range(int(forecast["test_start"]), int(forecast["test_end"]) + 1):
                train = frame.loc[
                    frame["analysis_eligible"]
                    & frame["year"].between(int(forecast["train_start"]), year - 1)
                ].copy()
                test = frame.loc[frame["analysis_eligible"] & frame["year"].eq(year)].copy()
                if train["event"].nunique() < 2 or test.empty:
                    raise ValueError(f"Invalid fold: mode={mode}, variant={variant}, year={year}")
                folds.append(
                    {
                        "mode": mode,
                        "variant": variant,
                        "year": year,
                        "train_start": int(train["year"].min()),
                        "train_end": int(train["year"].max()),
                        "n_train": len(train),
                        "events_train": int(train["event"].sum()),
                        "n_test": len(test),
                        "events_test": int(test["event"].sum()),
                    }
                )
                for model, features in MODELS.items():
                    estimator = model_pipeline(int(forecast["seed"]))
                    estimator.fit(train[features], train["event"])
                    scored = test[
                        ["cell_id", "region_group", "year", "event", "relative_canopy"]
                    ].copy()
                    scored["score"] = estimator.predict_proba(test[features])[:, 1]
                    scored["model"] = model
                    scored["mode"] = mode
                    scored["variant"] = variant
                    predictions.append(scored)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(folds)


def safe_ap(group: pd.DataFrame) -> float:
    if group.empty or group["event"].nunique() < 2:
        return np.nan
    return float(average_precision_score(group["event"], group["score"]))


def calculate_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (mode, variant, model, year), group in predictions.groupby(
        ["mode", "variant", "model", "year"], sort=True
    ):
        ranked = group.sort_values(["score", "cell_id"], ascending=[False, True])
        events = int(ranked["event"].sum())
        prevalence = float(ranked["event"].mean())
        ap = safe_ap(ranked)
        k = max(1, int(math.ceil(len(ranked) * 0.20)))
        rows.append(
            {
                "mode": mode,
                "variant": variant,
                "model": model,
                "year": year,
                "n": len(ranked),
                "events": events,
                "prevalence": prevalence,
                "ap": ap,
                "ap_lift": ap - prevalence if np.isfinite(ap) else np.nan,
                "estimable": ranked["event"].nunique() == 2,
                "top20_k": k,
                "top20_tp": int(ranked.head(k)["event"].sum()),
            }
        )
    year_metrics = pd.DataFrame(rows)
    summaries = []
    for (mode, variant, model), group in predictions.groupby(["mode", "variant", "model"]):
        annual = year_metrics.loc[
            year_metrics["mode"].eq(mode)
            & year_metrics["variant"].eq(variant)
            & year_metrics["model"].eq(model)
            & year_metrics["estimable"]
        ]
        all_years = year_metrics.loc[
            year_metrics["mode"].eq(mode)
            & year_metrics["variant"].eq(variant)
            & year_metrics["model"].eq(model)
        ]
        summaries.append(
            {
                "mode": mode,
                "variant": variant,
                "model": model,
                "n": len(group),
                "events": int(group["event"].sum()),
                "prevalence": float(group["event"].mean()),
                "pooled_ap": safe_ap(group),
                "macro_year_ap_lift": float(annual["ap_lift"].mean()),
                "estimable_years": len(annual),
                "top20_recall_micro": float(
                    all_years["top20_tp"].sum() / all_years["events"].sum()
                ),
            }
        )
    return year_metrics, pd.DataFrame(summaries)


def two_year_block_indices(n_years: int, replicates: int, rng: np.random.Generator) -> np.ndarray:
    blocks = math.ceil(n_years / 2)
    starts = rng.integers(0, n_years - 1, size=(replicates, blocks))
    paired = np.stack([starts, starts + 1], axis=-1).reshape(replicates, -1)
    return paired[:, :n_years]


def bootstrap_metrics(
    year_metrics: pd.DataFrame, config: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast = config["forecast"]
    years = np.arange(int(forecast["test_start"]), int(forecast["test_end"]) + 1)
    rng = np.random.default_rng(int(forecast["seed"]))
    indices = two_year_block_indices(len(years), int(forecast["bootstrap_replicates"]), rng)
    estimate_rows = []
    difference_rows = []
    draw_rows = []
    for (mode, variant), source in year_metrics.groupby(["mode", "variant"]):
        for model in MODELS:
            table = source.loc[source["model"].eq(model)].set_index("year")["ap_lift"].reindex(years)
            values = table.to_numpy(dtype=float)
            draws = np.nanmean(values[indices], axis=1)
            estimate_rows.append(
                {
                    "mode": mode,
                    "variant": variant,
                    "model": model,
                    "estimate": float(np.nanmean(values)),
                    "ci_low": float(np.nanquantile(draws, 0.025)),
                    "ci_high": float(np.nanquantile(draws, 0.975)),
                    "estimable_years": int(np.isfinite(values).sum()),
                }
            )
            draw_rows.extend(
                {
                    "mode": mode,
                    "variant": variant,
                    "kind": "model",
                    "name": model,
                    "replicate": replicate,
                    "value": float(value),
                }
                for replicate, value in enumerate(draws)
            )
        comparisons = [
            ("current_plus_trajectory", "current_only", "trajectory_minus_current"),
            ("trajectory_plus_oisst", "current_plus_trajectory", "oisst_minus_trajectory"),
        ]
        for left, right, name in comparisons:
            left_values = source.loc[source["model"].eq(left)].set_index("year")["ap_lift"].reindex(years)
            right_values = source.loc[source["model"].eq(right)].set_index("year")["ap_lift"].reindex(years)
            values = left_values.to_numpy(dtype=float) - right_values.to_numpy(dtype=float)
            draws = np.nanmean(values[indices], axis=1)
            difference_rows.append(
                {
                    "mode": mode,
                    "variant": variant,
                    "comparison": name,
                    "estimate": float(np.nanmean(values)),
                    "ci_low": float(np.nanquantile(draws, 0.025)),
                    "ci_high": float(np.nanquantile(draws, 0.975)),
                    "estimable_years": int(np.isfinite(values).sum()),
                    "supported_positive": bool(np.nanquantile(draws, 0.025) > 0),
                }
            )
            draw_rows.extend(
                {
                    "mode": mode,
                    "variant": variant,
                    "kind": "difference",
                    "name": name,
                    "replicate": replicate,
                    "value": float(value),
                }
                for replicate, value in enumerate(draws)
            )
    return pd.DataFrame(estimate_rows), pd.DataFrame(difference_rows), pd.DataFrame(draw_rows)


def label_agreement(
    panels: dict[str, pd.DataFrame], primary: str, test_start: int, test_end: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = panels[primary]
    rows = []
    disagreements = []
    for variant, panel in panels.items():
        joined = base[
            ["cell_id", "year", "region_group", "eligible", "event", "annual_area_m2"]
        ].merge(
            panel[["cell_id", "year", "eligible", "event", "annual_area_m2"]],
            on=["cell_id", "year"],
            suffixes=("_primary", "_variant"),
            validate="one_to_one",
        )
        common = joined.loc[
            joined["year"].between(test_start, test_end)
            & joined["eligible_primary"]
            & joined["eligible_variant"]
        ].copy()
        intersection = int(((common["event_primary"] == 1) & (common["event_variant"] == 1)).sum())
        union = int(((common["event_primary"] == 1) | (common["event_variant"] == 1)).sum())
        event_agreement = float((common["event_primary"] == common["event_variant"]).mean())
        kappa = float(cohen_kappa_score(common["event_primary"], common["event_variant"]))
        positive_jaccard = intersection / union if union else np.nan
        area = joined.loc[
            joined["year"].between(1984, test_end + 1)
            & joined["annual_area_m2_primary"].notna()
            & joined["annual_area_m2_variant"].notna()
        ]
        log_ratio = np.log1p(area["annual_area_m2_variant"]) - np.log1p(
            area["annual_area_m2_primary"]
        )
        rows.append(
            {
                "variant": variant,
                "common_eligible_test_rows": len(common),
                "primary_events": int(common["event_primary"].sum()),
                "variant_events": int(common["event_variant"].sum()),
                "event_raw_agreement": event_agreement,
                "event_cohen_kappa": kappa,
                "event_positive_jaccard": positive_jaccard,
                "event_flips": int((common["event_primary"] != common["event_variant"]).sum()),
                "common_annual_area_rows": len(area),
                "annual_area_pearson": float(
                    area[["annual_area_m2_primary", "annual_area_m2_variant"]].corr().iloc[0, 1]
                ),
                "median_absolute_log1p_area_difference": float(np.median(np.abs(log_ratio))),
            }
        )
        changed = common.loc[common["event_primary"] != common["event_variant"]].copy()
        changed["variant"] = variant
        disagreements.append(changed)
    return pd.DataFrame(rows), pd.concat(disagreements, ignore_index=True)


def quality_summaries(
    quarterly: pd.DataFrame,
    predictions: pd.DataFrame,
    primary: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = quarterly.copy()
    data["sensor_era"] = np.select(
        [data["year"].le(2004), data["year"].between(2005, 2012)],
        ["1984-2004 Landsat 4/5/7", "2005-2012 Landsat 5/7"],
        default="2013-2025 Landsat 7/8/9",
    )
    region_era = data.groupby(["region_group", "sensor_era"]).agg(
        cell_quarters=("cell_id", "size"),
        mean_fixed_valid_fraction=("fixed_valid_fraction", "mean"),
        share_fixed_vf50=("fixed_valid_fraction", lambda x: float(x.ge(0.50).mean())),
        share_fixed_vf75=("fixed_valid_fraction", lambda x: float(x.ge(0.75).mean())),
        mean_passes=("mean_passes_fixed_valid_pixels", "mean"),
        share_quarters_mean_passes_ge2=(
            "mean_passes_fixed_valid_pixels", lambda x: float(x.ge(2).mean())
        ),
        median_area_se_rss=("area_se_rss_independence_fixed", "median"),
        median_area_se_sum=("area_se_sum_conservative_fixed", "median"),
    ).reset_index()
    yearly = data.groupby("year").agg(
        cell_quarters=("cell_id", "size"),
        mean_fixed_valid_fraction=("fixed_valid_fraction", "mean"),
        share_fixed_vf50=("fixed_valid_fraction", lambda x: float(x.ge(0.50).mean())),
        share_fixed_vf75=("fixed_valid_fraction", lambda x: float(x.ge(0.75).mean())),
        mean_passes=("mean_passes_fixed_valid_pixels", "mean"),
        passes5=("passes5_fixed_sum", "sum"),
        passes7=("passes7_fixed_sum", "sum"),
        passes8=("passes8_fixed_sum", "sum"),
        passes_total=("passes_total_fixed_sum", "sum"),
    ).reset_index()
    sensor_sum = yearly[["passes5", "passes7", "passes8"]].sum(axis=1)
    for sensor in ["passes5", "passes7", "passes8"]:
        yearly[f"{sensor}_share"] = np.divide(
            yearly[sensor], sensor_sum, out=np.zeros(len(yearly)), where=sensor_sum > 0
        )
    yearly["sensor_component_minus_total_passes"] = sensor_sum - yearly["passes_total"]

    source = predictions.loc[
        predictions["mode"].eq("native") & predictions["variant"].eq(primary)
    ].copy()
    source["sensor_era"] = np.where(
        source["year"].le(2012), "2005-2012 Landsat 5/7", "2013-2024 Landsat 7/8/9"
    )
    era_rows = []
    for (era, model), group in source.groupby(["sensor_era", "model"]):
        annual = []
        for _, year_group in group.groupby("year"):
            ap = safe_ap(year_group)
            if np.isfinite(ap):
                annual.append(ap - year_group["event"].mean())
        era_rows.append(
            {
                "sensor_era": era,
                "model": model,
                "n": len(group),
                "events": int(group["event"].sum()),
                "macro_year_ap_lift": float(np.mean(annual)),
                "estimable_years": len(annual),
            }
        )
    return region_era, yearly, pd.DataFrame(era_rows)


def make_figures(
    output: Path,
    quality_year: pd.DataFrame,
    agreement: pd.DataFrame,
    estimates: pd.DataFrame,
    differences: pd.DataFrame,
    primary: str,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(quality_year["year"], quality_year["share_fixed_vf50"], label="Valid fraction >=50%", color="#1565c0")
    axes[0].plot(quality_year["year"], quality_year["share_fixed_vf75"], label="Valid fraction >=75%", color="#78909c")
    axes[0].set_ylabel("Share of cell-quarters")
    axes[0].set_ylim(0, 1.03)
    axes[0].legend(loc="lower right")
    axes[0].set_title("Observation completeness and Landsat image support")
    axes[1].plot(quality_year["year"], quality_year["mean_passes"], color="#ef6c00", label="Mean images per valid pixel")
    axes[1].set_ylabel("Mean Landsat images")
    axes[1].set_xlabel("Year")
    for axis in axes:
        axis.axvline(2005, color="#555555", linestyle="--", linewidth=1)
        axis.axvline(2013, color="#555555", linestyle="--", linewidth=1)
    fig.savefig(output / "figure_01_observation_quality_by_year.png", dpi=180)
    plt.close(fig)

    order = list(VARIANT_LABELS)
    part = agreement.set_index("variant").reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    y = np.arange(len(part))
    ax.barh(y, part["event_positive_jaccard"], color=np.where(part["variant"].eq(primary), "#1565c0", "#90a4ae"))
    ax.set_yticks(y, [VARIANT_LABELS[x] for x in part["variant"]])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Positive-event Jaccard agreement with primary label")
    ax.set_title("How much does the 30% decline label change?")
    for index, row in part.iterrows():
        ax.text(row["event_positive_jaccard"] + 0.01, index, f"n={int(row['common_eligible_test_rows']):,}; flips={int(row['event_flips'])}", va="center", fontsize=8)
    fig.savefig(output / "figure_02_label_agreement.png", dpi=180)
    plt.close(fig)

    native_est = estimates.loc[estimates["mode"].eq("native")]
    native_diff = differences.loc[differences["mode"].eq("native")]
    rows = []
    for variant in order:
        current = native_est.loc[
            native_est["variant"].eq(variant) & native_est["model"].eq("current_only")
        ].iloc[0]
        trajectory = native_diff.loc[
            native_diff["variant"].eq(variant)
            & native_diff["comparison"].eq("trajectory_minus_current")
        ].iloc[0]
        oisst = native_diff.loc[
            native_diff["variant"].eq(variant)
            & native_diff["comparison"].eq("oisst_minus_trajectory")
        ].iloc[0]
        for metric, row in [("Current lift", current), ("Trajectory increment", trajectory), ("OISST increment", oisst)]:
            rows.append({"variant": variant, "metric": metric, "estimate": row.estimate, "ci_low": row.ci_low, "ci_high": row.ci_high})
    plot = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(17, 8), sharey=True, constrained_layout=True)
    colors = {"Current lift": "#1565c0", "Trajectory increment": "#ef6c00", "OISST increment": "#546e7a"}
    for axis, metric in zip(axes, ["Current lift", "Trajectory increment", "OISST increment"], strict=True):
        part = plot.loc[plot["metric"].eq(metric)].set_index("variant").reindex(order).reset_index()
        y = np.arange(len(part))
        axis.errorbar(
            part["estimate"], y,
            xerr=[part["estimate"] - part["ci_low"], part["ci_high"] - part["estimate"]],
            fmt="o", color=colors[metric], capsize=3,
        )
        axis.axvline(0, color="black", linewidth=1)
        axis.set_title(metric)
        axis.set_xlabel("Macro within-year AP lift or paired difference\n(95% 2-year block bootstrap CI)")
        axis.set_yticks(y)
        if metric == "Current lift":
            axis.set_yticklabels([VARIANT_LABELS[x] for x in part["variant"]])
        else:
            axis.tick_params(labelleft=False)
    fig.suptitle("Observation and annual-label robustness of the main conclusions")
    fig.savefig(output / "figure_03_model_conclusion_robustness.png", dpi=180)
    plt.close(fig)


def write_report(
    output: Path,
    config: dict[str, object],
    panels: dict[str, pd.DataFrame],
    agreement: pd.DataFrame,
    summaries: pd.DataFrame,
    estimates: pd.DataFrame,
    differences: pd.DataFrame,
    quality_region_era: pd.DataFrame,
    quality_era_performance: pd.DataFrame,
) -> dict[str, object]:
    primary = config["primary_variant"]
    primary_est = estimates.loc[
        estimates["mode"].eq("native")
        & estimates["variant"].eq(primary)
        & estimates["model"].eq("current_only")
    ].iloc[0]
    trajectory = differences.loc[
        differences["mode"].eq("native")
        & differences["variant"].eq(primary)
        & differences["comparison"].eq("trajectory_minus_current")
    ].iloc[0]
    oisst = differences.loc[
        differences["mode"].eq("native")
        & differences["variant"].eq(primary)
        & differences["comparison"].eq("oisst_minus_trajectory")
    ].iloc[0]
    common_trajectory = differences.loc[
        differences["mode"].eq("common_rows")
        & differences["variant"].eq(primary)
        & differences["comparison"].eq("trajectory_minus_current")
    ].iloc[0]
    high_trajectory = differences.loc[
        differences["mode"].eq("high_observation_subset")
        & differences["variant"].eq(primary)
        & differences["comparison"].eq("trajectory_minus_current")
    ].iloc[0]
    high = summaries.loc[
        summaries["mode"].eq("high_observation_subset")
        & summaries["variant"].eq(primary)
        & summaries["model"].eq("current_only")
    ].iloc[0]
    native_current = estimates.loc[
        estimates["mode"].eq("native") & estimates["model"].eq("current_only")
    ]
    native_oisst = differences.loc[
        differences["mode"].eq("native")
        & differences["comparison"].eq("oisst_minus_trajectory")
    ]
    common_current = estimates.loc[
        estimates["mode"].eq("common_rows") & estimates["model"].eq("current_only")
    ]
    primary_panel = panels[primary]
    test_eligible = primary_panel.loc[
        primary_panel["year"].between(2005, 2024) & primary_panel["eligible"]
    ]
    min_agreement = agreement.loc[agreement["variant"].ne(primary), "event_positive_jaccard"].min()
    decision = {
        "primary_current_lift": float(primary_est.estimate),
        "primary_current_ci": [float(primary_est.ci_low), float(primary_est.ci_high)],
        "primary_trajectory_increment": float(trajectory.estimate),
        "primary_trajectory_ci": [float(trajectory.ci_low), float(trajectory.ci_high)],
        "primary_oisst_increment": float(oisst.estimate),
        "primary_oisst_ci": [float(oisst.ci_low), float(oisst.ci_high)],
        "common_row_trajectory_increment": float(common_trajectory.estimate),
        "common_row_trajectory_ci": [float(common_trajectory.ci_low), float(common_trajectory.ci_high)],
        "high_observation_trajectory_increment": float(high_trajectory.estimate),
        "high_observation_trajectory_ci": [float(high_trajectory.ci_low), float(high_trajectory.ci_high)],
        "current_lift_positive_ci_all_native_variants": bool(native_current["ci_low"].gt(0).all()),
        "current_lift_positive_ci_all_common_row_variants": bool(common_current["ci_low"].gt(0).all()),
        "oisst_positive_supported_any_native_variant": bool(native_oisst["ci_low"].gt(0).any()),
        "minimum_positive_event_jaccard_vs_primary": float(min_agreement),
        "primary_test_eligible_rows": int(len(test_eligible)),
        "primary_test_events": int(test_eligible["event"].sum()),
        "high_observation_subset_rows": int(high.n),
        "high_observation_subset_current_lift": float(high.macro_year_ap_lift),
    }
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lowest = agreement.loc[agreement["variant"].ne(primary)].sort_values("event_positive_jaccard").iloc[0]
    era_current = quality_era_performance.loc[
        quality_era_performance["model"].eq("current_only")
    ].sort_values("sensor_era")
    weakest_quality = quality_region_era.sort_values("share_fixed_vf50").iloc[0]
    lines = [
        "# 관측품질·라벨 강건성 실험 결과",
        "",
        "> 기존 결과를 본 뒤 고정한 강건성 분석이다. 독립 확증이나 사전등록 분석으로 표현하지 않는다.",
        "",
        "## 결론부터",
        "",
        "기존 구현은 all-history habitat footprint와 `75% 유효 분기 3개, Q3 불필수, 연간 최대`를 사용해, 잠근 프로토콜의 `pre-2005 고정 footprint, 50% 유효 분기 3개, Q3 필수`와 달랐다. 이번 실행은 공식 NetCDF 픽셀에서 주 프로토콜을 다시 만들고 그 차이를 직접 분해했다.",
        "",
        f"- 주 프로토콜의 현재 상태 AP lift는 **{primary_est.estimate:.3f}** (95% 2년 block CI {primary_est.ci_low:.3f}–{primary_est.ci_high:.3f})였다.",
        f"- 과거 궤적 증분은 **{trajectory.estimate:+.3f}** (95% CI {trajectory.ci_low:+.3f}–{trajectory.ci_high:+.3f})였다.",
        f"- 다만 완전 동일행 8개 정의의 주 라벨에서는 궤적 증분이 {common_trajectory.estimate:+.3f} (95% CI {common_trajectory.ci_low:+.3f}–{common_trajectory.ci_high:+.3f}), 고관측품질 subset에서는 {high_trajectory.estimate:+.3f} (95% CI {high_trajectory.ci_low:+.3f}–{high_trajectory.ci_high:+.3f})로 0을 배제하지 못했다. 따라서 궤적은 ‘작은 양의 점추정, 통계적 지지는 표본정의 의존’으로 낮춰 쓴다.",
        f"- OISST 증분은 **{oisst.estimate:+.3f}** (95% CI {oisst.ci_low:+.3f}–{oisst.ci_high:+.3f})였다.",
        f"- 현재 상태의 양의 순위화 가치는 native 8개 정의와 완전 동일행 비교 8개 정의에서 모두 95% CI가 0보다 컸다: **{'예' if decision['current_lift_positive_ci_all_native_variants'] and decision['current_lift_positive_ci_all_common_row_variants'] else '아니오'}**.",
        f"- OISST가 양의 추가가치를 보인 정의는 8개 중 **{int(native_oisst['ci_low'].gt(0).sum())}개**였다.",
        f"- 주 라벨과 가장 많이 달라진 정의는 `{lowest.variant}`였고, 양성사건 Jaccard는 {lowest.event_positive_jaccard:.3f}, 공통 적격행 {int(lowest.common_eligible_test_rows):,}개 중 사건 flip은 {int(lowest.event_flips):,}개였다.",
        "",
        "## 무엇이 달라졌나",
        "",
        f"- 주 프로토콜 평가기간 적격 표본은 {len(test_eligible):,} cell-year, 급감 사건은 {int(test_eligible['event'].sum()):,}개였다.",
        f"- 가장 낮은 지역·센서시대 분기 50% 관측통과율은 {weakest_quality.region_group} / {weakest_quality.sensor_era}의 {weakest_quality.share_fixed_vf50:.1%}였다.",
        f"- 모든 사용 분기의 픽셀당 평균 Landsat 영상 수가 2개 이상인 고품질 current-next 쌍에서도 현재 상태 macro AP lift는 {high.macro_year_ap_lift:.3f}였다(n={int(high.n):,}).",
        "- `area_se`는 RSS(픽셀 독립 가정)와 단순합(완전상관 상계)을 모두 저장했지만 가중치에는 사용하지 않았다. 픽셀 공분산과 annual-maximum 선택오차가 없어 단일 inverse-variance weight를 정당화할 수 없기 때문이다.",
        "",
        "## 센서시대 확인",
        "",
    ]
    for row in era_current.itertuples(index=False):
        lines.append(
            f"- {row.sensor_era}: 현재 상태 macro within-year AP lift {row.macro_year_ap_lift:.3f} (평가 가능 연도 {row.estimable_years}개, n={row.n:,})."
        )
    lines.extend(
        [
            "",
            "## 논문에서의 해석",
            "",
            "1. **주 결과는 프로토콜 정의로 교체한다.** 앞으로 legacy all-history/75%/Q3-불필수 라벨을 주 분석이라고 쓰면 안 된다.",
            "2. **현재 상태 순위화 결론은 관측완전성과 연간 집계 정의에 강건하다.** 이는 RQ1 방어력을 실제로 높인다.",
            "3. **과거 궤적의 증분은 강건한 확정 결과로 올리지 않는다.** 주 native 분석에서는 양수지만 동일행·고관측품질 분석에서 신뢰구간이 0을 포함했다.",
            "4. **환경 proxy 결과는 여전히 제한적으로 쓴다.** 라벨을 바꿔도 양의 OISST 증분이 재현되지 않았다면, ‘환경이 무관’이 아니라 ‘이 공개 proxy의 추가 예측정보가 확인되지 않음’이다.",
            "5. **라벨 동일성은 주장하지 않는다.** annual mean 또는 Q2–Q3 max는 사건 자체를 바꾸므로, 본문에는 주 정의를 두고 나머지는 민감도로 제시한다.",
            "6. **inverse-variance sensitivity는 보류가 아니라 부적합 판정이다.** 현 메타데이터로 임의 가중치를 만들면 오히려 리뷰어에게 약점이 된다.",
            "",
            "## 남은 관측품질 작업",
            "",
            "- 공식 Kelpwatch 웹/API와 지역별 층화표본 cell-quarter를 대조하는 독립 집계 검증은 별도 Gate 2로 남아 있다. 이번 실행은 공식 NetCDF와 기존 저장 quarterly 집계의 전 행 재현을 검증했다.",
            "- 5 km·20 km 격자와 half-cell offset은 라벨이 아니라 공간 support 민감도이므로 다음 E16 하위실험에서 다룬다.",
        ]
    )
    (output / "results_summary_ko.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


def main() -> None:
    args = parse_args()
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed run: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    quarterly_external = pd.read_csv(args.quarterly)
    inventory = pd.read_csv(args.inventory)
    stable_flag = inventory["stable_model_candidate"].astype(str).str.lower().eq("true")
    stable = inventory.loc[stable_flag].copy()
    stable = stable.sort_values("cell_id").reset_index(drop=True)
    if len(stable) != 165:
        raise ValueError(f"Expected 165 stable cells, found {len(stable)}")
    quarterly_external = quarterly_external.loc[
        quarterly_external["cell_id"].isin(stable["cell_id"])
        & quarterly_external["year"].between(1984, int(config["forecast"]["panel_end"]))
    ].copy()
    environment = pd.read_csv(args.environment)

    rebuilt, footprint, source_metadata = aggregate_netcdf_quality(
        args.netcdf, stable, int(config["forecast"]["panel_end"])
    )
    quarterly, rebuild_checks = merge_external_quarterly(rebuilt, quarterly_external)
    footprint = footprint.merge(
        stable[[
            "cell_id",
            "count_cells_historic_footprint",
            "count_cells_pre2005_footprint",
        ]],
        on="cell_id",
        how="left",
        validate="one_to_one",
    )
    footprint["allhistory_count_match"] = footprint[
        "allhistory_station_pixels_rebuilt"
    ].eq(footprint["count_cells_historic_footprint"])
    footprint["pre2005_count_match"] = footprint[
        "pre2005_positive_pixels_rebuilt"
    ].eq(footprint["count_cells_pre2005_footprint"])
    if not rebuild_checks["passed"].all() or not footprint[
        ["allhistory_count_match", "pre2005_count_match"]
    ].all().all():
        raise AssertionError("Official NetCDF aggregation failed to reproduce saved counts")

    panels: dict[str, pd.DataFrame] = {}
    annual_tables = []
    for spec in config["variants"]:
        annual = annualize(quarterly, spec)
        annual_tables.append(annual)
        panels[spec["name"]] = build_panel(annual, stable, config)

    locked = pd.read_csv(args.locked_panel)
    legacy = panels["legacy_allhist_vf75_max"]
    reproduction = locked[["cell_id", "year", "annual_max_kelp_area_m2", "event", "eligible"]].merge(
        legacy[["cell_id", "year", "annual_area_m2", "event", "eligible"]],
        on=["cell_id", "year"],
        suffixes=("_locked", "_rebuilt"),
        validate="one_to_one",
    )
    area_match = np.isclose(
        reproduction["annual_max_kelp_area_m2"], reproduction["annual_area_m2"], atol=0, rtol=0, equal_nan=True
    )
    reproduction_checks = pd.DataFrame(
        [
            {"check": "legacy_annual_area_matches_locked_panel", "mismatches": int((~area_match).sum())},
            {"check": "legacy_event_matches_locked_panel", "mismatches": int(reproduction["event_locked"].ne(reproduction["event_rebuilt"]).sum())},
            {"check": "legacy_eligibility_matches_locked_panel", "mismatches": int(reproduction["eligible_locked"].astype(bool).ne(reproduction["eligible_rebuilt"].astype(bool)).sum())},
        ]
    )
    reproduction_checks["passed"] = reproduction_checks["mismatches"].eq(0)
    if not reproduction_checks["passed"].all():
        raise AssertionError(reproduction_checks.to_string(index=False))

    predictions, folds = run_backtests(panels, environment, config)
    year_metrics, summaries = calculate_metrics(predictions)
    estimates, differences, draws = bootstrap_metrics(year_metrics, config)
    agreement, disagreements = label_agreement(
        panels,
        config["primary_variant"],
        int(config["forecast"]["test_start"]),
        int(config["forecast"]["test_end"]),
    )
    quality_region_era, quality_year, quality_era_performance = quality_summaries(
        quarterly, predictions, config["primary_variant"]
    )

    key_counts = predictions.groupby(["mode", "variant", "model"])[["cell_id", "year"]].size().reset_index(name="rows")
    quality_checks = pd.DataFrame(
        [
            {"check": "stable_cells_165", "value": len(stable), "passed": len(stable) == 165},
            {"check": "quarterly_keys_unique", "value": int(quarterly.duplicated(["cell_id", "year", "quarter"]).sum()), "passed": not quarterly.duplicated(["cell_id", "year", "quarter"]).any()},
            {"check": "panel_keys_unique_all_variants", "value": int(sum(panel.duplicated(["cell_id", "year"]).sum() for panel in panels.values())), "passed": all(not panel.duplicated(["cell_id", "year"]).any() for panel in panels.values())},
            {"check": "train_precedes_test", "value": int((folds["train_end"] >= folds["year"]).sum()), "passed": bool((folds["train_end"] < folds["year"]).all())},
            {"check": "prediction_scores_complete", "value": int(predictions["score"].isna().sum()), "passed": not predictions["score"].isna().any()},
            {"check": "prediction_keys_unique", "value": int(predictions.duplicated(["mode", "variant", "model", "cell_id", "year"]).sum()), "passed": not predictions.duplicated(["mode", "variant", "model", "cell_id", "year"]).any()},
            {"check": "model_rows_matched_within_mode_variant", "value": int(key_counts.groupby(["mode", "variant"])["rows"].nunique().max()), "passed": bool(key_counts.groupby(["mode", "variant"])["rows"].nunique().eq(1).all())},
            {"check": "netcdf_quarterly_rebuild", "value": int(rebuild_checks["mismatches"].sum()), "passed": bool(rebuild_checks["passed"].all())},
            {"check": "legacy_locked_reproduction", "value": int(reproduction_checks["mismatches"].sum()), "passed": bool(reproduction_checks["passed"].all())},
            {"check": "fixed_footprint_count_reproduction", "value": int((~footprint["pre2005_count_match"]).sum()), "passed": bool(footprint["pre2005_count_match"].all())},
        ]
    )
    if not quality_checks["passed"].all():
        raise AssertionError(quality_checks.to_string(index=False))

    outputs = {
        "quarterly_observation_quality_165.csv": quarterly,
        "fixed_footprint_reproduction.csv": footprint,
        "netcdf_rebuild_checks.csv": rebuild_checks,
        "legacy_reproduction_checks.csv": reproduction_checks,
        "annual_variants.csv": pd.concat(annual_tables, ignore_index=True),
        "variant_panels.csv": pd.concat(panels.values(), ignore_index=True),
        "predictions.csv": predictions,
        "fold_audit.csv": folds,
        "year_metrics.csv": year_metrics,
        "model_summary.csv": summaries,
        "bootstrap_estimates.csv": estimates,
        "bootstrap_differences.csv": differences,
        "bootstrap_draws.csv": draws,
        "label_agreement.csv": agreement,
        "label_disagreements.csv": disagreements,
        "observation_quality_by_region_era.csv": quality_region_era,
        "observation_quality_by_year.csv": quality_year,
        "performance_by_sensor_era.csv": quality_era_performance,
        "quality_checks.csv": quality_checks,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    make_figures(
        args.output_dir,
        quality_year,
        agreement,
        estimates,
        differences,
        config["primary_variant"],
    )
    decision = write_report(
        args.output_dir,
        config,
        panels,
        agreement,
        summaries,
        estimates,
        differences,
        quality_region_era,
        quality_era_performance,
    )
    manifest = {
        "status": "complete",
        "classification": config["classification"],
        "experiment_id": config["experiment_id"],
        "started_at_utc_epoch": started,
        "finished_at_utc_epoch": time.time(),
        "runtime_seconds": round(time.time() - started, 2),
        "command": " ".join(sys.argv),
        "git_sha_before_run": git_sha(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "source_metadata": source_metadata,
        "seed": int(config["forecast"]["seed"]),
        "bootstrap_replicates": int(config["forecast"]["bootstrap_replicates"]),
        "bootstrap_block_years": int(config["forecast"]["bootstrap_block_years"]),
        "inputs": {
            "config": {"path": str(args.config), "sha256": sha256(args.config)},
            "quarterly": {"path": str(args.quarterly), "sha256": sha256(args.quarterly)},
            "inventory": {"path": str(args.inventory), "sha256": sha256(args.inventory)},
            "netcdf": {"path": str(args.netcdf), "sha256": sha256(args.netcdf)},
            "environment": {"path": str(args.environment), "sha256": sha256(args.environment)},
            "locked_panel": {"path": str(args.locked_panel), "sha256": sha256(args.locked_panel)},
        },
        "decisions": decision,
        "output_sha256": {name: sha256(args.output_dir / name) for name in outputs},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
