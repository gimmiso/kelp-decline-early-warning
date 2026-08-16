"""Run grid-size, origin-shift, and cohort-threshold sensitivity analyses.

The experiment rebuilds every panel from the official Kelpwatch NetCDF using
only pixels with positive kelp area in 1984-2004 as a fixed footprint.  It
evaluates 5, 10, and 20 km equal-area grids at four half-cell origin shifts.
Cell-area-scaled and literal 500-pixel cohort thresholds are separated so that
grid support is not confused with an arbitrary change in inclusion density.
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
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from netCDF4 import Dataset
from pyproj import Transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EPSILON = 1e-9
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
FULL_MODELS = {
    "current_only": ["relative_canopy"],
    "current_plus_trajectory": ["relative_canopy", *TRAJECTORY],
}
OISST_MODELS = {
    **FULL_MODELS,
    "trajectory_plus_oisst": ["relative_canopy", *TRAJECTORY, *OISST],
}
ORIGIN_LABELS = {
    "o00": "Base origin",
    "ox50": "X + half cell",
    "oy50": "Y + half cell",
    "oxy50": "X/Y + half cell",
}


@dataclass
class GridWork:
    spec_id: str
    grid_size_m: int
    grid_size_km: int
    threshold_policy: str
    origin_name: str
    origin_x_m: float
    origin_y_m: float
    minimum_footprint_pixels: int
    grid_x: np.ndarray
    grid_y: np.ndarray
    center_x: np.ndarray
    center_y: np.ndarray
    center_lon: np.ndarray
    center_lat: np.ndarray
    pre2005_footprint_pixels: np.ndarray
    pre_to_candidate: np.ndarray
    area_by_time: np.ndarray
    valid_by_time: np.ndarray
    stable_pre_mapping: np.ndarray | None = None
    stable_cell_ids: list[str] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--netcdf", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--base-inventory", type=Path, required=True)
    parser.add_argument("--base-primary-panel", type=Path, required=True)
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


def region_from_latitude(latitude: float) -> str:
    if latitude >= 46.0:
        return "Washington outer coast"
    if latitude >= 42.0:
        return "Oregon"
    if latitude >= 37.0:
        return "Northern California"
    if latitude >= 34.45:
        return "Central California"
    if latitude >= 32.45:
        return "Southern California"
    if latitude >= 28.0:
        return "Baja California Norte"
    return "Baja California Sur"


def spec_id(grid_size_m: int, policy: str, origin: str) -> str:
    return f"g{grid_size_m // 1000:02d}_{policy}_{origin}"


def footprint_threshold(grid_size_m: int, policy: str, base_pixels: int) -> int:
    if policy == "fixed_500":
        return int(base_pixels)
    if policy == "area_scaled":
        return int(round(base_pixels * (grid_size_m / 10_000) ** 2))
    raise ValueError(f"Unknown threshold policy: {policy}")


def build_specifications(config: dict[str, object]) -> list[dict[str, object]]:
    specs = []
    base_pixels = int(config["cohort"]["base_10km_footprint_pixels"])
    for size in config["grid_sizes_m"]:
        for policy in config["cohort_threshold_policies"]:
            if size in policy.get("exclude_grid_sizes_m", []):
                continue
            for origin in config["origins"]:
                specs.append(
                    {
                        "spec_id": spec_id(int(size), policy["name"], origin["name"]),
                        "grid_size_m": int(size),
                        "grid_size_km": int(size) // 1000,
                        "threshold_policy": policy["name"],
                        "origin_name": origin["name"],
                        "origin_x_m": float(origin["x_fraction"]) * int(size),
                        "origin_y_m": float(origin["y_fraction"]) * int(size),
                        "minimum_footprint_pixels": footprint_threshold(
                            int(size), policy["name"], base_pixels
                        ),
                    }
                )
    return specs


def read_coordinates_and_pre2005_mask(
    dataset: Dataset, config: dict[str, object]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    longitude = np.asarray(dataset.variables["longitude"][:], dtype=float)
    latitude = np.asarray(dataset.variables["latitude"][:], dtype=float)
    coordinate_ok = np.isfinite(longitude) & np.isfinite(latitude)
    source_indices = np.flatnonzero(coordinate_ok)
    longitude = longitude[coordinate_ok]
    latitude = latitude[coordinate_ok]
    forward = Transformer.from_crs(
        config["source_crs"], config["grid_crs"], always_xy=True
    )
    x, y = forward.transform(longitude, latitude)
    prepositive = np.zeros(len(source_indices), dtype=bool)
    years = np.asarray(dataset.variables["year"][:], dtype=int)
    area_variable = dataset.variables["area"]
    for time_index, year in enumerate(years):
        if year > int(config["cohort"]["reference_end_year"]):
            continue
        selected = area_variable[time_index, :][source_indices]
        mask = np.ma.getmaskarray(selected)
        values = np.asarray(selected.filled(0), dtype=float)
        prepositive |= (~mask) & (values > 0)
    return source_indices, longitude, latitude, np.asarray(x), np.asarray(y), prepositive


def build_grid_work(
    spec: dict[str, object],
    pre_x: np.ndarray,
    pre_y: np.ndarray,
    n_times: int,
    backward: Transformer,
) -> GridWork:
    size = int(spec["grid_size_m"])
    gx = np.floor((pre_x - float(spec["origin_x_m"])) / size).astype(np.int64)
    gy = np.floor((pre_y - float(spec["origin_y_m"])) / size).astype(np.int64)
    unique_keys, inverse = np.unique(np.column_stack([gx, gy]), axis=0, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique_keys)).astype(int)
    candidate_old = np.flatnonzero(counts >= int(spec["minimum_footprint_pixels"]))
    old_to_candidate = np.full(len(unique_keys), -1, dtype=np.int32)
    old_to_candidate[candidate_old] = np.arange(len(candidate_old), dtype=np.int32)
    pre_to_candidate = old_to_candidate[inverse]
    keys = unique_keys[candidate_old]
    center_x = float(spec["origin_x_m"]) + (keys[:, 0] + 0.5) * size
    center_y = float(spec["origin_y_m"]) + (keys[:, 1] + 0.5) * size
    center_lon, center_lat = backward.transform(center_x, center_y)
    return GridWork(
        spec_id=str(spec["spec_id"]),
        grid_size_m=size,
        grid_size_km=int(spec["grid_size_km"]),
        threshold_policy=str(spec["threshold_policy"]),
        origin_name=str(spec["origin_name"]),
        origin_x_m=float(spec["origin_x_m"]),
        origin_y_m=float(spec["origin_y_m"]),
        minimum_footprint_pixels=int(spec["minimum_footprint_pixels"]),
        grid_x=keys[:, 0],
        grid_y=keys[:, 1],
        center_x=center_x,
        center_y=center_y,
        center_lon=np.asarray(center_lon),
        center_lat=np.asarray(center_lat),
        pre2005_footprint_pixels=counts[candidate_old],
        pre_to_candidate=pre_to_candidate,
        area_by_time=np.zeros((n_times, len(candidate_old)), dtype=float),
        valid_by_time=np.zeros((n_times, len(candidate_old)), dtype=np.int32),
    )


def aggregate_all_grids(
    dataset: Dataset,
    source_indices: np.ndarray,
    prepositive: np.ndarray,
    works: list[GridWork],
    time_indices: np.ndarray,
) -> None:
    area_variable = dataset.variables["area"]
    for local_time, source_time in enumerate(time_indices):
        selected = area_variable[source_time, :][source_indices]
        mask = np.ma.getmaskarray(selected)[prepositive]
        values = np.asarray(selected.filled(0), dtype=float)[prepositive]
        valid = ~mask
        for work in works:
            candidate = work.pre_to_candidate
            keep = candidate >= 0
            cell_index = candidate[keep]
            valid_keep = valid[keep]
            work.valid_by_time[local_time] = np.bincount(
                cell_index,
                weights=valid_keep.astype(np.int16),
                minlength=work.area_by_time.shape[1],
            ).astype(np.int32)
            work.area_by_time[local_time] = np.bincount(
                cell_index,
                weights=np.where(valid_keep, values[keep], 0.0),
                minlength=work.area_by_time.shape[1],
            )


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


def annualize_and_select_cohort(
    work: GridWork,
    years: np.ndarray,
    quarters: np.ndarray,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_years = np.arange(1984, int(config["forecast"]["panel_end"]) + 1)
    if len(unique_years) * 4 != len(years):
        raise ValueError("Expected four complete quarters for every analysis year")
    expected_years = np.repeat(unique_years, 4)
    expected_quarters = np.tile(np.arange(1, 5), len(unique_years))
    if not np.array_equal(years, expected_years) or not np.array_equal(quarters, expected_quarters):
        raise ValueError("NetCDF time axis is not ordered as complete year-quarter blocks")
    n_cells = work.area_by_time.shape[1]
    area = work.area_by_time.reshape(len(unique_years), 4, n_cells)
    valid_count = work.valid_by_time.reshape(len(unique_years), 4, n_cells)
    valid_fraction = valid_count / work.pre2005_footprint_pixels[np.newaxis, np.newaxis, :]
    quarter_valid = valid_fraction >= float(
        config["annual_observation_rule"]["minimum_valid_habitat_fraction"]
    )
    complete = quarter_valid.sum(axis=1) >= int(
        config["annual_observation_rule"]["minimum_valid_quarters"]
    )
    if bool(config["annual_observation_rule"]["require_q3"]):
        complete &= quarter_valid[:, 2, :]
    masked_area = np.where(quarter_valid, area, -np.inf)
    annual_area = masked_area.max(axis=1)
    selected_quarter = masked_area.argmax(axis=1) + 1
    annual_area[~complete] = np.nan
    selected_quarter[~complete] = 0
    ref_mask = unique_years <= int(config["cohort"]["reference_end_year"])
    positive_years = np.sum(
        np.isfinite(annual_area[ref_mask]) & (annual_area[ref_mask] > 0), axis=0
    )
    stable = positive_years >= int(config["cohort"]["minimum_positive_reference_years"])
    stable_indices = np.flatnonzero(stable)
    cell_ids = [
        f"{work.spec_id}_x{int(work.grid_x[index])}_y{int(work.grid_y[index])}"
        for index in stable_indices
    ]
    candidate_to_stable = np.full(n_cells, -1, dtype=np.int32)
    candidate_to_stable[stable_indices] = np.arange(len(stable_indices), dtype=np.int32)
    pre_mapping = np.full(len(work.pre_to_candidate), -1, dtype=np.int32)
    candidate_mask = work.pre_to_candidate >= 0
    pre_mapping[candidate_mask] = candidate_to_stable[work.pre_to_candidate[candidate_mask]]
    work.stable_pre_mapping = pre_mapping
    work.stable_cell_ids = cell_ids

    inventory = pd.DataFrame(
        {
            "spec_id": work.spec_id,
            "grid_size_km": work.grid_size_km,
            "threshold_policy": work.threshold_policy,
            "origin_name": work.origin_name,
            "origin_x_m": work.origin_x_m,
            "origin_y_m": work.origin_y_m,
            "minimum_footprint_pixels": work.minimum_footprint_pixels,
            "grid_x_index": work.grid_x,
            "grid_y_index": work.grid_y,
            "center_lon": work.center_lon,
            "center_lat": work.center_lat,
            "region_group": [region_from_latitude(value) for value in work.center_lat],
            "pre2005_footprint_pixels": work.pre2005_footprint_pixels,
            "positive_reference_years": positive_years,
            "stable_cohort": stable,
        }
    )
    inventory["cell_id"] = [
        f"{work.spec_id}_x{int(x)}_y{int(y)}"
        for x, y in zip(inventory["grid_x_index"], inventory["grid_y_index"], strict=True)
    ]

    stable_area = annual_area[:, stable_indices]
    stable_complete = complete[:, stable_indices]
    stable_quarters = quarter_valid[:, :, stable_indices].sum(axis=1)
    stable_selected = selected_quarter[:, stable_indices]
    stable_inventory = inventory.loc[stable].reset_index(drop=True)
    panel = pd.DataFrame(
        {
            "cell_id": np.repeat(cell_ids, len(unique_years)),
            "year": np.tile(unique_years, len(cell_ids)),
            "annual_area_m2": stable_area.T.reshape(-1),
            "annual_complete": stable_complete.T.reshape(-1),
            "valid_quarters": stable_quarters.T.reshape(-1),
            "selected_quarter": stable_selected.T.reshape(-1),
        }
    ).merge(
        stable_inventory[
            [
                "cell_id",
                "spec_id",
                "grid_size_km",
                "threshold_policy",
                "origin_name",
                "center_lon",
                "center_lat",
                "region_group",
                "pre2005_footprint_pixels",
                "positive_reference_years",
            ]
        ],
        on="cell_id",
        how="left",
        validate="many_to_one",
    )
    reference = panel.loc[
        panel["year"].between(
            int(config["cohort"]["reference_start_year"]),
            int(config["cohort"]["reference_end_year"]),
        )
        & panel["annual_area_m2"].notna()
    ].groupby("cell_id")["annual_area_m2"].quantile(0.95)
    panel["p95_annual_area_pre2005"] = panel["cell_id"].map(reference)
    if panel["p95_annual_area_pre2005"].isna().any() or panel[
        "p95_annual_area_pre2005"
    ].le(0).any():
        raise ValueError(f"Invalid reference canopy in {work.spec_id}")
    panel["relative_canopy"] = panel["annual_area_m2"] / panel[
        "p95_annual_area_pre2005"
    ]
    grouped = panel.groupby("cell_id", group_keys=False)
    panel["next_year_area_m2"] = grouped["annual_area_m2"].shift(-1)
    panel["relative_drop_next"] = (
        panel["annual_area_m2"] - panel["next_year_area_m2"]
    ) / np.maximum(panel["annual_area_m2"], EPSILON)
    panel["event"] = panel["relative_drop_next"].ge(
        float(config["outcome"]["decline_fraction"])
    ).astype(int)
    panel = add_features(panel)
    panel["eligible"] = (
        panel["relative_canopy"].gt(
            float(config["outcome"]["eligibility_relative_canopy_gt"])
        )
        & panel["annual_area_m2"].notna()
        & panel["next_year_area_m2"].notna()
        & panel["year"].between(
            int(config["forecast"]["train_start"]),
            int(config["forecast"]["test_end"]),
        )
    )
    return inventory, panel


def haversine_matrix_km(
    lat: np.ndarray, lon: np.ndarray, source_lat: np.ndarray, source_lon: np.ndarray
) -> np.ndarray:
    radius = 6371.0088
    lat1 = np.radians(lat)[:, None]
    lon1 = np.radians(lon)[:, None]
    lat2 = np.radians(source_lat)[None, :]
    lon2 = np.radians(source_lon)[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def prepare_oisst_sources(environment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["oisst_source_lat", "oisst_source_lon", "year"]
    valid_coordinates = environment["oisst_source_lat"].notna() & environment[
        "oisst_source_lon"
    ].notna()
    environment = environment.loc[valid_coordinates].copy()
    consistency = environment.groupby(key)[OISST].nunique(dropna=False)
    if consistency.gt(1).any().any():
        raise ValueError("OISST source-grid features are not unique by source point and year")
    source_year = environment[key + OISST].drop_duplicates(key).copy()
    sources = source_year[["oisst_source_lat", "oisst_source_lon"]].drop_duplicates().reset_index(drop=True)
    sources["source_index"] = np.arange(len(sources))
    return sources, source_year


def attach_oisst(
    panel: pd.DataFrame,
    sources: pd.DataFrame,
    source_year: pd.DataFrame,
    max_distance_km: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = panel[
        ["cell_id", "spec_id", "center_lat", "center_lon"]
    ].drop_duplicates("cell_id")
    distances = haversine_matrix_km(
        cells["center_lat"].to_numpy(),
        cells["center_lon"].to_numpy(),
        sources["oisst_source_lat"].to_numpy(),
        sources["oisst_source_lon"].to_numpy(),
    )
    nearest = distances.argmin(axis=1)
    match = cells.copy()
    match["oisst_source_lat"] = sources.iloc[nearest]["oisst_source_lat"].to_numpy()
    match["oisst_source_lon"] = sources.iloc[nearest]["oisst_source_lon"].to_numpy()
    match["oisst_source_distance_km"] = distances[np.arange(len(cells)), nearest]
    match["oisst_supported"] = match["oisst_source_distance_km"].le(max_distance_km)
    merged = panel.merge(
        match[
            [
                "cell_id",
                "oisst_source_lat",
                "oisst_source_lon",
                "oisst_source_distance_km",
                "oisst_supported",
            ]
        ],
        on="cell_id",
        how="left",
        validate="many_to_one",
    ).merge(
        source_year,
        on=["oisst_source_lat", "oisst_source_lon", "year"],
        how="left",
        validate="many_to_one",
    )
    return merged, match


def model_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=5000, random_state=seed)),
        ]
    )


def run_backtests(
    panels: dict[str, pd.DataFrame], config: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = []
    folds = []
    for spec, data in panels.items():
        domains = {
            "full_kelp_domain": (FULL_MODELS, data["eligible"]),
            "oisst_matched_domain": (
                OISST_MODELS,
                data["eligible"]
                & data["oisst_supported"].fillna(False)
                & data[OISST].notna().all(axis=1),
            ),
        }
        for domain, (models, eligible) in domains.items():
            for year in range(
                int(config["forecast"]["test_start"]),
                int(config["forecast"]["test_end"]) + 1,
            ):
                train = data.loc[
                    eligible
                    & data["year"].between(
                        int(config["forecast"]["train_start"]), year - 1
                    )
                ].copy()
                test = data.loc[eligible & data["year"].eq(year)].copy()
                if train["event"].nunique() < 2 or test.empty:
                    eligible_by_year = (
                        data.loc[eligible]
                        .groupby("year")["event"]
                        .agg(["size", "sum", "nunique"])
                        .to_dict("index")
                    )
                    raise ValueError(
                        "Invalid fold: "
                        f"{spec}, {domain}, {year}; "
                        f"train_n={len(train)}, train_events={int(train['event'].sum())}, "
                        f"train_classes={train['event'].nunique()}, test_n={len(test)}, "
                        f"eligible_by_year={eligible_by_year}"
                    )
                folds.append(
                    {
                        "spec_id": spec,
                        "domain": domain,
                        "year": year,
                        "train_start": int(train["year"].min()),
                        "train_end": int(train["year"].max()),
                        "n_train": len(train),
                        "events_train": int(train["event"].sum()),
                        "n_test": len(test),
                        "events_test": int(test["event"].sum()),
                    }
                )
                for model, features in models.items():
                    estimator = model_pipeline(int(config["forecast"]["seed"]))
                    estimator.fit(train[features], train["event"])
                    scored = test[
                        [
                            "cell_id",
                            "spec_id",
                            "grid_size_km",
                            "threshold_policy",
                            "origin_name",
                            "region_group",
                            "year",
                            "event",
                        ]
                    ].copy()
                    scored["score"] = estimator.predict_proba(test[features])[:, 1]
                    scored["model"] = model
                    scored["domain"] = domain
                    predictions.append(scored)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(folds)


def safe_ap(group: pd.DataFrame) -> float:
    if group.empty or group["event"].nunique() < 2:
        return np.nan
    return float(average_precision_score(group["event"], group["score"]))


def calculate_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (spec, domain, model, year), group in predictions.groupby(
        ["spec_id", "domain", "model", "year"], sort=True
    ):
        ranked = group.sort_values(["score", "cell_id"], ascending=[False, True])
        events = int(ranked["event"].sum())
        prevalence = float(ranked["event"].mean())
        ap = safe_ap(ranked)
        k = max(1, int(math.ceil(len(ranked) * 0.20)))
        rows.append(
            {
                "spec_id": spec,
                "domain": domain,
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
    for (spec, domain, model), group in predictions.groupby(["spec_id", "domain", "model"]):
        annual = year_metrics.loc[
            year_metrics["spec_id"].eq(spec)
            & year_metrics["domain"].eq(domain)
            & year_metrics["model"].eq(model)
        ]
        estimable = annual.loc[annual["estimable"]]
        summaries.append(
            {
                "spec_id": spec,
                "domain": domain,
                "model": model,
                "n": len(group),
                "events": int(group["event"].sum()),
                "prevalence": float(group["event"].mean()),
                "pooled_ap": safe_ap(group),
                "macro_year_ap_lift": float(estimable["ap_lift"].mean()),
                "estimable_years": len(estimable),
                "top20_recall_micro": float(
                    annual["top20_tp"].sum() / annual["events"].sum()
                ),
            }
        )
    return year_metrics, pd.DataFrame(summaries)


def two_year_block_indices(n_years: int, replicates: int, rng: np.random.Generator) -> np.ndarray:
    blocks = math.ceil(n_years / 2)
    starts = rng.integers(0, n_years - 1, size=(replicates, blocks))
    return np.stack([starts, starts + 1], axis=-1).reshape(replicates, -1)[:, :n_years]


def bootstrap_metrics(
    year_metrics: pd.DataFrame, config: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    years = np.arange(
        int(config["forecast"]["test_start"]),
        int(config["forecast"]["test_end"]) + 1,
    )
    rng = np.random.default_rng(int(config["forecast"]["seed"]))
    indices = two_year_block_indices(
        len(years), int(config["forecast"]["bootstrap_replicates"]), rng
    )
    estimate_rows = []
    difference_rows = []
    draw_rows = []
    for (spec, domain), source in year_metrics.groupby(["spec_id", "domain"]):
        models = list(FULL_MODELS if domain == "full_kelp_domain" else OISST_MODELS)
        for model in models:
            values = (
                source.loc[source["model"].eq(model)]
                .set_index("year")["ap_lift"]
                .reindex(years)
                .to_numpy(dtype=float)
            )
            distribution = np.nanmean(values[indices], axis=1)
            estimate_rows.append(
                {
                    "spec_id": spec,
                    "domain": domain,
                    "model": model,
                    "estimate": float(np.nanmean(values)),
                    "ci_low": float(np.nanquantile(distribution, 0.025)),
                    "ci_high": float(np.nanquantile(distribution, 0.975)),
                    "estimable_years": int(np.isfinite(values).sum()),
                }
            )
            draw_rows.extend(
                {
                    "spec_id": spec,
                    "domain": domain,
                    "kind": "model",
                    "name": model,
                    "replicate": replicate,
                    "value": float(value),
                }
                for replicate, value in enumerate(distribution)
            )
        comparisons = [
            ("current_plus_trajectory", "current_only", "trajectory_minus_current")
        ]
        if domain == "oisst_matched_domain":
            comparisons.append(
                ("trajectory_plus_oisst", "current_plus_trajectory", "oisst_minus_trajectory")
            )
        for left, right, name in comparisons:
            left_values = (
                source.loc[source["model"].eq(left)]
                .set_index("year")["ap_lift"]
                .reindex(years)
            )
            right_values = (
                source.loc[source["model"].eq(right)]
                .set_index("year")["ap_lift"]
                .reindex(years)
            )
            values = left_values.to_numpy(dtype=float) - right_values.to_numpy(dtype=float)
            distribution = np.nanmean(values[indices], axis=1)
            difference_rows.append(
                {
                    "spec_id": spec,
                    "domain": domain,
                    "comparison": name,
                    "estimate": float(np.nanmean(values)),
                    "ci_low": float(np.nanquantile(distribution, 0.025)),
                    "ci_high": float(np.nanquantile(distribution, 0.975)),
                    "estimable_years": int(np.isfinite(values).sum()),
                    "supported_positive": bool(np.nanquantile(distribution, 0.025) > 0),
                }
            )
            draw_rows.extend(
                {
                    "spec_id": spec,
                    "domain": domain,
                    "kind": "difference",
                    "name": name,
                    "replicate": replicate,
                    "value": float(value),
                }
                for replicate, value in enumerate(distribution)
            )
    return pd.DataFrame(estimate_rows), pd.DataFrame(difference_rows), pd.DataFrame(draw_rows)


def confusion_metrics(tp: int, fp: int, fn: int, tn: int) -> tuple[float, float, float]:
    total = tp + fp + fn + tn
    raw = (tp + tn) / total if total else np.nan
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    if not total:
        return raw, np.nan, jaccard
    expected = (
        (tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)
    ) / (total * total)
    kappa = (raw - expected) / (1 - expected) if expected < 1 else 1.0
    return raw, kappa, jaccard


def pair_spatial_agreement(
    spec: str,
    reference: str,
    panels: dict[str, pd.DataFrame],
    works: dict[str, GridWork],
    predictions: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, object]:
    panel = panels[spec]
    ref_panel = panels[reference]
    mapping = works[spec].stable_pre_mapping
    ref_mapping = works[reference].stable_pre_mapping
    if mapping is None or ref_mapping is None:
        raise ValueError("Stable pixel mapping is missing")
    common_static = (mapping >= 0) & (ref_mapping >= 0)
    tp = fp = fn = tn = 0
    top_intersection = top_union = 0
    common_pixel_years = 0
    prediction_source = predictions.loc[
        predictions["domain"].eq("full_kelp_domain")
        & predictions["model"].eq("current_only")
    ]
    for year in range(
        int(config["forecast"]["test_start"]), int(config["forecast"]["test_end"]) + 1
    ):
        current = panel.loc[panel["year"].eq(year)].set_index("cell_id").reindex(
            works[spec].stable_cell_ids
        )
        ref_current = ref_panel.loc[ref_panel["year"].eq(year)].set_index("cell_id").reindex(
            works[reference].stable_cell_ids
        )
        eligible = current["eligible"].to_numpy(dtype=bool)
        ref_eligible = ref_current["eligible"].to_numpy(dtype=bool)
        common = common_static.copy()
        common[common_static] &= eligible[mapping[common_static]] & ref_eligible[
            ref_mapping[common_static]
        ]
        if not common.any():
            continue
        y = current["event"].to_numpy(dtype=bool)[mapping[common]]
        y_ref = ref_current["event"].to_numpy(dtype=bool)[ref_mapping[common]]
        tp += int(np.sum(y & y_ref))
        fp += int(np.sum(y & ~y_ref))
        fn += int(np.sum(~y & y_ref))
        tn += int(np.sum(~y & ~y_ref))
        common_pixel_years += int(common.sum())

        score = prediction_source.loc[
            prediction_source["spec_id"].eq(spec) & prediction_source["year"].eq(year)
        ].sort_values(["score", "cell_id"], ascending=[False, True])
        ref_score = prediction_source.loc[
            prediction_source["spec_id"].eq(reference)
            & prediction_source["year"].eq(year)
        ].sort_values(["score", "cell_id"], ascending=[False, True])
        selected = set(score.head(max(1, math.ceil(len(score) * 0.20)))["cell_id"])
        ref_selected = set(
            ref_score.head(max(1, math.ceil(len(ref_score) * 0.20)))["cell_id"]
        )
        selected_cells = np.array(
            [cell_id in selected for cell_id in works[spec].stable_cell_ids], dtype=bool
        )
        ref_selected_cells = np.array(
            [cell_id in ref_selected for cell_id in works[reference].stable_cell_ids],
            dtype=bool,
        )
        selected_pixels = selected_cells[mapping[common]]
        ref_selected_pixels = ref_selected_cells[ref_mapping[common]]
        top_intersection += int(np.sum(selected_pixels & ref_selected_pixels))
        top_union += int(np.sum(selected_pixels | ref_selected_pixels))
    raw, kappa, label_jaccard = confusion_metrics(tp, fp, fn, tn)
    return {
        "spec_id": spec,
        "reference_spec": reference,
        "common_pixel_years": common_pixel_years,
        "label_raw_agreement": raw,
        "label_cohen_kappa": kappa,
        "label_positive_jaccard": label_jaccard,
        "top20_spatial_jaccard": top_intersection / top_union if top_union else np.nan,
        "label_tp_pixels": tp,
        "label_fp_pixels": fp,
        "label_fn_pixels": fn,
        "label_tn_pixels": tn,
    }


def spatial_agreements(
    panels: dict[str, pd.DataFrame],
    works: dict[str, GridWork],
    predictions: pd.DataFrame,
    spec_table: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    global_reference = config["primary_reference_spec"]
    rows = []
    for row in spec_table.itertuples(index=False):
        same_size_reference = spec_table.loc[
            spec_table["grid_size_km"].eq(row.grid_size_km)
            & spec_table["threshold_policy"].eq(row.threshold_policy)
            & spec_table["origin_name"].eq("o00"),
            "spec_id",
        ].iloc[0]
        for scope, reference in [
            ("global_10km_o00", global_reference),
            ("same_size_policy_o00", same_size_reference),
        ]:
            result = pair_spatial_agreement(
                row.spec_id, reference, panels, works, predictions, config
            )
            result["reference_scope"] = scope
            rows.append(result)
    return pd.DataFrame(rows)


def reproduce_base_panel(
    base_panel: pd.DataFrame,
    base_inventory: pd.DataFrame,
    reference_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key_to_original = base_inventory.set_index(["grid_x_index", "grid_y_index"])[
        "cell_id"
    ]
    cells = base_panel[
        ["cell_id", "center_lat", "center_lon"]
    ].drop_duplicates("cell_id")
    # The base equal-area grid is globally anchored, so its integer keys can be
    # parsed from the deterministic cell id without spatial tolerance joins.
    parsed = cells["cell_id"].str.extract(r"_x(-?\d+)_y(-?\d+)$").astype(int)
    cells["grid_x_index"] = parsed[0]
    cells["grid_y_index"] = parsed[1]
    cells["original_cell_id"] = [
        key_to_original.get((x, y), np.nan)
        for x, y in zip(cells["grid_x_index"], cells["grid_y_index"], strict=True)
    ]
    mapped = base_panel.merge(
        cells[["cell_id", "original_cell_id"]], on="cell_id", how="left", validate="many_to_one"
    )
    reference = reference_panel.loc[
        reference_panel["variant"].eq("fixedpre_protocol_max")
    ][["cell_id", "year", "annual_area_m2", "event", "eligible"]].rename(
        columns={
            "cell_id": "original_cell_id",
            "annual_area_m2": "reference_area_m2",
            "event": "reference_event",
            "eligible": "reference_eligible",
        }
    )
    joined = mapped.merge(
        reference, on=["original_cell_id", "year"], how="inner", validate="one_to_one"
    )
    area_match = np.isclose(
        joined["annual_area_m2"], joined["reference_area_m2"], atol=0, rtol=0, equal_nan=True
    )
    checks = pd.DataFrame(
        [
            {
                "check": "locked_165_cells_present_in_protocol_base",
                "mismatches": int(reference["original_cell_id"].nunique() - joined["original_cell_id"].nunique()),
            },
            {"check": "base_annual_area_matches_previous_primary", "mismatches": int((~area_match).sum())},
            {"check": "base_event_matches_previous_primary", "mismatches": int(joined["event"].ne(joined["reference_event"]).sum())},
            {"check": "base_eligibility_matches_previous_primary", "mismatches": int(joined["eligible"].astype(bool).ne(joined["reference_eligible"].astype(bool)).sum())},
        ]
    )
    checks["passed"] = checks["mismatches"].eq(0)
    return checks, cells


def build_result_table(
    specs: pd.DataFrame,
    inventory: pd.DataFrame,
    panels: dict[str, pd.DataFrame],
    matches: pd.DataFrame,
    summaries: pd.DataFrame,
    estimates: pd.DataFrame,
    differences: pd.DataFrame,
    agreement: pd.DataFrame,
    config: dict[str, object],
) -> pd.DataFrame:
    rows = []
    for spec_row in specs.itertuples(index=False):
        spec = spec_row.spec_id
        panel = panels[spec]
        test = panel.loc[panel["year"].between(2005, 2024) & panel["eligible"]]
        current = estimates.loc[
            estimates["spec_id"].eq(spec)
            & estimates["domain"].eq("full_kelp_domain")
            & estimates["model"].eq("current_only")
        ].iloc[0]
        trajectory = differences.loc[
            differences["spec_id"].eq(spec)
            & differences["domain"].eq("full_kelp_domain")
            & differences["comparison"].eq("trajectory_minus_current")
        ].iloc[0]
        oisst = differences.loc[
            differences["spec_id"].eq(spec)
            & differences["domain"].eq("oisst_matched_domain")
            & differences["comparison"].eq("oisst_minus_trajectory")
        ].iloc[0]
        current_summary = summaries.loc[
            summaries["spec_id"].eq(spec)
            & summaries["domain"].eq("full_kelp_domain")
            & summaries["model"].eq("current_only")
        ].iloc[0]
        trajectory_summary = summaries.loc[
            summaries["spec_id"].eq(spec)
            & summaries["domain"].eq("full_kelp_domain")
            & summaries["model"].eq("current_plus_trajectory")
        ].iloc[0]
        spatial = agreement.loc[
            agreement["spec_id"].eq(spec)
            & agreement["reference_scope"].eq("global_10km_o00")
        ].iloc[0]
        same_size = agreement.loc[
            agreement["spec_id"].eq(spec)
            & agreement["reference_scope"].eq("same_size_policy_o00")
        ].iloc[0]
        spec_match = matches.loc[matches["spec_id"].eq(spec)]
        rows.append(
            {
                **spec_row._asdict(),
                "candidate_cells": int(inventory.loc[inventory["spec_id"].eq(spec)].shape[0]),
                "stable_cells": int(panel["cell_id"].nunique()),
                "eligible_test_rows": len(test),
                "test_events": int(test["event"].sum()),
                "test_prevalence": float(test["event"].mean()),
                "oisst_supported_cells": int(spec_match["oisst_supported"].sum()),
                "oisst_cell_coverage": float(spec_match["oisst_supported"].mean()),
                "oisst_max_distance_km": float(spec_match["oisst_source_distance_km"].max()),
                "current_lift": float(current.estimate),
                "current_ci_low": float(current.ci_low),
                "current_ci_high": float(current.ci_high),
                "trajectory_increment": float(trajectory.estimate),
                "trajectory_ci_low": float(trajectory.ci_low),
                "trajectory_ci_high": float(trajectory.ci_high),
                "oisst_increment": float(oisst.estimate),
                "oisst_ci_low": float(oisst.ci_low),
                "oisst_ci_high": float(oisst.ci_high),
                "current_top20_recall": float(current_summary.top20_recall_micro),
                "trajectory_top20_recall": float(trajectory_summary.top20_recall_micro),
                "trajectory_top20_recall_delta": float(
                    trajectory_summary.top20_recall_micro - current_summary.top20_recall_micro
                ),
                "global_label_jaccard": float(spatial.label_positive_jaccard),
                "global_top20_spatial_jaccard": float(spatial.top20_spatial_jaccard),
                "same_size_label_jaccard": float(same_size.label_positive_jaccard),
                "same_size_top20_spatial_jaccard": float(same_size.top20_spatial_jaccard),
            }
        )
    return pd.DataFrame(rows)


def origin_spread(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (size, policy), group in results.groupby(["grid_size_km", "threshold_policy"]):
        rows.append(
            {
                "grid_size_km": size,
                "threshold_policy": policy,
                "origin_count": len(group),
                "stable_cells_min": int(group["stable_cells"].min()),
                "stable_cells_max": int(group["stable_cells"].max()),
                "current_lift_min": float(group["current_lift"].min()),
                "current_lift_max": float(group["current_lift"].max()),
                "current_lift_range": float(group["current_lift"].max() - group["current_lift"].min()),
                "trajectory_increment_min": float(group["trajectory_increment"].min()),
                "trajectory_increment_max": float(group["trajectory_increment"].max()),
                "oisst_increment_min": float(group["oisst_increment"].min()),
                "oisst_increment_max": float(group["oisst_increment"].max()),
                "same_size_label_jaccard_min": float(group["same_size_label_jaccard"].min()),
                "same_size_top20_spatial_jaccard_min": float(group["same_size_top20_spatial_jaccard"].min()),
            }
        )
    return pd.DataFrame(rows)


def make_figures(output: Path, results: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    primary = results.loc[results["threshold_policy"].eq("area_scaled")].copy()
    origins = ["o00", "ox50", "oy50", "oxy50"]
    colors = {"o00": "#1565c0", "ox50": "#ef6c00", "oy50": "#546e7a", "oxy50": "#8d6e63"}
    markers = {"o00": "o", "ox50": "s", "oy50": "^", "oxy50": "D"}
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    jitter = {"o00": -0.18, "ox50": -0.06, "oy50": 0.06, "oxy50": 0.18}
    for origin in origins:
        part = primary.loc[primary["origin_name"].eq(origin)].sort_values("grid_size_km")
        x = np.arange(len(part)) + jitter[origin]
        ax.errorbar(
            x,
            part["current_lift"],
            yerr=[part["current_lift"] - part["current_ci_low"], part["current_ci_high"] - part["current_lift"]],
            fmt=markers[origin],
            color=colors[origin],
            capsize=3,
            label=ORIGIN_LABELS[origin],
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(np.arange(3), ["5 km", "10 km", "20 km"])
    ax.set_ylabel("Macro within-year AP lift\n(95% two-year block bootstrap CI)")
    ax.set_title("Current-canopy ranking across grid sizes and origins")
    ax.legend(ncol=2)
    fig.savefig(output / "figure_01_grid_origin_current_lift.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.4), sharex=True, sharey=True)
    size_colors = {5: "#1565c0", 10: "#ef6c00", 20: "#546e7a"}
    for ax, size in zip(axes, [5, 10, 20], strict=True):
        part = primary.loc[primary["grid_size_km"].eq(size)]
        for row in part.itertuples(index=False):
            ax.scatter(
                row.global_label_jaccard,
                row.global_top20_spatial_jaccard,
                s=90,
                color=size_colors[size],
                marker=markers[row.origin_name],
                edgecolor="white",
                linewidth=0.8,
                label=ORIGIN_LABELS[row.origin_name],
            )
        ax.set_xlim(0.70, 1.015)
        ax.set_ylim(0.48, 1.015)
        ax.set_title(f"{size} km cells")
        ax.set_xlabel("Decline-label Jaccard")
    axes[0].set_ylabel("Top-20% selection Jaccard")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Pixel-weighted spatial agreement versus the 10 km base grid", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.80, wspace=0.05)
    fig.savefig(output / "figure_02_spatial_agreement.png", dpi=180)
    plt.close(fig)

    policy = results.loc[results["grid_size_km"].isin([5, 20])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    x_labels = []
    positions = []
    position = 0
    for size in [5, 20]:
        for origin in origins:
            positions.append(position)
            x_labels.append(f"{size}k\n{origin}")
            position += 1
        position += 1
    for policy_name, offset, color in [("area_scaled", -0.14, "#1565c0"), ("fixed_500", 0.14, "#90a4ae")]:
        part = policy.loc[policy["threshold_policy"].eq(policy_name)].copy()
        ordered = []
        for size in [5, 20]:
            for origin in origins:
                ordered.append(part.loc[part["grid_size_km"].eq(size) & part["origin_name"].eq(origin)].iloc[0])
        table = pd.DataFrame(ordered)
        axes[0].bar(np.asarray(positions) + offset, table["stable_cells"], width=0.26, color=color, label=policy_name)
        axes[1].scatter(np.asarray(positions) + offset, table["current_lift"], color=color, label=policy_name)
    axes[0].set_ylabel("Stable cohort cells")
    axes[0].set_title("Cohort size under footprint-threshold policies")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Macro within-year AP lift")
    axes[1].set_title("Current-canopy lift under threshold policies")
    for axis in axes:
        axis.set_xticks(positions, x_labels, rotation=45, ha="right")
        axis.legend()
    fig.savefig(output / "figure_03_cohort_threshold_policy.png", dpi=180)
    plt.close(fig)


def write_summary(
    output: Path,
    results: pd.DataFrame,
    spread: pd.DataFrame,
    reproduction: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, object]:
    primary = results.loc[results["spec_id"].eq(config["primary_reference_spec"])].iloc[0]
    scaled = results.loc[results["threshold_policy"].eq("area_scaled")]
    current_all_supported = bool(scaled["current_ci_low"].gt(0).all())
    trajectory_supported = int(scaled["trajectory_ci_low"].gt(0).sum())
    oisst_supported = int(scaled["oisst_ci_low"].gt(0).sum())
    lowest_label = scaled.loc[scaled["spec_id"].ne(config["primary_reference_spec"]), "global_label_jaccard"].min()
    lowest_top20 = scaled.loc[scaled["spec_id"].ne(config["primary_reference_spec"]), "global_top20_spatial_jaccard"].min()
    max_origin_range = spread.loc[spread["threshold_policy"].eq("area_scaled"), "current_lift_range"].max()
    decisions = {
        "primary_protocol_cells": int(primary.stable_cells),
        "primary_current_lift": float(primary.current_lift),
        "primary_current_ci": [float(primary.current_ci_low), float(primary.current_ci_high)],
        "current_lift_supported_all_12_scaled_specs": current_all_supported,
        "trajectory_supported_scaled_specs": trajectory_supported,
        "oisst_supported_scaled_specs": oisst_supported,
        "scaled_spec_count": len(scaled),
        "maximum_within_size_origin_current_lift_range": float(max_origin_range),
        "minimum_global_label_jaccard": float(lowest_label),
        "minimum_global_top20_spatial_jaccard": float(lowest_top20),
    }
    (output / "decision.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# 격자 크기·원점 민감도 결과",
        "",
        "> 기존 결과를 확인한 뒤 고정한 강건성 분석이다. 독립 확증이나 사전등록 분석으로 표현하지 않는다.",
        "",
        "## 결론부터",
        "",
        f"- 셀 면적에 맞춰 footprint 기준을 5 km=125, 10 km=500, 20 km=2,000 pixels로 조정한 12개 격자에서 현재 상태 AP lift의 95% CI가 모두 0보다 컸다: **{'예' if current_all_supported else '아니오'}**.",
        f"- 기본 10 km 원점의 protocol cohort는 {int(primary.stable_cells)}셀이고 현재 상태 AP lift는 {primary.current_lift:.3f} (95% CI {primary.current_ci_low:.3f}–{primary.current_ci_high:.3f})였다.",
        f"- 같은 크기 안에서 원점을 반 칸 이동했을 때 current lift 최대 범위는 {max_origin_range:.3f}였다.",
        f"- 과거 궤적 증분의 CI가 0보다 큰 경우는 12개 중 {trajectory_supported}개, OISST 증분은 {oisst_supported}개였다. 따라서 추가 데이터 블록의 결론은 현재 상태보다 훨씬 덜 안정적이다.",
        f"- 10 km 기본 격자와 비교한 픽셀 가중 급감라벨 Jaccard 최솟값은 {lowest_label:.3f}, Top-20% 조사공간 Jaccard 최솟값은 {lowest_top20:.3f}였다. 예측 성능이 유지되어도 실제 조사대상 지도는 격자에 따라 달라질 수 있다.",
        "",
        "## 비교가 공정하도록 고정한 것",
        "",
        "- 모든 격자는 EPSG:6933 equal-area 좌표계에서 만들었다.",
        "- 각 크기마다 기본 원점, x 반 칸, y 반 칸, x/y 반 칸 이동의 네 원점을 사용했다.",
        "- 모든 결과는 1984–2004년에 양성 관측된 30 m pixels만 고정 footprint로 사용했다.",
        "- 셀 면적 변화에 따른 cohort-selection 왜곡을 막기 위해 area-scaled threshold를 주 비교로 사용하고, literal 500-pixel 기준은 별도 민감도로 분리했다.",
        "- 모델·라벨·예측연도·expanding window·2년 block bootstrap은 이전 관측품질 실험과 동일하다.",
        "",
        "## 중요한 해석",
        "",
        "1. **RQ1의 현재 상태 순위화는 공간 support 변화에도 유지된다.** 특정 10 km 원점 하나에서만 생긴 결과라는 반론은 상당히 약해졌다.",
        "2. **그러나 우선조사 위치는 완전히 고정되지 않는다.** AP lift와 공간선정 Jaccard는 다른 질문이므로 둘을 분리해 보고해야 한다.",
        "3. **궤적과 OISST 추가가치는 더 보수적으로 쓴다.** 격자별 CI 지지 횟수가 제한적이면 평균 점추정 하나로 일반화하지 않는다.",
        "4. **고정 500 pixels 결과를 주 분석으로 섞지 않는다.** 5 km에서는 지나치게 엄격하고 20 km에서는 느슨해져 서로 다른 habitat-density cohort를 비교하게 된다.",
        "",
        "## 재현과 남은 범위",
        "",
        f"- 이전 165셀 주 패널 재현 검사는 {int(reproduction['passed'].sum())}/{len(reproduction)}개 통과했다. protocol cohort에서는 legacy 선택에 묶여 있던 Southern California 2셀이 추가될 수 있다.",
        "- OISST는 기존 환경파일의 유효 source-grid 좌표 81개에 최근접 매칭한 보조 민감도다. 결측 좌표 표식은 제외했으며, 40 km 지원범위와 매칭거리를 결과표에 공개한다.",
        "- 이 실험은 격자 support 강건성이지 새로운 생태적 외부검증이 아니다.",
    ]
    (output / "results_summary_ko.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decisions


def main() -> None:
    args = parse_args()
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed run: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    specifications = build_specifications(config)
    spec_table = pd.DataFrame(specifications)
    environment = pd.read_csv(args.environment)
    base_inventory = pd.read_csv(args.base_inventory)
    reference_panel = pd.read_csv(args.base_primary_panel)

    with Dataset(args.netcdf) as dataset:
        source_indices, longitude, latitude, x, y, prepositive = read_coordinates_and_pre2005_mask(
            dataset, config
        )
        years_all = np.asarray(dataset.variables["year"][:], dtype=int)
        quarters_all = np.asarray(dataset.variables["quarter"][:], dtype=int)
        time_indices = np.flatnonzero(years_all <= int(config["forecast"]["panel_end"]))
        years = years_all[time_indices]
        quarters = quarters_all[time_indices]
        backward = Transformer.from_crs(
            config["grid_crs"], config["source_crs"], always_xy=True
        )
        works_list = [
            build_grid_work(
                spec, x[prepositive], y[prepositive], len(time_indices), backward
            )
            for spec in specifications
        ]
        aggregate_all_grids(dataset, source_indices, prepositive, works_list, time_indices)

    inventories = []
    raw_panels: dict[str, pd.DataFrame] = {}
    works = {work.spec_id: work for work in works_list}
    for work in works_list:
        inventory, panel = annualize_and_select_cohort(work, years, quarters, config)
        inventories.append(inventory)
        raw_panels[work.spec_id] = panel
    inventory = pd.concat(inventories, ignore_index=True)

    sources, source_year = prepare_oisst_sources(environment)
    panels: dict[str, pd.DataFrame] = {}
    matches = []
    for spec, panel in raw_panels.items():
        merged, match = attach_oisst(
            panel,
            sources,
            source_year,
            float(config["oisst"]["maximum_nearest_source_distance_km"]),
        )
        panels[spec] = merged
        matches.append(match)
    matches = pd.concat(matches, ignore_index=True)

    base_spec = config["primary_reference_spec"]
    reproduction, base_cells = reproduce_base_panel(
        panels[base_spec], base_inventory, reference_panel
    )
    if not reproduction["passed"].all():
        raise AssertionError(reproduction.to_string(index=False))

    predictions, folds = run_backtests(panels, config)
    year_metrics, summaries = calculate_metrics(predictions)
    estimates, differences, draws = bootstrap_metrics(year_metrics, config)
    agreement = spatial_agreements(panels, works, predictions, spec_table, config)
    results = build_result_table(
        spec_table,
        inventory,
        panels,
        matches,
        summaries,
        estimates,
        differences,
        agreement,
        config,
    )
    spread = origin_spread(results)

    key_counts = predictions.groupby(["spec_id", "domain", "model"])[
        ["cell_id", "year"]
    ].size().reset_index(name="rows")
    quality_checks = pd.DataFrame(
        [
            {"check": "expected_grid_specifications", "value": len(spec_table), "passed": len(spec_table) == 20},
            {"check": "coordinate_rows_unique", "value": len(source_indices), "passed": len(source_indices) == len(np.unique(np.column_stack([longitude, latitude]), axis=0))},
            {"check": "pre2005_positive_pixels", "value": int(prepositive.sum()), "passed": int(prepositive.sum()) > 500_000},
            {"check": "panel_keys_unique", "value": int(sum(panel.duplicated(["cell_id", "year"]).sum() for panel in panels.values())), "passed": all(not panel.duplicated(["cell_id", "year"]).any() for panel in panels.values())},
            {"check": "forward_only_folds", "value": int((folds["train_end"] >= folds["year"]).sum()), "passed": bool((folds["train_end"] < folds["year"]).all())},
            {"check": "prediction_scores_complete", "value": int(predictions["score"].isna().sum()), "passed": not predictions["score"].isna().any()},
            {"check": "prediction_keys_unique", "value": int(predictions.duplicated(["spec_id", "domain", "model", "cell_id", "year"]).sum()), "passed": not predictions.duplicated(["spec_id", "domain", "model", "cell_id", "year"]).any()},
            {"check": "model_rows_matched_within_domain", "value": int(key_counts.groupby(["spec_id", "domain"])["rows"].nunique().max()), "passed": bool(key_counts.groupby(["spec_id", "domain"])["rows"].nunique().eq(1).all())},
            {"check": "base_locked_panel_reproduction", "value": int(reproduction["mismatches"].sum()), "passed": bool(reproduction["passed"].all())},
            {"check": "all_specs_have_20_forecast_years", "value": int(year_metrics.groupby(["spec_id", "domain", "model"])["year"].nunique().min()), "passed": bool(year_metrics.groupby(["spec_id", "domain", "model"])["year"].nunique().eq(20).all())},
            {"check": "oisst_source_grid_coordinates_valid", "value": len(sources), "passed": bool(len(sources) == 81 and sources[["oisst_source_lat", "oisst_source_lon"]].notna().all().all())},
        ]
    )
    if not quality_checks["passed"].all():
        raise AssertionError(quality_checks.to_string(index=False))

    outputs = {
        "grid_specifications.csv": spec_table,
        "grid_inventory.csv": inventory,
        "grid_panels.csv": pd.concat(panels.values(), ignore_index=True),
        "oisst_cell_matches.csv": matches,
        "base_reproduction_checks.csv": reproduction,
        "base_protocol_cell_mapping.csv": base_cells,
        "predictions.csv": predictions,
        "fold_audit.csv": folds,
        "year_metrics.csv": year_metrics,
        "model_summary.csv": summaries,
        "bootstrap_estimates.csv": estimates,
        "bootstrap_differences.csv": differences,
        "bootstrap_draws.csv": draws,
        "pixel_spatial_agreement.csv": agreement,
        "grid_sensitivity_results.csv": results,
        "origin_spread_summary.csv": spread,
        "quality_checks.csv": quality_checks,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)
    make_figures(args.output_dir, results)
    decisions = write_summary(args.output_dir, results, spread, reproduction, config)
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
        "seed": int(config["forecast"]["seed"]),
        "bootstrap_replicates": int(config["forecast"]["bootstrap_replicates"]),
        "grid_specifications": len(spec_table),
        "pre2005_positive_station_pixels": int(prepositive.sum()),
        "inputs": {
            "config": {"path": str(args.config), "sha256": sha256(args.config)},
            "netcdf": {"path": str(args.netcdf), "sha256": sha256(args.netcdf)},
            "environment": {"path": str(args.environment), "sha256": sha256(args.environment)},
            "base_inventory": {"path": str(args.base_inventory), "sha256": sha256(args.base_inventory)},
            "base_primary_panel": {"path": str(args.base_primary_panel), "sha256": sha256(args.base_primary_panel)},
        },
        "decisions": decisions,
        "output_sha256": {name: sha256(args.output_dir / name) for name in outputs},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decisions, indent=2))


if __name__ == "__main__":
    main()
