"""Run the Giraldo California field-data error-diagnostic case study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score


AXES = [
    "grazing_log1p",
    "rock_probability",
    "depth_m",
    "kelp_species_contrast",
]
AXIS_LABELS = {
    "grazing_log1p": "Urchin grazing pressure",
    "rock_probability": "Rock probability",
    "depth_m": "Depth",
    "kelp_species_contrast": "Bull-vs-giant kelp contrast",
}
FEATURE_SETS = [
    "current_only",
    "current_plus_trajectory",
    "trajectory_plus_oisst",
]
FEATURE_LABELS = {
    "current_only": "Current canopy",
    "current_plus_trajectory": "Current + trajectory",
    "trajectory_plus_oisst": "Current + trajectory + OISST",
}
OUTCOME_LABELS = {
    "trajectory_rank_quality_pp": "Trajectory rank quality (pp)",
    "trajectory_false_negative": "False-negative probability",
    "trajectory_false_positive": "False-positive probability",
    "oisst_signed_rank_gain_pp": "OISST signed rank gain (pp)",
    "oisst_brier_gain": "OISST Brier-score improvement",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field",
        type=Path,
        default=Path(
            "data/external/restoration_site_selection/giraldo_2025/"
            "Kelp_Predictors_All_CA_2.csv"
        ),
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/processed/kelpwatch_west_coast_panel_165.csv"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "outputs/experiments/20260814_journal_extension_v1/"
            "expanding_predictions.csv"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/giraldo_error_case_study_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/experiments/20260816_giraldo_error_case_study_v1"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.SubprocessError:
        return "unknown"


def haversine_matrix_km(
    latitude: np.ndarray,
    longitude: np.ndarray,
    center_latitude: np.ndarray,
    center_longitude: np.ndarray,
) -> np.ndarray:
    lat1 = np.radians(latitude)[:, None]
    lon1 = np.radians(longitude)[:, None]
    lat2 = np.radians(center_latitude)[None, :]
    lon2 = np.radians(center_longitude)[None, :]
    haversine = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(haversine))


def iqr(series: pd.Series) -> float:
    return float(series.quantile(0.75) - series.quantile(0.25))


def prepare_field_site_years(field: pd.DataFrame) -> pd.DataFrame:
    required = {
        "site_campus_unique_ID",
        "survey_year",
        "latitude",
        "longitude",
        "transect",
        "den_STRPURAD",
        "den_MESFRAAD",
        "den_MACPYRAD",
        "den_NERLUE",
        "mean_prob_of_rock",
        "mean_depth",
    }
    missing = sorted(required.difference(field.columns))
    if missing:
        raise ValueError(f"Giraldo field table is missing required columns: {missing}")
    transformed = field[list(required)].copy()
    transformed["grazing_log1p"] = np.log1p(
        transformed[["den_STRPURAD", "den_MESFRAAD"]].sum(
            axis=1, min_count=1
        )
    )
    transformed["rock_probability"] = transformed["mean_prob_of_rock"]
    transformed["depth_m"] = -transformed["mean_depth"]
    transformed["kelp_species_contrast"] = np.log1p(
        transformed["den_NERLUE"]
    ) - np.log1p(transformed["den_MACPYRAD"])
    aggregations: dict[str, str | tuple[str, str]] = {
        "latitude": "median",
        "longitude": "median",
        **{axis: "median" for axis in AXES},
    }
    site_year = (
        transformed.groupby(
            ["site_campus_unique_ID", "survey_year"], as_index=False
        )
        .agg(aggregations)
        .merge(
            transformed.groupby(
                ["site_campus_unique_ID", "survey_year"], as_index=False
            ).size().rename(columns={"size": "n_transects"}),
            on=["site_campus_unique_ID", "survey_year"],
            validate="one_to_one",
        )
    )
    return site_year


def match_and_aggregate_field(
    site_year: pd.DataFrame,
    cells: pd.DataFrame,
    max_distance_km: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = site_year.dropna(subset=["latitude", "longitude"]).copy()
    distances = haversine_matrix_km(
        valid["latitude"].to_numpy(),
        valid["longitude"].to_numpy(),
        cells["center_lat"].to_numpy(),
        cells["center_lon"].to_numpy(),
    )
    nearest_index = distances.argmin(axis=1)
    valid["cell_id"] = cells["cell_id"].to_numpy()[nearest_index]
    valid["region_group"] = cells["region_group"].to_numpy()[nearest_index]
    valid["match_distance_km"] = distances[
        np.arange(len(valid)), nearest_index
    ]
    matched = valid.loc[valid["match_distance_km"].le(max_distance_km)].copy()

    rows: list[dict[str, object]] = []
    for (cell_id, year), group in matched.groupby(
        ["cell_id", "survey_year"], sort=True
    ):
        row: dict[str, object] = {
            "cell_id": cell_id,
            "year": int(year),
            "region_group": group["region_group"].iloc[0],
            "n_sites": int(group["site_campus_unique_ID"].nunique()),
            "n_transects": int(group["n_transects"].sum()),
            "median_match_distance_km": float(group["match_distance_km"].median()),
            "max_match_distance_km": float(group["match_distance_km"].max()),
        }
        for axis in AXES:
            row[axis] = float(group[axis].median())
            row[f"{axis}_iqr"] = iqr(group[axis])
        rows.append(row)
    return matched, pd.DataFrame(rows)


def add_ranks_and_budget_flags(
    predictions: pd.DataFrame, budget_fraction: float
) -> pd.DataFrame:
    rows = []
    group_keys = ["model_family", "feature_set", "year"]
    for _, group in predictions.groupby(group_keys, sort=True):
        ranked = group.sort_values(
            ["score", "cell_id"], ascending=[False, True]
        ).copy()
        selected_count = max(1, int(math.ceil(len(ranked) * budget_fraction)))
        ranked["selected_at_budget"] = False
        ranked.iloc[
            :selected_count, ranked.columns.get_loc("selected_at_budget")
        ] = True
        ranked["risk_percentile"] = ranked["score"].rank(
            method="average", pct=True
        )
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True)


def prepare_prediction_diagnostics(
    predictions: pd.DataFrame,
    first_year: int,
    last_year: int,
    domain: str,
    budget_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "cell_id",
        "region_group",
        "year",
        "event",
        "score",
        "model_family",
        "feature_set",
        "domain",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing required columns: {missing}")
    filtered = predictions.loc[
        predictions["domain"].eq(domain)
        & predictions["year"].between(first_year, last_year)
        & predictions["feature_set"].isin(FEATURE_SETS)
    ].copy()
    prediction_key = [
        "cell_id",
        "year",
        "model_family",
        "feature_set",
    ]
    if filtered.duplicated(prediction_key).any():
        raise ValueError("Out-of-fold prediction keys are not unique")
    ranked = add_ranks_and_budget_flags(filtered, budget_fraction)
    wide = ranked.pivot(
        index=["cell_id", "region_group", "year", "event", "model_family"],
        columns="feature_set",
        values=["score", "risk_percentile", "selected_at_budget"],
    ).reset_index()
    wide.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in wide.columns
    ]
    rename: dict[str, str] = {}
    short_names = {
        "current_only": "current",
        "current_plus_trajectory": "trajectory",
        "trajectory_plus_oisst": "oisst",
    }
    for full_name, short_name in short_names.items():
        for metric in ["score", "risk_percentile", "selected_at_budget"]:
            rename[f"{metric}_{full_name}"] = f"{metric}_{short_name}"
    wide = wide.rename(columns=rename)
    for feature in ["current", "trajectory", "oisst"]:
        wide[f"false_negative_{feature}"] = (
            wide["event"].eq(1)
            & ~wide[f"selected_at_budget_{feature}"].astype(bool)
        ).astype(int)
        wide[f"false_positive_{feature}"] = (
            wide["event"].eq(0)
            & wide[f"selected_at_budget_{feature}"].astype(bool)
        ).astype(int)
    signed_event = 2 * wide["event"] - 1
    wide["trajectory_rank_quality_pp"] = (
        signed_event * (wide["risk_percentile_trajectory"] - 0.5) * 100
    )
    wide["oisst_signed_rank_gain_pp"] = (
        signed_event
        * (
            wide["risk_percentile_oisst"]
            - wide["risk_percentile_trajectory"]
        )
        * 100
    )
    wide["oisst_brier_gain"] = (
        (wide["event"] - wide["score_trajectory"]) ** 2
        - (wide["event"] - wide["score_oisst"]) ** 2
    )
    wide["trajectory_false_negative"] = wide[
        "false_negative_trajectory"
    ]
    wide["trajectory_false_positive"] = wide[
        "false_positive_trajectory"
    ]
    return ranked, wide


def safe_ap_lift(group: pd.DataFrame) -> float:
    if group.empty or group["event"].nunique() < 2:
        return np.nan
    return float(average_precision_score(group["event"], group["score"])) - float(
        group["event"].mean()
    )


def bootstrap_mean(
    values: np.ndarray, replicates: int, rng: np.random.Generator
) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan
    samples = rng.choice(finite, size=(replicates, finite.size), replace=True).mean(
        axis=1
    )
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def performance_by_scope(
    ranked_predictions: pd.DataFrame,
    matched_keys: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matched_key_set = matched_keys[["cell_id", "year"]].drop_duplicates()
    scoped = {
        "full_165_domain": ranked_predictions.copy(),
        "giraldo_matched": ranked_predictions.merge(
            matched_key_set, on=["cell_id", "year"], how="inner"
        ),
    }
    year_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    increment_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for scope, frame in scoped.items():
        for keys, group in frame.groupby(
            ["model_family", "feature_set", "year"], sort=True
        ):
            year_rows.append(
                {
                    "scope": scope,
                    "model_family": keys[0],
                    "feature_set": keys[1],
                    "year": int(keys[2]),
                    "n": len(group),
                    "events": int(group["event"].sum()),
                    "prevalence": float(group["event"].mean()),
                    "ap_lift": safe_ap_lift(group),
                }
            )
    annual = pd.DataFrame(year_rows)
    for keys, group in annual.groupby(
        ["scope", "model_family", "feature_set"], sort=True
    ):
        values = group["ap_lift"].dropna().to_numpy()
        ci_low, ci_high = bootstrap_mean(values, replicates, rng)
        summary_rows.append(
            {
                "scope": keys[0],
                "model_family": keys[1],
                "feature_set": keys[2],
                "estimable_years": len(values),
                "macro_within_year_ap_lift": float(values.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    comparisons = [
        (
            "trajectory_minus_current",
            "current_plus_trajectory",
            "current_only",
        ),
        (
            "oisst_minus_trajectory",
            "trajectory_plus_oisst",
            "current_plus_trajectory",
        ),
    ]
    for (scope, family), group in annual.groupby(
        ["scope", "model_family"], sort=True
    ):
        pivot = group.pivot(index="year", columns="feature_set", values="ap_lift")
        for comparison, left, right in comparisons:
            paired = (pivot[left] - pivot[right]).dropna().to_numpy()
            ci_low, ci_high = bootstrap_mean(paired, replicates, rng)
            increment_rows.append(
                {
                    "scope": scope,
                    "model_family": family,
                    "comparison": comparison,
                    "left_feature_set": left,
                    "right_feature_set": right,
                    "paired_years": len(paired),
                    "estimate": float(paired.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )
    return annual, pd.DataFrame(summary_rows), pd.DataFrame(increment_rows)


def cluster_meat(
    design: np.ndarray,
    residuals: np.ndarray,
    groups: pd.Series,
) -> tuple[np.ndarray, int]:
    n, parameters = design.shape
    reset_groups = groups.reset_index(drop=True)
    meat = np.zeros((parameters, parameters))
    for positions in reset_groups.groupby(reset_groups, sort=False).groups.values():
        index = np.fromiter(positions, dtype=int)
        score = design[index].T @ residuals[index]
        meat += np.outer(score, score)
    cluster_count = int(reset_groups.nunique())
    if cluster_count > 1 and n > parameters:
        meat *= (cluster_count / (cluster_count - 1)) * (
            (n - 1) / (n - parameters)
        )
    return meat, cluster_count


def two_way_cluster_ols(
    frame: pd.DataFrame,
    outcome: str,
    axis: str,
) -> dict[str, object]:
    columns = [outcome, axis, "cell_id", "year", "region_group", "event"]
    data = frame[columns].dropna().reset_index(drop=True)
    axis_sd = float(data[axis].std(ddof=0))
    if not np.isfinite(axis_sd) or axis_sd == 0:
        raise ValueError(f"Axis {axis} has no usable variation for {outcome}")
    data["axis_z"] = (data[axis] - data[axis].mean()) / axis_sd
    design_frame = pd.concat(
        [
            pd.Series(1.0, index=data.index, name="intercept"),
            data[["axis_z"]],
            pd.get_dummies(
                data["region_group"], prefix="region", drop_first=True, dtype=float
            ),
            pd.get_dummies(data["year"], prefix="year", drop_first=True, dtype=float),
        ],
        axis=1,
    )
    design = design_frame.to_numpy(dtype=float)
    response = data[outcome].to_numpy(dtype=float)
    bread = np.linalg.pinv(design.T @ design)
    coefficients = bread @ design.T @ response
    residuals = response - design @ coefficients
    cell_meat, cell_clusters = cluster_meat(
        design, residuals, data["cell_id"]
    )
    year_meat, year_clusters = cluster_meat(design, residuals, data["year"])
    observation_meat, _ = cluster_meat(
        design, residuals, pd.Series(np.arange(len(data)))
    )
    covariance = bread @ (cell_meat + year_meat - observation_meat) @ bread
    variance = max(0.0, float(covariance[1, 1]))
    standard_error = math.sqrt(variance)
    degrees_freedom = max(1, min(cell_clusters, year_clusters) - 1)
    critical = float(student_t.ppf(0.975, degrees_freedom))
    estimate = float(coefficients[1])
    if standard_error > 0:
        statistic = estimate / standard_error
        p_value = float(
            2 * student_t.sf(abs(statistic), degrees_freedom)
        )
    else:
        statistic = np.nan
        p_value = np.nan
    return {
        "n": len(data),
        "cells": int(data["cell_id"].nunique()),
        "years": int(data["year"].nunique()),
        "events": int(data["event"].sum()),
        "axis_mean": float(data[axis].mean()),
        "axis_sd": axis_sd,
        "estimate_per_sd": estimate,
        "standard_error": standard_error,
        "ci_low": estimate - critical * standard_error,
        "ci_high": estimate + critical * standard_error,
        "t_statistic": statistic,
        "degrees_freedom": degrees_freedom,
        "p_value": p_value,
        "cell_clusters": cell_clusters,
        "year_clusters": year_clusters,
    }


def benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    finite = values.dropna().sort_values()
    count = len(finite)
    if count == 0:
        return result
    adjusted = finite.to_numpy() * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[finite.index] = np.minimum(adjusted, 1.0)
    return result


def association_table(
    analysis: pd.DataFrame,
    scope: str,
    q_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, family_frame in analysis.groupby("model_family", sort=True):
        for outcome in OUTCOME_LABELS:
            outcome_frame = family_frame
            if outcome == "trajectory_false_negative":
                outcome_frame = outcome_frame.loc[outcome_frame["event"].eq(1)]
            elif outcome == "trajectory_false_positive":
                outcome_frame = outcome_frame.loc[outcome_frame["event"].eq(0)]
            for axis in AXES:
                estimate = two_way_cluster_ols(outcome_frame, outcome, axis)
                rows.append(
                    {
                        "scope": scope,
                        "model_family": family,
                        "outcome": outcome,
                        "outcome_label": OUTCOME_LABELS[outcome],
                        "axis": axis,
                        "axis_label": AXIS_LABELS[axis],
                        **estimate,
                    }
                )
    results = pd.DataFrame(rows)
    results["q_value"] = np.nan
    group_columns = ["scope", "model_family", "outcome"]
    for _, index in results.groupby(group_columns).groups.items():
        results.loc[index, "q_value"] = benjamini_hochberg(
            results.loc[index, "p_value"]
        )
    results["ci_excludes_zero"] = (
        results["ci_low"].gt(0) | results["ci_high"].lt(0)
    )
    results["supported_bh"] = results["ci_excludes_zero"] & results[
        "q_value"
    ].lt(q_threshold)
    return results


def build_condition_support(
    associations: pd.DataFrame,
    q_threshold: float,
) -> pd.DataFrame:
    primary = associations.loc[
        associations["scope"].eq("primary_7p2km")
        & associations["model_family"].eq("logistic")
    ].copy()
    rows: list[dict[str, object]] = []
    for row in primary.itertuples(index=False):
        same_primary = associations.loc[
            associations["scope"].eq("primary_7p2km")
            & associations["outcome"].eq(row.outcome)
            & associations["axis"].eq(row.axis)
        ]
        direction = np.sign(row.estimate_per_sd)
        same_direction_models = int(
            (np.sign(same_primary["estimate_per_sd"]) == direction).sum()
        )
        strict = associations.loc[
            associations["scope"].eq("strict_5km")
            & associations["model_family"].eq("logistic")
            & associations["outcome"].eq(row.outcome)
            & associations["axis"].eq(row.axis)
        ].iloc[0]
        strict_same_direction = bool(
            np.sign(strict["estimate_per_sd"]) == direction
        )
        stable = bool(
            row.ci_excludes_zero
            and row.q_value < q_threshold
            and same_direction_models >= 2
            and strict_same_direction
        )
        rows.append(
            {
                "outcome": row.outcome,
                "axis": row.axis,
                "primary_estimate_per_sd": row.estimate_per_sd,
                "primary_ci_low": row.ci_low,
                "primary_ci_high": row.ci_high,
                "primary_q_value": row.q_value,
                "same_direction_model_count": same_direction_models,
                "strict_5km_estimate_per_sd": strict["estimate_per_sd"],
                "strict_same_direction": strict_same_direction,
                "stable_condition_supported": stable,
            }
        )
    return pd.DataFrame(rows)


def repeated_error_cells(
    analysis: pd.DataFrame, cells: pd.DataFrame, primary_family: str
) -> pd.DataFrame:
    primary = analysis.loc[analysis["model_family"].eq(primary_family)].copy()
    aggregations: dict[str, tuple[str, object]] = {
        "observed_years": ("year", "nunique"),
        "event_opportunities": ("event", "sum"),
        "trajectory_false_negatives": ("trajectory_false_negative", "sum"),
        "trajectory_false_positives": ("trajectory_false_positive", "sum"),
    }
    for axis in AXES:
        aggregations[f"mean_{axis}"] = (axis, "mean")
    summary = (
        primary.groupby(["cell_id", "region_group"], as_index=False)
        .agg(**aggregations)
        .merge(
            cells[["cell_id", "center_lat", "center_lon"]],
            on="cell_id",
            validate="one_to_one",
        )
    )
    summary["non_event_opportunities"] = (
        summary["observed_years"] - summary["event_opportunities"]
    )
    summary["trajectory_false_negative_rate"] = summary[
        "trajectory_false_negatives"
    ] / summary["event_opportunities"].replace(0, np.nan)
    summary["trajectory_false_positive_rate"] = summary[
        "trajectory_false_positives"
    ] / summary["non_event_opportunities"].replace(0, np.nan)
    summary["repeated_false_negative"] = summary[
        "trajectory_false_negatives"
    ].ge(2)
    summary["repeated_false_positive"] = summary[
        "trajectory_false_positives"
    ].ge(2)
    return summary


def plot_coverage(
    cells: pd.DataFrame,
    matched_site_years: pd.DataFrame,
    matched_cell_years: pd.DataFrame,
    output: Path,
) -> None:
    matched_cells = set(matched_cell_years["cell_id"])
    covered = cells["cell_id"].isin(matched_cells)
    fig, ax = plt.subplots(figsize=(7.6, 10.5))
    ax.scatter(
        cells.loc[~covered, "center_lon"],
        cells.loc[~covered, "center_lat"],
        s=24,
        facecolors="none",
        edgecolors="#A6ADB4",
        linewidths=0.8,
        label="165-cell domain without matched field data",
    )
    ax.scatter(
        cells.loc[covered, "center_lon"],
        cells.loc[covered, "center_lat"],
        s=38,
        c="#3F6F8F",
        edgecolors="#16384C",
        linewidths=0.5,
        label="Matched 10-km cells",
    )
    unique_sites = matched_site_years.sort_values("survey_year").drop_duplicates(
        "site_campus_unique_ID", keep="last"
    )
    ax.scatter(
        unique_sites["longitude"],
        unique_sites["latitude"],
        s=8,
        c="#D39A2C",
        alpha=0.65,
        linewidths=0,
        label="Giraldo field sites",
    )
    ax.set_title("Giraldo field-data coverage in the 165-cell study domain", loc="left")
    fig.text(
        0.11,
        0.925,
        "Primary match: nearest cell center within 7.2 km; field data end in 2021",
        fontsize=9,
        color="#4C5358",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(color="#D9DDE0", linewidth=0.5, alpha=0.6)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_associations(associations: pd.DataFrame, output: Path) -> None:
    outcomes = [
        "trajectory_rank_quality_pp",
        "trajectory_false_negative",
        "trajectory_false_positive",
        "oisst_signed_rank_gain_pp",
    ]
    primary = associations.loc[
        associations["scope"].eq("primary_7p2km")
        & associations["model_family"].eq("logistic")
        & associations["outcome"].isin(outcomes)
    ].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))
    for ax, outcome in zip(axes.flat, outcomes, strict=True):
        plot_data = primary.loc[primary["outcome"].eq(outcome)].set_index("axis").loc[
            AXES
        ].reset_index()
        scale = 100.0 if outcome in {
            "trajectory_false_negative",
            "trajectory_false_positive",
        } else 1.0
        y = np.arange(len(plot_data))
        estimate = plot_data["estimate_per_sd"].to_numpy() * scale
        low = plot_data["ci_low"].to_numpy() * scale
        high = plot_data["ci_high"].to_numpy() * scale
        ax.errorbar(
            estimate,
            y,
            xerr=np.vstack([estimate - low, high - estimate]),
            fmt="o",
            color="#3F6F8F",
            ecolor="#7E98AA",
            capsize=3,
            markersize=5,
        )
        ax.axvline(0, color="#33383C", linewidth=0.8, linestyle="--")
        ax.set_yticks(y, plot_data["axis_label"])
        ax.invert_yaxis()
        ax.set_title(OUTCOME_LABELS[outcome], loc="left", fontsize=10)
        ax.set_xlabel("Change per 1 SD of the field axis" + (" (pp)" if scale == 100 else ""))
        ax.grid(axis="x", color="#D9DDE0", linewidth=0.5)
    fig.suptitle(
        "Ecological-axis associations with out-of-fold prediction diagnostics",
        x=0.07,
        ha="left",
        fontsize=14,
    )
    fig.text(
        0.07,
        0.93,
        "Logistic model; region and year fixed effects; two-way cell/year clustered 95% intervals",
        fontsize=9,
        color="#4C5358",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_repeated_errors(summary: pd.DataFrame, output: Path) -> None:
    plot_data = summary.loc[summary["event_opportunities"].ge(1)].copy()
    colors = {
        "Northern California": "#8A6F3D",
        "Central California": "#3F6F8F",
        "Southern California": "#A55C45",
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for region, group in plot_data.groupby("region_group", sort=True):
        ax.scatter(
            group["mean_grazing_log1p"],
            group["trajectory_false_negative_rate"],
            s=25 + 12 * group["event_opportunities"],
            c=colors.get(region, "#777777"),
            alpha=0.75,
            edgecolors="#FFFFFF",
            linewidths=0.6,
            label=region,
        )
    ax.set_title("Trajectory-model false-negative rate and surveyed urchin pressure", loc="left")
    fig.text(
        0.11,
        0.92,
        "Cell-level summary; marker area increases with the number of observed decline events",
        fontsize=9,
        color="#4C5358",
    )
    ax.set_xlabel("Mean log1p(red + purple urchin density)")
    ax.set_ylabel("False-negative rate at a 20% annual monitoring budget")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(color="#D9DDE0", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cohort_performance(summary: pd.DataFrame, output: Path) -> None:
    data = summary.loc[summary["model_family"].eq("logistic")].copy()
    scope_order = ["full_165_domain", "giraldo_matched"]
    feature_order = FEATURE_SETS
    colors = {"full_165_domain": "#A6ADB4", "giraldo_matched": "#3F6F8F"}
    offsets = {"full_165_domain": -0.12, "giraldo_matched": 0.12}
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for scope in scope_order:
        subset = data.loc[data["scope"].eq(scope)].set_index("feature_set").loc[
            feature_order
        ]
        x = np.arange(len(feature_order)) + offsets[scope]
        estimate = subset["macro_within_year_ap_lift"].to_numpy()
        ax.errorbar(
            x,
            estimate,
            yerr=np.vstack(
                [
                    estimate - subset["ci_low"].to_numpy(),
                    subset["ci_high"].to_numpy() - estimate,
                ]
            ),
            fmt="o",
            color=colors[scope],
            capsize=3,
            label=("Full 165-cell domain" if scope == "full_165_domain" else "Giraldo-matched subset"),
        )
    ax.set_xticks(np.arange(len(feature_order)), [FEATURE_LABELS[x] for x in feature_order])
    ax.set_ylabel("Macro within-year AP lift")
    ax.set_title("Prediction performance in the full and field-observed cohorts", loc="left")
    fig.text(
        0.11,
        0.92,
        "Logistic expanding-window predictions, 2005–2021; intervals resample forecast years",
        fontsize=9,
        color="#4C5358",
    )
    ax.grid(axis="y", color="#D9DDE0", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_results_summary(
    output: Path,
    decision: dict[str, object],
) -> None:
    matched = decision["matched_population"]
    increments = decision["matched_subset_increments"]
    logistic_oisst = increments["logistic"]["oisst_minus_trajectory"]
    lines = [
        "# Giraldo 현장자료 기반 오류진단 보조 사례연구 결과",
        "",
        "## 결론",
        "",
        "Giraldo 현장자료가 존재하는 California 부분표본에서도 공개 OISST의 추가가치나 기존 궤적모형의 반복오류를 안정적으로 설명하는 단일 생태축을 확인하지 못했다. 이 결과는 환경 또는 생태조건이 중요하지 않다는 증거가 아니라, 불균형한 현장표본과 60㎡–10 km 공간 support 불일치 아래에서 해당 네 축의 조건부 정보가치를 식별하지 못했다는 뜻이다.",
        "",
        "## 분석범위",
        "",
        f"- 주 결합: {matched['cells']}셀, {matched['cell_years']} cell-year, {matched['events']}개 급감 사건",
        f"- 기간: {matched['first_year']}–{matched['last_year']}",
        f"- 지역: 북부 {matched['northern_cell_years']}, 중부 {matched['central_cell_years']}, 남부 {matched['southern_cell_years']} cell-year",
        "- Oregon, Washington 및 Baja 현장자료는 포함되지 않았다.",
        "",
        "## 조건부 OISST 가치",
        "",
        f"Giraldo matched 표본에서 Logistic OISST AP-lift 증분은 {logistic_oisst['estimate']:+.4f}였고 연도 bootstrap 95% 구간은 [{logistic_oisst['ci_low']:+.4f}, {logistic_oisst['ci_high']:+.4f}]였다. 세 모형군 모두 신뢰구간이 0을 포함했다. 따라서 현장조사가 존재하는 California 표본에서 OISST가 평균적으로 더 유용하다는 결론은 지지되지 않았다.",
        "",
        "## 생태적 오류조건",
        "",
        f"성게 방목압, 암반확률, 수심, bull–giant kelp 구성의 네 축과 다섯 오류·증분가치 결과를 사전 고정해 검사했다. 안정성 기준을 통과한 조합은 {decision['stable_condition_count']}개였다. 즉 특정 생태조건을 이용해 공개 환경모형의 실패를 재현 가능하게 구분하지 못했다.",
        "",
        "## 논문에서 가능한 표현",
        "",
        "> California의 제한된 현장 관측과 결합한 탐색적 오류분석에서도 성게 방목압, 암반 서식처, 수심 및 켈프 종 구성에 따른 환경 프록시의 안정적인 조건부 증분가치는 확인되지 않았다. 이 음성결과는 현장자료의 생태적 무관성보다 불균형한 표본선택, 작은 유효 공간복제 및 현장 transect와 10 km 의사결정 셀 간 support 불일치로 해석해야 한다.",
        "",
        "## 금지되는 해석",
        "",
        "- 성게가 켈프 급감에 중요하지 않다고 결론내리지 않는다.",
        "- Giraldo 현장자료가 전 해안 예측을 개선하지 않는다고 일반화하지 않는다.",
        "- 52셀 부분표본의 성능을 165셀 전체 성능처럼 제시하지 않는다.",
        "- 현장자료를 주 정보블록 B5로 표현하지 않는다.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty run directory: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed = int(config["reproducibility"]["random_seed"])
    replicates = int(config["reproducibility"]["year_bootstrap_replicates"])
    first_year = int(config["population"]["first_forecast_year"])
    last_year = int(config["population"]["last_field_year"])
    domain = config["population"]["prediction_domain"]
    primary_distance = float(
        config["spatial_matching"]["primary_max_distance_km"]
    )
    strict_distance = float(
        config["spatial_matching"]["strict_sensitivity_max_distance_km"]
    )
    budget = float(
        config["prediction_diagnostics"]["annual_monitoring_budget_fraction"]
    )
    q_threshold = float(config["association_model"]["false_discovery_rate"])

    field = pd.read_csv(args.field, low_memory=False)
    panel = pd.read_csv(args.panel)
    predictions = pd.read_csv(args.predictions)
    if panel["cell_id"].nunique() != 165 or panel.duplicated(
        ["cell_id", "year"]
    ).any():
        raise ValueError("Expected a duplicate-free locked 165-cell panel")
    cells = panel[
        ["cell_id", "region_group", "center_lat", "center_lon"]
    ].drop_duplicates()
    if len(cells) != 165:
        raise ValueError("Cell coordinate table is not one row per locked cell")

    site_year = prepare_field_site_years(field)
    primary_site_matches, primary_field = match_and_aggregate_field(
        site_year, cells, primary_distance
    )
    strict_site_matches, strict_field = match_and_aggregate_field(
        site_year, cells, strict_distance
    )
    ranked, prediction_wide = prepare_prediction_diagnostics(
        predictions, first_year, last_year, domain, budget
    )
    primary_analysis = prediction_wide.merge(
        primary_field,
        on=["cell_id", "year", "region_group"],
        how="inner",
        validate="many_to_one",
    )
    strict_analysis = prediction_wide.merge(
        strict_field,
        on=["cell_id", "year", "region_group"],
        how="inner",
        validate="many_to_one",
    )
    two_site_analysis = primary_analysis.loc[
        primary_analysis["n_sites"].ge(2)
    ].copy()
    if primary_analysis[["cell_id", "year"]].drop_duplicates().shape[0] != 424:
        raise ValueError("Primary match no longer reproduces the locked 424 cell-year scope")

    annual_performance, performance_summary, performance_increments = (
        performance_by_scope(ranked, primary_field, replicates, seed)
    )
    association_frames = [
        association_table(primary_analysis, "primary_7p2km", q_threshold),
        association_table(strict_analysis, "strict_5km", q_threshold),
        association_table(two_site_analysis, "primary_n_sites_ge2", q_threshold),
    ]
    associations = pd.concat(association_frames, ignore_index=True)
    condition_support = build_condition_support(associations, q_threshold)
    cell_errors = repeated_error_cells(
        primary_analysis, cells, config["population"]["primary_model_family"]
    )

    primary_keys = primary_analysis[["cell_id", "year"]].drop_duplicates()
    full_reference = ranked.loc[
        ranked["model_family"].eq(config["population"]["primary_model_family"])
        & ranked["feature_set"].eq("current_only")
    ]
    matched_reference = full_reference.merge(
        primary_keys, on=["cell_id", "year"], how="inner"
    )
    coverage_rows = []
    for region, group in full_reference.groupby("region_group", sort=True):
        matched_region = matched_reference.loc[
            matched_reference["region_group"].eq(region)
        ]
        coverage_rows.append(
            {
                "region_group": region,
                "full_cell_years": len(group),
                "full_cells": int(group["cell_id"].nunique()),
                "full_events": int(group["event"].sum()),
                "matched_cell_years": len(matched_region),
                "matched_cells": int(matched_region["cell_id"].nunique()),
                "matched_events": int(matched_region["event"].sum()),
                "matched_cell_year_share": len(matched_region) / len(group),
            }
        )
    coverage = pd.DataFrame(coverage_rows)

    axis_correlations = (
        primary_analysis.loc[primary_analysis["model_family"].eq("logistic"), AXES]
        .corr(method="spearman")
        .rename_axis("axis")
        .reset_index()
    )
    quality_rows = [
        {"metric": "raw_field_rows", "value": len(field)},
        {"metric": "raw_field_columns", "value": len(field.columns)},
        {
            "metric": "raw_unique_sites",
            "value": int(field["site_campus_unique_ID"].nunique()),
        },
        {
            "metric": "raw_year_start",
            "value": int(field["survey_year"].min()),
        },
        {"metric": "raw_year_end", "value": int(field["survey_year"].max())},
        {"metric": "raw_exact_duplicate_rows", "value": int(field.duplicated().sum())},
        {
            "metric": "raw_coordinate_missing_rate",
            "value": float(field[["latitude", "longitude"]].isna().any(axis=1).mean()),
        },
        {"metric": "unique_site_years", "value": len(site_year)},
        {
            "metric": "primary_matched_site_years",
            "value": len(primary_site_matches),
        },
        {
            "metric": "strict_matched_site_years",
            "value": len(strict_site_matches),
        },
        {
            "metric": "primary_analysis_cell_years",
            "value": len(primary_keys),
        },
        {
            "metric": "primary_analysis_cells",
            "value": int(primary_keys["cell_id"].nunique()),
        },
        {
            "metric": "primary_analysis_events",
            "value": int(matched_reference["event"].sum()),
        },
        {
            "metric": "median_sites_per_cell_year",
            "value": float(primary_field.merge(primary_keys)["n_sites"].median()),
        },
        {
            "metric": "median_transects_per_cell_year",
            "value": float(primary_field.merge(primary_keys)["n_transects"].median()),
        },
    ]
    for axis in AXES:
        quality_rows.append(
            {
                "metric": f"primary_{axis}_missing_rate",
                "value": float(primary_analysis[axis].isna().mean()),
            }
        )
    quality = pd.DataFrame(quality_rows)

    matched_region_counts = (
        matched_reference.groupby("region_group")
        .agg(cell_years=("year", "size"), cells=("cell_id", "nunique"), events=("event", "sum"))
        .reset_index()
        .set_index("region_group")
    )
    matched_increment_decision: dict[str, dict[str, dict[str, float]]] = {}
    matched_increment_rows = performance_increments.loc[
        performance_increments["scope"].eq("giraldo_matched")
    ]
    for family, group in matched_increment_rows.groupby("model_family"):
        matched_increment_decision[family] = {}
        for row in group.itertuples(index=False):
            matched_increment_decision[family][row.comparison] = {
                "estimate": float(row.estimate),
                "ci_low": float(row.ci_low),
                "ci_high": float(row.ci_high),
                "paired_years": int(row.paired_years),
            }
    decision: dict[str, object] = {
        "classification": "pilot-informed exploratory diagnostic; not independent confirmation",
        "decision": (
            "stable_ecological_failure_condition_detected"
            if condition_support["stable_condition_supported"].any()
            else "no_stable_ecological_failure_condition_detected"
        ),
        "matched_population": {
            "cells": int(primary_keys["cell_id"].nunique()),
            "cell_years": len(primary_keys),
            "events": int(matched_reference["event"].sum()),
            "first_year": int(matched_reference["year"].min()),
            "last_year": int(matched_reference["year"].max()),
            "northern_cell_years": int(
                matched_region_counts.loc["Northern California", "cell_years"]
            ),
            "central_cell_years": int(
                matched_region_counts.loc["Central California", "cell_years"]
            ),
            "southern_cell_years": int(
                matched_region_counts.loc["Southern California", "cell_years"]
            ),
        },
        "strict_match_population": {
            "cells": int(strict_analysis["cell_id"].nunique()),
            "cell_years": int(
                strict_analysis[["cell_id", "year"]].drop_duplicates().shape[0]
            ),
        },
        "matched_subset_increments": matched_increment_decision,
        "stable_condition_count": int(
            condition_support["stable_condition_supported"].sum()
        ),
        "tested_primary_condition_count": len(condition_support),
        "interpretation": (
            "No prespecified ecological axis robustly identified where the public OISST block "
            "helped or where trajectory-model errors recurred. This is an identification limit "
            "of the partial, cross-scale California case study, not evidence that ecology is unimportant."
        ),
    }

    outputs = {
        "field_site_year_matches.csv": primary_site_matches,
        "field_cell_years.csv": primary_field,
        "analysis_dataset.csv": primary_analysis,
        "cohort_coverage_by_region.csv": coverage,
        "data_quality_profile.csv": quality,
        "axis_spearman_correlations.csv": axis_correlations,
        "performance_year_metrics.csv": annual_performance,
        "performance_summary.csv": performance_summary,
        "performance_increment_summary.csv": performance_increments,
        "ecological_axis_associations.csv": associations,
        "condition_support_summary.csv": condition_support,
        "repeated_error_cells.csv": cell_errors,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False)

    plot_coverage(
        cells,
        primary_site_matches,
        primary_field.merge(primary_keys, on=["cell_id", "year"]),
        args.output_dir / "figure_01_field_coverage.png",
    )
    plot_associations(
        associations, args.output_dir / "figure_02_primary_ecological_associations.png"
    )
    plot_repeated_errors(
        cell_errors, args.output_dir / "figure_03_recurrent_miss_vs_grazing.png"
    )
    plot_cohort_performance(
        performance_summary,
        args.output_dir / "figure_04_matched_vs_full_performance.png",
    )
    chart_map = pd.DataFrame(
        [
            {
                "figure": "figure_01_field_coverage.png",
                "question": "Where does the field subset overlap the 165-cell domain?",
                "chart_family": "spatial point map",
                "supported_takeaway": "Field coverage is California-only and partial.",
            },
            {
                "figure": "figure_02_primary_ecological_associations.png",
                "question": "Do four prespecified field axes explain out-of-fold errors or OISST rank gain?",
                "chart_family": "faceted dot and interval",
                "supported_takeaway": "Uncertainty intervals include zero for the primary effects.",
            },
            {
                "figure": "figure_03_recurrent_miss_vs_grazing.png",
                "question": "Are recurrent misses concentrated at high surveyed urchin pressure?",
                "chart_family": "scatter",
                "supported_takeaway": "No stable monotonic relationship is visible or supported.",
            },
            {
                "figure": "figure_04_matched_vs_full_performance.png",
                "question": "How does the selected field-observed cohort differ in apparent model performance?",
                "chart_family": "grouped dot and interval",
                "supported_takeaway": "The matched cohort is not a representative replacement for the full domain.",
            },
        ]
    )
    chart_map.to_csv(args.output_dir / "chart_map.csv", index=False)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_results_summary(args.output_dir / "results_summary_ko.md", decision)

    finished = time.time()
    output_files = sorted(
        path for path in args.output_dir.iterdir() if path.name != "manifest.json"
    )
    manifest = {
        "status": "complete",
        "classification": decision["classification"],
        "started_at_utc_epoch": started,
        "finished_at_utc_epoch": finished,
        "runtime_seconds": round(finished - started, 2),
        "command": (
            f"scripts/60_run_giraldo_error_case_study.py --field {args.field} "
            f"--panel {args.panel} --predictions {args.predictions} "
            f"--config {args.config} --output-dir {args.output_dir}"
        ),
        "git_sha_before_run": git_sha(),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "seed": seed,
        "year_bootstrap_replicates": replicates,
        "inputs": {
            "field": str(args.field),
            "field_sha256": sha256(args.field),
            "panel": str(args.panel),
            "panel_sha256": sha256(args.panel),
            "predictions": str(args.predictions),
            "predictions_sha256": sha256(args.predictions),
            "config": str(args.config),
            "config_sha256": sha256(args.config),
        },
        "decision": decision,
        "output_sha256": {path.name: sha256(path) for path in output_files},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
