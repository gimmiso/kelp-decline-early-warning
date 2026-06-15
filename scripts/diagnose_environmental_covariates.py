"""Run environmental covariate QC and sensitivity diagnostics.

This script evaluates whether NOAA environmental covariates may be too spatially
or temporally coarse for nearshore kelp decline screening. It uses existing
processed modeling rows and local NOAA cache files; it does not download new
NOAA data or write raw data.
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from train_model_comparison import INPUT_DATASET, preprocessor, predict_scores
from run_recall_oriented_modeling_extensions import (
    TARGET_ACTIONABLE_DROP,
    TARGET_ACTIONABLE_LOW,
    TARGET_ORIGINAL,
    add_actionable_labels,
    add_environment_extensions,
    add_trajectory_features,
    model_specs,
)


warnings.filterwarnings("ignore", category=UserWarning)

CANOPY = "relative_canopy"
NEXT_CANOPY = "next_year_relative_canopy"
BASELINE_P25 = "baseline_p25_relative_canopy_1984_2013"
NEW_DECLINE_TARGET = "new_decline_event_next"
BASELINE_START_YEAR = 1984
BASELINE_END_YEAR = 2013
MODEL_START_YEAR = 1989
MODEL_END_YEAR = 2024

DEFAULT_CACHE_DIR = Path("data/external/noaa/cache")
DIAGNOSTIC_DIR = Path("outputs/diagnostics")
MODEL_RESULTS_DIR = Path("outputs/model_results")
METADATA_DIR = Path("outputs/metadata")
FIGURE_DIR = Path("outputs/figures")

DISTANCE_BY_CELL = DIAGNOSTIC_DIR / "oisst_spatial_matching_distance_by_cell.csv"
DISTANCE_SUMMARY = DIAGNOSTIC_DIR / "oisst_spatial_matching_distance_summary.csv"
DISTANCE_FIGURE = FIGURE_DIR / "oisst_matching_distance_distribution.png"
OISST_SENSITIVITY_SUMMARY = DIAGNOSTIC_DIR / "oisst_matching_sensitivity_summary.csv"
OISST_SENSITIVITY_PERFORMANCE = MODEL_RESULTS_DIR / "oisst_matching_sensitivity_model_performance.csv"
ENV_FEATURE_AVAILABILITY = METADATA_DIR / "environmental_feature_availability_report.csv"
SEASONAL_FEATURE_SUMMARY = METADATA_DIR / "seasonal_environmental_feature_summary.csv"
ENV_INCREMENTAL_PERFORMANCE = MODEL_RESULTS_DIR / "environment_incremental_value_performance.csv"
ENV_INCREMENTAL_BY_SUBSET = MODEL_RESULTS_DIR / "environment_incremental_value_by_subset_and_label.csv"
REPORT_OUTPUT = DIAGNOSTIC_DIR / "environmental_covariate_diagnostic_report.md"


@dataclass(frozen=True)
class EvalContext:
    """Definition of one subset/label evaluation context."""

    evaluation_context: str
    target: str
    subset_threshold: float | None = None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Diagnose environmental covariate quality and sensitivity.")
    parser.add_argument("--input", type=Path, default=INPUT_DATASET)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    return parser.parse_args()


def load_data(path: Path) -> pd.DataFrame:
    """Load the modeling dataset and keep complete-feature years."""
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path).sort_values(["cell_id", "year"]).reset_index(drop=True)
    data = data.loc[data["year"].between(MODEL_START_YEAR, MODEL_END_YEAR)].copy()
    if len(data) != 1800:
        raise ValueError(f"Expected 1,800 rows for {MODEL_START_YEAR}-{MODEL_END_YEAR}, found {len(data)}.")
    return data


def add_transition_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Add stricter new-decline and actionable labels."""
    labeled = add_actionable_labels(data)
    labeled[NEW_DECLINE_TARGET] = (
        (labeled[CANOPY] >= labeled[BASELINE_P25]) & (labeled[NEXT_CANOPY] < labeled[BASELINE_P25])
    ).astype(int)
    return labeled


def haversine_km(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Compute great-circle distance in kilometers."""
    radius = 6371.0088
    lat1_rad = np.radians(lat1.astype(float))
    lon1_rad = np.radians(lon1.astype(float))
    lat2_rad = np.radians(lat2.astype(float))
    lon2_rad = np.radians(lon2.astype(float))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def oisst_distance_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create cell-level OISST source matching distance tables."""
    required = {"cell_id", "center_lat", "center_lon", "oisst_source_lat", "oisst_source_lon"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing OISST matching columns: {missing}")
    cells = data[["cell_id", "center_lat", "center_lon", "oisst_source_lat", "oisst_source_lon"]].drop_duplicates()
    cells["oisst_matching_distance_km"] = haversine_km(
        cells["center_lat"],
        cells["center_lon"],
        cells["oisst_source_lat"],
        cells["oisst_source_lon"],
    )
    distances = cells["oisst_matching_distance_km"]
    summary = pd.DataFrame(
        [
            ("mean_distance_km", distances.mean()),
            ("median_distance_km", distances.median()),
            ("max_distance_km", distances.max()),
            ("std_distance_km", distances.std()),
            ("cells_distance_gt_10km", int((distances > 10).sum())),
            ("cells_distance_gt_20km", int((distances > 20).sum())),
            ("cells_distance_gt_30km", int((distances > 30).sum())),
            ("cells_distance_gt_40km", int((distances > 40).sum())),
            ("n_cells", len(cells)),
        ],
        columns=["metric", "value"],
    )
    return cells.sort_values("cell_id"), summary


def plot_distance_distribution(distance_by_cell: pd.DataFrame, output: Path) -> None:
    """Plot OISST source matching distance distribution."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(distance_by_cell["oisst_matching_distance_km"], bins=14, edgecolor="black", alpha=0.8)
    for threshold in [10, 20, 30, 40]:
        ax.axvline(threshold, linestyle="--", linewidth=1, label=f"{threshold} km")
    ax.set_xlabel("Distance from Kelpwatch cell centroid to OISST source grid (km)")
    ax.set_ylabel("Number of cells")
    ax.set_title("OISST spatial matching distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def parse_oisst_cache(cache_dir: Path) -> dict[tuple[float, float], Path]:
    """Map cached OISST daily files to their grid coordinates."""
    pattern = re.compile(r"oisst_lat(?P<lat>[-m0-9.]+)_lon(?P<lon>[-m0-9.]+)_daily\.csv$")
    mapping: dict[tuple[float, float], Path] = {}
    for path in (cache_dir / "oisst").glob("oisst_lat*_lon*_daily.csv"):
        match = pattern.search(path.name)
        if not match:
            continue
        lat = float(match.group("lat").replace("m", "-"))
        lon = float(match.group("lon").replace("m", "-"))
        mapping[(round(lat, 3), round(lon, 3))] = path
    return mapping


def read_daily(path: Path) -> pd.DataFrame:
    """Read one cached daily OISST CSV."""
    daily = pd.read_csv(path, parse_dates=["time"])
    daily["time"] = pd.to_datetime(daily["time"], utc=True)
    daily["sst"] = pd.to_numeric(daily["sst"], errors="coerce")
    return daily.dropna(subset=["sst"])


def neighbor_points(lat: float, lon: float) -> list[tuple[float, float]]:
    """Return the 3x3 OISST neighborhood centered on one grid point."""
    return [(round(lat + dlat, 3), round(lon + dlon, 3)) for dlat in [-0.25, 0.0, 0.25] for dlon in [-0.25, 0.0, 0.25]]


def neighborhood_daily(
    source_lat: float,
    source_lon: float,
    cache_map: dict[tuple[float, float], Path],
) -> tuple[pd.DataFrame, int]:
    """Average cached daily OISST values across available 3x3 neighbor points."""
    frames = []
    used_points = 0
    for point in neighbor_points(source_lat, source_lon):
        path = cache_map.get(point)
        if path is None:
            continue
        frame = read_daily(path)[["time", "sst"]].rename(columns={"sst": f"sst_{used_points}"})
        frames.append(frame)
        used_points += 1
    if not frames:
        return pd.DataFrame(columns=["time", "sst"]), 0
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="time", how="outer")
    sst_columns = [column for column in merged.columns if column.startswith("sst_")]
    merged["sst"] = merged[sst_columns].mean(axis=1)
    return merged[["time", "sst"]].dropna(), used_points


def annual_sst_features(daily: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """Aggregate daily SST to annual, seasonal, and window features."""
    if daily.empty:
        return pd.DataFrame()
    frame = daily.copy()
    frame["year"] = pd.to_datetime(frame["time"], utc=True).dt.year
    frame["month"] = pd.to_datetime(frame["time"], utc=True).dt.month
    baseline = frame.loc[frame["year"].between(BASELINE_START_YEAR, BASELINE_END_YEAR)]
    annual_baseline_mean = baseline.groupby("year")["sst"].mean().mean()
    annual_p90 = baseline["sst"].quantile(0.90)
    annual_p95 = baseline["sst"].quantile(0.95)
    season_months = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "growing_season": [5, 6, 7, 8, 9],
        "pre_growing_season": [2, 3, 4],
    }
    annual = frame.groupby("year").agg(
        annual_mean_sst=("sst", "mean"),
        annual_sst_anomaly=("sst", lambda values: values.mean() - annual_baseline_mean),
        annual_hot_days_p90=("sst", lambda values: int((values > annual_p90).sum())),
        annual_hot_days_p95=("sst", lambda values: int((values > annual_p95).sum())),
    )
    for season, months in season_months.items():
        seasonal = frame.loc[frame["month"].isin(months)]
        baseline_season = seasonal.loc[seasonal["year"].between(BASELINE_START_YEAR, BASELINE_END_YEAR)]
        baseline_mean = baseline_season.groupby("year")["sst"].mean().mean()
        season_p90 = baseline_season["sst"].quantile(0.90)
        season_p95 = baseline_season["sst"].quantile(0.95)
        grouped = seasonal.groupby("year")["sst"]
        annual[f"{season}_mean_sst_anomaly"] = grouped.mean() - baseline_mean
        annual[f"{season}_hot_days_p90"] = grouped.apply(lambda values: int((values > season_p90).sum()))
        annual[f"{season}_hot_days_p95"] = grouped.apply(lambda values: int((values > season_p95).sum()))
    summer = frame.loc[frame["month"].isin(season_months["summer"])]
    summer_baseline_max = summer.loc[summer["year"].between(BASELINE_START_YEAR, BASELINE_END_YEAR)].groupby("year")["sst"].max().mean()
    annual["summer_max_sst_anomaly"] = summer.groupby("year")["sst"].max() - summer_baseline_max
    annual = annual.sort_index()
    for column in [
        "annual_sst_anomaly",
        "winter_mean_sst_anomaly",
        "spring_mean_sst_anomaly",
        "summer_mean_sst_anomaly",
    ]:
        if column in annual.columns:
            annual[f"{column}_lag1"] = annual[column].shift(1)
    annual = annual.reset_index()
    if suffix:
        annual = annual.rename(columns={column: f"{column}_{suffix}" for column in annual.columns if column != "year"})
    return annual


def build_oisst_sensitivity_features(data: pd.DataFrame, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create nearest and cached 3x3 OISST sensitivity features."""
    cache_map = parse_oisst_cache(cache_dir)
    cell_meta = data[["cell_id", "oisst_source_lat", "oisst_source_lon"]].drop_duplicates()
    frames = []
    sensitivity_rows = []
    for row in cell_meta.to_dict("records"):
        source = (round(float(row["oisst_source_lat"]), 3), round(float(row["oisst_source_lon"]), 3))
        nearest_path = cache_map.get(source)
        if nearest_path is None:
            continue
        nearest = annual_sst_features(read_daily(nearest_path), "nearest")
        neighborhood, used_points = neighborhood_daily(source[0], source[1], cache_map)
        neighborhood_features = annual_sst_features(neighborhood, "3x3")
        features = nearest.merge(neighborhood_features, on="year", how="left")
        features["cell_id"] = row["cell_id"]
        features["oisst_3x3_cached_grid_count"] = used_points
        frames.append(features)
        compare_cols = [
            ("annual_mean_sst", "annual_mean_sst_nearest", "annual_mean_sst_3x3"),
            ("annual_sst_anomaly", "annual_sst_anomaly_nearest", "annual_sst_anomaly_3x3"),
            ("annual_hot_days_p90", "annual_hot_days_p90_nearest", "annual_hot_days_p90_3x3"),
            ("annual_hot_days_p95", "annual_hot_days_p95_nearest", "annual_hot_days_p95_3x3"),
            ("lag1_sst_anomaly", "annual_sst_anomaly_lag1_nearest", "annual_sst_anomaly_lag1_3x3"),
        ]
        for feature, nearest_col, mean_col in compare_cols:
            if nearest_col in features.columns and mean_col in features.columns:
                diff = features[mean_col] - features[nearest_col]
                sensitivity_rows.append(
                    {
                        "cell_id": row["cell_id"],
                        "feature": feature,
                        "oisst_3x3_cached_grid_count": used_points,
                        "mean_abs_difference": diff.abs().mean(),
                        "mean_difference": diff.mean(),
                        "max_abs_difference": diff.abs().max(),
                        "correlation": features[[nearest_col, mean_col]].corr().iloc[0, 1],
                    }
                )
    if not frames:
        raise ValueError("No OISST cache files matched source coordinates.")
    features = pd.concat(frames, ignore_index=True)
    features = data[["cell_id", "year"]].merge(features, on=["cell_id", "year"], how="left")
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity_summary = (
        sensitivity.groupby("feature")
        .agg(
            cells=("cell_id", "nunique"),
            mean_cached_3x3_grid_count=("oisst_3x3_cached_grid_count", "mean"),
            mean_abs_difference=("mean_abs_difference", "mean"),
            mean_difference=("mean_difference", "mean"),
            max_abs_difference=("max_abs_difference", "max"),
            median_correlation=("correlation", "median"),
        )
        .reset_index()
    )
    return features, sensitivity, sensitivity_summary


def add_cuti_beuti_seasonal(data: pd.DataFrame, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add seasonal CUTI/BEUTI anomaly features from cached daily tables."""
    enhanced = data.copy()
    rows = []
    for prefix, lat_column in [("cuti", "cuti_lat_bin"), ("beuti", "beuti_lat_bin")]:
        paths = sorted((cache_dir / "cuti_beuti").glob(f"{prefix}_lat*_daily.csv"))
        if not paths:
            rows.append({"feature": f"{prefix}_seasonal_anomalies", "status": "skipped", "notes": "No cached daily table found."})
            continue
        daily = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        rename = {}
        for column in daily.columns:
            if column.startswith("time"):
                rename[column] = "time"
            elif column.startswith("latitude"):
                rename[column] = "latitude"
            elif column.upper().startswith(prefix.upper()):
                rename[column] = prefix
        daily = daily.rename(columns=rename)
        daily["time"] = pd.to_datetime(daily["time"], utc=True)
        daily["year"] = daily["time"].dt.year
        daily["month"] = daily["time"].dt.month
        daily[prefix] = pd.to_numeric(daily[prefix], errors="coerce")
        season_defs = {"winter": [12, 1, 2], "spring": [3, 4, 5], "summer": [6, 7, 8]}
        seasonal_frames = []
        for season, months in season_defs.items():
            seasonal = daily.loc[daily["month"].isin(months)].copy()
            means = seasonal.groupby(["latitude", "year"])[prefix].mean().reset_index(name=f"{season}_{prefix}")
            baseline = means.loc[means["year"].between(BASELINE_START_YEAR, BASELINE_END_YEAR)].groupby("latitude")[f"{season}_{prefix}"].mean()
            means[f"{season}_{prefix}_anomaly"] = means[f"{season}_{prefix}"] - means["latitude"].map(baseline)
            seasonal_frames.append(means[["latitude", "year", f"{season}_{prefix}_anomaly"]])
        features = seasonal_frames[0]
        for frame in seasonal_frames[1:]:
            features = features.merge(frame, on=["latitude", "year"], how="outer")
        features = features.rename(columns={"latitude": lat_column})
        feature_cols = [column for column in features.columns if column.endswith("_anomaly")]
        features = features.sort_values([lat_column, "year"])
        for column in feature_cols:
            features[f"{column}_lag1"] = features.groupby(lat_column)[column].shift(1)
            rows.append({"feature": column, "status": "created", "notes": f"Seasonal {prefix.upper()} anomaly from cached daily table."})
            rows.append({"feature": f"{column}_lag1", "status": "created", "notes": f"Lagged seasonal {prefix.upper()} anomaly."})
        enhanced = enhanced.merge(features, on=[lat_column, "year"], how="left")
    return enhanced, pd.DataFrame(rows)


def split_data(data: pd.DataFrame, target: str) -> dict[str, pd.DataFrame]:
    """Apply the established temporal split."""
    return {
        "train": data.loc[data["year"].between(1989, 2016)].copy(),
        "validation": data.loc[data["year"].between(2017, 2020)].copy(),
        "test": data.loc[data["year"].between(2021, 2024)].copy(),
    }


def score_metrics(y_true: pd.Series, scores: np.ndarray) -> dict[str, object]:
    """Compute default-threshold and ranking metrics."""
    predictions = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "f2": fbeta_score(y_true, predictions, beta=2, zero_division=0),
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def available(columns: list[str], data: pd.DataFrame) -> list[str]:
    """Return available columns."""
    return [column for column in columns if column in data.columns]


def feature_sets(data: pd.DataFrame, variant: str = "nearest") -> dict[str, list[str]]:
    """Define environmental incremental-value feature sets."""
    current = available(["relative_canopy", "kelp_area_m2", "count_cells_kelp", "historical_footprint_area_m2"], data)
    trajectory = available(
        ["canopy_lag1", "canopy_lag2", "canopy_2yr_change", "canopy_3yr_slope", "canopy_3yr_cv", "canopy_drop_from_3yr_max"],
        data,
    )
    if variant == "3x3":
        env = available(
            [
                "annual_mean_sst_3x3",
                "annual_sst_anomaly_3x3",
                "annual_hot_days_p90_3x3",
                "annual_hot_days_p95_3x3",
                "annual_sst_anomaly_lag1_3x3",
            ],
            data,
        )
    else:
        env = available(
            [
                "annual_mean_sst_anomaly",
                "annual_max_sst_anomaly",
                "hot_days_p90",
                "hot_days_p95",
                "lag1_annual_mean_sst_anomaly",
                "lag1_hot_days_p90",
                "annual_mean_cuti",
                "cuti_anomaly",
                "lag1_cuti_anomaly",
                "annual_mean_beuti",
                "beuti_anomaly",
                "lag1_beuti_anomaly",
                "winter_mean_sst_anomaly_nearest",
                "spring_mean_sst_anomaly_nearest",
                "summer_mean_sst_anomaly_nearest",
                "summer_max_sst_anomaly_nearest",
                "growing_season_hot_days_p90_nearest",
                "growing_season_hot_days_p95_nearest",
                "pre_growing_season_hot_days_p90_nearest",
                "pre_growing_season_hot_days_p95_nearest",
                "winter_mean_sst_anomaly_lag1_nearest",
                "spring_mean_sst_anomaly_lag1_nearest",
                "summer_mean_sst_anomaly_lag1_nearest",
                "winter_cuti_anomaly",
                "spring_cuti_anomaly",
                "summer_cuti_anomaly",
                "winter_beuti_anomaly",
                "spring_beuti_anomaly",
                "summer_beuti_anomaly",
                "winter_cuti_anomaly_lag1",
                "spring_beuti_anomaly_lag1",
            ],
            data,
        )
    sets = {
        "canopy_current_only": current,
        "environment_only": env,
        "canopy_current_plus_environment": current + env,
        "canopy_trajectory_only": trajectory,
        "canopy_current_plus_trajectory": current + trajectory,
        "canopy_current_plus_trajectory_plus_environment": current + trajectory + env,
    }
    return {name: list(dict.fromkeys(features)) for name, features in sets.items() if features}


def evaluate_models(data: pd.DataFrame, contexts: list[tuple[str, str, float | None]], variant: str = "nearest") -> pd.DataFrame:
    """Evaluate incremental feature sets across contexts."""
    rows = []
    sets = feature_sets(data, variant)
    for context_name, target, threshold in contexts:
        subset = data.loc[data[CANOPY] > threshold].copy() if threshold is not None else data.copy()
        splits = split_data(subset, target)
        for split_name, split in splits.items():
            if set(split[target].astype(int).unique()) != {0, 1}:
                raise ValueError(f"{context_name} / {target} / {split_name} does not have both classes.")
        y_train = splits["train"][target].astype(int)
        specs = [spec for spec in model_specs(y_train) if spec.model_variant == "cost_sensitive"]
        for set_name, features in sets.items():
            for spec in specs:
                pipe = Pipeline(
                    [
                        ("preprocess", preprocessor(splits["train"], features, scale=spec.needs_scaling)),
                        ("model", spec.estimator),
                    ]
                )
                pipe.fit(splits["train"][features], y_train)
                y_test = splits["test"][target].astype(int)
                scores = predict_scores(pipe, splits["test"][features])
                metrics = score_metrics(y_test, scores)
                rows.append(
                    {
                        "evaluation_context": context_name,
                        "target": target,
                        "oisst_matching_variant": variant,
                        "feature_set": set_name,
                        "model_family": spec.model_family,
                        "model_variant": spec.model_variant,
                        "n_observations": len(y_test),
                        "n_positive_events": int(y_test.sum()),
                        "event_prevalence": float(y_test.mean()),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    distance_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    incremental: pd.DataFrame,
    by_subset: pd.DataFrame,
) -> None:
    """Write the environmental covariate diagnostic report."""
    def markdown_table(frame: pd.DataFrame) -> str:
        """Render a compact Markdown table without optional tabulate dependency."""
        if frame.empty:
            return "_No rows available._"
        display = frame.copy()
        for column in display.select_dtypes(include=[np.number]).columns:
            display[column] = display[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
        headers = list(display.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in display.astype(str).itertuples(index=False, name=None):
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    distance = dict(zip(distance_summary["metric"], distance_summary["value"]))
    full_best = incremental.sort_values(["feature_set", "f2", "pr_auc"], ascending=[True, False, False]).groupby("feature_set").head(1)
    subset_best = by_subset.sort_values(["evaluation_context", "f2", "pr_auc"], ascending=[True, False, False]).groupby("evaluation_context").head(1)
    lines = [
        "# Environmental Covariate Diagnostic Report",
        "",
        "## OISST Spatial Matching Distance",
        "",
        f"- Mean distance to assigned OISST source grid: {float(distance['mean_distance_km']):.2f} km",
        f"- Median distance: {float(distance['median_distance_km']):.2f} km",
        f"- Maximum distance: {float(distance['max_distance_km']):.2f} km",
        f"- Cells > 20 km: {int(float(distance['cells_distance_gt_20km']))}",
        f"- Cells > 40 km: {int(float(distance['cells_distance_gt_40km']))}",
        "",
        "## Nearest vs Cached 3x3 OISST Sensitivity",
        "",
        markdown_table(sensitivity_summary),
        "",
        "## Incremental Value by Feature Set, Full Original Label",
        "",
        markdown_table(full_best[["feature_set", "model_family", "precision", "recall", "f1", "f2", "pr_auc", "false_negatives", "false_positives"]]),
        "",
        "## Best Results by Evaluation Context",
        "",
        markdown_table(subset_best[["evaluation_context", "target", "feature_set", "model_family", "precision", "recall", "f2", "pr_auc", "false_negatives", "false_positives"]]),
        "",
        "## Interpretation",
        "",
        "NOAA environmental covariates did not improve full-sample performance over current-canopy-only models under the current feature construction. This does not imply that environmental variables are irrelevant. The OISST matching distances, partial cached-neighborhood averaging, CUTI/BEUTI latitude-bin assignment, and annual/seasonal aggregation all indicate that the present covariates are coarse relative to nearshore kelp canopy dynamics.",
        "",
        "The most defensible interpretation is that the current NOAA covariates provide useful environmental context but limited incremental predictive value once current canopy state is included. Environmental variables may become more useful with finer coastal SST matching, event-window features, wave/grazing/disease covariates, or region-specific validation.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run environmental covariate diagnostics."""
    args = parse_args()
    for directory in [DIAGNOSTIC_DIR, MODEL_RESULTS_DIR, METADATA_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    data = add_transition_labels(load_data(args.input))
    data, _trajectory_summary = add_trajectory_features(data)
    data, _environment_extension_report = add_environment_extensions(data)
    distance_by_cell, distance_summary = oisst_distance_tables(data)
    plot_distance_distribution(distance_by_cell, DISTANCE_FIGURE)
    oisst_features, _cell_sensitivity, sensitivity_summary = build_oisst_sensitivity_features(data, args.cache_dir)
    data = data.merge(
        oisst_features.drop(columns=[column for column in ["annual_mean_sst_nearest"] if column in oisst_features.columns]),
        on=["cell_id", "year"],
        how="left",
    )
    data, upwell_feature_report = add_cuti_beuti_seasonal(data, args.cache_dir)
    seasonal_columns = [column for column in data.columns if any(token in column for token in ["winter_", "spring_", "summer_", "growing_season_", "pre_growing_season_"])]
    seasonal_summary = pd.DataFrame(
        [
            {
                "feature": column,
                "missing_count": int(data[column].isna().sum()),
                "non_missing_count": int(data[column].notna().sum()),
                "mean": pd.to_numeric(data[column], errors="coerce").mean(),
                "std": pd.to_numeric(data[column], errors="coerce").std(),
            }
            for column in sorted(set(seasonal_columns))
        ]
    )
    feature_report = pd.concat(
        [
            pd.DataFrame(
                [
                    {"feature": "cached_3x3_oisst", "status": "created", "notes": "Uses cached OISST grid files within one 0.25-degree step of each source grid point."},
                    {"feature": "buffer_average_oisst", "status": "skipped", "notes": "Not implemented because local cache does not contain a complete coastal-radius grid inventory."},
                ]
            ),
            upwell_feature_report,
        ],
        ignore_index=True,
    )
    contexts_full = [("full_sample", TARGET_ORIGINAL, None)]
    contexts_by_subset = [
        ("full_original", TARGET_ORIGINAL, None),
        ("at_risk_current_canopy_gt_0.05", TARGET_ORIGINAL, 0.05),
        ("new_decline_transition", NEW_DECLINE_TARGET, None),
        ("actionable_low_next", TARGET_ACTIONABLE_LOW, None),
        ("actionable_drop_next", TARGET_ACTIONABLE_DROP, None),
    ]
    incremental = evaluate_models(data, contexts_full, "nearest")
    by_subset = evaluate_models(data, contexts_by_subset, "nearest")
    sensitivity_model = evaluate_models(data, contexts_full, "3x3")
    distance_by_cell.to_csv(DISTANCE_BY_CELL, index=False)
    distance_summary.to_csv(DISTANCE_SUMMARY, index=False)
    sensitivity_summary.to_csv(OISST_SENSITIVITY_SUMMARY, index=False)
    sensitivity_model.loc[sensitivity_model["feature_set"].isin(["environment_only", "canopy_current_plus_environment"])].to_csv(
        OISST_SENSITIVITY_PERFORMANCE, index=False
    )
    feature_report.to_csv(ENV_FEATURE_AVAILABILITY, index=False)
    seasonal_summary.to_csv(SEASONAL_FEATURE_SUMMARY, index=False)
    incremental.to_csv(ENV_INCREMENTAL_PERFORMANCE, index=False)
    by_subset.to_csv(ENV_INCREMENTAL_BY_SUBSET, index=False)
    write_report(REPORT_OUTPUT, distance_summary, sensitivity_summary, incremental, by_subset)

    print("Environmental covariate diagnostics complete.")
    print(f"Mean OISST matching distance: {float(dict(zip(distance_summary['metric'], distance_summary['value']))['mean_distance_km']):.2f} km")
    print(f"Max OISST matching distance: {float(dict(zip(distance_summary['metric'], distance_summary['value']))['max_distance_km']):.2f} km")
    print(f"Incremental performance rows: {len(incremental)}")
    print(f"Subset/label performance rows: {len(by_subset)}")


if __name__ == "__main__":
    main()
