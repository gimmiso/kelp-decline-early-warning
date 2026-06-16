"""Build or plan NOAA Coral Reef Watch 5 km SST exposure features.

This script adds a CRW CoralTemp 5 km SST exposure family without replacing the
existing OISST V1/V2 workflows. If local CRW daily point caches are not present,
the script runs in dry-run mode and writes planning/diagnostic outputs that
document the required access path.
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_recall_oriented_modeling_extensions import (
    BASELINE_P25,
    CANOPY,
    NEXT_CANOPY,
    TARGET_ACTIONABLE_DROP,
    add_actionable_labels,
)
from train_model_comparison import INPUT_DATASET, main_subset


warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_CACHE_DIR = Path("data/external/noaa/cache/crw5km")
DEFAULT_FEATURE_OUTPUT = Path("data/processed/crw5km_sst_features.csv")
FEATURE_DIAGNOSTICS_OUTPUT = Path("results/tables/crw5km_vs_oisst_feature_diagnostics.csv")
MODEL_COMPARISON_OUTPUT = Path("results/tables/crw5km_model_comparison.csv")
REPORT_OUTPUT = Path("outputs/diagnostics/crw5km_sst_feature_report.md")

CRW_ERDDAP_DATASET = "dhw_5km"
CRW_ERDDAP_BASE_URL = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km"
CRW_PRODUCT_PAGE = "https://coralreefwatch.noaa.gov/product/5km/index_5km_sst.php"
CRW_METHOD_PAGE = "https://coralreefwatch.noaa.gov/product/5km/methodology.php"
CRW_METADATA_PAGE = "https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc%3ACRW-5km-HeatStressProducts"

BASELINE_START_YEAR = 1985
BASELINE_END_YEAR = 2012
MODEL_START_YEAR = 1989
TRAIN_END_YEAR = 2016
TEST_START_YEAR = 2021
TEST_END_YEAR = 2024
CRW_GRID_START_LAT = -89.975
CRW_GRID_START_LON = -179.975
CRW_GRID_STEP = 0.05
BUFFER_RADII_KM = [10, 25]
TARGET_NEW_DECLINE = "new_decline_event_next"
TARGET_AT_RISK = "decline_event_next_at_risk_gt005"

CRW_NEAREST_FEATURES = [
    "annual_mean_sst_crw5km",
    "annual_max_sst_crw5km",
    "spring_mean_sst_crw5km",
    "summer_max_sst_crw5km",
    "warmest_month_mean_sst_crw5km",
    "sst_anomaly_crw5km",
    "hot_days_p90_crw5km",
    "cumulative_heat_stress_crw5km",
    "lag1_sst_anomaly_crw5km",
    "lag1_hot_days_p90_crw5km",
]

OISST_COMPARISON_FEATURES = [
    "annual_mean_sst",
    "annual_max_sst",
    "annual_mean_sst_anomaly",
    "hot_days_p90",
    "lag1_annual_mean_sst_anomaly",
    "lag1_hot_days_p90",
]


@dataclass(frozen=True)
class TargetSpec:
    """Target and optional row filter for model comparison."""

    name: str
    target: str
    filter_column: str | None = None


TARGETS = [
    TargetSpec("original_decline", "decline_event_next"),
    TargetSpec("at_risk_original_gt005", TARGET_AT_RISK, "at_risk_gt005"),
    TargetSpec("new_decline_transition", TARGET_NEW_DECLINE),
    TargetSpec("actionable_decline_drop", TARGET_ACTIONABLE_DROP),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build or plan CRW CoralTemp 5 km SST features.")
    parser.add_argument("--input", type=Path, default=INPUT_DATASET)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--feature-output", type=Path, default=DEFAULT_FEATURE_OUTPUT)
    parser.add_argument("--feature-diagnostics-output", type=Path, default=FEATURE_DIAGNOSTICS_OUTPUT)
    parser.add_argument("--model-comparison-output", type=Path, default=MODEL_COMPARISON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    parser.add_argument("--limit-cells", type=int, default=None, help="Limit cells for local smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="Force planning mode even if local CRW cache exists.")
    return parser.parse_args()


def snap_to_crw_grid(value: float, start: float) -> float:
    """Snap a coordinate to the nearest 0.05 degree CRW grid center."""
    return round(round((value - start) / CRW_GRID_STEP) * CRW_GRID_STEP + start, 3)


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in kilometers."""
    earth_radius_km = 6371.0088
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = lat2_rad - lat1_rad
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(a))


def load_base_rows(input_path: Path, limit_cells: int | None) -> pd.DataFrame:
    """Load V1 modeling rows and add transition/actionable targets."""
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    data = main_subset(pd.read_csv(input_path).sort_values(["cell_id", "year"]).reset_index(drop=True))
    data = add_actionable_labels(data)
    data[TARGET_NEW_DECLINE] = ((data[CANOPY] >= data[BASELINE_P25]) & (data[NEXT_CANOPY] < data[BASELINE_P25])).astype(int)
    data[TARGET_AT_RISK] = data["decline_event_next"].astype(int)
    data["at_risk_gt005"] = data[CANOPY] > 0.05
    if limit_cells is not None:
        keep = data["cell_id"].drop_duplicates().head(limit_cells).tolist()
        data = data.loc[data["cell_id"].isin(keep)].copy()
    return data


def cell_metadata(data: pd.DataFrame) -> pd.DataFrame:
    """Return unique Kelpwatch cell centroid metadata."""
    required = {"cell_id", "center_lat", "center_lon"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing cell metadata columns: {missing}")
    cells = data[["cell_id", "center_lat", "center_lon"]].drop_duplicates().reset_index(drop=True)
    cells["nearest_crw_lat"] = cells["center_lat"].map(lambda value: snap_to_crw_grid(float(value), CRW_GRID_START_LAT))
    cells["nearest_crw_lon"] = cells["center_lon"].map(lambda value: snap_to_crw_grid(float(value), CRW_GRID_START_LON))
    cells["distance_to_crw_grid_km"] = haversine_km(
        cells["center_lat"].to_numpy(),
        cells["center_lon"].to_numpy(),
        cells["nearest_crw_lat"].to_numpy(),
        cells["nearest_crw_lon"].to_numpy(),
    )
    return cells


def parse_crw_cache_filename(path: Path) -> tuple[float, float] | None:
    """Extract CRW point latitude and longitude from a cached CSV name."""
    match = re.search(r"crw5km_lat(?P<lat>[-m0-9.]+)_lon(?P<lon>[-m0-9.]+)_daily\.csv$", path.name)
    if not match:
        return None
    return float(match.group("lat").replace("m", "-")), float(match.group("lon").replace("m", "-"))


def read_crw_cache(cache_dir: Path) -> pd.DataFrame:
    """Read local daily CRW point CSV files from the expected cache directory."""
    frames = []
    for path in sorted(cache_dir.glob("crw5km_lat*_lon*_daily.csv")):
        coords = parse_crw_cache_filename(path)
        if coords is None:
            continue
        lat, lon = coords
        daily = pd.read_csv(path)
        if "time" not in daily.columns:
            continue
        sst_column = next((column for column in ["CRW_SST", "crw_sst", "sst"] if column in daily.columns), None)
        if sst_column is None:
            continue
        daily["time"] = pd.to_datetime(daily["time"], utc=True, errors="coerce")
        daily["sst"] = pd.to_numeric(daily[sst_column], errors="coerce")
        if "CRW_SSTANOMALY" in daily.columns:
            daily["crw_native_sst_anomaly"] = pd.to_numeric(daily["CRW_SSTANOMALY"], errors="coerce")
        elif "sst_anomaly" in daily.columns:
            daily["crw_native_sst_anomaly"] = pd.to_numeric(daily["sst_anomaly"], errors="coerce")
        else:
            daily["crw_native_sst_anomaly"] = np.nan
        daily = daily.dropna(subset=["time", "sst"])
        daily["crw_lat"] = round(lat, 3)
        daily["crw_lon"] = round(lon, 3)
        daily["year"] = daily["time"].dt.year.astype(int)
        daily["month"] = daily["time"].dt.month.astype(int)
        frames.append(daily[["crw_lat", "crw_lon", "time", "year", "month", "sst", "crw_native_sst_anomaly"]])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def annual_crw_point_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Summarize daily CRW point records to annual point features."""
    baseline = daily.loc[daily["year"].between(BASELINE_START_YEAR, BASELINE_END_YEAR)].copy()
    thresholds = (
        baseline.groupby(["crw_lat", "crw_lon"])["sst"]
        .quantile(0.90)
        .rename("sst_p90_baseline")
        .reset_index()
    )
    climatology = (
        baseline.groupby(["crw_lat", "crw_lon"])["sst"]
        .mean()
        .rename("baseline_mean_sst")
        .reset_index()
    )

    daily = daily.merge(thresholds, on=["crw_lat", "crw_lon"], how="left").merge(
        climatology, on=["crw_lat", "crw_lon"], how="left"
    )
    daily["hot_day_p90"] = daily["sst"] > daily["sst_p90_baseline"]
    daily["heat_stress_excess"] = (daily["sst"] - daily["sst_p90_baseline"]).clip(lower=0)

    annual = (
        daily.groupby(["crw_lat", "crw_lon", "year"])["sst"]
        .agg(annual_mean_sst_crw5km="mean", annual_max_sst_crw5km="max")
        .reset_index()
    )
    spring = (
        daily.loc[daily["month"].between(4, 6)]
        .groupby(["crw_lat", "crw_lon", "year"])["sst"]
        .mean()
        .rename("spring_mean_sst_crw5km")
        .reset_index()
    )
    summer = (
        daily.loc[daily["month"].between(6, 8)]
        .groupby(["crw_lat", "crw_lon", "year"])["sst"]
        .max()
        .rename("summer_max_sst_crw5km")
        .reset_index()
    )
    monthly = (
        daily.groupby(["crw_lat", "crw_lon", "year", "month"])["sst"]
        .mean()
        .reset_index()
        .groupby(["crw_lat", "crw_lon", "year"])["sst"]
        .max()
        .rename("warmest_month_mean_sst_crw5km")
        .reset_index()
    )
    hot = (
        daily.groupby(["crw_lat", "crw_lon", "year"])
        .agg(hot_days_p90_crw5km=("hot_day_p90", "sum"), cumulative_heat_stress_crw5km=("heat_stress_excess", "sum"))
        .reset_index()
    )
    annual = annual.merge(spring, on=["crw_lat", "crw_lon", "year"], how="left")
    annual = annual.merge(summer, on=["crw_lat", "crw_lon", "year"], how="left")
    annual = annual.merge(monthly, on=["crw_lat", "crw_lon", "year"], how="left")
    annual = annual.merge(hot, on=["crw_lat", "crw_lon", "year"], how="left")
    annual = annual.merge(climatology, on=["crw_lat", "crw_lon"], how="left")
    annual["sst_anomaly_crw5km"] = annual["annual_mean_sst_crw5km"] - annual["baseline_mean_sst"]
    return annual.drop(columns=["baseline_mean_sst"])


def nearest_crw_features(cells: pd.DataFrame, annual_points: pd.DataFrame, years: pd.Series) -> pd.DataFrame:
    """Attach nearest CRW annual point features to each Kelpwatch cell-year."""
    point_meta = annual_points[["crw_lat", "crw_lon"]].drop_duplicates().reset_index(drop=True)
    if point_meta.empty:
        return pd.DataFrame()
    assignments = []
    for cell in cells.itertuples():
        distances = haversine_km(
            np.repeat(float(cell.center_lat), len(point_meta)),
            np.repeat(float(cell.center_lon), len(point_meta)),
            point_meta["crw_lat"].to_numpy(),
            point_meta["crw_lon"].to_numpy(),
        )
        nearest_idx = int(np.nanargmin(distances))
        nearest_point = point_meta.iloc[nearest_idx]
        assignments.append(
            {
                "cell_id": cell.cell_id,
                "nearest_crw_lat": float(nearest_point["crw_lat"]),
                "nearest_crw_lon": float(nearest_point["crw_lon"]),
                "distance_to_crw_grid_km": float(distances[nearest_idx]),
            }
        )
    assigned = pd.DataFrame(assignments)
    skeleton = assigned.merge(
        pd.DataFrame({"year": sorted(years.unique())}), how="cross"
    )
    features = skeleton.merge(
        annual_points,
        left_on=["nearest_crw_lat", "nearest_crw_lon", "year"],
        right_on=["crw_lat", "crw_lon", "year"],
        how="left",
    )
    features["crw5km_nearest_grid_points_used"] = np.where(features["annual_mean_sst_crw5km"].notna(), 1, 0)
    features = features.drop(columns=[column for column in ["crw_lat", "crw_lon"] if column in features.columns])
    return features


def buffer_crw_features(cells: pd.DataFrame, annual_points: pd.DataFrame) -> pd.DataFrame:
    """Compute 10 km and 25 km buffer means from locally cached CRW point supports."""
    point_meta = annual_points[["crw_lat", "crw_lon"]].drop_duplicates().reset_index(drop=True)
    if point_meta.empty:
        return pd.DataFrame({"cell_id": cells["cell_id"].unique()})
    memberships = []
    for cell in cells.itertuples():
        distances = haversine_km(
            np.repeat(float(cell.center_lat), len(point_meta)),
            np.repeat(float(cell.center_lon), len(point_meta)),
            point_meta["crw_lat"].to_numpy(),
            point_meta["crw_lon"].to_numpy(),
        )
        for radius in BUFFER_RADII_KM:
            inside = point_meta.loc[distances <= radius].copy()
            for point in inside.itertuples():
                memberships.append(
                    {
                        "cell_id": cell.cell_id,
                        "radius_km": radius,
                        "crw_lat": point.crw_lat,
                        "crw_lon": point.crw_lon,
                    }
                )
    if not memberships:
        return pd.DataFrame({"cell_id": cells["cell_id"].unique()})
    memberships_df = pd.DataFrame(memberships)
    merged = memberships_df.merge(annual_points, on=["crw_lat", "crw_lon"], how="left")
    value_columns = [column for column in CRW_NEAREST_FEATURES if column in merged.columns and not column.startswith("lag1_")]
    grouped = (
        merged.groupby(["cell_id", "radius_km", "year"])[value_columns]
        .mean()
        .reset_index()
    )
    counts = (
        memberships_df.groupby(["cell_id", "radius_km"])
        .size()
        .rename("grid_points_used")
        .reset_index()
    )
    grouped = grouped.merge(counts, on=["cell_id", "radius_km"], how="left")
    outputs = []
    for radius in BUFFER_RADII_KM:
        subset = grouped.loc[grouped["radius_km"] == radius].drop(columns=["radius_km"]).copy()
        rename = {
            column: f"{column.replace('_crw5km', '')}_crw5km_buffer{radius}km"
            for column in value_columns
        }
        rename["grid_points_used"] = f"crw5km_buffer{radius}km_grid_points_used"
        outputs.append(subset.rename(columns=rename))
    if not outputs:
        return pd.DataFrame({"cell_id": cells["cell_id"].unique()})
    result = outputs[0]
    for extra in outputs[1:]:
        result = result.merge(extra, on=["cell_id", "year"], how="outer")
    return result


def add_lag_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add lag-1 CRW features where feasible."""
    output = features.sort_values(["cell_id", "year"]).copy()
    grouped = output.groupby("cell_id", group_keys=False)
    for column in list(output.columns):
        if column in {"sst_anomaly_crw5km", "hot_days_p90_crw5km", "cumulative_heat_stress_crw5km"}:
            output[f"lag1_{column}"] = grouped[column].shift(1)
        if column.endswith("_buffer10km") or column.endswith("_buffer25km"):
            if "sst_anomaly" in column or "hot_days_p90" in column:
                output[f"lag1_{column}"] = grouped[column].shift(1)
    return output


def build_crw_features(data: pd.DataFrame, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build CRW cell-year features from local daily point cache files."""
    daily = read_crw_cache(cache_dir)
    if daily.empty:
        return pd.DataFrame(), cell_metadata(data)
    cells = cell_metadata(data)
    annual_points = annual_crw_point_features(daily)
    nearest = nearest_crw_features(cells, annual_points, data["year"])
    buffers = buffer_crw_features(cells, annual_points)
    features = nearest.merge(buffers, on=["cell_id", "year"], how="left") if "year" in buffers.columns else nearest
    features = add_lag_features(features)
    return features, cells


def feature_diagnostics(data: pd.DataFrame, crw_features: pd.DataFrame, cells: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Compare CRW features with OISST and summarize missingness/support."""
    rows = []
    if crw_features.empty:
        rows.append(
            {
                "diagnostic": "data_access",
                "feature": "CRW_SST",
                "comparison_feature": "",
                "value": np.nan,
                "status": "dry_run_no_local_crw_cache",
                "notes": f"Place daily point CSV files under {DEFAULT_CACHE_DIR} or adapt the script to download ERDDAP subsets.",
            }
        )
        rows.append(
            {
                "diagnostic": "nearest_grid_distance",
                "feature": "distance_to_crw_grid_km",
                "comparison_feature": "",
                "value": float(cells["distance_to_crw_grid_km"].mean()) if not cells.empty else np.nan,
                "status": "planned",
                "notes": "Distance to theoretical nearest 0.05 degree CRW grid center from Kelpwatch cell centroid.",
            }
        )
        return pd.DataFrame(rows)

    merged = data.merge(crw_features, on=["cell_id", "year"], how="left")
    pairs = [
        ("annual_mean_sst_crw5km", "annual_mean_sst"),
        ("annual_max_sst_crw5km", "annual_max_sst"),
        ("sst_anomaly_crw5km", "annual_mean_sst_anomaly"),
        ("hot_days_p90_crw5km", "hot_days_p90"),
        ("lag1_sst_anomaly_crw5km", "lag1_annual_mean_sst_anomaly"),
        ("lag1_hot_days_p90_crw5km", "lag1_hot_days_p90"),
    ]
    for crw_column, oisst_column in pairs:
        if crw_column not in merged.columns or oisst_column not in merged.columns:
            continue
        valid = merged[[crw_column, oisst_column]].dropna()
        rows.append(
            {
                "diagnostic": "feature_correlation",
                "feature": crw_column,
                "comparison_feature": oisst_column,
                "value": valid[crw_column].corr(valid[oisst_column]) if len(valid) >= 3 else np.nan,
                "status": "computed",
                "notes": "Pearson correlation across matched cell-year rows.",
            }
        )
        rows.append(
            {
                "diagnostic": "missingness_crw",
                "feature": crw_column,
                "comparison_feature": oisst_column,
                "value": float(merged[crw_column].isna().mean()),
                "status": "computed",
                "notes": "Fraction of cell-year rows missing CRW feature.",
            }
        )
        rows.append(
            {
                "diagnostic": "missingness_oisst",
                "feature": crw_column,
                "comparison_feature": oisst_column,
                "value": float(merged[oisst_column].isna().mean()),
                "status": "computed",
                "notes": "Fraction of cell-year rows missing OISST comparison feature.",
            }
        )

    for metric, value in {
        "mean_distance_to_crw_grid_km": cells["distance_to_crw_grid_km"].mean(),
        "median_distance_to_crw_grid_km": cells["distance_to_crw_grid_km"].median(),
        "max_distance_to_crw_grid_km": cells["distance_to_crw_grid_km"].max(),
        "unique_nearest_crw_grid_points": cells[["nearest_crw_lat", "nearest_crw_lon"]].drop_duplicates().shape[0],
    }.items():
        rows.append(
            {
                "diagnostic": "distance_support",
                "feature": metric,
                "comparison_feature": "",
                "value": float(value),
                "status": "computed",
                "notes": "CRW nearest-grid centroid/support diagnostic.",
            }
        )
    rows.append(
        {
            "diagnostic": "run_mode",
            "feature": "mode",
            "comparison_feature": "",
            "value": np.nan,
            "status": mode,
            "notes": "actual_run means local CRW cache was available; dry_run means planning only.",
        }
    )
    return pd.DataFrame(rows)


def available(features: list[str], data: pd.DataFrame) -> list[str]:
    """Return available columns from a feature list."""
    return [feature for feature in features if feature in data.columns]


def model_feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    """Define CRW-vs-OISST feature families for model comparison."""
    buffer_features = [
        column
        for column in data.columns
        if column.startswith("annual_mean_sst_crw5km_buffer")
        or column.startswith("sst_anomaly_crw5km_buffer")
        or column.startswith("hot_days_p90_crw5km_buffer")
        or column.startswith("lag1_hot_days_p90_crw5km_buffer")
    ]
    return {
        "oisst_v1": available(OISST_COMPARISON_FEATURES, data),
        "crw5km_nearest": available(CRW_NEAREST_FEATURES, data),
        "crw5km_nearest_plus_oisst": available(CRW_NEAREST_FEATURES + OISST_COMPARISON_FEATURES, data),
        "crw5km_buffer_sensitivity": available(buffer_features, data),
    }


def has_two_classes(frame: pd.DataFrame, target: str) -> bool:
    """Return whether both target classes are present."""
    return set(frame[target].dropna().astype(int).unique()) == {0, 1}


def preprocess(features: list[str]) -> ColumnTransformer:
    """Create numeric preprocessing."""
    return ColumnTransformer(
        transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), features)]
    )


def evaluate_models(data: pd.DataFrame, crw_features: pd.DataFrame) -> pd.DataFrame:
    """Compare CRW and OISST environmental-only models on transition targets."""
    if crw_features.empty:
        return pd.DataFrame(
            [
                {
                    "target_definition": target.name,
                    "feature_family": "crw5km",
                    "model": "not_run",
                    "n_train": 0,
                    "n_test": 0,
                    "positive_events_test": 0,
                    "event_prevalence_test": np.nan,
                    "pr_auc": np.nan,
                    "recall": np.nan,
                    "precision": np.nan,
                    "f1": np.nan,
                    "balanced_accuracy": np.nan,
                    "status": "dry_run_no_local_crw_cache",
                }
                for target in TARGETS
            ]
        )

    data = data.merge(crw_features, on=["cell_id", "year"], how="left", validate="one_to_one")
    sets = model_feature_sets(data)
    estimators = {
        "Logistic Regression L2": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=3,
            n_jobs=-1,
        ),
    }
    rows = []
    for target in TARGETS:
        working = data.copy()
        if target.filter_column:
            working = working.loc[working[target.filter_column]].copy()
        train = working.loc[working["year"].between(MODEL_START_YEAR, TRAIN_END_YEAR)].copy()
        test = working.loc[working["year"].between(TEST_START_YEAR, TEST_END_YEAR)].copy()
        if train.empty or test.empty or not has_two_classes(train, target.target) or not has_two_classes(test, target.target):
            rows.append(
                {
                    "target_definition": target.name,
                    "feature_family": "all",
                    "model": "not_run",
                    "n_train": len(train),
                    "n_test": len(test),
                    "positive_events_test": int(test[target.target].sum()) if target.target in test else 0,
                    "event_prevalence_test": float(test[target.target].mean()) if len(test) else np.nan,
                    "pr_auc": np.nan,
                    "recall": np.nan,
                    "precision": np.nan,
                    "f1": np.nan,
                    "balanced_accuracy": np.nan,
                    "status": "insufficient_target_classes",
                }
            )
            continue
        for family, features in sets.items():
            if not features:
                continue
            for model_name, estimator in estimators.items():
                pipeline = Pipeline([("preprocess", preprocess(features)), ("model", estimator)])
                pipeline.fit(train[features], train[target.target].astype(int))
                scores = pipeline.predict_proba(test[features])[:, 1]
                predictions = (scores >= 0.5).astype(int)
                y_true = test[target.target].astype(int)
                rows.append(
                    {
                        "target_definition": target.name,
                        "feature_family": family,
                        "model": model_name,
                        "n_train": len(train),
                        "n_test": len(test),
                        "positive_events_test": int(y_true.sum()),
                        "event_prevalence_test": float(y_true.mean()),
                        "pr_auc": average_precision_score(y_true, scores),
                        "recall": recall_score(y_true, predictions, zero_division=0),
                        "precision": precision_score(y_true, predictions, zero_division=0),
                        "f1": f1_score(y_true, predictions, zero_division=0),
                        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
                        "status": "computed",
                    }
                )
    return pd.DataFrame(rows)


def sample_erddap_url(cells: pd.DataFrame) -> str:
    """Return a sample ERDDAP URL for one cell's nearest CRW grid point."""
    if cells.empty:
        lat, lon = 39.425, -123.775
    else:
        first = cells.iloc[0]
        lat, lon = float(first["nearest_crw_lat"]), float(first["nearest_crw_lon"])
    query = f"CRW_SST[(1985-04-01T12:00:00Z):1:(2024-12-31T12:00:00Z)][({lat:.3f})][({lon:.3f})]"
    return f"{CRW_ERDDAP_BASE_URL}.csv?{quote(query, safe='?,=&[]():')}"


def small_markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without optional dependencies."""
    if frame.empty:
        return "No rows."
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
        else:
            display[column] = display[column].astype(str)
    columns = list(display.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(row[column] for column in columns) + " |" for _, row in display.iterrows()]
    return "\n".join([header, divider, *rows])


def write_report(
    output: Path,
    mode: str,
    cells: pd.DataFrame,
    crw_features: pd.DataFrame,
    diagnostics: pd.DataFrame,
    model_results: pd.DataFrame,
    cache_dir: Path,
) -> None:
    """Write a CRW feature report."""
    n_cells = int(cells["cell_id"].nunique()) if not cells.empty else 0
    n_features = int(len([column for column in crw_features.columns if column.endswith("crw5km")])) if not crw_features.empty else 0
    mean_distance = cells["distance_to_crw_grid_km"].mean() if not cells.empty else np.nan
    max_distance = cells["distance_to_crw_grid_km"].max() if not cells.empty else np.nan
    computed_models = int((model_results["status"] == "computed").sum()) if "status" in model_results else 0
    best = model_results.loc[model_results["status"] == "computed"].sort_values("pr_auc", ascending=False).head(5)

    lines = [
        "# CRW 5 km SST Feature Report",
        "",
        "## Purpose",
        "",
        "This report adds NOAA Coral Reef Watch CoralTemp 5 km SST as a candidate exposure family.",
        "It does not remove or overwrite the existing OISST V1/V2 workflow.",
        "",
        "## Data Access Feasibility",
        "",
        f"- Run mode: `{mode}`",
        f"- Expected local CRW cache directory: `{cache_dir}`",
        f"- ERDDAP dataset ID: `{CRW_ERDDAP_DATASET}`",
        f"- CRW SST variable: `CRW_SST`",
        f"- CRW SST anomaly variable: `CRW_SSTANOMALY`",
        "- Grid resolution: 0.05 degree, approximately 5 km.",
        "- Time coverage in the ERDDAP metadata begins on 1985-04-01 for this operational griddap endpoint.",
        f"- Sample point CSV request: `{sample_erddap_url(cells)}`",
        "",
        "Source pages:",
        "",
        f"- CRW product page: {CRW_PRODUCT_PAGE}",
        f"- CRW methodology page: {CRW_METHOD_PAGE}",
        f"- NOAA/NCEI metadata page: {CRW_METADATA_PAGE}",
        "",
        "## Spatial Matching Strategy",
        "",
        "- Baseline: nearest CRW 5 km ocean grid cell to each Kelpwatch 10 km cell centroid.",
        "- Sensitivity: 10 km and 25 km buffer means if sufficient local CRW grid-point caches are available.",
        "- Diagnostics record `distance_to_crw_grid_km` and the number of CRW grid points used.",
        "- CRW 5 km SST is interpreted as a higher-resolution satellite SST exposure layer, not true local in-situ nearshore temperature.",
        "",
        "## Current Run Summary",
        "",
        f"- Kelpwatch cells inspected: `{n_cells}`",
        f"- Mean distance to theoretical nearest CRW grid center: `{mean_distance:.3f}` km",
        f"- Max distance to theoretical nearest CRW grid center: `{max_distance:.3f}` km",
        f"- CRW feature rows built: `{len(crw_features)}`",
        f"- CRW nearest feature columns built: `{n_features}`",
        f"- Computed model-comparison rows: `{computed_models}`",
        "",
    ]
    if crw_features.empty:
        lines.extend(
            [
                "## Dry-Run Interpretation",
                "",
                "No local CRW 5 km daily point cache was found, so feature construction and model comparison were not run.",
                "The diagnostic tables were still written to document required inputs and access steps.",
                "",
                "Expected local daily point-cache naming pattern:",
                "",
                "```text",
                "data/external/noaa/cache/crw5km/crw5km_lat39.425_lonm123.775_daily.csv",
                "```",
                "",
                "Each CSV should include at least:",
                "",
                "```text",
                "time, CRW_SST",
                "```",
                "",
                "`CRW_SSTANOMALY` may also be included if downloaded from ERDDAP.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Model Comparison Summary",
                "",
                "The table below lists the top computed CRW/OISST environmental-only results from this run.",
                "",
                small_markdown_table(best) if not best.empty else "No model rows were computed.",
                "",
            ]
        )

    lines.extend(
        [
            "## Output Files",
            "",
            f"- `{FEATURE_DIAGNOSTICS_OUTPUT}`",
            f"- `{MODEL_COMPARISON_OUTPUT}`",
            f"- `{REPORT_OUTPUT}`",
            "",
            "## Interpretation",
            "",
            "CRW 5 km SST should be treated as a higher-resolution satellite SST exposure alternative to OISST, not as true local in-situ temperature. The goal is to test whether a less coarse SST product improves at-risk and transition-oriented kelp decline prediction.",
            "",
            "If CRW features do not improve transition-oriented targets, the result still helps support the interpretation that abrupt kelp transitions require local ecological drivers such as grazing pressure, predator/community state, wave disturbance, substrate, and disease context.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run CRW feature construction or dry-run planning."""
    args = parse_args()
    for path in [args.feature_output, args.feature_diagnostics_output, args.model_comparison_output, args.report_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    data = load_base_rows(args.input, args.limit_cells)
    cells = cell_metadata(data)
    mode = "dry_run_forced" if args.dry_run else "actual_run"
    if args.dry_run:
        crw_features = pd.DataFrame()
    else:
        crw_features, cells = build_crw_features(data, args.cache_dir)
        if crw_features.empty:
            mode = "dry_run_no_local_crw_cache"
        else:
            actual_assignments = crw_features[
                ["cell_id", "nearest_crw_lat", "nearest_crw_lon", "distance_to_crw_grid_km"]
            ].drop_duplicates("cell_id")
            cells = cells.drop(columns=["nearest_crw_lat", "nearest_crw_lon", "distance_to_crw_grid_km"]).merge(
                actual_assignments, on="cell_id", how="left"
            )
            crw_features.to_csv(args.feature_output, index=False)

    diagnostics = feature_diagnostics(data, crw_features, cells, mode)
    model_results = evaluate_models(data, crw_features)
    diagnostics.to_csv(args.feature_diagnostics_output, index=False)
    model_results.to_csv(args.model_comparison_output, index=False)
    write_report(args.report_output, mode, cells, crw_features, diagnostics, model_results, args.cache_dir)

    print(f"Run mode: {mode}")
    print(f"Wrote feature diagnostics: {args.feature_diagnostics_output}")
    print(f"Wrote model comparison: {args.model_comparison_output}")
    print(f"Wrote report: {args.report_output}")
    if not crw_features.empty:
        print(f"Wrote CRW feature table: {args.feature_output}")


if __name__ == "__main__":
    main()
