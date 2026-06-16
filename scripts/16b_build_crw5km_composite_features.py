"""Build CRW 5 km monthly-composite SST features for Kelpwatch cells.

This script keeps the existing OISST workflow intact and adds a lighter NOAA
Coral Reef Watch (CRW) 5 km composite exposure layer. It avoids the slow daily
point-cache path and the inconsistent ERDDAP yearly-bbox path by streaming
predictable NOAA STAR monthly NetCDF files, extracting only retained Kelpwatch
cell points, and deleting raw NetCDF files unless explicitly told otherwise.
"""

from __future__ import annotations

import argparse
import calendar
import os
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from netCDF4 import Dataset
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_recall_oriented_modeling_extensions import (
    BASELINE_P25,
    CANOPY,
    NEXT_CANOPY,
    TARGET_ACTIONABLE_DROP,
    add_actionable_labels,
)
from train_model_comparison import CANOPY_FEATURES, INPUT_DATASET, OISST_FEATURES, main_subset


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

STAR_MONTHLY_ROOT = "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/5km/v3.1_op/nc/v1.0/monthly"
CRW_COMPOSITE_PAGE = "https://coralreefwatch.noaa.gov/product/5km/index_5km_composite.php"

DEFAULT_EXTRACTED_CACHE = Path("data/external/noaa/cache/crw5km_composites/extracted/crw5km_monthly_points_extracted.csv")
DEFAULT_RAW_CACHE_DIR = Path("data/external/noaa/cache/crw5km_composites/raw_monthly_tmp")
DEFAULT_FEATURE_OUTPUT = Path("data/processed/crw5km_composite_features.csv")
FEATURE_DIAGNOSTICS_OUTPUT = Path("results/tables/crw5km_composite_feature_diagnostics.csv")
MODEL_COMPARISON_OUTPUT = Path("results/tables/crw5km_composite_model_comparison.csv")
REPORT_OUTPUT = Path("outputs/diagnostics/crw5km_composite_feature_report.md")

MODEL_START_YEAR = 1989
TRAIN_END_YEAR = 2016
TEST_START_YEAR = 2021
TEST_END_YEAR = 2024
LAG_START_YEAR = MODEL_START_YEAR - 1
TARGET_NEW_DECLINE = "new_decline_event_next"
TARGET_AT_RISK = "decline_event_next_at_risk_gt005"
SEARCH_RADIUS_GRID_CELLS = 16

CRW_BASE_FEATURES = [
    "annual_mean_sst_crw5km",
    "spring_mean_sst_crw5km",
    "summer_mean_sst_crw5km",
    "warmest_month_mean_sst_crw5km",
    "annual_mean_ssta_crw5km",
    "spring_ssta_crw5km",
    "summer_ssta_crw5km",
    "annual_max_monthly_ssta_crw5km",
]
CRW_LAG_FEATURES = [f"lag1_{feature}" for feature in CRW_BASE_FEATURES]
CRW_FEATURES = CRW_BASE_FEATURES + CRW_LAG_FEATURES


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
    parser = argparse.ArgumentParser(description="Build CRW 5 km monthly-composite SST features.")
    parser.add_argument("--input", type=Path, default=INPUT_DATASET)
    parser.add_argument("--extracted-cache", type=Path, default=DEFAULT_EXTRACTED_CACHE)
    parser.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW_CACHE_DIR)
    parser.add_argument("--feature-output", type=Path, default=DEFAULT_FEATURE_OUTPUT)
    parser.add_argument("--feature-diagnostics-output", type=Path, default=FEATURE_DIAGNOSTICS_OUTPUT)
    parser.add_argument("--model-comparison-output", type=Path, default=MODEL_COMPARISON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    parser.add_argument("--start-year", type=int, default=LAG_START_YEAR)
    parser.add_argument("--end-year", type=int, default=TEST_END_YEAR)
    parser.add_argument("--limit-cells", type=int, default=None, help="Limit cells for smoke tests.")
    parser.add_argument("--keep-raw-cache", action="store_true", help="Keep downloaded monthly NetCDF files.")
    parser.add_argument("--force-refresh-extracted", action="store_true", help="Ignore existing extracted monthly point cache.")
    parser.add_argument("--delay-seconds", type=float, default=0.1, help="Polite delay between monthly file requests.")
    parser.add_argument("--max-workers", type=int, default=4, help="Small parallelism for monthly STAR file extraction.")
    return parser.parse_args()


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
    """Load the V1 modeling rows and add transition/actionable targets."""
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
    return data[["cell_id", "center_lat", "center_lon"]].drop_duplicates().reset_index(drop=True)


def star_monthly_url(year: int, month: int, product: str) -> str:
    """Return a predictable NOAA STAR monthly CRW NetCDF URL."""
    yyyymm = f"{year}{month:02d}"
    return f"{STAR_MONTHLY_ROOT}/{year}/ct5km_{product}_v3.1_{yyyymm}.nc"


def download_monthly_file(url: str, output_path: Path, delay_seconds: float) -> Path:
    """Download one monthly NetCDF file to a temporary path."""
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    response = requests.get(url, timeout=180)
    if response.status_code != 200:
        raise RuntimeError(f"Download failed: HTTP {response.status_code} for {url}")
    output_path.write_bytes(response.content)
    time.sleep(delay_seconds)
    return output_path


def nearest_index(values: np.ndarray, target: float) -> int:
    """Return nearest coordinate index."""
    return int(np.nanargmin(np.abs(values - target)))


def scalar_value(variable, lat_index: int, lon_index: int) -> float:
    """Read a scaled scalar value from a CRW NetCDF variable."""
    value = variable[0, lat_index, lon_index]
    if np.ma.is_masked(value):
        return np.nan
    return float(value)


def build_assignments(cells: pd.DataFrame, sample_sst_path: Path) -> pd.DataFrame:
    """Assign each Kelpwatch cell to the nearest valid CRW 5 km ocean grid point."""
    rows: list[dict[str, object]] = []
    with Dataset(sample_sst_path) as dataset:
        latitudes = np.array(dataset.variables["lat"][:], dtype=float)
        longitudes = np.array(dataset.variables["lon"][:], dtype=float)
        sst = dataset.variables["sea_surface_temperature"]
        for cell in cells.itertuples(index=False):
            center_lat = float(cell.center_lat)
            center_lon = float(cell.center_lon)
            nearest_lat_idx = nearest_index(latitudes, center_lat)
            nearest_lon_idx = nearest_index(longitudes, center_lon)
            lat_start = max(0, nearest_lat_idx - SEARCH_RADIUS_GRID_CELLS)
            lat_end = min(len(latitudes), nearest_lat_idx + SEARCH_RADIUS_GRID_CELLS + 1)
            lon_start = max(0, nearest_lon_idx - SEARCH_RADIUS_GRID_CELLS)
            lon_end = min(len(longitudes), nearest_lon_idx + SEARCH_RADIUS_GRID_CELLS + 1)
            candidates: list[tuple[float, int, int, float, float]] = []
            for lat_idx in range(lat_start, lat_end):
                for lon_idx in range(lon_start, lon_end):
                    value = scalar_value(sst, lat_idx, lon_idx)
                    if np.isfinite(value):
                        lat = float(latitudes[lat_idx])
                        lon = float(longitudes[lon_idx])
                        distance = float(haversine_km(np.array([center_lat]), np.array([center_lon]), np.array([lat]), np.array([lon]))[0])
                        candidates.append((distance, lat_idx, lon_idx, lat, lon))
            if candidates:
                distance, lat_idx, lon_idx, crw_lat, crw_lon = min(candidates, key=lambda item: item[0])
                status = "nearest_valid_ocean_grid"
            else:
                lat_idx = nearest_lat_idx
                lon_idx = nearest_lon_idx
                crw_lat = float(latitudes[lat_idx])
                crw_lon = float(longitudes[lon_idx])
                distance = float(haversine_km(np.array([center_lat]), np.array([center_lon]), np.array([crw_lat]), np.array([crw_lon]))[0])
                status = "nearest_grid_no_valid_ocean_in_search_window"
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                    "crw_lat": crw_lat,
                    "crw_lon": crw_lon,
                    "lat_index": int(lat_idx),
                    "lon_index": int(lon_idx),
                    "distance_to_crw_grid_km": distance,
                    "assignment_status": status,
                }
            )
    return pd.DataFrame(rows)


def load_extracted_cache(path: Path, force_refresh: bool) -> pd.DataFrame:
    """Load the compact extracted monthly point cache if available."""
    if force_refresh or not path.exists():
        return pd.DataFrame()
    cache = pd.read_csv(path)
    if cache.empty:
        return cache
    cache["year"] = cache["year"].astype(int)
    cache["month"] = cache["month"].astype(int)
    return cache


def cache_has_month(cache: pd.DataFrame, year: int, month: int, expected_cells: set[str]) -> bool:
    """Return whether cache has successful rows for all expected cells in a month."""
    if cache.empty:
        return False
    subset = cache.loc[(cache["year"] == year) & (cache["month"] == month)]
    if subset.empty:
        return False
    successful = set(subset.loc[subset["extraction_status"] == "ok", "cell_id"].astype(str))
    return expected_cells.issubset(successful)


def append_cache(path: Path, rows: list[dict[str, object]]) -> None:
    """Append extracted monthly point rows to the compact CSV cache."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    write_header = not path.exists() or path.stat().st_size == 0
    frame.to_csv(path, mode="a", index=False, header=write_header)


def extract_month(
    year: int,
    month: int,
    assignments: pd.DataFrame,
    raw_cache_dir: Path,
    keep_raw_cache: bool,
    delay_seconds: float,
) -> list[dict[str, object]]:
    """Download one month of SST/SSTA, extract assigned cell values, and delete raw files."""
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    if keep_raw_cache:
        temp_dir = raw_cache_dir
        cleanup_paths = False
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="crw5km_monthly_")
        temp_dir = Path(temp_context.name)
        cleanup_paths = True

    sst_url = star_monthly_url(year, month, "sst-mean")
    ssta_url = star_monthly_url(year, month, "ssta-mean")
    sst_path = temp_dir / Path(sst_url).name
    ssta_path = temp_dir / Path(ssta_url).name
    rows: list[dict[str, object]] = []
    try:
        download_monthly_file(sst_url, sst_path, delay_seconds)
        download_monthly_file(ssta_url, ssta_path, delay_seconds)
        with Dataset(sst_path) as sst_dataset, Dataset(ssta_path) as ssta_dataset:
            sst = sst_dataset.variables["sea_surface_temperature"]
            ssta = ssta_dataset.variables["sea_surface_temperature_anomaly"]
            for assignment in assignments.itertuples(index=False):
                rows.append(
                    {
                        "cell_id": assignment.cell_id,
                        "year": year,
                        "month": month,
                        "days_in_month": calendar.monthrange(year, month)[1],
                        "crw_lat": assignment.crw_lat,
                        "crw_lon": assignment.crw_lon,
                        "distance_to_crw_grid_km": assignment.distance_to_crw_grid_km,
                        "sst_mean_crw5km": scalar_value(sst, int(assignment.lat_index), int(assignment.lon_index)),
                        "ssta_mean_crw5km": scalar_value(ssta, int(assignment.lat_index), int(assignment.lon_index)),
                        "source_file_sst": Path(sst_url).name,
                        "source_file_ssta": Path(ssta_url).name,
                        "extraction_status": "ok",
                    }
                )
    except Exception as error:
        for assignment in assignments.itertuples(index=False):
            rows.append(
                {
                    "cell_id": assignment.cell_id,
                    "year": year,
                    "month": month,
                    "days_in_month": calendar.monthrange(year, month)[1],
                    "crw_lat": assignment.crw_lat,
                    "crw_lon": assignment.crw_lon,
                    "distance_to_crw_grid_km": assignment.distance_to_crw_grid_km,
                    "sst_mean_crw5km": np.nan,
                    "ssta_mean_crw5km": np.nan,
                    "source_file_sst": Path(sst_url).name,
                    "source_file_ssta": Path(ssta_url).name,
                    "extraction_status": f"failed: {error}",
                }
            )
    finally:
        if cleanup_paths:
            temp_context.cleanup()
        else:
            for path in [sst_path, ssta_path]:
                if not keep_raw_cache and path.exists():
                    path.unlink()
    return rows


def ensure_extracted_monthly_cache(
    cells: pd.DataFrame,
    start_year: int,
    end_year: int,
    extracted_cache: Path,
    raw_cache_dir: Path,
    keep_raw_cache: bool,
    force_refresh: bool,
    delay_seconds: float,
    max_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Build or resume the compact monthly point extraction cache."""
    raw_cache_dir.mkdir(parents=True, exist_ok=True)
    extracted_cache.parent.mkdir(parents=True, exist_ok=True)
    if force_refresh and extracted_cache.exists():
        extracted_cache.unlink()

    sample_sst_path = raw_cache_dir / Path(star_monthly_url(TEST_START_YEAR, 1, "sst-mean")).name
    download_monthly_file(star_monthly_url(TEST_START_YEAR, 1, "sst-mean"), sample_sst_path, delay_seconds)
    assignments = build_assignments(cells, sample_sst_path)
    if not keep_raw_cache and sample_sst_path.exists():
        sample_sst_path.unlink()

    cache = load_extracted_cache(extracted_cache, False)
    expected_cells = set(assignments["cell_id"].astype(str))
    months_to_process: list[tuple[int, int]] = []
    skipped_months = 0
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if cache_has_month(cache, year, month, expected_cells):
                skipped_months += 1
                continue
            months_to_process.append((year, month))

    processed_months = 0
    failed_months = 0
    workers = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(extract_month, year, month, assignments, raw_cache_dir, keep_raw_cache, delay_seconds): (year, month)
            for year, month in months_to_process
        }
        for future in as_completed(futures):
            year, month = futures[future]
            print(f"Extracted CRW monthly composite points for {year}-{month:02d}")
            rows = future.result()
            append_cache(extracted_cache, rows)
            processed_months += 1
            if any(str(row["extraction_status"]).startswith("failed") for row in rows):
                failed_months += 1
            cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True) if not cache.empty else pd.DataFrame(rows)

    final_cache = load_extracted_cache(extracted_cache, False)
    stats = {
        "processed_months_this_run": processed_months,
        "skipped_cached_months": skipped_months,
        "failed_months_this_run": failed_months,
    }
    return final_cache, assignments, stats


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    """Return a days-weighted average while ignoring missing values."""
    valid = values.notna() & weights.notna()
    if not valid.any():
        return np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def annual_feature_rows(monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate extracted monthly point values to annual/seasonal CRW features."""
    monthly = monthly.loc[monthly["extraction_status"] == "ok"].copy()
    monthly["days_in_month"] = pd.to_numeric(monthly["days_in_month"], errors="coerce")
    rows: list[dict[str, object]] = []
    for (cell_id, year), group in monthly.groupby(["cell_id", "year"]):
        spring = group.loc[group["month"].between(4, 6)]
        summer = group.loc[group["month"].between(7, 9)]
        rows.append(
            {
                "cell_id": cell_id,
                "year": int(year),
                "annual_mean_sst_crw5km": weighted_average(group["sst_mean_crw5km"], group["days_in_month"]),
                "spring_mean_sst_crw5km": weighted_average(spring["sst_mean_crw5km"], spring["days_in_month"]),
                "summer_mean_sst_crw5km": weighted_average(summer["sst_mean_crw5km"], summer["days_in_month"]),
                "warmest_month_mean_sst_crw5km": float(group["sst_mean_crw5km"].max(skipna=True)),
                "annual_mean_ssta_crw5km": weighted_average(group["ssta_mean_crw5km"], group["days_in_month"]),
                "spring_ssta_crw5km": weighted_average(spring["ssta_mean_crw5km"], spring["days_in_month"]),
                "summer_ssta_crw5km": weighted_average(summer["ssta_mean_crw5km"], summer["days_in_month"]),
                "annual_max_monthly_ssta_crw5km": float(group["ssta_mean_crw5km"].max(skipna=True)),
                "nearest_crw_lat": float(group["crw_lat"].iloc[0]),
                "nearest_crw_lon": float(group["crw_lon"].iloc[0]),
                "distance_to_crw_grid_km": float(group["distance_to_crw_grid_km"].iloc[0]),
                "n_crw_grid_points": 1,
                "monthly_records_available": int(group[["sst_mean_crw5km", "ssta_mean_crw5km"]].notna().all(axis=1).sum()),
            }
        )
    features = pd.DataFrame(rows).sort_values(["cell_id", "year"]).reset_index(drop=True)
    grouped = features.groupby("cell_id", group_keys=False)
    for feature in CRW_BASE_FEATURES:
        features[f"lag1_{feature}"] = grouped[feature].shift(1)
    return features.loc[features["year"].between(MODEL_START_YEAR, TEST_END_YEAR)].reset_index(drop=True)


def feature_diagnostics(data: pd.DataFrame, features: pd.DataFrame, extracted: pd.DataFrame, assignments: pd.DataFrame, stats: dict[str, int]) -> pd.DataFrame:
    """Summarize CRW composite support, missingness, and correlation with OISST."""
    merged = data.merge(features, on=["cell_id", "year"], how="left")
    rows: list[dict[str, object]] = []
    for feature in CRW_FEATURES:
        if feature in merged.columns:
            rows.append(
                {
                    "diagnostic": "missingness",
                    "feature": feature,
                    "comparison_feature": "",
                    "value": float(merged[feature].isna().mean()),
                    "status": "computed",
                    "notes": "Fraction of retained Kelpwatch cell-year rows missing the CRW composite feature.",
                }
            )
    pairs = [
        ("annual_mean_sst_crw5km", "annual_mean_sst"),
        ("annual_mean_ssta_crw5km", "annual_mean_sst_anomaly"),
        ("lag1_annual_mean_ssta_crw5km", "lag1_annual_mean_sst_anomaly"),
    ]
    for crw_feature, oisst_feature in pairs:
        if crw_feature in merged.columns and oisst_feature in merged.columns:
            valid = merged[[crw_feature, oisst_feature]].dropna()
            rows.append(
                {
                    "diagnostic": "feature_correlation",
                    "feature": crw_feature,
                    "comparison_feature": oisst_feature,
                    "value": float(valid[crw_feature].corr(valid[oisst_feature])) if len(valid) >= 3 else np.nan,
                    "status": "computed",
                    "notes": "Pearson correlation across matched cell-year rows.",
                }
            )
    support_values = {
        "monthly_files_successfully_processed": int(extracted.loc[extracted["extraction_status"] == "ok", ["year", "month"]].drop_duplicates().shape[0]),
        "monthly_point_rows_extracted": int(len(extracted)),
        "annual_feature_rows": int(len(features)),
        "unique_cells": int(features["cell_id"].nunique()) if not features.empty else 0,
        "mean_distance_to_crw_grid_km": float(assignments["distance_to_crw_grid_km"].mean()),
        "max_distance_to_crw_grid_km": float(assignments["distance_to_crw_grid_km"].max()),
        **stats,
    }
    for feature, value in support_values.items():
        rows.append(
            {
                "diagnostic": "support",
                "feature": feature,
                "comparison_feature": "",
                "value": value,
                "status": "computed",
                "notes": "CRW monthly composite extraction support diagnostic.",
            }
        )
    rows.append(
        {
            "diagnostic": "data_access",
            "feature": "NOAA_STAR_monthly_netcdf",
            "comparison_feature": "",
            "value": np.nan,
            "status": "computed",
            "notes": f"Monthly SST mean and SSTA mean files were streamed from {STAR_MONTHLY_ROOT}/{{year}}/.",
        }
    )
    return pd.DataFrame(rows)


def available(features: list[str], data: pd.DataFrame) -> list[str]:
    """Return features available in the dataset."""
    return [feature for feature in features if feature in data.columns]


def model_feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    """Define OISST/CRW/canopy comparison feature families."""
    return {
        "oisst_only": available(OISST_FEATURES, data),
        "crw_composite_only": available(CRW_FEATURES, data),
        "oisst_plus_crw_composite": available(OISST_FEATURES + CRW_FEATURES, data),
        "canopy_only": available(CANOPY_FEATURES, data),
        "canopy_plus_crw_composite": available(CANOPY_FEATURES + CRW_FEATURES, data),
    }


def has_two_classes(frame: pd.DataFrame, target: str) -> bool:
    """Return whether both target classes are present."""
    return set(frame[target].dropna().astype(int).unique()) == {0, 1}


def preprocess(features: list[str]) -> ColumnTransformer:
    """Create numeric preprocessing."""
    return ColumnTransformer(
        transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), features)]
    )


def evaluate_models(data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Compare OISST, CRW composite, combined, and canopy feature families."""
    merged = data.merge(features, on=["cell_id", "year"], how="left", validate="one_to_one")
    sets = model_feature_sets(merged)
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
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        working = merged.copy()
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
                    "roc_auc": np.nan,
                    "recall": np.nan,
                    "precision": np.nan,
                    "f1": np.nan,
                    "balanced_accuracy": np.nan,
                    "status": "insufficient_target_classes",
                }
            )
            continue
        for family, family_features in sets.items():
            if not family_features:
                continue
            for model_name, estimator in estimators.items():
                pipeline = Pipeline([("preprocess", preprocess(family_features)), ("model", estimator)])
                pipeline.fit(train[family_features], train[target.target].astype(int))
                scores = pipeline.predict_proba(test[family_features])[:, 1]
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
                        "roc_auc": roc_auc_score(y_true, scores) if len(set(y_true)) == 2 else np.nan,
                        "recall": recall_score(y_true, predictions, zero_division=0),
                        "precision": precision_score(y_true, predictions, zero_division=0),
                        "f1": f1_score(y_true, predictions, zero_division=0),
                        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
                        "status": "computed",
                    }
                )
    return pd.DataFrame(rows)


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


def best_by_target_and_family(model_results: pd.DataFrame) -> pd.DataFrame:
    """Return the best PR-AUC row for each target and feature family."""
    computed = model_results.loc[model_results["status"] == "computed"].copy()
    if computed.empty:
        return computed
    return (
        computed.sort_values(["target_definition", "feature_family", "pr_auc"], ascending=[True, True, False])
        .groupby(["target_definition", "feature_family"])
        .head(1)
        .reset_index(drop=True)
    )


def write_report(
    output: Path,
    features: pd.DataFrame,
    diagnostics: pd.DataFrame,
    model_results: pd.DataFrame,
    assignments: pd.DataFrame,
    extracted: pd.DataFrame,
    stats: dict[str, int],
) -> None:
    """Write a CRW composite feature and model-comparison report."""
    computed = model_results.loc[model_results["status"] == "computed"].copy()
    best = best_by_target_and_family(model_results)
    top_by_target = computed.sort_values(["target_definition", "pr_auc"], ascending=[True, False]).groupby("target_definition").head(1)
    missingness = diagnostics.loc[diagnostics["diagnostic"] == "missingness", ["feature", "value"]].copy()
    mean_missingness = float(missingness["value"].mean()) if not missingness.empty else np.nan
    successful_months = int(extracted.loc[extracted["extraction_status"] == "ok", ["year", "month"]].drop_duplicates().shape[0])
    lines = [
        "# CRW 5 km Composite SST Feature Report",
        "",
        "## Purpose",
        "",
        "This report adds a NOAA Coral Reef Watch 5 km monthly-composite SST exposure layer.",
        "It does not replace the existing OISST V1/V2 workflow or the optional daily CRW point-cache path.",
        "",
        "## Data Access",
        "",
        "- Daily CRW point-cache extraction was tested previously but was too slow for the current environment.",
        "- The ERDDAP monthly bbox path was also inconsistent for full 1988-2024 extraction in this environment.",
        "- This run streams predictable NOAA STAR monthly NetCDF files, extracts retained Kelpwatch cell points, and deletes raw NetCDF files unless `--keep-raw-cache` is used.",
        f"- NOAA STAR monthly root: {STAR_MONTHLY_ROOT}/",
        f"- CRW 5 km composite product page: {CRW_COMPOSITE_PAGE}",
        "",
        "## Spatial Matching",
        "",
        "- Baseline extraction uses the nearest valid CRW 5 km ocean grid cell to each retained Kelpwatch 10 km cell centroid.",
        "- The compact extracted cache stores only point-level monthly SST/SSTA values, source filenames, and extraction status.",
        "- CRW 5 km remains satellite SST exposure, not true in-situ nearshore temperature.",
        "",
        "## Feature Construction Summary",
        "",
        f"- Monthly files successfully processed: `{successful_months}` month pairs",
        f"- Extracted monthly point rows: `{len(extracted)}`",
        f"- Annual CRW feature rows built: `{len(features)}`",
        f"- Unique Kelpwatch cells: `{features['cell_id'].nunique() if not features.empty else 0}`",
        f"- Mean CRW feature missingness: `{mean_missingness:.4f}`",
        f"- Mean nearest-grid distance: `{assignments['distance_to_crw_grid_km'].mean():.3f}` km",
        f"- Max nearest-grid distance: `{assignments['distance_to_crw_grid_km'].max():.3f}` km",
        f"- Processed months this run: `{stats['processed_months_this_run']}`",
        f"- Cached months reused: `{stats['skipped_cached_months']}`",
        f"- Failed months this run: `{stats['failed_months_this_run']}`",
        "",
        "Composite features summarize monthly mean SST and SSTA. They cannot fully reproduce daily hot-day counts, cumulative heat stress, or short marine-heatwave duration metrics.",
        "",
        "## Model Comparison",
        "",
        f"- Computed model-comparison rows: `{len(computed)}`",
        "",
        "Best result per target:",
        "",
        small_markdown_table(
            top_by_target[
                ["target_definition", "feature_family", "model", "pr_auc", "recall", "precision", "f1", "roc_auc"]
            ]
        )
        if not top_by_target.empty
        else "No computed model rows.",
        "",
        "Best row by target and feature family:",
        "",
        small_markdown_table(
            best[
                ["target_definition", "feature_family", "model", "pr_auc", "recall", "precision", "f1", "roc_auc"]
            ]
        )
        if not best.empty
        else "No computed model rows.",
        "",
        "## Interpretation",
        "",
        "CRW composite features should be interpreted as a higher-resolution satellite SST exposure alternative to OISST. They are not local in-situ temperature measurements.",
        "Improvements should be claimed only where the computed comparison table supports them. If CRW composite features do not improve transition-oriented targets, the result still supports the broader interpretation that abrupt kelp transitions likely require local ecological covariates in addition to satellite SST exposure.",
        "",
        "## Output Files",
        "",
        f"- `{DEFAULT_FEATURE_OUTPUT}`",
        f"- `{FEATURE_DIAGNOSTICS_OUTPUT}`",
        f"- `{MODEL_COMPARISON_OUTPUT}`",
        f"- `{REPORT_OUTPUT}`",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run CRW monthly-composite feature construction and model comparison."""
    args = parse_args()
    for output in [args.feature_output, args.feature_diagnostics_output, args.model_comparison_output, args.report_output]:
        output.parent.mkdir(parents=True, exist_ok=True)

    data = load_base_rows(args.input, args.limit_cells)
    cells = cell_metadata(data)
    extracted, assignments, stats = ensure_extracted_monthly_cache(
        cells=cells,
        start_year=args.start_year,
        end_year=args.end_year,
        extracted_cache=args.extracted_cache,
        raw_cache_dir=args.raw_cache_dir,
        keep_raw_cache=args.keep_raw_cache,
        force_refresh=args.force_refresh_extracted,
        delay_seconds=args.delay_seconds,
        max_workers=args.max_workers,
    )
    features = annual_feature_rows(extracted)
    diagnostics = feature_diagnostics(data, features, extracted, assignments, stats)
    model_results = evaluate_models(data, features)

    features.to_csv(args.feature_output, index=False)
    diagnostics.to_csv(args.feature_diagnostics_output, index=False)
    model_results.to_csv(args.model_comparison_output, index=False)
    write_report(args.report_output, features, diagnostics, model_results, assignments, extracted, stats)

    computed_rows = int((model_results["status"] == "computed").sum())
    missingness = diagnostics.loc[diagnostics["diagnostic"] == "missingness", "value"]
    successful_months = int(extracted.loc[extracted["extraction_status"] == "ok", ["year", "month"]].drop_duplicates().shape[0])
    print(f"Monthly file pairs successfully processed: {successful_months}")
    print(f"Extracted monthly point rows: {len(extracted)}")
    print(f"Annual CRW feature rows built: {len(features)}")
    print(f"Mean CRW feature missingness: {missingness.mean():.4f}")
    print(f"Computed model-comparison rows: {computed_rows}")
    print(f"Wrote features: {args.feature_output}")
    print(f"Wrote diagnostics: {args.feature_diagnostics_output}")
    print(f"Wrote model comparison: {args.model_comparison_output}")
    print(f"Wrote report: {args.report_output}")


if __name__ == "__main__":
    main()
