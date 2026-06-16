"""Build static bathymetry and habitat covariates for retained Kelpwatch cells.

The script downloads a small GEBCO California coastal subset through the GEBCO
queue API when no local bathymetry file is supplied, extracts ocean-pixel
summaries for the retained 10 km Kelpwatch cells, and evaluates whether static
habitat context improves decline prediction beyond canopy persistence and SST
exposure layers.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import shapely
from netCDF4 import Dataset
from shapely.geometry import shape
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
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


GEBCO_DOWNLOAD_BASE = "https://download.gebco.net"
GEBCO_GRID_ID = 1
GEBCO_BATHYMETRY_DATA_SOURCE_ID = 1
GEBCO_NETCDF_FORMAT_ID = 1
GEBCO_PRODUCT_PAGE = "https://www.gebco.net/data-products/gridded-bathymetry-data"

FILTERED_CELLS = Path("geometries/regular_10km_fishnet/filtered_cells_historic_footprint_ge500.csv")
GEOJSON_DIR = Path("geometries/regular_10km_fishnet/single_cell_geojsons")
DEFAULT_FEATURE_OUTPUT = Path("data/processed/bathymetry_habitat_features.csv")
FEATURE_DIAGNOSTICS_OUTPUT = Path("results/tables/bathymetry_habitat_feature_diagnostics.csv")
MODEL_COMPARISON_OUTPUT = Path("results/tables/bathymetry_habitat_model_comparison.csv")
REPORT_OUTPUT = Path("outputs/diagnostics/bathymetry_habitat_feature_report.md")

MODEL_START_YEAR = 1989
TRAIN_END_YEAR = 2016
VALIDATION_START_YEAR = 2017
VALIDATION_END_YEAR = 2020
TEST_START_YEAR = 2021
TEST_END_YEAR = 2024
TARGET_NEW_DECLINE = "new_decline_event_next"
TARGET_AT_RISK = "decline_event_next_at_risk_gt005"
COASTAL_BBOX_PADDING_DEGREES = 0.05
DEFAULT_QUEUE_TIMEOUT_SECONDS = 300

HABITAT_FEATURES = [
    "mean_depth_m",
    "min_depth_m",
    "max_depth_m",
    "depth_range_m",
    "shallow_area_share_0_30m",
    "shallow_area_share_0_50m",
    "slope_mean",
    "slope_std",
    "n_bathymetry_pixels_used",
    "ocean_pixel_share",
    "bathymetry_missing_rate",
]
CRW_COMPOSITE_FEATURES = [
    "annual_mean_sst_crw5km",
    "spring_mean_sst_crw5km",
    "summer_mean_sst_crw5km",
    "warmest_month_mean_sst_crw5km",
    "annual_mean_ssta_crw5km",
    "spring_ssta_crw5km",
    "summer_ssta_crw5km",
    "annual_max_monthly_ssta_crw5km",
    "lag1_annual_mean_sst_crw5km",
    "lag1_annual_mean_ssta_crw5km",
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
    parser = argparse.ArgumentParser(description="Build GEBCO bathymetry/habitat features.")
    parser.add_argument("--input", type=Path, default=INPUT_DATASET)
    parser.add_argument("--filtered-cells", type=Path, default=FILTERED_CELLS)
    parser.add_argument("--geojson-dir", type=Path, default=GEOJSON_DIR)
    parser.add_argument("--gebco-netcdf", type=Path, default=None, help="Optional local GEBCO NetCDF subset.")
    parser.add_argument("--feature-output", type=Path, default=DEFAULT_FEATURE_OUTPUT)
    parser.add_argument("--feature-diagnostics-output", type=Path, default=FEATURE_DIAGNOSTICS_OUTPUT)
    parser.add_argument("--model-comparison-output", type=Path, default=MODEL_COMPARISON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    parser.add_argument("--keep-raw-download", action="store_true", help="Keep downloaded GEBCO zip/NetCDF under data/external.")
    parser.add_argument("--dry-run", action="store_true", help="Write access diagnostics without downloading GEBCO.")
    return parser.parse_args()


def load_base_rows(input_path: Path) -> pd.DataFrame:
    """Load V1 modeling rows and add transition/actionable targets."""
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    data = main_subset(pd.read_csv(input_path).sort_values(["cell_id", "year"]).reset_index(drop=True))
    data = add_actionable_labels(data)
    data[TARGET_NEW_DECLINE] = ((data[CANOPY] >= data[BASELINE_P25]) & (data[NEXT_CANOPY] < data[BASELINE_P25])).astype(int)
    data[TARGET_AT_RISK] = data["decline_event_next"].astype(int)
    data["at_risk_gt005"] = data[CANOPY] > 0.05
    return data


def load_cell_polygons(filtered_cells: Path, geojson_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load retained-cell metadata and single-cell GeoJSON polygons."""
    cells = pd.read_csv(filtered_cells)
    polygons: dict[str, object] = {}
    for row in cells.itertuples(index=False):
        path = geojson_dir / str(row.geojson_file)
        if not path.exists():
            raise FileNotFoundError(path)
        geojson = json.loads(path.read_text(encoding="utf-8"))
        feature = geojson["features"][0] if geojson.get("type") == "FeatureCollection" else geojson
        polygons[str(row.cell_id)] = shape(feature["geometry"])
    return cells, polygons


def padded_bbox(polygons: dict[str, object]) -> tuple[float, float, float, float]:
    """Return west, east, south, north bounding box with a small padding."""
    bounds = np.array([polygon.bounds for polygon in polygons.values()])
    west = float(bounds[:, 0].min() - COASTAL_BBOX_PADDING_DEGREES)
    south = float(bounds[:, 1].min() - COASTAL_BBOX_PADDING_DEGREES)
    east = float(bounds[:, 2].max() + COASTAL_BBOX_PADDING_DEGREES)
    north = float(bounds[:, 3].max() + COASTAL_BBOX_PADDING_DEGREES)
    return west, east, south, north


def submit_gebco_queue(west: float, east: float, south: float, north: float) -> str:
    """Submit a small GEBCO NetCDF subset request and return the basket id."""
    payload = {
        "id": "0",
        "email": None,
        "submission_date": "2026-06-17T00:00:00.000000",
        "processing_status": "new",
        "items": [
            {
                "id": 0,
                "grid_id": GEBCO_GRID_ID,
                "data_source_ids": [GEBCO_BATHYMETRY_DATA_SOURCE_ID],
                "formats": [GEBCO_NETCDF_FORMAT_ID],
                "left": west,
                "right": east,
                "top": north,
                "bottom": south,
            }
        ],
    }
    response = requests.post(
        f"{GEBCO_DOWNLOAD_BASE}/api/queue",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return str(response.json()["basketId"])


def wait_for_gebco_queue(basket_id: str, timeout_seconds: int = DEFAULT_QUEUE_TIMEOUT_SECONDS) -> None:
    """Wait for a GEBCO queue item to finish."""
    start = time.time()
    while time.time() - start < timeout_seconds:
        response = requests.get(f"{GEBCO_DOWNLOAD_BASE}/api/queue/status/{basket_id}", timeout=30)
        response.raise_for_status()
        status = response.json().get("status")
        if status == "finished":
            return
        if status in {"failed", "error"}:
            raise RuntimeError(f"GEBCO queue failed for {basket_id}: {response.text}")
        time.sleep(5)
    raise TimeoutError(f"GEBCO queue timed out after {timeout_seconds} seconds for {basket_id}")


def download_gebco_subset(west: float, east: float, south: float, north: float, keep_raw: bool) -> tuple[Path, tempfile.TemporaryDirectory | None, dict[str, object]]:
    """Download a GEBCO subset zip and extract the NetCDF path."""
    basket_id = submit_gebco_queue(west, east, south, north)
    wait_for_gebco_queue(basket_id)
    response = requests.get(f"{GEBCO_DOWNLOAD_BASE}/api/queue/download/{basket_id}", timeout=120)
    response.raise_for_status()
    if keep_raw:
        raw_dir = Path("data/external/bathymetry/gebco_2026")
        raw_dir.mkdir(parents=True, exist_ok=True)
        zip_path = raw_dir / f"{basket_id}.zip"
        zip_path.write_bytes(response.content)
        extract_dir = raw_dir / basket_id
        extract_dir.mkdir(parents=True, exist_ok=True)
        temp_context = None
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="gebco_bathymetry_")
        zip_path = Path(temp_context.name) / f"{basket_id}.zip"
        zip_path.write_bytes(response.content)
        extract_dir = Path(temp_context.name) / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
        netcdf_members = [member for member in archive.namelist() if member.endswith(".nc")]
    if not netcdf_members:
        raise FileNotFoundError("GEBCO subset zip did not contain a NetCDF file.")
    metadata = {
        "basket_id": basket_id,
        "download_size_bytes": len(response.content),
        "netcdf_file": netcdf_members[0],
        "bbox_west": west,
        "bbox_east": east,
        "bbox_south": south,
        "bbox_north": north,
    }
    return extract_dir / netcdf_members[0], temp_context, metadata


def slope_from_elevation(elevation: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Approximate terrain slope magnitude from GEBCO elevation in m/m."""
    lat_spacing_m = abs(float(np.nanmedian(np.diff(latitudes)))) * 111_320.0
    lon_spacing_degrees = abs(float(np.nanmedian(np.diff(longitudes))))
    mean_lat = np.radians(float(np.nanmean(latitudes)))
    lon_spacing_m = lon_spacing_degrees * 111_320.0 * max(float(np.cos(mean_lat)), 0.1)
    grad_y, grad_x = np.gradient(elevation.astype(float), lat_spacing_m, lon_spacing_m)
    return np.sqrt(grad_x**2 + grad_y**2)


def bathymetry_features_from_netcdf(netcdf_path: Path, polygons: dict[str, object]) -> pd.DataFrame:
    """Compute ocean-only bathymetry/habitat summaries for retained cells."""
    with Dataset(netcdf_path) as dataset:
        latitudes = np.array(dataset.variables["lat"][:], dtype=float)
        longitudes = np.array(dataset.variables["lon"][:], dtype=float)
        elevation = np.ma.filled(dataset.variables["elevation"][:], np.nan).astype(float)
    slope = slope_from_elevation(elevation, latitudes, longitudes)

    rows: list[dict[str, object]] = []
    for cell_id, polygon in polygons.items():
        west, south, east, north = polygon.bounds
        lat_idx = np.where((latitudes >= south) & (latitudes <= north))[0]
        lon_idx = np.where((longitudes >= west) & (longitudes <= east))[0]
        if len(lat_idx) == 0 or len(lon_idx) == 0:
            rows.append({"cell_id": cell_id, "feature_status": "no_bathymetry_pixels_in_cell_bounds"})
            continue
        lat_subset = latitudes[lat_idx]
        lon_subset = longitudes[lon_idx]
        lon_grid, lat_grid = np.meshgrid(lon_subset, lat_subset)
        inside = shapely.contains_xy(polygon, lon_grid.ravel(), lat_grid.ravel()).reshape(lon_grid.shape)
        elev_subset = elevation[np.ix_(lat_idx, lon_idx)]
        slope_subset = slope[np.ix_(lat_idx, lon_idx)]
        candidate = inside & np.isfinite(elev_subset)
        ocean = candidate & (elev_subset < 0)
        total_inside = int(inside.sum())
        valid_inside = int(candidate.sum())
        ocean_count = int(ocean.sum())
        if ocean_count == 0:
            rows.append(
                {
                    "cell_id": cell_id,
                    "mean_depth_m": np.nan,
                    "min_depth_m": np.nan,
                    "max_depth_m": np.nan,
                    "depth_range_m": np.nan,
                    "shallow_area_share_0_30m": np.nan,
                    "shallow_area_share_0_50m": np.nan,
                    "slope_mean": np.nan,
                    "slope_std": np.nan,
                    "n_bathymetry_pixels_used": 0,
                    "ocean_pixel_share": 0.0,
                    "bathymetry_missing_rate": 1.0 - valid_inside / total_inside if total_inside else np.nan,
                    "feature_status": "no_ocean_pixels",
                }
            )
            continue
        depth = -elev_subset[ocean]
        ocean_slope = slope_subset[ocean]
        rows.append(
            {
                "cell_id": cell_id,
                "mean_depth_m": float(np.nanmean(depth)),
                "min_depth_m": float(np.nanmin(depth)),
                "max_depth_m": float(np.nanmax(depth)),
                "depth_range_m": float(np.nanmax(depth) - np.nanmin(depth)),
                "shallow_area_share_0_30m": float(np.nanmean((depth >= 0) & (depth <= 30))),
                "shallow_area_share_0_50m": float(np.nanmean((depth >= 0) & (depth <= 50))),
                "slope_mean": float(np.nanmean(ocean_slope)),
                "slope_std": float(np.nanstd(ocean_slope)),
                "n_bathymetry_pixels_used": ocean_count,
                "ocean_pixel_share": float(ocean_count / total_inside) if total_inside else np.nan,
                "bathymetry_missing_rate": float(1.0 - valid_inside / total_inside) if total_inside else np.nan,
                "feature_status": "computed",
            }
        )
    return pd.DataFrame(rows).sort_values("cell_id").reset_index(drop=True)


def add_crw_if_available(data: pd.DataFrame) -> pd.DataFrame:
    """Merge CRW composite features if the local ignored processed file exists."""
    crw_path = Path("data/processed/crw5km_composite_features.csv")
    if not crw_path.exists():
        return data
    crw = pd.read_csv(crw_path)
    return data.merge(crw, on=["cell_id", "year"], how="left", validate="one_to_one")


def model_feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    """Define feature families for bathymetry/habitat comparison."""
    available = lambda columns: [column for column in columns if column in data.columns]
    sets = {
        "canopy_only": available(CANOPY_FEATURES),
        "oisst_only": available(OISST_FEATURES),
        "habitat_only": available(HABITAT_FEATURES),
        "oisst_plus_habitat": available(OISST_FEATURES + HABITAT_FEATURES),
        "canopy_plus_oisst_plus_habitat": available(CANOPY_FEATURES + OISST_FEATURES + HABITAT_FEATURES),
    }
    crw_features = available(CRW_COMPOSITE_FEATURES)
    if crw_features:
        sets.update(
            {
                "crw_composite_only": crw_features,
                "crw_plus_habitat": available(CRW_COMPOSITE_FEATURES + HABITAT_FEATURES),
                "canopy_plus_crw_plus_habitat": available(CANOPY_FEATURES + CRW_COMPOSITE_FEATURES + HABITAT_FEATURES),
            }
        )
    return {name: features for name, features in sets.items() if features}


def has_two_classes(frame: pd.DataFrame, target: str) -> bool:
    """Return whether both target classes are present."""
    return set(frame[target].dropna().astype(int).unique()) == {0, 1}


def preprocess(features: list[str]) -> ColumnTransformer:
    """Create numeric preprocessing."""
    return ColumnTransformer(
        transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), features)]
    )


def validation_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    """Select a validation threshold that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(y_true.astype(int), scores)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metric_row(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Return test metrics for scores and a fixed threshold."""
    predictions = (scores >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) == 2 else np.nan,
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
    }


def evaluate_naive_persistence(working: pd.DataFrame, target: TargetSpec) -> dict[str, object]:
    """Evaluate a simple current-low-canopy persistence risk score."""
    train = working.loc[working["year"].between(MODEL_START_YEAR, TRAIN_END_YEAR)].copy()
    validation = working.loc[working["year"].between(VALIDATION_START_YEAR, VALIDATION_END_YEAR)].copy()
    test = working.loc[working["year"].between(TEST_START_YEAR, TEST_END_YEAR)].copy()
    risk_column = "relative_canopy"
    if train.empty or validation.empty or test.empty or not has_two_classes(validation, target.target) or not has_two_classes(test, target.target):
        return {
            "target_definition": target.name,
            "feature_family": "naive_persistence_baseline",
            "model": "current_low_canopy_score",
            "status": "insufficient_target_classes",
            "n_train": len(train),
            "n_validation": len(validation),
            "n_test": len(test),
            "positive_events_test": int(test[target.target].sum()) if target.target in test else 0,
            "event_prevalence_test": float(test[target.target].mean()) if len(test) else np.nan,
        }
    validation_scores = 1.0 - validation[risk_column].to_numpy(dtype=float)
    test_scores = 1.0 - test[risk_column].to_numpy(dtype=float)
    threshold = validation_threshold(validation[target.target], validation_scores)
    row = metric_row(test[target.target].astype(int), test_scores, threshold)
    row.update(
        {
            "target_definition": target.name,
            "feature_family": "naive_persistence_baseline",
            "model": "current_low_canopy_score",
            "n_train": len(train),
            "n_validation": len(validation),
            "n_test": len(test),
            "positive_events_test": int(test[target.target].sum()),
            "event_prevalence_test": float(test[target.target].mean()),
            "decision_threshold": threshold,
            "status": "computed",
        }
    )
    return row


def evaluate_models(data: pd.DataFrame) -> pd.DataFrame:
    """Run model comparisons for habitat and existing exposure feature families."""
    feature_sets = model_feature_sets(data)
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
        working = data.copy()
        if target.filter_column:
            working = working.loc[working[target.filter_column]].copy()
        rows.append(evaluate_naive_persistence(working, target))
        train = working.loc[working["year"].between(MODEL_START_YEAR, TRAIN_END_YEAR)].copy()
        validation = working.loc[working["year"].between(VALIDATION_START_YEAR, VALIDATION_END_YEAR)].copy()
        test = working.loc[working["year"].between(TEST_START_YEAR, TEST_END_YEAR)].copy()
        if train.empty or validation.empty or test.empty or not has_two_classes(train, target.target) or not has_two_classes(validation, target.target) or not has_two_classes(test, target.target):
            rows.append(
                {
                    "target_definition": target.name,
                    "feature_family": "all",
                    "model": "not_run",
                    "n_train": len(train),
                    "n_validation": len(validation),
                    "n_test": len(test),
                    "positive_events_test": int(test[target.target].sum()) if target.target in test else 0,
                    "event_prevalence_test": float(test[target.target].mean()) if len(test) else np.nan,
                    "status": "insufficient_target_classes",
                }
            )
            continue
        for family, features in feature_sets.items():
            for model_name, estimator in estimators.items():
                pipeline = Pipeline([("preprocess", preprocess(features)), ("model", estimator)])
                pipeline.fit(train[features], train[target.target].astype(int))
                validation_scores = pipeline.predict_proba(validation[features])[:, 1]
                test_scores = pipeline.predict_proba(test[features])[:, 1]
                threshold = validation_threshold(validation[target.target], validation_scores)
                row = metric_row(test[target.target].astype(int), test_scores, threshold)
                row.update(
                    {
                        "target_definition": target.name,
                        "feature_family": family,
                        "model": model_name,
                        "n_train": len(train),
                        "n_validation": len(validation),
                        "n_test": len(test),
                        "positive_events_test": int(test[target.target].sum()),
                        "event_prevalence_test": float(test[target.target].mean()),
                        "decision_threshold": threshold,
                        "status": "computed",
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def diagnostics_table(features: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    """Create feature diagnostics rows."""
    rows: list[dict[str, object]] = []
    for feature in HABITAT_FEATURES:
        if feature in features.columns:
            rows.append(
                {
                    "diagnostic": "feature_summary",
                    "feature": feature,
                    "value_min": float(features[feature].min(skipna=True)),
                    "value_max": float(features[feature].max(skipna=True)),
                    "value_mean": float(features[feature].mean(skipna=True)),
                    "missing_rate": float(features[feature].isna().mean()),
                    "notes": "Static GEBCO-derived bathymetry/habitat covariate.",
                }
            )
    status_counts = features["feature_status"].value_counts(dropna=False).to_dict()
    for status, count in status_counts.items():
        rows.append(
            {
                "diagnostic": "feature_status",
                "feature": str(status),
                "value_min": np.nan,
                "value_max": np.nan,
                "value_mean": float(count),
                "missing_rate": np.nan,
                "notes": "Number of retained cells with this bathymetry extraction status.",
            }
        )
    for key, value in metadata.items():
        rows.append(
            {
                "diagnostic": "data_access",
                "feature": key,
                "value_min": np.nan,
                "value_max": np.nan,
                "value_mean": np.nan if isinstance(value, str) else value,
                "missing_rate": np.nan,
                "notes": str(value),
            }
        )
    return pd.DataFrame(rows)


def small_markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table."""
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


def best_by_target_family(results: pd.DataFrame) -> pd.DataFrame:
    """Return best PR-AUC row by target and family."""
    computed = results.loc[results["status"] == "computed"].copy()
    if computed.empty:
        return computed
    return (
        computed.sort_values(["target_definition", "feature_family", "pr_auc"], ascending=[True, True, False])
        .groupby(["target_definition", "feature_family"])
        .head(1)
        .reset_index(drop=True)
    )


def write_report(output: Path, features: pd.DataFrame, diagnostics: pd.DataFrame, results: pd.DataFrame, metadata: dict[str, object]) -> None:
    """Write a bathymetry/habitat feature report."""
    computed = results.loc[results["status"] == "computed"].copy()
    best = best_by_target_family(results)
    top_by_target = computed.sort_values(["target_definition", "pr_auc"], ascending=[True, False]).groupby("target_definition").head(1)
    lines = [
        "# Bathymetry and Habitat Feature Report",
        "",
        "## Purpose",
        "",
        "This report adds static GEBCO-derived bathymetry and habitat-context covariates to the retained 10 km Kelpwatch cells.",
        "These features are intended to reduce over-reliance on canopy persistence and SST-only exposure by representing habitat suitability and exposure context.",
        "",
        "## Data Access",
        "",
        f"- Source: GEBCO 2026 gridded bathymetry/elevation subset via `{GEBCO_DOWNLOAD_BASE}`.",
        f"- GEBCO product page: {GEBCO_PRODUCT_PAGE}",
        "- Raw GEBCO zip and NetCDF files are temporary by default and are not committed.",
        f"- Basket id: `{metadata.get('basket_id', 'local_or_dry_run')}`",
        f"- NetCDF file: `{metadata.get('netcdf_file', metadata.get('local_gebco_netcdf', 'not_available'))}`",
        "",
        "## Feature Definitions",
        "",
        "- GEBCO elevation is in meters relative to mean sea level.",
        "- Ocean pixels are identified as `elevation_m < 0`.",
        "- Positive depth is computed as `depth_m = -elevation_m` for ocean pixels.",
        "- Coastal cells with land are summarized using valid ocean pixels only.",
        "- `ocean_pixel_share` and `bathymetry_missing_rate` are retained for coverage diagnostics.",
        "",
        "## Feature Summary",
        "",
        f"- Retained cells with habitat features: `{int((features['feature_status'] == 'computed').sum())}` / `{len(features)}`",
        f"- Mean ocean pixel share: `{features['ocean_pixel_share'].mean():.3f}`",
        f"- Mean bathymetry missing rate: `{features['bathymetry_missing_rate'].mean():.3f}`",
        f"- Mean depth: `{features['mean_depth_m'].mean():.2f}` m",
        f"- Mean shallow 0-30 m share: `{features['shallow_area_share_0_30m'].mean():.3f}`",
        f"- Mean shallow 0-50 m share: `{features['shallow_area_share_0_50m'].mean():.3f}`",
        "",
        "## Model Comparison",
        "",
        f"- Computed model-comparison rows: `{len(computed)}`",
        "",
        "Best result per target:",
        "",
        small_markdown_table(top_by_target[["target_definition", "feature_family", "model", "pr_auc", "recall", "precision", "f1", "roc_auc"]])
        if not top_by_target.empty
        else "No computed model rows.",
        "",
        "Best row by target and feature family:",
        "",
        small_markdown_table(best[["target_definition", "feature_family", "model", "pr_auc", "recall", "precision", "f1", "roc_auc"]])
        if not best.empty
        else "No computed model rows.",
        "",
        "## Interpretation",
        "",
        "Bathymetry and habitat features are static covariates. They are not direct biological drivers and should not be interpreted as operational early-warning signals by themselves.",
        "Habitat-only performance indicates spatial risk-screening structure, not detection of future ecological transition. Improvement on the original broad decline target does not necessarily imply at-risk, new-transition, or actionable early-warning skill.",
        "No future target information is used in these static features.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def write_dry_run_outputs(args: argparse.Namespace, cells: pd.DataFrame, metadata: dict[str, object]) -> None:
    """Write dry-run diagnostics when bathymetry is unavailable."""
    diagnostics = pd.DataFrame(
        [
            {
                "diagnostic": "data_access",
                "feature": "GEBCO_download_plan",
                "value_min": np.nan,
                "value_max": np.nan,
                "value_mean": np.nan,
                "missing_rate": np.nan,
                "notes": f"Use GEBCO queue API with bbox {metadata['bbox']} or provide --gebco-netcdf.",
            }
        ]
    )
    diagnostics.to_csv(args.feature_diagnostics_output, index=False)
    pd.DataFrame().to_csv(args.model_comparison_output, index=False)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        "# Bathymetry and Habitat Feature Report\n\nDry-run only. GEBCO bathymetry was not downloaded.\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run bathymetry/habitat feature construction and model comparison."""
    args = parse_args()
    for path in [args.feature_output, args.feature_diagnostics_output, args.model_comparison_output, args.report_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    data = load_base_rows(args.input)
    cells, polygons = load_cell_polygons(args.filtered_cells, args.geojson_dir)
    west, east, south, north = padded_bbox(polygons)
    metadata: dict[str, object] = {"bbox": f"west={west:.4f}, east={east:.4f}, south={south:.4f}, north={north:.4f}"}
    temp_context = None
    if args.dry_run:
        write_dry_run_outputs(args, cells, metadata)
        print("Dry run complete; bathymetry features were not built.")
        return

    if args.gebco_netcdf is not None:
        netcdf_path = args.gebco_netcdf
        metadata["local_gebco_netcdf"] = str(netcdf_path)
    else:
        netcdf_path, temp_context, download_metadata = download_gebco_subset(west, east, south, north, args.keep_raw_download)
        metadata.update(download_metadata)

    try:
        features = bathymetry_features_from_netcdf(netcdf_path, polygons)
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    features.to_csv(args.feature_output, index=False)
    diagnostics = diagnostics_table(features, metadata)
    diagnostics.to_csv(args.feature_diagnostics_output, index=False)

    modeling = data.merge(features.drop(columns=["feature_status"]), on="cell_id", how="left", validate="many_to_one")
    modeling = add_crw_if_available(modeling)
    results = evaluate_models(modeling)
    results.to_csv(args.model_comparison_output, index=False)
    write_report(args.report_output, features, diagnostics, results, metadata)

    print(f"Retained cells with habitat features: {(features['feature_status'] == 'computed').sum()} / {len(features)}")
    print(f"Mean ocean pixel share: {features['ocean_pixel_share'].mean():.4f}")
    print(f"Mean bathymetry missing rate: {features['bathymetry_missing_rate'].mean():.4f}")
    print(f"Computed model-comparison rows: {(results['status'] == 'computed').sum()}")
    print(f"Wrote features: {args.feature_output}")
    print(f"Wrote diagnostics: {args.feature_diagnostics_output}")
    print(f"Wrote model comparison: {args.model_comparison_output}")
    print(f"Wrote report: {args.report_output}")


if __name__ == "__main__":
    main()
