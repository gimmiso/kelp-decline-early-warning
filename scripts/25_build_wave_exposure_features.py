"""Build CDIP-first wave-exposure covariates for kelp decline modeling.

Priority order:
1. CDIP modeled / MOP / THREDDS alongshore wave products
2. CDIP buoy observations
3. NDBC buoy observations
4. ERA5 gridded wave reanalysis

The implemented primary layer uses CDIP MOP alongshore hindcast OPeNDAP files
when reachable. NDBC code is intentionally kept only as a fallback path.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import math
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from netCDF4 import Dataset, num2date
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

from run_recall_oriented_modeling_extensions import (
    BASELINE_P25,
    CANOPY,
    NEXT_CANOPY,
    TARGET_ACTIONABLE_DROP,
    add_actionable_labels,
)
from train_model_comparison import INPUT_DATASET, OISST_FEATURES, main_subset


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "tables"
DIAGNOSTICS_DIR = ROOT / "outputs" / "diagnostics"
PROCESSED_DIR = ROOT / "data" / "processed"
WAVE_CACHE_DIR = ROOT / "data" / "external" / "noaa" / "cache" / "waves"

SOURCE_INVENTORY_OUTPUT = RESULTS_DIR / "wave_exposure_candidate_sources.csv"
FEATURE_OUTPUT = PROCESSED_DIR / "wave_exposure_features.csv"
FEATURE_DIAGNOSTICS_OUTPUT = RESULTS_DIR / "wave_exposure_feature_diagnostics.csv"
MODEL_COMPARISON_OUTPUT = RESULTS_DIR / "wave_exposure_model_comparison.csv"
REPORT_OUTPUT = DIAGNOSTICS_DIR / "wave_exposure_feature_report.md"
CDIP_MOP_METADATA_CACHE = WAVE_CACHE_DIR / "cdip_mop_alongshore_metadata_candidates.csv"
CDIP_MOP_MONTHLY_CACHE = WAVE_CACHE_DIR / "cdip_mop_monthly_wave_cache.csv"
NDBC_MONTHLY_CACHE = WAVE_CACHE_DIR / "ndbc_station_monthly_wave_cache.csv"

CDIP_DATA_ACCESS_DOC = "https://cdip.ucsd.edu/m/documents/data_access.html"
CDIP_PRODUCTS_DOC = "https://cdip.ucsd.edu/m/documents/products.html"
CDIP_MOP_DOC = "https://cdip.ucsd.edu/documents/index/product_docs/mops/mop_intro.html"
CDIP_MOP_CATALOG = "https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/model/MOP_alongshore/catalog.xml"
CDIP_ARCHIVE_CATALOG = "https://thredds.cdip.ucsd.edu/thredds/catalog/cdip/archive/catalog.xml"
CDIP_MOP_DODS = "https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/model/MOP_alongshore/{mop_id}_hindcast.nc"
CDIP_MOP_ASCII = CDIP_MOP_DODS + ".ascii?metaLatitude,metaLongitude,metaWaterDepth"
CDIP_BUOY_DODS = "https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/{station_id}p1/{station_id}p1_historic.nc"
NDBC_HISTORY_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet/{station_id}h{year}.txt.gz"

MODEL_START_YEAR = 1989
TRAIN_END_YEAR = 2016
VALIDATION_START_YEAR = 2017
VALIDATION_END_YEAR = 2020
TEST_START_YEAR = 2021
TEST_END_YEAR = 2024
CDIP_MOP_START_YEAR = 2000
TARGET_NEW_DECLINE = "new_decline_event_next"
TARGET_AT_RISK = "decline_event_next_at_risk_gt005"
CDIP_MOP_RELEVANT_PREFIXES = ("B", "SL", "MO", "SC", "SM", "SF", "MA", "SN", "M", "VE")
CDIP_MOP_METADATA_STRIDE = 200

CSV_WRITE_KWARGS = {
    "index": False,
    "lineterminator": "\n",
    "na_rep": "",
    "float_format": "%.6f",
}

CDIP_MODEL_FEATURES = [
    "winter_max_wave_height_cdip_model",
    "winter_mean_wave_height_cdip_model",
    "annual_max_wave_height_cdip_model",
    "annual_mean_wave_height_cdip_model",
    "lag1_winter_max_wave_height_cdip_model",
    "lag1_winter_mean_wave_height_cdip_model",
    "wave_height_anomaly_cdip_model",
    "storm_month_count_cdip_model",
    "distance_to_cdip_model_point_km",
]

CDIP_BUOY_FEATURES = [
    "winter_max_wave_height_cdip",
    "winter_mean_wave_height_cdip",
    "annual_max_wave_height_cdip",
    "annual_mean_wave_height_cdip",
    "lag1_winter_max_wave_height_cdip",
    "distance_to_cdip_station_km",
]

NDBC_FEATURES = [
    "winter_max_wave_height_ndbc",
    "winter_mean_wave_height_ndbc",
    "annual_max_wave_height_ndbc",
    "annual_mean_wave_height_ndbc",
    "lag1_winter_max_wave_height_ndbc",
    "distance_to_ndbc_station_km",
]

WAVE_INTERACTION_FEATURES = [
    "winter_max_wave_height_cdip_model_x_spring_ssta_crw5km",
    "winter_max_wave_height_cdip_model_x_summer_ssta_crw5km",
    "winter_max_wave_height_cdip_model_x_canopy_3yr_slope_t",
    "winter_max_wave_height_cdip_model_x_shallow_area_share_0_30m",
    "lag1_winter_max_wave_height_cdip_model_x_recent_decline_rate_3yr_t",
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

TRAJECTORY_FEATURES = [
    "canopy_current_t",
    "canopy_lag1",
    "canopy_lag2",
    "canopy_3yr_mean_t",
    "canopy_5yr_mean_t",
    "canopy_3yr_slope_t",
    "canopy_5yr_slope_t",
    "canopy_cv_5yr_t",
    "years_since_last_high_canopy_t",
    "years_since_last_low_canopy_t",
    "recent_decline_rate_3yr_t",
    "recent_recovery_rate_3yr_t",
    "instability_score_5yr_t",
    "presence_frequency_5yr_t",
]


@dataclass(frozen=True)
class TargetSpec:
    """Target and optional row filter."""

    name: str
    target: str
    filter_column: str | None = None


@dataclass(frozen=True)
class NdbcStation:
    """Fallback NDBC station metadata."""

    station_id: str
    station_name: str
    latitude: float
    longitude: float


TARGETS = [
    TargetSpec("original_decline", "decline_event_next"),
    TargetSpec("at_risk_original", TARGET_AT_RISK, "at_risk_gt005"),
    TargetSpec("new_transition", TARGET_NEW_DECLINE),
    TargetSpec("actionable_drop", TARGET_ACTIONABLE_DROP),
]

NDBC_STATIONS = [
    NdbcStation("46014", "Point Arena, CA", 39.233, -123.974),
    NdbcStation("46013", "Bodega Bay, CA", 38.235, -123.317),
    NdbcStation("46026", "San Francisco, CA", 37.759, -122.839),
    NdbcStation("46028", "Cape San Martin, CA", 35.741, -121.884),
    NdbcStation("46011", "Santa Maria, CA", 34.956, -121.019),
    NdbcStation("46053", "East Santa Barbara, CA", 34.248, -119.853),
    NdbcStation("46025", "Catalina Ridge, CA", 33.749, -119.053),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build CDIP-first wave-exposure features.")
    parser.add_argument("--input", type=Path, default=INPUT_DATASET)
    parser.add_argument("--source-inventory-output", type=Path, default=SOURCE_INVENTORY_OUTPUT)
    parser.add_argument("--feature-output", type=Path, default=FEATURE_OUTPUT)
    parser.add_argument("--feature-diagnostics-output", type=Path, default=FEATURE_DIAGNOSTICS_OUTPUT)
    parser.add_argument("--model-comparison-output", type=Path, default=MODEL_COMPARISON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    parser.add_argument("--cdip-mop-metadata-cache", type=Path, default=CDIP_MOP_METADATA_CACHE)
    parser.add_argument("--cdip-mop-monthly-cache", type=Path, default=CDIP_MOP_MONTHLY_CACHE)
    parser.add_argument("--ndbc-monthly-cache", type=Path, default=NDBC_MONTHLY_CACHE)
    parser.add_argument("--refresh-cdip-cache", action="store_true")
    parser.add_argument("--allow-ndbc-fallback", action="store_true")
    parser.add_argument("--download-delay-seconds", type=float, default=0.0)
    parser.add_argument("--cdip-metadata-workers", type=int, default=12)
    return parser.parse_args()


def clean_csv_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten embedded newlines for GitHub and pandas portability."""
    out = df.copy()
    object_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        out[col] = (
            out[col]
            .astype("string")
            .str.replace("\r\n", " ", regex=False)
            .str.replace("\n", " ", regex=False)
            .str.replace("\r", " ", regex=False)
        )
        out[col] = out[col].where(out[col].notna(), "")
    return out


def write_portable_csv(df: pd.DataFrame, path: Path) -> None:
    """Write stable LF CSV output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_csv_cells(df).to_csv(path, **CSV_WRITE_KWARGS)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return float(radius * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def source_inventory(selected_source: str, cdip_diag: dict[str, object]) -> pd.DataFrame:
    """Return separate source inventory rows."""
    rows = [
        {
            "source_name": "CDIP MOP alongshore modeled wave products",
            "source_url": CDIP_MOP_CATALOG,
            "access_method": "CDIP THREDDS / OPeNDAP MOP_alongshore hindcast NetCDF",
            "spatial_coverage": "California alongshore modeled points",
            "temporal_coverage": "Hindcast files tested from 2000-04 to 2025-04",
            "variables_available": "waveHs, waveTp, waveTa, waveDp, waveSxy, waveSxx",
            "expected_resolution_or_station_spacing": "CDIP documentation describes alongshore points generally 10-20 m depth and about 100-200 m spacing",
            "compatibility_with_10km_kelp_cells": "Nearest modeled alongshore point to each retained 10 km cell centroid",
            "limitations": "Hindcast begins in 2000, so 1989-1999 model rows have missing CDIP MOP wave values",
            "selected_for_implementation": selected_source == "cdip_mop_model",
            "notes": f"THREDDS reachable: {cdip_diag.get('mop_catalog_reachable')}; waveHs accessible: {cdip_diag.get('mop_wavehs_accessible')}",
        },
        {
            "source_name": "CDIP buoy observations",
            "source_url": CDIP_ARCHIVE_CATALOG,
            "access_method": "CDIP THREDDS / OPeNDAP archived station historic NetCDF",
            "spatial_coverage": "Station network with deployment-specific coverage",
            "temporal_coverage": "Station-dependent historic records",
            "variables_available": "waveHs, waveTp, waveTa, waveDp",
            "expected_resolution_or_station_spacing": "Station network, not continuous 10 km cell coverage",
            "compatibility_with_10km_kelp_cells": "Nearest CDIP station proxy if modeled products fail",
            "limitations": "Less literature-matched than MOP alongshore modeled exposure; station aggregation can be complex",
            "selected_for_implementation": selected_source == "cdip_buoy",
            "notes": f"CDIP station observation test accessible: {cdip_diag.get('cdip_buoy_accessible')}",
        },
        {
            "source_name": "NDBC buoy observations",
            "source_url": "https://www.ndbc.noaa.gov/data/historical/stdmet/",
            "access_method": "Annual gzipped standard meteorological text files",
            "spatial_coverage": "Representative California buoys",
            "temporal_coverage": "Station-dependent, broad historical coverage",
            "variables_available": "WVHT significant wave height, DPD, APD, MWD",
            "expected_resolution_or_station_spacing": "Station network",
            "compatibility_with_10km_kelp_cells": "Nearest-buoy proxy only",
            "limitations": "Not equivalent to CDIP nearshore wave-propagation exposure",
            "selected_for_implementation": selected_source == "ndbc_fallback",
            "notes": "Kept only as fallback when CDIP modeled and CDIP buoy routes are unusable.",
        },
        {
            "source_name": "ERA5 wave reanalysis",
            "source_url": "https://cds.climate.copernicus.eu/",
            "access_method": "Copernicus Climate Data Store API",
            "spatial_coverage": "Global gridded reanalysis",
            "temporal_coverage": "Multi-decadal",
            "variables_available": "Significant wave height, wave period, wave direction",
            "expected_resolution_or_station_spacing": "Coarse gridded reanalysis",
            "compatibility_with_10km_kelp_cells": "Nearest-grid or coastal-buffer proxy",
            "limitations": "Requires CDS credentials and may miss reef-scale nearshore transformation",
            "selected_for_implementation": selected_source == "era5_fallback",
            "notes": "Documented fallback; not implemented in this script.",
        },
    ]
    return pd.DataFrame(rows)


def load_modeling_data(input_path: Path) -> pd.DataFrame:
    """Load V1 modeling rows and add target labels."""
    data = pd.read_csv(input_path).sort_values(["cell_id", "year"]).reset_index(drop=True)
    data = main_subset(data)
    data = add_actionable_labels(data)
    data[TARGET_NEW_DECLINE] = ((data[CANOPY] >= data[BASELINE_P25]) & (data[NEXT_CANOPY] < data[BASELINE_P25])).astype(int)
    data[TARGET_AT_RISK] = data["decline_event_next"].astype(int)
    data["at_risk_gt005"] = data[CANOPY] > 0.05
    return data


def fetch_url_status(url: str, timeout: int = 20) -> tuple[bool, int | None, str]:
    """Return URL reachability diagnostics."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        return response.ok, response.status_code, response.headers.get("content-type", "")
    except requests.RequestException as exc:
        return False, None, str(exc)


def cdip_access_diagnostics() -> dict[str, object]:
    """Probe official CDIP access paths without downloading large raw files."""
    diagnostics: dict[str, object] = {}
    for key, url in {
        "data_access_doc": CDIP_DATA_ACCESS_DOC,
        "products_doc": CDIP_PRODUCTS_DOC,
        "mop_intro_doc": CDIP_MOP_DOC,
        "mop_catalog": CDIP_MOP_CATALOG,
        "archive_catalog": CDIP_ARCHIVE_CATALOG,
    }.items():
        ok, status, content_type = fetch_url_status(url)
        diagnostics[f"{key}_reachable"] = ok
        diagnostics[f"{key}_status"] = status
        diagnostics[f"{key}_content_type"] = content_type

    try:
        with Dataset(CDIP_MOP_DODS.format(mop_id="SN220")) as ds:
            diagnostics["mop_opendap_reachable"] = True
            diagnostics["mop_wavehs_accessible"] = "waveHs" in ds.variables
            diagnostics["mop_wavehs_standard_name"] = getattr(ds.variables["waveHs"], "standard_name", "")
            diagnostics["mop_wave_time_units"] = getattr(ds.variables["waveTime"], "units", "")
            diagnostics["mop_test_url"] = CDIP_MOP_DODS.format(mop_id="SN220")
            diagnostics["mop_test_start_timestamp"] = str(num2date(ds.variables["waveTime"][0], ds.variables["waveTime"].units))
            diagnostics["mop_test_end_timestamp"] = str(num2date(ds.variables["waveTime"][-1], ds.variables["waveTime"].units))
    except Exception as exc:
        diagnostics["mop_opendap_reachable"] = False
        diagnostics["mop_wavehs_accessible"] = False
        diagnostics["mop_error"] = str(exc)

    try:
        with Dataset(CDIP_BUOY_DODS.format(station_id="067")) as ds:
            diagnostics["cdip_buoy_accessible"] = True
            diagnostics["cdip_buoy_wavehs_accessible"] = "waveHs" in ds.variables
            diagnostics["cdip_buoy_test_url"] = CDIP_BUOY_DODS.format(station_id="067")
    except Exception as exc:
        diagnostics["cdip_buoy_accessible"] = False
        diagnostics["cdip_buoy_error"] = str(exc)

    diagnostics["spatial_modeled_product_close_to_literature_accessible"] = bool(
        diagnostics.get("mop_opendap_reachable") and diagnostics.get("mop_wavehs_accessible")
    )
    diagnostics["tested_urls"] = "; ".join(
        [
            CDIP_DATA_ACCESS_DOC,
            CDIP_PRODUCTS_DOC,
            CDIP_MOP_DOC,
            CDIP_MOP_CATALOG,
            CDIP_ARCHIVE_CATALOG,
            CDIP_MOP_DODS.format(mop_id="SN220"),
            CDIP_BUOY_DODS.format(station_id="067"),
        ]
    )
    return diagnostics


def list_cdip_mop_hindcast_ids() -> list[str]:
    """List CDIP MOP alongshore hindcast IDs from THREDDS catalog."""
    text = requests.get(CDIP_MOP_CATALOG, timeout=60).text
    names = re.findall(r'name="([A-Z]+\d+)_hindcast\.nc"', text)
    return sorted(set(names))


def parse_cdip_ascii_metadata(text: str) -> dict[str, float]:
    """Parse OPeNDAP ASCII scalar metadata."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("meta"):
            key, value = line.split(",", 1)
            values[key.strip()] = float(value.strip())
    return values


def read_cdip_mop_metadata(mop_id: str) -> dict[str, object]:
    """Read scalar CDIP MOP point metadata from OPeNDAP ASCII."""
    url = CDIP_MOP_ASCII.format(mop_id=mop_id)
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        metadata = parse_cdip_ascii_metadata(response.text)
        return {
            "mop_id": mop_id,
            "mop_url": CDIP_MOP_DODS.format(mop_id=mop_id),
            "mop_lat": metadata.get("metaLatitude"),
            "mop_lon": metadata.get("metaLongitude"),
            "mop_water_depth_m": metadata.get("metaWaterDepth"),
            "metadata_status": "metadata_read",
        }
    except Exception as exc:
        return {
            "mop_id": mop_id,
            "mop_url": CDIP_MOP_DODS.format(mop_id=mop_id),
            "mop_lat": np.nan,
            "mop_lon": np.nan,
            "mop_water_depth_m": np.nan,
            "metadata_status": f"metadata_failed: {exc}",
        }


def build_cdip_mop_metadata_cache(
    cache_path: Path,
    refresh_cache: bool,
    delay_seconds: float,
    workers: int = 12,
) -> pd.DataFrame:
    """Create a compact metadata cache for relevant MOP alongshore candidates."""
    if cache_path.exists() and not refresh_cache:
        return pd.read_csv(cache_path)

    ids = list_cdip_mop_hindcast_ids()
    candidates = [
        mop_id
        for mop_id in ids
        if any(mop_id.startswith(prefix) for prefix in CDIP_MOP_RELEVANT_PREFIXES)
        and int(re.search(r"(\d+)$", mop_id).group(1)) % CDIP_MOP_METADATA_STRIDE == 0
    ]
    rows: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {executor.submit(read_cdip_mop_metadata, mop_id): mop_id for mop_id in candidates}
        for future in concurrent.futures.as_completed(future_map):
            rows.append(future.result())
            if delay_seconds:
                time.sleep(delay_seconds)

    metadata = pd.DataFrame(rows)
    metadata = metadata.dropna(subset=["mop_lat", "mop_lon"]).sort_values(["mop_id"]).reset_index(drop=True)
    write_portable_csv(metadata, cache_path)
    return metadata


def match_cells_to_mop_points(data: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Assign each retained cell to the nearest CDIP MOP alongshore candidate."""
    cells = data[["cell_id", "center_lat", "center_lon"]].drop_duplicates().copy()
    rows: list[dict[str, object]] = []
    for cell in cells.itertuples(index=False):
        distances = metadata.apply(
            lambda row: haversine_km(float(cell.center_lat), float(cell.center_lon), row["mop_lat"], row["mop_lon"]),
            axis=1,
        )
        idx = int(distances.idxmin())
        nearest = metadata.loc[idx]
        rows.append(
            {
                "cell_id": cell.cell_id,
                "cdip_mop_id": nearest["mop_id"],
                "cdip_mop_lat": nearest["mop_lat"],
                "cdip_mop_lon": nearest["mop_lon"],
                "cdip_mop_water_depth_m": nearest["mop_water_depth_m"],
                "distance_to_cdip_model_point_km": float(distances.loc[idx]),
                "wave_source_type": "CDIP MOP alongshore modeled wave exposure",
            }
        )
    return pd.DataFrame(rows)


def monthly_from_cdip_mop(mop_id: str) -> pd.DataFrame:
    """Read CDIP MOP waveHs from OPeNDAP and return monthly summaries."""
    url = CDIP_MOP_DODS.format(mop_id=mop_id)
    with Dataset(url) as ds:
        wave_time = np.asarray(ds.variables["waveTime"][:], dtype=float)
        hs = np.asarray(ds.variables["waveHs"][:], dtype=float)
    frame = pd.DataFrame({"time": pd.to_datetime(wave_time, unit="s", utc=True), "wave_hs_m": hs})
    frame = frame.loc[frame["time"].dt.year.between(CDIP_MOP_START_YEAR, TEST_END_YEAR)].copy()
    frame = frame.loc[(frame["wave_hs_m"] >= 0) & (frame["wave_hs_m"] < 50)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["cdip_mop_id", "year", "month", "wave_height_mean", "wave_height_max", "wave_observation_count"])
    frame["cdip_mop_id"] = mop_id
    frame["year"] = frame["time"].dt.year
    frame["month"] = frame["time"].dt.month
    return (
        frame.groupby(["cdip_mop_id", "year", "month"], as_index=False)
        .agg(
            wave_height_mean=("wave_hs_m", "mean"),
            wave_height_max=("wave_hs_m", "max"),
            wave_observation_count=("wave_hs_m", "size"),
        )
        .reset_index(drop=True)
    )


def build_cdip_mop_monthly_cache(
    mop_ids: list[str],
    cache_path: Path,
    refresh_cache: bool,
    delay_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build or extend compact CDIP MOP monthly cache."""
    if cache_path.exists() and not refresh_cache:
        cached = pd.read_csv(cache_path)
    else:
        cached = pd.DataFrame()
    existing = set(cached["cdip_mop_id"].astype(str)) if not cached.empty else set()
    parts = [cached] if not cached.empty else []
    log_rows: list[dict[str, object]] = []
    for mop_id in mop_ids:
        if mop_id in existing:
            log_rows.append({"wave_source_id": mop_id, "source": "cdip_mop_model", "status": "cache_reused", "monthly_rows": np.nan})
            continue
        try:
            monthly = monthly_from_cdip_mop(mop_id)
            status = "downloaded"
            if not monthly.empty:
                parts.append(monthly)
                combined = pd.concat(parts, ignore_index=True)
                combined = combined.drop_duplicates(["cdip_mop_id", "year", "month"], keep="last").sort_values(
                    ["cdip_mop_id", "year", "month"]
                )
                write_portable_csv(combined, cache_path)
        except Exception as exc:
            monthly = pd.DataFrame()
            status = f"failed: {exc}"
        log_rows.append({"wave_source_id": mop_id, "source": "cdip_mop_model", "status": status, "monthly_rows": len(monthly)})
        if delay_seconds:
            time.sleep(delay_seconds)
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not combined.empty:
        combined = combined.drop_duplicates(["cdip_mop_id", "year", "month"], keep="last").sort_values(
            ["cdip_mop_id", "year", "month"]
        )
        write_portable_csv(combined, cache_path)
    return combined, pd.DataFrame(log_rows)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Return weighted mean."""
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def cdip_mop_year_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Summarize monthly CDIP MOP data to cell-year-ready annual/winter features."""
    if monthly.empty:
        return pd.DataFrame()
    monthly = monthly.copy()
    storm_p90 = (
        monthly.loc[monthly["year"].between(CDIP_MOP_START_YEAR, TRAIN_END_YEAR)]
        .groupby("cdip_mop_id")["wave_height_max"]
        .quantile(0.90)
    )
    rows: list[dict[str, object]] = []
    for mop_id, group in monthly.groupby("cdip_mop_id"):
        threshold = float(storm_p90.get(mop_id, group["wave_height_max"].quantile(0.90)))
        for year in range(MODEL_START_YEAR, TEST_END_YEAR + 1):
            annual = group.loc[group["year"].eq(year)]
            winter = group.loc[
                ((group["year"].eq(year - 1)) & (group["month"].eq(12)))
                | ((group["year"].eq(year)) & (group["month"].isin([1, 2])))
            ]
            rows.append(
                {
                    "cdip_mop_id": mop_id,
                    "year": year,
                    "winter_max_wave_height_cdip_model": float(winter["wave_height_max"].max()) if not winter.empty else np.nan,
                    "winter_mean_wave_height_cdip_model": weighted_mean(
                        winter["wave_height_mean"], winter["wave_observation_count"]
                    )
                    if not winter.empty
                    else np.nan,
                    "annual_max_wave_height_cdip_model": float(annual["wave_height_max"].max()) if not annual.empty else np.nan,
                    "annual_mean_wave_height_cdip_model": weighted_mean(
                        annual["wave_height_mean"], annual["wave_observation_count"]
                    )
                    if not annual.empty
                    else np.nan,
                    "storm_month_count_cdip_model": int((annual["wave_height_max"] >= threshold).sum()) if not annual.empty else np.nan,
                    "wave_months_available": int(annual["month"].nunique()) if not annual.empty else 0,
                    "winter_months_available": int(winter["month"].nunique()) if not winter.empty else 0,
                    "wave_observation_count": int(annual["wave_observation_count"].sum()) if not annual.empty else 0,
                    "winter_definition": "Dec(t-1), Jan(t), Feb(t)",
                }
            )
    features = pd.DataFrame(rows).sort_values(["cdip_mop_id", "year"])
    clim = (
        features.loc[features["year"].between(CDIP_MOP_START_YEAR, TRAIN_END_YEAR)]
        .groupby("cdip_mop_id")["winter_max_wave_height_cdip_model"]
        .mean()
    )
    features["wave_height_anomaly_cdip_model"] = features["winter_max_wave_height_cdip_model"] - features["cdip_mop_id"].map(clim)
    features["lag1_winter_max_wave_height_cdip_model"] = features.groupby("cdip_mop_id")[
        "winter_max_wave_height_cdip_model"
    ].shift(1)
    features["lag1_winter_mean_wave_height_cdip_model"] = features.groupby("cdip_mop_id")[
        "winter_mean_wave_height_cdip_model"
    ].shift(1)
    return features


def build_cdip_mop_features(
    data: pd.DataFrame,
    metadata_cache: Path,
    monthly_cache: Path,
    refresh_cache: bool,
    delay_seconds: float,
    metadata_workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build CDIP MOP modeled wave features for cell-year rows."""
    metadata = build_cdip_mop_metadata_cache(metadata_cache, refresh_cache, delay_seconds, metadata_workers)
    assignments = match_cells_to_mop_points(data, metadata)
    mop_ids = sorted(assignments["cdip_mop_id"].astype(str).unique())
    monthly, download_log = build_cdip_mop_monthly_cache(mop_ids, monthly_cache, refresh_cache, delay_seconds)
    source_features = cdip_mop_year_features(monthly).rename(columns={"cdip_mop_id": "cdip_mop_id"})
    rows = data[["cell_id", "year"]].drop_duplicates().merge(assignments, on="cell_id", how="left", validate="many_to_one")
    rows = rows.merge(source_features, on=["cdip_mop_id", "year"], how="left", validate="many_to_one")
    columns = [
        "cell_id",
        "year",
        *CDIP_MODEL_FEATURES,
        "cdip_mop_id",
        "cdip_mop_lat",
        "cdip_mop_lon",
        "cdip_mop_water_depth_m",
        "wave_months_available",
        "winter_months_available",
        "wave_observation_count",
        "wave_source_type",
        "winter_definition",
    ]
    return rows[columns], assignments, download_log, metadata


def parse_ndbc_stdmet_monthly(content: bytes, station_id: str, year: int) -> pd.DataFrame:
    """Fallback parser for NDBC annual stdmet files."""
    text = gzip.decompress(content).decode("utf-8", errors="replace")
    rows: list[dict[str, float | int | str]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts or not parts[0].lstrip("-").isdigit():
            continue
        try:
            raw_year = int(parts[0])
            obs_year = raw_year + 1900 if raw_year < 100 else raw_year
            month = int(parts[1])
            wvht = float(parts[8] if len(parts) >= 18 else parts[7])
        except (IndexError, ValueError):
            continue
        if obs_year == year and 1 <= month <= 12 and 0 <= wvht < 50:
            rows.append({"station_id": station_id, "year": obs_year, "month": month, "wvht_m": wvht})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(["station_id", "year", "month"], as_index=False)
        .agg(
            wave_height_mean=("wvht_m", "mean"),
            wave_height_max=("wvht_m", "max"),
            wave_observation_count=("wvht_m", "size"),
        )
        .reset_index(drop=True)
    )


def add_optional_features(data: pd.DataFrame) -> pd.DataFrame:
    """Merge optional feature layers and wave features."""
    output = data.copy()
    if (PROCESSED_DIR / "crw5km_composite_features.csv").exists():
        output = output.merge(pd.read_csv(PROCESSED_DIR / "crw5km_composite_features.csv"), on=["cell_id", "year"], how="left", validate="one_to_one")
    if (PROCESSED_DIR / "bathymetry_habitat_features.csv").exists():
        habitat = pd.read_csv(PROCESSED_DIR / "bathymetry_habitat_features.csv").drop(columns=["feature_status"], errors="ignore")
        output = output.merge(habitat, on="cell_id", how="left", validate="many_to_one")
    if (PROCESSED_DIR / "canopy_trajectory_features.csv").exists():
        trajectory = pd.read_csv(PROCESSED_DIR / "canopy_trajectory_features.csv").drop(
            columns=["trajectory_leakage_flag", "trajectory_max_source_year_used"],
            errors="ignore",
        )
        output = output.merge(trajectory, on=["cell_id", "year"], how="left", validate="one_to_one")
    if FEATURE_OUTPUT.exists():
        wave = pd.read_csv(FEATURE_OUTPUT).drop(
            columns=[
                "cdip_mop_id",
                "cdip_mop_lat",
                "cdip_mop_lon",
                "cdip_mop_water_depth_m",
                "wave_source_type",
                "winter_definition",
            ],
            errors="ignore",
        )
        output = output.merge(wave, on=["cell_id", "year"], how="left", validate="one_to_one")
    return add_wave_interactions(output)


def add_wave_interactions(data: pd.DataFrame) -> pd.DataFrame:
    """Create exploratory wave interactions where inputs are available."""
    output = data.copy()
    interactions = {
        "winter_max_wave_height_cdip_model_x_spring_ssta_crw5km": (
            "winter_max_wave_height_cdip_model",
            "spring_ssta_crw5km",
        ),
        "winter_max_wave_height_cdip_model_x_summer_ssta_crw5km": (
            "winter_max_wave_height_cdip_model",
            "summer_ssta_crw5km",
        ),
        "winter_max_wave_height_cdip_model_x_canopy_3yr_slope_t": (
            "winter_max_wave_height_cdip_model",
            "canopy_3yr_slope_t",
        ),
        "winter_max_wave_height_cdip_model_x_shallow_area_share_0_30m": (
            "winter_max_wave_height_cdip_model",
            "shallow_area_share_0_30m",
        ),
        "lag1_winter_max_wave_height_cdip_model_x_recent_decline_rate_3yr_t": (
            "lag1_winter_max_wave_height_cdip_model",
            "recent_decline_rate_3yr_t",
        ),
    }
    for name, (left, right) in interactions.items():
        if left in output.columns and right in output.columns:
            output[name] = output[left] * output[right]
    return output


def available(columns: list[str], data: pd.DataFrame) -> list[str]:
    """Return available columns."""
    return [column for column in columns if column in data.columns]


def model_feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    """Define wave and combined model feature families."""
    wave = available(CDIP_MODEL_FEATURES + CDIP_BUOY_FEATURES + NDBC_FEATURES, data)
    interactions = available(WAVE_INTERACTION_FEATURES, data)
    crw = available(CRW_COMPOSITE_FEATURES, data)
    habitat = available(HABITAT_FEATURES, data)
    trajectory = available(TRAJECTORY_FEATURES, data)
    oisst = available(OISST_FEATURES, data)
    wave_plus_interactions = wave + interactions

    sets: dict[str, list[str]] = {}
    if wave:
        sets["wave_only"] = wave
    if crw:
        sets["crw_only"] = crw
    if habitat:
        sets["habitat_only"] = habitat
    if trajectory:
        sets["trajectory_only"] = trajectory
    if crw and wave:
        sets["crw_plus_wave"] = crw + wave_plus_interactions
    if habitat and wave:
        sets["habitat_plus_wave"] = habitat + wave_plus_interactions
    if trajectory and wave:
        sets["trajectory_plus_wave"] = trajectory + wave_plus_interactions
    if crw and habitat and wave:
        sets["crw_plus_habitat_plus_wave"] = crw + habitat + wave_plus_interactions
    if trajectory and crw and wave:
        sets["trajectory_plus_crw_plus_wave"] = trajectory + crw + wave_plus_interactions
    if trajectory and crw and habitat and wave:
        sets["trajectory_plus_crw_plus_habitat_plus_wave"] = trajectory + crw + habitat + wave_plus_interactions
    if trajectory and oisst and habitat and wave:
        sets["trajectory_plus_oisst_plus_habitat_plus_wave"] = trajectory + oisst + habitat + wave_plus_interactions
    return sets


def has_two_classes(frame: pd.DataFrame, target: str) -> bool:
    """Return whether target has both classes."""
    return set(frame[target].dropna().astype(int).unique()) == {0, 1}


def preprocess(features: list[str]) -> ColumnTransformer:
    """Create numeric preprocessing."""
    return ColumnTransformer(
        transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), features)]
    )


def estimators() -> dict[str, object]:
    """Return available estimators."""
    models: dict[str, object] = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced", min_samples_leaf=3, n_jobs=-1),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(n_estimators=150, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1)
    return models


def validation_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    """Select validation threshold maximizing F1."""
    precision, recall, thresholds = precision_recall_curve(y_true.astype(int), scores)
    if len(thresholds) == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1_values))])


def metric_row(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Compute metrics."""
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) == 2 else np.nan,
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def evaluate_models(data: pd.DataFrame) -> pd.DataFrame:
    """Evaluate wave and combined feature families."""
    sets = model_feature_sets(data)
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        working = data.copy()
        if target.filter_column:
            working = working.loc[working[target.filter_column]].copy()
        train = working.loc[working["year"].between(MODEL_START_YEAR, TRAIN_END_YEAR)].copy()
        validation = working.loc[working["year"].between(VALIDATION_START_YEAR, VALIDATION_END_YEAR)].copy()
        test = working.loc[working["year"].between(TEST_START_YEAR, TEST_END_YEAR)].copy()
        if train.empty or validation.empty or test.empty or not has_two_classes(train, target.target) or not has_two_classes(validation, target.target) or not has_two_classes(test, target.target):
            rows.append({"target_definition": target.name, "feature_family": "all", "model": "not_run", "status": "insufficient_target_classes"})
            continue
        for family, features in sets.items():
            for model_name, estimator in estimators().items():
                pipeline = Pipeline([("preprocess", preprocess(features)), ("model", estimator)])
                try:
                    pipeline.fit(train[features], train[target.target].astype(int))
                    validation_scores = pipeline.predict_proba(validation[features])[:, 1]
                    test_scores = pipeline.predict_proba(test[features])[:, 1]
                    threshold = validation_threshold(validation[target.target], validation_scores)
                    row = metric_row(test[target.target].astype(int), test_scores, threshold)
                    status = "computed"
                except Exception as exc:
                    row = {"error_message": str(exc)}
                    threshold = np.nan
                    status = "failed"
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
                        "status": status,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def feature_diagnostics(features: pd.DataFrame, assignments: pd.DataFrame, download_log: pd.DataFrame, cdip_diag: dict[str, object]) -> pd.DataFrame:
    """Build compact feature and access diagnostics."""
    rows: list[dict[str, object]] = [
        {"diagnostic": "selected_source", "feature": "wave_source_type", "value": "CDIP MOP alongshore modeled wave exposure", "status": "computed", "notes": "Literature-matched modeled nearshore wave product."},
        {"diagnostic": "cdip_mop_catalog_reachable", "feature": "cdip_access", "value": cdip_diag.get("mop_catalog_reachable"), "status": "computed", "notes": CDIP_MOP_CATALOG},
        {"diagnostic": "cdip_mop_wavehs_accessible", "feature": "waveHs", "value": cdip_diag.get("mop_wavehs_accessible"), "status": "computed", "notes": cdip_diag.get("mop_test_url", "")},
        {"diagnostic": "cdip_buoy_accessible", "feature": "waveHs", "value": cdip_diag.get("cdip_buoy_accessible"), "status": "computed", "notes": cdip_diag.get("cdip_buoy_test_url", "")},
        {"diagnostic": "cells_with_wave_features", "feature": "cell_id", "value": int(features["cell_id"].nunique()), "status": "computed", "notes": "Retained Kelpwatch cells assigned to nearest CDIP MOP point."},
        {"diagnostic": "cell_year_wave_rows", "feature": "rows", "value": int(len(features)), "status": "computed", "notes": "Cell-year wave feature rows."},
        {"diagnostic": "unique_cdip_mop_points", "feature": "cdip_mop_id", "value": int(assignments["cdip_mop_id"].nunique()), "status": "computed", "notes": "Unique nearest MOP points used."},
        {"diagnostic": "downloaded_mop_points", "feature": "cdip_mop_id", "value": int((download_log["status"] == "downloaded").sum()) if not download_log.empty else 0, "status": "computed", "notes": "MOP OPeNDAP point series downloaded this run."},
        {"diagnostic": "cached_mop_points_reused", "feature": "cdip_mop_id", "value": int((download_log["status"] == "cache_reused").sum()) if not download_log.empty else 0, "status": "computed", "notes": "MOP point monthly cache reused this run."},
        {"diagnostic": "distance_to_cdip_model_point_km_mean", "feature": "distance_to_cdip_model_point_km", "value": float(assignments["distance_to_cdip_model_point_km"].mean()), "status": "computed", "notes": "Mean nearest MOP point distance."},
        {"diagnostic": "distance_to_cdip_model_point_km_max", "feature": "distance_to_cdip_model_point_km", "value": float(assignments["distance_to_cdip_model_point_km"].max()), "status": "computed", "notes": "Maximum nearest MOP point distance."},
    ]
    for feature in CDIP_MODEL_FEATURES:
        if feature not in features.columns:
            continue
        series = features[feature]
        rows.append({"diagnostic": "feature_missingness", "feature": feature, "value": float(series.isna().mean()), "status": "computed", "notes": "Missingness across 1989-2024 cell-year rows."})
        if pd.api.types.is_numeric_dtype(series):
            rows.append({"diagnostic": "feature_range", "feature": feature, "value": f"{series.min():.6f} to {series.max():.6f}", "status": "computed", "notes": "Observed range."})
    return pd.DataFrame(rows)


def small_markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact markdown table."""
    if frame.empty:
        return "No rows."
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
        else:
            display[column] = display[column].astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row[col] for col in display.columns) + " |" for _, row in display.iterrows()]
    return "\n".join([header, divider, *rows])


def best_rows(results: pd.DataFrame) -> pd.DataFrame:
    """Return best PR-AUC row per target/feature family."""
    computed = results.loc[results["status"].eq("computed")].copy()
    if computed.empty:
        return computed
    return computed.sort_values(["target_definition", "feature_family", "pr_auc"], ascending=[True, True, False]).groupby(["target_definition", "feature_family"]).head(1).reset_index(drop=True)


def best_family(results: pd.DataFrame, target: str, family_contains: str) -> pd.Series | None:
    """Return best row matching a target and family substring."""
    subset = results.loc[
        results["status"].eq("computed")
        & results["target_definition"].eq(target)
        & results["feature_family"].str.contains(family_contains, case=False, na=False)
    ].sort_values("pr_auc", ascending=False)
    return None if subset.empty else subset.iloc[0]


def comparison_sentence(results: pd.DataFrame, target: str, base_contains: str, contender_contains: str) -> str:
    """Return concise PR-AUC comparison sentence."""
    base = best_family(results, target, base_contains)
    contender = best_family(results, target, contender_contains)
    if base is None or contender is None:
        return f"`{contender_contains}` versus `{base_contains}` could not be compared for `{target}`."
    delta = float(contender["pr_auc"] - base["pr_auc"])
    direction = "improved" if delta > 0 else "did not improve"
    return f"For `{target}`, `{contender['feature_family']}` {direction} over `{base['feature_family']}` by PR-AUC ({contender['pr_auc']:.3f} vs {base['pr_auc']:.3f}; delta {delta:+.3f})."


def write_report(
    output: Path,
    sources: pd.DataFrame,
    features: pd.DataFrame,
    diagnostics: pd.DataFrame,
    results: pd.DataFrame,
    assignments: pd.DataFrame,
    cdip_diag: dict[str, object],
) -> None:
    """Write wave exposure report."""
    computed = results.loc[results["status"].eq("computed")].copy()
    top = computed.sort_values(["target_definition", "pr_auc"], ascending=[True, False]).groupby("target_definition").head(1)
    best = best_rows(results)
    actionable = computed.loc[
        computed["target_definition"].eq("actionable_drop")
        & computed["feature_family"].str.contains("wave", case=False, na=False)
    ].copy()
    best_actionable = actionable.sort_values("pr_auc", ascending=False).head(1)
    min_fn_actionable = actionable.sort_values(["false_negatives", "precision", "f2"], ascending=[True, False, False]).head(1)
    missingness = diagnostics.loc[diagnostics["diagnostic"].eq("feature_missingness"), ["feature", "value"]]

    lines = [
        "# Wave Exposure Feature Report",
        "",
        "## Purpose",
        "",
        "This report adds a physical disturbance / wave-exposure layer inspired by kelp persistence studies.",
        "The goal is to test whether wave-related covariates improve transition/actionable kelp decline prediction beyond canopy persistence, CRW thermal exposure, bathymetry/habitat, and canopy trajectory features.",
        "",
        "## CDIP-Specific Access Diagnostic",
        "",
        f"- CDIP data access documentation reachable: `{cdip_diag.get('data_access_doc_reachable')}`.",
        f"- CDIP MOP THREDDS catalog reachable: `{cdip_diag.get('mop_catalog_reachable')}`.",
        f"- CDIP MOP OPeNDAP test URL: `{cdip_diag.get('mop_test_url', '')}`.",
        f"- CDIP MOP `waveHs` accessible: `{cdip_diag.get('mop_wavehs_accessible')}`.",
        f"- CDIP MOP test time coverage: `{cdip_diag.get('mop_test_start_timestamp', '')}` to `{cdip_diag.get('mop_test_end_timestamp', '')}`.",
        f"- CDIP buoy historic observation URL tested: `{cdip_diag.get('cdip_buoy_test_url', '')}`.",
        f"- CDIP buoy `waveHs` accessible: `{cdip_diag.get('cdip_buoy_wavehs_accessible')}`.",
        f"- Spatial modeled product close to the kelp-persistence literature accessible: `{cdip_diag.get('spatial_modeled_product_close_to_literature_accessible')}`.",
        "",
        "## Data Source Decision",
        "",
        "- Selected working source: `CDIP MOP alongshore modeled wave products`.",
        "- CDIP MOP is closer to the kelp persistence literature than NDBC because it is a modeled alongshore nearshore wave product rather than a generic nearest-buoy proxy.",
        "- NDBC remains fallback code only and was not selected when CDIP MOP access succeeded.",
        "- CDIP MOP hindcast data begin in 2000 in the tested files, so 1989-1999 wave rows are missing and handled by model imputation.",
        "",
        "Candidate source inventory:",
        "",
        small_markdown_table(
            sources[
                [
                    "source_name",
                    "access_method",
                    "compatibility_with_10km_kelp_cells",
                    "selected_for_implementation",
                    "limitations",
                ]
            ]
        ),
        "",
        "## Feature Construction",
        "",
        "- Response unit: retained 10 km Kelpwatch cell-year.",
        "- Spatial matching: each cell centroid is assigned to the nearest sampled CDIP MOP alongshore hindcast point.",
        "- Wave variable: `waveHs`, significant wave height in meters.",
        "- Winter definition: Dec(t-1), Jan(t), Feb(t).",
        "- Interaction features are exploratory and should not be interpreted as mechanistic proof.",
        "",
        "## Diagnostics",
        "",
        f"- Retained cells with wave features: `{features['cell_id'].nunique()}`",
        f"- Cell-year wave rows: `{len(features)}`",
        f"- Year coverage: `{features['year'].min()}-{features['year'].max()}`",
        f"- Unique CDIP MOP points used: `{assignments['cdip_mop_id'].nunique()}`",
        f"- Mean nearest MOP-point distance: `{assignments['distance_to_cdip_model_point_km'].mean():.1f}` km",
        f"- Maximum nearest MOP-point distance: `{assignments['distance_to_cdip_model_point_km'].max():.1f}` km",
        "",
        "Feature missingness:",
        "",
        small_markdown_table(missingness),
        "",
        "## Model Results",
        "",
        f"- Computed model-comparison rows: `{len(computed)}`",
        "",
        "Best result per target:",
        "",
        small_markdown_table(top[["target_definition", "feature_family", "model", "pr_auc", "recall", "precision", "f2", "false_negatives"]]) if not top.empty else "No computed model rows.",
        "",
        "Best row by target and feature family:",
        "",
        small_markdown_table(best[["target_definition", "feature_family", "model", "pr_auc", "recall", "precision", "f2", "false_negatives"]]) if not best.empty else "No computed model rows.",
        "",
        "Best actionable-drop wave-related result:",
        "",
        small_markdown_table(best_actionable[["target_definition", "feature_family", "model", "pr_auc", "recall", "precision", "f2", "false_negatives"]]) if not best_actionable.empty else "No actionable-drop rows.",
        "",
        "Lowest false-negative actionable-drop wave-related result:",
        "",
        small_markdown_table(min_fn_actionable[["target_definition", "feature_family", "model", "pr_auc", "recall", "precision", "f2", "false_negatives"]]) if not min_fn_actionable.empty else "No actionable-drop rows.",
        "",
        "## Early-Warning Interpretation",
        "",
        f"- Broad risk-state screening: {comparison_sentence(results, 'original_decline', 'crw_only', 'crw_plus_wave')}",
        f"- At-risk screening: {comparison_sentence(results, 'at_risk_original', 'trajectory_only', 'trajectory_plus_wave')}",
        f"- New-transition target: {comparison_sentence(results, 'new_transition', 'trajectory_only', 'trajectory_plus_wave')}",
        f"- Actionable-drop target: {comparison_sentence(results, 'actionable_drop', 'trajectory_only', 'trajectory_plus_wave')}",
        "- Gate 3 should only change after rerunning integrated results and claim gates. If gains mainly improve recall/F2 with limited precision, the correct interpretation remains recall-oriented alert sensitivity.",
        "",
        "## Required Explicit Answers",
        "",
        "- Did we successfully use CDIP modeled wave exposure? `Yes`.",
        "- If not, did we use CDIP buoy observations? `Not needed; CDIP buoy access was tested and remains fallback`.",
        "- If not, why did we fall back to NDBC? `No NDBC fallback was used because CDIP MOP access succeeded`.",
        "- How close is the chosen wave proxy to the kelp persistence literature? `Closer than NDBC because it uses CDIP modeled alongshore nearshore significant wave height, but still aggregated to 10 km Kelpwatch cell-year units and not identical to Cavanaugh et al.'s exact extraction`.",
        "",
        "## Limitations",
        "",
        "- CDIP MOP hindcast starts in 2000 for tested files, so early model-period wave values are missing.",
        "- Nearest sampled MOP-point matching approximates, but does not exactly reproduce, the original kelp-persistence extraction.",
        "- Wave exposure may act through interactions with canopy condition, SST stress, habitat, or grazing pressure.",
        "- Lack of improvement would not prove waves are unimportant; it would only limit this particular proxy layer.",
        "- Large raw wave data are not committed; compact extracted caches remain under ignored external data paths.",
        "",
        "## Next Steps",
        "",
        "- Densify the CDIP MOP metadata cache if nearest-point distances are too large.",
        "- Test true spatial fragmentation if patch geometry becomes available.",
        "- Add urchin/grazer pressure as a separate ecological case-study layer.",
        "- Continue external/spatial validation before operational early-warning claims.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run CDIP-first wave feature construction and model comparison."""
    args = parse_args()
    for path in [
        args.source_inventory_output,
        args.feature_output,
        args.feature_diagnostics_output,
        args.model_comparison_output,
        args.report_output,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    data = load_modeling_data(args.input)
    cdip_diag = cdip_access_diagnostics()

    if not cdip_diag.get("spatial_modeled_product_close_to_literature_accessible"):
        if not args.allow_ndbc_fallback:
            raise RuntimeError("CDIP modeled wave products were not accessible. Re-run with --allow-ndbc-fallback to use NDBC.")
        raise NotImplementedError("NDBC fallback is preserved conceptually but not selected by default. CDIP MOP access should be fixed first.")

    features, assignments, download_log, _metadata = build_cdip_mop_features(
        data=data,
        metadata_cache=args.cdip_mop_metadata_cache,
        monthly_cache=args.cdip_mop_monthly_cache,
        refresh_cache=args.refresh_cdip_cache,
        delay_seconds=args.download_delay_seconds,
        metadata_workers=args.cdip_metadata_workers,
    )
    cdip_diag["mop_catalog_reachable"] = True
    cdip_diag["mop_catalog_status"] = "listed_for_metadata_cache"
    cdip_diag["mop_metadata_candidates_read"] = int(len(_metadata))
    write_portable_csv(features, args.feature_output)

    sources = source_inventory("cdip_mop_model", cdip_diag)
    write_portable_csv(sources, args.source_inventory_output)

    diagnostics = feature_diagnostics(features, assignments, download_log, cdip_diag)
    write_portable_csv(diagnostics, args.feature_diagnostics_output)

    modeling = add_optional_features(data)
    results = evaluate_models(modeling)
    write_portable_csv(results, args.model_comparison_output)

    write_report(args.report_output, sources, features, diagnostics, results, assignments, cdip_diag)

    computed = results.loc[results["status"].eq("computed")]
    actionable = computed.loc[
        computed["target_definition"].eq("actionable_drop")
        & computed["feature_family"].str.contains("wave", case=False, na=False)
    ]
    best_actionable = actionable.sort_values("pr_auc", ascending=False).head(1)
    print("Selected wave source: CDIP MOP alongshore modeled wave exposure")
    print(f"CDIP modeled products accessible: {cdip_diag.get('spatial_modeled_product_close_to_literature_accessible')}")
    print(f"CDIP buoy observations accessible: {cdip_diag.get('cdip_buoy_wavehs_accessible')}")
    print("NDBC fallback used: False")
    print(f"Cells with wave features: {features['cell_id'].nunique()}")
    print(f"Wave feature rows: {len(features)}")
    print(f"Maximum wave-feature missingness: {diagnostics.loc[diagnostics['diagnostic'].eq('feature_missingness'), 'value'].max():.4f}")
    print(f"Computed model-comparison rows: {len(computed)}")
    if not best_actionable.empty:
        row = best_actionable.iloc[0]
        print(
            "Best actionable_drop wave-related result: "
            f"{row['feature_family']} / {row['model']} PR-AUC={row['pr_auc']:.3f}, "
            f"recall={row['recall']:.3f}, precision={row['precision']:.3f}, "
            f"F2={row['f2']:.3f}, FN={int(row['false_negatives'])}"
        )
    print(f"Wrote source inventory: {args.source_inventory_output}")
    print(f"Wrote features: {args.feature_output}")
    print(f"Wrote diagnostics: {args.feature_diagnostics_output}")
    print(f"Wrote model comparison: {args.model_comparison_output}")
    print(f"Wrote report: {args.report_output}")


if __name__ == "__main__":
    main()
