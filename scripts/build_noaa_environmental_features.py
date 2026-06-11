"""Build cached NOAA-only environmental features for Kelpwatch modeling rows."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from netCDF4 import Dataset, num2date


EXPECTED_CELLS = 50
EXPECTED_MODELING_ROWS = 2050
BASELINE_START_YEAR = 1984
BASELINE_END_YEAR = 2013
OISST_START_YEAR = 1984
OISST_END_YEAR = 2024
OISST_2024_END_DATE = "2024-11-27"
CUTI_BEUTI_START_YEAR = 1988
CUTI_BEUTI_END_YEAR = 2024

OISST_DATASET = "SST_OI_DAILY_1981_PRESENT_HL"
OISST_BASE_URL = f"https://erddap.aoml.noaa.gov/hdb/erddap/griddap/{OISST_DATASET}.nc"
UPWELL_BASE_URL = "https://upwell.pfeg.noaa.gov/erddap"
CUTI_DATASET = "erdCUTIdaily"
BEUTI_DATASET = "erdBEUTIdaily"

DEFAULT_LABELS = Path("data/processed/modeling_kelpwatch_labels_ge500.csv")
DEFAULT_FILTERED_CELLS = Path(
    "geometries/regular_10km_fishnet/filtered_cells_historic_footprint_ge500.csv"
)
DEFAULT_INVENTORY = Path("geometries/regular_10km_fishnet/aoi_inventory_regular_10km_fishnet.csv")
DEFAULT_CACHE_DIR = Path("data/external/noaa/cache")
DEFAULT_FEATURE_OUTPUT = Path("data/processed/noaa_environmental_features_ge500.csv")
DEFAULT_MERGED_OUTPUT = Path("data/processed/modeling_dataset_ge500_noaa_v1.csv")
DEFAULT_FEATURE_SUMMARY = Path("outputs/metadata/noaa_environmental_feature_summary_ge500.csv")
DEFAULT_FEATURE_REPORT = Path("outputs/metadata/noaa_environmental_feature_report.md")
DEFAULT_MODELING_SUMMARY = Path("outputs/metadata/modeling_dataset_ge500_noaa_v1_summary.csv")
DEFAULT_PROGRESS_LOG = Path("outputs/metadata/noaa_environmental_feature_progress_log.csv")

REQUIRED_LABEL_COLUMNS = {
    "cell_id",
    "year",
    "decline_event_next",
    "decline_event_next_p25_full",
    "decline_50pct_next",
}
REQUIRED_CELL_COLUMNS = {"cell_id", "region_group", "center_lat", "center_lon"}
NOAA_FEATURE_COLUMNS = [
    "nearest_oisst_lat",
    "nearest_oisst_lon",
    "oisst_source_lat",
    "oisst_source_lon",
    "oisst_observed_days",
    "oisst_gap_filled",
    "annual_mean_sst",
    "annual_max_sst",
    "annual_min_sst",
    "annual_sst_std",
    "annual_mean_sst_anomaly",
    "annual_max_sst_anomaly",
    "hot_days_p90",
    "hot_days_p95",
    "lag1_annual_mean_sst_anomaly",
    "lag1_hot_days_p90",
    "cuti_lat_bin",
    "beuti_lat_bin",
    "annual_mean_cuti",
    "spring_mean_cuti",
    "summer_mean_cuti",
    "cuti_anomaly",
    "lag1_cuti_anomaly",
    "annual_mean_beuti",
    "spring_mean_beuti",
    "summer_mean_beuti",
    "beuti_anomaly",
    "lag1_beuti_anomaly",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build cached NOAA environmental features.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--filtered-cells", type=Path, default=DEFAULT_FILTERED_CELLS)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--feature-output", type=Path, default=DEFAULT_FEATURE_OUTPUT)
    parser.add_argument("--merged-output", type=Path, default=DEFAULT_MERGED_OUTPUT)
    parser.add_argument("--feature-summary-output", type=Path, default=DEFAULT_FEATURE_SUMMARY)
    parser.add_argument("--feature-report-output", type=Path, default=DEFAULT_FEATURE_REPORT)
    parser.add_argument("--modeling-summary-output", type=Path, default=DEFAULT_MODELING_SUMMARY)
    parser.add_argument("--progress-log-output", type=Path, default=DEFAULT_PROGRESS_LOG)
    parser.add_argument("--limit-cells", type=int, default=None, help="Limit retained cells for smoke tests.")
    parser.add_argument("--output-tag", default="", help="Optional suffix inserted before output extensions.")
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def tagged_path(path: Path, tag: str) -> Path:
    """Insert an optional tag before the file suffix."""
    if not tag:
        return path
    clean = tag if tag.startswith("_") else f"_{tag}"
    return path.with_name(f"{path.stem}{clean}{path.suffix}")


def require_columns(data: pd.DataFrame, required: set[str], name: str) -> None:
    """Validate required columns."""
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def oisst_nearest_grid(value: float, start: float) -> float:
    """Snap a coordinate to the nearest OISST 0.25-degree grid center."""
    return round((value - start) / 0.25) * 0.25 + start


def nearest_cuti_beuti_lat(value: float) -> float:
    """Assign a centroid latitude to the nearest available CUTI/BEUTI latitude bin."""
    return float(min(max(round(value), 31), 47))


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load label rows and retained cell metadata."""
    if not args.labels.exists():
        raise FileNotFoundError(args.labels)
    if not args.filtered_cells.exists():
        raise FileNotFoundError(args.filtered_cells)
    if not args.inventory.exists():
        raise FileNotFoundError(args.inventory)

    labels = pd.read_csv(args.labels)
    retained = pd.read_csv(args.filtered_cells)
    inventory = pd.read_csv(args.inventory)
    require_columns(labels, REQUIRED_LABEL_COLUMNS, str(args.labels))
    require_columns(retained, {"cell_id"}, str(args.filtered_cells))
    require_columns(inventory, REQUIRED_CELL_COLUMNS, str(args.inventory))

    retained_ids = retained["cell_id"].drop_duplicates().tolist()
    if args.limit_cells is not None:
        retained_ids = retained_ids[: args.limit_cells]

    cell_meta = inventory.loc[inventory["cell_id"].isin(retained_ids)].copy()
    labels = labels.loc[labels["cell_id"].isin(retained_ids)].copy()
    labels["year"] = pd.to_numeric(labels["year"], errors="raise").astype(int)

    if len(cell_meta) != len(retained_ids):
        missing = sorted(set(retained_ids) - set(cell_meta["cell_id"]))
        raise ValueError(f"Missing centroid metadata for cells: {missing}")
    if args.limit_cells is None and cell_meta["cell_id"].nunique() != EXPECTED_CELLS:
        raise ValueError(f"Expected {EXPECTED_CELLS} cells, found {cell_meta['cell_id'].nunique()}.")
    if args.limit_cells is None and len(labels) != EXPECTED_MODELING_ROWS:
        raise ValueError(f"Expected {EXPECTED_MODELING_ROWS} label rows, found {len(labels)}.")

    cell_meta["nearest_oisst_lat"] = cell_meta["center_lat"].map(
        lambda value: oisst_nearest_grid(float(value), -89.875)
    )
    cell_meta["nearest_oisst_lon"] = cell_meta["center_lon"].map(
        lambda value: oisst_nearest_grid(float(value), -179.875)
    )
    cell_meta["cuti_lat_bin"] = cell_meta["center_lat"].map(nearest_cuti_beuti_lat)
    cell_meta["beuti_lat_bin"] = cell_meta["center_lat"].map(nearest_cuti_beuti_lat)
    return labels, cell_meta


def request_bytes(url: str, delay_seconds: float) -> bytes:
    """Download bytes with retry/backoff."""
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.get(url, timeout=180)
            if response.status_code == 200:
                time.sleep(delay_seconds)
                return response.content
            last_error = RuntimeError(f"{response.status_code}: {response.text[:300]}")
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(delay_seconds + attempt * 2)
    raise RuntimeError(f"NOAA request failed after retries: {url}") from last_error


def oisst_url(lat: float, lon: float, start_year: int, end_year: int) -> str:
    """Build an OISST NetCDF subset URL for one grid point and year range."""
    end_date = f"{end_year}-12-31"
    if end_year == 2024:
        end_date = OISST_2024_END_DATE
    query = (
        f"sst[({start_year}-01-01T00:00:00Z):1:({end_date}T00:00:00Z)]"
        f"[({lat:.3f})][({lon:.3f})]"
    )
    return f"{OISST_BASE_URL}?{quote(query, safe='?,=&[]():')}"


def read_oisst_netcdf(path: Path) -> pd.DataFrame:
    """Read an OISST NetCDF point subset into daily rows."""
    with Dataset(path) as dataset:
        time_var = dataset.variables["time"]
        times = num2date(time_var[:], units=time_var.units, only_use_cftime_datetimes=False)
        sst = np.asarray(dataset.variables["sst"][:], dtype=float)
    sst = np.squeeze(sst)
    sst = np.where(sst <= -9, np.nan, sst)
    return pd.DataFrame({"time": pd.to_datetime(times, utc=True), "sst": sst}).dropna()


def oisst_neighbor_candidates(lat: float, lon: float) -> list[tuple[float, float]]:
    """Return nearby OISST grid candidates, starting with the requested point."""
    candidates = [(lat, lon)]
    offsets = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
    for dlat in offsets:
        for dlon in offsets:
            candidate = (round(lat + dlat, 3), round(lon + dlon, 3))
            if candidate == (lat, lon):
                continue
            candidates.append(candidate)
    return sorted(candidates, key=lambda point: (point[0] - lat) ** 2 + (point[1] - lon) ** 2)


def build_single_oisst_daily_cache(
    lat: float,
    lon: float,
    cache_dir: Path,
    force: bool,
    delay_seconds: float,
    progress: dict[str, object],
) -> pd.DataFrame:
    """Build or read one exact cached daily OISST time series."""
    cache_path = cache_dir / "oisst" / f"oisst_lat{lat:.3f}_lon{lon:.3f}_daily.csv".replace("-", "m")
    if cache_path.exists() and not force:
        progress["cached_files_reused"].append(str(cache_path))
        return pd.read_csv(cache_path, parse_dates=["time"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    chunk_ranges = [(1984, 1991), (1992, 1999), (2000, 2007), (2008, 2015), (2016, 2023), (2024, 2024)]
    tmp_dir = cache_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for start_year, end_year in chunk_ranges:
        tmp_path = tmp_dir / f"oisst_{lat:.3f}_{lon:.3f}_{start_year}_{end_year}.nc".replace("-", "m")
        if force or not tmp_path.exists():
            tmp_path.write_bytes(request_bytes(oisst_url(lat, lon, start_year, end_year), delay_seconds))
        frames.append(read_oisst_netcdf(tmp_path))

    daily = pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time")
    daily.to_csv(cache_path, index=False)
    progress["downloaded_files"].append(str(cache_path))
    return daily


def build_oisst_daily_cache(
    lat: float,
    lon: float,
    cache_dir: Path,
    force: bool,
    delay_seconds: float,
    progress: dict[str, object],
) -> tuple[pd.DataFrame, float, float]:
    """Build or read a daily OISST time series, with nearest valid ocean-grid fallback."""
    attempted: list[tuple[float, float]] = []
    for candidate_lat, candidate_lon in oisst_neighbor_candidates(lat, lon):
        attempted.append((candidate_lat, candidate_lon))
        daily = build_single_oisst_daily_cache(
            candidate_lat, candidate_lon, cache_dir, force, delay_seconds, progress
        )
        if not daily.empty and daily["sst"].notna().any():
            return daily, candidate_lat, candidate_lon
    raise ValueError(f"No valid OISST SST data found near {lat}, {lon}. Attempted: {attempted}")


def annual_oisst_features(
    daily: pd.DataFrame,
    requested_lat: float,
    requested_lon: float,
    source_lat: float,
    source_lon: float,
) -> pd.DataFrame:
    """Aggregate daily OISST into annual thermal stress features."""
    daily = daily.copy()
    daily["year"] = pd.to_datetime(daily["time"], utc=True).dt.year
    daily["sst"] = pd.to_numeric(daily["sst"], errors="coerce")
    baseline = daily.loc[daily["year"].between(BASELINE_START_YEAR, BASELINE_END_YEAR)]
    p90 = baseline["sst"].quantile(0.90)
    p95 = baseline["sst"].quantile(0.95)
    annual = (
        daily.groupby("year")
        .agg(
            annual_mean_sst=("sst", "mean"),
            annual_max_sst=("sst", "max"),
            annual_min_sst=("sst", "min"),
            annual_sst_std=("sst", "std"),
            oisst_observed_days=("sst", "size"),
            hot_days_p90=("sst", lambda values: int((values > p90).sum())),
            hot_days_p95=("sst", lambda values: int((values > p95).sum())),
        )
        .reindex(range(OISST_START_YEAR, OISST_END_YEAR + 1))
    )
    baseline_annual = annual.loc[BASELINE_START_YEAR:BASELINE_END_YEAR]
    annual["annual_mean_sst_anomaly"] = annual["annual_mean_sst"] - baseline_annual["annual_mean_sst"].mean()
    annual["annual_max_sst_anomaly"] = annual["annual_max_sst"] - baseline_annual["annual_max_sst"].mean()
    annual["oisst_gap_filled"] = annual["annual_mean_sst"].isna()
    fill_cols = [
        "annual_mean_sst",
        "annual_max_sst",
        "annual_min_sst",
        "annual_sst_std",
        "annual_mean_sst_anomaly",
        "annual_max_sst_anomaly",
        "hot_days_p90",
        "hot_days_p95",
    ]
    annual[fill_cols] = annual[fill_cols].interpolate(method="linear", limit_direction="both")
    annual["hot_days_p90"] = annual["hot_days_p90"].round()
    annual["hot_days_p95"] = annual["hot_days_p95"].round()
    annual["oisst_observed_days"] = annual["oisst_observed_days"].fillna(0).astype(int)
    annual = annual.reset_index(names="year")
    annual["nearest_oisst_lat"] = requested_lat
    annual["nearest_oisst_lon"] = requested_lon
    annual["oisst_source_lat"] = source_lat
    annual["oisst_source_lon"] = source_lon
    return annual


def build_oisst_features(
    labels: pd.DataFrame,
    cell_meta: pd.DataFrame,
    args: argparse.Namespace,
    progress: dict[str, object],
) -> pd.DataFrame:
    """Build OISST features once per unique grid point and reuse by cell."""
    point_frames = []
    unique_points = cell_meta[["nearest_oisst_lat", "nearest_oisst_lon"]].drop_duplicates()
    for point in unique_points.to_dict("records"):
        lat = float(point["nearest_oisst_lat"])
        lon = float(point["nearest_oisst_lon"])
        daily, source_lat, source_lon = build_oisst_daily_cache(
            lat, lon, args.cache_dir, args.force_download, args.delay_seconds, progress
        )
        point_frames.append(annual_oisst_features(daily, lat, lon, source_lat, source_lon))

    point_features = pd.concat(point_frames, ignore_index=True)
    features = cell_meta[["cell_id", "nearest_oisst_lat", "nearest_oisst_lon"]].merge(
        point_features, on=["nearest_oisst_lat", "nearest_oisst_lon"], how="left"
    )
    features = labels[["cell_id", "year"]].merge(features, on=["cell_id", "year"], how="left")
    features = features.sort_values(["cell_id", "year"])
    features["lag1_annual_mean_sst_anomaly"] = features.groupby("cell_id")["annual_mean_sst_anomaly"].shift(1)
    features["lag1_hot_days_p90"] = features.groupby("cell_id")["hot_days_p90"].shift(1)
    return features


def upwell_url(dataset: str, variable: str, min_lat: float, max_lat: float) -> str:
    """Build a CUTI/BEUTI daily ERDDAP CSV URL for all needed latitude bins."""
    query = (
        f"{variable}[({CUTI_BEUTI_START_YEAR}-01-01T00:00:00Z):1:"
        f"({CUTI_BEUTI_END_YEAR}-12-31T00:00:00Z)][({min_lat:.1f}):1:({max_lat:.1f})]"
    )
    return f"{UPWELL_BASE_URL}/griddap/{dataset}.csvp?{quote(query, safe='?,=&[]():')}"


def normalize_noaa_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize ERDDAP csvp column names."""
    renamed = {}
    for column in data.columns:
        if column.startswith("time"):
            renamed[column] = "time"
        elif column.startswith("latitude"):
            renamed[column] = "latitude"
        elif column.startswith("CUTI"):
            renamed[column] = "cuti"
        elif column.startswith("BEUTI"):
            renamed[column] = "beuti"
    return data.rename(columns=renamed)


def build_upwell_cache(
    variable: str,
    dataset: str,
    lat_bins: list[float],
    cache_dir: Path,
    force: bool,
    delay_seconds: float,
    progress: dict[str, object],
) -> pd.DataFrame:
    """Download or read one cached CUTI/BEUTI daily table for all needed bins."""
    min_lat = min(lat_bins)
    max_lat = max(lat_bins)
    cache_path = cache_dir / "cuti_beuti" / f"{variable.lower()}_lat{min_lat:.0f}_{max_lat:.0f}_daily.csv"
    if cache_path.exists() and not force:
        progress["cached_files_reused"].append(str(cache_path))
        data = normalize_noaa_columns(pd.read_csv(cache_path))
        data["time"] = pd.to_datetime(data["time"], utc=True)
        return data

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    content = request_bytes(upwell_url(dataset, variable, min_lat, max_lat), delay_seconds)
    cache_path.write_bytes(content)
    progress["downloaded_files"].append(str(cache_path))
    data = normalize_noaa_columns(pd.read_csv(cache_path))
    data["time"] = pd.to_datetime(data["time"], utc=True)
    return data


def annual_upwell_features(daily: pd.DataFrame, value_column: str, lat_column_name: str, prefix: str) -> pd.DataFrame:
    """Aggregate daily CUTI/BEUTI into annual, spring, and summer features."""
    daily = normalize_noaa_columns(daily).copy()
    daily["year"] = pd.to_datetime(daily["time"], utc=True).dt.year
    daily["month"] = pd.to_datetime(daily["time"], utc=True).dt.month
    daily[value_column] = pd.to_numeric(daily[value_column], errors="coerce")
    annual = (
        daily.groupby(["latitude", "year"])
        .agg(**{f"annual_mean_{prefix}": (value_column, "mean")})
        .reset_index()
    )
    spring = (
        daily.loc[daily["month"].isin([3, 4, 5])]
        .groupby(["latitude", "year"])[value_column]
        .mean()
        .reset_index(name=f"spring_mean_{prefix}")
    )
    summer = (
        daily.loc[daily["month"].isin([6, 7, 8])]
        .groupby(["latitude", "year"])[value_column]
        .mean()
        .reset_index(name=f"summer_mean_{prefix}")
    )
    features = annual.merge(spring, on=["latitude", "year"], how="left").merge(
        summer, on=["latitude", "year"], how="left"
    )
    baseline = features.loc[features["year"].between(BASELINE_START_YEAR, BASELINE_END_YEAR)]
    baseline_mean = baseline.groupby("latitude")[f"annual_mean_{prefix}"].mean()
    features[f"{prefix}_anomaly"] = features[f"annual_mean_{prefix}"] - features["latitude"].map(baseline_mean)
    features = features.rename(columns={"latitude": lat_column_name})
    return features


def build_cuti_beuti_features(
    labels: pd.DataFrame,
    cell_meta: pd.DataFrame,
    args: argparse.Namespace,
    progress: dict[str, object],
) -> pd.DataFrame:
    """Build CUTI/BEUTI features once for all needed latitude bins and reuse by cell."""
    bins = sorted(cell_meta["cuti_lat_bin"].drop_duplicates().astype(float).tolist())
    cuti_daily = build_upwell_cache("CUTI", CUTI_DATASET, bins, args.cache_dir, args.force_download, args.delay_seconds, progress)
    beuti_daily = build_upwell_cache("BEUTI", BEUTI_DATASET, bins, args.cache_dir, args.force_download, args.delay_seconds, progress)
    cuti = annual_upwell_features(cuti_daily, "cuti", "cuti_lat_bin", "cuti")
    beuti = annual_upwell_features(beuti_daily, "beuti", "beuti_lat_bin", "beuti")
    features = cell_meta[["cell_id", "cuti_lat_bin", "beuti_lat_bin"]].merge(cuti, on="cuti_lat_bin", how="left")
    features = features.merge(beuti, on=["beuti_lat_bin", "year"], how="left")
    features = labels[["cell_id", "year"]].merge(features, on=["cell_id", "year"], how="left")
    features = features.sort_values(["cell_id", "year"])
    features["lag1_cuti_anomaly"] = features.groupby("cell_id")["cuti_anomaly"].shift(1)
    features["lag1_beuti_anomaly"] = features.groupby("cell_id")["beuti_anomaly"].shift(1)
    return features


def summarize_missing(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Summarize missing values for selected columns."""
    return pd.DataFrame(
        {
            "feature": columns,
            "missing_count": [int(data[column].isna().sum()) for column in columns],
            "missing_rate": [float(data[column].isna().mean()) for column in columns],
        }
    )


def write_report(path: Path, features: pd.DataFrame, merged: pd.DataFrame, cell_meta: pd.DataFrame, missing: pd.DataFrame, progress: dict[str, object]) -> None:
    """Write the NOAA feature report."""
    access_notes = (
        f"OISST: {OISST_DATASET} via {OISST_BASE_URL}; "
        f"CUTI: {CUTI_DATASET} via {UPWELL_BASE_URL}; "
        f"BEUTI: {BEUTI_DATASET} via {UPWELL_BASE_URL}; "
        "CUTI/BEUTI daily values are aggregated to annual, spring, and summer summaries."
    )
    target_counts = merged["decline_event_next"].value_counts().sort_index()
    lines = [
        "# NOAA Environmental Feature Report",
        "",
        "## Summary",
        "",
        f"Number of cells: {features['cell_id'].nunique()}",
        f"Year range: {int(features['year'].min())}-{int(features['year'].max())}",
        f"Number of unique nearest OISST grid points: {cell_meta[['nearest_oisst_lat', 'nearest_oisst_lon']].drop_duplicates().shape[0]}",
        f"Number of CUTI/BEUTI latitude bins: {cell_meta['cuti_lat_bin'].nunique()}",
        f"Merged dataset row count: {len(merged)}",
        f"Baseline period: {BASELINE_START_YEAR}-{BASELINE_END_YEAR}",
        f"Runtime seconds: {progress['runtime_seconds']:.2f}",
        "",
        "## NOAA Data Access Notes",
        "",
        f"noaa_data_access_notes: {access_notes}",
        "",
        access_notes,
        "",
        "## Feature List",
        "",
    ]
    lines.extend([f"- `{column}`" for column in NOAA_FEATURE_COLUMNS])
    lines.extend(["", "## Target Distribution After Merge", ""])
    for target, count in target_counts.items():
        lines.append(f"- decline_event_next = {int(target)}: {int(count)}")
    lines.extend(["", "## Missing Value Counts", ""])
    for row in missing.to_dict("records"):
        lines.append(f"- `{row['feature']}`: {int(row['missing_count'])} ({row['missing_rate']:.4f})")
    lines.extend(
        [
            "",
            "## Progress Log Summary",
            "",
            f"- cells processed: {progress['number_of_cells']}",
            f"- unique OISST grid points: {progress['number_of_unique_oisst_grid_points']}",
            f"- CUTI/BEUTI latitude bins: {progress['number_of_cuti_beuti_latitude_bins']}",
            f"- downloaded files: {len(progress['downloaded_files'])}",
            f"- cached files reused: {len(progress['cached_files_reused'])}",
            "",
            "## Notes on Limitations",
            "",
            "- OISST is assigned using the nearest 0.25-degree grid point. A coastal-buffer average is reserved for sensitivity analysis.",
            f"- 2024 OISST annual features use data available through {OISST_2024_END_DATE}; rerun after NOAA archive updates for complete 2024 daily coverage.",
            "- CUTI/BEUTI begin in 1988, so 1984-1987 rows have missing CUTI/BEUTI values in Version 1.",
            "- Daily CUTI/BEUTI are aggregated to annual, spring, and summer summaries; daily-window features can be added later.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def validate(labels: pd.DataFrame, features: pd.DataFrame, merged: pd.DataFrame, limit_cells: int | None) -> None:
    """Validate feature and merged tables."""
    if features.duplicated(["cell_id", "year"]).any():
        raise ValueError("NOAA features contain duplicate cell_id x year rows.")
    if merged.duplicated(["cell_id", "year"]).any():
        raise ValueError("Merged dataset contains duplicate cell_id x year rows.")
    if len(features) != len(labels) or len(merged) != len(labels):
        raise ValueError(f"Row mismatch: labels={len(labels)}, features={len(features)}, merged={len(merged)}")
    if limit_cells is None and len(merged) != EXPECTED_MODELING_ROWS:
        raise ValueError(f"Expected {EXPECTED_MODELING_ROWS} rows, found {len(merged)}.")
    if limit_cells is None and set(merged["decline_event_next"].dropna().astype(int).unique()) != {0, 1}:
        raise ValueError("Main target does not contain both 0 and 1 classes.")


def write_outputs(args: argparse.Namespace, features: pd.DataFrame, merged: pd.DataFrame, cell_meta: pd.DataFrame, progress: dict[str, object]) -> None:
    """Write processed outputs and commit-friendly metadata."""
    feature_output = tagged_path(args.feature_output, args.output_tag)
    merged_output = tagged_path(args.merged_output, args.output_tag)
    feature_summary_output = tagged_path(args.feature_summary_output, args.output_tag)
    feature_report_output = tagged_path(args.feature_report_output, args.output_tag)
    modeling_summary_output = tagged_path(args.modeling_summary_output, args.output_tag)
    progress_log_output = tagged_path(args.progress_log_output, args.output_tag)

    for path in [feature_output, merged_output, feature_summary_output, modeling_summary_output, progress_log_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(feature_output, index=False)
    merged.to_csv(merged_output, index=False)
    missing = summarize_missing(features, NOAA_FEATURE_COLUMNS)
    missing.to_csv(feature_summary_output, index=False)
    pd.DataFrame(
        [
            ("rows", len(merged)),
            ("columns", merged.shape[1]),
            ("cells", merged["cell_id"].nunique()),
            ("year_range", f"{int(merged['year'].min())}-{int(merged['year'].max())}"),
            ("decline_event_next_rate", float(merged["decline_event_next"].mean())),
            (
                "noaa_data_access_notes",
                f"OISST={OISST_DATASET}; CUTI={CUTI_DATASET}; BEUTI={BEUTI_DATASET}; base={UPWELL_BASE_URL}",
            ),
        ],
        columns=["metric", "value"],
    ).to_csv(modeling_summary_output, index=False)
    pd.DataFrame(
        [
            ("number_of_cells", progress["number_of_cells"]),
            ("number_of_unique_oisst_grid_points", progress["number_of_unique_oisst_grid_points"]),
            ("number_of_cuti_beuti_latitude_bins", progress["number_of_cuti_beuti_latitude_bins"]),
            ("downloaded_files", "|".join(progress["downloaded_files"])),
            ("cached_files_reused", "|".join(progress["cached_files_reused"])),
            ("runtime_seconds", f"{progress['runtime_seconds']:.2f}"),
        ],
        columns=["metric", "value"],
    ).to_csv(progress_log_output, index=False)
    write_report(feature_report_output, features, merged, cell_meta, missing, progress)


def main() -> None:
    """Run the cached NOAA environmental feature workflow."""
    started = time.time()
    args = parse_args()
    labels, cell_meta = load_inputs(args)
    progress: dict[str, object] = {
        "number_of_cells": cell_meta["cell_id"].nunique(),
        "number_of_unique_oisst_grid_points": cell_meta[["nearest_oisst_lat", "nearest_oisst_lon"]].drop_duplicates().shape[0],
        "number_of_cuti_beuti_latitude_bins": cell_meta["cuti_lat_bin"].nunique(),
        "downloaded_files": [],
        "cached_files_reused": [],
        "runtime_seconds": 0.0,
    }
    oisst = build_oisst_features(labels, cell_meta, args, progress)
    cuti_beuti = build_cuti_beuti_features(labels, cell_meta, args, progress)
    features = labels[["cell_id", "year"]].merge(oisst, on=["cell_id", "year"], how="left").merge(
        cuti_beuti, on=["cell_id", "year"], how="left"
    )
    merged = labels.merge(features, on=["cell_id", "year"], how="left")
    validate(labels, features, merged, args.limit_cells)
    progress["runtime_seconds"] = time.time() - started
    write_outputs(args, features, merged, cell_meta, progress)

    missing = summarize_missing(features, NOAA_FEATURE_COLUMNS)
    print("NOAA environmental feature workflow complete.")
    print(f"cells processed: {progress['number_of_cells']}")
    print(f"unique OISST grid points: {progress['number_of_unique_oisst_grid_points']}")
    print(f"CUTI/BEUTI latitude bins: {progress['number_of_cuti_beuti_latitude_bins']}")
    print(f"downloaded files: {len(progress['downloaded_files'])}")
    print(f"cached files reused: {len(progress['cached_files_reused'])}")
    print(f"missing values: {int(missing['missing_count'].sum())}")
    print(f"runtime seconds: {progress['runtime_seconds']:.2f}")


if __name__ == "__main__":
    main()
