"""Build leakage-safe NOAA features for the locked 165-cell west-coast panel.

The complete weekly NOAA OISST v2.1 record is used because the previously used
AOML daily mirror has multi-year gaps and stops in November 2024. CUTI/BEUTI
are assigned only to centroids inside their 31--47 N latitude support; no
boundary clamping or geographic extrapolation is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from netCDF4 import Dataset, num2date

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_noaa_environmental_features import (  # noqa: E402
    BEUTI_DATASET,
    CUTI_DATASET,
    build_upwell_cache,
    normalize_noaa_columns,
    oisst_neighbor_candidates,
)


BASELINE_START = 1984
BASELINE_END = 2004
UPWELL_BASELINE_START = 1988
UPWELL_MIN_LAT = 31.0
UPWELL_MAX_LAT = 47.0
UPWELL_US_REGIONS = {
    "Southern California",
    "Central California",
    "Northern California",
    "Oregon",
    "Washington outer coast",
}
OISST_DATASET = "noaa_psl_62b6_f192_98f7"
OISST_URL = f"https://comet.nefsc.noaa.gov/erddap/griddap/{OISST_DATASET}.nc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/processed/kelpwatch_west_coast_panel_165.csv"),
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/external/noaa/cache")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping-output", type=Path, required=True)
    parser.add_argument("--coverage-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snap(value: float, start: float) -> float:
    return round((value - start) / 0.25) * 0.25 + start


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def annual_oisst(
    daily: pd.DataFrame,
    requested_lat: float,
    requested_lon: float,
    source_lat: float,
    source_lon: float,
) -> pd.DataFrame:
    data = daily.copy()
    data["time"] = pd.to_datetime(data["time"], utc=True)
    data["year"] = data["time"].dt.year
    data["month"] = data["time"].dt.month
    data["sst"] = pd.to_numeric(data["sst"], errors="coerce")
    baseline_daily = data.loc[data["year"].between(BASELINE_START, BASELINE_END), "sst"]
    if baseline_daily.notna().sum() < 1_000:
        raise ValueError(f"Incomplete pre-2005 OISST baseline at {source_lat}, {source_lon}")
    p90 = float(baseline_daily.quantile(0.90))
    annual = (
        data.loc[data["year"].between(BASELINE_START, 2024)]
        .groupby("year")
        .agg(
            annual_mean_sst=("sst", "mean"),
            annual_max_sst=("sst", "max"),
            oisst_observed_weeks=("sst", "count"),
            hot_weeks_p90=("sst", lambda x: int((x > p90).sum())),
        )
        .reset_index()
    )
    expected_years = set(range(BASELINE_START, 2025))
    if set(annual["year"]) != expected_years:
        missing = sorted(expected_years - set(annual["year"]))
        raise ValueError(f"Missing OISST years at {source_lat}, {source_lon}: {missing}")
    baseline = annual.loc[annual["year"].between(BASELINE_START, BASELINE_END)]
    annual["annual_mean_sst_anomaly"] = annual["annual_mean_sst"] - baseline["annual_mean_sst"].mean()
    annual["annual_max_sst_anomaly"] = annual["annual_max_sst"] - baseline["annual_max_sst"].mean()
    for season, months in {"winter": [1, 2, 3], "summer": [7, 8, 9]}.items():
        column = f"{season}_mean_sst"
        seasonal = (
            data.loc[data["month"].isin(months)]
            .groupby("year")["sst"]
            .mean()
            .rename(column)
        )
        annual = annual.merge(seasonal, on="year", how="left")
        seasonal_baseline = annual.loc[
            annual["year"].between(BASELINE_START, BASELINE_END), column
        ].mean()
        annual[f"{column}_anomaly"] = annual[column] - seasonal_baseline
    annual["requested_oisst_lat"] = requested_lat
    annual["requested_oisst_lon"] = requested_lon
    annual["oisst_source_lat"] = source_lat
    annual["oisst_source_lon"] = source_lon
    annual["oisst_source_distance_km"] = haversine_km(
        requested_lat, requested_lon, source_lat, source_lon
    )
    return annual


def oisst_query(lat: float, lon: float) -> str:
    longitude_360 = lon % 360
    query = (
        "sst[(1984-01-01T00:00:00Z):1:(2024-12-31T00:00:00Z)]"
        f"[({lat:.3f})][({longitude_360:.3f})]"
    )
    return f"{OISST_URL}?{quote(query, safe='?,=&[]():')}"


def read_weekly_netcdf(path: Path) -> pd.DataFrame:
    with Dataset(path) as dataset:
        time_variable = dataset.variables["time"]
        times = num2date(
            time_variable[:],
            units=time_variable.units,
            only_use_cftime_datetimes=False,
        )
        values = np.ma.filled(dataset.variables["sst"][:], np.nan).astype(float)
    return pd.DataFrame(
        {"time": pd.to_datetime(times, utc=True), "sst": np.squeeze(values)}
    ).dropna()


def download_oisst_point(
    requested_lat: float,
    requested_lon: float,
    cache_dir: Path,
    delay: float,
) -> tuple[pd.DataFrame, float, float]:
    """Read or download one complete weekly OISST v2.1 point series."""
    cache_root = cache_dir / "oisst_weekly"
    temp_root = cache_dir / "tmp_weekly"
    cache_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    for source_lat, source_lon in oisst_neighbor_candidates(requested_lat, requested_lon):
        stem = f"oisst_weekly_lat{source_lat:.3f}_lon{source_lon:.3f}.csv".replace("-", "m")
        cache_path = cache_root / stem
        if cache_path.exists():
            cached = pd.read_csv(cache_path, parse_dates=["time"])
            if len(cached) >= 2_130 and cached["sst"].notna().any():
                return cached, float(source_lat), float(source_lon)
        token = f"{source_lat:.3f}_{source_lon:.3f}".replace("-", "m")
        full_path = temp_root / f"weekly_{token}.nc"
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = requests.get(oisst_query(source_lat, source_lon), timeout=300)
                if response.status_code == 404:
                    break
                response.raise_for_status()
                full_path.write_bytes(response.content)
                weekly = read_weekly_netcdf(full_path)
                if len(weekly) < 2_130 or not weekly["sst"].between(-3, 45).all():
                    break
                weekly.to_csv(cache_path, index=False)
                time.sleep(delay)
                return weekly, float(source_lat), float(source_lon)
            except (requests.RequestException, OSError, ValueError, OverflowError) as exc:
                last_error = exc
                time.sleep(2 + attempt * 3)
        if last_error is not None and attempt == 2:
            raise RuntimeError(f"Failed OISST at {source_lat}, {source_lon}") from last_error
    raise ValueError(f"No valid ocean OISST point near {requested_lat}, {requested_lon}")


def build_oisst(panel: pd.DataFrame, cache_dir: Path, workers: int, delay: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = panel[["cell_id", "center_lat", "center_lon"]].drop_duplicates().copy()
    cells["requested_oisst_lat"] = cells["center_lat"].map(lambda x: snap(float(x), -89.875))
    cells["requested_oisst_lon"] = cells["center_lon"].map(lambda x: snap(float(x), -179.875))
    points = list(
        cells[["requested_oisst_lat", "requested_oisst_lon"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    def load_point(point: tuple[float, float]):
        lat, lon = map(float, point)
        weekly, source_lat, source_lon = download_oisst_point(
            lat, lon, cache_dir, delay
        )
        return lat, lon, weekly, float(source_lat), float(source_lon)

    loaded = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(load_point, point): point for point in points}
        for future in as_completed(futures):
            lat, lon, weekly, source_lat, source_lon = future.result()
            loaded[(lat, lon)] = (weekly, source_lat, source_lon)
            print(f"OISST point {len(loaded)}/{len(points)}: {lat:.3f}, {lon:.3f} -> {source_lat:.3f}, {source_lon:.3f}", flush=True)

    frames = []
    mapping_rows = []
    for (requested_lat, requested_lon), (weekly, source_lat, source_lon) in loaded.items():
        frames.append(
            annual_oisst(
                weekly, requested_lat, requested_lon, source_lat, source_lon
            )
        )
        mapping_rows.append(
            {
                "requested_oisst_lat": requested_lat,
                "requested_oisst_lon": requested_lon,
                "oisst_source_lat": source_lat,
                "oisst_source_lon": source_lon,
                "oisst_source_distance_km": haversine_km(
                    requested_lat, requested_lon, source_lat, source_lon
                ),
            }
        )
    point_features = pd.concat(frames, ignore_index=True)
    features = cells.merge(
        point_features,
        on=["requested_oisst_lat", "requested_oisst_lon"],
        how="left",
    )
    features = features.sort_values(["cell_id", "year"])
    features["lag1_annual_mean_sst_anomaly"] = features.groupby("cell_id")[
        "annual_mean_sst_anomaly"
    ].shift(1)
    features["lag1_hot_weeks_p90"] = features.groupby("cell_id")["hot_weeks_p90"].shift(1)
    mapping = pd.DataFrame(mapping_rows).sort_values(
        ["requested_oisst_lat", "requested_oisst_lon"]
    )
    mapping["oisst_endpoint"] = OISST_URL
    mapping["temporal_resolution"] = "weekly"
    return features, mapping


def annual_upwell(daily: pd.DataFrame, value: str, prefix: str) -> pd.DataFrame:
    data = normalize_noaa_columns(daily).copy()
    data["time"] = pd.to_datetime(data["time"], utc=True)
    data["year"] = data["time"].dt.year
    data["month"] = data["time"].dt.month
    data[value] = pd.to_numeric(data[value], errors="coerce")
    annual = data.groupby(["latitude", "year"])[value].mean().reset_index(
        name=f"annual_mean_{prefix}"
    )
    baseline = annual.loc[
        annual["year"].between(UPWELL_BASELINE_START, BASELINE_END)
    ].groupby("latitude")[f"annual_mean_{prefix}"].mean()
    annual[f"{prefix}_anomaly"] = annual[f"annual_mean_{prefix}"] - annual[
        "latitude"
    ].map(baseline)
    for season, months in {
        "winter": [1, 2, 3],
        "spring": [4, 5, 6],
        "upwelling_season": [3, 4, 5, 6, 7, 8, 9],
    }.items():
        column = f"{season}_mean_{prefix}"
        seasonal = (
            data.loc[data["month"].isin(months)]
            .groupby(["latitude", "year"])[value]
            .mean()
            .reset_index(name=column)
        )
        annual = annual.merge(seasonal, on=["latitude", "year"], how="left")
        seasonal_baseline = (
            annual.loc[annual["year"].between(UPWELL_BASELINE_START, BASELINE_END)]
            .groupby("latitude")[column]
            .mean()
        )
        annual[f"{season}_{prefix}_anomaly"] = annual[column] - annual[
            "latitude"
        ].map(seasonal_baseline)
    return annual


def build_upwelling(panel: pd.DataFrame, cache_dir: Path, delay: float) -> pd.DataFrame:
    cells = panel[["cell_id", "center_lat", "region_group"]].drop_duplicates().copy()
    cells["upwelling_supported"] = (
        cells["center_lat"].between(UPWELL_MIN_LAT, UPWELL_MAX_LAT, inclusive="both")
        & cells["region_group"].isin(UPWELL_US_REGIONS)
    )
    cells["upwelling_lat_bin"] = np.where(
        cells["upwelling_supported"], cells["center_lat"].round(), np.nan
    )
    progress: dict[str, list[str]] = {"downloaded_files": [], "cached_files_reused": []}
    bins = list(np.arange(UPWELL_MIN_LAT, UPWELL_MAX_LAT + 1))
    cuti_daily = build_upwell_cache(
        "CUTI", CUTI_DATASET, bins, cache_dir, False, delay, progress
    )
    beuti_daily = build_upwell_cache(
        "BEUTI", BEUTI_DATASET, bins, cache_dir, False, delay, progress
    )
    cuti = annual_upwell(cuti_daily, "cuti", "cuti")
    beuti = annual_upwell(beuti_daily, "beuti", "beuti")
    annual = cuti.merge(beuti, on=["latitude", "year"], how="outer")
    features = cells.merge(
        annual,
        left_on="upwelling_lat_bin",
        right_on="latitude",
        how="left",
    ).drop(columns="latitude")
    features = features.sort_values(["cell_id", "year"])
    features["lag1_cuti_anomaly"] = features.groupby("cell_id")["cuti_anomaly"].shift(1)
    features["lag1_beuti_anomaly"] = features.groupby("cell_id")["beuti_anomaly"].shift(1)
    return features


def main() -> None:
    args = parse_args()
    started = time.time()
    panel = pd.read_csv(args.panel)
    required = {"cell_id", "year", "center_lat", "center_lon", "region_group"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"Panel missing columns: {missing}")
    if panel["cell_id"].nunique() != 165 or panel.duplicated(["cell_id", "year"]).any():
        raise ValueError("Expected a duplicate-free 165-cell panel")

    oisst, mapping = build_oisst(
        panel, args.cache_dir, args.workers, args.delay_seconds
    )
    upwelling = build_upwelling(panel, args.cache_dir, args.delay_seconds)
    keys = panel[["cell_id", "year"]]
    features = keys.merge(oisst, on=["cell_id", "year"], how="left").merge(
        upwelling, on=["cell_id", "year"], how="left"
    )
    if features.duplicated(["cell_id", "year"]).any() or len(features) != len(panel):
        raise ValueError("Environmental feature merge changed panel keys")

    coverage = pd.DataFrame(
        [
            {
                "check": "panel_cells",
                "value": panel["cell_id"].nunique(),
                "expected": 165,
                "passed": panel["cell_id"].nunique() == 165,
            },
            {
                "check": "unique_oisst_requested_points",
                "value": len(mapping),
                "expected": ">0",
                "passed": len(mapping) > 0,
            },
            {
                "check": "oisst_2024_complete_weeks_min",
                "value": int(
                    features.loc[features["year"].eq(2024), "oisst_observed_weeks"].min()
                ),
                "expected": ">=52",
                "passed": bool(
                    features.loc[features["year"].eq(2024), "oisst_observed_weeks"].ge(52).all()
                ),
            },
            {
                "check": "upwelling_supported_cells",
                "value": int(
                    features.loc[features["upwelling_supported"].fillna(False), "cell_id"].nunique()
                ),
                "expected": "107 U.S. cells with centroid latitude 31-47N",
                "passed": int(
                    features.loc[features["upwelling_supported"].fillna(False), "cell_id"].nunique()
                ) == 107,
            },
            {
                "check": "max_oisst_source_distance_km",
                "value": float(mapping["oisst_source_distance_km"].max()),
                "expected": "reported_not_hidden",
                "passed": True,
            },
        ]
    )
    if not coverage.loc[
        coverage["check"].isin(
            ["panel_cells", "oisst_2024_complete_weeks_min", "upwelling_supported_cells"]
        ),
        "passed",
    ].all():
        raise AssertionError(coverage.to_string(index=False))

    for path in [args.output, args.mapping_output, args.coverage_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    mapping.to_csv(args.mapping_output, index=False)
    coverage.to_csv(args.coverage_output, index=False)
    metadata = {
        "panel": str(args.panel),
        "panel_sha256": sha256(args.panel),
        "output_sha256": sha256(args.output),
        "oisst_endpoint": OISST_URL,
        "oisst_dataset": OISST_DATASET,
        "oisst_temporal_resolution": "weekly",
        "thermal_baseline": [BASELINE_START, BASELINE_END],
        "upwelling_baseline": [UPWELL_BASELINE_START, BASELINE_END],
        "upwelling_geographic_support": [UPWELL_MIN_LAT, UPWELL_MAX_LAT],
        "runtime_seconds": round(time.time() - started, 2),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
