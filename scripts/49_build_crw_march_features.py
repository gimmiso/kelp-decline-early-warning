"""Build March-cutoff NOAA CRW 5 km SST features for the locked 165-cell panel.

This workflow is intentionally narrower than the older CRW composite scripts.
It downloads only January-March monthly mean SST for 1985-2025, assigns every
locked Kelpwatch cell to the nearest valid CRW ocean pixel within 15 km, and
writes compact cell-year features for the March 31 forecast-clock experiment.

Raw monthly NetCDF files are streamed from NOAA STAR and deleted after the
needed West Coast subset is extracted. The compact extracted cache is resumable.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from netCDF4 import Dataset


STAR_MONTHLY_ROOT = (
    "https://www.star.nesdis.noaa.gov/pub/socd/mecb/crw/data/5km/"
    "v3.1_op/nc/v1.0/monthly"
)
START_YEAR = 1985
END_YEAR = 2025
BASELINE_END_YEAR = 2004
MONTHS = (1, 2, 3)
MAX_MATCH_DISTANCE_KM = 15.0
SEARCH_RADIUS_GRID_CELLS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/processed/kelpwatch_west_coast_panel_165.csv"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "data/external/noaa/cache/crw5km_march/"
            "crw5km_january_march_monthly_points_165.csv"
        ),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "data/processed/crw5km_march_features_165.csv"
        ),
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("outputs/metadata/crw5km_march_cell_mapping_165.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("outputs/metadata/crw5km_march_features_165.metadata.json"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> np.ndarray:
    radius = 6371.0088
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def monthly_url(year: int, month: int) -> str:
    return (
        f"{STAR_MONTHLY_ROOT}/{year}/"
        f"ct5km_sst-mean_v3.1_{year}{month:02d}.nc"
    )


def download(url: str, output: Path) -> None:
    response = requests.get(url, timeout=240)
    response.raise_for_status()
    output.write_bytes(response.content)


def load_cells(panel_path: Path) -> pd.DataFrame:
    panel = pd.read_csv(panel_path)
    required = {"cell_id", "center_lat", "center_lon"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Missing panel columns: {sorted(missing)}")
    cells = panel[["cell_id", "center_lat", "center_lon"]].drop_duplicates()
    if len(cells) != 165 or cells["cell_id"].nunique() != 165:
        raise ValueError(f"Expected locked 165-cell panel, found {len(cells)} rows")
    if cells["cell_id"].duplicated().any():
        raise ValueError("Cell metadata is not unique")
    return cells.sort_values("cell_id").reset_index(drop=True)


def build_mapping(cells: pd.DataFrame, sample_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with Dataset(sample_path) as dataset:
        latitudes = np.asarray(dataset.variables["lat"][:], dtype=float)
        longitudes = np.asarray(dataset.variables["lon"][:], dtype=float)
        sst = dataset.variables["sea_surface_temperature"]
        for cell in cells.itertuples(index=False):
            lat_idx = int(np.nanargmin(np.abs(latitudes - float(cell.center_lat))))
            lon_idx = int(np.nanargmin(np.abs(longitudes - float(cell.center_lon))))
            lat0 = max(0, lat_idx - SEARCH_RADIUS_GRID_CELLS)
            lat1 = min(len(latitudes), lat_idx + SEARCH_RADIUS_GRID_CELLS + 1)
            lon0 = max(0, lon_idx - SEARCH_RADIUS_GRID_CELLS)
            lon1 = min(len(longitudes), lon_idx + SEARCH_RADIUS_GRID_CELLS + 1)
            window = np.ma.filled(sst[0, lat0:lat1, lon0:lon1], np.nan).astype(float)
            valid_i, valid_j = np.where(np.isfinite(window))
            if len(valid_i) == 0:
                raise ValueError(f"No valid CRW ocean pixel near {cell.cell_id}")
            candidate_lats = latitudes[lat0 + valid_i]
            candidate_lons = longitudes[lon0 + valid_j]
            distances = haversine_km(
                float(cell.center_lat),
                float(cell.center_lon),
                candidate_lats,
                candidate_lons,
            )
            selected = int(np.argmin(distances))
            distance = float(distances[selected])
            if distance > MAX_MATCH_DISTANCE_KM:
                raise ValueError(
                    f"Nearest CRW ocean pixel for {cell.cell_id} is {distance:.2f} km away"
                )
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "center_lat": float(cell.center_lat),
                    "center_lon": float(cell.center_lon),
                    "crw_lat": float(candidate_lats[selected]),
                    "crw_lon": float(candidate_lons[selected]),
                    "lat_index": int(lat0 + valid_i[selected]),
                    "lon_index": int(lon0 + valid_j[selected]),
                    "distance_to_crw_grid_km": distance,
                    "assignment_status": "nearest_valid_ocean_within_15km",
                }
            )
    mapping = pd.DataFrame(rows)
    if len(mapping) != 165 or mapping["distance_to_crw_grid_km"].gt(15).any():
        raise AssertionError("CRW mapping gate failed")
    return mapping


def extract_month(year: int, month: int, mapping: pd.DataFrame) -> pd.DataFrame:
    url = monthly_url(year, month)
    with tempfile.TemporaryDirectory(prefix="kelp_crw_month_") as temp_dir:
        path = Path(temp_dir) / Path(url).name
        download(url, path)
        with Dataset(path) as dataset:
            sst = dataset.variables["sea_surface_temperature"]
            lat_indices = mapping["lat_index"].to_numpy(dtype=int)
            lon_indices = mapping["lon_index"].to_numpy(dtype=int)
            lat0, lat1 = int(lat_indices.min()), int(lat_indices.max()) + 1
            lon0, lon1 = int(lon_indices.min()), int(lon_indices.max()) + 1
            west_coast = np.ma.filled(sst[0, lat0:lat1, lon0:lon1], np.nan).astype(float)
            values = west_coast[lat_indices - lat0, lon_indices - lon0]
    out = mapping[["cell_id", "crw_lat", "crw_lon", "distance_to_crw_grid_km"]].copy()
    out["year"] = year
    out["month"] = month
    out["days_in_month"] = calendar.monthrange(year, month)[1]
    out["sst_mean_crw5km"] = values
    out["source_url"] = url
    out["extraction_status"] = np.where(np.isfinite(values), "ok", "missing")
    return out


def load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    cache = pd.read_csv(path)
    if cache.empty:
        return cache
    cache["year"] = cache["year"].astype(int)
    cache["month"] = cache["month"].astype(int)
    return cache


def complete_months(cache: pd.DataFrame) -> set[tuple[int, int]]:
    if cache.empty:
        return set()
    counts = cache.loc[cache["extraction_status"].eq("ok")].groupby(["year", "month"])["cell_id"].nunique()
    return set(counts.loc[counts.eq(165)].index.tolist())


def append_cache(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def build_features(cache: pd.DataFrame) -> pd.DataFrame:
    ok = cache.loc[cache["extraction_status"].eq("ok")].copy()
    expected = (END_YEAR - START_YEAR + 1) * len(MONTHS) * 165
    if len(ok) != expected or ok.duplicated(["cell_id", "year", "month"]).any():
        raise ValueError(f"Expected {expected} unique valid monthly rows, found {len(ok)}")
    ok["weighted_sst"] = ok["sst_mean_crw5km"] * ok["days_in_month"]
    grouped = ok.groupby(["cell_id", "year"], as_index=False)
    features = grouped.agg(
        q1_weighted_sst_sum=("weighted_sst", "sum"),
        q1_days=("days_in_month", "sum"),
        q1_max_monthly_sst_crw5km=("sst_mean_crw5km", "max"),
        q1_min_monthly_sst_crw5km=("sst_mean_crw5km", "min"),
        q1_months_available=("month", "nunique"),
        crw_lat=("crw_lat", "first"),
        crw_lon=("crw_lon", "first"),
        distance_to_crw_grid_km=("distance_to_crw_grid_km", "first"),
    )
    features["q1_mean_sst_crw5km"] = features["q1_weighted_sst_sum"] / features["q1_days"]
    baseline = features.loc[features["year"].between(START_YEAR, BASELINE_END_YEAR)]
    climatology = baseline.groupby("cell_id").agg(
        q1_mean_sst_baseline_crw5km=("q1_mean_sst_crw5km", "mean"),
        q1_max_monthly_sst_baseline_crw5km=("q1_max_monthly_sst_crw5km", "mean"),
    )
    features = features.merge(climatology, on="cell_id", how="left")
    features["q1_mean_sst_anomaly_crw5km"] = (
        features["q1_mean_sst_crw5km"] - features["q1_mean_sst_baseline_crw5km"]
    )
    features["q1_max_monthly_sst_anomaly_crw5km"] = (
        features["q1_max_monthly_sst_crw5km"]
        - features["q1_max_monthly_sst_baseline_crw5km"]
    )
    keep = [
        "cell_id",
        "year",
        "q1_mean_sst_crw5km",
        "q1_mean_sst_anomaly_crw5km",
        "q1_max_monthly_sst_crw5km",
        "q1_max_monthly_sst_anomaly_crw5km",
        "q1_min_monthly_sst_crw5km",
        "q1_months_available",
        "crw_lat",
        "crw_lon",
        "distance_to_crw_grid_km",
    ]
    return features[keep].sort_values(["cell_id", "year"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    started = time.time()
    cells = load_cells(args.panel)
    if args.force and args.cache.exists():
        args.cache.unlink()

    with tempfile.TemporaryDirectory(prefix="kelp_crw_assignment_") as temp_dir:
        sample = Path(temp_dir) / "ct5km_sst-mean_v3.1_202101.nc"
        download(monthly_url(2021, 1), sample)
        mapping = build_mapping(cells, sample)

    args.mapping.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.mapping, index=False)
    cache = load_cache(args.cache)
    done = complete_months(cache)
    pending = [
        (year, month)
        for year in range(START_YEAR, END_YEAR + 1)
        for month in MONTHS
        if (year, month) not in done
    ]
    print(f"CRW March months: cached={len(done)} pending={len(pending)}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(extract_month, year, month, mapping): (year, month)
            for year, month in pending
        }
        for number, future in enumerate(as_completed(futures), start=1):
            year, month = futures[future]
            frame = future.result()
            if frame["extraction_status"].ne("ok").any():
                raise ValueError(f"CRW extraction missing values for {year}-{month:02d}")
            append_cache(args.cache, frame)
            print(f"CRW month {number}/{len(pending)}: {year}-{month:02d}", flush=True)

    cache = load_cache(args.cache)
    features = build_features(cache)
    args.features.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.features, index=False)
    metadata = {
        "source": STAR_MONTHLY_ROOT,
        "product": "NOAA Coral Reef Watch CoralTemp v3.1 monthly mean SST",
        "months": list(MONTHS),
        "years": [START_YEAR, END_YEAR],
        "baseline_years": [START_YEAR, BASELINE_END_YEAR],
        "cells": int(features["cell_id"].nunique()),
        "cell_years": int(len(features)),
        "mapping_distance_km": {
            "mean": float(mapping["distance_to_crw_grid_km"].mean()),
            "median": float(mapping["distance_to_crw_grid_km"].median()),
            "max": float(mapping["distance_to_crw_grid_km"].max()),
        },
        "panel_sha256": sha256(args.panel),
        "cache_sha256": sha256(args.cache),
        "features_sha256": sha256(args.features),
        "runtime_seconds": round(time.time() - started, 2),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
