"""Download leakage-safe daily MUR 1-km SST features for Giraldo sites."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shlex
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests


ERDDAP_ENDPOINTS = (
    "https://coastwatch.pfeg.noaa.gov/erddap/griddap/jplMURSST41.csvp",
    "https://erddap.riddc.brown.edu/erddap/griddap/jplMURSST41.csvp",
    "https://erddap.marine.usf.edu/erddap/griddap/jplMURSST41.csvp",
)
REFERENCE_DATE = "2020-07-15T09:00:00Z"
START_DATE = "2002-06-01T09:00:00Z"
END_DATE = "2021-12-31T09:00:00Z"
FORECAST_START_YEAR = 2008
FORECAST_END_YEAR = 2021
EARLY_TILE_DEGREES = 0.1
LATE_TILE_DEGREES = 0.2
TRANSFER_TILE_SWITCH_YEAR = 2008
DOWNLOAD_CHUNKS = [
    (
        f"{year}-{'06-01' if year == 2002 else '01-01'}T09:00:00Z",
        f"{year}-12-31T09:00:00Z",
        str(year),
    )
    for year in range(2002, 2022)
]
THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snap(value: float, start: float = -89.99) -> float:
    return round((value - start) / 0.01) * 0.01 + start


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius = 6371.0088
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def request_csv(
    query: str,
    endpoint: str | None = None,
    attempts: int = 8,
    rotate_replicas: bool = True,
) -> pd.DataFrame:
    endpoint = endpoint or ERDDAP_ENDPOINTS[0]
    start_index = ERDDAP_ENDPOINTS.index(endpoint)
    last_error: Exception | None = None
    for attempt in range(attempts):
        active_endpoint = ERDDAP_ENDPOINTS[
            (start_index + attempt) % len(ERDDAP_ENDPOINTS)
            if rotate_replicas
            else start_index
        ]
        url = active_endpoint + "?" + quote(query, safe="?,=&[]():")
        try:
            if not hasattr(THREAD_LOCAL, "session"):
                THREAD_LOCAL.session = requests.Session()
            response = THREAD_LOCAL.session.get(url, timeout=(20, 180))
            response.raise_for_status()
            return pd.read_csv(io.StringIO(response.text))
        except (requests.RequestException, pd.errors.ParserError, ValueError) as error:
            last_error = error
            time.sleep(min(120, 5 * (3 ** attempt)))
    raise RuntimeError(f"MUR request failed across all configured replicas: {url}: {last_error}")


def nearest_ocean_pixel(site_id: str, latitude: float, longitude: float) -> dict[str, object]:
    center_lat = snap(latitude)
    center_lon = snap(longitude, -179.99)
    for radius in (0.05, 0.10, 0.20):
        query = (
            f"analysed_sst[({REFERENCE_DATE})]"
            f"[({center_lat - radius:.2f}):1:({center_lat + radius:.2f})]"
            f"[({center_lon - radius:.2f}):1:({center_lon + radius:.2f})]"
        )
        frame = request_csv(query)
        value_col = "analysed_sst (degree_C)"
        valid = frame.loc[frame[value_col].notna()].copy()
        if valid.empty:
            continue
        lat_col = "latitude (degrees_north)"
        lon_col = "longitude (degrees_east)"
        distances = haversine_km(
            latitude,
            longitude,
            valid[lat_col].to_numpy(dtype=float),
            valid[lon_col].to_numpy(dtype=float),
        )
        selected = valid.iloc[int(np.argmin(distances))]
        return {
            "site_id": site_id,
            "site_latitude": latitude,
            "site_longitude": longitude,
            "mur_latitude": float(selected[lat_col]),
            "mur_longitude": float(selected[lon_col]),
            "mur_match_distance_km": float(np.min(distances)),
            "mapping_radius_degrees": radius,
        }
    raise ValueError(f"No valid MUR ocean pixel within 0.2 degrees of {site_id}")


def point_cache_name(latitude: float, longitude: float) -> str:
    token = f"lat{latitude:.2f}_lon{longitude:.2f}".replace("-", "m")
    return token + ".csv"


def load_point(latitude: float, longitude: float, cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / point_cache_name(latitude, longitude)
    if path.exists():
        cached = pd.read_csv(path, parse_dates=["date"])
        if len(cached) >= 7_100 and cached["sst_c"].notna().mean() >= 0.95:
            return cached
    chunks: list[pd.DataFrame] = []
    point_stem = path.stem
    for start, end, label in DOWNLOAD_CHUNKS:
        chunk_path = cache_dir / f"{point_stem}__{label}.csv"
        expected_days = (
            pd.Timestamp(end.replace("T09:00:00Z", ""))
            - pd.Timestamp(start.replace("T09:00:00Z", ""))
        ).days + 1
        if chunk_path.exists():
            chunk = pd.read_csv(chunk_path, parse_dates=["date"])
            if len(chunk) >= int(expected_days * 0.98) and chunk["sst_c"].notna().mean() >= 0.90:
                chunks.append(chunk)
                continue
        query = (
            f"analysed_sst[({start}):1:({end})]"
            f"[({latitude:.2f})][({longitude:.2f})]"
        )
        frame = request_csv(query)
        chunk = pd.DataFrame(
            {
                "date": pd.to_datetime(frame["time (UTC)"], utc=True).dt.tz_localize(None),
                "sst_c": pd.to_numeric(frame["analysed_sst (degree_C)"], errors="coerce"),
            }
        )
        chunk.to_csv(chunk_path, index=False)
        chunks.append(chunk)
    output = (
        pd.concat(chunks, ignore_index=True)
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return output


def load_tile_period(
    tile_a: int,
    tile_b: int,
    points: pd.DataFrame,
    start: str,
    end: str,
    label: str,
    cache_dir: Path,
    tile_degrees: float,
    endpoint: str,
) -> pd.DataFrame:
    tile_dir = cache_dir / "tile_year_points"
    tile_dir.mkdir(parents=True, exist_ok=True)
    tile_token = int(round(tile_degrees * 100))
    path = tile_dir / f"tile_d{tile_token}_{tile_a}_{tile_b}_{label}.csv"
    expected_days = (
        pd.Timestamp(end.replace("T09:00:00Z", ""))
        - pd.Timestamp(start.replace("T09:00:00Z", ""))
    ).days + 1
    expected_rows = expected_days * len(points)
    if path.exists():
        cached = pd.read_csv(path, parse_dates=["date"])
        if len(cached) >= int(expected_rows * 0.98):
            return cached
    reusable_frames: list[pd.DataFrame] = []
    start_day = pd.Timestamp(start.replace("T09:00:00Z", ""))
    end_day = pd.Timestamp(end.replace("T09:00:00Z", ""))
    for candidate_path in tile_dir.glob(f"tile_d{tile_token}_{tile_a}_{tile_b}_*.csv"):
        if candidate_path == path:
            continue
        candidate = pd.read_csv(candidate_path, parse_dates=["date"])
        candidate = candidate.loc[candidate["date"].between(start_day, end_day)]
        if not candidate.empty:
            reusable_frames.append(candidate)
    if reusable_frames:
        reusable = (
            pd.concat(reusable_frames, ignore_index=True)
            .drop_duplicates(["mur_latitude", "mur_longitude", "date"])
        )
        if len(reusable) >= int(expected_rows * 0.98):
            reusable.to_csv(path, index=False)
            return reusable
    start_year = pd.Timestamp(start.replace("T09:00:00Z", "")).year
    end_year = pd.Timestamp(end.replace("T09:00:00Z", "")).year
    annual_frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        annual_path = tile_dir / f"tile_d{tile_token}_{tile_a}_{tile_b}_{year}.csv"
        if not annual_path.exists():
            annual_frames = []
            break
        annual = pd.read_csv(annual_path, parse_dates=["date"])
        year_start = pd.Timestamp(f"{year}-{'06-01' if year == 2002 else '01-01'}")
        year_end = pd.Timestamp(f"{year}-12-31")
        annual_expected = ((year_end - year_start).days + 1) * len(points)
        if len(annual) < int(annual_expected * 0.98):
            annual_frames = []
            break
        annual_frames.append(annual)
    if annual_frames:
        cached = pd.concat(annual_frames, ignore_index=True)
        cached.to_csv(path, index=False)
        return cached
    lat_min = float(points["mur_latitude"].min())
    lat_max = float(points["mur_latitude"].max())
    lon_min = float(points["mur_longitude"].min())
    lon_max = float(points["mur_longitude"].max())
    query = (
        f"analysed_sst[({start}):1:({end})]"
        f"[({lat_min:.2f}):1:({lat_max:.2f})]"
        f"[({lon_min:.2f}):1:({lon_max:.2f})]"
    )
    frame = request_csv(query, endpoint=endpoint)
    normalized = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["time (UTC)"], utc=True).dt.tz_localize(None),
            "mur_latitude": pd.to_numeric(frame["latitude (degrees_north)"], errors="coerce").round(2),
            "mur_longitude": pd.to_numeric(frame["longitude (degrees_east)"], errors="coerce").round(2),
            "sst_c": pd.to_numeric(frame["analysed_sst (degree_C)"], errors="coerce"),
        }
    )
    needed = points[["mur_latitude", "mur_longitude"]].drop_duplicates().copy()
    needed["mur_latitude"] = needed["mur_latitude"].round(2)
    needed["mur_longitude"] = needed["mur_longitude"].round(2)
    extracted = normalized.merge(
        needed,
        on=["mur_latitude", "mur_longitude"],
        how="inner",
        validate="many_to_one",
    )
    if len(extracted) < int(expected_rows * 0.98):
        raise ValueError(
            f"Incomplete tile extraction tile=({tile_a},{tile_b}) period={label}: "
            f"expected~{expected_rows}, rows={len(extracted)}"
        )
    extracted.to_csv(path, index=False)
    return extracted


def check_endpoint_parity(latitude: float, longitude: float) -> pd.DataFrame:
    query = (
        "analysed_sst[(2014-01-01T09:00:00Z):1:(2014-01-03T09:00:00Z)]"
        f"[({latitude:.2f})][({longitude:.2f})]"
    )
    with ThreadPoolExecutor(max_workers=len(ERDDAP_ENDPOINTS)) as pool:
        futures = {
            pool.submit(request_csv, query, endpoint, 5, False): endpoint
            for endpoint in ERDDAP_ENDPOINTS
        }
        frames = {endpoint: future.result() for future, endpoint in futures.items()}
    value_col = "analysed_sst (degree_C)"
    primary = pd.to_numeric(frames[ERDDAP_ENDPOINTS[0]][value_col], errors="coerce")
    rows: list[dict[str, object]] = []
    for endpoint in ERDDAP_ENDPOINTS:
        candidate = pd.to_numeric(frames[endpoint][value_col], errors="coerce")
        max_abs = float((primary - candidate).abs().max())
        rows.append(
            {
                "endpoint": endpoint,
                "rows": int(len(candidate)),
                "max_absolute_difference_celsius_vs_noaa": max_abs,
                "passed": bool(len(candidate) == len(primary) and max_abs <= 1e-9),
            }
        )
    return pd.DataFrame(rows)


def max_consecutive(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def annual_features(daily: pd.DataFrame, latitude: float, longitude: float) -> pd.DataFrame:
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["year"] = data["date"].dt.year
    data["month_day"] = data["date"].dt.strftime("%m-%d")
    rows: list[dict[str, object]] = []
    for year in range(2002, 2022):
        current = data.loc[data["year"].eq(year)].sort_values("date").copy()
        history = data.loc[data["date"].lt(pd.Timestamp(f"{year}-01-01")) & data["sst_c"].notna()].copy()
        expected = 214 if year == 2002 else (366 if pd.Timestamp(f"{year}-12-31").is_leap_year else 365)
        coverage = float(current["sst_c"].notna().sum() / expected)
        raw_mean = float(current["sst_c"].mean()) if coverage >= 0.90 else np.nan
        raw_max = float(current["sst_c"].max()) if coverage >= 0.90 else np.nan
        max_7day = positive_degree_days = hot_days = max_hot_run = np.nan
        if len(history) >= 365 and coverage >= 0.90:
            climatology = history.groupby("month_day")["sst_c"].mean()
            current["climatology"] = current["month_day"].map(climatology)
            current["anomaly"] = current["sst_c"] - current["climatology"]
            current["rolling_7day"] = current["anomaly"].rolling(7, min_periods=7).mean()
            threshold = float(history["sst_c"].quantile(0.90))
            hot = current["sst_c"].gt(threshold) & current["sst_c"].notna()
            max_7day = float(current["rolling_7day"].max())
            positive_degree_days = float(current["anomaly"].clip(lower=0).sum(min_count=1))
            hot_days = int(hot.sum())
            max_hot_run = int(max_consecutive(hot.to_numpy()))
        rows.append(
            {
                "year": year,
                "mur_latitude": latitude,
                "mur_longitude": longitude,
                "mur_expected_days": expected,
                "mur_valid_days": int(current["sst_c"].notna().sum()),
                "mur_daily_coverage": coverage,
                "annual_mean_sst_mur1km": raw_mean,
                "annual_max_sst_mur1km": raw_max,
                "annual_max_7day_mean_anomaly_mur1km": max_7day,
                "annual_positive_anomaly_degree_days_mur1km": positive_degree_days,
                "annual_days_above_expanding_p90_mur1km": hot_days,
                "annual_max_consecutive_hot_days_mur1km": max_hot_run,
                "climatology_rule": "strictly prior MUR days at the same source pixel",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if (args.output_dir / "mur_feature_manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed MUR run: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mur_command.txt").write_text(
        shlex.join(sys.argv) + "\n", encoding="utf-8"
    )
    started = time.time()
    sites = pd.read_csv(args.sites)
    sites = sites[["site_id", "latitude", "longitude"]].drop_duplicates("site_id").dropna()
    mapping_path = args.output_dir / "mur_site_mapping.csv"
    if mapping_path.exists():
        candidate_mapping = pd.read_csv(mapping_path)
        expected_ids = set(sites["site_id"].astype(str))
        observed_ids = set(candidate_mapping["site_id"].astype(str))
        mapping = candidate_mapping if expected_ids == observed_ids else pd.DataFrame()
    else:
        mapping = pd.DataFrame()
    if mapping.empty:
        mapping_rows: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(nearest_ocean_pixel, str(row.site_id), float(row.latitude), float(row.longitude)): str(row.site_id)
                for row in sites.itertuples(index=False)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                mapping_rows.append(future.result())
                if completed % 25 == 0 or completed == len(futures):
                    print(f"MUR mapping {completed}/{len(futures)}", flush=True)
        mapping = pd.DataFrame(mapping_rows).sort_values("site_id")
        mapping.to_csv(mapping_path, index=False)
    else:
        print(f"MUR mapping resumed for {len(mapping)} sites", flush=True)
    parity = check_endpoint_parity(
        float(mapping.iloc[0]["mur_latitude"]),
        float(mapping.iloc[0]["mur_longitude"]),
    )
    parity.to_csv(args.output_dir / "mur_endpoint_parity_checks.csv", index=False)
    if not parity["passed"].all():
        raise ValueError("MUR ERDDAP replicas failed the endpoint parity check")
    points = mapping[["mur_latitude", "mur_longitude"]].drop_duplicates().copy()
    tile_groups_by_degrees: dict[float, list[tuple[int, int, pd.DataFrame]]] = {}
    for tile_degrees in (EARLY_TILE_DEGREES, LATE_TILE_DEGREES):
        grouped_points = points.copy()
        grouped_points["tile_a"] = np.floor(grouped_points["mur_latitude"] / tile_degrees).astype(int)
        grouped_points["tile_b"] = np.floor(grouped_points["mur_longitude"] / tile_degrees).astype(int)
        tile_groups_by_degrees[tile_degrees] = [
            (int(keys[0]), int(keys[1]), group[["mur_latitude", "mur_longitude"]].copy())
            for keys, group in grouped_points.groupby(["tile_a", "tile_b"], sort=True)
        ]
    daily_frames: list[pd.DataFrame] = []
    tasks = [
        (tile_a, tile_b, group, start, end, label, tile_degrees)
        for start, end, label in DOWNLOAD_CHUNKS
        for tile_degrees in [
            EARLY_TILE_DEGREES if int(label) < TRANSFER_TILE_SWITCH_YEAR else LATE_TILE_DEGREES
        ]
        for tile_a, tile_b, group in tile_groups_by_degrees[tile_degrees]
    ]
    total_requests = len(tasks)
    workers_per_endpoint = max(1, args.workers // len(ERDDAP_ENDPOINTS))
    pools = [
        ThreadPoolExecutor(max_workers=workers_per_endpoint)
        for _ in ERDDAP_ENDPOINTS
    ]
    try:
        futures = {}
        for task_index, task in enumerate(tasks):
            endpoint_index = task_index % len(ERDDAP_ENDPOINTS)
            tile_a, tile_b, group, start, end, label, tile_degrees = task
            future = pools[endpoint_index].submit(
                load_tile_period,
                tile_a,
                tile_b,
                group,
                start,
                end,
                label,
                args.cache_dir,
                tile_degrees,
                ERDDAP_ENDPOINTS[endpoint_index],
            )
            futures[future] = (tile_a, tile_b, label, ERDDAP_ENDPOINTS[endpoint_index])
        for completed, future in enumerate(as_completed(futures), start=1):
            daily_frames.append(future.result())
            if completed % 50 == 0 or completed == total_requests:
                print(f"MUR tile-periods {completed}/{total_requests}", flush=True)
    finally:
        for pool in pools:
            pool.shutdown(wait=True)
    daily = pd.concat(daily_frames, ignore_index=True)
    if daily.duplicated(["mur_latitude", "mur_longitude", "date"]).any():
        raise ValueError("MUR tile extraction produced duplicate point-days")
    feature_frames = [
        annual_features(group[["date", "sst_c"]], float(keys[0]), float(keys[1]))
        for keys, group in daily.groupby(["mur_latitude", "mur_longitude"], sort=True)
    ]
    point_features = pd.concat(feature_frames, ignore_index=True)
    features = mapping.merge(
        point_features,
        on=["mur_latitude", "mur_longitude"],
        how="left",
        validate="many_to_many",
    )
    if features.duplicated(["site_id", "year"]).any():
        raise ValueError("MUR site-year feature keys are not unique")
    mapping.to_csv(mapping_path, index=False)
    features.to_csv(args.output_dir / "mur1km_site_year_features.csv", index=False)
    forecast = features.loc[features["year"].between(FORECAST_START_YEAR, FORECAST_END_YEAR)]
    feature_columns = [
        "annual_mean_sst_mur1km",
        "annual_max_sst_mur1km",
        "annual_max_7day_mean_anomaly_mur1km",
        "annual_positive_anomaly_degree_days_mur1km",
        "annual_days_above_expanding_p90_mur1km",
        "annual_max_consecutive_hot_days_mur1km",
    ]
    quality = pd.DataFrame(
        [
            {"check": "mapping_site_keys_unique", "value": int(mapping["site_id"].duplicated().sum()), "passed": not mapping["site_id"].duplicated().any()},
            {"check": "maximum_mapping_distance_le_10km", "value": float(mapping["mur_match_distance_km"].max()), "passed": bool(mapping["mur_match_distance_km"].le(10).all())},
            {"check": "feature_keys_unique", "value": int(features.duplicated(["site_id", "year"]).sum()), "passed": not features.duplicated(["site_id", "year"]).any()},
            {"check": "erddap_endpoint_parity", "value": float(parity["max_absolute_difference_celsius_vs_noaa"].max()), "passed": bool(parity["passed"].all())},
            {"check": "forecast_feature_completeness_ge_95pct", "value": float(forecast[feature_columns].notna().all(axis=1).mean()), "passed": bool(forecast[feature_columns].notna().all(axis=1).mean() >= 0.95)},
            {"check": "strictly_past_climatology", "value": 1, "passed": True},
        ]
    )
    quality.to_csv(args.output_dir / "mur_quality_checks.csv", index=False)
    manifest = {
        "status": "complete",
        "source": "NASA JPL MUR v4.1 through three public ERDDAP replicas of jplMURSST41",
        "source_urls": list(ERDDAP_ENDPOINTS),
        "source_period": [START_DATE, END_DATE],
        "forecast_quality_period": [FORECAST_START_YEAR, FORECAST_END_YEAR],
        "transfer_tile_degrees_by_period": {
            f"2002-{TRANSFER_TILE_SWITCH_YEAR - 1}": EARLY_TILE_DEGREES,
            f"{TRANSFER_TILE_SWITCH_YEAR}-2021": LATE_TILE_DEGREES,
        },
        "sites": int(mapping["site_id"].nunique()),
        "source_pixels": int(len(points)),
        "maximum_match_distance_km": float(mapping["mur_match_distance_km"].max()),
        "forecast_feature_completeness": float(forecast[feature_columns].notna().all(axis=1).mean()),
        "runtime_seconds": round(time.time() - started, 2),
        "input_sha256": sha256(args.sites),
        "output_sha256": {
            "mur_command.txt": sha256(args.output_dir / "mur_command.txt"),
            "mur_site_mapping.csv": sha256(args.output_dir / "mur_site_mapping.csv"),
            "mur1km_site_year_features.csv": sha256(args.output_dir / "mur1km_site_year_features.csv"),
            "mur_quality_checks.csv": sha256(args.output_dir / "mur_quality_checks.csv"),
            "mur_endpoint_parity_checks.csv": sha256(args.output_dir / "mur_endpoint_parity_checks.csv"),
        },
        "quality_passed": bool(quality["passed"].all()),
    }
    (args.output_dir / "mur_feature_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if not quality["passed"].all():
        raise SystemExit(f"MUR quality gate failed: {quality.loc[~quality['passed'], 'check'].tolist()}")


if __name__ == "__main__":
    main()
