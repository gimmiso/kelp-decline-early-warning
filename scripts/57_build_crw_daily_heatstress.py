"""Download daily Q1 CoralTemp and build four locked heat-stress features.

The 165-cell mapping is reused verbatim from the monthly CRW experiment. 1985
Q1 is extracted from NOAA STAR daily files because the PaciOOS ERDDAP mirror
starts on 1985-04-01. Years 1986-2024 are downloaded as small latitude-band
subsets from the mirror. A 2021 daily file is independently extracted from the
official archive and compared with the mirror before features are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from netCDF4 import Dataset, num2date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_dates(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year}-03-31", freq="D")


def source_points(mapping: pd.DataFrame) -> pd.DataFrame:
    points = (
        mapping[["crw_lat", "crw_lon", "lat_index", "lon_index"]]
        .drop_duplicates(["crw_lat", "crw_lon"])
        .sort_values(["crw_lat", "crw_lon"], ascending=[False, True])
        .reset_index(drop=True)
    )
    points.insert(0, "source_point_id", [f"crw_{i:03d}" for i in range(len(points))])
    points["band_id"] = np.floor((points["crw_lat"] - 24.0) / 3.0).astype(int)
    return points


def get_to_path(url: str, output: Path, attempts: int = 4) -> tuple[int, float]:
    last_error: Exception | None = None
    started = time.monotonic()
    for attempt in range(attempts):
        try:
            with requests.get(url, timeout=(30, 360), stream=True) as response:
                response.raise_for_status()
                with output.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            return output.stat().st_size, time.monotonic() - started
        except (requests.RequestException, OSError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Download failed after {attempts} attempts: {url}: {last_error}")


def erddap_url(year: int, band: pd.DataFrame, root: str) -> str:
    lat_max = float(band["crw_lat"].max())
    lat_min = float(band["crw_lat"].min())
    lon_min = float(band["crw_lon"].min())
    lon_max = float(band["crw_lon"].max())
    end = date(year, 3, 31).isoformat()
    return (
        f"{root}.nc?CRW_SST"
        f"[({year}-01-01T12:00:00Z):1:({end}T12:00:00Z)]"
        f"[({lat_max:.3f}):1:({lat_min:.3f})]"
        f"[({lon_min:.3f}):1:({lon_max:.3f})]"
    )


def complete_cache(path: Path, points: pd.DataFrame, year: int) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, usecols=["source_point_id", "date"])
    except Exception:
        return False
    return (
        not frame.duplicated(["source_point_id", "date"]).any()
        and len(frame) == len(points) * len(expected_dates(year))
    )


def extract_erddap_year_band(
    year: int,
    band_id: int,
    band: pd.DataFrame,
    cache_path: Path,
    root: str,
    force: bool,
) -> dict[str, object]:
    if not force and complete_cache(cache_path, band, year):
        return {
            "route": "erddap_band_subset",
            "year": year,
            "band_id": band_id,
            "status": "cached",
            "cache_path": str(cache_path),
            "source_url": erddap_url(year, band, root),
            "bytes": cache_path.stat().st_size,
            "elapsed_seconds": 0.0,
        }
    url = erddap_url(year, band, root)
    with tempfile.TemporaryDirectory(prefix="kelp_crw_erddap_") as temp_dir:
        netcdf_path = Path(temp_dir) / f"crw_{year}_{band_id}.nc"
        size, elapsed = get_to_path(url, netcdf_path)
        with Dataset(netcdf_path) as dataset:
            lat = np.asarray(dataset.variables["latitude"][:], dtype=float)
            lon = np.asarray(dataset.variables["longitude"][:], dtype=float)
            tvar = dataset.variables["time"]
            timestamps = num2date(
                tvar[:],
                units=tvar.units,
                calendar=getattr(tvar, "calendar", "standard"),
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )
            dates = pd.to_datetime([item.isoformat() for item in timestamps]).normalize()
            cube = np.ma.filled(dataset.variables["CRW_SST"][:], np.nan).astype(float)
            rows: list[pd.DataFrame] = []
            for point in band.itertuples(index=False):
                i = int(np.argmin(np.abs(lat - float(point.crw_lat))))
                j = int(np.argmin(np.abs(lon - float(point.crw_lon))))
                if abs(float(lat[i]) - float(point.crw_lat)) > 0.001 or abs(float(lon[j]) - float(point.crw_lon)) > 0.001:
                    raise ValueError(f"ERDDAP grid mismatch for {point.source_point_id}")
                rows.append(
                    pd.DataFrame(
                        {
                            "source_point_id": point.source_point_id,
                            "date": dates,
                            "sst_c": cube[:, i, j],
                            "source_route": "pacioos_erddap_dhw_5km",
                        }
                    )
                )
    out = pd.concat(rows, ignore_index=True)
    calendar_frame = pd.MultiIndex.from_product(
        [band["source_point_id"], expected_dates(year)], names=["source_point_id", "date"]
    ).to_frame(index=False)
    out = calendar_frame.merge(out, on=["source_point_id", "date"], how="left", validate="one_to_one")
    out["source_route"] = out["source_route"].fillna("pacioos_erddap_dhw_5km")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return {
        "route": "erddap_band_subset",
        "year": year,
        "band_id": band_id,
        "status": "downloaded",
        "cache_path": str(cache_path),
        "source_url": url,
        "bytes": size,
        "elapsed_seconds": elapsed,
    }


def official_url(day: pd.Timestamp, root: str) -> str:
    stamp = day.strftime("%Y%m%d")
    return f"{root}/{day.year}/coraltemp_v3.1_{stamp}.nc"


def extract_official_date(
    day: pd.Timestamp,
    points: pd.DataFrame,
    cache_path: Path,
    root: str,
    variable: str,
    force: bool,
) -> dict[str, object]:
    if not force and cache_path.exists():
        cached = pd.read_csv(cache_path)
        if len(cached) == len(points) and not cached["source_point_id"].duplicated().any():
            return {
                "route": "official_star_daily",
                "year": int(day.year),
                "band_id": "all_points",
                "date": day.strftime("%Y-%m-%d"),
                "status": "cached",
                "cache_path": str(cache_path),
                "source_url": official_url(day, root),
                "bytes": cache_path.stat().st_size,
                "elapsed_seconds": 0.0,
            }
    url = official_url(day, root)
    with tempfile.TemporaryDirectory(prefix="kelp_crw_star_") as temp_dir:
        netcdf_path = Path(temp_dir) / Path(url).name
        size, elapsed = get_to_path(url, netcdf_path)
        with Dataset(netcdf_path) as dataset:
            cube_var = dataset.variables[variable]
            lat = np.asarray(dataset.variables["lat"][:], dtype=float)
            lon = np.asarray(dataset.variables["lon"][:], dtype=float)
            # Monthly CRW files store latitude north-to-south, whereas the
            # official daily files store it south-to-north. The locked source
            # coordinates are authoritative; array indices are product-local.
            lat_indices = np.asarray(
                [int(np.argmin(np.abs(lat - value))) for value in points["crw_lat"]],
                dtype=int,
            )
            lon_indices = np.asarray(
                [int(np.argmin(np.abs(lon - value))) for value in points["crw_lon"]],
                dtype=int,
            )
            lat0, lat1 = int(lat_indices.min()), int(lat_indices.max()) + 1
            lon0, lon1 = int(lon_indices.min()), int(lon_indices.max()) + 1
            subset = np.ma.filled(cube_var[0, lat0:lat1, lon0:lon1], np.nan).astype(float)
            values = subset[lat_indices - lat0, lon_indices - lon0]
            if np.max(np.abs(lat[lat_indices] - points["crw_lat"].to_numpy())) > 0.001:
                raise ValueError("Official latitude coordinates do not match locked source points")
            if np.max(np.abs(lon[lon_indices] - points["crw_lon"].to_numpy())) > 0.001:
                raise ValueError("Official longitude coordinates do not match locked source points")
    out = points[["source_point_id"]].copy()
    out["date"] = day.strftime("%Y-%m-%d")
    out["sst_c"] = values
    out["source_route"] = "noaa_star_official_daily"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache_path, index=False)
    return {
        "route": "official_star_daily",
        "year": int(day.year),
        "band_id": "all_points",
        "date": day.strftime("%Y-%m-%d"),
        "status": "downloaded",
        "cache_path": str(cache_path),
        "source_url": url,
        "bytes": size,
        "elapsed_seconds": elapsed,
    }


def max_consecutive_hot(valid: pd.Series, hot: pd.Series) -> int:
    longest = 0
    current = 0
    for is_valid, is_hot in zip(valid.to_numpy(), hot.to_numpy()):
        if bool(is_valid) and bool(is_hot):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def build_features(daily: pd.DataFrame, points: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = daily.merge(points[["source_point_id", "crw_lat", "crw_lon"]], on="source_point_id", how="left", validate="many_to_one")
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    daily["month_day"] = daily["date"].dt.strftime("%m-%d")
    baseline = daily.loc[daily["year"].between(1985, 2004) & daily["sst_c"].notna()].copy()
    climatology = (
        baseline.loc[baseline["month_day"].ne("02-29")]
        .groupby(["source_point_id", "month_day"], as_index=False)["sst_c"]
        .mean()
        .rename(columns={"sst_c": "calendar_day_climatology_c"})
    )
    leap = (
        climatology.loc[climatology["month_day"].isin(["02-28", "03-01"])]
        .groupby("source_point_id", as_index=False)["calendar_day_climatology_c"]
        .mean()
    )
    leap["month_day"] = "02-29"
    climatology = pd.concat([climatology, leap], ignore_index=True)
    thresholds = (
        baseline.groupby("source_point_id", as_index=False)["sst_c"]
        .quantile(0.90)
        .rename(columns={"sst_c": "local_q1_p90_c"})
    )
    daily = daily.merge(climatology, on=["source_point_id", "month_day"], how="left", validate="many_to_one")
    daily = daily.merge(thresholds, on="source_point_id", how="left", validate="many_to_one")
    daily["anomaly_c"] = daily["sst_c"] - daily["calendar_day_climatology_c"]
    daily["positive_anomaly_c"] = daily["anomaly_c"].clip(lower=0)
    daily["hot_day"] = daily["sst_c"].gt(daily["local_q1_p90_c"])
    daily["valid"] = daily["sst_c"].notna()
    daily = daily.sort_values(["source_point_id", "year", "date"])
    daily["rolling_7day_mean_anomaly_c"] = (
        daily.groupby(["source_point_id", "year"], sort=False)["anomaly_c"]
        .rolling(window=7, min_periods=7)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    rows: list[dict[str, object]] = []
    for (point_id, year), group in daily.groupby(["source_point_id", "year"], sort=True):
        expected = len(expected_dates(int(year)))
        n_valid = int(group["valid"].sum())
        passes = n_valid / expected >= 0.95
        rows.append(
            {
                "source_point_id": point_id,
                "year": int(year),
                "q1_expected_days": expected,
                "q1_valid_days": n_valid,
                "q1_daily_coverage": n_valid / expected,
                "q1_daily_coverage_pass": passes,
                "q1_max_7day_mean_anomaly_crw5km": float(group["rolling_7day_mean_anomaly_c"].max()) if passes else np.nan,
                "q1_positive_anomaly_degree_days_crw5km": float(group["positive_anomaly_c"].sum(min_count=1)) if passes else np.nan,
                "q1_days_above_local_p90_crw5km": int(group.loc[group["valid"], "hot_day"].sum()) if passes else np.nan,
                "q1_max_consecutive_hot_days_crw5km": max_consecutive_hot(group["valid"], group["hot_day"]) if passes else np.nan,
            }
        )
    point_features = pd.DataFrame(rows)
    point_features = point_features.merge(thresholds, on="source_point_id", how="left", validate="many_to_one")
    map_points = mapping.merge(points, on=["crw_lat", "crw_lon", "lat_index", "lon_index"], how="left", validate="many_to_one")
    if map_points["cell_id"].duplicated().any():
        raise ValueError("Cell-to-source-point mapping is not unique by cell")
    if point_features.duplicated(["source_point_id", "year"]).any():
        raise ValueError("Daily features are not unique by source pixel and year")
    features = map_points[["cell_id", "source_point_id", "crw_lat", "crw_lon", "distance_to_crw_grid_km"]].merge(
        point_features, on="source_point_id", how="left", validate="many_to_many"
    )
    if features.duplicated(["cell_id", "year"]).any():
        raise ValueError("Expanded daily features are not unique by cell and year")
    profile = (
        features.groupby("year", as_index=False)
        .agg(
            cells=("cell_id", "nunique"),
            source_pixels=("source_point_id", "nunique"),
            mean_daily_coverage=("q1_daily_coverage", "mean"),
            min_daily_coverage=("q1_daily_coverage", "min"),
            passing_cells=("q1_daily_coverage_pass", "sum"),
        )
    )
    return features, daily, profile


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    mapping = pd.read_csv(args.mapping)
    if len(mapping) != 165 or mapping["cell_id"].nunique() != 165:
        raise ValueError("Locked CRW mapping must contain 165 unique cells")
    points = source_points(mapping)
    if len(points) != 164:
        raise ValueError(f"Expected 164 unique source pixels, found {len(points)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    points.to_csv(args.output_dir / "source_points_and_bands.csv", index=False)

    source = config["source"]
    logs: list[dict[str, object]] = []
    jobs = []
    for year in range(1986, 2025):
        for band_id, band in points.groupby("band_id"):
            path = args.cache_dir / "erddap" / f"crw_q1_{year}_band_{int(band_id):02d}.csv"
            jobs.append((year, int(band_id), band.copy(), path))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                extract_erddap_year_band,
                year,
                band_id,
                band,
                path,
                source["erddap_dataset"],
                args.force,
            ): (year, band_id)
            for year, band_id, band, path in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            logs.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(f"ERDDAP subsets complete: {completed}/{len(futures)}", flush=True)

    official_jobs = []
    for day in expected_dates(1985):
        path = args.cache_dir / "official_1985" / f"crw_{day.strftime('%Y%m%d')}.csv"
        official_jobs.append((day, path))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                extract_official_date,
                day,
                points,
                path,
                source["official_daily_root"],
                source["official_variable"],
                args.force,
            ): day
            for day, path in official_jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            logs.append(future.result())
            if completed % 15 == 0 or completed == len(futures):
                print(f"Official 1985 files complete: {completed}/{len(futures)}", flush=True)

    daily_frames = [pd.read_csv(path) for _, _, _, path in jobs]
    daily_frames.extend(pd.read_csv(path) for _, path in official_jobs)
    daily = pd.concat(daily_frames, ignore_index=True)
    daily["date"] = pd.to_datetime(daily["date"]).dt.strftime("%Y-%m-%d")
    if daily.duplicated(["source_point_id", "date"]).any():
        raise ValueError("Duplicate source-point dates in combined daily extraction")

    overlap_day = pd.Timestamp("2021-01-15")
    overlap_cache = args.cache_dir / "overlap" / "crw_20210115_official.csv"
    logs.append(
        extract_official_date(
            overlap_day,
            points,
            overlap_cache,
            source["official_daily_root"],
            source["official_variable"],
            args.force,
        )
    )
    official = pd.read_csv(overlap_cache).rename(columns={"sst_c": "official_sst_c"})
    mirror = daily.loc[daily["date"].eq("2021-01-15"), ["source_point_id", "sst_c"]].rename(columns={"sst_c": "erddap_sst_c"})
    overlap = official[["source_point_id", "official_sst_c"]].merge(mirror, on="source_point_id", how="outer", validate="one_to_one")
    overlap["absolute_difference_c"] = (overlap["official_sst_c"] - overlap["erddap_sst_c"]).abs()
    overlap.to_csv(args.output_dir / "official_erddap_overlap_20210115.csv", index=False)

    features, daily_enriched, profile = build_features(daily, points, mapping)
    feature_path = args.output_dir / "crw_daily_q1_heatstress_features_165.csv"
    features.to_csv(feature_path, index=False)
    profile.to_csv(args.output_dir / "daily_coverage_by_year.csv", index=False)
    pd.DataFrame(logs).sort_values(["route", "year", "band_id"], key=lambda s: s.astype(str)).to_csv(
        args.output_dir / "download_inventory.csv", index=False
    )

    feature_columns = config["daily_feature_lock"]["features"]
    forecast_features = features.loc[features["year"].between(2005, 2024)]
    valid_overlap = overlap.dropna(subset=["official_sst_c", "erddap_sst_c"])
    max_overlap_difference = float(valid_overlap["absolute_difference_c"].max()) if len(valid_overlap) else math.inf
    checks = [
        ("mapping_cell_count_165", len(mapping) == 165),
        ("mapping_unique_source_pixels_164", len(points) == 164),
        ("daily_key_unique", not daily.duplicated(["source_point_id", "date"]).any()),
        ("daily_years_1985_2024", set(pd.to_datetime(daily["date"]).dt.year.unique()) == set(range(1985, 2025))),
        ("overlap_has_at_least_150_valid_pixels", len(valid_overlap) >= 150),
        ("official_erddap_max_abs_difference_le_0_001c", max_overlap_difference <= 0.001),
        ("forecast_feature_rows_3300", len(forecast_features) == 165 * 20),
        ("forecast_daily_coverage_pass_at_least_95pct", float(forecast_features["q1_daily_coverage_pass"].mean()) >= 0.95),
        ("forecast_four_features_complete_at_least_95pct", float(forecast_features[feature_columns].notna().all(axis=1).mean()) >= 0.95),
    ]
    quality = pd.DataFrame(checks, columns=["check", "passed"])
    quality["value"] = ""
    quality.loc[quality["check"].eq("official_erddap_max_abs_difference_le_0_001c"), "value"] = str(max_overlap_difference)
    quality.loc[quality["check"].eq("forecast_daily_coverage_pass_at_least_95pct"), "value"] = str(float(forecast_features["q1_daily_coverage_pass"].mean()))
    quality.loc[quality["check"].eq("forecast_four_features_complete_at_least_95pct"), "value"] = str(float(forecast_features[feature_columns].notna().all(axis=1).mean()))
    quality.to_csv(args.output_dir / "source_quality_checks.csv", index=False)
    metadata = {
        "experiment_id": config["experiment_id"],
        "classification": config["classification"],
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "mapping": str(args.mapping),
        "mapping_sha256": sha256(args.mapping),
        "feature_file": str(feature_path),
        "feature_sha256": sha256(feature_path),
        "daily_rows": len(daily),
        "source_pixels": len(points),
        "cell_year_feature_rows": len(features),
        "official_erddap_overlap_valid_pixels": len(valid_overlap),
        "official_erddap_max_absolute_difference_c": max_overlap_difference,
        "quality_passed": bool(quality["passed"].all()),
        "raw_daily_cache_retained": True,
        "raw_daily_cache_path": str(args.cache_dir),
        "daily_enriched_not_persisted": True,
    }
    (args.output_dir / "daily_feature_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if not quality["passed"].all():
        failed = quality.loc[~quality["passed"], "check"].tolist()
        raise SystemExit(f"Daily source quality gate failed: {failed}")


if __name__ == "__main__":
    main()
