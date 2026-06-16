"""Build V2 multi-scale OISST exposure features from local NOAA cache.

This script adds a new analysis layer without replacing the Version 1 nearest
grid workflow. It treats cached OISST grid cells as environmental point
supports, buffers Kelpwatch cell centroids in a projected CRS, and aggregates
annual exposure summaries at multiple spatial supports.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from train_model_comparison import INPUT_DATASET, main_subset


DEFAULT_CACHE_DIR = Path("data/external/noaa/cache/oisst")
DEFAULT_OUTPUT = Path("data/processed/multiscale_environmental_features.csv")
DEFAULT_REPORT = Path("outputs/diagnostics/multiscale_environmental_feature_report.md")
DEFAULT_SUMMARY = Path("outputs/metadata/multiscale_environmental_feature_summary.csv")

BASELINE_START_YEAR = 1984
BASELINE_END_YEAR = 2013
MODEL_START_YEAR = 1989
MODEL_END_YEAR = 2024
PROJECTED_CRS = "EPSG:3310"
GEOGRAPHIC_CRS = "EPSG:4326"
SCALES_KM = [10, 25, 30, 50, 75]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build multi-scale OISST exposure features.")
    parser.add_argument("--input", type=Path, default=INPUT_DATASET)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--scales-km", type=int, nargs="+", default=SCALES_KM)
    return parser.parse_args()


def parse_oisst_filename(path: Path) -> tuple[float, float] | None:
    """Extract OISST point latitude and longitude from a cached CSV name."""
    match = re.search(r"oisst_lat(?P<lat>[-m0-9.]+)_lon(?P<lon>[-m0-9.]+)_daily\.csv$", path.name)
    if not match:
        return None
    lat = float(match.group("lat").replace("m", "-"))
    lon = float(match.group("lon").replace("m", "-"))
    return round(lat, 3), round(lon, 3)


def read_cached_oisst(cache_dir: Path) -> pd.DataFrame:
    """Read all cached daily OISST point series and attach coordinates."""
    rows = []
    for path in sorted(cache_dir.glob("oisst_lat*_lon*_daily.csv")):
        coords = parse_oisst_filename(path)
        if coords is None:
            continue
        lat, lon = coords
        daily = pd.read_csv(path, parse_dates=["time"])
        if "sst" not in daily.columns:
            continue
        daily["sst"] = pd.to_numeric(daily["sst"], errors="coerce")
        daily = daily.dropna(subset=["sst"])
        daily["year"] = pd.to_datetime(daily["time"], utc=True).dt.year.astype(int)
        daily["oisst_lat"] = lat
        daily["oisst_lon"] = lon
        rows.append(daily[["oisst_lat", "oisst_lon", "year", "sst"]])
    if not rows:
        raise FileNotFoundError(f"No cached daily OISST CSV files found in {cache_dir}")
    return pd.concat(rows, ignore_index=True)


def annual_point_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Convert daily OISST point series to annual exposure summaries."""
    thresholds = (
        daily.loc[daily["year"].between(BASELINE_START_YEAR, BASELINE_END_YEAR)]
        .groupby(["oisst_lat", "oisst_lon"])["sst"]
        .quantile([0.90, 0.95])
        .unstack()
        .rename(columns={0.90: "sst_p90_baseline", 0.95: "sst_p95_baseline"})
        .reset_index()
    )
    annual = (
        daily.groupby(["oisst_lat", "oisst_lon", "year"])["sst"]
        .agg(annual_mean_sst="mean", annual_max_sst="max", annual_min_sst="min", annual_sst_std="std")
        .reset_index()
    )
    annual = annual.merge(thresholds, on=["oisst_lat", "oisst_lon"], how="left")
    hot_counts = daily.merge(thresholds, on=["oisst_lat", "oisst_lon"], how="left")
    hot_counts["hot_days_p90"] = hot_counts["sst"] > hot_counts["sst_p90_baseline"]
    hot_counts["hot_days_p95"] = hot_counts["sst"] > hot_counts["sst_p95_baseline"]
    hot_counts = (
        hot_counts.groupby(["oisst_lat", "oisst_lon", "year"])[["hot_days_p90", "hot_days_p95"]]
        .sum()
        .reset_index()
    )
    annual = annual.merge(hot_counts, on=["oisst_lat", "oisst_lon", "year"], how="left")
    baseline = (
        annual.loc[annual["year"].between(BASELINE_START_YEAR, BASELINE_END_YEAR)]
        .groupby(["oisst_lat", "oisst_lon"])[["annual_mean_sst", "annual_max_sst"]]
        .mean()
        .rename(columns={"annual_mean_sst": "baseline_mean_sst", "annual_max_sst": "baseline_max_sst"})
        .reset_index()
    )
    annual = annual.merge(baseline, on=["oisst_lat", "oisst_lon"], how="left")
    annual["annual_mean_sst_anomaly"] = annual["annual_mean_sst"] - annual["baseline_mean_sst"]
    annual["annual_max_sst_anomaly"] = annual["annual_max_sst"] - annual["baseline_max_sst"]
    return annual.drop(columns=["sst_p90_baseline", "sst_p95_baseline", "baseline_mean_sst", "baseline_max_sst"])


def kelp_cells(data: pd.DataFrame) -> pd.DataFrame:
    """Return unique Kelpwatch cell centroids from the modeling dataset."""
    required = {"cell_id", "center_lat", "center_lon", "region_group"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing cell metadata columns: {missing}")
    return data[["cell_id", "region_group", "center_lat", "center_lon"]].drop_duplicates().reset_index(drop=True)


def point_geodata(cells: pd.DataFrame, points: pd.DataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create projected GeoDataFrames for cell and OISST point centroids."""
    cell_gdf = gpd.GeoDataFrame(
        cells,
        geometry=gpd.points_from_xy(cells["center_lon"], cells["center_lat"]),
        crs=GEOGRAPHIC_CRS,
    ).to_crs(PROJECTED_CRS)
    point_meta = points[["oisst_lat", "oisst_lon"]].drop_duplicates().reset_index(drop=True)
    point_gdf = gpd.GeoDataFrame(
        point_meta,
        geometry=gpd.points_from_xy(point_meta["oisst_lon"], point_meta["oisst_lat"]),
        crs=GEOGRAPHIC_CRS,
    ).to_crs(PROJECTED_CRS)
    return cell_gdf, point_gdf


def nearest_assignments(cell_gdf: gpd.GeoDataFrame, point_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Assign each cell to the nearest cached OISST point."""
    joined = gpd.sjoin_nearest(
        cell_gdf[["cell_id", "geometry"]],
        point_gdf[["oisst_lat", "oisst_lon", "geometry"]],
        how="left",
        distance_col="distance_m",
    )
    return joined[["cell_id", "oisst_lat", "oisst_lon", "distance_m"]].rename(
        columns={"distance_m": "oisst_nearest_distance_km"}
    ).assign(oisst_nearest_distance_km=lambda x: x["oisst_nearest_distance_km"] / 1000.0)


def buffer_memberships(
    cell_gdf: gpd.GeoDataFrame,
    point_gdf: gpd.GeoDataFrame,
    scales_km: list[int],
) -> pd.DataFrame:
    """Return cell-to-OISST point memberships for each buffer radius."""
    rows = []
    for scale in scales_km:
        buffered = cell_gdf[["cell_id", "geometry"]].copy()
        buffered["geometry"] = buffered.geometry.buffer(scale * 1000)
        joined = gpd.sjoin(
            point_gdf[["oisst_lat", "oisst_lon", "geometry"]],
            buffered,
            how="inner",
            predicate="within",
        )
        for row in joined.itertuples():
            rows.append(
                {
                    "scale_km": scale,
                    "cell_id": row.cell_id,
                    "oisst_lat": row.oisst_lat,
                    "oisst_lon": row.oisst_lon,
                }
            )
    return pd.DataFrame(rows)


def aggregate_exposures(
    annual: pd.DataFrame,
    nearest: pd.DataFrame,
    memberships: pd.DataFrame,
    scales_km: list[int],
) -> pd.DataFrame:
    """Aggregate annual point features for nearest and buffer supports."""
    nearest_features = annual.merge(nearest, on=["oisst_lat", "oisst_lon"], how="inner")
    nearest_grouped = nearest_features.groupby(["cell_id", "year"], as_index=False).agg(
        oisst_nearest_annual_mean_sst_mean=("annual_mean_sst", "mean"),
        oisst_nearest_annual_max_sst_max=("annual_max_sst", "max"),
        oisst_nearest_annual_mean_sst_anomaly_mean=("annual_mean_sst_anomaly", "mean"),
        oisst_nearest_annual_max_sst_anomaly_max=("annual_max_sst_anomaly", "max"),
        oisst_nearest_hot_days_p90_mean=("hot_days_p90", "mean"),
        oisst_nearest_hot_days_p90_max=("hot_days_p90", "max"),
        oisst_nearest_hot_days_p95_mean=("hot_days_p95", "mean"),
        oisst_nearest_hot_days_p95_max=("hot_days_p95", "max"),
        oisst_nearest_n_grid_points=("oisst_lat", "nunique"),
        oisst_nearest_distance_km=("oisst_nearest_distance_km", "first"),
    )
    output = nearest_grouped.copy()
    for scale in scales_km:
        members = memberships.loc[memberships["scale_km"] == scale]
        joined = members.merge(annual, on=["oisst_lat", "oisst_lon"], how="inner")
        prefix = f"oisst_{scale}km"
        grouped = joined.groupby(["cell_id", "year"], as_index=False).agg(
            **{
                f"{prefix}_annual_mean_sst_mean": ("annual_mean_sst", "mean"),
                f"{prefix}_annual_max_sst_max": ("annual_max_sst", "max"),
                f"{prefix}_annual_mean_sst_anomaly_mean": ("annual_mean_sst_anomaly", "mean"),
                f"{prefix}_annual_max_sst_anomaly_max": ("annual_max_sst_anomaly", "max"),
                f"{prefix}_hot_days_p90_mean": ("hot_days_p90", "mean"),
                f"{prefix}_hot_days_p90_max": ("hot_days_p90", "max"),
                f"{prefix}_hot_days_p95_mean": ("hot_days_p95", "mean"),
                f"{prefix}_hot_days_p95_max": ("hot_days_p95", "max"),
                f"{prefix}_n_grid_points": ("oisst_lat", "nunique"),
            }
        )
        output = output.merge(grouped, on=["cell_id", "year"], how="left")
    return output.sort_values(["cell_id", "year"]).reset_index(drop=True)


def feature_summary(features: pd.DataFrame, scales_km: list[int]) -> pd.DataFrame:
    """Summarize feature coverage by spatial scale."""
    rows = []
    scale_names = ["nearest", *[f"{scale}km" for scale in scales_km]]
    for scale in scale_names:
        point_col = f"oisst_{scale}_n_grid_points"
        feature_cols = [column for column in features.columns if column.startswith(f"oisst_{scale}_")]
        rows.append(
            {
                "scale": scale,
                "feature_columns": len(feature_cols),
                "rows": len(features),
                "rows_with_any_missing": int(features[feature_cols].isna().any(axis=1).sum()) if feature_cols else 0,
                "mean_grid_points": float(features[point_col].mean()) if point_col in features else np.nan,
                "min_grid_points": float(features[point_col].min()) if point_col in features else np.nan,
                "max_grid_points": float(features[point_col].max()) if point_col in features else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_report(output: Path, summary: pd.DataFrame, cache_dir: Path, scales_km: list[int]) -> None:
    """Write a concise construction report."""
    lines = [
        "# Multi-Scale Environmental Feature Construction Report",
        "",
        "## Purpose",
        "",
        "This V2 layer constructs OISST exposure variables at multiple spatial supports around each retained Kelpwatch cell. It keeps the Version 1 nearest-grid assignment as the baseline and adds buffer-based summaries to evaluate support mismatch.",
        "",
        "## Implementation",
        "",
        f"- OISST cache directory: `{cache_dir}`.",
        f"- Buffer scales: `{', '.join(str(scale) + ' km' for scale in scales_km)}`.",
        f"- Distance operations use projected CRS `{PROJECTED_CRS}` rather than degree buffers.",
        "- OISST cached grid cells are treated as point supports at their grid centroids.",
        "- CUTI and BEUTI remain latitude-bin proxies in Version 1 and are not converted to radial buffer supports here.",
        "",
        "## Feature Coverage",
        "",
        "| scale | feature_columns | rows | rows_with_any_missing | mean_grid_points | min_grid_points | max_grid_points |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.scale} | {row.feature_columns} | {row.rows} | {row.rows_with_any_missing} | "
            f"{row.mean_grid_points:.2f} | {row.min_grid_points:.0f} | {row.max_grid_points:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "Buffer aggregation reduces sensitivity to a single nearest OISST grid cell, but it does not create true nearshore in-situ temperature. The current local cache contains OISST points previously needed by the Version 1 workflow, so this V2 output should be interpreted as a reproducible multi-scale prototype. A publication-grade run should cache all OISST grid points intersecting each candidate buffer.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Build multi-scale OISST exposure features."""
    args = parse_args()
    data = main_subset(pd.read_csv(args.input).sort_values(["cell_id", "year"]).reset_index(drop=True))
    cells = kelp_cells(data)
    daily = read_cached_oisst(args.cache_dir)
    annual = annual_point_features(daily)
    annual = annual.loc[annual["year"].between(BASELINE_START_YEAR, MODEL_END_YEAR)].copy()
    cell_gdf, point_gdf = point_geodata(cells, annual)
    nearest = nearest_assignments(cell_gdf, point_gdf)
    memberships = buffer_memberships(cell_gdf, point_gdf, args.scales_km)
    features = aggregate_exposures(annual, nearest, memberships, args.scales_km)
    features = features.loc[features["year"].between(BASELINE_START_YEAR, MODEL_END_YEAR)].copy()
    summary = feature_summary(features, args.scales_km)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    summary.to_csv(args.summary_output, index=False)
    write_report(args.report_output, summary, args.cache_dir, args.scales_km)

    print(f"Wrote multi-scale features: {args.output}")
    print(f"Wrote feature summary: {args.summary_output}")
    print(f"Wrote report: {args.report_output}")


if __name__ == "__main__":
    main()
