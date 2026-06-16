"""Build V2 multi-scale OISST exposure features from local NOAA cache.

This script adds a new analysis layer without replacing the Version 1 nearest
grid workflow. It treats cached OISST grid cells as environmental point
supports, interpolates OISST exposure to Kelpwatch cell centroids, buffers
Kelpwatch cell centroids in a projected CRS, and aggregates annual exposure
summaries at multiple spatial supports.
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
IDW_K_VALUES = [4, 8]
IDW_POWER = 2.0
POINT_FEATURES = [
    "annual_mean_sst",
    "annual_max_sst",
    "annual_min_sst",
    "annual_sst_std",
    "annual_mean_sst_anomaly",
    "annual_max_sst_anomaly",
    "hot_days_p90",
    "hot_days_p95",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build multi-scale OISST exposure features.")
    parser.add_argument("--input", type=Path, default=INPUT_DATASET)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--scales-km", type=int, nargs="+", default=SCALES_KM)
    parser.add_argument("--idw-k", type=int, nargs="+", default=IDW_K_VALUES)
    parser.add_argument("--idw-power", type=float, default=IDW_POWER)
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


def distance_assignments(cell_gdf: gpd.GeoDataFrame, point_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return distance from every cell centroid to every cached OISST point."""
    rows = []
    point_records = point_gdf[["oisst_lat", "oisst_lon", "geometry"]].to_dict("records")
    for cell in cell_gdf[["cell_id", "geometry"]].itertuples():
        for point in point_records:
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "oisst_lat": point["oisst_lat"],
                    "oisst_lon": point["oisst_lon"],
                    "distance_km": float(cell.geometry.distance(point["geometry"]) / 1000.0),
                }
            )
    return pd.DataFrame(rows)


def idw_memberships(distances: pd.DataFrame, k_values: list[int], power: float) -> pd.DataFrame:
    """Return source-aware IDW memberships for each requested k."""
    rows = []
    for k in k_values:
        for cell_id, group in distances.groupby("cell_id"):
            nearest = group.sort_values("distance_km").head(k).copy()
            zero_distance = nearest["distance_km"] <= 1e-9
            if zero_distance.any():
                weights = np.where(zero_distance, 1.0, 0.0)
            else:
                raw_weights = 1.0 / np.power(nearest["distance_km"].to_numpy(), power)
                weights = raw_weights / raw_weights.sum()
            for (_, row), weight in zip(nearest.iterrows(), weights):
                rows.append(
                    {
                        "method": f"idw_k{k}",
                        "k": k,
                        "cell_id": cell_id,
                        "oisst_lat": row["oisst_lat"],
                        "oisst_lon": row["oisst_lon"],
                        "distance_km": row["distance_km"],
                        "weight": float(weight),
                    }
                )
    return pd.DataFrame(rows)


def bilinear_memberships(cells: pd.DataFrame, point_gdf: gpd.GeoDataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return bilinear memberships only if every cell has four cached vertices."""
    cached_points = {
        (round(float(row.oisst_lat), 3), round(float(row.oisst_lon), 3))
        for row in point_gdf[["oisst_lat", "oisst_lon"]].drop_duplicates().itertuples()
    }
    rows = []
    incomplete_cells = []
    for cell in cells.itertuples():
        lat = float(cell.center_lat)
        lon = float(cell.center_lon)
        lat0 = np.floor((lat + 89.875) / 0.25) * 0.25 - 89.875
        lon0 = np.floor((lon + 179.875) / 0.25) * 0.25 - 179.875
        lat1 = lat0 + 0.25
        lon1 = lon0 + 0.25
        vertices = [
            (round(lat0, 3), round(lon0, 3)),
            (round(lat0, 3), round(lon1, 3)),
            (round(lat1, 3), round(lon0, 3)),
            (round(lat1, 3), round(lon1, 3)),
        ]
        if not all(vertex in cached_points for vertex in vertices):
            incomplete_cells.append(cell.cell_id)
            continue
        lat_fraction = 0.0 if lat1 == lat0 else (lat - lat0) / (lat1 - lat0)
        lon_fraction = 0.0 if lon1 == lon0 else (lon - lon0) / (lon1 - lon0)
        weights = {
            (round(lat0, 3), round(lon0, 3)): (1 - lat_fraction) * (1 - lon_fraction),
            (round(lat0, 3), round(lon1, 3)): (1 - lat_fraction) * lon_fraction,
            (round(lat1, 3), round(lon0, 3)): lat_fraction * (1 - lon_fraction),
            (round(lat1, 3), round(lon1, 3)): lat_fraction * lon_fraction,
        }
        for (point_lat, point_lon), weight in weights.items():
            rows.append(
                {
                    "method": "bilinear",
                    "cell_id": cell.cell_id,
                    "oisst_lat": point_lat,
                    "oisst_lon": point_lon,
                    "weight": float(weight),
                }
            )
    diagnostics = {
        "bilinear_complete": len(incomplete_cells) == 0,
        "bilinear_complete_cells": int(cells["cell_id"].nunique() - len(incomplete_cells)),
        "bilinear_incomplete_cells": len(incomplete_cells),
    }
    if incomplete_cells:
        diagnostics["bilinear_incomplete_examples"] = ",".join(incomplete_cells[:10])
        return pd.DataFrame(), diagnostics
    return pd.DataFrame(rows), diagnostics


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
    idw: pd.DataFrame,
    bilinear: pd.DataFrame,
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
        oisst_nearest_n_grid_points=("oisst_lat", "count"),
        oisst_nearest_distance_km=("oisst_nearest_distance_km", "first"),
    )
    output = nearest_grouped.copy()

    for method in sorted(idw["method"].unique()) if not idw.empty else []:
        members = idw.loc[idw["method"] == method]
        joined = members.merge(annual, on=["oisst_lat", "oisst_lon"], how="inner")
        for feature in POINT_FEATURES:
            joined[f"{feature}_weighted"] = joined[feature] * joined["weight"]
        grouped = joined.groupby(["cell_id", "year"], as_index=False).agg(
            **{
                f"oisst_{method}_{feature}_idw": (f"{feature}_weighted", "sum")
                for feature in POINT_FEATURES
            },
            **{
                f"oisst_{method}_n_grid_points": ("oisst_lat", "count"),
                f"oisst_{method}_max_distance_km": ("distance_km", "max"),
            },
        )
        output = output.merge(grouped, on=["cell_id", "year"], how="left")

    if not bilinear.empty:
        joined = bilinear.merge(annual, on=["oisst_lat", "oisst_lon"], how="inner")
        for feature in POINT_FEATURES:
            joined[f"{feature}_weighted"] = joined[feature] * joined["weight"]
        grouped = joined.groupby(["cell_id", "year"], as_index=False).agg(
            **{
                f"oisst_bilinear_{feature}_interpolated": (f"{feature}_weighted", "sum")
                for feature in POINT_FEATURES
            },
            oisst_bilinear_n_grid_points=("oisst_lat", "count"),
        )
        output = output.merge(grouped, on=["cell_id", "year"], how="left")

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
                f"{prefix}_n_grid_points": ("oisst_lat", "count"),
            }
        )
        output = output.merge(grouped, on=["cell_id", "year"], how="left")
    return output.sort_values(["cell_id", "year"]).reset_index(drop=True)


def feature_summary(
    features: pd.DataFrame,
    scales_km: list[int],
    idw_k_values: list[int],
    bilinear_diagnostics: dict[str, object],
) -> pd.DataFrame:
    """Summarize feature coverage by spatial scale."""
    rows = []
    scale_names = ["nearest", *[f"idw_k{k}" for k in idw_k_values]]
    if bilinear_diagnostics.get("bilinear_complete"):
        scale_names.append("bilinear")
    scale_names.extend([f"{scale}km" for scale in scales_km])
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


def write_report(
    output: Path,
    summary: pd.DataFrame,
    cache_dir: Path,
    scales_km: list[int],
    idw_k_values: list[int],
    bilinear_diagnostics: dict[str, object],
) -> None:
    """Write a concise construction report."""
    lines = [
        "# Multi-Scale Environmental Feature Construction Report",
        "",
        "## Purpose",
        "",
        "This V2 layer constructs source-aware OISST exposure variables around each retained Kelpwatch cell. It keeps the Version 1 nearest-grid assignment as the baseline, adds IDW-interpolated OISST exposure at kelp-cell centroids, and adds broader coastal-neighborhood buffer summaries to evaluate support mismatch.",
        "",
        "## Implementation",
        "",
        f"- OISST cache directory: `{cache_dir}`.",
        "- Nearest-grid assignment is retained as the baseline.",
        f"- IDW interpolation uses k = `{', '.join(map(str, idw_k_values))}` and power = `{IDW_POWER}`.",
        "- IDW is source-aware interpolation from a coarse 0.25-degree gridded SST field to kelp cell centroids; it is not ordinary missing-value imputation and does not create true 10 km SST.",
        f"- Bilinear interpolation included: `{bool(bilinear_diagnostics.get('bilinear_complete'))}`.",
        f"- Bilinear complete cells: `{bilinear_diagnostics.get('bilinear_complete_cells')}`; incomplete cells: `{bilinear_diagnostics.get('bilinear_incomplete_cells')}`.",
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
            "IDW-interpolated OISST exposure at kelp-cell centroids is the main practical interpolation method in this V2 layer because 10 km buffers are under-supported relative to the 0.25-degree OISST grid spacing. Buffer aggregation reduces sensitivity to a single nearest OISST grid cell, but it does not create true nearshore in-situ temperature. The current local cache contains OISST points previously needed by the Version 1 workflow, so this V2 output should be interpreted as a reproducible multi-scale prototype. A publication-grade run should cache all OISST grid points intersecting each candidate buffer.",
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
    distances = distance_assignments(cell_gdf, point_gdf)
    idw = idw_memberships(distances, args.idw_k, args.idw_power)
    bilinear, bilinear_diagnostics = bilinear_memberships(cells, point_gdf)
    memberships = buffer_memberships(cell_gdf, point_gdf, args.scales_km)
    features = aggregate_exposures(annual, nearest, idw, bilinear, memberships, args.scales_km)
    features = features.loc[features["year"].between(BASELINE_START_YEAR, MODEL_END_YEAR)].copy()
    summary = feature_summary(features, args.scales_km, args.idw_k, bilinear_diagnostics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    summary.to_csv(args.summary_output, index=False)
    write_report(args.report_output, summary, args.cache_dir, args.scales_km, args.idw_k, bilinear_diagnostics)

    print(f"Wrote multi-scale features: {args.output}")
    print(f"Wrote feature summary: {args.summary_output}")
    print(f"Wrote report: {args.report_output}")


if __name__ == "__main__":
    main()
