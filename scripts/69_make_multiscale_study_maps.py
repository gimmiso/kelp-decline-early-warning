"""Create coastwide 5-km and California local-support study-unit maps."""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
import pandas as pd
from pyproj import Transformer
import requests
from shapely.geometry import box


NATURAL_EARTH_URL = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
REGION_COLORS = {
    "Baja California Sur": "#8C6D9E",
    "Baja California Norte": "#6F4C8B",
    "Southern California": "#D08C3C",
    "Central California": "#C85C4A",
    "Northern California": "#3D7C78",
    "Oregon": "#4C78A8",
    "Washington outer coast": "#2F4B7C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coast-panel", type=Path, required=True)
    parser.add_argument("--coast-output", type=Path, required=True)
    parser.add_argument("--california-sites", type=Path)
    parser.add_argument("--california-output", type=Path)
    parser.add_argument(
        "--cartography-cache",
        type=Path,
        default=Path("data/external/cartography/natural_earth_admin0_10m"),
    )
    return parser.parse_args()


def country_boundaries(cache: Path) -> gpd.GeoDataFrame:
    shapefile = cache / "ne_10m_admin_0_countries.shp"
    if not shapefile.exists():
        response = requests.get(NATURAL_EARTH_URL, timeout=(20, 180))
        response.raise_for_status()
        cache.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            archive.extractall(cache)
    countries = gpd.read_file(shapefile)
    name_column = "ADMIN" if "ADMIN" in countries else "NAME"
    return countries.loc[countries[name_column].isin(["United States of America", "Mexico", "Canada"])].to_crs(4326)


def make_coast_map(panel_path: Path, output: Path, countries: gpd.GeoDataFrame) -> None:
    cells = pd.read_csv(panel_path, usecols=["cell_id", "region_group", "center_lat", "center_lon"]).drop_duplicates("cell_id")
    transformer = Transformer.from_crs(4326, 6933, always_xy=True)
    x, y = transformer.transform(cells["center_lon"].to_numpy(), cells["center_lat"].to_numpy())
    half = 2_500
    geometry = [box(xv - half, yv - half, xv + half, yv + half) for xv, yv in zip(x, y, strict=True)]
    grid = gpd.GeoDataFrame(cells, geometry=geometry, crs=6933).to_crs(4326)
    figure, axis = plt.subplots(figsize=(7.5, 10.2), constrained_layout=True)
    axis.set_facecolor("#F6F8FA")
    countries.plot(ax=axis, facecolor="#ECE8DF", edgecolor="#A9A9A9", linewidth=0.45, zorder=1)
    for region, group in grid.groupby("region_group", sort=False):
        group.plot(
            ax=axis,
            facecolor=REGION_COLORS.get(region, "#777777"),
            edgecolor="white",
            linewidth=0.18,
            alpha=0.92,
            zorder=2,
        )
    handles = [Patch(facecolor=color, edgecolor="none", label=region) for region, color in REGION_COLORS.items() if region in set(grid["region_group"])]
    axis.legend(handles=handles, title="Monitoring region", loc="upper right", frameon=False, fontsize=8, title_fontsize=9)
    axis.set_xlim(-127.3, -112.7)
    axis.set_ylim(25.7, 50.0)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(color="#D7DCE0", linewidth=0.4, alpha=0.7)
    axis.set_title(f"Coastwide 5-km monitoring units (n={len(grid)})", loc="left", fontsize=14, pad=10)
    axis.text(0.01, 0.012, "Fixed 5-km equal-area units; boundaries are analytical, not ecological.", transform=axis.transAxes, fontsize=7.5, color="#555B61", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5})
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def make_california_map(metadata_path: Path, output: Path, countries: gpd.GeoDataFrame) -> None:
    metadata = pd.read_csv(metadata_path)
    stable_flag = metadata["stable_support"].astype(str).str.lower().eq("true")
    stable = metadata.loc[stable_flag].copy()
    all_sites = metadata.drop_duplicates("site_id")
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 8.0), constrained_layout=True, gridspec_kw={"width_ratios": [0.95, 1.05]})
    map_axis, inset_axis = axes
    map_axis.set_facecolor("#F6F8FA")
    countries.plot(ax=map_axis, facecolor="#ECE8DF", edgecolor="#A9A9A9", linewidth=0.45, zorder=1)
    map_axis.scatter(all_sites["longitude"], all_sites["latitude"], s=9, color="#A7ADB2", alpha=0.7, label="Candidate field site", zorder=2)
    for support_m, color, marker, size in [(1000, "#D49A32", "o", 32), (300, "#2F5D7C", ".", 24)]:
        part = stable.loc[stable["support_m"].eq(support_m)]
        map_axis.scatter(part["longitude"], part["latitude"], s=size, facecolors="none" if support_m == 1000 else color, edgecolors=color, marker=marker, linewidths=0.8, label=f"Stable {support_m:,}-m support (n={part['site_id'].nunique()})", zorder=3)
    map_axis.set_xlim(-125.0, -116.4)
    map_axis.set_ylim(31.8, 42.4)
    map_axis.set_xlabel("Longitude")
    map_axis.set_ylabel("Latitude")
    map_axis.grid(color="#D7DCE0", linewidth=0.4, alpha=0.7)
    map_axis.legend(loc="lower left", frameon=False, fontsize=8)
    map_axis.set_title("California Giraldo field-site coverage", loc="left", fontsize=13)

    one_km = stable.loc[stable["support_m"].eq(1000)].copy()
    cluster = one_km["overlap_cluster"].value_counts().index[0]
    cluster_sites = one_km.loc[one_km["overlap_cluster"].eq(cluster)].copy()
    transformer = Transformer.from_crs(4326, 6933, always_xy=True)
    cluster_sites["x"], cluster_sites["y"] = transformer.transform(cluster_sites["longitude"].to_numpy(), cluster_sites["latitude"].to_numpy())
    x_reference = float(cluster_sites["x"].mean())
    y_reference = float(cluster_sites["y"].mean())
    cluster_sites["x_local"] = cluster_sites["x"] - x_reference
    cluster_sites["y_local"] = cluster_sites["y"] - y_reference
    three_hundred_ids = set(stable.loc[stable["support_m"].eq(300), "site_id"])
    for row in cluster_sites.itertuples(index=False):
        inset_axis.add_patch(Circle((row.x_local, row.y_local), 1000, facecolor="#D49A32", edgecolor="#B77816", alpha=0.16, linewidth=1.0))
        if row.site_id in three_hundred_ids:
            inset_axis.add_patch(Circle((row.x_local, row.y_local), 300, facecolor="#2F5D7C", edgecolor="#21435B", alpha=0.32, linewidth=0.9))
        inset_axis.scatter(row.x_local, row.y_local, s=18, color="#22272B", zorder=4)
    padding = 1_300
    inset_axis.set_xlim(cluster_sites["x_local"].min() - padding, cluster_sites["x_local"].max() + padding)
    inset_axis.set_ylim(cluster_sites["y_local"].min() - padding, cluster_sites["y_local"].max() + padding)
    inset_axis.set_aspect("equal")
    inset_axis.set_xlabel("Local equal-area easting (m)")
    inset_axis.set_ylabel("Local equal-area northing (m)")
    inset_axis.grid(color="#D7DCE0", linewidth=0.45, alpha=0.7)
    inset_axis.set_title(f"Actual overlapping buffers in example cluster ({len(cluster_sites)} sites)", loc="left", fontsize=11)
    inset_axis.legend(handles=[Patch(facecolor="#2F5D7C", alpha=0.32, label="300-m support"), Patch(facecolor="#D49A32", alpha=0.16, label="1-km support")], frameon=False, loc="lower left", fontsize=8)
    figure.suptitle("California local ecological case-study units", x=0.02, ha="left", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    countries = country_boundaries(args.cartography_cache)
    make_coast_map(args.coast_panel, args.coast_output, countries)
    if args.california_sites is not None:
        if args.california_output is None:
            raise ValueError("--california-output is required with --california-sites")
        make_california_map(args.california_sites, args.california_output, countries)
    print(args.coast_output)
    if args.california_output is not None:
        print(args.california_output)


if __name__ == "__main__":
    main()
