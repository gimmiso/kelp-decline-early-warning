"""Map the 165-cell study area and the 107-cell upwelling support subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pyogrio
from matplotlib.patches import Patch, Polygon as MplPolygon


REGION_ORDER = [
    "Washington outer coast",
    "Oregon",
    "Northern California",
    "Central California",
    "Southern California",
    "Baja California Norte",
    "Baja California Sur",
]
COLORS = {
    "Washington outer coast": "#4C78A8",
    "Oregon": "#72B7B2",
    "Northern California": "#54A24B",
    "Central California": "#ECA82C",
    "Southern California": "#F58518",
    "Baja California Norte": "#B279A2",
    "Baja California Sur": "#9C755F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/processed/kelpwatch_west_coast_panel_165.csv"),
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=Path(
            "outputs/experiments/20260814_minimal_paper_extension_v2/environment_features.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/experiments/20260814_minimal_paper_extension_v2/figure_00_study_area_165_cells.png"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def background_countries() -> gpd.GeoDataFrame | None:
    fixture = (
        Path(pyogrio.__file__).resolve().parent
        / "tests"
        / "fixtures"
        / "naturalearth_lowres"
        / "naturalearth_lowres.shp"
    )
    if not fixture.exists():
        return None
    world = gpd.read_file(fixture)
    return world.loc[world["name"].isin(["Canada", "United States of America", "Mexico"])]


def display_polygon(lon: float, lat: float) -> list[tuple[float, float]]:
    """Return an approximately 10-by-10 km display box around a centroid."""
    half_lat = 5.0 / 111.32
    half_lon = 5.0 / (111.32 * math.cos(math.radians(lat)))
    return [
        (lon - half_lon, lat - half_lat),
        (lon + half_lon, lat - half_lat),
        (lon + half_lon, lat + half_lat),
        (lon - half_lon, lat + half_lat),
    ]


def main() -> None:
    args = parse_args()
    panel = pd.read_csv(args.panel)
    environment = pd.read_csv(args.environment)
    cells = panel[["cell_id", "region_group", "center_lat", "center_lon"]].drop_duplicates()
    support = (
        environment.assign(
            upwelling_supported=environment["upwelling_supported"]
            .astype(str)
            .str.lower()
            .eq("true")
        )
        .groupby("cell_id", as_index=False)["upwelling_supported"]
        .max()
    )
    cells = cells.merge(support, on="cell_id", how="left")
    if len(cells) != 165 or cells["upwelling_supported"].sum() != 107:
        raise ValueError("Expected 165 total cells and 107 upwelling-supported cells")

    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(8.8, 11.5))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.09, top=0.91)
    land = background_countries()
    if land is not None:
        land.plot(ax=ax, facecolor="#F1F0EB", edgecolor="#7B817F", linewidth=0.7, zorder=0)

    for row in cells.itertuples(index=False):
        edge = "#111111" if row.upwelling_supported else "#FFFFFF"
        width = 0.65 if row.upwelling_supported else 0.25
        ax.add_patch(
            MplPolygon(
                display_polygon(float(row.center_lon), float(row.center_lat)),
                closed=True,
                facecolor=COLORS[row.region_group],
                edgecolor=edge,
                linewidth=width,
                alpha=0.92,
                zorder=3,
            )
        )

    counts = cells["region_group"].value_counts()
    region_legend = [
        Patch(
            facecolor=COLORS[region],
            edgecolor="none",
            label=f"{region} (n={int(counts.get(region, 0))})",
        )
        for region in REGION_ORDER
        if counts.get(region, 0) > 0
    ]
    first = ax.legend(
        handles=region_legend,
        title="Study regions",
        loc="center right",
        bbox_to_anchor=(0.99, 0.55),
        fontsize=8.5,
        title_fontsize=9,
        frameon=True,
    )
    ax.add_artist(first)
    support_legend = [
        Patch(facecolor="#B0BEC5", edgecolor="#111111", linewidth=1.2, label="CUTI/BEUTI supported (n=107)"),
        Patch(facecolor="#B0BEC5", edgecolor="#FFFFFF", linewidth=1.2, label="OISST only (n=58)"),
    ]
    ax.legend(handles=support_legend, loc="upper right", fontsize=8.5, frameon=True)

    ax.annotate(
        "N",
        xy=(-114.2, 48.3),
        xytext=(-114.2, 46.8),
        arrowprops={"arrowstyle": "-|>", "color": "#111111", "lw": 1.5},
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_xlim(-127.2, -113.1)
    ax.set_ylim(26.4, 49.2)
    ax.set_aspect(1 / math.cos(math.radians(38)))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "Expanded KelpWatch study area\n"
        "165 nominal 10-km cells; black borders mark valid CUTI/BEUTI support",
        fontsize=15,
        pad=14,
    )
    fig.text(
        0.5,
        0.025,
        "Cell boxes reconstructed for display from reported centroids and nominal 10-km size.\n"
        "Analyses use the stored cell IDs and time series; boxes are not used for spatial intersection.",
        fontsize=7.2,
        color="#555555",
        va="center",
        ha="center",
    )
    ax.grid(color="#D9DDDF", linewidth=0.5, alpha=0.7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, facecolor="white")
    plt.close(fig)

    metadata = {
        "figure": str(args.output),
        "panel": str(args.panel),
        "panel_sha256": sha256(args.panel),
        "environment": str(args.environment),
        "environment_sha256": sha256(args.environment),
        "total_cells": int(len(cells)),
        "upwelling_supported_cells": int(cells["upwelling_supported"].sum()),
        "region_counts": {key: int(value) for key, value in counts.sort_index().items()},
        "geometry_note": "Approximate 10-km display boxes reconstructed from centroids; not analysis geometry.",
        "background": "Natural Earth low-resolution country polygons distributed with pyogrio test fixtures.",
    }
    metadata_path = args.output.with_name("figure_00_study_area_165_cells.metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
