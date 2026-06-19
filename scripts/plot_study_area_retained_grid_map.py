#!/usr/bin/env python3
"""Plot Figure 1: study area and retained 10 km Kelpwatch grid cells.

This script uses the actual fishnet and filtering outputs committed in this
repository. It does not generate, move, or approximate grid cells.

Input files:
- geometries/regular_10km_fishnet/kelpwatch_regular_10km_fishnet_preview.geojson
  Candidate 10 km fishnet polygons generated for the Northern and Central
  California coastal corridor.
- geometries/regular_10km_fishnet/filtered_cells_historic_footprint_gt0.csv
  Exploratory retained cells with count_cells_historic_footprint > 0.
- geometries/regular_10km_fishnet/filtered_cells_historic_footprint_ge500.csv
  Main-analysis retained cells with count_cells_historic_footprint >= 500.

Natural Earth land/coastline data are downloaded to a temporary directory only
as a cartographic background. The grid geometry and selection counts are read
from repository files.
"""

from __future__ import annotations

import tempfile
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
FISHNET_GEOJSON = (
    ROOT
    / "geometries"
    / "regular_10km_fishnet"
    / "kelpwatch_regular_10km_fishnet_preview.geojson"
)
FOOTPRINT_GT0_CSV = (
    ROOT
    / "geometries"
    / "regular_10km_fishnet"
    / "filtered_cells_historic_footprint_gt0.csv"
)
FOOTPRINT_GE500_CSV = (
    ROOT
    / "geometries"
    / "regular_10km_fishnet"
    / "filtered_cells_historic_footprint_ge500.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "maps"
PNG_OUT = OUTPUT_DIR / "figure1_study_area_retained_10km_grid_cells.png"
PDF_OUT = OUTPUT_DIR / "figure1_study_area_retained_10km_grid_cells.pdf"

PROJECT_CRS = "EPSG:3310"  # California Albers, useful for local map scale.

NE_LAND_URL = "https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_land.zip"
NE_COAST_URL = "https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_coastline.zip"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def download_natural_earth(url: str, tmpdir: Path) -> Path | None:
    """Download a small Natural Earth zip to a temporary path for background use."""
    out = tmpdir / Path(url).name
    try:
        urllib.request.urlretrieve(url, out)
    except Exception as exc:  # pragma: no cover - cartographic fallback only.
        print(f"Warning: could not download {url}: {exc}")
        return None
    return out


def read_natural_earth(zip_path: Path | None) -> gpd.GeoDataFrame | None:
    if zip_path is None:
        return None
    try:
        gdf = gpd.read_file(f"zip://{zip_path}")
    except Exception as exc:  # pragma: no cover - cartographic fallback only.
        print(f"Warning: could not read {zip_path}: {exc}")
        return None
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def load_grid_and_filters() -> tuple[gpd.GeoDataFrame, set[str], set[str]]:
    """Load candidate, exploratory, and main-analysis cells and verify counts."""
    for path in [FISHNET_GEOJSON, FOOTPRINT_GT0_CSV, FOOTPRINT_GE500_CSV]:
        require_file(path)

    grid = gpd.read_file(FISHNET_GEOJSON)
    if "cell_id" not in grid.columns:
        raise ValueError(f"{FISHNET_GEOJSON} must contain a cell_id column.")
    if grid.crs is None:
        grid = grid.set_crs("EPSG:4326")

    footprint_gt0 = pd.read_csv(FOOTPRINT_GT0_CSV)
    footprint_ge500 = pd.read_csv(FOOTPRINT_GE500_CSV)
    for label, df, path in [
        ("footprint > 0", footprint_gt0, FOOTPRINT_GT0_CSV),
        ("footprint >= 500", footprint_ge500, FOOTPRINT_GE500_CSV),
    ]:
        if "cell_id" not in df.columns:
            raise ValueError(f"{path} must contain a cell_id column for {label}.")

    candidate_ids = set(grid["cell_id"].astype(str))
    gt0_ids = set(footprint_gt0["cell_id"].astype(str))
    ge500_ids = set(footprint_ge500["cell_id"].astype(str))

    missing_gt0 = sorted(gt0_ids - candidate_ids)
    missing_ge500 = sorted(ge500_ids - candidate_ids)
    if missing_gt0 or missing_ge500:
        raise ValueError(
            "Filtered cell CSVs contain IDs not present in the candidate grid: "
            f"gt0={missing_gt0[:5]}, ge500={missing_ge500[:5]}"
        )
    if not ge500_ids.issubset(gt0_ids):
        raise ValueError("Main retained cells (>=500) should be a subset of >0 cells.")

    grid["cell_id"] = grid["cell_id"].astype(str)
    grid["selection_group"] = "candidate"
    grid.loc[grid["cell_id"].isin(gt0_ids), "selection_group"] = "footprint_gt0"
    grid.loc[grid["cell_id"].isin(ge500_ids), "selection_group"] = "retained_ge500"
    return grid, gt0_ids, ge500_ids


def add_scale_bar(ax, length_km: int = 50) -> None:
    """Add a simple scale bar in projected coordinates."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bar_len = length_km * 1000
    x = x0 + 0.07 * (x1 - x0)
    y = y0 + 0.06 * (y1 - y0)
    ax.plot([x, x + bar_len], [y, y], color="#222222", linewidth=1.6)
    ax.plot([x, x], [y - 2500, y + 2500], color="#222222", linewidth=1.0)
    ax.plot(
        [x + bar_len, x + bar_len],
        [y - 2500, y + 2500],
        color="#222222",
        linewidth=1.0,
    )
    ax.text(
        x + bar_len / 2,
        y + 6000,
        f"{length_km} km",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#222222",
    )


def add_north_arrow(ax) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x1 - 0.08 * (x1 - x0)
    y = y1 - 0.27 * (y1 - y0)
    ax.annotate(
        "N",
        xy=(x, y + 28000),
        xytext=(x, y),
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#222222", linewidth=1.2),
    )


def plot_map() -> tuple[int, int, int]:
    grid, gt0_ids, ge500_ids = load_grid_and_filters()
    candidate_count = len(grid)
    gt0_count = len(gt0_ids)
    ge500_count = len(ge500_ids)

    grid_projected = grid.to_crs(PROJECT_CRS)
    all_cells = grid_projected
    gt0_cells = grid_projected[grid_projected["cell_id"].isin(gt0_ids)]
    retained_cells = grid_projected[grid_projected["cell_id"].isin(ge500_ids)]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        land = read_natural_earth(download_natural_earth(NE_LAND_URL, tmpdir))
        coast = read_natural_earth(download_natural_earth(NE_COAST_URL, tmpdir))
        if land is not None:
            land = land.to_crs(PROJECT_CRS)
        if coast is not None:
            coast = coast.to_crs(PROJECT_CRS)

        fig, ax = plt.subplots(figsize=(8.0, 8.7))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#F7FAFC")

        minx, miny, maxx, maxy = grid_projected.total_bounds
        x_margin = 55000
        y_margin = 35000
        extent = (minx - x_margin, maxx + x_margin, miny - y_margin, maxy + y_margin)

        if land is not None:
            land.cx[extent[0] : extent[1], extent[2] : extent[3]].plot(
                ax=ax,
                facecolor="#F1F1EC",
                edgecolor="none",
                zorder=0,
            )
        if coast is not None:
            coast.cx[extent[0] : extent[1], extent[2] : extent[3]].plot(
                ax=ax,
                color="#7B8078",
                linewidth=0.45,
                zorder=1,
            )

        all_cells.plot(
            ax=ax,
            facecolor="#E7E9ED",
            edgecolor="#B8BDC7",
            linewidth=0.35,
            alpha=0.78,
            zorder=2,
        )
        gt0_cells.plot(
            ax=ax,
            facecolor="#88B7D5",
            edgecolor="#2F5F7E",
            linewidth=0.45,
            alpha=0.88,
            zorder=3,
        )
        retained_cells.plot(
            ax=ax,
            facecolor="#173B63",
            edgecolor="#071D34",
            linewidth=0.58,
            alpha=0.98,
            zorder=4,
        )

        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.axis("off")

        legend_items = [
            Patch(
                facecolor="#E7E9ED",
                edgecolor="#B8BDC7",
                label=f"Candidate fishnet cells (n={candidate_count})",
            ),
            Patch(
                facecolor="#88B7D5",
                edgecolor="#2F5F7E",
                label=f"Historical footprint > 0 (n={gt0_count})",
            ),
            Patch(
                facecolor="#173B63",
                edgecolor="#071D34",
                label=f"Main retained cells >=500 pixels (n={ge500_count})",
            ),
        ]
        ax.legend(
            handles=legend_items,
            loc="center left",
            bbox_to_anchor=(1.02, 0.42),
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="#C8C8C8",
            fontsize=8.5,
            title="Cell selection",
            title_fontsize=9,
        )

        annotation = (
            "Verified counts\n"
            f"Candidate cells: {candidate_count}\n"
            f"Footprint cells: {gt0_count}\n"
            f"Retained cells: {ge500_count}"
        )
        ax.text(
            0.025,
            0.065,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            linespacing=1.35,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="#BFC4CB",
                linewidth=0.7,
                alpha=0.96,
            ),
            zorder=5,
        )

        add_scale_bar(ax)
        add_north_arrow(ax)

        # Optional locator inset. It uses Natural Earth land only as a reference
        # background and marks the actual grid extent with a rectangle.
        if land is not None:
            inset = ax.inset_axes([0.70, 0.735, 0.24, 0.205])
            inset.set_facecolor("#F7FAFC")
            land_ll = land.to_crs("EPSG:4326")
            grid_ll = grid.to_crs("EPSG:4326")
            land_ll.cx[-128:-113, 31:43].plot(
                ax=inset,
                facecolor="#F1F1EC",
                edgecolor="#9A9A92",
                linewidth=0.28,
            )
            gx0, gy0, gx1, gy1 = grid_ll.total_bounds
            inset.add_patch(
                Rectangle(
                    (gx0, gy0),
                    gx1 - gx0,
                    gy1 - gy0,
                    fill=False,
                    edgecolor="#173B63",
                    linewidth=1.1,
                )
            )
            inset.set_xlim(-126.5, -113.5)
            inset.set_ylim(31.5, 42.5)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_color("#BFC4CB")
                spine.set_linewidth(0.6)
            inset.set_title("California context", fontsize=7.5, pad=2)

        fig.savefig(PNG_OUT, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(PDF_OUT, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return candidate_count, gt0_count, ge500_count


def main() -> None:
    candidate_count, gt0_count, ge500_count = plot_map()
    print("Verified counts:")
    print(f"  Candidate fishnet cells: {candidate_count}")
    print(f"  Historical footprint > 0 cells: {gt0_count}")
    print(f"  Main retained cells >=500 pixels: {ge500_count}")
    print("Output files:")
    print(f"  {PNG_OUT}")
    print(f"  {PDF_OUT}")


if __name__ == "__main__":
    main()
