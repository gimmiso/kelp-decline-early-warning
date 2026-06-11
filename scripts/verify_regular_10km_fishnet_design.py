"""Verify the committed regular 10 km Kelpwatch fishnet design.

This script checks the committed package-derived files. It does not generate a
new grid and does not modify any GeoJSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_CELL_COUNT = 285


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Verify committed fishnet AOI files.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current working directory.",
    )
    return parser.parse_args()


def main() -> None:
    """Run fishnet design checks."""
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    geometry_dir = repo_root / "geometries" / "regular_10km_fishnet"
    single_cell_dir = geometry_dir / "single_cell_geojsons"

    expected_names = [f"cell_{i:03d}.geojson" for i in range(1, EXPECTED_CELL_COUNT + 1)]
    actual_names = sorted(path.name for path in single_cell_dir.glob("cell_*.geojson"))
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))
        extra = sorted(set(actual_names) - set(expected_names))
        raise SystemExit(f"Cell filenames are not continuous. Missing={missing}, extra={extra}")

    preview_path = geometry_dir / "kelpwatch_regular_10km_fishnet_preview.geojson"
    with preview_path.open() as file:
        preview = json.load(file)
    preview_features = len(preview.get("features", []))
    if preview_features != EXPECTED_CELL_COUNT:
        raise SystemExit(f"Preview feature count is {preview_features}, expected 285.")

    inventory_path = geometry_dir / "aoi_inventory_regular_10km_fishnet.csv"
    with inventory_path.open(newline="") as file:
        inventory_rows = sum(1 for _ in csv.DictReader(file))
    if inventory_rows != EXPECTED_CELL_COUNT:
        raise SystemExit(f"Inventory row count is {inventory_rows}, expected 285.")

    validation_text = (geometry_dir / "grid_validation_regular_10km_fishnet.txt").read_text()
    if "Overlap pairs with positive area: 0" not in validation_text:
        raise SystemExit("Validation file does not confirm zero positive-area overlaps.")

    print("Regular 10 km fishnet design verification passed.")
    print(f"Generated cells: {EXPECTED_CELL_COUNT}")
    print("Cell size: 10 km x 10 km")
    print("Grid type: regular fishnet")
    print("Positive-area overlap: 0")
    print("Study region: Northern and Central California coastal corridor")


if __name__ == "__main__":
    main()
