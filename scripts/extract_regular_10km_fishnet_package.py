"""Extract the accepted regular 10 km Kelpwatch fishnet package.

This script does not regenerate or redesign the fishnet grid. The zip package
is treated as the source of truth for the accepted 285-cell AOI design.

Example
-------
python scripts/extract_regular_10km_fishnet_package.py \
    /path/to/kelpwatch_regular_10km_fishnet_package.zip
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory


EXPECTED_CELL_COUNT = 285


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract package-derived regular 10 km fishnet AOI files."
    )
    parser.add_argument(
        "zip_path",
        type=Path,
        help="Path to kelpwatch_regular_10km_fishnet_package.zip.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current working directory.",
    )
    return parser.parse_args()


def copy_package_files(extracted_dir: Path, repo_root: Path) -> None:
    """Copy package files into their canonical repository locations."""
    geometry_dir = repo_root / "geometries" / "regular_10km_fishnet"
    scripts_dir = repo_root / "scripts"
    maps_dir = repo_root / "docs" / "maps"

    geometry_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    for filename in [
        "README_grid_method.md",
        "aoi_inventory_regular_10km_fishnet.csv",
        "grid_validation_regular_10km_fishnet.txt",
        "kelpwatch_regular_10km_fishnet_preview.geojson",
    ]:
        shutil.copy2(extracted_dir / filename, geometry_dir / filename)

    single_cell_dst = geometry_dir / "single_cell_geojsons"
    if single_cell_dst.exists():
        shutil.rmtree(single_cell_dst)
    shutil.copytree(extracted_dir / "single_cell_geojsons", single_cell_dst)

    shutil.copy2(
        extracted_dir / "check_regular_fishnet_distribution.py",
        scripts_dir / "check_regular_fishnet_distribution.py",
    )
    shutil.copy2(
        extracted_dir / "kelpwatch_regular_10km_fishnet_preview_map.html",
        maps_dir / "kelpwatch_regular_10km_fishnet_preview_map.html",
    )


def validate_extracted_design(repo_root: Path) -> None:
    """Validate source-of-truth fishnet outputs after extraction."""
    geometry_dir = repo_root / "geometries" / "regular_10km_fishnet"
    single_cell_dir = geometry_dir / "single_cell_geojsons"

    expected_names = [f"cell_{i:03d}.geojson" for i in range(1, EXPECTED_CELL_COUNT + 1)]
    actual_names = sorted(path.name for path in single_cell_dir.glob("cell_*.geojson"))
    if actual_names != expected_names:
        raise ValueError("Single-cell GeoJSON names are missing, extra, or not continuous.")

    preview_path = geometry_dir / "kelpwatch_regular_10km_fishnet_preview.geojson"
    with preview_path.open() as file:
        preview = json.load(file)
    feature_count = len(preview.get("features", []))
    if feature_count != EXPECTED_CELL_COUNT:
        raise ValueError(f"Preview GeoJSON has {feature_count} features, expected 285.")

    inventory_path = geometry_dir / "aoi_inventory_regular_10km_fishnet.csv"
    with inventory_path.open(newline="") as file:
        row_count = sum(1 for _ in csv.DictReader(file))
    if row_count != EXPECTED_CELL_COUNT:
        raise ValueError(f"AOI inventory has {row_count} rows, expected 285.")

    validation_text = (geometry_dir / "grid_validation_regular_10km_fishnet.txt").read_text()
    if "Overlap pairs with positive area: 0" not in validation_text:
        raise ValueError("Validation file does not confirm zero positive-area overlaps.")


def main() -> None:
    """Extract and validate the accepted package-derived fishnet design."""
    args = parse_args()
    zip_path = args.zip_path.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()

    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with TemporaryDirectory() as temp_dir:
        extracted_dir = Path(temp_dir)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extracted_dir)
        copy_package_files(extracted_dir, repo_root)

    validate_extracted_design(repo_root)
    print("Extracted accepted regular 10 km fishnet AOI package.")
    print("Generated cells: 285")
    print("Cell size: 10 km x 10 km")
    print("Grid type: regular fishnet")
    print("Positive-area overlap: 0")
    print("Study region: Northern and Central California coastal corridor")


if __name__ == "__main__":
    main()
