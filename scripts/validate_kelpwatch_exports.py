"""Validate local Kelpwatch cell-level CSV exports."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = {
    "year",
    "quarter",
    "kelp_area_m2",
    "count_cells_kelp",
    "count_cells_no_clouds",
    "count_cells_historic_footprint",
}
DEFAULT_CELLS = ("001", "002", "003")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Validate Kelpwatch raw CSV exports.")
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("data/raw/kelpwatch_aoi"),
        help="Directory containing kelpwatch_cell_XXX.csv files.",
    )
    parser.add_argument(
        "--cells",
        nargs="+",
        default=list(DEFAULT_CELLS),
        help="Cell numbers or IDs to validate, for example 001 002 003 or cell_001.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all kelpwatch_cell_*.csv files in the CSV directory.",
    )
    return parser.parse_args()


def normalize_cell_id(cell: str) -> str:
    """Return a canonical cell_id like cell_001."""
    value = cell.strip()
    if value.startswith("cell_"):
        return value
    return f"cell_{int(value):03d}"


def expected_csv_paths(csv_dir: Path, cells: list[str], include_all: bool) -> list[Path]:
    """Return CSV paths to validate."""
    if include_all:
        return sorted(csv_dir.glob("kelpwatch_cell_*.csv"))
    return [csv_dir / f"kelpwatch_{normalize_cell_id(cell)}.csv" for cell in cells]


def validate_csv(path: Path) -> tuple[str, str]:
    """Validate one CSV and return status plus error message."""
    if not path.exists():
        return "missing", "CSV file does not exist."
    if path.stat().st_size == 0:
        return "failed", "CSV file is empty."

    try:
        with path.open(newline="") as file:
            reader = csv.DictReader(file)
            columns = set(reader.fieldnames or [])
            missing_columns = sorted(REQUIRED_COLUMNS - columns)
            if missing_columns:
                return "failed", f"Missing required columns: {missing_columns}"
            rows = list(reader)
    except Exception as exc:
        return "failed", str(exc)

    if not rows:
        return "failed", "CSV has a header but no data rows."
    if not any(row.get("quarter") == "max" for row in rows):
        return "failed", "CSV does not contain a quarter=max row."
    return "ok", ""


def main() -> None:
    """Run validation checks."""
    args = parse_args()
    paths = expected_csv_paths(args.csv_dir, args.cells, args.all)
    if not paths:
        raise SystemExit("No CSV files selected for validation.")

    failures = []
    for path in paths:
        status, error = validate_csv(path)
        print(f"{path}: {status}{' - ' + error if error else ''}")
        if status != "ok":
            failures.append((path, status, error))

    if failures:
        raise SystemExit(f"{len(failures)} CSV export(s) failed validation.")
    print(f"Validated {len(paths)} Kelpwatch CSV export(s).")


if __name__ == "__main__":
    main()
