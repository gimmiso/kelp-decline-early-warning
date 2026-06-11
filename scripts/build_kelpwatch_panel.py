"""Build an annual Kelpwatch canopy panel for retained 10 km fishnet cells."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean, median


REQUIRED_COLUMNS = {
    "year",
    "quarter",
    "kelp_area_m2",
    "count_cells_kelp",
    "count_cells_no_clouds",
    "count_cells_historic_footprint",
}

DEFAULT_FILTERED_CELLS = Path(
    "geometries/regular_10km_fishnet/filtered_cells_historic_footprint_ge500.csv"
)
DEFAULT_RAW_DIR = Path("data/raw/kelpwatch_aoi")
DEFAULT_PANEL_OUTPUT = Path("data/processed/kelpwatch_panel_ge500.csv")
DEFAULT_SUMMARY_OUTPUT = Path("outputs/metadata/kelpwatch_panel_ge500_summary.csv")


PANEL_COLUMNS = [
    "cell_id",
    "region_group",
    "center_lat",
    "center_lon",
    "year",
    "quarter",
    "kelp_area_m2",
    "count_cells_kelp",
    "count_cells_no_clouds",
    "count_cells_historic_footprint",
    "historical_footprint_area_m2",
    "relative_canopy",
    "source_csv_file",
]

SUMMARY_COLUMNS = [
    "cell_id",
    "region_group",
    "center_lat",
    "center_lon",
    "n_panel_years",
    "first_panel_year",
    "last_panel_year",
    "historical_footprint_cells",
    "historical_footprint_area_m2",
    "mean_kelp_area_m2",
    "max_kelp_area_m2",
    "median_kelp_area_m2",
    "mean_relative_canopy",
    "max_relative_canopy",
    "n_years_positive_kelp",
    "mean_count_cells_no_clouds",
    "panel_rows_expected",
    "current_year_used_for_exclusion",
    "current_and_future_years_excluded",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Construct an annual Kelpwatch panel from retained fishnet cells."
    )
    parser.add_argument(
        "--filtered-cells",
        type=Path,
        default=DEFAULT_FILTERED_CELLS,
        help="Filtered main modeling cell list.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory containing raw kelpwatch_cell_XXX.csv files.",
    )
    parser.add_argument(
        "--panel-output",
        type=Path,
        default=DEFAULT_PANEL_OUTPUT,
        help="Output annual panel CSV path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="Output cell-level metadata summary CSV path.",
    )
    parser.add_argument(
        "--current-year",
        type=int,
        default=date.today().year,
        help="Current year. Rows from this year and later are excluded by default.",
    )
    parser.add_argument(
        "--keep-current-year",
        action="store_true",
        help="Keep current/future years instead of dropping them as incomplete.",
    )
    return parser.parse_args()


def to_float(value: str | None) -> float:
    """Parse a numeric CSV value, treating blanks as zero."""
    if value in {None, ""}:
        return 0.0
    return float(value)


def to_int(value: str | None) -> int:
    """Parse an integer-like CSV value, treating blanks as zero."""
    return int(to_float(value))


def format_value(value: object) -> str:
    """Format values for stable CSV output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.8f}".rstrip("0").rstrip(".")
    return str(value)


def expected_csv_name(cell_id: str) -> str:
    """Return the expected raw Kelpwatch CSV name for a cell_id."""
    suffix = cell_id.replace("cell_", "")
    return f"kelpwatch_cell_{suffix}.csv"


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read CSV rows and return fieldnames."""
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        rows = list(reader)
    return columns, rows


def read_filtered_cells(path: Path) -> list[dict[str, str]]:
    """Read and validate the retained main modeling cell list."""
    columns, rows = read_csv_rows(path)
    if "cell_id" not in columns:
        raise ValueError(f"{path} is missing required column: cell_id")
    if not rows:
        raise ValueError(f"{path} contains no retained cells.")

    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        cell_id = row["cell_id"]
        if cell_id in seen:
            duplicates.append(cell_id)
        seen.add(cell_id)
    if duplicates:
        raise ValueError(f"Duplicate cell_id values found: {sorted(set(duplicates))}")

    return rows


def read_cell_panel_rows(
    cell_row: dict[str, str],
    raw_dir: Path,
    current_year: int,
    keep_current_year: bool,
) -> list[dict[str, object]]:
    """Read one raw Kelpwatch CSV and return annual max rows with derived fields."""
    cell_id = cell_row["cell_id"]
    csv_name = cell_row.get("kelpwatch_csv_file") or expected_csv_name(cell_id)
    csv_path = raw_dir / csv_name
    columns, rows = read_csv_rows(csv_path)

    missing = sorted(REQUIRED_COLUMNS - set(columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    panel_rows: list[dict[str, object]] = []
    for row in rows:
        quarter = str(row.get("quarter", "")).strip().lower()
        if quarter != "max":
            continue

        year = to_int(row.get("year"))
        if not keep_current_year and year >= current_year:
            continue

        footprint_cells = to_int(row.get("count_cells_historic_footprint"))
        footprint_area = footprint_cells * 900
        kelp_area = to_float(row.get("kelp_area_m2"))
        relative_canopy = kelp_area / footprint_area if footprint_area > 0 else None

        panel_rows.append(
            {
                "cell_id": cell_id,
                "region_group": cell_row.get("region_group", ""),
                "center_lat": cell_row.get("center_lat", ""),
                "center_lon": cell_row.get("center_lon", ""),
                "year": year,
                "quarter": "max",
                "kelp_area_m2": kelp_area,
                "count_cells_kelp": to_int(row.get("count_cells_kelp")),
                "count_cells_no_clouds": to_int(row.get("count_cells_no_clouds")),
                "count_cells_historic_footprint": footprint_cells,
                "historical_footprint_area_m2": footprint_area,
                "relative_canopy": relative_canopy,
                "source_csv_file": csv_name,
            }
        )

    return panel_rows


def build_panel(
    cells: list[dict[str, str]],
    raw_dir: Path,
    current_year: int,
    keep_current_year: bool,
) -> list[dict[str, object]]:
    """Build a clean annual panel from raw Kelpwatch cell CSV files."""
    panel: list[dict[str, object]] = []
    for cell_row in cells:
        panel.extend(read_cell_panel_rows(cell_row, raw_dir, current_year, keep_current_year))

    panel.sort(key=lambda row: (str(row["cell_id"]), int(row["year"])))
    if not panel:
        raise ValueError("No panel rows were produced.")
    return panel


def build_summary(
    panel: list[dict[str, object]],
    cells: list[dict[str, str]],
    current_year: int,
    keep_current_year: bool,
) -> list[dict[str, object]]:
    """Create cell-level metadata summary for the constructed panel."""
    rows_by_cell: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in panel:
        rows_by_cell[str(row["cell_id"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for cell in cells:
        cell_id = cell["cell_id"]
        rows = rows_by_cell.get(cell_id, [])
        years = [int(row["year"]) for row in rows]
        kelp_values = [float(row["kelp_area_m2"]) for row in rows]
        relative_values = [
            float(row["relative_canopy"])
            for row in rows
            if row.get("relative_canopy") not in {None, ""}
        ]
        no_cloud_values = [float(row["count_cells_no_clouds"]) for row in rows]
        footprint_cells = max([int(row["count_cells_historic_footprint"]) for row in rows], default="")
        footprint_area = max([int(row["historical_footprint_area_m2"]) for row in rows], default="")

        summary_rows.append(
            {
                "cell_id": cell_id,
                "region_group": cell.get("region_group", ""),
                "center_lat": cell.get("center_lat", ""),
                "center_lon": cell.get("center_lon", ""),
                "n_panel_years": len(set(years)),
                "first_panel_year": min(years) if years else "",
                "last_panel_year": max(years) if years else "",
                "historical_footprint_cells": footprint_cells,
                "historical_footprint_area_m2": footprint_area,
                "mean_kelp_area_m2": mean(kelp_values) if kelp_values else "",
                "max_kelp_area_m2": max(kelp_values) if kelp_values else "",
                "median_kelp_area_m2": median(kelp_values) if kelp_values else "",
                "mean_relative_canopy": mean(relative_values) if relative_values else "",
                "max_relative_canopy": max(relative_values) if relative_values else "",
                "n_years_positive_kelp": sum(1 for value in kelp_values if value > 0),
                "mean_count_cells_no_clouds": mean(no_cloud_values) if no_cloud_values else "",
                "panel_rows_expected": "yes" if rows else "no",
                "current_year_used_for_exclusion": current_year,
                "current_and_future_years_excluded": not keep_current_year,
            }
        )

    return summary_rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: format_value(row.get(column)) for column in columns})


def main() -> None:
    """Run the Kelpwatch annual panel construction workflow."""
    args = parse_args()
    cells = read_filtered_cells(args.filtered_cells)
    panel = build_panel(cells, args.raw_dir, args.current_year, args.keep_current_year)
    summary = build_summary(panel, cells, args.current_year, args.keep_current_year)

    write_csv(args.panel_output, PANEL_COLUMNS, panel)
    write_csv(args.summary_output, SUMMARY_COLUMNS, summary)

    years = [int(row["year"]) for row in panel]
    print("Kelpwatch annual canopy panel construction complete.")
    print(f"Retained cells: {len(cells)}")
    print(f"Panel rows: {len(panel)}")
    print(f"Panel years: {min(years)}-{max(years)}")
    print(f"Panel output: {args.panel_output}")
    print(f"Summary output: {args.summary_output}")


if __name__ == "__main__":
    main()
