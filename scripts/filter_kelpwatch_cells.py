"""Filter regular 10 km Kelpwatch fishnet cells by historical kelp footprint."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


EXPECTED_CANDIDATE_CELLS = 285
EXPECTED_EXPLORATORY_CELLS = 74
EXPECTED_MAIN_CELLS = 50
REQUIRED_SUMMARY_COLUMNS = {
    "cell_id",
    "region_group",
    "kelpwatch_csv_exists",
    "kelpwatch_has_required_columns",
    "kelpwatch_has_quarter_max",
    "historic_footprint_cells",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter Kelpwatch fishnet cells by historical kelp footprint."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "geometries/regular_10km_fishnet/"
            "aoi_inventory_regular_10km_fishnet_with_kelpwatch_summary.csv"
        ),
        help="Kelpwatch summary inventory CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("geometries/regular_10km_fishnet"),
        help="Directory for filtered outputs.",
    )
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    """Parse summary boolean values."""
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_int(value: str) -> int:
    """Parse integer-like values from CSV."""
    if value in {"", None}:
        return 0
    return int(float(value))


def read_summary(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read and validate the summary inventory."""
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames or []
        rows = list(reader)

    missing = sorted(REQUIRED_SUMMARY_COLUMNS - set(columns))
    if missing:
        raise ValueError(f"Summary file is missing required columns: {missing}")
    if len(rows) != EXPECTED_CANDIDATE_CELLS:
        raise ValueError(f"Expected 285 candidate cells, found {len(rows)}.")

    invalid = [
        row["cell_id"]
        for row in rows
        if not (
            parse_bool(row["kelpwatch_csv_exists"])
            and parse_bool(row["kelpwatch_has_required_columns"])
            and parse_bool(row["kelpwatch_has_quarter_max"])
        )
    ]
    if invalid:
        raise ValueError(f"Cells with invalid Kelpwatch summary status: {invalid}")

    return columns, rows


def add_filter_flags(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Add exploratory and main filter boolean flags."""
    enriched = []
    for row in rows:
        footprint = parse_int(row["historic_footprint_cells"])
        updated = dict(row)
        updated["passes_exploratory_filter_gt0"] = "true" if footprint > 0 else "false"
        updated["passes_main_filter_ge500"] = "true" if footprint >= 500 else "false"
        enriched.append(updated)
    return enriched


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def count_by_region(rows: list[dict[str, str]], flag_column: str) -> dict[str, int]:
    """Count passing rows by region group."""
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if parse_bool(row[flag_column]):
            counts[row["region_group"]] += 1
    return dict(sorted(counts.items()))


def write_report(path: Path, rows: list[dict[str, str]], exploratory: list[dict[str, str]], main: list[dict[str, str]]) -> None:
    """Write a Markdown filtering report."""
    exploratory_counts = count_by_region(rows, "passes_exploratory_filter_gt0")
    main_counts = count_by_region(rows, "passes_main_filter_ge500")

    regions = sorted({row["region_group"] for row in rows})
    lines = [
        "# Kelpwatch Fishnet Cell Filtering Report",
        "",
        "## Summary",
        "",
        f"Total candidate cells: {len(rows)}",
        f"Cells with historical footprint > 0: {len(exploratory)}",
        f"Cells with historical footprint >= 500: {len(main)}",
        "",
        "## Counts by Region Group",
        "",
    ]
    for region in regions:
        lines.extend(
            [
                f"### {region}",
                "",
                f"- footprint > 0: {exploratory_counts.get(region, 0)}",
                f"- footprint >= 500: {main_counts.get(region, 0)}",
                "",
            ]
        )

    lines.extend(
        [
            "## Filtering Logic",
            "",
            "The regular 10 km fishnet generated 285 candidate cells across the Northern and Central California coastal corridor. After querying Kelpwatch, many candidate cells reported zero historical kelp footprint, indicating that no kelp canopy was observed within those cells during the historical observation period. These cells are excluded before modeling.",
            "",
            "The exploratory dataset retains all cells with `count_cells_historic_footprint > 0`. The main modeling dataset applies a stricter threshold of `count_cells_historic_footprint >= 500`, following the logic of previous Kelpwatch-based studies that excluded 10 km cells with very low potential kelp habitat.",
            "",
            "## Outputs",
            "",
            "- `aoi_inventory_regular_10km_fishnet_with_filters.csv`",
            "- `filtered_cells_historic_footprint_gt0.csv`",
            "- `filtered_cells_historic_footprint_ge500.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run the filtering workflow."""
    args = parse_args()
    columns, rows = read_summary(args.summary)
    enriched = add_filter_flags(rows)

    enriched_columns = list(columns)
    for column in ["passes_exploratory_filter_gt0", "passes_main_filter_ge500"]:
        if column not in enriched_columns:
            enriched_columns.append(column)

    exploratory = [row for row in enriched if parse_bool(row["passes_exploratory_filter_gt0"])]
    main = [row for row in enriched if parse_bool(row["passes_main_filter_ge500"])]

    if len(exploratory) != EXPECTED_EXPLORATORY_CELLS:
        raise ValueError(f"Expected 74 exploratory cells, found {len(exploratory)}.")
    if len(main) != EXPECTED_MAIN_CELLS:
        raise ValueError(f"Expected 50 main modeling cells, found {len(main)}.")

    output_dir = args.output_dir
    write_csv(output_dir / "aoi_inventory_regular_10km_fishnet_with_filters.csv", enriched_columns, enriched)
    write_csv(output_dir / "filtered_cells_historic_footprint_gt0.csv", enriched_columns, exploratory)
    write_csv(output_dir / "filtered_cells_historic_footprint_ge500.csv", enriched_columns, main)
    write_report(output_dir / "cell_filtering_report.md", enriched, exploratory, main)

    print("Kelpwatch fishnet cell filtering complete.")
    print(f"Total candidate cells: {len(enriched)}")
    print(f"Cells with historical footprint > 0: {len(exploratory)}")
    print(f"Cells with historical footprint >= 500: {len(main)}")


if __name__ == "__main__":
    main()
