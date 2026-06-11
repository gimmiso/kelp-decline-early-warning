"""Summarize Kelpwatch cell-level CSV exports and augment the AOI inventory."""

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


SUMMARY_COLUMNS = [
    "kelpwatch_csv_exists",
    "kelpwatch_csv_rows",
    "kelpwatch_has_required_columns",
    "kelpwatch_has_quarter_max",
    "historic_footprint_cells",
    "has_historic_kelp_footprint",
    "passes_initial_filter",
    "passes_robustness_filter_500",
    "n_years_total",
    "n_years_with_positive_kelp",
    "first_positive_kelp_year",
    "last_positive_kelp_year",
    "max_growing_season_kelp_area_m2",
    "mean_growing_season_kelp_area_m2",
    "median_growing_season_kelp_area_m2",
    "p25_growing_season_kelp_area_m2",
    "p75_growing_season_kelp_area_m2",
    "historical_p25_kelp_area_m2",
    "historical_mean_kelp_area_m2",
    "historical_max_kelp_area_m2",
    "latest_year",
    "latest_growing_season_kelp_area_m2",
    "latest_vs_historical_p25",
    "latest_vs_historical_mean_ratio",
    "mean_count_cells_no_clouds_max",
    "min_count_cells_no_clouds_max",
    "mean_cloud_free_fraction",
    "n_missing_or_empty_max_rows",
    "summary_error",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create an AOI inventory augmented with Kelpwatch CSV summaries."
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path("geometries/regular_10km_fishnet/aoi_inventory_regular_10km_fishnet.csv"),
        help="Input AOI inventory CSV.",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("data/raw/kelpwatch_aoi"),
        help="Directory containing kelpwatch_cell_XXX.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "geometries/regular_10km_fishnet/"
            "aoi_inventory_regular_10km_fishnet_with_kelpwatch_summary.csv"
        ),
        help="Output augmented inventory CSV.",
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


def quantile(values: list[float], probability: float) -> float | None:
    """Compute a linear-interpolated quantile without external dependencies."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def mean(values: list[float]) -> float | None:
    """Return mean or None for empty input."""
    if not values:
        return None
    return sum(values) / len(values)


def format_value(value: object) -> str:
    """Format summary values for stable CSV output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def empty_summary(csv_path: Path, error: str = "") -> dict[str, object]:
    """Return an empty summary row for missing or invalid CSVs."""
    summary = {column: "" for column in SUMMARY_COLUMNS}
    summary.update(
        {
            "kelpwatch_csv_exists": csv_path.exists(),
            "kelpwatch_csv_rows": 0,
            "kelpwatch_has_required_columns": False,
            "kelpwatch_has_quarter_max": False,
            "has_historic_kelp_footprint": False,
            "passes_initial_filter": False,
            "passes_robustness_filter_500": False,
            "summary_error": error,
        }
    )
    return summary


def summarize_cell_csv(csv_path: Path) -> dict[str, object]:
    """Summarize one raw Kelpwatch cell CSV."""
    if not csv_path.exists():
        return empty_summary(csv_path, "CSV file missing")
    if csv_path.stat().st_size == 0:
        return empty_summary(csv_path, "CSV file empty")

    try:
        with csv_path.open(newline="") as file:
            reader = csv.DictReader(file)
            fieldnames = set(reader.fieldnames or [])
            rows = list(reader)
    except Exception as exc:
        return empty_summary(csv_path, str(exc))

    has_required = REQUIRED_COLUMNS.issubset(fieldnames)
    if not has_required:
        return empty_summary(csv_path, f"Missing columns: {sorted(REQUIRED_COLUMNS - fieldnames)}")

    max_rows = [row for row in rows if row.get("quarter") == "max"]
    kelp_values = [to_float(row.get("kelp_area_m2")) for row in max_rows]
    years = [to_int(row.get("year")) for row in max_rows]
    positive = [(to_int(row.get("year")), to_float(row.get("kelp_area_m2"))) for row in max_rows]
    positive_years = [year for year, area in positive if area > 0]
    no_clouds = [to_float(row.get("count_cells_no_clouds")) for row in max_rows]
    historic_footprints = [to_int(row.get("count_cells_historic_footprint")) for row in rows]
    historic_footprint = max(historic_footprints) if historic_footprints else 0
    cloud_free_fractions = [
        to_float(row.get("count_cells_no_clouds")) / historic_footprint
        for row in max_rows
        if historic_footprint > 0
    ]
    missing_or_empty = sum(1 for row in max_rows if row.get("kelp_area_m2") in {"", None})
    latest_year = max(years) if years else None
    latest_area = None
    if latest_year is not None:
        for row in max_rows:
            if to_int(row.get("year")) == latest_year:
                latest_area = to_float(row.get("kelp_area_m2"))
                break

    p25 = quantile(kelp_values, 0.25)
    historical_mean = mean(kelp_values)
    latest_vs_p25 = None
    if latest_area is not None and p25 not in {None, 0}:
        latest_vs_p25 = latest_area - p25
    latest_vs_mean_ratio = None
    if latest_area is not None and historical_mean not in {None, 0}:
        latest_vs_mean_ratio = latest_area / historical_mean

    return {
        "kelpwatch_csv_exists": True,
        "kelpwatch_csv_rows": len(rows),
        "kelpwatch_has_required_columns": has_required,
        "kelpwatch_has_quarter_max": bool(max_rows),
        "historic_footprint_cells": historic_footprint,
        "has_historic_kelp_footprint": historic_footprint > 0,
        "passes_initial_filter": historic_footprint > 0,
        "passes_robustness_filter_500": historic_footprint >= 500,
        "n_years_total": len(max_rows),
        "n_years_with_positive_kelp": len(positive_years),
        "first_positive_kelp_year": min(positive_years) if positive_years else None,
        "last_positive_kelp_year": max(positive_years) if positive_years else None,
        "max_growing_season_kelp_area_m2": max(kelp_values) if kelp_values else None,
        "mean_growing_season_kelp_area_m2": mean(kelp_values),
        "median_growing_season_kelp_area_m2": quantile(kelp_values, 0.5),
        "p25_growing_season_kelp_area_m2": p25,
        "p75_growing_season_kelp_area_m2": quantile(kelp_values, 0.75),
        "historical_p25_kelp_area_m2": p25,
        "historical_mean_kelp_area_m2": historical_mean,
        "historical_max_kelp_area_m2": max(kelp_values) if kelp_values else None,
        "latest_year": latest_year,
        "latest_growing_season_kelp_area_m2": latest_area,
        "latest_vs_historical_p25": latest_vs_p25,
        "latest_vs_historical_mean_ratio": latest_vs_mean_ratio,
        "mean_count_cells_no_clouds_max": mean(no_clouds),
        "min_count_cells_no_clouds_max": min(no_clouds) if no_clouds else None,
        "mean_cloud_free_fraction": mean(cloud_free_fractions),
        "n_missing_or_empty_max_rows": missing_or_empty,
        "summary_error": "",
    }


def main() -> None:
    """Build the augmented AOI inventory."""
    args = parse_args()
    with args.inventory.open(newline="") as file:
        reader = csv.DictReader(file)
        inventory_rows = list(reader)
        inventory_columns = reader.fieldnames or []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_columns = inventory_columns + [col for col in SUMMARY_COLUMNS if col not in inventory_columns]

    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=output_columns)
        writer.writeheader()
        for row in inventory_rows:
            cell_id = row["cell_id"]
            csv_path = args.csv_dir / f"kelpwatch_{cell_id}.csv"
            summary = summarize_cell_csv(csv_path)
            output_row = dict(row)
            output_row.update({key: format_value(value) for key, value in summary.items()})
            writer.writerow(output_row)

    print(f"Wrote augmented inventory: {args.output}")
    print(f"Rows: {len(inventory_rows)}")


if __name__ == "__main__":
    main()
