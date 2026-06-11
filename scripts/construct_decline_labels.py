"""Construct next-year kelp canopy decline labels for early-warning modeling."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_CELLS = 50
EXPECTED_YEARS_PER_CELL = 42
EXPECTED_START_YEAR = 1984
EXPECTED_END_YEAR = 2025
EXPECTED_MODELING_ROWS = 2050
BASELINE_START_YEAR = 1984
BASELINE_END_YEAR = 2013
FOOTPRINT_CELL_AREA_M2 = 900

DEFAULT_PANEL = Path("data/processed/kelpwatch_panel_ge500.csv")
DEFAULT_LABELED_PANEL = Path("data/processed/kelpwatch_panel_ge500_labeled.csv")
DEFAULT_MODELING_PANEL = Path("data/processed/modeling_kelpwatch_labels_ge500.csv")
DEFAULT_SUMMARY = Path("outputs/metadata/decline_label_summary_ge500.csv")
DEFAULT_COUNTS_BY_YEAR = Path("outputs/metadata/decline_label_counts_by_year.csv")
DEFAULT_COUNTS_BY_REGION = Path("outputs/metadata/decline_label_counts_by_region.csv")
DEFAULT_REPORT = Path("outputs/metadata/decline_label_report.md")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Construct next-year decline labels from the Kelpwatch annual panel."
    )
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL, help="Input annual panel CSV.")
    parser.add_argument(
        "--labeled-output",
        type=Path,
        default=DEFAULT_LABELED_PANEL,
        help="Output full labeled panel CSV.",
    )
    parser.add_argument(
        "--modeling-output",
        type=Path,
        default=DEFAULT_MODELING_PANEL,
        help="Output modeling-ready labeled panel CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="Output label summary metadata CSV.",
    )
    parser.add_argument(
        "--counts-by-year-output",
        type=Path,
        default=DEFAULT_COUNTS_BY_YEAR,
        help="Output decline label counts by year CSV.",
    )
    parser.add_argument(
        "--counts-by-region-output",
        type=Path,
        default=DEFAULT_COUNTS_BY_REGION,
        help="Output decline label counts by region CSV.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_REPORT,
        help="Output Markdown decline label report.",
    )
    return parser.parse_args()


def require_columns(data: pd.DataFrame, columns: set[str], name: str) -> None:
    """Raise an error if required columns are missing."""
    missing = sorted(columns - set(data.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_panel(path: Path) -> pd.DataFrame:
    """Load and validate the input Kelpwatch annual canopy panel."""
    if not path.exists():
        raise FileNotFoundError(path)

    panel = pd.read_csv(path)
    require_columns(panel, {"cell_id", "year", "kelp_area_m2", "count_cells_historic_footprint"}, str(path))

    if "quarter" in panel.columns:
        quarters = set(panel["quarter"].astype(str).str.lower().unique())
        if quarters != {"max"}:
            raise ValueError(f"Expected quarter=max only, found: {sorted(quarters)}")

    if panel["cell_id"].nunique() != EXPECTED_CELLS:
        raise ValueError(f"Expected {EXPECTED_CELLS} unique cells, found {panel['cell_id'].nunique()}.")

    duplicates = panel.duplicated(["cell_id", "year"])
    if duplicates.any():
        duplicate_rows = panel.loc[duplicates, ["cell_id", "year"]].head(10).to_dict("records")
        raise ValueError(f"cell_id x year duplicates found. Examples: {duplicate_rows}")

    years_by_cell = panel.groupby("cell_id")["year"].agg(["count", "min", "max"])
    invalid_cells = years_by_cell.loc[
        (years_by_cell["count"] != EXPECTED_YEARS_PER_CELL)
        | (years_by_cell["min"] != EXPECTED_START_YEAR)
        | (years_by_cell["max"] != EXPECTED_END_YEAR)
    ]
    if not invalid_cells.empty:
        raise ValueError(f"Cells with invalid annual coverage: {invalid_cells.to_dict('index')}")

    panel = panel.sort_values(["cell_id", "year"]).reset_index(drop=True)
    return panel


def ensure_relative_canopy(panel: pd.DataFrame) -> pd.DataFrame:
    """Ensure relative_canopy and historical_footprint_area_m2 are present and valid."""
    panel = panel.copy()
    numeric_columns = ["kelp_area_m2", "count_cells_historic_footprint"]
    for column in numeric_columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")

    if "historical_footprint_area_m2" not in panel.columns:
        panel["historical_footprint_area_m2"] = (
            panel["count_cells_historic_footprint"] * FOOTPRINT_CELL_AREA_M2
        )
    else:
        panel["historical_footprint_area_m2"] = pd.to_numeric(
            panel["historical_footprint_area_m2"], errors="coerce"
        )

    if "relative_canopy" not in panel.columns:
        panel["relative_canopy"] = panel["kelp_area_m2"] / panel["historical_footprint_area_m2"]
    else:
        panel["relative_canopy"] = pd.to_numeric(panel["relative_canopy"], errors="coerce")

    relative = panel["relative_canopy"]
    if relative.isna().any() or not np.isfinite(relative).all():
        raise ValueError("relative_canopy contains missing or non-finite values.")
    if (relative < 0).any():
        raise ValueError("relative_canopy contains negative values.")

    return panel


def add_decline_labels(panel: pd.DataFrame) -> pd.DataFrame:
    """Add next-year canopy values and decline labels."""
    labeled = panel.sort_values(["cell_id", "year"]).copy()

    labeled["next_year_kelp_area_m2"] = labeled.groupby("cell_id")["kelp_area_m2"].shift(-1)
    labeled["next_year_relative_canopy"] = labeled.groupby("cell_id")["relative_canopy"].shift(-1)

    baseline = labeled.loc[
        labeled["year"].between(BASELINE_START_YEAR, BASELINE_END_YEAR)
    ].groupby("cell_id")["relative_canopy"].quantile(0.25)
    full_history = labeled.groupby("cell_id")["relative_canopy"].quantile(0.25)

    labeled["baseline_p25_relative_canopy_1984_2013"] = labeled["cell_id"].map(baseline)
    labeled["p25_relative_canopy_full_history"] = labeled["cell_id"].map(full_history)

    has_next = labeled["next_year_relative_canopy"].notna()
    labeled["decline_event_next"] = pd.NA
    labeled.loc[has_next, "decline_event_next"] = (
        labeled.loc[has_next, "next_year_relative_canopy"]
        < labeled.loc[has_next, "baseline_p25_relative_canopy_1984_2013"]
    ).astype(int)

    labeled["decline_event_next_p25_full"] = pd.NA
    labeled.loc[has_next, "decline_event_next_p25_full"] = (
        labeled.loc[has_next, "next_year_relative_canopy"]
        < labeled.loc[has_next, "p25_relative_canopy_full_history"]
    ).astype(int)

    labeled["relative_canopy_change_next"] = (
        labeled["next_year_relative_canopy"] - labeled["relative_canopy"]
    )
    labeled["relative_canopy_pct_change_next"] = np.where(
        labeled["relative_canopy"] > 0,
        labeled["relative_canopy_change_next"] / labeled["relative_canopy"],
        np.nan,
    )

    labeled["decline_50pct_next"] = pd.NA
    valid_pct_decline = has_next & labeled["relative_canopy"].gt(0)
    labeled.loc[has_next, "decline_50pct_next"] = 0
    labeled.loc[valid_pct_decline, "decline_50pct_next"] = (
        labeled.loc[valid_pct_decline, "next_year_relative_canopy"]
        <= 0.5 * labeled.loc[valid_pct_decline, "relative_canopy"]
    ).astype(int)

    for column in ["decline_event_next", "decline_event_next_p25_full", "decline_50pct_next"]:
        labeled[column] = labeled[column].astype("Int64")

    return labeled


def validate_labeled_panel(labeled: pd.DataFrame, modeling: pd.DataFrame) -> None:
    """Run requested validation checks on the labeled panel."""
    final_year = labeled.loc[labeled["year"] == EXPECTED_END_YEAR]
    if final_year.empty:
        raise ValueError(f"No rows found for final year {EXPECTED_END_YEAR}.")
    if final_year["next_year_relative_canopy"].notna().any():
        raise ValueError(f"{EXPECTED_END_YEAR} rows should have missing next-year canopy values.")
    if final_year["decline_event_next"].notna().any():
        raise ValueError(f"{EXPECTED_END_YEAR} rows should have missing main next-year labels.")

    if len(modeling) != EXPECTED_MODELING_ROWS:
        raise ValueError(f"Expected {EXPECTED_MODELING_ROWS} modeling rows, found {len(modeling)}.")

    target_classes = set(modeling["decline_event_next"].dropna().astype(int).unique())
    if target_classes != {0, 1}:
        raise ValueError(f"decline_event_next must contain both 0 and 1 classes, found {target_classes}.")


def label_counts(data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Summarize decline labels by a grouping column."""
    grouped = (
        data.groupby(group_column, dropna=False)
        .agg(
            modeling_rows=("decline_event_next", "size"),
            decline_event_count=("decline_event_next", "sum"),
            decline_event_p25_full_count=("decline_event_next_p25_full", "sum"),
            decline_50pct_count=("decline_50pct_next", "sum"),
        )
        .reset_index()
    )
    grouped["decline_event_rate"] = grouped["decline_event_count"] / grouped["modeling_rows"]
    return grouped


def create_summary(panel: pd.DataFrame, labeled: pd.DataFrame, modeling: pd.DataFrame) -> pd.DataFrame:
    """Create a compact metadata summary as key-value rows."""
    main_count = int(modeling["decline_event_next"].sum())
    full_count = int(modeling["decline_event_next_p25_full"].sum())
    pct50_count = int(modeling["decline_50pct_next"].sum())
    rows = [
        ("input_rows", len(panel)),
        ("number_of_cells", panel["cell_id"].nunique()),
        ("year_range", f"{int(panel['year'].min())}-{int(panel['year'].max())}"),
        ("baseline_period", f"{BASELINE_START_YEAR}-{BASELINE_END_YEAR}"),
        ("rows_with_valid_next_year_labels", len(modeling)),
        ("main_decline_event_count", main_count),
        ("main_decline_event_rate", main_count / len(modeling)),
        ("full_history_p25_decline_event_count", full_count),
        ("full_history_p25_decline_event_rate", full_count / len(modeling)),
        ("decline_50pct_next_count", pct50_count),
        ("decline_50pct_next_rate", pct50_count / len(modeling)),
        ("full_labeled_rows", len(labeled)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def write_report(
    path: Path,
    summary: pd.DataFrame,
    counts_by_region: pd.DataFrame,
    counts_by_year: pd.DataFrame,
) -> None:
    """Write a Markdown report for the decline label construction workflow."""
    values = dict(zip(summary["metric"], summary["value"]))
    lines = [
        "# Kelpwatch Decline Label Report",
        "",
        "## Summary",
        "",
        f"Input rows: {values['input_rows']}",
        f"Number of cells: {values['number_of_cells']}",
        f"Year range: {values['year_range']}",
        f"Baseline period: {values['baseline_period']}",
        f"Rows with valid next-year labels: {values['rows_with_valid_next_year_labels']}",
        f"Main decline-event count: {values['main_decline_event_count']}",
        f"Main decline-event rate: {float(values['main_decline_event_rate']):.4f}",
        "",
        "## Robustness Label Counts",
        "",
        f"Full-history p25 decline-event count: {values['full_history_p25_decline_event_count']}",
        f"Full-history p25 decline-event rate: {float(values['full_history_p25_decline_event_rate']):.4f}",
        f"50 percent next-year decline count: {values['decline_50pct_next_count']}",
        f"50 percent next-year decline rate: {float(values['decline_50pct_next_rate']):.4f}",
        "",
        "## Decline-Event Count by Region",
        "",
    ]

    for row in counts_by_region.to_dict("records"):
        lines.append(
            f"- {row['region_group']}: {int(row['decline_event_count'])} / "
            f"{int(row['modeling_rows'])} rows ({row['decline_event_rate']:.4f})"
        )

    lines.extend(["", "## Decline-Event Count by Year", ""])
    for row in counts_by_year.to_dict("records"):
        lines.append(
            f"- {int(row['year'])}: {int(row['decline_event_count'])} / "
            f"{int(row['modeling_rows'])} rows ({row['decline_event_rate']:.4f})"
        )

    lines.extend(
        [
            "",
            "## Label Definitions",
            "",
            "The main early-warning target is `decline_event_next`, which equals 1 when the following year's `relative_canopy` falls below the cell-specific 25th percentile of `relative_canopy` during the 1984-2013 baseline period.",
            "",
            "Robustness labels include `decline_event_next_p25_full`, based on each cell's full-history 25th percentile, and `decline_50pct_next`, based on a next-year canopy decline of at least 50 percent from the current year.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_outputs(args: argparse.Namespace, labeled: pd.DataFrame, modeling: pd.DataFrame) -> None:
    """Write processed data, metadata summaries, and report outputs."""
    args.labeled_output.parent.mkdir(parents=True, exist_ok=True)
    args.modeling_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    labeled.to_csv(args.labeled_output, index=False)
    modeling.to_csv(args.modeling_output, index=False)

    summary = create_summary(labeled, labeled, modeling)
    counts_by_year = label_counts(modeling, "year")
    counts_by_region = label_counts(modeling, "region_group")

    summary.to_csv(args.summary_output, index=False)
    counts_by_year.to_csv(args.counts_by_year_output, index=False)
    counts_by_region.to_csv(args.counts_by_region_output, index=False)
    write_report(args.report_output, summary, counts_by_region, counts_by_year)


def main() -> None:
    """Run the decline-label construction workflow."""
    args = parse_args()
    panel = ensure_relative_canopy(load_panel(args.panel))
    labeled = add_decline_labels(panel)
    modeling = labeled.loc[labeled["decline_event_next"].notna()].copy()
    validate_labeled_panel(labeled, modeling)
    write_outputs(args, labeled, modeling)

    main_count = int(modeling["decline_event_next"].sum())
    print("Kelpwatch decline-label construction complete.")
    print(f"Input rows: {len(panel)}")
    print(f"Cells: {panel['cell_id'].nunique()}")
    print(f"Modeling rows: {len(modeling)}")
    print(f"Main decline events: {main_count}")
    print(f"Main decline-event rate: {main_count / len(modeling):.4f}")
    print(f"Labeled output: {args.labeled_output}")
    print(f"Modeling output: {args.modeling_output}")


if __name__ == "__main__":
    main()
