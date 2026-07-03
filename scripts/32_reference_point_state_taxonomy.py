#!/usr/bin/env python3
"""Build site-specific canopy reference-state and transition taxonomy tables.

This script reuses the processed Kelpwatch annual modeling panel and saved
out-of-sample prediction probabilities. It does not retrain models. The goal
is to describe ecological state transitions around cell-specific canopy
reference points so that broad risk-state screening can be separated from
stricter new-transition or actionable-drop interpretation.

Run from the repository root:

    python scripts/32_reference_point_state_taxonomy.py
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


DEFAULT_INPUT = Path("data/processed/modeling_dataset_ge500_noaa_v1.csv")
DEFAULT_PREDICTIONS = Path("outputs/metadata/model_comparison_test_predictions.csv")
TABLE_DIR = Path("outputs/tables")
REPORT_DIR = Path("outputs/reports")
FIGURE_DIR = Path("outputs/figures")

TAXONOMY_OUTPUT = TABLE_DIR / "reference_point_state_taxonomy.csv"
TRANSITION_MATRIX_OUTPUT = TABLE_DIR / "reference_point_transition_matrix.csv"
TRANSITION_MATRIX_BY_YEAR_OUTPUT = TABLE_DIR / "reference_point_transition_matrix_by_year.csv"
TRANSITION_SUMMARY_OUTPUT = TABLE_DIR / "reference_point_transition_summary.csv"
TRANSITION_SUMMARY_BY_YEAR_OUTPUT = TABLE_DIR / "reference_point_transition_summary_by_year.csv"
MODEL_RISK_OUTPUT = TABLE_DIR / "model_risk_by_transition_type.csv"
TOPK_COMPOSITION_OUTPUT = TABLE_DIR / "topk_transition_composition.csv"
REPORT_OUTPUT = REPORT_DIR / "reference_point_state_taxonomy_summary.md"

MATRIX_COUNTS_FIGURE = FIGURE_DIR / "reference_point_transition_matrix_counts.png"
MATRIX_PROBABILITIES_FIGURE = FIGURE_DIR / "reference_point_transition_matrix_probabilities.png"
TYPES_BY_YEAR_FIGURE = FIGURE_DIR / "reference_point_transition_types_by_year.png"
CURRENT_NEXT_FIGURE = FIGURE_DIR / "reference_point_current_vs_next_canopy.png"
MODEL_RISK_FIGURE = FIGURE_DIR / "model_risk_by_transition_type.png"
TOPK_COMPOSITION_FIGURE = FIGURE_DIR / "topk_transition_composition.png"

BASELINE_START_YEAR = 1984
BASELINE_END_YEAR = 2013
MAIN_MODEL_START_YEAR = 1989
VALIDATION_START_YEAR = 2017
TEST_START_YEAR = 2021
TEST_END_YEAR = 2024
NEAR_ZERO_THRESHOLD = 0.05
FIXED_TOPK_BUDGETS = [1, 3, 5, 10, 15, 20]

STATE_ORDER = [
    "low_below_p25",
    "lower_mid_p25_p50",
    "upper_mid_p50_p75",
    "high_above_p75",
    "missing_reference_state",
]
STATE_LABELS = {
    "low_below_p25": "Low\n< p25",
    "lower_mid_p25_p50": "Lower-mid\np25-p50",
    "upper_mid_p50_p75": "Upper-mid\np50-p75",
    "high_above_p75": "High\n>= p75",
    "missing_reference_state": "Missing",
}

TRANSITION_ORDER = [
    "persistent_low",
    "recovery_from_low",
    "new_low_transition",
    "stable_non_low",
    "missing_transition",
]
TRANSITION_COLORS = {
    "persistent_low": "#2E4780",
    "recovery_from_low": "#71B436",
    "new_low_transition": "#CC6F47",
    "stable_non_low": "#C5CAD3",
    "missing_transition": "#7A828F",
}

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

CSV_WRITE_KWARGS = {
    "index": False,
    "lineterminator": "\n",
    "na_rep": "",
    "float_format": "%.6f",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build reference-point canopy state taxonomy.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--taxonomy-output", type=Path, default=TAXONOMY_OUTPUT)
    parser.add_argument("--transition-matrix-output", type=Path, default=TRANSITION_MATRIX_OUTPUT)
    parser.add_argument("--transition-matrix-by-year-output", type=Path, default=TRANSITION_MATRIX_BY_YEAR_OUTPUT)
    parser.add_argument("--transition-summary-output", type=Path, default=TRANSITION_SUMMARY_OUTPUT)
    parser.add_argument("--transition-summary-by-year-output", type=Path, default=TRANSITION_SUMMARY_BY_YEAR_OUTPUT)
    parser.add_argument("--model-risk-output", type=Path, default=MODEL_RISK_OUTPUT)
    parser.add_argument("--topk-composition-output", type=Path, default=TOPK_COMPOSITION_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument(
        "--label-target",
        default="decline_event_next",
        help="Default label target name when prediction file lacks a label target column.",
    )
    return parser.parse_args()


def clean_csv_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten embedded line breaks in object cells before writing CSV."""
    out = df.copy()
    object_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        out[col] = (
            out[col]
            .astype("string")
            .str.replace("\r\n", " ", regex=False)
            .str.replace("\n", " ", regex=False)
            .str.replace("\r", " ", regex=False)
        )
        out[col] = out[col].where(out[col].notna(), "")
    return out


def write_portable_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a stable LF-delimited CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_csv_cells(df).to_csv(path, **CSV_WRITE_KWARGS)


def infer_column(columns: Iterable[str], candidates: list[str], required: bool, description: str) -> str | None:
    """Infer a column name from possible alternatives."""
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    if required:
        raise ValueError(f"Could not infer {description}. Tried: {candidates}")
    return None


def require_columns(data: pd.DataFrame, columns: set[str], path: Path) -> None:
    """Raise a clear error if required columns are missing."""
    missing = sorted(columns - set(data.columns))
    if missing:
        raise ValueError(
            f"{path} is missing required columns for reference-state taxonomy: {missing}. "
            "Run `python scripts/construct_decline_labels.py` and "
            "`python scripts/build_noaa_environmental_features.py` first."
        )


def load_taxonomy_input(path: Path) -> tuple[pd.DataFrame, dict[str, str | None], list[str]]:
    """Load the processed annual modeling panel and normalize core columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Input modeling panel not found: {path}. Expected the processed V1 dataset."
        )
    data = pd.read_csv(path)
    columns = list(data.columns)
    cell_col = infer_column(columns, ["cell_id", "grid_id", "site_id", "aoi_id"], True, "cell identifier")
    year_col = infer_column(columns, ["year"], True, "year")
    current_col = infer_column(
        columns,
        ["relative_canopy", "current_relative_canopy", "canopy_current_t"],
        True,
        "current relative canopy",
    )
    next_col = infer_column(
        columns,
        ["next_year_relative_canopy", "next_relative_canopy"],
        True,
        "next-year relative canopy",
    )
    q25_col = infer_column(
        columns,
        ["baseline_p25_relative_canopy_1984_2013", "q25", "p25_relative_canopy"],
        False,
        "q25 reference point",
    )
    region_col = infer_column(columns, ["region_group", "region", "region_name"], False, "region")
    lat_col = infer_column(columns, ["center_lat", "latitude", "lat"], False, "latitude")
    lon_col = infer_column(columns, ["center_lon", "longitude", "lon"], False, "longitude")
    split_col = infer_column(columns, ["split", "data_split"], False, "split")

    require_columns(data, {cell_col, year_col, current_col, next_col}, path)
    data = data.copy()
    data[cell_col] = data[cell_col].astype(str)
    data[year_col] = pd.to_numeric(data[year_col], errors="coerce").astype("Int64")
    data[current_col] = pd.to_numeric(data[current_col], errors="coerce")
    data[next_col] = pd.to_numeric(data[next_col], errors="coerce")
    if q25_col:
        data[q25_col] = pd.to_numeric(data[q25_col], errors="coerce")

    duplicate_mask = data.duplicated([cell_col, year_col])
    if duplicate_mask.any():
        examples = data.loc[duplicate_mask, [cell_col, year_col]].head(10).to_dict("records")
        raise ValueError(f"Duplicate cell-year rows found in {path}. Examples: {examples}")

    notes: list[str] = []
    if split_col is None:
        data["split"] = data[year_col].map(infer_temporal_split)
        split_col = "split"
        notes.append(
            "No split column was present in the modeling panel; split was inferred as "
            "pre_modeling=1984-1988, train=1989-2016, validation=2017-2020, test=2021-2024."
        )

    mapping = {
        "cell_id": cell_col,
        "year": year_col,
        "current": current_col,
        "next": next_col,
        "q25": q25_col,
        "region": region_col,
        "latitude": lat_col,
        "longitude": lon_col,
        "split": split_col,
    }
    return data, mapping, notes


def infer_temporal_split(year: object) -> str:
    """Infer the temporal split used by the existing model-comparison workflow."""
    if pd.isna(year):
        return "missing_year"
    year_int = int(year)
    if year_int < MAIN_MODEL_START_YEAR:
        return "pre_modeling"
    if year_int < VALIDATION_START_YEAR:
        return "train"
    if year_int < TEST_START_YEAR:
        return "validation"
    if year_int <= TEST_END_YEAR:
        return "test"
    return "post_test"


def compute_reference_points(
    data: pd.DataFrame,
    mapping: dict[str, str | None],
) -> tuple[pd.DataFrame, str, list[str]]:
    """Compute or reuse q25/q50/q75 cell-specific reference points."""
    cell_col = str(mapping["cell_id"])
    year_col = str(mapping["year"])
    canopy_col = str(mapping["current"])
    q25_col = mapping["q25"]
    notes: list[str] = []

    baseline = data.loc[data[year_col].between(BASELINE_START_YEAR, BASELINE_END_YEAR)].copy()
    if baseline.empty:
        baseline = data.copy()
        baseline_period = f"full available years {int(data[year_col].min())}-{int(data[year_col].max())}"
        notes.append(
            f"No rows were available for {BASELINE_START_YEAR}-{BASELINE_END_YEAR}; "
            "q25/q50/q75 were computed from the full available series."
        )
    else:
        baseline_period = f"{BASELINE_START_YEAR}-{BASELINE_END_YEAR}"

    quantiles = (
        baseline.groupby(cell_col)[canopy_col]
        .quantile([0.25, 0.50, 0.75])
        .unstack()
        .rename(columns={0.25: "computed_q25", 0.50: "q50", 0.75: "q75"})
        .reset_index()
    )

    if q25_col:
        q25_existing = (
            data[[cell_col, str(q25_col)]]
            .drop_duplicates(subset=[cell_col])
            .rename(columns={str(q25_col): "q25"})
        )
        reference = quantiles.merge(q25_existing, on=cell_col, how="left")
        missing_q25 = reference["q25"].isna()
        if missing_q25.any():
            reference.loc[missing_q25, "q25"] = reference.loc[missing_q25, "computed_q25"]
            notes.append(
                f"Existing q25 column `{q25_col}` had missing values for {int(missing_q25.sum())} cells; "
                "computed baseline q25 was used for those cells."
            )
        notes.append(f"q25 was reused from `{q25_col}`; q50 and q75 were computed from {baseline_period}.")
    else:
        reference = quantiles.rename(columns={"computed_q25": "q25"})
        notes.append(f"q25/q50/q75 were computed from {baseline_period}.")

    reference = reference[[cell_col, "q25", "q50", "q75"]]
    for column in ["q25", "q50", "q75"]:
        reference[column] = pd.to_numeric(reference[column], errors="coerce")
    return reference, baseline_period, notes


def assign_reference_state(value: float, q25: float, q50: float, q75: float) -> str:
    """Assign a relative-canopy value to a cell-specific reference state."""
    if pd.isna(value) or pd.isna(q25) or pd.isna(q50) or pd.isna(q75):
        return "missing_reference_state"
    if value < q25:
        return "low_below_p25"
    if value < q50:
        return "lower_mid_p25_p50"
    if value < q75:
        return "upper_mid_p50_p75"
    return "high_above_p75"


def assign_transition_type(current_state: str, next_state: str) -> str:
    """Assign the primary current-to-next reference-state transition type."""
    if "missing" in {current_state, next_state}:
        return "missing_transition"
    current_low = current_state == "low_below_p25"
    next_low = next_state == "low_below_p25"
    if current_low and next_low:
        return "persistent_low"
    if current_low and not next_low:
        return "recovery_from_low"
    if not current_low and next_low:
        return "new_low_transition"
    return "stable_non_low"


def build_taxonomy(data: pd.DataFrame, mapping: dict[str, str | None], reference: pd.DataFrame) -> pd.DataFrame:
    """Build the main reference-point state taxonomy table."""
    cell_col = str(mapping["cell_id"])
    year_col = str(mapping["year"])
    current_col = str(mapping["current"])
    next_col = str(mapping["next"])
    split_col = str(mapping["split"])

    taxonomy = data.merge(reference, on=cell_col, how="left", validate="many_to_one")
    taxonomy["current_reference_state"] = taxonomy.apply(
        lambda row: assign_reference_state(row[current_col], row["q25"], row["q50"], row["q75"]),
        axis=1,
    )
    taxonomy["next_reference_state"] = taxonomy.apply(
        lambda row: assign_reference_state(row[next_col], row["q25"], row["q50"], row["q75"]),
        axis=1,
    )
    taxonomy["transition_type"] = [
        assign_transition_type(current_state, next_state)
        for current_state, next_state in zip(
            taxonomy["current_reference_state"], taxonomy["next_reference_state"], strict=True
        )
    ]

    current = taxonomy[current_col]
    next_value = taxonomy[next_col]
    taxonomy["near_zero_current_flag"] = current.le(NEAR_ZERO_THRESHOLD).where(current.notna(), pd.NA)
    taxonomy["near_zero_next_flag"] = next_value.le(NEAR_ZERO_THRESHOLD).where(next_value.notna(), pd.NA)
    taxonomy["current_at_risk_flag"] = current.gt(NEAR_ZERO_THRESHOLD).where(current.notna(), pd.NA)
    taxonomy["current_reference_or_high_flag"] = current.ge(taxonomy["q50"]).where(current.notna() & taxonomy["q50"].notna(), pd.NA)
    taxonomy["current_high_flag"] = current.ge(taxonomy["q75"]).where(current.notna() & taxonomy["q75"].notna(), pd.NA)

    taxonomy["persistent_low_flag"] = taxonomy["transition_type"].eq("persistent_low")
    taxonomy["recovery_from_low_flag"] = taxonomy["transition_type"].eq("recovery_from_low")
    taxonomy["new_low_transition_flag"] = taxonomy["transition_type"].eq("new_low_transition")
    taxonomy["stable_non_low_flag"] = taxonomy["transition_type"].eq("stable_non_low")
    taxonomy["actionable_low_entry_flag"] = current.gt(NEAR_ZERO_THRESHOLD) & taxonomy["next_reference_state"].eq("low_below_p25")
    taxonomy["sharp_drop_30_flag"] = current.gt(NEAR_ZERO_THRESHOLD) & next_value.le(0.70 * current)
    taxonomy["sharp_drop_50_flag"] = current.gt(NEAR_ZERO_THRESHOLD) & next_value.le(0.50 * current)
    taxonomy["high_to_low_transition_flag"] = (
        taxonomy["current_reference_state"].eq("high_above_p75")
        & taxonomy["next_reference_state"].eq("low_below_p25")
    )
    taxonomy["reference_to_low_transition_flag"] = (
        current.ge(taxonomy["q50"]) & taxonomy["next_reference_state"].eq("low_below_p25")
    )

    taxonomy["absolute_canopy_change"] = next_value - current
    taxonomy["proportional_drop"] = np.where(
        current.gt(0) & current.notna() & next_value.notna(),
        (current - next_value) / current,
        np.nan,
    )

    rename_map = {
        cell_col: "cell_id",
        year_col: "year",
        current_col: "current_relative_canopy",
        next_col: "next_relative_canopy",
        split_col: "split",
    }
    optional_columns = []
    for source_key, output_name in [
        ("region", "region"),
        ("latitude", "latitude"),
        ("longitude", "longitude"),
    ]:
        source_col = mapping[source_key]
        if source_col:
            rename_map[str(source_col)] = output_name
            optional_columns.append(output_name)

    taxonomy = taxonomy.rename(columns=rename_map)

    existing_label_cols = [
        column
        for column in [
            "decline_event_next",
            "decline_event_next_p25_full",
            "decline_50pct_next",
            "relative_canopy_change_next",
            "relative_canopy_pct_change_next",
        ]
        if column in taxonomy.columns
    ]

    base_columns = [
        "cell_id",
        "year",
        "split",
        *optional_columns,
        "current_relative_canopy",
        "next_relative_canopy",
        "q25",
        "q50",
        "q75",
        "current_reference_state",
        "next_reference_state",
        "transition_type",
        "near_zero_current_flag",
        "near_zero_next_flag",
        "current_at_risk_flag",
        "current_reference_or_high_flag",
        "current_high_flag",
        "persistent_low_flag",
        "recovery_from_low_flag",
        "new_low_transition_flag",
        "stable_non_low_flag",
        "actionable_low_entry_flag",
        "sharp_drop_30_flag",
        "sharp_drop_50_flag",
        "high_to_low_transition_flag",
        "reference_to_low_transition_flag",
        "absolute_canopy_change",
        "proportional_drop",
        *existing_label_cols,
    ]
    return taxonomy[base_columns].sort_values(["cell_id", "year"]).reset_index(drop=True)


def transition_matrix(taxonomy: pd.DataFrame, by_year: bool = False) -> pd.DataFrame:
    """Build transition matrix counts and row-normalized probabilities."""
    group_cols = ["split", "current_reference_state", "next_reference_state"]
    denominator_cols = ["split", "current_reference_state"]
    total_cols = ["split"]
    if by_year:
        group_cols = ["year", *group_cols]
        denominator_cols = ["year", *denominator_cols]
        total_cols = ["year", *total_cols]

    counts = taxonomy.groupby(group_cols, dropna=False).size().reset_index(name="count")
    row_totals = counts.groupby(denominator_cols, dropna=False)["count"].sum().reset_index(name="row_total")
    totals = counts.groupby(total_cols, dropna=False)["count"].sum().reset_index(name="split_total")
    out = counts.merge(row_totals, on=denominator_cols, how="left").merge(totals, on=total_cols, how="left")
    out["transition_probability"] = out["count"] / out["row_total"]
    out["overall_share"] = out["count"] / out["split_total"]
    out = out.drop(columns=["split_total"])
    return out.sort_values(group_cols).reset_index(drop=True)


def transition_summary(taxonomy: pd.DataFrame, by_year: bool = False) -> pd.DataFrame:
    """Summarize transition types by split and optionally by year."""
    group_cols = ["split", "transition_type"]
    denominator_cols = ["split"]
    share_column = "share_of_observations"
    if by_year:
        group_cols = ["year", *group_cols]
        denominator_cols = ["year", *denominator_cols]
        share_column = "share_of_year_split_observations"

    summary = (
        taxonomy.groupby(group_cols, dropna=False)
        .agg(
            n_observations=("transition_type", "size"),
            mean_current_relative_canopy=("current_relative_canopy", "mean"),
            mean_next_relative_canopy=("next_relative_canopy", "mean"),
            mean_absolute_change=("absolute_canopy_change", "mean"),
            mean_proportional_drop=("proportional_drop", "mean"),
            n_actionable_low_entry=("actionable_low_entry_flag", "sum"),
            n_sharp_drop_30=("sharp_drop_30_flag", "sum"),
            n_sharp_drop_50=("sharp_drop_50_flag", "sum"),
            n_high_to_low_transition=("high_to_low_transition_flag", "sum"),
            n_reference_to_low_transition=("reference_to_low_transition_flag", "sum"),
        )
        .reset_index()
    )
    totals = summary.groupby(denominator_cols, dropna=False)["n_observations"].sum().reset_index(name="group_total")
    summary = summary.merge(totals, on=denominator_cols, how="left")
    summary[share_column] = summary["n_observations"] / summary["group_total"]
    summary = summary.drop(columns=["group_total"])
    ordered = [*group_cols, "n_observations", share_column]
    other_cols = [column for column in summary.columns if column not in ordered]
    return summary[ordered + other_cols].sort_values(group_cols).reset_index(drop=True)


def load_predictions(path: Path, default_label_target: str) -> tuple[pd.DataFrame | None, list[str]]:
    """Load prediction probabilities for model-risk linkage, if available."""
    notes: list[str] = []
    if not path.exists():
        notes.append(f"Prediction file not found: {path}; model-risk linkage was skipped.")
        return None, notes

    raw = pd.read_csv(path)
    try:
        cell_col = infer_column(raw.columns, ["cell_id", "grid_id", "site_id", "aoi_id"], True, "prediction cell id")
        year_col = infer_column(raw.columns, ["year", "eval_year", "prediction_year"], True, "prediction year")
        true_col = infer_column(raw.columns, ["y_true", "label", "target", "actual", "observed"], True, "true label")
        score_col = infer_column(
            raw.columns,
            ["y_proba", "predicted_probability", "probability", "risk_score", "score", "prediction_score"],
            True,
            "predicted probability",
        )
    except ValueError as error:
        notes.append(f"Prediction merge skipped: {error}")
        return None, notes

    feature_col = infer_column(raw.columns, ["feature_set", "feature_family"], False, "feature set")
    model_col = infer_column(raw.columns, ["model", "model_name"], False, "model")
    split_col = infer_column(raw.columns, ["split", "evaluation_split"], False, "split")
    label_col = infer_column(raw.columns, ["label_target", "target_definition", "target_column"], False, "label target")

    predictions = pd.DataFrame(
        {
            "cell_id": raw[cell_col].astype(str),
            "year": pd.to_numeric(raw[year_col], errors="coerce").astype("Int64"),
            "label_target": raw[label_col].astype(str) if label_col else default_label_target,
            "feature_set": raw[feature_col].astype(str) if feature_col else "unknown_feature_set",
            "model": raw[model_col].astype(str) if model_col else "unknown_model",
            "split": raw[split_col].astype(str) if split_col else "out_of_sample",
            "y_true": pd.to_numeric(raw[true_col], errors="coerce"),
            "y_proba": pd.to_numeric(raw[score_col], errors="coerce"),
            "original_row": np.arange(len(raw)),
        }
    )
    before = len(predictions)
    predictions = predictions.dropna(subset=["cell_id", "year", "y_true", "y_proba"]).copy()
    if len(predictions) < before:
        notes.append(f"Dropped {before - len(predictions)} prediction rows with missing keys/labels/scores.")
    predictions["year"] = predictions["year"].astype(int)
    predictions["y_true"] = predictions["y_true"].astype(int)
    if not label_col:
        notes.append(f"No label target column found in predictions; assigned `{default_label_target}`.")
    if predictions.empty:
        notes.append(f"No usable prediction rows found in {path}; model-risk linkage was skipped.")
        return None, notes
    return predictions, notes


def merge_predictions_with_taxonomy(
    taxonomy: pd.DataFrame,
    predictions: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, list[str]]:
    """Merge prediction probabilities with taxonomy on cell-year keys."""
    if predictions is None:
        return None, ["Prediction file was unavailable or unusable."]

    merge_cols = ["cell_id", "year"]
    minimal_taxonomy = taxonomy[
        [
            "cell_id",
            "year",
            "split",
            "transition_type",
            "actionable_low_entry_flag",
            "sharp_drop_30_flag",
            "new_low_transition_flag",
            "persistent_low_flag",
        ]
    ].copy()
    merged = predictions.merge(
        minimal_taxonomy,
        on=merge_cols,
        how="left",
        suffixes=("_prediction", "_taxonomy"),
        validate="many_to_one",
    )
    missing = int(merged["transition_type"].isna().sum())
    notes: list[str] = []
    if missing:
        notes.append(f"{missing} prediction rows could not be matched to taxonomy rows on cell_id/year.")

    if "split_taxonomy" in merged.columns:
        mismatch = merged["split_prediction"].ne(merged["split_taxonomy"]).sum()
        if mismatch:
            notes.append(f"{int(mismatch)} prediction rows had prediction/taxonomy split mismatches.")
        merged = merged.rename(columns={"split_prediction": "split"}).drop(columns=["split_taxonomy"])

    merged = merged.dropna(subset=["transition_type"]).copy()
    if merged.empty:
        notes.append("No prediction rows remained after taxonomy merge.")
        return None, notes
    return merged, notes


def model_risk_by_transition(merged: pd.DataFrame | None) -> pd.DataFrame:
    """Summarize predicted probabilities by transition type."""
    if merged is None or merged.empty:
        return pd.DataFrame()
    group_cols = ["label_target", "feature_set", "model", "split", "transition_type"]
    summary = (
        merged.groupby(group_cols, dropna=False)
        .agg(
            n_observations=("y_proba", "size"),
            mean_predicted_probability=("y_proba", "mean"),
            median_predicted_probability=("y_proba", "median"),
            q25_predicted_probability=("y_proba", lambda x: x.quantile(0.25)),
            q75_predicted_probability=("y_proba", lambda x: x.quantile(0.75)),
            mean_true_label=("y_true", "mean"),
            share_actionable_low_entry=("actionable_low_entry_flag", "mean"),
            share_sharp_drop_30=("sharp_drop_30_flag", "mean"),
            share_new_low_transition=("new_low_transition_flag", "mean"),
            share_persistent_low=("persistent_low_flag", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(group_cols).reset_index(drop=True)


def choose_main_prediction_group(merged: pd.DataFrame) -> dict[str, str]:
    """Choose the preferred model group for figures and top-k composition."""
    preferred = merged.loc[
        merged["label_target"].eq("decline_event_next")
        & merged["feature_set"].eq("canopy_only")
        & merged["model"].eq("Random Forest")
        & merged["split"].eq("test")
    ]
    if not preferred.empty:
        return {
            "label_target": "decline_event_next",
            "feature_set": "canopy_only",
            "model": "Random Forest",
            "split": "test",
        }
    fallback = (
        merged.groupby(["label_target", "feature_set", "model", "split"], dropna=False)
        .agg(n_observations=("y_proba", "size"), mean_true_label=("y_true", "mean"))
        .reset_index()
        .sort_values(["n_observations", "mean_true_label"], ascending=False)
        .iloc[0]
    )
    return {key: str(fallback[key]) for key in ["label_target", "feature_set", "model", "split"]}


def filter_prediction_group(merged: pd.DataFrame, group: dict[str, str]) -> pd.DataFrame:
    """Filter merged predictions to one model group."""
    mask = (
        merged["label_target"].eq(group["label_target"])
        & merged["feature_set"].eq(group["feature_set"])
        & merged["model"].eq(group["model"])
        & merged["split"].eq(group["split"])
    )
    out = merged.loc[mask].copy()
    if out.empty:
        raise ValueError(f"No rows found for prediction group: {group}")
    return out


def topk_transition_composition(merged: pd.DataFrame | None) -> tuple[pd.DataFrame, dict[str, str] | None]:
    """Aggregate annual top-k alert composition by transition type for the main group."""
    if merged is None or merged.empty:
        return pd.DataFrame(), None
    group = choose_main_prediction_group(merged)
    data = filter_prediction_group(merged, group)
    rows: list[dict[str, object]] = []
    for k in FIXED_TOPK_BUDGETS:
        selected_parts = []
        for year, year_df in data.groupby("year", sort=True):
            ranked = year_df.sort_values(
                ["y_proba", "cell_id", "original_row"],
                ascending=[False, True, True],
                kind="mergesort",
            ).head(min(k, len(year_df)))
            selected_parts.append(ranked)
        selected = pd.concat(selected_parts, ignore_index=True)
        total_selected = len(selected)
        for transition_type, part in selected.groupby("transition_type", dropna=False, sort=True):
            selected_count = int(len(part))
            rows.append(
                {
                    **group,
                    "budget_k": int(k),
                    "transition_type": transition_type,
                    "selected_count": selected_count,
                    "selected_share": selected_count / total_selected if total_selected else np.nan,
                    "true_decline_count": int(part["y_true"].sum()),
                    "false_alert_count": int((part["y_true"] == 0).sum()),
                    "actionable_low_entry_count": int(part["actionable_low_entry_flag"].sum()),
                    "sharp_drop_30_count": int(part["sharp_drop_30_flag"].sum()),
                    "new_low_transition_count": int(part["new_low_transition_flag"].sum()),
                    "persistent_low_count": int(part["persistent_low_flag"].sum()),
                }
            )
    return pd.DataFrame(rows), group


def setup_axis(ax: plt.Axes, xlabel: str | None = None, ylabel: str | None = None) -> None:
    """Apply restrained report-friendly chart styling."""
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(True, color=TOKENS["grid"], linewidth=0.8, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])


def add_header(fig: plt.Figure, ax: plt.Axes, title: str, subtitle: str) -> None:
    """Add a chart title and subtitle with consistent spacing."""
    ax.set_title("")
    fig.subplots_adjust(top=0.84)
    left = ax.get_position().x0
    fig.text(
        left,
        0.96,
        textwrap.fill(title, width=78),
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.90,
        textwrap.fill(subtitle, width=112),
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )


def state_matrix_for_plot(taxonomy: pd.DataFrame, value: str) -> pd.DataFrame:
    """Return a full state-by-state matrix for plotting."""
    counts = (
        taxonomy.groupby(["current_reference_state", "next_reference_state"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    if value == "probability":
        counts["row_total"] = counts.groupby("current_reference_state")["count"].transform("sum")
        counts[value] = counts["count"] / counts["row_total"]
    else:
        counts[value] = counts["count"]
    matrix = counts.pivot(
        index="current_reference_state",
        columns="next_reference_state",
        values=value,
    ).reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
    return matrix.fillna(0)


def plot_transition_matrix(taxonomy: pd.DataFrame, output: Path, value: str) -> None:
    """Plot state transition heatmap as counts or row-normalized probabilities."""
    matrix = state_matrix_for_plot(taxonomy, value)
    cmap = LinearSegmentedColormap.from_list("reference_blue", ["#FFFFFF", "#CEDFFE", "#5477C4", "#2E4780"])
    fig, ax = plt.subplots(figsize=(8.0, 6.2), facecolor=TOKENS["surface"])
    image = ax.imshow(matrix.values, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_xticklabels([STATE_LABELS.get(label, label) for label in matrix.columns], fontsize=9)
    ax.set_yticklabels([STATE_LABELS.get(label, label) for label in matrix.index], fontsize=9)
    ax.set_xlabel("Next-year reference state")
    ax.set_ylabel("Current-year reference state")
    threshold = np.nanmax(matrix.values) * 0.55 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            label = f"{val:.2f}" if value == "probability" else f"{int(val)}"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color="#FFFFFF" if val > threshold else TOKENS["ink"],
                fontsize=8,
            )
    title = "Reference-state transition matrix"
    subtitle = (
        "Counts of current-to-next-year canopy reference-state transitions."
        if value == "count"
        else "Row-normalized probabilities within each current canopy reference state."
    )
    add_header(fig, ax, title, subtitle)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.set_ylabel("Count" if value == "count" else "Transition probability")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_transition_types_by_year(taxonomy: pd.DataFrame, output: Path) -> None:
    """Plot yearly transition type counts as stacked bars."""
    counts = (
        taxonomy.groupby(["year", "transition_type"], dropna=False)
        .size()
        .reset_index(name="count")
        .pivot(index="year", columns="transition_type", values="count")
        .reindex(columns=TRANSITION_ORDER, fill_value=0)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(11.5, 6.2), facecolor=TOKENS["surface"])
    bottom = np.zeros(len(counts))
    x = np.arange(len(counts.index))
    for transition_type in TRANSITION_ORDER:
        values = counts[transition_type].to_numpy()
        if values.sum() == 0:
            continue
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=TRANSITION_COLORS[transition_type],
            edgecolor="#FFFFFF",
            linewidth=0.3,
            label=transition_type.replace("_", " "),
        )
        bottom += values
    ax.set_xticks(x[::4])
    ax.set_xticklabels(counts.index[::4].astype(int), rotation=0)
    setup_axis(ax, "Year", "Grid-year observations")
    ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.03))
    add_header(
        fig,
        ax,
        "Transition type counts by year",
        "Annual count of persistent low, recovery, new low-transition, and stable non-low states.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_current_vs_next(taxonomy: pd.DataFrame, output: Path) -> None:
    """Plot current versus next-year canopy colored by transition type."""
    fig, ax = plt.subplots(figsize=(7.2, 6.4), facecolor=TOKENS["surface"])
    for transition_type in TRANSITION_ORDER:
        part = taxonomy.loc[taxonomy["transition_type"].eq(transition_type)]
        if part.empty:
            continue
        ax.scatter(
            part["current_relative_canopy"],
            part["next_relative_canopy"],
            s=18,
            alpha=0.62,
            color=TRANSITION_COLORS[transition_type],
            edgecolor="none",
            label=transition_type.replace("_", " "),
        )
    max_value = float(
        np.nanmax(
            [
                taxonomy["current_relative_canopy"].max(),
                taxonomy["next_relative_canopy"].max(),
            ]
        )
    )
    ax.plot([0, max_value], [0, max_value], linestyle="--", color="#7A828F", linewidth=1.2, label="1:1 reference")
    setup_axis(ax, "Current relative canopy", "Next-year relative canopy")
    ax.set_xlim(0, max_value * 1.04)
    ax.set_ylim(0, max_value * 1.04)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    add_header(
        fig,
        ax,
        "Current versus next-year relative canopy",
        "Each point is a retained 10 km cell-year, colored by the reference-point transition taxonomy.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_risk_by_transition(merged: pd.DataFrame | None, output: Path) -> bool:
    """Plot predicted risk distributions by transition type for the main model group."""
    if merged is None or merged.empty:
        return False
    group = choose_main_prediction_group(merged)
    data = filter_prediction_group(merged, group)
    ordered_types = [transition_type for transition_type in TRANSITION_ORDER if transition_type in set(data["transition_type"])]
    if not ordered_types:
        return False
    values = [data.loc[data["transition_type"].eq(transition_type), "y_proba"].dropna().to_numpy() for transition_type in ordered_types]
    fig, ax = plt.subplots(figsize=(9.0, 6.0), facecolor=TOKENS["surface"])
    box = ax.boxplot(values, patch_artist=True, widths=0.6, showfliers=False)
    for patch, transition_type in zip(box["boxes"], ordered_types, strict=True):
        patch.set_facecolor(TRANSITION_COLORS[transition_type])
        patch.set_alpha(0.85)
        patch.set_edgecolor("#464C55")
    for element in ["whiskers", "caps", "medians"]:
        for artist in box[element]:
            artist.set_color("#464C55")
            artist.set_linewidth(1.2)
    ax.set_xticklabels([label.replace("_", "\n") for label in ordered_types], fontsize=8)
    setup_axis(ax, None, "Predicted decline probability")
    ax.set_ylim(0, 1.02)
    add_header(
        fig,
        ax,
        "Model risk by transition type",
        f"Distribution for {group['feature_set']} / {group['model']} / {group['split']}; risk scores are not causal evidence.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_topk_composition(composition: pd.DataFrame, output: Path) -> bool:
    """Plot top-k alert composition by transition type."""
    if composition.empty:
        return False
    pivot = (
        composition.pivot_table(
            index="budget_k",
            columns="transition_type",
            values="selected_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=TRANSITION_ORDER, fill_value=0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(8.6, 6.0), facecolor=TOKENS["surface"])
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot.index))
    for transition_type in TRANSITION_ORDER:
        values = pivot[transition_type].to_numpy()
        if values.sum() == 0:
            continue
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=TRANSITION_COLORS[transition_type],
            edgecolor="#FFFFFF",
            linewidth=0.4,
            label=transition_type.replace("_", " "),
        )
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.astype(int))
    setup_axis(ax, "Annual alert budget: top k cells per year", "Selected cell-year alerts")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.03))
    add_header(
        fig,
        ax,
        "Annual top-k alert composition",
        "Selected alerts for the main model group, aggregated across test years and colored by transition taxonomy.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return True


def dataframe_to_markdown(data: pd.DataFrame) -> str:
    """Render a compact dataframe as Markdown."""
    display = data.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
        else:
            display[column] = display[column].astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy()]
    return "\n".join([header, separator, *rows])


def transition_share_lookup(summary: pd.DataFrame) -> dict[str, float]:
    """Return full-observation shares by transition type."""
    total = summary["n_observations"].sum()
    if total == 0:
        return {}
    grouped = summary.groupby("transition_type")["n_observations"].sum()
    return {key: float(value / total) for key, value in grouped.items()}


def summarize_topk_for_report(composition: pd.DataFrame) -> pd.DataFrame:
    """Return compact top 5/top 20 composition rows."""
    if composition.empty:
        return pd.DataFrame()
    selected = composition.loc[composition["budget_k"].isin([5, 20])].copy()
    return selected[
        [
            "budget_k",
            "transition_type",
            "selected_count",
            "selected_share",
            "true_decline_count",
            "false_alert_count",
            "actionable_low_entry_count",
            "new_low_transition_count",
            "persistent_low_count",
        ]
    ].sort_values(["budget_k", "selected_count"], ascending=[True, False])


def write_report(
    output: Path,
    input_path: Path,
    prediction_path: Path,
    taxonomy: pd.DataFrame,
    matrix: pd.DataFrame,
    summary: pd.DataFrame,
    model_risk: pd.DataFrame,
    composition: pd.DataFrame,
    baseline_period: str,
    reference_notes: list[str],
    input_notes: list[str],
    prediction_notes: list[str],
    main_group: dict[str, str] | None,
) -> None:
    """Write a Markdown report for the reference taxonomy analysis."""
    transition_counts = (
        taxonomy["transition_type"]
        .value_counts()
        .rename_axis("transition_type")
        .reset_index(name="n_observations")
    )
    transition_counts["share_of_observations"] = transition_counts["n_observations"] / len(taxonomy)
    transition_counts = transition_counts.set_index("transition_type").reindex(TRANSITION_ORDER).dropna(how="all").reset_index()
    transition_counts["n_observations"] = transition_counts["n_observations"].astype(int)

    state_matrix = state_matrix_for_plot(taxonomy, "count").reset_index().rename(columns={"current_reference_state": "current_state"})
    topk_table = summarize_topk_for_report(composition)

    model_risk_note = "Model predictions were not merged."
    if not model_risk.empty and main_group is not None:
        main_risk = model_risk.loc[
            model_risk["label_target"].eq(main_group["label_target"])
            & model_risk["feature_set"].eq(main_group["feature_set"])
            & model_risk["model"].eq(main_group["model"])
            & model_risk["split"].eq(main_group["split"])
        ].sort_values("mean_predicted_probability", ascending=False)
        if not main_risk.empty:
            top_row = main_risk.iloc[0]
            model_risk_note = (
                f"For the main group `{main_group['label_target']} / {main_group['feature_set']} / "
                f"{main_group['model']} / {main_group['split']}`, the highest mean predicted probability "
                f"was in `{top_row['transition_type']}` (`{top_row['mean_predicted_probability']:.3f}`)."
            )

    lines = [
        "# Reference-Point State Taxonomy Summary",
        "",
        "## Purpose",
        "",
        "This analysis classifies retained 10 km Kelpwatch cell-years into site-specific canopy reference states and transition types. It is an ecological state-assessment and decision-support diagnostic, not proof of fully operational early-warning skill.",
        "",
        "## Inputs",
        "",
        f"- Modeling panel: `{input_path}`",
        f"- Prediction file: `{prediction_path}`" if not model_risk.empty else f"- Prediction file: `{prediction_path}` (not merged)",
        f"- Reference period: `{baseline_period}`",
        f"- Number of cells: `{taxonomy['cell_id'].nunique():,}`",
        f"- Grid-year observations: `{len(taxonomy):,}`",
        f"- Available splits: `{', '.join(sorted(taxonomy['split'].astype(str).unique()))}`",
        "",
        "## Reference-State Definitions",
        "",
        "- `low_below_p25`: relative canopy < cell-specific q25.",
        "- `lower_mid_p25_p50`: q25 <= relative canopy < q50.",
        "- `upper_mid_p50_p75`: q50 <= relative canopy < q75.",
        "- `high_above_p75`: relative canopy >= q75.",
        "",
        "## Transition-Type Definitions",
        "",
        "- `persistent_low`: current and next-year state are both below p25.",
        "- `recovery_from_low`: current state is below p25 and next-year state is not below p25.",
        "- `new_low_transition`: current state is not below p25 and next-year state is below p25.",
        "- `stable_non_low`: neither current nor next-year state is below p25.",
        "- `missing_transition`: current or next-year state could not be assigned.",
        "",
        "## Transition Counts",
        "",
        dataframe_to_markdown(transition_counts),
        "",
        "## State-Transition Matrix Counts",
        "",
        dataframe_to_markdown(state_matrix),
        "",
        "## Model-Risk Linkage",
        "",
        model_risk_note,
        "",
    ]

    if not composition.empty and main_group is not None:
        lines.extend(
            [
                "## Annual Top-k Transition Composition",
                "",
                f"Main group: `{main_group['label_target']} / {main_group['feature_set']} / {main_group['model']} / {main_group['split']}`.",
                "",
                dataframe_to_markdown(topk_table),
                "",
            ]
        )
    else:
        lines.extend(["## Annual Top-k Transition Composition", "", "Top-k transition composition was not generated because prediction merge was unavailable.", ""])

    all_notes = [*input_notes, *reference_notes, *prediction_notes]
    if all_notes:
        lines.extend(["## Assumptions and Warnings", ""])
        lines.extend([f"- {note}" for note in all_notes])
        lines.append("")

    lines.extend(
        [
            "## Limitations",
            "",
            "- Reference states are retrospective labels used for evaluation and interpretation; they are not model inputs in the original workflow.",
            "- q25/q50/q75 depend on the selected baseline period.",
            "- Persistent low states should not be interpreted as new early-warning success.",
            "- Sharp drops and low-state entry are related but not identical outcomes.",
            "- The taxonomy supports cautious decision-support interpretation, not causal attribution.",
            "",
            "## Outputs",
            "",
            "- `outputs/tables/reference_point_state_taxonomy.csv`",
            "- `outputs/tables/reference_point_transition_matrix.csv`",
            "- `outputs/tables/reference_point_transition_matrix_by_year.csv`",
            "- `outputs/tables/reference_point_transition_summary.csv`",
            "- `outputs/tables/reference_point_transition_summary_by_year.csv`",
            "- `outputs/tables/model_risk_by_transition_type.csv`",
            "- `outputs/tables/topk_transition_composition.csv`",
            "- `outputs/figures/reference_point_transition_matrix_counts.png`",
            "- `outputs/figures/reference_point_transition_matrix_probabilities.png`",
            "- `outputs/figures/reference_point_transition_types_by_year.png`",
            "- `outputs/figures/reference_point_current_vs_next_canopy.png`",
            "- `outputs/figures/model_risk_by_transition_type.png`",
            "- `outputs/figures/topk_transition_composition.png`",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the reference-point state taxonomy workflow."""
    args = parse_args()
    data, mapping, input_notes = load_taxonomy_input(args.input)
    reference, baseline_period, reference_notes = compute_reference_points(data, mapping)
    taxonomy = build_taxonomy(data, mapping, reference)
    matrix = transition_matrix(taxonomy, by_year=False)
    matrix_by_year = transition_matrix(taxonomy, by_year=True)
    summary = transition_summary(taxonomy, by_year=False)
    summary_by_year = transition_summary(taxonomy, by_year=True)

    predictions, prediction_notes = load_predictions(args.predictions, args.label_target)
    merged_predictions, merge_notes = merge_predictions_with_taxonomy(taxonomy, predictions)
    prediction_notes.extend(merge_notes)
    model_risk = model_risk_by_transition(merged_predictions)
    composition, main_group = topk_transition_composition(merged_predictions)

    write_portable_csv(taxonomy, args.taxonomy_output)
    write_portable_csv(matrix, args.transition_matrix_output)
    write_portable_csv(matrix_by_year, args.transition_matrix_by_year_output)
    write_portable_csv(summary, args.transition_summary_output)
    write_portable_csv(summary_by_year, args.transition_summary_by_year_output)
    if not model_risk.empty:
        write_portable_csv(model_risk, args.model_risk_output)
    if not composition.empty:
        write_portable_csv(composition, args.topk_composition_output)

    plot_transition_matrix(taxonomy, args.figure_dir / MATRIX_COUNTS_FIGURE.name, "count")
    plot_transition_matrix(taxonomy, args.figure_dir / MATRIX_PROBABILITIES_FIGURE.name, "probability")
    plot_transition_types_by_year(taxonomy, args.figure_dir / TYPES_BY_YEAR_FIGURE.name)
    plot_current_vs_next(taxonomy, args.figure_dir / CURRENT_NEXT_FIGURE.name)
    risk_figure_written = plot_model_risk_by_transition(merged_predictions, args.figure_dir / MODEL_RISK_FIGURE.name)
    topk_figure_written = plot_topk_composition(composition, args.figure_dir / TOPK_COMPOSITION_FIGURE.name)

    write_report(
        args.report_output,
        args.input,
        args.predictions,
        taxonomy,
        matrix,
        summary,
        model_risk,
        composition,
        baseline_period,
        reference_notes,
        input_notes,
        prediction_notes,
        main_group,
    )

    shares = transition_share_lookup(summary)
    print("Reference-point state taxonomy complete.")
    print(f"Input panel: {args.input}")
    print(f"Baseline/reference period: {baseline_period}")
    print(f"Taxonomy rows: {len(taxonomy):,}")
    print(f"Cells: {taxonomy['cell_id'].nunique():,}")
    print(f"Transition matrix rows: {len(matrix):,}")
    print(f"Transition summary rows: {len(summary):,}")
    print(f"Prediction merge: {'yes' if merged_predictions is not None and not merged_predictions.empty else 'no'}")
    print(f"Top-k transition composition: {'yes' if not composition.empty else 'no'}")
    print(f"Risk figure written: {risk_figure_written}")
    print(f"Top-k composition figure written: {topk_figure_written}")
    for key in ["persistent_low", "new_low_transition", "recovery_from_low", "stable_non_low"]:
        print(f"Share {key}: {shares.get(key, 0.0):.3f}")
    print(f"Wrote taxonomy: {args.taxonomy_output}")
    print(f"Wrote report: {args.report_output}")


if __name__ == "__main__":
    main()
