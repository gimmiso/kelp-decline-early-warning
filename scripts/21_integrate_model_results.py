"""Integrate model-result tables into one synthesis layer.

This script does not rerun models or feature construction. It reads existing
result CSVs, harmonizes their schemas, and writes portfolio-level comparison
tables plus a diagnostic report.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "tables"
DIAGNOSTICS_DIR = ROOT / "outputs" / "diagnostics"

MASTER_OUT = RESULTS_DIR / "integrated_model_comparison_master.csv"
BEST_OUT = RESULTS_DIR / "integrated_best_by_target.csv"
GAP_OUT = RESULTS_DIR / "integrated_baseline_gap_summary.csv"
REPORT_OUT = DIAGNOSTICS_DIR / "integrated_model_results_report.md"

CSV_WRITE_KWARGS = {
    "index": False,
    "lineterminator": "\n",
    "na_rep": "",
    "float_format": "%.6f",
}


EXPECTED_RESULT_FILES = [
    "results/tables/model_comparison_results.csv",
    "results/tables/multiscale_model_comparison.csv",
    "results/tables/naive_persistence_baseline_comparison.csv",
    "results/tables/ml_vs_naive_baseline_gap.csv",
    "results/tables/crw5km_composite_model_comparison.csv",
    "results/tables/bathymetry_habitat_model_comparison.csv",
    "results/tables/canopy_trajectory_model_comparison.csv",
    "results/tables/crw5km_model_comparison.csv",
    "results/tables/high_canopy_subgroup_performance.csv",
    "outputs/metadata/model_comparison_results.csv",
    "outputs/metadata/model_comparison_test_metrics.csv",
    "outputs/metadata/model_diagnostics_same_model_comparison.csv",
    "outputs/metadata/threshold_tuning_test_results.csv",
    "outputs/model_results/actionable_decline_model_performance.csv",
    "outputs/model_results/cost_sensitive_model_performance.csv",
    "outputs/model_results/environment_incremental_value_performance.csv",
    "outputs/model_results/feature_ablation_performance.csv",
    "outputs/model_results/extended_threshold_selection_summary.csv",
    "outputs/model_results/oisst_matching_sensitivity_model_performance.csv",
    "outputs/model_results/threshold_selection_summary.csv",
    "outputs/diagnostics/at_risk_subset_model_performance.csv",
    "outputs/diagnostics/new_decline_transition_model_performance.csv",
]

MASTER_COLUMNS = [
    "source_table",
    "experiment_layer",
    "target_definition",
    "normalized_target",
    "feature_family",
    "normalized_feature_family",
    "model",
    "learning_strategy",
    "threshold",
    "pr_auc",
    "roc_auc",
    "recall",
    "precision",
    "f1",
    "f2",
    "false_negatives",
    "false_positives",
    "true_positives",
    "true_negatives",
    "train_positive_count",
    "validation_positive_count",
    "test_positive_count",
    "n_train",
    "n_validation",
    "n_test",
    "event_prevalence_test",
    "status",
    "notes",
]

TARGET_ORDER = [
    "original_decline",
    "at_risk_original",
    "new_transition",
    "actionable_drop",
    "high_canopy_subgroup",
    "unknown_target",
]

ENVIRONMENT_ONLY_FAMILIES = {
    "OISST_only",
    "CRW_composite_only",
    "bathymetry_habitat_only",
    "multiscale_environment",
}

CANOPY_FAMILIES = {
    "canopy_only",
    "canopy_trajectory",
    "canopy_plus_CRW",
    "canopy_plus_trajectory",
    "canopy_plus_CRW_plus_habitat",
}

BASELINE_EXCLUDE_FOR_COMBINED = {
    "naive_persistence",
    "canopy_only",
    "OISST_only",
    "CRW_composite_only",
    "bathymetry_habitat_only",
}


def relative_path(path: Path) -> str:
    return str(path.relative_to(ROOT))


def safe_float(value) -> float:
    if value is None:
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def clean_csv_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with object-cell line breaks flattened for GitHub previews."""
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
    """Write CSVs with stable LF newlines and pandas/GitHub-friendly formatting."""
    clean_csv_cells(df).to_csv(path, **CSV_WRITE_KWARGS)


def first_present(row: pd.Series, names: Iterable[str], default=None):
    for name in names:
        if name in row.index:
            value = row[name]
            if not pd.isna(value):
                return value
    return default


def normalize_target(raw: object, context: str = "") -> str:
    text = f"{raw} {context}".lower()
    text = text.replace("-", "_")

    if "high_canopy" in text or "historical_p75" in text or "p75" in text:
        return "high_canopy_subgroup"
    if "actionable" in text or "drop" in text or "low_next" in text:
        return "actionable_drop"
    if "new_decline" in text or "new_transition" in text or "transition" in text:
        return "new_transition"
    if "at_risk" in text or "gt005" in text or "gt0" in text or "current_canopy_" in text:
        return "at_risk_original"
    if "original" in text or "decline_event_next" in text or "decline_state" in text:
        return "original_decline"
    return "unknown_target"


def normalize_feature_family(raw: object, source_table: str = "", context: str = "") -> str:
    text = f"{raw} {context}".lower()
    source_text = source_table.lower()
    text = text.replace("-", "_")

    if "threshold" in source_text or "threshold" in text:
        return "threshold_tuned"
    if "cost_sensitive" in text or "rare_event" in text or "class_weight" in text:
        return "rare_event_learning"
    if "current_lag_slope" in text or "lag_slope" in text:
        return "canopy_trajectory"
    if "naive" in text or "persistence_rule" in text or "persistence_baseline" in text:
        return "naive_persistence"
    if "multiscale" in source_text or "multiscale" in text or "buffer" in text or "idw" in text:
        return "multiscale_environment"
    if "oisst" in text and "crw" in text and "canopy" not in text:
        return "multiscale_environment"
    if "trajectory_plus_crw_plus_habitat" in text or (
        "trajectory" in text and "crw" in text and "habitat" in text
    ):
        return "canopy_plus_CRW_plus_habitat"
    if "canopy" in text and "crw" in text and "habitat" in text:
        return "canopy_plus_CRW_plus_habitat"
    if "trajectory" in text and "crw" in text:
        return "canopy_plus_CRW"
    if "canopy" in text and "crw" in text:
        return "canopy_plus_CRW"
    if "trajectory" in text and ("plus" in text or "canopy" in text):
        return "canopy_plus_trajectory" if "plus" in text else "canopy_trajectory"
    if "canopy_trajectory" in text or "trajectory_only" in text:
        return "canopy_trajectory"
    if "canopy_current" in text or "canopy_only" in text or "existing_canopy" in text:
        return "canopy_only"
    if "oisst" in text and "habitat" in text:
        return "OISST_plus_habitat"
    if "crw" in text and "habitat" in text:
        return "CRW_plus_habitat"
    if "crw" in text or "coraltemp" in text or "crw5km" in text:
        return "CRW_composite_only"
    if "habitat" in text or "bathymetry" in text:
        return "bathymetry_habitat_only"
    if "oisst" in text or "environment_only" in text or "noaa" in text:
        return "OISST_only"
    if "crw5km" in source_text and raw in (None, "", "other"):
        return "CRW_composite_only"
    return "other"


def infer_experiment_layer(path: Path) -> str:
    name = path.name.lower()
    parent = path.parent.name.lower()
    rel = relative_path(path).lower()

    if "canopy_trajectory" in name:
        return "trajectory"
    if "bathymetry" in name or "habitat" in name:
        return "bathymetry_habitat"
    if "crw5km_composite" in name:
        return "CRW_composite"
    if "crw5km" in name:
        return "CRW_daily_feasibility"
    if "multiscale" in name:
        return "multiscale_environment"
    if "naive_persistence" in name:
        return "naive_persistence"
    if "threshold" in name:
        return "threshold_tuned"
    if "cost_sensitive" in name or "actionable" in name or "feature_ablation" in name:
        return "rare_event_learning"
    if "at_risk" in name or "new_decline" in name:
        return "validity_diagnostics"
    if "model_comparison" in name and parent == "metadata":
        return "V1"
    if "environment_incremental" in name or "oisst_matching" in name:
        return "OISST_sensitivity"
    if "results/tables" in rel:
        return "V2_extension"
    return "other"


def discover_result_files() -> tuple[list[Path], list[Path]]:
    expected = [ROOT / p for p in EXPECTED_RESULT_FILES]
    found = [p for p in expected if p.exists()]
    missing = [p for p in expected if not p.exists()]

    optional_patterns = [
        "results/tables/*model*comparison*.csv",
        "results/tables/*performance*.csv",
        "outputs/model_results/*performance*.csv",
        "outputs/model_results/*selection_summary*.csv",
        "outputs/metadata/*model_comparison*.csv",
        "outputs/metadata/*threshold*tuning*test*.csv",
        "outputs/diagnostics/*model_performance*.csv",
    ]
    for pattern in optional_patterns:
        for path in ROOT.glob(pattern):
            if any(
                skip in path.name.lower()
                for skip in [
                    "integrated_",
                    "predictions",
                    "confusion_matrices",
                    "selected_thresholds",
                    "validation_grid",
                    "tuning_results",
                ]
            ):
                continue
            if path.exists() and path not in found:
                found.append(path)

    found = sorted(found, key=lambda p: relative_path(p))
    return found, missing


def row_status(row: pd.Series, path: Path) -> str:
    status = first_present(row, ["status"], default="computed")
    status = str(status)
    if "dry_run" in status.lower() or "not_run" in status.lower():
        return status
    if "tuning_results" in path.name and "selection_summary" not in path.name:
        return "validation_grid_not_ranked"
    return status


def standardize_rows(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    rows: list[dict] = []
    source = relative_path(path)
    experiment_layer = infer_experiment_layer(path)

    for _, row in df.iterrows():
        if "split" in row.index and str(row["split"]).lower() != "test":
            continue

        status = row_status(row, path)

        default_target = "unknown"
        if experiment_layer == "V1" or "threshold_tuning_test_results" in source:
            default_target = "decline_event_next"
        if "threshold_selection_summary" in source:
            default_target = "decline_event_next"

        target_raw = first_present(
            row,
            ["target_definition", "target", "evaluation_subset", "evaluation_subgroup"],
            default=default_target,
        )
        context = " ".join(
            str(first_present(row, [c], default=""))
            for c in ["evaluation_context", "evaluation_subset", "evaluation_subgroup"]
        )
        normalized_target = normalize_target(target_raw, context)

        feature_raw = first_present(
            row,
            ["feature_family", "feature_set", "model_group", "comparison", "best_ml_group"],
            default="other",
        )
        model = str(first_present(row, ["model", "model_name", "model_family", "best_ml_model"], default="unknown"))
        learning_strategy = str(
            first_present(
                row,
                ["model_variant", "selection_rule", "validation_design", "evaluation_context", "scale"],
                default="",
            )
        )
        normalized_family = normalize_feature_family(feature_raw, source, f"{model} {learning_strategy}")

        threshold = first_present(row, ["decision_threshold", "threshold", "selected_threshold"], default=np.nan)

        pr_auc = first_present(row, ["pr_auc", "test_pr_auc", "best_ml_pr_auc"], default=np.nan)
        roc_auc = first_present(row, ["roc_auc", "test_roc_auc"], default=np.nan)
        recall = first_present(row, ["recall", "test_recall", "best_ml_recall"], default=np.nan)
        precision = first_present(row, ["precision", "test_precision", "best_ml_precision"], default=np.nan)
        f1 = first_present(row, ["f1", "test_f1"], default=np.nan)
        f2 = first_present(row, ["f2", "test_f2"], default=np.nan)

        false_negatives = first_present(row, ["false_negatives", "fn", "test_false_negatives"], default=np.nan)
        false_positives = first_present(row, ["false_positives", "fp", "test_false_positives"], default=np.nan)
        true_positives = first_present(row, ["true_positives", "tp", "test_true_positives"], default=np.nan)
        true_negatives = first_present(row, ["true_negatives", "tn", "test_true_negatives"], default=np.nan)

        n_train = first_present(row, ["n_train"], default=np.nan)
        n_validation = first_present(row, ["n_validation"], default=np.nan)
        n_test = first_present(row, ["n_test", "n_observations", "n_rows"], default=np.nan)
        test_positive_count = first_present(
            row,
            [
                "positive_events_test",
                "test_positive_events",
                "n_positive_events",
                "n_positive_events_test",
            ],
            default=np.nan,
        )
        event_prevalence = first_present(
            row,
            ["event_prevalence_test", "test_event_prevalence", "event_prevalence", "positive_rate"],
            default=np.nan,
        )

        notes_parts = []
        if status != "computed":
            notes_parts.append(f"status={status}")
        if "evaluation_subset" in row.index and not pd.isna(row["evaluation_subset"]):
            notes_parts.append(f"subset={row['evaluation_subset']}")
        if "evaluation_subgroup" in row.index and not pd.isna(row["evaluation_subgroup"]):
            notes_parts.append(f"subgroup={row['evaluation_subgroup']}")
        if "current_canopy_threshold" in row.index and not pd.isna(row["current_canopy_threshold"]):
            notes_parts.append(f"current_canopy_threshold={row['current_canopy_threshold']}")
        if "oisst_matching_variant" in row.index and not pd.isna(row["oisst_matching_variant"]):
            notes_parts.append(f"oisst_matching_variant={row['oisst_matching_variant']}")
        if "held_out_region" in row.index and not pd.isna(row["held_out_region"]):
            notes_parts.append(f"held_out_region={row['held_out_region']}")

        rows.append(
            {
                "source_table": source,
                "experiment_layer": experiment_layer,
                "target_definition": str(target_raw),
                "normalized_target": normalized_target,
                "feature_family": str(feature_raw),
                "normalized_feature_family": normalized_family,
                "model": model,
                "learning_strategy": learning_strategy if learning_strategy != "nan" else "",
                "threshold": safe_float(threshold),
                "pr_auc": safe_float(pr_auc),
                "roc_auc": safe_float(roc_auc),
                "recall": safe_float(recall),
                "precision": safe_float(precision),
                "f1": safe_float(f1),
                "f2": safe_float(f2),
                "false_negatives": safe_float(false_negatives),
                "false_positives": safe_float(false_positives),
                "true_positives": safe_float(true_positives),
                "true_negatives": safe_float(true_negatives),
                "train_positive_count": np.nan,
                "validation_positive_count": np.nan,
                "test_positive_count": safe_float(test_positive_count),
                "n_train": safe_float(n_train),
                "n_validation": safe_float(n_validation),
                "n_test": safe_float(n_test),
                "event_prevalence_test": safe_float(event_prevalence),
                "status": status,
                "notes": "; ".join(notes_parts),
            }
        )

    return rows


def rankable(master: pd.DataFrame) -> pd.DataFrame:
    excluded_statuses = {"dry_run_no_local_crw_cache", "validation_grid_not_ranked"}
    out = master[~master["status"].isin(excluded_statuses)].copy()
    out = out[out["pr_auc"].notna() | out["recall"].notna() | out["f2"].notna()]
    return out


def max_value(df: pd.DataFrame, column: str, mask=None) -> float:
    if mask is not None:
        df = df[mask]
    if df.empty or column not in df.columns:
        return np.nan
    value = df[column].dropna().max()
    return float(value) if pd.notna(value) else np.nan


def min_value(df: pd.DataFrame, column: str, mask=None) -> float:
    if mask is not None:
        df = df[mask]
    if df.empty or column not in df.columns:
        return np.nan
    value = df[column].dropna().min()
    return float(value) if pd.notna(value) else np.nan


def best_row(df: pd.DataFrame, column: str, ascending: bool = False) -> pd.Series | None:
    candidates = df[df[column].notna()].copy()
    if candidates.empty:
        return None
    candidates = candidates.sort_values(
        by=[column, "pr_auc", "recall"],
        ascending=[ascending, False, False],
        na_position="last",
    )
    return candidates.iloc[0]


def format_model(row: pd.Series | None) -> tuple[str, str, str, float]:
    if row is None:
        return "", "", "", np.nan
    return (
        str(row["experiment_layer"]),
        str(row["normalized_feature_family"]),
        str(row["model"]),
        safe_float(row["pr_auc"]),
    )


def target_interpretation(target: str, row: pd.Series | None) -> str:
    family = "" if row is None else str(row["normalized_feature_family"])
    if target == "original_decline":
        return (
            "Broad decline-state screening; high PR-AUC should be interpreted as risk-state "
            f"prediction, especially when led by {family or 'the top feature family'}."
        )
    if target == "at_risk_original":
        return (
            "At-risk evaluation removes already-low canopy states and is a stronger early-warning "
            "stress test, but still may reflect persistence and spatial risk."
        )
    if target == "new_transition":
        return (
            "Strict transition-oriented target; useful performance here would be stronger evidence "
            "for early-warning skill than original-label performance."
        )
    if target == "actionable_drop":
        return (
            "Actionable drop target emphasizes warning sensitivity; false-negative reduction must "
            "be balanced against precision."
        )
    if target == "high_canopy_subgroup":
        return (
            "High-canopy subgroup asks whether models detect decline from healthier states; this is "
            "closer to transition monitoring than broad low-state detection."
        )
    return "Target parsing was incomplete; inspect source table notes before making claims."


def build_best_by_target(master: pd.DataFrame) -> pd.DataFrame:
    usable = rankable(master)
    records = []

    for target in TARGET_ORDER:
        target_df = usable[usable["normalized_target"] == target]
        if target_df.empty:
            continue

        best_pr = best_row(target_df, "pr_auc")
        best_recall = best_row(target_df, "recall")
        best_f2 = best_row(target_df, "f2")
        best_fn = best_row(target_df, "false_negatives", ascending=True)

        env_mask = target_df["normalized_feature_family"].isin(ENVIRONMENT_ONLY_FAMILIES)
        canopy_mask = target_df["normalized_feature_family"].isin(CANOPY_FAMILIES)
        no_canopy_mask = ~target_df["normalized_feature_family"].isin(CANOPY_FAMILIES)
        combined_mask = ~target_df["normalized_feature_family"].isin(BASELINE_EXCLUDE_FOR_COMBINED)

        rec = {
            "target_definition": target,
            "best_pr_auc_experiment": "" if best_pr is None else best_pr["experiment_layer"],
            "best_pr_auc_feature_family": "" if best_pr is None else best_pr["normalized_feature_family"],
            "best_pr_auc_model": "" if best_pr is None else best_pr["model"],
            "best_pr_auc_value": np.nan if best_pr is None else best_pr["pr_auc"],
            "best_recall_experiment": "" if best_recall is None else best_recall["experiment_layer"],
            "best_recall_feature_family": "" if best_recall is None else best_recall["normalized_feature_family"],
            "best_recall_model": "" if best_recall is None else best_recall["model"],
            "best_recall_value": np.nan if best_recall is None else best_recall["recall"],
            "best_f2_experiment": "" if best_f2 is None else best_f2["experiment_layer"],
            "best_f2_feature_family": "" if best_f2 is None else best_f2["normalized_feature_family"],
            "best_f2_model": "" if best_f2 is None else best_f2["model"],
            "best_f2_value": np.nan if best_f2 is None else best_f2["f2"],
            "best_false_negative_model": "" if best_fn is None else best_fn["model"],
            "best_false_negative_count": np.nan if best_fn is None else best_fn["false_negatives"],
            "canopy_only_pr_auc": max_value(
                target_df,
                "pr_auc",
                target_df["normalized_feature_family"].eq("canopy_only"),
            ),
            "naive_persistence_pr_auc": max_value(
                target_df,
                "pr_auc",
                target_df["normalized_feature_family"].eq("naive_persistence"),
            ),
            "best_environment_only_pr_auc": max_value(target_df, "pr_auc", env_mask),
            "best_canopy_including_pr_auc": max_value(target_df, "pr_auc", canopy_mask),
            "best_canopy_excluding_pr_auc": max_value(target_df, "pr_auc", no_canopy_mask),
            "best_combined_pr_auc": max_value(target_df, "pr_auc", combined_mask),
            "main_interpretation": target_interpretation(target, best_pr),
        }
        records.append(rec)

    return pd.DataFrame(records)


def gap(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a - b)


def build_gap_summary(master: pd.DataFrame) -> pd.DataFrame:
    usable = rankable(master)
    records = []
    for target in TARGET_ORDER:
        target_df = usable[usable["normalized_target"] == target]
        if target_df.empty:
            continue

        best = max_value(target_df, "pr_auc")
        naive = max_value(target_df, "pr_auc", target_df["normalized_feature_family"].eq("naive_persistence"))
        canopy = max_value(target_df, "pr_auc", target_df["normalized_feature_family"].eq("canopy_only"))
        oisst = max_value(target_df, "pr_auc", target_df["normalized_feature_family"].eq("OISST_only"))
        crw = max_value(target_df, "pr_auc", target_df["normalized_feature_family"].eq("CRW_composite_only"))
        habitat = max_value(
            target_df,
            "pr_auc",
            target_df["normalized_feature_family"].eq("bathymetry_habitat_only"),
        )
        trajectory = max_value(
            target_df,
            "pr_auc",
            target_df["normalized_feature_family"].isin(["canopy_trajectory", "canopy_plus_trajectory"]),
        )
        environment = max_value(
            target_df,
            "pr_auc",
            target_df["normalized_feature_family"].isin(ENVIRONMENT_ONLY_FAMILIES),
        )

        canopy_fn = min_value(target_df, "false_negatives", target_df["normalized_feature_family"].eq("canopy_only"))
        best_fn = min_value(target_df, "false_negatives")

        records.append(
            {
                "target_definition": target,
                "best_model_pr_auc": best,
                "naive_persistence_pr_auc": naive,
                "canopy_only_pr_auc": canopy,
                "OISST_only_pr_auc": oisst,
                "CRW_only_pr_auc": crw,
                "habitat_only_pr_auc": habitat,
                "trajectory_pr_auc": trajectory,
                "best_environment_only_pr_auc": environment,
                "best_model_minus_naive_pr_auc": gap(best, naive),
                "best_model_minus_canopy_only_pr_auc": gap(best, canopy),
                "best_environment_minus_OISST_pr_auc": gap(environment, oisst),
                "CRW_minus_OISST_pr_auc": gap(crw, oisst),
                "habitat_minus_CRW_pr_auc": gap(habitat, crw),
                "trajectory_minus_canopy_pr_auc": gap(trajectory, canopy),
                "actionable_false_negative_reduction": gap(canopy_fn, best_fn),
            }
        )

    return pd.DataFrame(records)


def fmt(value: float, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def feature_layer_notes(master: pd.DataFrame) -> list[str]:
    usable = rankable(master)
    notes = []
    for family, label in [
        ("CRW_composite_only", "CRW 5 km composite"),
        ("bathymetry_habitat_only", "bathymetry/habitat"),
        ("canopy_trajectory", "canopy trajectory"),
        ("threshold_tuned", "threshold-tuned"),
        ("rare_event_learning", "rare-event/cost-sensitive"),
    ]:
        family_df = usable[usable["normalized_feature_family"].eq(family)]
        if family_df.empty:
            notes.append(f"- **{label}:** no rankable rows were found in the integrated tables.")
            continue
        best = best_row(family_df, "pr_auc")
        notes.append(
            "- **{}:** best PR-AUC {} on `{}` using `{}` / `{}`.".format(
                label,
                fmt(best["pr_auc"]) if best is not None else "NA",
                best["normalized_target"] if best is not None else "unknown",
                best["experiment_layer"] if best is not None else "unknown",
                best["model"] if best is not None else "unknown",
            )
        )
    return notes


def write_report(
    master: pd.DataFrame,
    best: pd.DataFrame,
    gaps: pd.DataFrame,
    found_files: list[Path],
    missing_files: list[Path],
) -> None:
    usable = rankable(master)
    targets = [t for t in TARGET_ORDER if t in set(master["normalized_target"])]
    found_lines = "\n".join(f"- `{relative_path(p)}`" for p in found_files)
    missing_lines = "\n".join(f"- `{relative_path(p)}`" for p in missing_files) or "- None"

    best_lines = []
    for _, row in best.iterrows():
        target = row["target_definition"]
        best_lines.append(
            "### {}\n\n"
            "- Best PR-AUC: `{}` / `{}` / `{}` = `{}`.\n"
            "- Best recall: `{}` / `{}` / `{}` = `{}`.\n"
            "- Best F2: `{}` / `{}` / `{}` = `{}`.\n"
            "- Best false-negative count: `{}` = `{}`.\n"
            "- Interpretation: {}\n".format(
                target,
                row["best_pr_auc_experiment"],
                row["best_pr_auc_feature_family"],
                row["best_pr_auc_model"],
                fmt(row["best_pr_auc_value"]),
                row["best_recall_experiment"],
                row["best_recall_feature_family"],
                row["best_recall_model"],
                fmt(row["best_recall_value"]),
                row["best_f2_experiment"],
                row["best_f2_feature_family"],
                row["best_f2_model"],
                fmt(row["best_f2_value"]),
                row["best_false_negative_model"],
                fmt(row["best_false_negative_count"], 0),
                row["main_interpretation"],
            )
        )

    gap_lines = []
    for _, row in gaps.iterrows():
        gap_lines.append(
            "- `{}`: best-minus-naive `{}`, best-minus-canopy `{}`, CRW-minus-OISST `{}`, "
            "trajectory-minus-canopy `{}`, false-negative reduction vs canopy `{}`.".format(
                row["target_definition"],
                fmt(row["best_model_minus_naive_pr_auc"]),
                fmt(row["best_model_minus_canopy_only_pr_auc"]),
                fmt(row["CRW_minus_OISST_pr_auc"]),
                fmt(row["trajectory_minus_canopy_pr_auc"]),
                fmt(row["actionable_false_negative_reduction"], 0),
            )
        )

    computed_rows = usable.shape[0]
    status_counts = master["status"].value_counts(dropna=False).to_dict()
    status_lines = "\n".join(f"- `{k}`: {v}" for k, v in status_counts.items())

    report = f"""# Integrated Model Results Report

## Overview

This report consolidates existing result tables across V1 model comparison, threshold tuning, recall-oriented extensions, OISST sensitivity, CRW 5 km composites, bathymetry/habitat, multiscale environmental exposure, naive persistence, and canopy trajectory diagnostics.

No heavy feature construction or model training was rerun. Missing expected files are recorded below rather than fabricated.

Integrated rows:

```text
Source result files found: {len(found_files)}
Missing expected files: {len(missing_files)}
Rows integrated: {len(master)}
Rankable computed rows: {computed_rows}
Targets detected: {len(targets)}
Detected target groups: {', '.join(targets)}
```

Status counts:

{status_lines}

Found result files:

{found_lines}

Missing expected result files:

{missing_lines}

## Target Hierarchy

- **Original broad decline-state screening (`original_decline`)** identifies whether next-year canopy falls below a cell-specific low-canopy threshold. Strong performance here can reflect current canopy state, persistent low canopy, or spatial risk structure.
- **At-risk original target (`at_risk_original`)** removes already-low or near-zero current canopy observations. This is a stronger test of whether models help before visible low-state persistence dominates.
- **New transition target (`new_transition`)** focuses on transition into a low-canopy state from a not-yet-low state. This is closer to true early-warning than broad original-label prediction.
- **Actionable decline/drop target (`actionable_drop`)** emphasizes meaningful next-year decline or low-canopy warning sensitivity. Recall and false negatives matter, but precision must remain interpretable.
- **High-canopy subgroup (`high_canopy_subgroup`)** asks whether decline from healthier canopy states can be predicted; this is also transition-oriented.

## Main Findings by Target

{chr(10).join(best_lines)}

## Baseline Gap Diagnosis

{chr(10).join(gap_lines)}

Interpretation rule: improvements on the original broad decline target support **risk-state screening** unless the same feature family also improves at-risk, new-transition, or actionable-drop targets. False-negative reductions are useful for screening, but a recall gain with low precision is a sensitivity trade-off rather than operational early-warning success.

## Feature-Layer Interpretation

{chr(10).join(feature_layer_notes(master))}

The integrated tables distinguish source-aware environmental exposure, static habitat context, and canopy-state or trajectory features. Habitat-only and environmental-only performance should be read as spatial or exposure-risk screening, not as proof of a causal mechanism. Trajectory features are leakage-audited time-series instability proxies; they are not true patch fragmentation unless patch geometry is added.

## Current Bottlenecks

- **Target definition:** the original decline target is partly a low-state/risk-state label, so it can overstate early-warning skill.
- **Rare-event class imbalance:** stricter transition and actionable targets have fewer positive events, making recall and precision unstable.
- **Persistence bias:** current canopy and low-canopy persistence remain strong predictors, especially for the original target.
- **Missing disturbance variables:** wave exposure, storm disturbance, grazing pressure, disease context, and restoration/intervention history are not yet represented.
- **Missing biological drivers:** urchin/grazer pressure and predator/community data are likely needed for a stronger ecological transition case study.
- **Limited spatial support / coarse covariates:** OISST and latitude-bin proxies remain broad exposure layers; CRW 5 km improves spatial resolution but is still satellite SST exposure, not in situ kelp-bed temperature.

## Recommended Next Steps

1. **Define claim gates:** decide which targets and minimum precision/recall criteria are required before using the phrase early-warning.
2. **Consolidate rare-event learning:** focus on transition and actionable targets with calibrated threshold selection, class weighting, and precision floors.
3. **Add wave exposure:** test whether disturbance exposure explains actionable drops better than SST alone.
4. **Assess true fragmentation feasibility:** only add fragmentation claims if patch geometry or within-cell spatial structure can be measured.
5. **Generate final figures and tables:** use the integrated master and best-by-target tables to build a concise portfolio or manuscript result section.
"""

    REPORT_OUT.write_text(report, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    found_files, missing_files = discover_result_files()

    rows: list[dict] = []
    read_errors: list[dict] = []
    for path in found_files:
        try:
            rows.extend(standardize_rows(path))
        except Exception as exc:  # pragma: no cover - defensive reporting
            read_errors.append(
                {
                    "source_table": relative_path(path),
                    "experiment_layer": infer_experiment_layer(path),
                    "target_definition": "unknown",
                    "normalized_target": "unknown_target",
                    "feature_family": "unknown",
                    "normalized_feature_family": "other",
                    "model": "not_read",
                    "learning_strategy": "",
                    "threshold": np.nan,
                    "pr_auc": np.nan,
                    "roc_auc": np.nan,
                    "recall": np.nan,
                    "precision": np.nan,
                    "f1": np.nan,
                    "f2": np.nan,
                    "false_negatives": np.nan,
                    "false_positives": np.nan,
                    "true_positives": np.nan,
                    "true_negatives": np.nan,
                    "train_positive_count": np.nan,
                    "validation_positive_count": np.nan,
                    "test_positive_count": np.nan,
                    "n_train": np.nan,
                    "n_validation": np.nan,
                    "n_test": np.nan,
                    "event_prevalence_test": np.nan,
                    "status": "read_error",
                    "notes": str(exc),
                }
            )

    rows.extend(read_errors)
    master = pd.DataFrame(rows, columns=MASTER_COLUMNS)

    # Make outputs stable and easier to scan.
    master["normalized_target"] = pd.Categorical(master["normalized_target"], TARGET_ORDER, ordered=True)
    master = master.sort_values(
        by=["normalized_target", "experiment_layer", "source_table", "normalized_feature_family", "model"],
        na_position="last",
    )
    master["normalized_target"] = master["normalized_target"].astype(str)

    best = build_best_by_target(master)
    gaps = build_gap_summary(master)

    write_portable_csv(master, MASTER_OUT)
    write_portable_csv(best, BEST_OUT)
    write_portable_csv(gaps, GAP_OUT)
    write_report(master, best, gaps, found_files, missing_files)

    print(f"Source result files found: {len(found_files)}")
    print(f"Missing expected files: {len(missing_files)}")
    print(f"Rows integrated: {len(master)}")
    print("Targets detected:", ", ".join(sorted(master["normalized_target"].dropna().unique())))
    print(f"Wrote: {relative_path(MASTER_OUT)}")
    print(f"Wrote: {relative_path(BEST_OUT)}")
    print(f"Wrote: {relative_path(GAP_OUT)}")
    print(f"Wrote: {relative_path(REPORT_OUT)}")


if __name__ == "__main__":
    main()
