"""Apply conservative claim gates to integrated model results.

Claim gates are interpretation rules, not formal statistical significance tests.
They use existing integrated result tables to decide which level of claim is
currently defensible without rerunning feature construction or model training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

MASTER_PATH = ROOT / "results" / "tables" / "integrated_model_comparison_master.csv"
BEST_PATH = ROOT / "results" / "tables" / "integrated_best_by_target.csv"
GAP_PATH = ROOT / "results" / "tables" / "integrated_baseline_gap_summary.csv"

SUMMARY_OUT = ROOT / "results" / "tables" / "claim_gate_summary.csv"
SENSITIVITY_OUT = ROOT / "results" / "tables" / "claim_gate_sensitivity.csv"
REPORT_OUT = ROOT / "outputs" / "diagnostics" / "claim_gate_interpretation_report.md"

PRECISION_FLOORS = [0.30, 0.40, 0.50]
DEFAULT_PRECISION_FLOOR = 0.40

CSV_WRITE_KWARGS = {
    "index": False,
    "lineterminator": "\n",
    "na_rep": "",
    "float_format": "%.6f",
}


@dataclass(frozen=True)
class Baseline:
    pr_auc: float = np.nan
    recall: float = np.nan
    precision: float = np.nan
    f2: float = np.nan
    false_negatives: float = np.nan


def clean_csv_cells(df: pd.DataFrame) -> pd.DataFrame:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_csv_cells(df).to_csv(path, **CSV_WRITE_KWARGS)


def safe_float(value) -> float:
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def has_value(value) -> bool:
    return pd.notna(value)


def max_or_nan(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.max()) if not series.empty else np.nan


def min_or_nan(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.min()) if not series.empty else np.nan


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [p for p in [MASTER_PATH, BEST_PATH, GAP_PATH] if not p.exists()]
    if missing:
        missing_text = "\n".join(str(p.relative_to(ROOT)) for p in missing)
        raise FileNotFoundError(f"Missing integrated result input(s):\n{missing_text}")

    master = pd.read_csv(MASTER_PATH)
    best = pd.read_csv(BEST_PATH)
    gaps = pd.read_csv(GAP_PATH)
    return master, best, gaps


def rankable(master: pd.DataFrame) -> pd.DataFrame:
    excluded = {"dry_run_no_local_crw_cache", "validation_grid_not_ranked"}
    out = master[~master["status"].isin(excluded)].copy()
    return out


def canopy_baseline(target_df: pd.DataFrame) -> Baseline:
    canopy = target_df[target_df["normalized_feature_family"].eq("canopy_only")]
    return Baseline(
        pr_auc=max_or_nan(canopy["pr_auc"]),
        recall=max_or_nan(canopy["recall"]),
        precision=max_or_nan(canopy["precision"]),
        f2=max_or_nan(canopy["f2"]),
        false_negatives=min_or_nan(canopy["false_negatives"]),
    )


def row_metric(row: pd.Series, name: str) -> float:
    return safe_float(row[name]) if name in row.index else np.nan


def false_negative_reduction(row_fn: float, baseline_fn: float) -> float:
    if not has_value(row_fn) or not has_value(baseline_fn) or baseline_fn <= 0:
        return np.nan
    return float((baseline_fn - row_fn) / baseline_fn)


def is_validation_selected_threshold(row: pd.Series) -> bool:
    experiment = str(row.get("experiment_layer", "")).lower()
    family = str(row.get("normalized_feature_family", "")).lower()
    source = str(row.get("source_table", "")).lower()
    strategy = str(row.get("learning_strategy", "")).lower()
    threshold = row_metric(row, "threshold")
    if "threshold" in experiment or "threshold" in family or "threshold" in source:
        return True
    if has_value(threshold) and any(token in strategy for token in ["max_f", "recall", "precision", "default"]):
        return True
    return False


def condition_text(conditions: dict[str, bool | None]) -> str:
    met = [name for name, passed in conditions.items() if passed is True]
    return "; ".join(met) if met else "none"


def available_condition_count(conditions: dict[str, bool | None]) -> int:
    return sum(1 for value in conditions.values() if value is not None)


def met_condition_count(conditions: dict[str, bool | None]) -> int:
    return sum(1 for value in conditions.values() if value is True)


def select_candidate(rows: list[dict]) -> dict:
    return sorted(
        rows,
        key=lambda r: (
            r["conditions_met_count"],
            safe_float(r.get("precision")),
            safe_float(r.get("recall")),
            safe_float(r.get("f2")),
            safe_float(r.get("pr_auc")),
        ),
        reverse=True,
    )[0]


def gate1_for_floor(best: pd.DataFrame, gaps: pd.DataFrame, precision_floor: float) -> dict:
    target = "original_decline"
    best_row = best[best["target_definition"].eq(target)]
    gap_row = gaps[gaps["target_definition"].eq(target)]
    if best_row.empty or gap_row.empty:
        return insufficient_row("G1", target, precision_floor, "Gate 1 inputs missing")

    b = best_row.iloc[0]
    g = gap_row.iloc[0]

    conditions = {
        "best_model_pr_auc_ge_0_75": safe_float(b["best_pr_auc_value"]) >= 0.75,
        "environment_or_combined_minus_OISST_pr_auc_ge_0_03": safe_float(
            g["best_environment_minus_OISST_pr_auc"]
        )
        >= 0.03,
        "best_model_minus_naive_pr_auc_ge_0_03": safe_float(g["best_model_minus_naive_pr_auc"]) >= 0.03
        if has_value(g["best_model_minus_naive_pr_auc"])
        else None,
    }
    met = met_condition_count(conditions)
    passed = met >= 1
    return {
        "gate_id": "G1",
        "gate_name": "Broad risk-state screening support",
        "target_definition": target,
        "precision_floor": precision_floor,
        "pass_fail": passed,
        "supporting_conditions_met": condition_text(conditions),
        "conditions_met_count": met,
        "conditions_available": available_condition_count(conditions),
        "total_conditions_required": 1,
        "best_model": b["best_pr_auc_model"],
        "feature_family": b["best_pr_auc_feature_family"],
        "pr_auc": safe_float(b["best_pr_auc_value"]),
        "recall": safe_float(b["best_recall_value"]),
        "precision": np.nan,
        "f2": safe_float(b["best_f2_value"]),
        "false_negatives": safe_float(b["best_false_negative_count"]),
        "false_positives": np.nan,
        "baseline_family": "naive_persistence / OISST_only",
        "baseline_pr_auc": safe_float(g["naive_persistence_pr_auc"]),
        "delta_pr_auc_vs_baseline": safe_float(g["best_model_minus_naive_pr_auc"]),
        "interpretation_label": "risk_state_screening_supported" if passed else "insufficient_information",
        "notes": "Original decline supports broad risk-state screening only; not operational early warning.",
    }


def evaluate_at_risk_candidate(row: pd.Series, baseline: Baseline, precision_floor: float) -> dict:
    pr_auc = row_metric(row, "pr_auc")
    f2 = row_metric(row, "f2")
    recall = row_metric(row, "recall")
    precision = row_metric(row, "precision")
    fn = row_metric(row, "false_negatives")
    fn_reduction = false_negative_reduction(fn, baseline.false_negatives)

    conditions = {
        "best_model_minus_canopy_pr_auc_ge_0_03": pr_auc - baseline.pr_auc >= 0.03
        if has_value(pr_auc) and has_value(baseline.pr_auc)
        else None,
        "best_model_minus_canopy_f2_ge_0_10": f2 - baseline.f2 >= 0.10
        if has_value(f2) and has_value(baseline.f2)
        else None,
        "false_negatives_decrease_ge_30pct_vs_canopy": fn_reduction >= 0.30
        if has_value(fn_reduction)
        else None,
        "recall_improves_ge_0_10_vs_canopy": recall - baseline.recall >= 0.10
        if has_value(recall) and has_value(baseline.recall)
        else None,
        "precision_ge_floor": precision >= precision_floor if has_value(precision) else None,
    }
    met = met_condition_count(conditions)
    pr_auc_condition = conditions["best_model_minus_canopy_pr_auc_ge_0_03"] is True
    if met >= 3 and pr_auc_condition:
        label = "at_risk_screening_supported"
        passed = True
    elif met >= 2:
        label = "at_risk_screening_partially_supported"
        passed = True
    elif available_condition_count(conditions) == 0:
        label = "insufficient_information"
        passed = False
    else:
        label = "insufficient_information"
        passed = False

    return row_to_gate_record(
        row=row,
        gate_id="G2",
        gate_name="At-risk screening support",
        precision_floor=precision_floor,
        pass_fail=passed,
        conditions=conditions,
        total_required=2,
        label=label,
        baseline=baseline,
        notes="At-risk support should not be described as robust transition early warning.",
    )


def evaluate_transition_candidate(row: pd.Series, baseline: Baseline, precision_floor: float) -> dict:
    pr_auc = row_metric(row, "pr_auc")
    f2 = row_metric(row, "f2")
    recall = row_metric(row, "recall")
    precision = row_metric(row, "precision")
    fn = row_metric(row, "false_negatives")
    fn_reduction = false_negative_reduction(fn, baseline.false_negatives)

    conditions = {
        "best_model_minus_canopy_pr_auc_ge_0_03": pr_auc - baseline.pr_auc >= 0.03
        if has_value(pr_auc) and has_value(baseline.pr_auc)
        else None,
        "best_model_minus_canopy_f2_ge_0_10": f2 - baseline.f2 >= 0.10
        if has_value(f2) and has_value(baseline.f2)
        else None,
        "false_negatives_decrease_ge_40pct_vs_canopy": fn_reduction >= 0.40
        if has_value(fn_reduction)
        else None,
        "recall_ge_0_70": recall >= 0.70 if has_value(recall) else None,
        "precision_ge_floor": precision >= precision_floor if has_value(precision) else None,
        "threshold_validation_selected": is_validation_selected_threshold(row),
    }
    met = met_condition_count(conditions)
    pr_auc_condition = conditions["best_model_minus_canopy_pr_auc_ge_0_03"] is True
    recall_or_fn = (
        conditions["recall_ge_0_70"] is True or conditions["false_negatives_decrease_ge_40pct_vs_canopy"] is True
    )
    precision_ok = conditions["precision_ge_floor"] is True

    if met >= 3 and pr_auc_condition:
        label = "transition_early_warning_supported"
        passed = True
    elif recall_or_fn and precision_ok and not pr_auc_condition:
        label = "transition_recall_oriented_sensitivity_only"
        passed = False
    elif met >= 3 and not pr_auc_condition:
        label = "transition_recall_oriented_sensitivity_only"
        passed = False
    elif available_condition_count(conditions) == 0:
        label = "insufficient_information"
        passed = False
    else:
        label = "transition_early_warning_not_supported"
        passed = False

    return row_to_gate_record(
        row=row,
        gate_id="G3",
        gate_name="Transition/actionable early-warning support",
        precision_floor=precision_floor,
        pass_fail=passed,
        conditions=conditions,
        total_required=3,
        label=label,
        baseline=baseline,
        notes="Gate 3 is the stricter early-warning-oriented gate.",
    )


def row_to_gate_record(
    row: pd.Series,
    gate_id: str,
    gate_name: str,
    precision_floor: float,
    pass_fail: bool,
    conditions: dict[str, bool | None],
    total_required: int,
    label: str,
    baseline: Baseline,
    notes: str,
) -> dict:
    return {
        "gate_id": gate_id,
        "gate_name": gate_name,
        "target_definition": row["normalized_target"],
        "precision_floor": precision_floor,
        "pass_fail": pass_fail,
        "supporting_conditions_met": condition_text(conditions),
        "conditions_met_count": met_condition_count(conditions),
        "conditions_available": available_condition_count(conditions),
        "total_conditions_required": total_required,
        "best_model": row["model"],
        "feature_family": row["normalized_feature_family"],
        "pr_auc": row_metric(row, "pr_auc"),
        "recall": row_metric(row, "recall"),
        "precision": row_metric(row, "precision"),
        "f2": row_metric(row, "f2"),
        "false_negatives": row_metric(row, "false_negatives"),
        "false_positives": row_metric(row, "false_positives"),
        "baseline_family": "canopy_only",
        "baseline_pr_auc": baseline.pr_auc,
        "delta_pr_auc_vs_baseline": row_metric(row, "pr_auc") - baseline.pr_auc
        if has_value(row_metric(row, "pr_auc")) and has_value(baseline.pr_auc)
        else np.nan,
        "baseline_recall": baseline.recall,
        "delta_recall_vs_baseline": row_metric(row, "recall") - baseline.recall
        if has_value(row_metric(row, "recall")) and has_value(baseline.recall)
        else np.nan,
        "baseline_f2": baseline.f2,
        "delta_f2_vs_baseline": row_metric(row, "f2") - baseline.f2
        if has_value(row_metric(row, "f2")) and has_value(baseline.f2)
        else np.nan,
        "baseline_false_negatives": baseline.false_negatives,
        "false_negative_reduction_rate": false_negative_reduction(
            row_metric(row, "false_negatives"), baseline.false_negatives
        ),
        "source_table": row["source_table"],
        "experiment_layer": row["experiment_layer"],
        "learning_strategy": row.get("learning_strategy", ""),
        "threshold": row_metric(row, "threshold"),
        "interpretation_label": label,
        "notes": notes,
    }


def insufficient_row(gate_id: str, target: str, precision_floor: float, note: str) -> dict:
    label = "transition_early_warning_not_supported" if gate_id == "G3" else "insufficient_information"
    return {
        "gate_id": gate_id,
        "gate_name": "",
        "target_definition": target,
        "precision_floor": precision_floor,
        "pass_fail": False,
        "supporting_conditions_met": "none",
        "conditions_met_count": 0,
        "conditions_available": 0,
        "total_conditions_required": 0,
        "best_model": "",
        "feature_family": "",
        "pr_auc": np.nan,
        "recall": np.nan,
        "precision": np.nan,
        "f2": np.nan,
        "false_negatives": np.nan,
        "false_positives": np.nan,
        "baseline_family": "",
        "baseline_pr_auc": np.nan,
        "delta_pr_auc_vs_baseline": np.nan,
        "baseline_recall": np.nan,
        "delta_recall_vs_baseline": np.nan,
        "baseline_f2": np.nan,
        "delta_f2_vs_baseline": np.nan,
        "baseline_false_negatives": np.nan,
        "false_negative_reduction_rate": np.nan,
        "source_table": "",
        "experiment_layer": "",
        "learning_strategy": "",
        "threshold": np.nan,
        "interpretation_label": label,
        "notes": note,
    }


def apply_target_gate(
    master: pd.DataFrame,
    target: str,
    precision_floor: float,
    evaluator,
) -> dict:
    usable = rankable(master)
    target_df = usable[usable["normalized_target"].eq(target)].copy()
    if target_df.empty:
        gate_id = "G2" if target == "at_risk_original" else "G3"
        return insufficient_row(gate_id, target, precision_floor, "No rankable rows found for target")

    baseline = canopy_baseline(target_df)
    candidates = []
    for _, row in target_df.iterrows():
        if not has_value(row_metric(row, "pr_auc")) and not has_value(row_metric(row, "recall")):
            continue
        candidates.append(evaluator(row, baseline, precision_floor))

    if not candidates:
        gate_id = "G2" if target == "at_risk_original" else "G3"
        return insufficient_row(gate_id, target, precision_floor, "No candidate rows with usable metrics")
    return select_candidate(candidates)


def build_sensitivity(master: pd.DataFrame, best: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    records = []
    for floor in PRECISION_FLOORS:
        records.append(gate1_for_floor(best, gaps, floor))
        records.append(apply_target_gate(master, "at_risk_original", floor, evaluate_at_risk_candidate))
        records.append(apply_target_gate(master, "new_transition", floor, evaluate_transition_candidate))
        records.append(apply_target_gate(master, "actionable_drop", floor, evaluate_transition_candidate))
    return pd.DataFrame(records)


def build_summary(sensitivity: pd.DataFrame) -> pd.DataFrame:
    summary = sensitivity[np.isclose(sensitivity["precision_floor"], DEFAULT_PRECISION_FLOOR)].copy()
    summary = summary.sort_values(["gate_id", "target_definition"]).reset_index(drop=True)

    gate3 = summary[summary["gate_id"].eq("G3")]
    if gate3.empty:
        aggregate = insufficient_row(
            "G3_overall",
            "new_transition_or_actionable_drop",
            DEFAULT_PRECISION_FLOOR,
            "No Gate 3 rows found",
        )
    else:
        any_supported = gate3["interpretation_label"].eq("transition_early_warning_supported").any()
        any_sensitivity = gate3["interpretation_label"].eq("transition_recall_oriented_sensitivity_only").any()
        if any_supported:
            label = "transition_early_warning_supported"
            passed = True
            note = "At least one strict early-warning-oriented target passed Gate 3."
        elif any_sensitivity:
            label = "transition_recall_oriented_sensitivity_only"
            passed = False
            note = "Gate 3 did not support robust transition early warning, but recall-oriented sensitivity was detected."
        else:
            label = "transition_early_warning_not_supported"
            passed = False
            note = "Neither transition/actionable target passed Gate 3."
        best_gate3 = select_candidate(gate3.to_dict("records"))
        aggregate = best_gate3.copy()
        aggregate.update(
            {
                "gate_id": "G3_overall",
                "gate_name": "Overall transition/actionable early-warning support",
                "target_definition": "new_transition_or_actionable_drop",
                "pass_fail": passed,
                "interpretation_label": label,
                "notes": note,
            }
        )

    return pd.concat([summary, pd.DataFrame([aggregate])], ignore_index=True)


def fmt(value: float, digits: int = 3) -> str:
    if not has_value(value):
        return "NA"
    return f"{value:.{digits}f}"


def target_report_lines(summary: pd.DataFrame) -> str:
    lines = []
    for _, row in summary[~summary["gate_id"].eq("G3_overall")].iterrows():
        lines.append(
            "### {target}\n\n"
            "- Gate: `{gate}` ({gate_name})\n"
            "- Result: `{label}`; pass = `{passed}`\n"
            "- Selected model: `{family}` / `{model}`\n"
            "- Metrics: PR-AUC `{pr_auc}`, recall `{recall}`, precision `{precision}`, F2 `{f2}`, false negatives `{fn}`\n"
            "- Baseline: `{baseline}` PR-AUC `{baseline_pr_auc}`, recall `{baseline_recall}`, false negatives `{baseline_fn}`\n"
            "- Supporting conditions: {conditions}\n".format(
                target=row["target_definition"],
                gate=row["gate_id"],
                gate_name=row["gate_name"],
                label=row["interpretation_label"],
                passed=row["pass_fail"],
                family=row["feature_family"],
                model=row["best_model"],
                pr_auc=fmt(row["pr_auc"]),
                recall=fmt(row["recall"]),
                precision=fmt(row["precision"]),
                f2=fmt(row["f2"]),
                fn=fmt(row["false_negatives"], 0),
                baseline=row["baseline_family"],
                baseline_pr_auc=fmt(row["baseline_pr_auc"]),
                baseline_recall=fmt(row["baseline_recall"]),
                baseline_fn=fmt(row["baseline_false_negatives"], 0),
                conditions=row["supporting_conditions_met"],
            )
        )
    return "\n".join(lines)


def sensitivity_summary_lines(sensitivity: pd.DataFrame) -> str:
    pivot = sensitivity.pivot_table(
        index=["gate_id", "target_definition"],
        columns="precision_floor",
        values="interpretation_label",
        aggfunc="first",
    )
    lines = []
    for idx, row in pivot.iterrows():
        floor_text = ", ".join(f"{floor:.2f}: `{label}`" for floor, label in row.dropna().items())
        lines.append(f"- `{idx[0]}` / `{idx[1]}` -> {floor_text}")
    return "\n".join(lines)


def final_claim_level(summary: pd.DataFrame) -> tuple[str, str]:
    gate1 = summary[summary["gate_id"].eq("G1")]
    gate2 = summary[summary["gate_id"].eq("G2")]
    gate3 = summary[summary["gate_id"].eq("G3_overall")]

    gate1_label = gate1["interpretation_label"].iloc[0] if not gate1.empty else "insufficient_information"
    gate2_label = gate2["interpretation_label"].iloc[0] if not gate2.empty else "insufficient_information"
    gate3_label = gate3["interpretation_label"].iloc[0] if not gate3.empty else "transition_early_warning_not_supported"

    if gate3_label == "transition_early_warning_supported":
        headline = "Transition/actionable early-warning support is provisionally supported."
    elif gate3_label == "transition_recall_oriented_sensitivity_only":
        headline = "Current results support recall-oriented warning sensitivity, not robust operational early warning."
    elif gate2_label in {"at_risk_screening_supported", "at_risk_screening_partially_supported"}:
        headline = "Current results support broad risk-state screening and partial at-risk screening."
    elif gate1_label == "risk_state_screening_supported":
        headline = "Current results support broad decline-risk state screening only."
    else:
        headline = "Current results are insufficient for claim support."

    details = (
        f"Gate 1: `{gate1_label}`. Gate 2: `{gate2_label}`. "
        f"Gate 3: `{gate3_label}`."
    )
    return headline, details


def write_report(summary: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    headline, details = final_claim_level(summary)
    gate3 = summary[summary["gate_id"].eq("G3_overall")].iloc[0]

    report = f"""# Claim-Gate Interpretation Report

## Purpose

Claim gates are conservative interpretation rules used to prevent overclaiming from integrated model-result tables. They are not formal statistical significance tests, confidence intervals, or causal tests. They translate existing model metrics into defensible claim levels.

No new models were trained and no heavy feature construction was rerun for this report.

## Gate Definitions

### Gate 1: Broad Risk-State Screening Support

Relevant target: `original_decline`.

This gate passes if at least one broad screening condition is met: best PR-AUC is at least `0.75`, environment/combined models improve over OISST-only by at least `0.03` PR-AUC, or the best model improves over naive persistence by at least `0.03` PR-AUC. Passing this gate supports broad decline-risk state screening only.

### Gate 2: At-Risk Screening Support

Relevant target: `at_risk_original`.

This gate evaluates whether a model improves over canopy-only baselines among observations that are more relevant to early warning. It checks PR-AUC gain, F2 gain, false-negative reduction, recall gain, and whether precision meets the selected floor. Passing this gate supports at-risk screening, not robust transition early warning.

### Gate 3: Transition/Actionable Early-Warning Support

Relevant targets: `new_transition` and `actionable_drop`.

This is the strict early-warning-oriented gate. It checks PR-AUC gain over canopy-only, F2 gain, false-negative reduction, recall, precision floor, and whether threshold information indicates validation-selected thresholding. A target must meet at least three conditions and include PR-AUC improvement over canopy-only to be labeled as supported. If recall or false negatives improve without PR-AUC support, the result is labeled as recall-oriented warning sensitivity only.

## Results by Target

{target_report_lines(summary)}

## Sensitivity Analysis

The same gates were evaluated with precision floors `0.30`, `0.40`, and `0.50`.

{sensitivity_summary_lines(sensitivity)}

## Final Claim Level

{headline}

{details}

The safest current claim is that this repository supports **broad decline-risk state screening** and provides a structured robustness framework for testing stricter early-warning claims. At-risk screening evidence is partial because gains over canopy-only are limited under the default precision floor. Gate 3 overall is `{gate3['interpretation_label']}`.

## What Can Be Claimed Safely

- Broad risk-state screening is supported for the original decline-state target.
- The integrated workflow can distinguish broad screening, at-risk screening, and stricter transition/actionable targets.
- Recall-oriented threshold and cost-sensitive settings can reduce false negatives in some settings, but these are sensitivity trade-offs.

## What Can Be Claimed Only Partially

- At-risk screening support is partial unless future runs show clearer improvement over canopy-only baselines while maintaining acceptable precision.
- Transition/actionable warning sensitivity can be discussed only as a diagnostic result when recall or false negatives improve without PR-AUC support.

## What Cannot Yet Be Claimed

- The results do not yet establish an operational kelp decline early-warning system.
- Strong original-label PR-AUC should not be used as evidence of true transition early-warning.
- Environmental or habitat covariate gains should not be interpreted as causal mechanisms.

## Recommended Next Steps

1. **Rare-event learning:** prioritize transition/actionable labels if Gate 3 failures are driven by low recall or false negatives.
2. **Wave exposure:** add disturbance exposure if environment, habitat, and trajectory features still fail transition targets.
3. **True fragmentation feasibility:** add patch geometry or within-cell spatial structure before making fragmentation claims.
4. **External/spatial validation:** if future gates become strong, test leave-region-out or external-region validation before stronger claims.
"""
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(report, encoding="utf-8")


def main() -> None:
    master, best, gaps = read_inputs()
    sensitivity = build_sensitivity(master, best, gaps)
    summary = build_summary(sensitivity)

    write_portable_csv(summary, SUMMARY_OUT)
    write_portable_csv(sensitivity, SENSITIVITY_OUT)
    write_report(summary, sensitivity)

    gate3 = summary[summary["gate_id"].eq("G3_overall")].iloc[0]
    print(f"Rows written to {SUMMARY_OUT.relative_to(ROOT)}: {len(summary)}")
    print(f"Rows written to {SENSITIVITY_OUT.relative_to(ROOT)}: {len(sensitivity)}")
    for _, row in summary.iterrows():
        print(
            f"{row['gate_id']} / {row['target_definition']}: "
            f"{row['interpretation_label']} (pass={row['pass_fail']})"
        )
    print("Final safe claim level:", final_claim_level(summary)[0])
    print("Gate 3 overall:", gate3["interpretation_label"])


if __name__ == "__main__":
    main()
