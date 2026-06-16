"""Benchmark ML models against naive canopy-persistence baselines.

This persistence-aware validity layer tests whether apparent next-year kelp
decline prediction skill exceeds simple rules based on current or recent canopy
state. It does not add new environmental predictors or overwrite existing V1/V2
workflows.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from run_recall_oriented_modeling_extensions import (
    BASELINE_P25,
    CANOPY,
    NEXT_CANOPY,
    TARGET_ACTIONABLE_DROP,
    add_actionable_labels,
)
from train_model_comparison import (
    INPUT_DATASET,
    feature_sets,
    load_dataset,
    model_specs,
    predict_scores,
    preprocessor,
)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_DIR = Path("results/tables")
DIAGNOSTIC_DIR = Path("outputs/diagnostics")
NAIVE_OUTPUT = OUTPUT_DIR / "naive_persistence_baseline_comparison.csv"
SUBGROUP_OUTPUT = OUTPUT_DIR / "high_canopy_subgroup_performance.csv"
GAP_OUTPUT = OUTPUT_DIR / "ml_vs_naive_baseline_gap.csv"
REPORT_OUTPUT = DIAGNOSTIC_DIR / "naive_persistence_baseline_report.md"
V2_RESULTS = Path("results/tables/multiscale_model_comparison.csv")

MODEL_START_YEAR = 1989
TRAIN_END_YEAR = 2016
TEST_START_YEAR = 2021
TEST_END_YEAR = 2024
BASELINE_START_YEAR = 1984
BASELINE_END_YEAR = 2013
NEW_DECLINE_TARGET = "new_decline_event_next"
AT_RISK_TARGET = "decline_event_next_at_risk_gt005"
EPSILON = 1e-6


@dataclass(frozen=True)
class TargetDefinition:
    """Target definition and optional base subset."""

    name: str
    target_column: str
    base_filter_column: str | None = None


@dataclass(frozen=True)
class SubgroupDefinition:
    """Evaluation subgroup definition."""

    name: str
    filter_column: str | None = None


TARGETS = [
    TargetDefinition("A_original_decline_state", "decline_event_next"),
    TargetDefinition("B_at_risk_original_decline_gt005", AT_RISK_TARGET, "subgroup_current_canopy_gt_0_05"),
    TargetDefinition("C_new_decline_transition", NEW_DECLINE_TARGET),
    TargetDefinition("D_actionable_decline_drop", TARGET_ACTIONABLE_DROP),
]

SUBGROUPS = [
    SubgroupDefinition("full_sample"),
    SubgroupDefinition("current_canopy_gt_0_05", "subgroup_current_canopy_gt_0_05"),
    SubgroupDefinition("current_canopy_ge_historical_p50", "subgroup_current_canopy_ge_historical_p50"),
    SubgroupDefinition("current_canopy_ge_historical_p75", "subgroup_current_canopy_ge_historical_p75"),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark naive persistence baselines against ML models.")
    parser.add_argument("--input", type=Path, default=INPUT_DATASET)
    parser.add_argument("--v2-results", type=Path, default=V2_RESULTS)
    return parser.parse_args()


def load_benchmark_data(path: Path) -> pd.DataFrame:
    """Load model rows and add labels, thresholds, and lagged canopy features."""
    data = load_dataset(path).sort_values(["cell_id", "year"]).reset_index(drop=True)
    data = add_actionable_labels(data)
    data[NEW_DECLINE_TARGET] = ((data[CANOPY] >= data[BASELINE_P25]) & (data[NEXT_CANOPY] < data[BASELINE_P25])).astype(int)
    data[AT_RISK_TARGET] = data["decline_event_next"].astype(int)

    baseline = data.loc[data["year"].between(BASELINE_START_YEAR, BASELINE_END_YEAR)]
    p50 = baseline.groupby("cell_id")[CANOPY].quantile(0.50)
    p75 = baseline.groupby("cell_id")[CANOPY].quantile(0.75)
    data["baseline_p50_relative_canopy_1984_2013"] = data["cell_id"].map(p50)
    data["baseline_p75_relative_canopy_1984_2013"] = data["cell_id"].map(p75)

    grouped = data.groupby("cell_id", group_keys=False)
    data["lag1_relative_canopy"] = grouped[CANOPY].shift(1)
    data["lag2_relative_canopy"] = grouped[CANOPY].shift(2)
    data["canopy_change_1yr"] = data[CANOPY] - data["lag1_relative_canopy"]
    data["canopy_change_2yr"] = data[CANOPY] - data["lag2_relative_canopy"]
    data["canopy_3yr_slope"] = (
        grouped[CANOPY]
        .rolling(3, min_periods=3)
        .apply(lambda values: float(np.polyfit(np.arange(len(values)), values, 1)[0]), raw=True)
        .reset_index(level=0, drop=True)
    )
    data["subgroup_current_canopy_gt_0_05"] = data[CANOPY] > 0.05
    data["subgroup_current_canopy_ge_historical_p50"] = data[CANOPY] >= data["baseline_p50_relative_canopy_1984_2013"]
    data["subgroup_current_canopy_ge_historical_p75"] = data[CANOPY] >= data["baseline_p75_relative_canopy_1984_2013"]

    return data.loc[data["year"].between(MODEL_START_YEAR, TEST_END_YEAR)].copy()


def filtered_split(data: pd.DataFrame, target_def: TargetDefinition, subgroup: SubgroupDefinition) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train and test rows after target and subgroup filters."""
    working = data.copy()
    for filter_column in [target_def.base_filter_column, subgroup.filter_column]:
        if filter_column:
            working = working.loc[working[filter_column]].copy()
    working = working.dropna(subset=[target_def.target_column])
    train = working.loc[working["year"].between(MODEL_START_YEAR, TRAIN_END_YEAR)].copy()
    test = working.loc[working["year"].between(TEST_START_YEAR, TEST_END_YEAR)].copy()
    return train, test


def has_two_classes(frame: pd.DataFrame, target: str) -> bool:
    """Return whether a frame contains both target classes."""
    return set(frame[target].dropna().astype(int).unique()) == {0, 1}


def score_metrics(y_true: pd.Series, scores: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    """Compute benchmark metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    try:
        pr_auc = average_precision_score(y_true, scores)
    except ValueError:
        pr_auc = np.nan
    return {
        "pr_auc": float(pr_auc),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "n_test": int(len(y_true)),
        "n_positive_events": int(y_true.sum()),
        "event_prevalence": float(y_true.mean()),
    }


def rule_baseline_scores(test: pd.DataFrame, baseline_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return continuous risk scores and binary predictions for one rule baseline."""
    if baseline_name == "A_cell_p25_persistence_rule":
        scores = test[BASELINE_P25] - test[CANOPY]
        predictions = (test[CANOPY] < test[BASELINE_P25]).astype(int)
    elif baseline_name == "B_fixed_low_canopy_005_rule":
        scores = 0.05 - test[CANOPY]
        predictions = (test[CANOPY] < 0.05).astype(int)
    elif baseline_name == "C_lag1_low_state_rule":
        scores = pd.concat([test[BASELINE_P25] - test["lag1_relative_canopy"], 0.05 - test["lag1_relative_canopy"]], axis=1).max(axis=1)
        predictions = ((test["lag1_relative_canopy"] < test[BASELINE_P25]) | (test["lag1_relative_canopy"] < 0.05)).astype(int)
    elif baseline_name == "D_recent_declining_trajectory_rule":
        scores = pd.concat([-test["canopy_change_1yr"], -test["canopy_change_2yr"], -test["canopy_3yr_slope"]], axis=1).max(axis=1)
        predictions = ((test["canopy_change_1yr"] < 0) | (test["canopy_change_2yr"] < 0)).astype(int)
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")
    return scores.fillna(scores.median()).to_numpy(dtype=float), predictions.fillna(0).to_numpy(dtype=int)


def evaluate_rule_baselines(data: pd.DataFrame) -> pd.DataFrame:
    """Evaluate official naive persistence baselines."""
    rows = []
    baseline_names = [
        "A_cell_p25_persistence_rule",
        "B_fixed_low_canopy_005_rule",
        "C_lag1_low_state_rule",
        "D_recent_declining_trajectory_rule",
    ]
    for target_def in TARGETS:
        for subgroup in SUBGROUPS:
            _, test = filtered_split(data, target_def, subgroup)
            if test.empty or not has_two_classes(test, target_def.target_column):
                continue
            y_true = test[target_def.target_column].astype(int)
            for baseline_name in baseline_names:
                scores, predictions = rule_baseline_scores(test, baseline_name)
                rows.append(
                    {
                        "model_group": "naive_persistence",
                        "model_name": baseline_name,
                        "target_definition": target_def.name,
                        "target": target_def.target_column,
                        "evaluation_subgroup": subgroup.name,
                        **score_metrics(y_true, scores, predictions),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_logistic_baselines(data: pd.DataFrame) -> pd.DataFrame:
    """Evaluate simple canopy-only logistic benchmark baselines."""
    rows = []
    variants = {
        "E1_logistic_current_canopy_only": [CANOPY],
        "E2_logistic_current_lag_slope": [CANOPY, "lag1_relative_canopy", "canopy_3yr_slope"],
    }
    for target_def in TARGETS:
        for subgroup in SUBGROUPS:
            train, test = filtered_split(data, target_def, subgroup)
            if train.empty or test.empty or not has_two_classes(train, target_def.target_column) or not has_two_classes(test, target_def.target_column):
                continue
            y_train = train[target_def.target_column].astype(int)
            y_test = test[target_def.target_column].astype(int)
            for model_name, features in variants.items():
                model = Pipeline(
                    [
                        ("preprocess", preprocessor(train, features, scale=True)),
                        ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
                    ]
                )
                model.fit(train[features], y_train)
                scores = model.predict_proba(test[features])[:, 1]
                predictions = (scores >= 0.5).astype(int)
                rows.append(
                    {
                        "model_group": "logistic_canopy_baseline",
                        "model_name": model_name,
                        "target_definition": target_def.name,
                        "target": target_def.target_column,
                        "evaluation_subgroup": subgroup.name,
                        **score_metrics(y_test, scores, predictions),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_repo_ml_models(data: pd.DataFrame) -> pd.DataFrame:
    """Refit the repository's established V1 feature sets for subgroup comparison."""
    rows = []
    for target_def in TARGETS:
        for subgroup in SUBGROUPS:
            train, test = filtered_split(data, target_def, subgroup)
            if train.empty or test.empty or not has_two_classes(train, target_def.target_column) or not has_two_classes(test, target_def.target_column):
                continue
            y_train = train[target_def.target_column].astype(int)
            y_test = test[target_def.target_column].astype(int)
            specs = model_specs(y_train)
            sets = feature_sets(train)
            for feature_set_name, features in sets.items():
                for model_family, (estimator, needs_scaling) in specs.items():
                    model = Pipeline(
                        [
                            ("preprocess", preprocessor(train, features, scale=needs_scaling)),
                            ("model", estimator),
                        ]
                    )
                    model.fit(train[features], y_train)
                    scores = predict_scores(model, test[features])
                    predictions = (scores >= 0.5).astype(int)
                    rows.append(
                        {
                            "model_group": "repo_ml_refit",
                            "model_name": f"{feature_set_name} / {model_family}",
                            "target_definition": target_def.name,
                            "target": target_def.target_column,
                            "evaluation_subgroup": subgroup.name,
                            **score_metrics(y_test, scores, predictions),
                        }
                    )
    return pd.DataFrame(rows)


def load_v2_ml_results(path: Path) -> pd.DataFrame:
    """Load V2 multi-scale results when available."""
    if not path.exists():
        return pd.DataFrame()
    v2 = pd.read_csv(path)
    v2 = v2.loc[v2["validation_design"].eq("temporal_holdout")].copy()
    if v2.empty:
        return pd.DataFrame()
    rows = []
    for row in v2.itertuples():
        tp = int(round(row.recall * row.test_positive_events))
        fn = int(row.test_positive_events - tp)
        if row.precision and row.precision > 0:
            fp = int(round(tp / row.precision - tp))
        else:
            fp = np.nan
        subgroup = "current_canopy_gt_0_05" if row.target_definition == "B_at_risk_original_decline_gt005" else "full_sample"
        rows.append(
            {
                "model_group": "v2_multiscale_existing",
                "model_name": f"{row.comparison} / {row.model_family}",
                "target_definition": row.target_definition,
                "target": row.target,
                "evaluation_subgroup": subgroup,
                "pr_auc": row.pr_auc,
                "recall": row.recall,
                "precision": row.precision,
                "f1": row.f1,
                "balanced_accuracy": row.balanced_accuracy,
                "false_negatives": fn,
                "false_positives": fp,
                "true_positives": tp,
                "true_negatives": np.nan,
                "n_test": row.n_test,
                "n_positive_events": row.test_positive_events,
                "event_prevalence": row.test_event_prevalence,
            }
        )
    return pd.DataFrame(rows)


def best_gap_table(naive: pd.DataFrame, ml: pd.DataFrame) -> pd.DataFrame:
    """Create target/subgroup table comparing best naive baseline with best ML model."""
    rows = []
    keys = sorted(set(zip(naive["target_definition"], naive["evaluation_subgroup"])))
    for target_definition, subgroup in keys:
        naive_subset = naive.loc[
            (naive["target_definition"] == target_definition) & (naive["evaluation_subgroup"] == subgroup)
        ].sort_values("pr_auc", ascending=False)
        ml_subset = ml.loc[
            (ml["target_definition"] == target_definition) & (ml["evaluation_subgroup"] == subgroup)
        ].sort_values("pr_auc", ascending=False)
        if naive_subset.empty:
            continue
        naive_best = naive_subset.iloc[0]
        if ml_subset.empty:
            rows.append(
                {
                    "target_definition": target_definition,
                    "evaluation_subgroup": subgroup,
                    "best_naive_model": naive_best["model_name"],
                    "best_naive_pr_auc": naive_best["pr_auc"],
                    "best_ml_model": "not_available",
                    "best_ml_pr_auc": np.nan,
                    "ml_minus_naive_pr_auc": np.nan,
                    "gap_interpretation": "no_available_ml_result",
                    "claim_gate": "diagnostic_only",
                }
            )
            continue
        ml_best = ml_subset.iloc[0]
        gap = float(ml_best["pr_auc"] - naive_best["pr_auc"])
        if gap >= 0.05:
            interpretation = "ml_clearly_outperforms_naive"
        elif gap <= -0.01:
            interpretation = "naive_matches_or_exceeds_ml"
        else:
            interpretation = "ml_similar_to_naive"
        supports_early_warning = (
            interpretation == "ml_clearly_outperforms_naive"
            and target_definition in {"C_new_decline_transition", "D_actionable_decline_drop"}
            and subgroup in {"current_canopy_ge_historical_p50", "current_canopy_ge_historical_p75"}
        )
        rows.append(
            {
                "target_definition": target_definition,
                "evaluation_subgroup": subgroup,
                "best_naive_model": naive_best["model_name"],
                "best_naive_pr_auc": naive_best["pr_auc"],
                "best_naive_recall": naive_best["recall"],
                "best_naive_precision": naive_best["precision"],
                "best_ml_model": ml_best["model_name"],
                "best_ml_group": ml_best["model_group"],
                "best_ml_pr_auc": ml_best["pr_auc"],
                "best_ml_recall": ml_best["recall"],
                "best_ml_precision": ml_best["precision"],
                "ml_minus_naive_pr_auc": gap,
                "gap_interpretation": interpretation,
                "claim_gate": "early_warning_signal_beyond_persistence" if supports_early_warning else "diagnostic_screening_only",
            }
        )
    return pd.DataFrame(rows)


def write_report(output: Path, naive: pd.DataFrame, subgroup_perf: pd.DataFrame, gap: pd.DataFrame) -> None:
    """Write persistence-aware benchmark report."""
    full_gap = gap.loc[gap["evaluation_subgroup"].eq("full_sample")]
    high_gap = gap.loc[gap["evaluation_subgroup"].isin(["current_canopy_ge_historical_p50", "current_canopy_ge_historical_p75"])]
    early_support = (gap["claim_gate"] == "early_warning_signal_beyond_persistence").any()
    original_full = full_gap.loc[full_gap["target_definition"].eq("A_original_decline_state")]
    transition_rows = gap.loc[gap["target_definition"].isin(["C_new_decline_transition", "D_actionable_decline_drop"])]
    lines = [
        "# Naive Persistence Baseline Benchmark Report",
        "",
        "## Purpose",
        "",
        "This diagnostic treats simple canopy-persistence rules as official benchmark models. The goal is to test whether ML/GeoAI results provide early-warning signal beyond current or recent canopy-state persistence.",
        "",
        "## Claim-Gating Rule",
        "",
        "- If ML clearly outperforms naive baselines on high-canopy and transition/actionable targets, this supports early-warning signal beyond persistence.",
        "- If ML performs well only on the full-sample original decline target, the repository should not claim operational early warning.",
        "- If naive baselines perform similarly to or better than ML, report persistence bias clearly.",
        "",
        "## Summary",
        "",
        f"- Naive/logistic baseline rows: `{len(naive)}`.",
        f"- High-canopy/subgroup performance rows: `{len(subgroup_perf)}`.",
        f"- ML-vs-naive comparison rows: `{len(gap)}`.",
        f"- Any high-canopy transition/actionable claim gate passed: `{bool(early_support)}`.",
        "",
    ]
    if not original_full.empty:
        row = original_full.iloc[0]
        lines.extend(
            [
                "### Full-Sample Original Decline",
                "",
                f"- Best naive baseline: `{row.best_naive_model}` with PR-AUC `{row.best_naive_pr_auc:.3f}`.",
                f"- Best ML model: `{row.best_ml_model}` with PR-AUC `{row.best_ml_pr_auc:.3f}`.",
                f"- ML minus naive PR-AUC: `{row.ml_minus_naive_pr_auc:.3f}`.",
                "",
            ]
        )
    if not transition_rows.empty:
        lines.extend(["### Transition and Actionable Targets", ""])
        for row in transition_rows.sort_values(["target_definition", "evaluation_subgroup"]).itertuples():
            lines.append(
                f"- `{row.target_definition}` / `{row.evaluation_subgroup}`: best naive PR-AUC `{row.best_naive_pr_auc:.3f}`, "
                f"best ML PR-AUC `{row.best_ml_pr_auc:.3f}`, gap `{row.ml_minus_naive_pr_auc:.3f}` ({row.gap_interpretation})."
            )
        lines.append("")
    if not high_gap.empty:
        passed = high_gap.loc[high_gap["claim_gate"].eq("early_warning_signal_beyond_persistence")]
        lines.extend(
            [
                "### High-Canopy Subgroups",
                "",
                f"- High-canopy comparisons evaluated: `{len(high_gap)}`.",
                f"- High-canopy transition/actionable comparisons passing the claim gate: `{len(passed)}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "High full-sample performance should not be interpreted as operational early-warning skill unless the model also outperforms simple canopy-persistence baselines under at-risk, high-canopy, and transition-oriented evaluation settings.",
            "",
            "This repository provides a persistence-aware diagnostic framework for separating apparent decline predictability from true early-warning signal. The current benchmark should be used to gate claims: strong performance on already-low or full-sample decline-state rows is useful for diagnostic screening, but it is not by itself evidence of operational early warning.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the persistence-aware baseline benchmark."""
    args = parse_args()
    data = load_benchmark_data(args.input)
    naive_rules = evaluate_rule_baselines(data)
    logistic_baselines = evaluate_logistic_baselines(data)
    naive_all = pd.concat([naive_rules, logistic_baselines], ignore_index=True)
    repo_ml = evaluate_repo_ml_models(data)
    v2_ml = load_v2_ml_results(args.v2_results)
    ml_all = pd.concat([repo_ml, v2_ml], ignore_index=True) if not v2_ml.empty else repo_ml
    subgroup_perf = pd.concat([naive_all, ml_all], ignore_index=True)
    gap = best_gap_table(naive_all, ml_all)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
    naive_all.to_csv(NAIVE_OUTPUT, index=False)
    subgroup_perf.to_csv(SUBGROUP_OUTPUT, index=False)
    gap.to_csv(GAP_OUTPUT, index=False)
    write_report(REPORT_OUTPUT, naive_all, subgroup_perf, gap)

    print(f"Wrote naive baseline comparison: {NAIVE_OUTPUT}")
    print(f"Wrote subgroup performance: {SUBGROUP_OUTPUT}")
    print(f"Wrote ML-vs-naive gap table: {GAP_OUTPUT}")
    print(f"Wrote report: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
