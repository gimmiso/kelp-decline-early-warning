"""Diagnose zero-state persistence and at-risk early-warning validity.

This script checks whether kelp decline model performance may be inflated by
easy zero-to-zero or already-low canopy persistence. It adds three diagnostics:

1. Current-to-next-year zero/near-zero canopy transition tables.
2. At-risk subset model evaluation for the original decline label.
3. Full and at-risk model evaluation for a stricter new-decline transition label.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from train_model_comparison import (
    INPUT_DATASET,
    TARGET,
    feature_sets,
    load_dataset,
    main_subset,
    model_specs,
    predict_scores,
    preprocessor,
)


warnings.filterwarnings("ignore", category=UserWarning)

CANOPY_COLUMN = "relative_canopy"
NEXT_CANOPY_COLUMN = "next_year_relative_canopy"
BASELINE_P25_COLUMN = "baseline_p25_relative_canopy_1984_2013"
NEW_DECLINE_TARGET = "new_decline_event_next"

ZERO_THRESHOLDS = [0.0, 0.01, 0.05, 0.10]
AT_RISK_THRESHOLDS = [0.0, 0.01, 0.05, 0.10]

DIAGNOSTIC_DIR = Path("outputs/diagnostics")
TRANSITION_COUNTS_OUTPUT = DIAGNOSTIC_DIR / "zero_persistence_transition_counts.csv"
TRANSITION_RATES_OUTPUT = DIAGNOSTIC_DIR / "zero_persistence_transition_rates.csv"
TRANSITION_FIGURE = DIAGNOSTIC_DIR / "zero_persistence_transition_rates.png"
AT_RISK_PERFORMANCE_OUTPUT = DIAGNOSTIC_DIR / "at_risk_subset_model_performance.csv"
NEW_DECLINE_PERFORMANCE_OUTPUT = DIAGNOSTIC_DIR / "new_decline_transition_model_performance.csv"
REPORT_OUTPUT = DIAGNOSTIC_DIR / "zero_persistence_diagnostic_report.md"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Diagnose zero persistence and at-risk kelp decline model performance."
    )
    parser.add_argument("--input", type=Path, default=INPUT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DIAGNOSTIC_DIR)
    return parser.parse_args()


def require_columns(data: pd.DataFrame, columns: set[str]) -> None:
    """Raise a clear error if required columns are unavailable."""
    missing = sorted(columns - set(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def add_new_decline_label(data: pd.DataFrame) -> pd.DataFrame:
    """Add a stricter transition-into-low-canopy label."""
    require_columns(data, {CANOPY_COLUMN, NEXT_CANOPY_COLUMN, BASELINE_P25_COLUMN})
    labeled = data.copy()
    labeled[NEW_DECLINE_TARGET] = (
        (labeled[CANOPY_COLUMN] >= labeled[BASELINE_P25_COLUMN])
        & (labeled[NEXT_CANOPY_COLUMN] < labeled[BASELINE_P25_COLUMN])
    ).astype(int)
    return labeled


def split_data_for_target(data: pd.DataFrame, target: str) -> dict[str, pd.DataFrame]:
    """Apply the established temporal split and validate class balance."""
    splits = {
        "train": data.loc[data["year"].between(1989, 2016)].copy(),
        "validation": data.loc[data["year"].between(2017, 2020)].copy(),
        "test": data.loc[data["year"].between(2021, 2024)].copy(),
    }
    for split_name, split in splits.items():
        if split.empty:
            raise ValueError(f"{split_name} split is empty for target {target}.")
        classes = set(split[target].dropna().astype(int).unique())
        if classes != {0, 1}:
            raise ValueError(
                f"{split_name} split must contain both classes for target {target}; found {classes}."
            )
    return splits


def transition_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create zero/near-zero current-to-next-year transition count and rate tables."""
    require_columns(data, {CANOPY_COLUMN, NEXT_CANOPY_COLUMN})
    counts_rows = []
    rates_rows = []
    valid = data.loc[data[NEXT_CANOPY_COLUMN].notna()].copy()

    for threshold in ZERO_THRESHOLDS:
        current_zero = valid[CANOPY_COLUMN] <= threshold
        next_zero = valid[NEXT_CANOPY_COLUMN] <= threshold
        transitions = pd.DataFrame(
            {
                "current_state": np.where(current_zero, "current_zero", "current_nonzero"),
                "next_state": np.where(next_zero, "next_zero", "next_nonzero"),
            },
            index=valid.index,
        )
        transitions["transition"] = transitions["current_state"] + " -> " + transitions["next_state"]
        transition_counts = transitions["transition"].value_counts().to_dict()
        current_totals = transitions["current_state"].value_counts().to_dict()
        total_rows = len(transitions)

        zero_zero_count = int(transition_counts.get("current_zero -> next_zero", 0))
        zero_nonzero_count = int(transition_counts.get("current_zero -> next_nonzero", 0))
        nonzero_zero_count = int(transition_counts.get("current_nonzero -> next_zero", 0))
        nonzero_nonzero_count = int(transition_counts.get("current_nonzero -> next_nonzero", 0))
        current_zero_total = int(current_totals.get("current_zero", 0))
        current_nonzero_total = int(current_totals.get("current_nonzero", 0))
        zero_zero_rate = zero_zero_count / current_zero_total if current_zero_total else np.nan
        nonzero_zero_rate = nonzero_zero_count / current_nonzero_total if current_nonzero_total else np.nan

        for current_state in ["current_zero", "current_nonzero"]:
            for next_state in ["next_zero", "next_nonzero"]:
                transition = f"{current_state} -> {next_state}"
                count = int(transition_counts.get(transition, 0))
                denominator = int(current_totals.get(current_state, 0))
                counts_rows.append(
                    {
                        "canopy_threshold": threshold,
                        "current_state": current_state,
                        "next_state": next_state,
                        "transition": transition,
                        "count": count,
                        "total_observations": total_rows,
                        "current_state_total": denominator,
                    }
                )
                rates_rows.append(
                    {
                        "canopy_threshold": threshold,
                        "current_state": current_state,
                        "next_state": next_state,
                        "transition": transition,
                        "count": count,
                        "current_state_total": denominator,
                        "transition_rate_within_current_state": count / denominator if denominator else np.nan,
                        "transition_rate_overall": count / total_rows if total_rows else np.nan,
                        "zero_to_zero_persistence_rate": zero_zero_rate,
                        "nonzero_to_zero_transition_rate": nonzero_zero_rate,
                        "current_zero_count": current_zero_total,
                        "current_nonzero_count": current_nonzero_total,
                        "current_zero_next_nonzero_count": zero_nonzero_count,
                        "current_nonzero_next_zero_count": nonzero_zero_count,
                        "current_nonzero_next_nonzero_count": nonzero_nonzero_count,
                    }
                )

    return pd.DataFrame(counts_rows), pd.DataFrame(rates_rows)


def metric_value_or_nan(metric_function, y_true: pd.Series, scores_or_predictions: np.ndarray) -> float:
    """Return a metric value, or NaN if it is undefined."""
    try:
        return float(metric_function(y_true, scores_or_predictions))
    except ValueError:
        return np.nan


def metrics_for_scores(y_true: pd.Series, scores: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    """Compute model metrics at a fixed decision threshold."""
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "pr_auc": metric_value_or_nan(average_precision_score, y_true, scores),
        "roc_auc": metric_value_or_nan(roc_auc_score, y_true, scores),
        "accuracy": accuracy_score(y_true, predictions),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def subset_data(data: pd.DataFrame, threshold: float | None) -> tuple[str, pd.DataFrame]:
    """Return full-sample or current-canopy at-risk subset data."""
    if threshold is None:
        return "full_sample", data.copy()
    label = f"current_canopy_gt_{threshold:g}"
    return label, data.loc[data[CANOPY_COLUMN] > threshold].copy()


def evaluate_models_for_target(
    data: pd.DataFrame,
    target: str,
    subset_thresholds: list[float],
    include_full_sample: bool,
) -> pd.DataFrame:
    """Train and evaluate existing model families for one target and subset plan."""
    subset_plan: list[float | None] = []
    if include_full_sample:
        subset_plan.append(None)
    subset_plan.extend(subset_thresholds)

    rows = []
    for threshold in subset_plan:
        subset_label, subset = subset_data(data, threshold)
        splits = split_data_for_target(subset, target)
        sets = feature_sets(subset)
        y_train = splits["train"][target].astype(int)
        specs = model_specs(y_train)

        for feature_set_name, features in sets.items():
            x_train = splits["train"][features]
            for model_name, (estimator, needs_scaling) in specs.items():
                pipe = Pipeline(
                    [
                        ("preprocess", preprocessor(splits["train"], features, scale=needs_scaling)),
                        ("model", estimator),
                    ]
                )
                pipe.fit(x_train, y_train)

                y_test = splits["test"][target].astype(int)
                scores = predict_scores(pipe, splits["test"][features])
                metrics = metrics_for_scores(y_test, scores)
                rows.append(
                    {
                        "target": target,
                        "evaluation_subset": subset_label,
                        "current_canopy_threshold": np.nan if threshold is None else threshold,
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "split": "test",
                        "n_observations": len(y_test),
                        "n_positive_events": int(y_test.sum()),
                        "event_prevalence": float(y_test.mean()),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def plot_transition_rates(rates: pd.DataFrame, output: Path) -> None:
    """Plot zero persistence and nonzero-to-zero transition rates by threshold."""
    summary = (
        rates[["canopy_threshold", "zero_to_zero_persistence_rate", "nonzero_to_zero_transition_rate"]]
        .drop_duplicates()
        .sort_values("canopy_threshold")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        summary["canopy_threshold"],
        summary["zero_to_zero_persistence_rate"],
        marker="o",
        label="current zero -> next zero",
    )
    ax.plot(
        summary["canopy_threshold"],
        summary["nonzero_to_zero_transition_rate"],
        marker="o",
        label="current nonzero -> next zero",
    )
    ax.set_xlabel("Zero / near-zero relative canopy threshold")
    ax.set_ylabel("Transition rate")
    ax.set_ylim(0, 1)
    ax.set_title("Zero-persistence and new-zero transition rates")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def dataframe_to_markdown(data: pd.DataFrame, float_digits: int = 3) -> str:
    """Convert a compact DataFrame to a Markdown table without extra dependencies."""
    display = data.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.{float_digits}f}")
        else:
            display[column] = display[column].map(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy()]
    return "\n".join([header, separator, *rows])


def write_report(
    output: Path,
    counts: pd.DataFrame,
    rates: pd.DataFrame,
    at_risk_performance: pd.DataFrame,
    new_decline_performance: pd.DataFrame,
) -> None:
    """Write a methodological diagnostic report."""
    transition_summary = (
        rates[["canopy_threshold", "zero_to_zero_persistence_rate", "nonzero_to_zero_transition_rate"]]
        .drop_duplicates()
        .sort_values("canopy_threshold")
    )
    best_original = at_risk_performance.sort_values(
        ["evaluation_subset", "f1", "recall", "pr_auc"], ascending=[True, False, False, False]
    ).groupby("evaluation_subset", sort=False).head(1)
    best_new_decline = new_decline_performance.sort_values(
        ["evaluation_subset", "f1", "recall", "pr_auc"], ascending=[True, False, False, False]
    ).groupby("evaluation_subset", sort=False).head(1)

    full_best = at_risk_performance.loc[at_risk_performance["evaluation_subset"] == "full_sample"].sort_values(
        ["pr_auc", "f1", "recall"], ascending=False
    ).iloc[0]
    at_risk_005 = at_risk_performance.loc[
        at_risk_performance["evaluation_subset"] == "current_canopy_gt_0.05"
    ].sort_values(["pr_auc", "f1", "recall"], ascending=False).iloc[0]
    new_full_best = new_decline_performance.loc[
        new_decline_performance["evaluation_subset"] == "full_sample"
    ].sort_values(["pr_auc", "f1", "recall"], ascending=False).iloc[0]

    lines = [
        "# Zero-Persistence and At-Risk Early-Warning Diagnostic",
        "",
        "## Purpose",
        "",
        "This diagnostic checks whether model performance is partly explained by persistence of already-zero or already-low canopy states. It distinguishes three cases: true early warning of future decline, persistence of already-low canopy states, and trivial zero-to-zero prediction.",
        "",
        "## Zero-Persistence Transition Rates",
        "",
        dataframe_to_markdown(transition_summary),
        "",
        "## Original Label, Highest Test F1 Model by Evaluation Subset",
        "",
        dataframe_to_markdown(
            best_original[
                [
                    "evaluation_subset",
                    "feature_set",
                    "model",
                    "n_observations",
                    "n_positive_events",
                    "event_prevalence",
                    "precision",
                    "recall",
                    "f1",
                    "pr_auc",
                    "roc_auc",
                ]
            ]
        ),
        "",
        "## New-Decline Transition Label, Highest Test F1 Model by Evaluation Subset",
        "",
        dataframe_to_markdown(
            best_new_decline[
                [
                    "evaluation_subset",
                    "feature_set",
                    "model",
                    "n_observations",
                    "n_positive_events",
                    "event_prevalence",
                    "precision",
                    "recall",
                    "f1",
                    "pr_auc",
                    "roc_auc",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
        f"- Full-sample original-label best PR-AUC: `{full_best['feature_set']} / {full_best['model']}` with PR-AUC={full_best['pr_auc']:.3f}, recall={full_best['recall']:.3f}, and F1={full_best['f1']:.3f}.",
        f"- At-risk original-label best PR-AUC for `current_canopy > 0.05`: `{at_risk_005['feature_set']} / {at_risk_005['model']}` with PR-AUC={at_risk_005['pr_auc']:.3f}, recall={at_risk_005['recall']:.3f}, and F1={at_risk_005['f1']:.3f}.",
        f"- Full-sample new-decline-label best PR-AUC: `{new_full_best['feature_set']} / {new_full_best['model']}` with PR-AUC={new_full_best['pr_auc']:.3f}, recall={new_full_best['recall']:.3f}, and F1={new_full_best['f1']:.3f}.",
        "",
        "If performance is much stronger in the full sample than in current-nonzero or new-decline-transition evaluations, the model should be interpreted primarily as detecting canopy-state persistence rather than robust early warning. If useful skill remains in at-risk and new-decline-transition subsets, that is preliminary evidence that the model may capture warning signals before visible collapse.",
        "",
        "## Output Files",
        "",
        "- `outputs/diagnostics/zero_persistence_transition_counts.csv`",
        "- `outputs/diagnostics/zero_persistence_transition_rates.csv`",
        "- `outputs/diagnostics/zero_persistence_transition_rates.png`",
        "- `outputs/diagnostics/at_risk_subset_model_performance.csv`",
        "- `outputs/diagnostics/new_decline_transition_model_performance.csv`",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run all zero-persistence and at-risk diagnostics."""
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = main_subset(load_dataset(args.input))
    data = add_new_decline_label(data)

    counts, rates = transition_tables(data)
    at_risk_performance = evaluate_models_for_target(
        data=data,
        target=TARGET,
        subset_thresholds=AT_RISK_THRESHOLDS,
        include_full_sample=True,
    )
    new_decline_performance = evaluate_models_for_target(
        data=data,
        target=NEW_DECLINE_TARGET,
        subset_thresholds=AT_RISK_THRESHOLDS,
        include_full_sample=True,
    )

    transition_counts_output = output_dir / TRANSITION_COUNTS_OUTPUT.name
    transition_rates_output = output_dir / TRANSITION_RATES_OUTPUT.name
    transition_figure = output_dir / TRANSITION_FIGURE.name
    at_risk_output = output_dir / AT_RISK_PERFORMANCE_OUTPUT.name
    new_decline_output = output_dir / NEW_DECLINE_PERFORMANCE_OUTPUT.name
    report_output = output_dir / REPORT_OUTPUT.name

    counts.to_csv(transition_counts_output, index=False)
    rates.to_csv(transition_rates_output, index=False)
    at_risk_performance.to_csv(at_risk_output, index=False)
    new_decline_performance.to_csv(new_decline_output, index=False)
    plot_transition_rates(rates, transition_figure)
    write_report(report_output, counts, rates, at_risk_performance, new_decline_performance)

    transition_summary = rates[
        ["canopy_threshold", "zero_to_zero_persistence_rate", "nonzero_to_zero_transition_rate"]
    ].drop_duplicates()
    print("Zero-persistence diagnostic complete.")
    print(f"Rows used: {len(data)}")
    print("Temporal split: train 1989-2016, validation 2017-2020, test 2021-2024")
    print("Zero-persistence summary:")
    print(transition_summary.to_string(index=False))


if __name__ == "__main__":
    main()
