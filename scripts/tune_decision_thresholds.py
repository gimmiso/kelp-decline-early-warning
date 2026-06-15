"""Tune decision thresholds for recall-oriented kelp decline screening.

This script adds a validation-based threshold tuning diagnostic to the existing
model comparison workflow. It keeps the original default-threshold comparison
unchanged, selects thresholds on the validation period only, and applies the
selected thresholds once to the held-out test period.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
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
    split_data,
)


VALIDATION_GRID_OUTPUT = Path("outputs/metadata/threshold_tuning_validation_grid.csv")
SELECTED_THRESHOLDS_OUTPUT = Path("outputs/metadata/threshold_tuning_selected_thresholds.csv")
TEST_RESULTS_OUTPUT = Path("outputs/metadata/threshold_tuning_test_results.csv")
REPORT_OUTPUT = Path("outputs/metadata/threshold_tuning_report.md")
PR_TRADEOFF_FIGURE = Path("outputs/figures/threshold_tuning_recall_precision_tradeoff.png")
FALSE_NEGATIVE_FIGURE = Path("outputs/figures/threshold_tuning_false_negatives.png")
MODEL_RESULTS_GRID_OUTPUT = Path("outputs/model_results/threshold_tuning_results.csv")
MODEL_RESULTS_SELECTION_OUTPUT = Path("outputs/model_results/threshold_selection_summary.csv")
THRESHOLD_CURVE_FIGURE = Path("outputs/figures/precision_recall_threshold_curve.png")

THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Tune model decision thresholds on validation data.")
    parser.add_argument("--input", type=Path, default=INPUT_DATASET)
    parser.add_argument("--validation-grid-output", type=Path, default=VALIDATION_GRID_OUTPUT)
    parser.add_argument("--selected-thresholds-output", type=Path, default=SELECTED_THRESHOLDS_OUTPUT)
    parser.add_argument("--test-results-output", type=Path, default=TEST_RESULTS_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    parser.add_argument("--pr-tradeoff-figure", type=Path, default=PR_TRADEOFF_FIGURE)
    parser.add_argument("--false-negative-figure", type=Path, default=FALSE_NEGATIVE_FIGURE)
    parser.add_argument("--model-results-grid-output", type=Path, default=MODEL_RESULTS_GRID_OUTPUT)
    parser.add_argument("--model-results-selection-output", type=Path, default=MODEL_RESULTS_SELECTION_OUTPUT)
    parser.add_argument("--threshold-curve-figure", type=Path, default=THRESHOLD_CURVE_FIGURE)
    return parser.parse_args()


def threshold_metrics(y_true: pd.Series | np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, object]:
    """Compute threshold-dependent classification metrics."""
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "f2": fbeta_score(y_true, predictions, beta=2, zero_division=0),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "predicted_positive_rate": float(predictions.mean()),
    }


def validation_grid_for_scores(
    model_name: str,
    feature_set_name: str,
    y_validation: pd.Series,
    validation_scores: np.ndarray,
) -> pd.DataFrame:
    """Evaluate all candidate thresholds on validation scores."""
    rows = []
    for threshold in THRESHOLDS:
        row = threshold_metrics(y_validation, validation_scores, float(threshold))
        row.update(
            {
                "model": model_name,
                "feature_set": feature_set_name,
                "selection_split": "validation",
            }
        )
        rows.append(row)
    columns = [
        "model",
        "feature_set",
        "threshold",
        "precision",
        "recall",
        "f1",
        "f2",
        "false_positives",
        "false_negatives",
        "true_positives",
        "true_negatives",
        "predicted_positive_rate",
        "selection_split",
    ]
    return pd.DataFrame(rows)[columns]


def precision_floor_choice(group: pd.DataFrame) -> tuple[pd.Series, float, bool]:
    """Select max recall subject to a precision floor, with documented fallback."""
    eligible = group.loc[group["precision"] >= 0.65].copy()
    if eligible.empty:
        eligible = group.loc[group["precision"] >= 0.50].copy()
        if eligible.empty:
            return group.sort_values(["precision", "recall", "f2", "f1"], ascending=False).iloc[0], np.nan, True
        return eligible.sort_values(["recall", "f2", "f1", "precision"], ascending=False).iloc[0], 0.50, True
    return eligible.sort_values(["recall", "f2", "f1", "precision"], ascending=False).iloc[0], 0.65, False


def select_thresholds(validation_grid: pd.DataFrame) -> pd.DataFrame:
    """Select default, F1, F2, recall-oriented, and precision-floor thresholds."""
    selected_rows = []
    for (model_name, feature_set_name), group in validation_grid.groupby(["model", "feature_set"], sort=False):
        eligible = group.loc[group["recall"] >= 0.70].copy()
        if eligible.empty:
            recall_choice = group.sort_values(["recall", "f1", "precision"], ascending=False).iloc[0]
        else:
            recall_choice = eligible.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
        max_f1_choice = group.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]
        max_f2_choice = group.sort_values(["f2", "recall", "precision", "f1"], ascending=False).iloc[0]
        default_choice = group.loc[np.isclose(group["threshold"], 0.50)].iloc[0]
        floor_choice, precision_floor_used, floor_fallback = precision_floor_choice(group)

        for selection_rule, choice, precision_floor_target, precision_floor_actual, fallback_used in [
            ("default_0.5", default_choice, np.nan, np.nan, False),
            ("max_f1", max_f1_choice, np.nan, np.nan, False),
            ("max_f2", max_f2_choice, np.nan, np.nan, False),
            ("recall_ge_0.70_then_max_f1", recall_choice, np.nan, np.nan, False),
            ("max_recall_precision_ge_0.65", floor_choice, 0.65, precision_floor_used, floor_fallback),
        ]:
            selected_rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "selection_rule": selection_rule,
                    "selected_threshold": float(choice["threshold"]),
                    "precision_floor_target": precision_floor_target,
                    "precision_floor_used": precision_floor_actual,
                    "precision_floor_fallback_used": bool(fallback_used),
                    "validation_precision": float(choice["precision"]),
                    "validation_recall": float(choice["recall"]),
                    "validation_f1": float(choice["f1"]),
                    "validation_f2": float(choice["f2"]),
                    "validation_false_positives": int(choice["false_positives"]),
                    "validation_false_negatives": int(choice["false_negatives"]),
                }
            )
    return pd.DataFrame(selected_rows)


def test_results_for_scores(
    selected_thresholds: pd.DataFrame,
    score_lookup: dict[tuple[str, str], dict[str, object]],
) -> pd.DataFrame:
    """Apply validation-selected thresholds to the held-out test scores."""
    rows = []
    for _, selected in selected_thresholds.iterrows():
        key = (selected["feature_set"], selected["model"])
        values = score_lookup[key]
        y_test = values["y_test"]
        test_scores = values["test_scores"]
        threshold = float(selected["selected_threshold"])
        row = threshold_metrics(y_test, test_scores, threshold)
        rows.append(
            {
                "model": selected["model"],
                "feature_set": selected["feature_set"],
                "selection_rule": selected["selection_rule"],
                "threshold": threshold,
                "test_precision": row["precision"],
                "test_recall": row["recall"],
                "test_f1": row["f1"],
                "test_f2": row["f2"],
                "test_pr_auc": average_precision_score(y_test, test_scores),
                "test_roc_auc": roc_auc_score(y_test, test_scores),
                "test_false_positives": row["false_positives"],
                "test_false_negatives": row["false_negatives"],
                "test_true_positives": row["true_positives"],
                "test_true_negatives": row["true_negatives"],
                "test_predicted_positive_rate": row["predicted_positive_rate"],
            }
        )
    return pd.DataFrame(rows)


def threshold_selection_summary(selected_thresholds: pd.DataFrame, test_results: pd.DataFrame) -> pd.DataFrame:
    """Combine validation-selected thresholds with held-out test performance."""
    merged = selected_thresholds.merge(
        test_results,
        left_on=["model", "feature_set", "selection_rule", "selected_threshold"],
        right_on=["model", "feature_set", "selection_rule", "threshold"],
        how="left",
    )
    return merged.drop(columns=["threshold"])


def train_and_score(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train existing model specifications and return threshold diagnostics."""
    splits = split_data(data)
    sets = feature_sets(data)
    y_train = splits["train"][TARGET].astype(int)
    specs = model_specs(y_train)
    validation_grids = []
    score_lookup: dict[tuple[str, str], dict[str, object]] = {}

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

            y_validation = splits["validation"][TARGET].astype(int)
            validation_scores = predict_scores(pipe, splits["validation"][features])
            validation_grids.append(
                validation_grid_for_scores(model_name, feature_set_name, y_validation, validation_scores)
            )

            y_test = splits["test"][TARGET].astype(int)
            test_scores = predict_scores(pipe, splits["test"][features])
            score_lookup[(feature_set_name, model_name)] = {
                "y_test": y_test,
                "test_scores": test_scores,
            }

    validation_grid = pd.concat(validation_grids, ignore_index=True)
    selected_thresholds = select_thresholds(validation_grid)
    test_results = test_results_for_scores(selected_thresholds, score_lookup)
    return validation_grid, selected_thresholds, test_results


def plot_precision_recall_tradeoff(validation_grid: pd.DataFrame, selected_thresholds: pd.DataFrame, output: Path) -> None:
    """Plot validation precision-recall tradeoffs across thresholds."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True, sharey=True)
    feature_sets = ["canopy_only", "oisst_only", "canopy_noaa"]
    for ax, feature_set_name in zip(axes, feature_sets, strict=True):
        subset = validation_grid.loc[validation_grid["feature_set"] == feature_set_name]
        for model_name, group in subset.groupby("model"):
            ax.plot(group["recall"], group["precision"], marker="o", markersize=3, linewidth=1.2, label=model_name)
        selected = selected_thresholds.loc[
            (selected_thresholds["feature_set"] == feature_set_name)
            & (selected_thresholds["selection_rule"] == "recall_ge_0.70_then_max_f1")
        ]
        ax.scatter(
            selected["validation_recall"],
            selected["validation_precision"],
            color="black",
            s=42,
            marker="x",
            label="Selected recall-oriented threshold",
        )
        ax.set_title(f"Validation precision-recall tradeoff: {feature_set_name}")
        ax.set_ylabel("Precision")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Recall")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_false_negatives(test_results: pd.DataFrame, output: Path) -> None:
    """Plot test false negatives for default and recall-oriented thresholds."""
    output.parent.mkdir(parents=True, exist_ok=True)
    subset = test_results.loc[
        test_results["selection_rule"].isin(["default_0.5", "recall_ge_0.70_then_max_f1"])
    ].copy()
    subset["model_feature"] = subset["feature_set"] + " / " + subset["model"]
    ordered = (
        subset.loc[subset["selection_rule"] == "default_0.5"]
        .sort_values(["test_false_negatives", "model_feature"])
        ["model_feature"]
        .tolist()
    )
    subset["model_feature"] = pd.Categorical(subset["model_feature"], categories=ordered, ordered=True)
    pivot = subset.pivot(index="model_feature", columns="selection_rule", values="test_false_negatives").sort_index()
    labels = ["Default 0.5", "Recall-oriented"]
    columns = ["default_0.5", "recall_ge_0.70_then_max_f1"]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(pivot))
    width = 0.38
    for offset, column, label in zip([-width / 2, width / 2], columns, labels, strict=True):
        ax.bar(x + offset, pivot[column], width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=75, ha="right")
    ax.set_ylabel("Test false negatives")
    ax.set_title("False negatives under default and recall-oriented thresholds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_precision_recall_threshold_curve(validation_grid: pd.DataFrame, output: Path) -> None:
    """Plot validation precision, recall, and F2 as functions of threshold."""
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_pairs = [
        ("canopy_only", "Random Forest"),
        ("canopy_noaa", "SVM"),
        ("canopy_only", "Logistic Regression"),
        ("canopy_noaa", "Random Forest"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, (feature_set_name, model_name) in zip(axes.flat, selected_pairs, strict=True):
        subset = validation_grid.loc[
            (validation_grid["feature_set"] == feature_set_name)
            & (validation_grid["model"] == model_name)
        ].sort_values("threshold")
        ax.plot(subset["threshold"], subset["precision"], marker="o", label="Precision")
        ax.plot(subset["threshold"], subset["recall"], marker="o", label="Recall")
        ax.plot(subset["threshold"], subset["f2"], marker="o", label="F2")
        ax.axhline(0.65, color="gray", linestyle="--", linewidth=1, label="Precision floor 0.65")
        ax.set_title(f"{feature_set_name} / {model_name}")
        ax.set_xlabel("Decision threshold")
        ax.set_ylabel("Validation score")
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Validation precision, recall, and F2 across decision thresholds")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_model_feature(row: pd.Series) -> str:
    """Return a compact model-feature label."""
    return f"{row['feature_set']} / {row['model']}"


def dataframe_to_markdown(data: pd.DataFrame, float_digits: int = 3) -> str:
    """Convert a small DataFrame to a GitHub-flavored Markdown table."""
    display = data.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.{float_digits}f}")
        else:
            display[column] = display[column].map(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy()]
    return "\n".join([header, separator, *rows])


def write_report(
    validation_grid: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    test_results: pd.DataFrame,
    output: Path,
) -> None:
    """Write a concise threshold tuning report."""
    default = test_results.loc[test_results["selection_rule"] == "default_0.5"].copy()
    recall = test_results.loc[test_results["selection_rule"] == "recall_ge_0.70_then_max_f1"].copy()
    precision_floor = test_results.loc[test_results["selection_rule"] == "max_recall_precision_ge_0.65"].copy()
    f1_optimal = test_results.loc[test_results["selection_rule"] == "max_f1"].copy()
    f2_optimal = test_results.loc[test_results["selection_rule"] == "max_f2"].copy()
    comparison = recall.merge(
        default,
        on=["feature_set", "model"],
        suffixes=("_recall", "_default"),
    )
    comparison["recall_gain"] = comparison["test_recall_recall"] - comparison["test_recall_default"]
    comparison["false_negative_reduction"] = (
        comparison["test_false_negatives_default"] - comparison["test_false_negatives_recall"]
    )
    comparison["precision_change"] = comparison["test_precision_recall"] - comparison["test_precision_default"]
    comparison["f1_change"] = comparison["test_f1_recall"] - comparison["test_f1_default"]

    most_recall_gain = comparison.sort_values(
        ["recall_gain", "false_negative_reduction", "test_f1_recall"], ascending=False
    ).iloc[0]
    most_fn_reduction = comparison.sort_values(
        ["false_negative_reduction", "recall_gain", "test_f1_recall"], ascending=False
    ).iloc[0]
    best_early_warning = recall.sort_values(
        ["test_recall", "test_precision", "test_f2", "test_f1", "test_pr_auc"], ascending=False
    ).iloc[0]
    best_f1_recall = recall.sort_values(["test_f1", "test_recall", "test_precision"], ascending=False).iloc[0]

    canopy_rf_default = default.loc[
        (default["feature_set"] == "canopy_only") & (default["model"] == "Random Forest")
    ].iloc[0]
    canopy_rf_recall = recall.loc[
        (recall["feature_set"] == "canopy_only") & (recall["model"] == "Random Forest")
    ].iloc[0]
    noaa_svm_default = default.loc[
        (default["feature_set"] == "canopy_noaa") & (default["model"] == "SVM")
    ].iloc[0]
    noaa_svm_recall = recall.loc[
        (recall["feature_set"] == "canopy_noaa") & (recall["model"] == "SVM")
    ].iloc[0]

    top_table = comparison[
        [
            "feature_set",
            "model",
            "threshold_recall",
            "test_recall_default",
            "test_recall_recall",
            "recall_gain",
            "test_false_negatives_default",
            "test_false_negatives_recall",
            "false_negative_reduction",
            "test_precision_default",
            "test_precision_recall",
            "precision_change",
            "test_f1_recall",
            "test_f2_recall",
        ]
    ].sort_values(["false_negative_reduction", "recall_gain"], ascending=False)

    lines = [
        "# Recall-Oriented Threshold Tuning Report",
        "",
        "## Purpose",
        "",
        "This diagnostic evaluates whether validation-selected decision thresholds can reduce false negatives for kelp decline early-warning screening. The original model comparison at the default 0.5 threshold remains unchanged.",
        "",
        "Threshold tuning changes the operating point of the classifier, trading precision for recall. It does not improve or change ranking metrics such as PR-AUC or ROC-AUC.",
        "",
        "Thresholds were selected using the validation period only and then fixed for the held-out test period to avoid test-set leakage.",
        "",
        "## Data and Split",
        "",
        "- Dataset: `data/processed/modeling_dataset_ge500_noaa_v1.csv`",
        "- Train: 1989-2016",
        "- Validation: 2017-2020",
        "- Test: 2021-2024",
        "- Candidate thresholds: 0.05 to 0.95 in 0.05 increments",
        "- Selection rules: default threshold 0.50; max F1; max F2; recall >= 0.70 then max F1; max recall subject to precision >= 0.65.",
        "- If no threshold satisfies precision >= 0.65, the precision-floor rule falls back to precision >= 0.50 and records that fallback.",
        "",
        "## Main Findings",
        "",
        f"- Largest test recall gain: {format_model_feature(most_recall_gain)} "
        f"({most_recall_gain['test_recall_default']:.3f} to {most_recall_gain['test_recall_recall']:.3f}; "
        f"false negatives {int(most_recall_gain['test_false_negatives_default'])} to {int(most_recall_gain['test_false_negatives_recall'])}).",
        f"- Largest false-negative reduction: {format_model_feature(most_fn_reduction)} "
        f"({int(most_fn_reduction['false_negative_reduction'])} fewer false negatives).",
        f"- Highest recall-oriented test recall: {format_model_feature(best_early_warning)} "
        f"(threshold={best_early_warning['threshold']:.2f}, recall={best_early_warning['test_recall']:.3f}, "
        f"precision={best_early_warning['test_precision']:.3f}, F1={best_early_warning['test_f1']:.3f}, "
        f"F2={best_early_warning['test_f2']:.3f}).",
        f"- Best recall-oriented F1: {format_model_feature(best_f1_recall)} "
        f"(threshold={best_f1_recall['threshold']:.2f}, recall={best_f1_recall['test_recall']:.3f}, "
        f"precision={best_f1_recall['test_precision']:.3f}, F1={best_f1_recall['test_f1']:.3f}, "
        f"F2={best_f1_recall['test_f2']:.3f}).",
        "",
        "## Specific Model Checks",
        "",
        f"- `canopy_only / Random Forest` remained strong after threshold tuning: recall changed from {canopy_rf_default['test_recall']:.3f} to {canopy_rf_recall['test_recall']:.3f}, false negatives from {int(canopy_rf_default['test_false_negatives'])} to {int(canopy_rf_recall['test_false_negatives'])}, precision from {canopy_rf_default['test_precision']:.3f} to {canopy_rf_recall['test_precision']:.3f}, and F1 from {canopy_rf_default['test_f1']:.3f} to {canopy_rf_recall['test_f1']:.3f}.",
        f"- `canopy_noaa / SVM` became more useful as an early-warning screen after threshold tuning: recall changed from {noaa_svm_default['test_recall']:.3f} to {noaa_svm_recall['test_recall']:.3f}, false negatives from {int(noaa_svm_default['test_false_negatives'])} to {int(noaa_svm_recall['test_false_negatives'])}, precision from {noaa_svm_default['test_precision']:.3f} to {noaa_svm_recall['test_precision']:.3f}, and F1 from {noaa_svm_default['test_f1']:.3f} to {noaa_svm_recall['test_f1']:.3f}.",
        "",
        "## Threshold-Tuned Test Comparison",
        "",
        dataframe_to_markdown(top_table),
        "",
        "## Interpretation",
        "",
        "For early-warning screening, a recall-oriented operating point can be appropriate when false negatives are more costly than false positives. The preferred threshold-tuned model depends on whether the priority is maximum recall or a more balanced precision-recall trade-off.",
        "",
        f"- Main threshold-tuned model for a balanced recall-precision trade-off: `{format_model_feature(best_f1_recall)}`.",
        f"- High-sensitivity screening scenario: `{format_model_feature(best_early_warning)}`.",
        f"- Precision-floor rule rows: {len(precision_floor)}; F2-optimal rows: {len(f2_optimal)}.",
        "",
        "These results should be interpreted as operating-point diagnostics, not as evidence that threshold tuning improves the underlying model ranking quality.",
        "",
        "## Output Files",
        "",
        "- `outputs/metadata/threshold_tuning_validation_grid.csv`",
        "- `outputs/metadata/threshold_tuning_selected_thresholds.csv`",
        "- `outputs/metadata/threshold_tuning_test_results.csv`",
        "- `outputs/model_results/threshold_tuning_results.csv`",
        "- `outputs/model_results/threshold_selection_summary.csv`",
        "- `outputs/figures/threshold_tuning_recall_precision_tradeoff.png`",
        "- `outputs/figures/threshold_tuning_false_negatives.png`",
        "- `outputs/figures/precision_recall_threshold_curve.png`",
        "",
        "## Validation Grid Summary",
        "",
        f"- Validation grid rows: {len(validation_grid)}",
        f"- Selected threshold rows: {len(selected_thresholds)}",
        f"- Test result rows: {len(test_results)}",
        f"- F1-optimal threshold rows: {len(f1_optimal)}",
        f"- F2-optimal threshold rows: {len(f2_optimal)}",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run validation-based threshold tuning."""
    args = parse_args()
    data = main_subset(load_dataset(args.input))
    validation_grid, selected_thresholds, test_results = train_and_score(data)
    selection_summary = threshold_selection_summary(selected_thresholds, test_results)

    for output in [
        args.validation_grid_output,
        args.selected_thresholds_output,
        args.test_results_output,
        args.report_output,
        args.pr_tradeoff_figure,
        args.false_negative_figure,
        args.model_results_grid_output,
        args.model_results_selection_output,
        args.threshold_curve_figure,
    ]:
        output.parent.mkdir(parents=True, exist_ok=True)

    validation_grid.to_csv(args.validation_grid_output, index=False)
    selected_thresholds.to_csv(args.selected_thresholds_output, index=False)
    test_results.to_csv(args.test_results_output, index=False)
    validation_grid.to_csv(args.model_results_grid_output, index=False)
    selection_summary.to_csv(args.model_results_selection_output, index=False)
    plot_precision_recall_tradeoff(validation_grid, selected_thresholds, args.pr_tradeoff_figure)
    plot_false_negatives(test_results, args.false_negative_figure)
    plot_precision_recall_threshold_curve(validation_grid, args.threshold_curve_figure)
    write_report(validation_grid, selected_thresholds, test_results, args.report_output)

    recall_results = test_results.loc[test_results["selection_rule"] == "recall_ge_0.70_then_max_f1"]
    best = recall_results.sort_values(["test_recall", "test_precision", "test_f2", "test_f1"], ascending=False).iloc[0]
    print("Threshold tuning complete.")
    print("Thresholds selected on validation period only: 2017-2020")
    print("Thresholds applied to held-out test period: 2021-2024")
    print(
        "Highest recall-oriented test recall: "
        f"{best['feature_set']} / {best['model']} = "
        f"recall {best['test_recall']:.3f}, precision {best['test_precision']:.3f}, "
        f"threshold {best['threshold']:.2f}"
    )


if __name__ == "__main__":
    main()
