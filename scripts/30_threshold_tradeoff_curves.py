#!/usr/bin/env python3
"""Build threshold trade-off curves for kelp decline risk screening.

This script reuses out-of-sample prediction probabilities from the existing
model-comparison workflow and sweeps decision thresholds from 0.00 to 1.00.
The goal is not to claim operational early-warning skill. Instead, the output
quantifies the decision-support trade-off between missed declines and false
alerts under different alert thresholds.

Run from the repository root:

    python scripts/30_threshold_tradeoff_curves.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_PREDICTIONS = Path("outputs/metadata/model_comparison_test_predictions.csv")
DEFAULT_METRICS_OUTPUT = Path("outputs/tables/threshold_tradeoff_metrics.csv")
DEFAULT_REPORT_OUTPUT = Path("outputs/reports/threshold_tradeoff_summary.md")
DEFAULT_FIGURE_DIR = Path("outputs/figures")

RECALL_ALERTS_FIGURE = DEFAULT_FIGURE_DIR / "threshold_tradeoff_recall_vs_alerts.png"
PRECISION_RECALL_FIGURE = DEFAULT_FIGURE_DIR / "threshold_tradeoff_precision_recall.png"
FALSE_ALERTS_MISSED_FIGURE = DEFAULT_FIGURE_DIR / "threshold_tradeoff_false_alerts_vs_missed.png"
F2_THRESHOLD_FIGURE = DEFAULT_FIGURE_DIR / "threshold_tradeoff_f2_vs_threshold.png"

SELECTED_THRESHOLDS = [0.10, 0.20, 0.30, 0.50]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sweep alert thresholds and quantify false-alert versus missed-decline trade-offs."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument(
        "--label-target",
        default="decline_event_next",
        help="Label target name to use when the prediction file does not contain a label_target column.",
    )
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or NaN when the denominator is zero."""
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def fbeta(precision: float, recall: float, beta: float) -> float:
    """Compute F-beta safely from precision and recall."""
    if np.isnan(precision) or np.isnan(recall):
        return float("nan")
    beta2 = beta**2
    denominator = beta2 * precision + recall
    if denominator == 0:
        return float("nan")
    return float((1 + beta2) * precision * recall / denominator)


def load_predictions(path: Path, default_label_target: str) -> pd.DataFrame:
    """Load and normalize tidy prediction probabilities."""
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}. Run `python scripts/train_model_comparison.py` "
            "first to regenerate out-of-sample model probabilities."
        )

    predictions = pd.read_csv(path)
    required = {"feature_set", "model", "y_true", "y_proba"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns for threshold trade-off analysis: {missing}")

    if "label_target" not in predictions.columns:
        if "target_definition" in predictions.columns:
            predictions["label_target"] = predictions["target_definition"].astype(str)
        else:
            predictions["label_target"] = default_label_target
    if "split" not in predictions.columns:
        predictions["split"] = "out_of_sample"

    predictions = predictions.copy()
    predictions["feature_set"] = predictions["feature_set"].astype(str)
    predictions["model"] = predictions["model"].astype(str)
    predictions["label_target"] = predictions["label_target"].astype(str)
    predictions["split"] = predictions["split"].astype(str)
    predictions["y_true"] = predictions["y_true"].astype(int)
    predictions["y_proba"] = pd.to_numeric(predictions["y_proba"], errors="coerce")
    predictions = predictions.dropna(subset=["y_proba"])

    if predictions.empty:
        raise ValueError(f"No usable prediction probabilities found in {path}.")
    if not predictions["y_true"].isin([0, 1]).all():
        raise ValueError("Prediction labels must be binary 0/1 in `y_true`.")
    return predictions


def threshold_grid() -> np.ndarray:
    """Return thresholds from 0.00 to 1.00 in increments of 0.01."""
    return np.round(np.arange(0.0, 1.0001, 0.01), 2)


def metrics_for_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Compute threshold-dependent alert and classification metrics."""
    predicted = (scores >= threshold).astype(int)
    positive = y_true == 1
    negative = y_true == 0
    tp = int(np.sum((predicted == 1) & positive))
    fp = int(np.sum((predicted == 1) & negative))
    tn = int(np.sum((predicted == 0) & negative))
    fn = int(np.sum((predicted == 0) & positive))

    n_obs = int(len(y_true))
    n_positive = int(np.sum(positive))
    n_negative = int(np.sum(negative))
    alerts = tp + fp
    precision = safe_divide(tp, alerts)
    recall = safe_divide(tp, n_positive)
    specificity = safe_divide(tn, n_negative)
    false_positive_rate = safe_divide(fp, n_negative)
    f1 = fbeta(precision, recall, beta=1)
    f2 = fbeta(precision, recall, beta=2)

    return {
        "threshold": float(threshold),
        "n_obs": n_obs,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": false_positive_rate,
        "f1": f1,
        "f2": f2,
        "alerts": int(alerts),
        "alerts_per_100_cell_years": 100 * safe_divide(alerts, n_obs),
        "false_alerts_per_100_cell_years": 100 * safe_divide(fp, n_obs),
        "missed_declines_per_100_true_declines": 100 * safe_divide(fn, n_positive),
    }


def build_threshold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Build tidy threshold metrics for every label/feature/model/split group."""
    rows: list[dict[str, object]] = []
    group_cols = ["label_target", "feature_set", "model", "split"]
    for keys, group in predictions.groupby(group_cols, sort=True):
        label_target, feature_set, model_name, split = keys
        y_true = group["y_true"].to_numpy(dtype=int)
        scores = group["y_proba"].to_numpy(dtype=float)
        for threshold in threshold_grid():
            row = metrics_for_threshold(y_true, scores, float(threshold))
            row.update(
                {
                    "label_target": label_target,
                    "feature_set": feature_set,
                    "model": model_name,
                    "split": split,
                }
            )
            rows.append(row)

    columns = [
        "label_target",
        "feature_set",
        "model",
        "split",
        "threshold",
        "n_obs",
        "n_positive",
        "n_negative",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "specificity",
        "false_positive_rate",
        "f1",
        "f2",
        "alerts",
        "alerts_per_100_cell_years",
        "false_alerts_per_100_cell_years",
        "missed_declines_per_100_true_declines",
    ]
    metrics = pd.DataFrame(rows)[columns]
    return metrics.sort_values(["label_target", "feature_set", "model", "split", "threshold"]).reset_index(drop=True)


def model_score_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize threshold-independent ranking scores for choosing overview plots."""
    rows: list[dict[str, object]] = []
    group_cols = ["label_target", "feature_set", "model", "split"]
    for keys, group in predictions.groupby(group_cols, sort=True):
        y_true = group["y_true"].to_numpy(dtype=int)
        scores = group["y_proba"].to_numpy(dtype=float)
        unique_classes = np.unique(y_true)
        pr_auc = average_precision_score(y_true, scores) if len(unique_classes) == 2 else float("nan")
        roc_auc = roc_auc_score(y_true, scores) if len(unique_classes) == 2 else float("nan")
        rows.append(
            {
                "label_target": keys[0],
                "feature_set": keys[1],
                "model": keys[2],
                "split": keys[3],
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "n_obs": int(len(group)),
                "n_positive": int(group["y_true"].sum()),
            }
        )
    return pd.DataFrame(rows)


def choose_overview_group(score_summary: pd.DataFrame) -> dict[str, str]:
    """Choose the main overview group using highest PR-AUC, then ROC-AUC."""
    ranked = score_summary.sort_values(["pr_auc", "roc_auc", "n_positive"], ascending=False)
    if ranked.empty:
        raise ValueError("No model groups available for overview plotting.")
    row = ranked.iloc[0]
    return {
        "label_target": str(row["label_target"]),
        "feature_set": str(row["feature_set"]),
        "model": str(row["model"]),
        "split": str(row["split"]),
    }


def subset_for_group(metrics: pd.DataFrame, group: dict[str, str]) -> pd.DataFrame:
    """Return threshold metrics for one label/feature/model/split group."""
    mask = (
        metrics["label_target"].eq(group["label_target"])
        & metrics["feature_set"].eq(group["feature_set"])
        & metrics["model"].eq(group["model"])
        & metrics["split"].eq(group["split"])
    )
    subset = metrics.loc[mask].sort_values("threshold").copy()
    if subset.empty:
        raise ValueError(f"No threshold metrics found for overview group: {group}")
    return subset


def annotate_selected_thresholds(ax: plt.Axes, data: pd.DataFrame, x_col: str, y_col: str) -> None:
    """Annotate a small set of commonly discussed thresholds."""
    for threshold in SELECTED_THRESHOLDS:
        row = data.loc[np.isclose(data["threshold"], threshold)]
        if row.empty:
            continue
        x = row.iloc[0][x_col]
        y = row.iloc[0][y_col]
        if pd.notna(x) and pd.notna(y):
            ax.scatter([x], [y], s=28, color="#1F2430", zorder=3)
            ax.annotate(
                f"{threshold:.2f}",
                xy=(x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="#1F2430",
            )


def setup_axis(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    """Apply a restrained report-friendly axis style."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#E6E8F0", linewidth=0.8)
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D7DBE7")
    ax.spines["bottom"].set_color("#D7DBE7")


def figure_caption(group: dict[str, str]) -> str:
    """Return compact caption text for plot titles."""
    return f"{group['label_target']} | {group['feature_set']} / {group['model']} | {group['split']}"


def plot_recall_vs_alerts(data: pd.DataFrame, group: dict[str, str], output: Path) -> None:
    """Plot recall versus alert burden."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(data["alerts_per_100_cell_years"], data["recall"], color="#2E4780", linewidth=2)
    annotate_selected_thresholds(ax, data, "alerts_per_100_cell_years", "recall")
    setup_axis(ax, "Alerts per 100 cell-years", "Recall")
    ax.set_title("Recall versus alert burden", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_precision_recall(data: pd.DataFrame, group: dict[str, str], output: Path) -> None:
    """Plot precision-recall threshold curve."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    valid = data.dropna(subset=["precision", "recall"])
    scatter = ax.scatter(
        valid["recall"],
        valid["precision"],
        c=valid["threshold"],
        cmap="viridis_r",
        s=22,
        edgecolor="none",
    )
    ax.plot(valid["recall"], valid["precision"], color="#7A828F", linewidth=1, alpha=0.6)
    annotate_selected_thresholds(ax, valid, "recall", "precision")
    setup_axis(ax, "Recall", "Precision")
    ax.set_title("Precision-recall trade-off by threshold", loc="left", fontsize=12, fontweight="bold")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Decision threshold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_false_alerts_vs_missed(data: pd.DataFrame, group: dict[str, str], output: Path) -> None:
    """Plot false-alert burden versus missed-decline burden."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    valid = data.dropna(subset=["false_alerts_per_100_cell_years", "missed_declines_per_100_true_declines"])
    ax.plot(
        valid["false_alerts_per_100_cell_years"],
        valid["missed_declines_per_100_true_declines"],
        color="#2E4780",
        linewidth=2,
    )
    annotate_selected_thresholds(
        ax,
        valid,
        "false_alerts_per_100_cell_years",
        "missed_declines_per_100_true_declines",
    )
    setup_axis(ax, "False alerts per 100 cell-years", "Missed declines per 100 true decline cases")
    ax.set_title("False alerts versus missed declines", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_f2_vs_threshold(data: pd.DataFrame, group: dict[str, str], output: Path) -> None:
    """Plot F2 score versus threshold."""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(data["threshold"], data["f2"], color="#2E4780", linewidth=2)
    annotate_selected_thresholds(ax, data, "threshold", "f2")
    setup_axis(ax, "Decision threshold", "F2 score")
    ax.set_title("Recall-weighted F2 by threshold", loc="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


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


def selected_threshold_table(metrics: pd.DataFrame, group: dict[str, str]) -> pd.DataFrame:
    """Return selected threshold rows for the summary report."""
    overview = subset_for_group(metrics, group)
    table = overview.loc[overview["threshold"].isin(SELECTED_THRESHOLDS)].copy()
    return table[
        [
            "threshold",
            "precision",
            "recall",
            "f1",
            "f2",
            "alerts_per_100_cell_years",
            "false_alerts_per_100_cell_years",
            "missed_declines_per_100_true_declines",
            "tp",
            "fp",
            "fn",
            "tn",
        ]
    ]


def write_summary(
    output: Path,
    predictions_path: Path,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    score_summary: pd.DataFrame,
    overview_group: dict[str, str],
) -> None:
    """Write a concise threshold trade-off report."""
    labels = sorted(predictions["label_target"].unique())
    feature_sets = sorted(predictions["feature_set"].unique())
    models = sorted(predictions["model"].unique())
    splits = sorted(predictions["split"].unique())
    selected_table = selected_threshold_table(metrics, overview_group)
    best_f2 = subset_for_group(metrics, overview_group).sort_values(["f2", "recall", "precision"], ascending=False).iloc[0]
    overview_scores = score_summary.loc[
        score_summary["label_target"].eq(overview_group["label_target"])
        & score_summary["feature_set"].eq(overview_group["feature_set"])
        & score_summary["model"].eq(overview_group["model"])
        & score_summary["split"].eq(overview_group["split"])
    ].iloc[0]

    lines = [
        "# Threshold Trade-off Summary",
        "",
        "## Purpose",
        "",
        "This analysis evaluates how alert thresholds change the balance between missed declines and false alerts. It is intended as a persistence-aware decision-support diagnostic rather than evidence of fully operational early-warning skill.",
        "",
        "## Inputs",
        "",
        f"- Prediction file: `{predictions_path}`",
        f"- Prediction rows: `{len(predictions):,}`",
        f"- Label targets evaluated: `{', '.join(labels)}`",
        f"- Feature sets evaluated: `{', '.join(feature_sets)}`",
        f"- Models evaluated: `{', '.join(models)}`",
        f"- Splits evaluated: `{', '.join(splits)}`",
        "",
        "## Threshold Grid",
        "",
        "- Thresholds: `0.00` to `1.00` in increments of `0.01`.",
        "- Lower thresholds increase recall but increase alert burden.",
        "- Higher thresholds reduce false alerts but increase missed declines.",
        "",
        "## Overview Figure Group",
        "",
        "The overview figures use the highest PR-AUC model group in the input prediction file:",
        "",
        f"- Label target: `{overview_group['label_target']}`",
        f"- Feature set: `{overview_group['feature_set']}`",
        f"- Model: `{overview_group['model']}`",
        f"- Split: `{overview_group['split']}`",
        f"- PR-AUC: `{overview_scores['pr_auc']:.3f}`",
        f"- ROC-AUC: `{overview_scores['roc_auc']:.3f}`",
        "",
        "## Selected Thresholds for Overview Group",
        "",
        dataframe_to_markdown(selected_table),
        "",
        "## Best F2 Operating Point for Overview Group",
        "",
        f"- Threshold: `{best_f2['threshold']:.2f}`",
        f"- Precision: `{best_f2['precision']:.3f}`",
        f"- Recall: `{best_f2['recall']:.3f}`",
        f"- F2: `{best_f2['f2']:.3f}`",
        f"- Alerts per 100 cell-years: `{best_f2['alerts_per_100_cell_years']:.1f}`",
        f"- False alerts per 100 cell-years: `{best_f2['false_alerts_per_100_cell_years']:.1f}`",
        f"- Missed declines per 100 true decline cases: `{best_f2['missed_declines_per_100_true_declines']:.1f}`",
        "",
        "## Interpretation",
        "",
        "- Threshold selection reveals a practical trade-off between missed declines and false alerts.",
        "- The curves help distinguish broad risk-state screening from stricter operational early-warning claims.",
        "- The analysis supports decision-threshold selection rather than proving causal early-warning skill.",
        "- Apparent performance should still be interpreted alongside persistence baselines, at-risk subsets, transition labels, and actionable-drop labels.",
        "",
        "## Outputs",
        "",
        "- `outputs/tables/threshold_tradeoff_metrics.csv`",
        "- `outputs/figures/threshold_tradeoff_recall_vs_alerts.png`",
        "- `outputs/figures/threshold_tradeoff_precision_recall.png`",
        "- `outputs/figures/threshold_tradeoff_false_alerts_vs_missed.png`",
        "- `outputs/figures/threshold_tradeoff_f2_vs_threshold.png`",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the threshold trade-off analysis."""
    args = parse_args()
    predictions = load_predictions(args.predictions, args.label_target)
    metrics = build_threshold_metrics(predictions)
    score_summary = model_score_summary(predictions)
    overview_group = choose_overview_group(score_summary)
    overview_metrics = subset_for_group(metrics, overview_group)

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_output, index=False, lineterminator="\n")

    plot_recall_vs_alerts(overview_metrics, overview_group, args.figure_dir / RECALL_ALERTS_FIGURE.name)
    plot_precision_recall(overview_metrics, overview_group, args.figure_dir / PRECISION_RECALL_FIGURE.name)
    plot_false_alerts_vs_missed(overview_metrics, overview_group, args.figure_dir / FALSE_ALERTS_MISSED_FIGURE.name)
    plot_f2_vs_threshold(overview_metrics, overview_group, args.figure_dir / F2_THRESHOLD_FIGURE.name)
    write_summary(args.report_output, args.predictions, predictions, metrics, score_summary, overview_group)

    print("Threshold trade-off analysis complete.")
    print(f"Input predictions: {args.predictions}")
    print(f"Rows evaluated: {len(predictions):,}")
    print(f"Threshold metric rows: {len(metrics):,}")
    print(
        "Overview group: "
        f"{overview_group['label_target']} / {overview_group['feature_set']} / "
        f"{overview_group['model']} / {overview_group['split']}"
    )
    print(f"Wrote metrics: {args.metrics_output}")
    print(f"Wrote report: {args.report_output}")
    print(f"Wrote figures to: {args.figure_dir}")


if __name__ == "__main__":
    main()
