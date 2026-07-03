#!/usr/bin/env python3
"""Evaluate top-k alert-budget prioritisation for kelp decline risk scores.

This script reuses saved out-of-sample prediction probabilities and asks a
decision-support question:

    If managers can inspect only a fixed number of grid cells, how well do
    model risk scores prioritise cells that actually decline?

The analysis is a monitoring-prioritisation diagnostic. It does not prove
causal or fully operational early-warning skill.

Run from the repository root:

    python scripts/31_alert_budget_topk_evaluation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_PREDICTIONS = Path("outputs/metadata/model_comparison_test_predictions.csv")
METRICS_OUTPUT = Path("outputs/tables/alert_budget_topk_metrics.csv")
ANNUAL_SUMMARY_OUTPUT = Path("outputs/tables/alert_budget_topk_annual_summary.csv")
REPORT_OUTPUT = Path("outputs/reports/alert_budget_topk_summary.md")
FIGURE_DIR = Path("outputs/figures")

RECALL_BUDGET_FIGURE = FIGURE_DIR / "topk_recall_vs_budget.png"
PRECISION_BUDGET_FIGURE = FIGURE_DIR / "topk_precision_vs_budget.png"
LIFT_BUDGET_FIGURE = FIGURE_DIR / "topk_lift_vs_budget.png"
CUMULATIVE_GAIN_FIGURE = FIGURE_DIR / "topk_cumulative_gain.png"
HITS_FALSE_ALERTS_FIGURE = FIGURE_DIR / "topk_hits_false_alerts_by_budget.png"

FIXED_ANNUAL_BUDGETS = [1, 3, 5, 10, 15, 20]
FIXED_POOLED_BUDGETS = [5, 10, 20, 50, 100]
PERCENT_BUDGETS = [0.10, 0.20, 0.30, 0.40, 0.50]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate top-k alert-budget prioritisation.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics-output", type=Path, default=METRICS_OUTPUT)
    parser.add_argument("--annual-summary-output", type=Path, default=ANNUAL_SUMMARY_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument(
        "--label-target",
        default="decline_event_next",
        help="Default target name when the prediction file does not include a label target column.",
    )
    return parser.parse_args()


def safe_divide(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, or NaN when denominator is zero."""
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def infer_column(columns: Iterable[str], candidates: list[str], required: bool, description: str) -> str | None:
    """Infer a column name from candidate names."""
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    if required:
        raise ValueError(f"Could not infer {description}. Tried columns: {candidates}")
    return None


def load_predictions(path: Path, default_label_target: str) -> tuple[pd.DataFrame, list[str]]:
    """Load prediction probabilities and normalize expected column names."""
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}. Run `python scripts/train_model_comparison.py` "
            "first to regenerate out-of-sample prediction probabilities."
        )

    raw = pd.read_csv(path)
    notes: list[str] = []
    true_col = infer_column(raw.columns, ["y_true", "label", "target", "actual", "observed"], True, "true label column")
    score_col = infer_column(
        raw.columns,
        ["y_proba", "predicted_probability", "probability", "risk_score", "score", "prediction_score"],
        True,
        "predicted probability/risk score column",
    )
    feature_col = infer_column(raw.columns, ["feature_set", "feature_family"], False, "feature-set column")
    model_col = infer_column(raw.columns, ["model", "model_name"], False, "model column")
    split_col = infer_column(raw.columns, ["split", "evaluation_split", "selection_split"], False, "split column")
    label_col = infer_column(raw.columns, ["label_target", "target_definition", "target_column"], False, "label target column")
    year_col = infer_column(raw.columns, ["year", "eval_year", "prediction_year"], False, "year column")
    cell_col = infer_column(raw.columns, ["cell_id", "grid_id", "site_id", "aoi_id"], False, "grid/cell identifier")

    data = pd.DataFrame(
        {
            "label_target": raw[label_col].astype(str) if label_col else default_label_target,
            "feature_set": raw[feature_col].astype(str) if feature_col else "unknown_feature_set",
            "model": raw[model_col].astype(str) if model_col else "unknown_model",
            "split": raw[split_col].astype(str) if split_col else "out_of_sample",
            "y_true": pd.to_numeric(raw[true_col], errors="coerce"),
            "risk_score": pd.to_numeric(raw[score_col], errors="coerce"),
            "original_row": np.arange(len(raw)),
        }
    )
    if year_col:
        data["year"] = pd.to_numeric(raw[year_col], errors="coerce")
    else:
        data["year"] = np.nan
        notes.append("No year column found; annual top-k evaluation was skipped.")
    if cell_col:
        data["cell_id"] = raw[cell_col].astype(str)
    else:
        data["cell_id"] = ""
        notes.append("No grid/cell identifier found; deterministic tie-breaking uses original row order.")
    if not label_col:
        notes.append(f"No label target column found; assigned `{default_label_target}` to all rows.")
    if not feature_col:
        notes.append("No feature-set column found; assigned `unknown_feature_set`.")
    if not model_col:
        notes.append("No model column found; assigned `unknown_model`.")
    if not split_col:
        notes.append("No split column found; assigned `out_of_sample`.")

    before = len(data)
    data = data.dropna(subset=["y_true", "risk_score"]).copy()
    if len(data) < before:
        notes.append(f"Dropped {before - len(data)} rows with missing labels or risk scores.")

    data["y_true"] = data["y_true"].astype(int)
    if not data["y_true"].isin([0, 1]).all():
        raise ValueError("True labels must be binary 0/1 after parsing.")
    if data.empty:
        raise ValueError(f"No usable prediction rows found in {path}.")
    return data, notes


def budget_to_k(n_candidates: int, budget_type: str, budget_value: float) -> int:
    """Convert fixed or percentage budget to a capped top-k count."""
    if n_candidates <= 0:
        return 0
    if budget_type == "fixed_k":
        k = int(budget_value)
    elif budget_type == "percent":
        k = int(np.ceil(n_candidates * float(budget_value)))
    else:
        raise ValueError(f"Unknown budget type: {budget_type}")
    return max(1, min(k, n_candidates))


def sort_by_risk(data: pd.DataFrame) -> pd.DataFrame:
    """Sort risk scores descending with deterministic tie-breaking."""
    sort_cols = ["risk_score"]
    ascending = [False]
    if "year" in data.columns:
        sort_cols.append("year")
        ascending.append(True)
    if "cell_id" in data.columns:
        sort_cols.append("cell_id")
        ascending.append(True)
    sort_cols.append("original_row")
    ascending.append(True)
    return data.sort_values(sort_cols, ascending=ascending, kind="mergesort")


def topk_metrics(data: pd.DataFrame, k_selected: int) -> dict[str, float | int]:
    """Calculate top-k prioritisation metrics."""
    n_candidates = int(len(data))
    n_positive = int(data["y_true"].sum())
    base_prevalence = safe_divide(n_positive, n_candidates)
    selected = sort_by_risk(data).head(k_selected)
    hits = int(selected["y_true"].sum())
    false_alerts = int(k_selected - hits)
    missed_declines = int(n_positive - hits)
    precision_at_k = safe_divide(hits, k_selected)
    recall_at_k = safe_divide(hits, n_positive)
    false_discovery_rate_at_k = safe_divide(false_alerts, k_selected)
    lift_at_k = safe_divide(precision_at_k, base_prevalence) if not np.isnan(base_prevalence) else float("nan")
    expected_random_hits = k_selected * base_prevalence if not np.isnan(base_prevalence) else float("nan")
    enrichment_over_random = safe_divide(hits, expected_random_hits) if not np.isnan(expected_random_hits) else float("nan")
    return {
        "k_selected": int(k_selected),
        "n_candidates": n_candidates,
        "n_positive": n_positive,
        "base_prevalence": base_prevalence,
        "hits": hits,
        "false_alerts": false_alerts,
        "missed_declines": missed_declines,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "false_discovery_rate_at_k": false_discovery_rate_at_k,
        "lift_at_k": lift_at_k,
        "expected_random_hits": expected_random_hits,
        "enrichment_over_random": enrichment_over_random,
    }


def budget_rows(budgets: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Return budget rows preserving requested order."""
    return budgets


def evaluate_annual_topk(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate top-k budgets separately within each year."""
    if predictions["year"].isna().all():
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["label_target", "feature_set", "model", "split"]
    budgets = budget_rows(
        [("fixed_k", float(k)) for k in FIXED_ANNUAL_BUDGETS]
        + [("percent", float(pct)) for pct in PERCENT_BUDGETS]
    )
    for keys, group in predictions.groupby(group_cols, sort=True):
        label_target, feature_set, model, split = keys
        for year, year_group in group.dropna(subset=["year"]).groupby("year", sort=True):
            year_group = year_group.copy()
            for budget_type, budget_value in budgets:
                k_selected = budget_to_k(len(year_group), budget_type, budget_value)
                metrics = topk_metrics(year_group, k_selected)
                rows.append(
                    {
                        "label_target": label_target,
                        "feature_set": feature_set,
                        "model": model,
                        "split": split,
                        "evaluation_type": "annual_topk",
                        "year": int(year),
                        "budget_type": budget_type,
                        "budget_value": budget_value,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def evaluate_pooled_topk(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate top-k budgets across the full split."""
    rows: list[dict[str, object]] = []
    group_cols = ["label_target", "feature_set", "model", "split"]
    budgets = budget_rows(
        [("fixed_k", float(k)) for k in FIXED_POOLED_BUDGETS]
        + [("percent", float(pct)) for pct in PERCENT_BUDGETS]
    )
    for keys, group in predictions.groupby(group_cols, sort=True):
        label_target, feature_set, model, split = keys
        for budget_type, budget_value in budgets:
            k_selected = budget_to_k(len(group), budget_type, budget_value)
            metrics = topk_metrics(group, k_selected)
            rows.append(
                {
                    "label_target": label_target,
                    "feature_set": feature_set,
                    "model": model,
                    "split": split,
                    "evaluation_type": "pooled_topk",
                    "year": np.nan,
                    "budget_type": budget_type,
                    "budget_value": budget_value,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summarize_annual_topk(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate annual top-k metrics across years."""
    annual = metrics.loc[metrics["evaluation_type"].eq("annual_topk")].copy()
    if annual.empty:
        return pd.DataFrame()
    group_cols = ["label_target", "feature_set", "model", "split", "budget_type", "budget_value"]
    summary = (
        annual.groupby(group_cols, dropna=False)
        .agg(
            mean_precision_at_k=("precision_at_k", "mean"),
            mean_recall_at_k=("recall_at_k", "mean"),
            mean_lift_at_k=("lift_at_k", "mean"),
            mean_hits=("hits", "mean"),
            mean_false_alerts=("false_alerts", "mean"),
            mean_missed_declines=("missed_declines", "mean"),
            total_hits=("hits", "sum"),
            total_false_alerts=("false_alerts", "sum"),
            total_missed_declines=("missed_declines", "sum"),
            n_years=("year", "nunique"),
        )
        .reset_index()
    )
    return summary.sort_values(group_cols).reset_index(drop=True)


def prediction_score_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute ranking metrics used to select an overview group."""
    rows: list[dict[str, object]] = []
    group_cols = ["label_target", "feature_set", "model", "split"]
    for keys, group in predictions.groupby(group_cols, sort=True):
        y_true = group["y_true"].to_numpy(dtype=int)
        scores = group["risk_score"].to_numpy(dtype=float)
        if len(np.unique(y_true)) == 2:
            pr_auc = average_precision_score(y_true, scores)
            roc_auc = roc_auc_score(y_true, scores)
        else:
            pr_auc = float("nan")
            roc_auc = float("nan")
        rows.append(
            {
                "label_target": keys[0],
                "feature_set": keys[1],
                "model": keys[2],
                "split": keys[3],
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "n_obs": len(group),
                "n_positive": int(group["y_true"].sum()),
            }
        )
    return pd.DataFrame(rows)


def choose_overview_group(score_summary: pd.DataFrame) -> dict[str, str]:
    """Select a main overview group, preferring the existing threshold report group."""
    preferred = score_summary.loc[
        score_summary["label_target"].eq("decline_event_next")
        & score_summary["feature_set"].eq("canopy_only")
        & score_summary["model"].eq("Random Forest")
        & score_summary["split"].eq("test")
    ]
    if not preferred.empty:
        row = preferred.iloc[0]
    else:
        row = score_summary.sort_values(["pr_auc", "roc_auc", "n_positive"], ascending=False).iloc[0]
    return {key: str(row[key]) for key in ["label_target", "feature_set", "model", "split"]}


def filter_group(data: pd.DataFrame, group: dict[str, str]) -> pd.DataFrame:
    """Filter rows to one label/feature/model/split group."""
    mask = (
        data["label_target"].eq(group["label_target"])
        & data["feature_set"].eq(group["feature_set"])
        & data["model"].eq(group["model"])
        & data["split"].eq(group["split"])
    )
    subset = data.loc[mask].copy()
    if subset.empty:
        raise ValueError(f"No rows found for overview group: {group}")
    return subset


def setup_axis(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    """Apply a restrained report-friendly plotting style."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#E6E8F0", linewidth=0.8)
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D7DBE7")
    ax.spines["bottom"].set_color("#D7DBE7")


def annual_fixed_summary_for_group(annual_summary: pd.DataFrame, group: dict[str, str]) -> pd.DataFrame:
    """Return fixed-k annual summary rows for the overview group."""
    subset = filter_group(annual_summary, group)
    subset = subset.loc[subset["budget_type"].eq("fixed_k")].sort_values("budget_value")
    return subset


def plot_annual_metric(
    annual_summary: pd.DataFrame,
    group: dict[str, str],
    metric: str,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    """Plot an annual top-k summary metric against fixed alert budget."""
    data = annual_fixed_summary_for_group(annual_summary, group)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(data["budget_value"], data[metric], marker="o", color="#2E4780", linewidth=2)
    setup_axis(ax, "Annual alert budget: top k cells per year", ylabel)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_xticks(data["budget_value"])
    valid_values = data[metric].dropna()
    if metric in {"mean_recall_at_k", "mean_precision_at_k"}:
        ax.set_ylim(0, 1.05)
    elif not valid_values.empty:
        ax.set_ylim(0, max(1.0, float(valid_values.max())) * 1.12)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cumulative_gain(predictions: pd.DataFrame, group: dict[str, str], output: Path) -> None:
    """Plot cumulative gain for pooled ranked observations."""
    data = sort_by_risk(filter_group(predictions, group)).reset_index(drop=True)
    total_positive = data["y_true"].sum()
    n_obs = len(data)
    data["rank"] = np.arange(1, n_obs + 1)
    data["fraction_reviewed"] = data["rank"] / n_obs
    data["cumulative_recall"] = data["y_true"].cumsum() / total_positive if total_positive else np.nan

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(data["fraction_reviewed"], data["cumulative_recall"], color="#2E4780", linewidth=2, label="Model ranking")
    ax.plot([0, 1], [0, 1], color="#7A828F", linestyle="--", linewidth=1, label="Random ordering")
    setup_axis(ax, "Fraction of grid-year observations selected", "Fraction of true declines captured")
    ax.set_title("Cumulative gain for pooled ranking", loc="left", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hits_false_alerts(annual_summary: pd.DataFrame, group: dict[str, str], output: Path) -> None:
    """Plot mean hits and false alerts by annual fixed-k budget."""
    data = annual_fixed_summary_for_group(annual_summary, group)
    x = np.arange(len(data))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.bar(x - width / 2, data["mean_hits"], width=width, color="#2E4780", label="Mean hits")
    ax.bar(x + width / 2, data["mean_false_alerts"], width=width, color="#A3BEFA", edgecolor="#5477C4", label="Mean false alerts")
    ax.set_xticks(x)
    ax.set_xticklabels(data["budget_value"].astype(int))
    setup_axis(ax, "Annual alert budget: top k cells per year", "Mean cells per year")
    ax.set_title("Hits and false alerts by annual budget", loc="left", fontsize=12, fontweight="bold")
    y_max = max(float(data["mean_hits"].max()), float(data["mean_false_alerts"].max()), 1.0)
    ax.set_ylim(0, y_max * 1.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(data: pd.DataFrame) -> str:
    """Render a compact dataframe as GitHub-flavored Markdown."""
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


def compact_annual_table(annual_summary: pd.DataFrame, group: dict[str, str]) -> pd.DataFrame:
    """Build compact annual selected-budget table for the report."""
    data = annual_fixed_summary_for_group(annual_summary, group)
    data = data.loc[data["budget_value"].isin([1, 3, 5, 10, 20])].copy()
    return data[
        [
            "budget_value",
            "mean_precision_at_k",
            "mean_recall_at_k",
            "mean_lift_at_k",
            "mean_hits",
            "mean_false_alerts",
            "mean_missed_declines",
            "total_hits",
            "total_false_alerts",
            "n_years",
        ]
    ].rename(columns={"budget_value": "annual_top_k"})


def compact_pooled_table(metrics: pd.DataFrame, group: dict[str, str]) -> pd.DataFrame:
    """Build compact pooled selected-budget table for the report."""
    data = filter_group(metrics, group)
    data = data.loc[
        data["evaluation_type"].eq("pooled_topk")
        & data["budget_type"].eq("fixed_k")
        & data["budget_value"].isin([10, 20, 50, 100])
    ].copy()
    return data[
        [
            "budget_value",
            "precision_at_k",
            "recall_at_k",
            "lift_at_k",
            "hits",
            "false_alerts",
            "missed_declines",
            "expected_random_hits",
            "enrichment_over_random",
        ]
    ].rename(columns={"budget_value": "pooled_top_k"})


def write_report(
    output: Path,
    predictions_path: Path,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    annual_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
    overview_group: dict[str, str],
    notes: list[str],
) -> None:
    """Write a Markdown summary report."""
    labels = sorted(predictions["label_target"].unique())
    feature_sets = sorted(predictions["feature_set"].unique())
    models = sorted(predictions["model"].unique())
    splits = sorted(predictions["split"].unique())
    score_row = filter_group(score_summary, overview_group).iloc[0]
    annual_table = compact_annual_table(annual_summary, overview_group) if not annual_summary.empty else pd.DataFrame()
    pooled_table = compact_pooled_table(metrics, overview_group)

    lines = [
        "# Alert-Budget Top-k Prioritisation Summary",
        "",
        "## Purpose",
        "",
        "Top-k evaluation measures whether model risk scores can prioritise follow-up monitoring under fixed alert budgets. It helps distinguish broad risk prioritisation from stricter operational early-warning claims.",
        "",
        "## Inputs",
        "",
        f"- Prediction file: `{predictions_path}`",
        f"- Prediction rows evaluated: `{len(predictions):,}`",
        f"- Label targets evaluated: `{', '.join(labels)}`",
        f"- Feature sets evaluated: `{', '.join(feature_sets)}`",
        f"- Models evaluated: `{', '.join(models)}`",
        f"- Splits evaluated: `{', '.join(splits)}`",
        "",
        "## Budget Definitions",
        "",
        f"- Annual fixed-k budgets: `{', '.join(map(str, FIXED_ANNUAL_BUDGETS))}` cells per year.",
        f"- Pooled fixed-k budgets: `{', '.join(map(str, FIXED_POOLED_BUDGETS))}` grid-year observations.",
        "- Percent budgets: top `10%`, `20%`, `30%`, `40%`, and `50%` of available candidates.",
        "- Percentage budgets are converted to integer k using ceiling and capped at the number of candidates.",
        "",
        "## Main Figure Group",
        "",
        f"- Label target: `{overview_group['label_target']}`",
        f"- Feature set: `{overview_group['feature_set']}`",
        f"- Model: `{overview_group['model']}`",
        f"- Split: `{overview_group['split']}`",
        f"- PR-AUC: `{score_row['pr_auc']:.3f}`",
        f"- ROC-AUC: `{score_row['roc_auc']:.3f}`",
        "",
        "## Selected Annual Budgets",
        "",
        dataframe_to_markdown(annual_table) if not annual_table.empty else "Annual top-k evaluation was skipped because no year column was available.",
        "",
        "## Selected Pooled Budgets",
        "",
        dataframe_to_markdown(pooled_table),
        "",
        "## Interpretation",
        "",
        "- Fixed alert budgets evaluate prioritisation rather than binary classification.",
        "- Small budgets may have high precision but low recall.",
        "- Larger budgets capture more true declines but create more false alerts.",
        "- Lift@k compares model prioritisation against random selection at the same budget.",
        "- Results should be interpreted alongside persistence baselines, transition labels, and threshold trade-off curves.",
        "- Top-k performance does not prove causal or fully operational early-warning skill.",
        "",
        "## Outputs",
        "",
        "- `outputs/tables/alert_budget_topk_metrics.csv`",
        "- `outputs/tables/alert_budget_topk_annual_summary.csv`",
        "- `outputs/figures/topk_recall_vs_budget.png`",
        "- `outputs/figures/topk_precision_vs_budget.png`",
        "- `outputs/figures/topk_lift_vs_budget.png`",
        "- `outputs/figures/topk_cumulative_gain.png`",
        "- `outputs/figures/topk_hits_false_alerts_by_budget.png`",
    ]
    if notes:
        lines.extend(["", "## Assumptions and Warnings", ""])
        lines.extend([f"- {note}" for note in notes])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the alert-budget top-k evaluation."""
    args = parse_args()
    predictions, notes = load_predictions(args.predictions, args.label_target)
    annual = evaluate_annual_topk(predictions)
    pooled = evaluate_pooled_topk(predictions)
    metrics = pd.concat([df for df in [annual, pooled] if not df.empty], ignore_index=True)
    if metrics.empty:
        raise ValueError("No top-k metrics were generated.")
    annual_summary = summarize_annual_topk(metrics)
    score_summary = prediction_score_summary(predictions)
    overview_group = choose_overview_group(score_summary)

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.annual_summary_output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.metrics_output, index=False, lineterminator="\n")
    annual_summary.to_csv(args.annual_summary_output, index=False, lineterminator="\n")

    if not annual_summary.empty:
        plot_annual_metric(
            annual_summary,
            overview_group,
            "mean_recall_at_k",
            "Mean annual Recall@k",
            "Recall@k versus annual alert budget",
            args.figure_dir / RECALL_BUDGET_FIGURE.name,
        )
        plot_annual_metric(
            annual_summary,
            overview_group,
            "mean_precision_at_k",
            "Mean annual Precision@k",
            "Precision@k versus annual alert budget",
            args.figure_dir / PRECISION_BUDGET_FIGURE.name,
        )
        plot_annual_metric(
            annual_summary,
            overview_group,
            "mean_lift_at_k",
            "Mean annual Lift@k",
            "Lift@k versus annual alert budget",
            args.figure_dir / LIFT_BUDGET_FIGURE.name,
        )
        plot_hits_false_alerts(annual_summary, overview_group, args.figure_dir / HITS_FALSE_ALERTS_FIGURE.name)
    else:
        notes.append("Annual top-k figures were skipped because annual evaluation was unavailable.")
    plot_cumulative_gain(predictions, overview_group, args.figure_dir / CUMULATIVE_GAIN_FIGURE.name)
    write_report(
        args.report_output,
        args.predictions,
        predictions,
        metrics,
        annual_summary,
        score_summary,
        overview_group,
        notes,
    )

    print("Alert-budget top-k evaluation complete.")
    print(f"Input predictions: {args.predictions}")
    print(f"Prediction rows evaluated: {len(predictions):,}")
    print(f"Top-k metric rows: {len(metrics):,}")
    print(f"Annual summary rows: {len(annual_summary):,}")
    print(
        "Overview group: "
        f"{overview_group['label_target']} / {overview_group['feature_set']} / "
        f"{overview_group['model']} / {overview_group['split']}"
    )
    if notes:
        print("Assumptions/warnings:")
        for note in notes:
            print(f"  - {note}")
    print(f"Wrote metrics: {args.metrics_output}")
    print(f"Wrote annual summary: {args.annual_summary_output}")
    print(f"Wrote report: {args.report_output}")
    print(f"Wrote figures to: {args.figure_dir}")


if __name__ == "__main__":
    main()
