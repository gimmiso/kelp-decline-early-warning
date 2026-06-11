"""Diagnose initial model comparison results for kelp decline prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = Path("outputs/metadata/model_comparison_results.csv")
TEST_METRICS = Path("outputs/metadata/model_comparison_test_metrics.csv")
PREDICTIONS = Path("outputs/metadata/model_comparison_test_predictions.csv")
DATASET = Path("data/processed/modeling_dataset_ge500_noaa_v1.csv")

FEATURE_SET_SUMMARY = Path("outputs/metadata/model_diagnostics_feature_set_summary.csv")
SAME_MODEL_COMPARISON = Path("outputs/metadata/model_diagnostics_same_model_comparison.csv")
FALSE_NEGATIVES = Path("outputs/metadata/model_diagnostics_false_negatives.csv")
FALSE_NEGATIVE_SUMMARY = Path("outputs/metadata/model_diagnostics_false_negative_summary.csv")
FEATURE_AUDIT = Path("outputs/metadata/model_diagnostics_feature_audit.csv")
TEMPORAL_SUMMARY = Path("outputs/metadata/model_diagnostics_temporal_summary.csv")
ENV_SIGNAL = Path("outputs/metadata/model_diagnostics_environmental_signal.csv")
REPORT = Path("outputs/metadata/model_diagnostics_report.md")

FIG_PR_AUC = Path("outputs/figures/model_diagnostics_feature_set_pr_auc.png")
FIG_RECALL = Path("outputs/figures/model_diagnostics_feature_set_recall.png")
FIG_FN_YEAR = Path("outputs/figures/model_diagnostics_false_negatives_by_year.png")
FIG_ENV = Path("outputs/figures/model_diagnostics_environmental_signal.png")
NOTEBOOK = Path("notebooks/05_model_diagnostics.ipynb")

LEAKAGE_VARIABLES = {
    "decline_event_next",
    "next_year_kelp_area_m2",
    "next_year_relative_canopy",
    "decline_event_next_p25_full",
    "decline_50pct_next",
    "relative_canopy_change_next",
    "relative_canopy_pct_change_next",
    "baseline_p25_relative_canopy_1984_2013",
    "p25_relative_canopy_full_history",
}
CANOPY_FEATURES = [
    "relative_canopy",
    "kelp_area_m2",
    "count_cells_kelp",
    "count_cells_no_clouds",
    "count_cells_historic_footprint",
    "historical_footprint_area_m2",
    "lag1_relative_canopy",
    "relative_canopy_change_lag1",
]
OISST_FEATURES = [
    "annual_mean_sst",
    "annual_max_sst",
    "annual_min_sst",
    "annual_sst_std",
    "annual_mean_sst_anomaly",
    "annual_max_sst_anomaly",
    "hot_days_p90",
    "hot_days_p95",
    "lag1_annual_mean_sst_anomaly",
    "lag1_hot_days_p90",
]
CUTI_BEUTI_FEATURES = [
    "annual_mean_cuti",
    "spring_mean_cuti",
    "summer_mean_cuti",
    "cuti_anomaly",
    "lag1_cuti_anomaly",
    "annual_mean_beuti",
    "spring_mean_beuti",
    "summer_mean_beuti",
    "beuti_anomaly",
    "lag1_beuti_anomaly",
]
ENVIRONMENTAL_SIGNAL_FEATURES = [
    "annual_mean_sst_anomaly",
    "annual_max_sst_anomaly",
    "hot_days_p90",
    "hot_days_p95",
    "annual_mean_cuti",
    "cuti_anomaly",
    "annual_mean_beuti",
    "beuti_anomaly",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Diagnose model comparison results.")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--test-metrics", type=Path, default=TEST_METRICS)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    return parser.parse_args()


def require(path: Path) -> None:
    """Raise if a required file does not exist."""
    if not path.exists():
        raise FileNotFoundError(path)


def feature_set_summary(test: pd.DataFrame) -> pd.DataFrame:
    """Summarize performance by feature set."""
    rows = []
    for feature_set, group in test.groupby("feature_set"):
        rows.append(
            {
                "feature_set": feature_set,
                "best_model_by_pr_auc": group.sort_values("pr_auc", ascending=False).iloc[0]["model"],
                "best_pr_auc": group["pr_auc"].max(),
                "best_model_by_recall": group.sort_values("recall", ascending=False).iloc[0]["model"],
                "best_recall": group["recall"].max(),
                "best_model_by_f1": group.sort_values("f1", ascending=False).iloc[0]["model"],
                "best_f1": group["f1"].max(),
                "mean_test_pr_auc": group["pr_auc"].mean(),
                "mean_test_recall": group["recall"].mean(),
                "mean_test_f1": group["f1"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("best_pr_auc", ascending=False)


def same_model_comparison(test: pd.DataFrame) -> pd.DataFrame:
    """Compare each model across feature sets."""
    rows = []
    for model, group in test.groupby("model"):
        wide = group.set_index("feature_set")
        for feature_set in ["canopy_only", "oisst_only", "canopy_noaa"]:
            row = wide.loc[feature_set]
            rows.append(
                {
                    "model": model,
                    "feature_set": feature_set,
                    "test_pr_auc": row["pr_auc"],
                    "test_recall": row["recall"],
                    "test_f1": row["f1"],
                    "test_precision": row["precision"],
                    "false_negatives": int(row["fn"]),
                    "false_positives": int(row["fp"]),
                    "delta_pr_auc_vs_canopy_only": row["pr_auc"] - wide.loc["canopy_only", "pr_auc"],
                    "delta_recall_vs_canopy_only": row["recall"] - wide.loc["canopy_only", "recall"],
                    "delta_f1_vs_canopy_only": row["f1"] - wide.loc["canopy_only", "f1"],
                    "delta_false_negatives_vs_canopy_only": int(row["fn"] - wide.loc["canopy_only", "fn"]),
                    "delta_false_positives_vs_canopy_only": int(row["fp"] - wide.loc["canopy_only", "fp"]),
                }
            )
    return pd.DataFrame(rows)


def best_model(test: pd.DataFrame, feature_set: str) -> str:
    """Return the best model by PR-AUC for one feature set."""
    return test.loc[test["feature_set"] == feature_set].sort_values(
        ["pr_auc", "recall", "f1"], ascending=False
    ).iloc[0]["model"]


def false_negative_tables(predictions: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create false-negative detail and summary tables."""
    best_canopy = best_model(test, "canopy_only")
    best_noaa = best_model(test, "canopy_noaa")
    selected = predictions.loc[
        ((predictions["feature_set"] == "canopy_only") & (predictions["model"] == best_canopy))
        | ((predictions["feature_set"] == "canopy_noaa") & (predictions["model"] == best_noaa))
    ].copy()
    selected["is_false_negative"] = (selected["y_true"] == 1) & (selected["y_pred"] == 0)

    fn = selected.loc[selected["is_false_negative"]].copy()
    missed_by_canopy = set(
        selected.loc[
            (selected["feature_set"] == "canopy_only")
            & (selected["model"] == best_canopy)
            & selected["is_false_negative"],
            ["cell_id", "year"],
        ].itertuples(index=False, name=None)
    )
    missed_by_noaa = set(
        selected.loc[
            (selected["feature_set"] == "canopy_noaa")
            & (selected["model"] == best_noaa)
            & selected["is_false_negative"],
            ["cell_id", "year"],
        ].itertuples(index=False, name=None)
    )
    caught_by_noaa = missed_by_canopy - missed_by_noaa
    caught_by_canopy = missed_by_noaa - missed_by_canopy

    summary_rows = []
    for (feature_set, model), group in selected.groupby(["feature_set", "model"]):
        fn_group = group.loc[group["is_false_negative"]]
        by_year = {int(key): int(value) for key, value in fn_group["year"].value_counts().sort_index().items()}
        by_region = {str(key): int(value) for key, value in fn_group["region_group"].value_counts().sort_index().items()}
        by_cell = {str(key): int(value) for key, value in fn_group["cell_id"].value_counts().sort_index().items()}
        summary_rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "false_negative_count": len(fn_group),
                "false_negatives_by_year": by_year,
                "false_negatives_by_region": by_region,
                "false_negatives_by_cell_id": by_cell,
                "decline_events_caught_vs_other_selected_model": len(
                    caught_by_noaa if feature_set == "canopy_noaa" else caught_by_canopy
                ),
            }
        )
    return fn, pd.DataFrame(summary_rows)


def feature_audit() -> pd.DataFrame:
    """Audit features used and leakage status."""
    rows = []
    feature_sets = {
        "canopy_only": CANOPY_FEATURES,
        "oisst_only": OISST_FEATURES,
        "canopy_noaa": CANOPY_FEATURES + OISST_FEATURES + CUTI_BEUTI_FEATURES + ["region_group", "center_lat", "center_lon"],
    }
    for feature_set, features in feature_sets.items():
        for feature in features:
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature": feature,
                    "is_leakage_variable": feature in LEAKAGE_VARIABLES,
                    "interpretation": (
                        "leakage: invalid if used"
                        if feature in LEAKAGE_VARIABLES
                        else "allowed current-year or lagged predictor"
                    ),
                }
            )
    return pd.DataFrame(rows)


def temporal_summary(predictions: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Summarize actual and predicted declines by test year."""
    best_canopy = best_model(test, "canopy_only")
    best_noaa = best_model(test, "canopy_noaa")
    selected = predictions.loc[
        ((predictions["feature_set"] == "canopy_only") & (predictions["model"] == best_canopy))
        | ((predictions["feature_set"] == "canopy_noaa") & (predictions["model"] == best_noaa))
    ].copy()
    return (
        selected.groupby(["feature_set", "model", "year"])
        .agg(
            actual_decline_count=("y_true", "sum"),
            predicted_decline_count=("y_pred", "sum"),
            mean_predicted_risk=("y_proba", "mean"),
            n_rows=("y_true", "size"),
        )
        .reset_index()
    )


def environmental_signal(data: pd.DataFrame) -> pd.DataFrame:
    """Compare NOAA feature distributions by decline outcome."""
    subset = data.loc[data["year"].between(1989, 2024)].copy()
    rows = []
    for feature in ENVIRONMENTAL_SIGNAL_FEATURES:
        decline = subset.loc[subset["decline_event_next"] == 1, feature].dropna()
        non_decline = subset.loc[subset["decline_event_next"] == 0, feature].dropna()
        pooled = subset[feature].dropna()
        pooled_std = pooled.std()
        diff = decline.mean() - non_decline.mean()
        rows.append(
            {
                "feature": feature,
                "mean_decline_event_1": decline.mean(),
                "mean_decline_event_0": non_decline.mean(),
                "difference_1_minus_0": diff,
                "standardized_difference": diff / pooled_std if pooled_std and not np.isnan(pooled_std) else np.nan,
                "n_decline_event_1": len(decline),
                "n_decline_event_0": len(non_decline),
            }
        )
    return pd.DataFrame(rows)


def plot_feature_set_bars(summary: pd.DataFrame) -> None:
    """Create feature-set PR-AUC and recall summary figures."""
    FIG_PR_AUC.parent.mkdir(parents=True, exist_ok=True)
    for metric, path, title in [
        ("mean_test_pr_auc", FIG_PR_AUC, "Mean Test PR-AUC by Feature Set"),
        ("mean_test_recall", FIG_RECALL, "Mean Test Recall by Feature Set"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        ordered = summary.sort_values(metric, ascending=False)
        ax.bar(ordered["feature_set"], ordered[metric], color="#377eb8")
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)


def plot_false_negatives(fn: pd.DataFrame) -> None:
    """Plot false negatives by year."""
    FIG_FN_YEAR.parent.mkdir(parents=True, exist_ok=True)
    counts = fn.groupby(["feature_set", "model", "year"]).size().reset_index(name="false_negatives")
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, group in counts.groupby(["feature_set", "model"]):
        ax.plot(group["year"], group["false_negatives"], marker="o", label=f"{label[0]} / {label[1]}")
    ax.set_ylabel("False negatives")
    ax.set_title("False Negatives by Test Year")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_FN_YEAR, dpi=200)
    plt.close(fig)


def plot_environmental_signal(signal: pd.DataFrame) -> None:
    """Plot standardized environmental signal differences."""
    FIG_ENV.parent.mkdir(parents=True, exist_ok=True)
    ordered = signal.sort_values("standardized_difference")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered["feature"], ordered["standardized_difference"], color="#4daf4a")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Standardized mean difference (decline - non-decline)")
    ax.set_title("NOAA Environmental Signal by Decline Outcome")
    fig.tight_layout()
    fig.savefig(FIG_ENV, dpi=200)
    plt.close(fig)


def write_report(
    test: pd.DataFrame,
    summary: pd.DataFrame,
    same_model: pd.DataFrame,
    audit: pd.DataFrame,
    signal: pd.DataFrame,
) -> None:
    """Write the main diagnostics report."""
    best_pr = test.sort_values(["pr_auc", "recall", "f1"], ascending=False).iloc[0]
    best_canopy_pr = test.loc[test["feature_set"] == "canopy_only", "pr_auc"].max()
    best_noaa_pr = test.loc[test["feature_set"] == "canopy_noaa", "pr_auc"].max()
    leakage_found = audit["is_leakage_variable"].any()
    noaa_improvements = same_model.loc[
        (same_model["feature_set"] == "canopy_noaa")
        & (
            (same_model["delta_recall_vs_canopy_only"] > 0)
            | (same_model["delta_f1_vs_canopy_only"] > 0)
            | (same_model["delta_false_negatives_vs_canopy_only"] < 0)
        )
    ]
    strongest_env = signal.reindex(signal["standardized_difference"].abs().sort_values(ascending=False).index).head(3)

    lines = [
        "# Model Comparison Diagnostics Report",
        "",
        "## Initial Model Comparison Summary",
        "",
        f"- Best overall test PR-AUC: {best_pr['feature_set']} / {best_pr['model']} ({best_pr['pr_auc']:.4f})",
        f"- Best canopy-only test PR-AUC: {best_canopy_pr:.4f}",
        f"- Best canopy+NOAA test PR-AUC: {best_noaa_pr:.4f}",
        f"- Canopy-only outperformed canopy+NOAA by best test PR-AUC: {best_canopy_pr > best_noaa_pr}",
        "",
        "## Same-Model NOAA Improvements",
        "",
    ]
    if noaa_improvements.empty:
        lines.append("- No same-model comparison showed canopy+NOAA improving recall, F1, or false negatives relative to canopy-only at the default 0.5 threshold.")
    else:
        for row in noaa_improvements.to_dict("records"):
            lines.append(
                f"- {row['model']}: delta recall={row['delta_recall_vs_canopy_only']:.4f}, "
                f"delta F1={row['delta_f1_vs_canopy_only']:.4f}, "
                f"delta false negatives={int(row['delta_false_negatives_vs_canopy_only'])}"
            )
    lines.extend(
        [
            "",
            "## Leakage Audit",
            "",
            f"- Leakage variables included in feature sets: {leakage_found}",
            "- Canopy-only features are current-year canopy/status variables plus lagged canopy features.",
            "- If only current-year and lagged canopy variables are used, strong canopy-only performance is likely due to temporal persistence and autocorrelation in kelp canopy condition, not target leakage.",
            "",
            "## Why Canopy-Only May Be Strong",
            "",
            "- Current canopy state is temporally persistent.",
            "- The decline label is defined from next-year canopy condition.",
            "- Canopy history is a direct biological response signal, while NOAA features are environmental exposure proxies.",
            "",
            "## Environmental Signal Interpretation",
            "",
        ]
    )
    for row in strongest_env.to_dict("records"):
        direction = "higher" if row["difference_1_minus_0"] > 0 else "lower"
        lines.append(
            f"- `{row['feature']}` is {direction} in decline rows "
            f"(standardized difference={row['standardized_difference']:.3f})."
        )
    lines.extend(
        [
            "",
            "NOAA variables still provide environmental context for interpretation, even when they do not outperform direct canopy observations in test PR-AUC. They should be compared through SHAP explanations rather than judged only as replacements for canopy variables.",
            "",
            "## Recommended Next Steps",
            "",
            "- Run SHAP for the best canopy-only model.",
            "- Run SHAP for the best canopy+NOAA model.",
            "- Compare explanations to determine whether NOAA variables clarify environmental exposure patterns even when predictive improvement is limited.",
            "",
            "## Limitations",
            "",
            "- Small number of cells.",
            "- Limited test years.",
            "- OISST nearest valid grid Version 1.",
            "- CUTI/BEUTI latitude-bin proxy.",
            "- No direct grazing or urchin variables.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n")


def write_notebook() -> None:
    """Write a lightweight diagnostics notebook."""
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(
        """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# Model Diagnostics\\n", "\\n", "Run the model comparison diagnostics workflow."]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["!python ../scripts/diagnose_model_results.py"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["import pandas as pd\\n", "pd.read_csv('../outputs/metadata/model_diagnostics_feature_set_summary.csv')"]
  }
 ],
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3"}
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
""",
        encoding="utf-8",
    )


def main() -> None:
    """Run diagnostics."""
    args = parse_args()
    for path in [args.results, args.test_metrics, args.predictions, args.dataset]:
        require(path)

    results = pd.read_csv(args.results)
    test = pd.read_csv(args.test_metrics)
    predictions = pd.read_csv(args.predictions)
    data = pd.read_csv(args.dataset)

    summary = feature_set_summary(test)
    same_model = same_model_comparison(test)
    fn, fn_summary = false_negative_tables(predictions, test)
    audit = feature_audit()
    temporal = temporal_summary(predictions, test)
    signal = environmental_signal(data)

    for path in [
        FEATURE_SET_SUMMARY,
        SAME_MODEL_COMPARISON,
        FALSE_NEGATIVES,
        FALSE_NEGATIVE_SUMMARY,
        FEATURE_AUDIT,
        TEMPORAL_SUMMARY,
        ENV_SIGNAL,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(FEATURE_SET_SUMMARY, index=False)
    same_model.to_csv(SAME_MODEL_COMPARISON, index=False)
    fn.to_csv(FALSE_NEGATIVES, index=False)
    fn_summary.to_csv(FALSE_NEGATIVE_SUMMARY, index=False)
    audit.to_csv(FEATURE_AUDIT, index=False)
    temporal.to_csv(TEMPORAL_SUMMARY, index=False)
    signal.to_csv(ENV_SIGNAL, index=False)

    plot_feature_set_bars(summary)
    plot_false_negatives(fn)
    plot_environmental_signal(signal)
    write_report(test, summary, same_model, audit, signal)
    write_notebook()

    print("Model diagnostics complete.")
    print(f"Best canopy-only model: {best_model(test, 'canopy_only')}")
    print(f"Best canopy+NOAA model: {best_model(test, 'canopy_noaa')}")
    print(f"Leakage variables included: {audit['is_leakage_variable'].any()}")


if __name__ == "__main__":
    main()
