"""Analyze canopy persistence and NOAA environmental context for kelp decline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET = Path("data/processed/modeling_dataset_ge500_noaa_v1.csv")
TEST_METRICS = Path("outputs/metadata/model_comparison_test_metrics.csv")
PREDICTIONS = Path("outputs/metadata/model_comparison_test_predictions.csv")

CANOPY_PERSISTENCE_SUMMARY = Path("outputs/metadata/canopy_persistence_summary.csv")
ENV_DECLINE_COMPARISON = Path("outputs/metadata/environmental_signal_decline_vs_nondecline.csv")
STRATIFIED_ENV_SIGNAL = Path("outputs/metadata/stratified_environmental_signal_by_canopy.csv")
FN_ENV_PROFILE = Path("outputs/metadata/canopy_only_false_negative_environment_profile.csv")
FEATURE_SET_ROLES = Path("outputs/metadata/feature_set_role_summary.csv")
REPORT = Path("outputs/metadata/canopy_environment_context_report.md")

FIG_CANOPY_SCATTER = Path("outputs/figures/canopy_persistence_scatter.png")
FIG_CANOPY_QUANTILE = Path("outputs/figures/canopy_quantile_decline_rate.png")
FIG_ENV_DECLINE = Path("outputs/figures/environmental_signal_decline_vs_nondecline.png")
FIG_STRAT_HOT = Path("outputs/figures/stratified_hot_days_by_canopy.png")
FIG_STRAT_BEUTI = Path("outputs/figures/stratified_beuti_by_canopy.png")
FIG_FN_ENV = Path("outputs/figures/canopy_only_false_negative_environment_profile.png")
NOTEBOOK = Path("notebooks/06_canopy_environment_context_analysis.ipynb")

TARGET = "decline_event_next"
YEAR_MIN = 1989
YEAR_MAX = 2024

ENVIRONMENTAL_VARIABLES = [
    "annual_mean_sst_anomaly",
    "annual_max_sst_anomaly",
    "hot_days_p90",
    "hot_days_p95",
    "cuti_anomaly",
    "beuti_anomaly",
    "annual_mean_cuti",
    "annual_mean_beuti",
]

STRATIFIED_STRESS_VARIABLES = [
    "hot_days_p90",
    "annual_mean_sst_anomaly",
    "beuti_anomaly",
    "cuti_anomaly",
]

FN_PROFILE_VARIABLES = [
    "annual_mean_sst_anomaly",
    "annual_max_sst_anomaly",
    "hot_days_p90",
    "hot_days_p95",
    "beuti_anomaly",
    "cuti_anomaly",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze canopy persistence and environmental context."
    )
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--test-metrics", type=Path, default=TEST_METRICS)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--year-min", type=int, default=YEAR_MIN)
    parser.add_argument("--year-max", type=int, default=YEAR_MAX)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    """Create the parent directory for an output path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """Save a CSV with its parent directory created."""
    ensure_parent(path)
    frame.to_csv(path, index=False)


def load_main_dataset(path: Path, year_min: int, year_max: int) -> pd.DataFrame:
    """Load the complete-feature modeling period."""
    data = pd.read_csv(path)
    subset = data.loc[
        data["year"].between(year_min, year_max)
        & data[TARGET].notna()
        & data["relative_canopy"].notna()
        & data["next_year_relative_canopy"].notna()
    ].copy()
    subset[TARGET] = subset[TARGET].astype(int)
    return subset


def safe_std(values: pd.Series) -> float:
    """Return sample standard deviation as a float, guarding all-missing values."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(clean.std(ddof=1))


def standardized_difference(decline: pd.Series, non_decline: pd.Series) -> float:
    """Compute a pooled-standard-deviation standardized mean difference."""
    d = pd.to_numeric(decline, errors="coerce").dropna()
    n = pd.to_numeric(non_decline, errors="coerce").dropna()
    if len(d) < 2 or len(n) < 2:
        return float("nan")
    pooled = np.sqrt((d.var(ddof=1) + n.var(ddof=1)) / 2)
    if pooled == 0 or np.isnan(pooled):
        return float("nan")
    return float((d.mean() - n.mean()) / pooled)


def canopy_persistence(data: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Summarize current canopy versus next-year canopy and decline probability."""
    working = data.copy()
    working["canopy_quintile"] = pd.qcut(
        working["relative_canopy"],
        q=5,
        labels=["Q1_lowest", "Q2", "Q3", "Q4", "Q5_highest"],
        duplicates="drop",
    )
    correlation = working["relative_canopy"].corr(working["next_year_relative_canopy"])
    summary = (
        working.groupby("canopy_quintile", observed=True)
        .agg(
            n_rows=("cell_id", "size"),
            mean_current_relative_canopy=("relative_canopy", "mean"),
            median_current_relative_canopy=("relative_canopy", "median"),
            mean_next_year_relative_canopy=("next_year_relative_canopy", "mean"),
            median_next_year_relative_canopy=("next_year_relative_canopy", "median"),
            decline_event_next_rate=(TARGET, "mean"),
            decline_event_next_count=(TARGET, "sum"),
        )
        .reset_index()
    )
    summary.insert(0, "correlation_current_vs_next_relative_canopy", correlation)
    return summary, float(correlation)


def plot_canopy_persistence(data: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Create canopy persistence figures."""
    ensure_parent(FIG_CANOPY_SCATTER)
    plt.figure(figsize=(7, 5))
    colors = data[TARGET].map({0: "#4c78a8", 1: "#f58518"})
    plt.scatter(
        data["relative_canopy"],
        data["next_year_relative_canopy"],
        c=colors,
        alpha=0.55,
        s=18,
        linewidths=0,
    )
    max_axis = max(data["relative_canopy"].max(), data["next_year_relative_canopy"].max())
    plt.plot([0, max_axis], [0, max_axis], color="#333333", linestyle="--", linewidth=1)
    plt.xlabel("Current-year relative canopy")
    plt.ylabel("Next-year relative canopy")
    plt.title("Canopy Persistence: Current vs Next-Year Condition")
    plt.tight_layout()
    plt.savefig(FIG_CANOPY_SCATTER, dpi=220)
    plt.close()

    ensure_parent(FIG_CANOPY_QUANTILE)
    plt.figure(figsize=(7, 5))
    plt.bar(
        summary["canopy_quintile"].astype(str),
        summary["decline_event_next_rate"],
        color="#f58518",
    )
    plt.ylabel("Next-year decline rate")
    plt.xlabel("Current relative canopy quintile")
    plt.title("Next-Year Decline Rate by Current Canopy Quintile")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, max(0.05, summary["decline_event_next_rate"].max() * 1.15))
    plt.tight_layout()
    plt.savefig(FIG_CANOPY_QUANTILE, dpi=220)
    plt.close()


def environmental_decline_comparison(data: pd.DataFrame) -> pd.DataFrame:
    """Compare environmental variables for decline and non-decline rows."""
    rows = []
    for variable in ENVIRONMENTAL_VARIABLES:
        if variable not in data.columns:
            continue
        decline = data.loc[data[TARGET] == 1, variable]
        non_decline = data.loc[data[TARGET] == 0, variable]
        rows.append(
            {
                "variable": variable,
                "decline_mean": decline.mean(),
                "non_decline_mean": non_decline.mean(),
                "difference": decline.mean() - non_decline.mean(),
                "standardized_difference": standardized_difference(decline, non_decline),
                "decline_median": decline.median(),
                "non_decline_median": non_decline.median(),
                "decline_non_missing": decline.notna().sum(),
                "non_decline_non_missing": non_decline.notna().sum(),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "standardized_difference", key=lambda s: s.abs(), ascending=False
    )


def plot_environmental_decline_comparison(comparison: pd.DataFrame) -> None:
    """Plot standardized differences for decline versus non-decline rows."""
    ensure_parent(FIG_ENV_DECLINE)
    plot_data = comparison.sort_values("standardized_difference")
    colors = np.where(plot_data["standardized_difference"] >= 0, "#f58518", "#4c78a8")
    plt.figure(figsize=(8, 5))
    plt.barh(plot_data["variable"], plot_data["standardized_difference"], color=colors)
    plt.axvline(0, color="#333333", linewidth=1)
    plt.xlabel("Standardized difference: decline minus non-decline")
    plt.title("NOAA Environmental Signal in Decline Rows")
    plt.tight_layout()
    plt.savefig(FIG_ENV_DECLINE, dpi=220)
    plt.close()


def add_canopy_groups(data: pd.DataFrame) -> pd.DataFrame:
    """Add tertile-based current canopy condition groups."""
    working = data.copy()
    working["canopy_group"] = pd.qcut(
        working["relative_canopy"],
        q=3,
        labels=["low_canopy", "medium_canopy", "high_canopy"],
        duplicates="drop",
    )
    return working


def stress_label(variable: str, value: float, threshold: float) -> str:
    """Classify environmental stress relative to a median threshold."""
    if pd.isna(value):
        return "missing"
    if variable in {"beuti_anomaly", "cuti_anomaly"}:
        return "high_stress" if value <= threshold else "low_stress"
    return "high_stress" if value >= threshold else "low_stress"


def stratified_environmental_signal(data: pd.DataFrame) -> pd.DataFrame:
    """Compare decline rates by canopy group and environmental stress class."""
    working = add_canopy_groups(data)
    rows = []
    for variable in STRATIFIED_STRESS_VARIABLES:
        if variable not in working.columns:
            continue
        threshold = working[variable].median(skipna=True)
        temp = working.loc[working[variable].notna()].copy()
        temp["stress_group"] = temp[variable].apply(lambda value: stress_label(variable, value, threshold))
        for (canopy_group, stress_group), group in temp.groupby(
            ["canopy_group", "stress_group"], observed=True
        ):
            rows.append(
                {
                    "canopy_group": canopy_group,
                    "environment_variable": variable,
                    "stress_group": stress_group,
                    "threshold_type": "median",
                    "threshold_value": threshold,
                    "n_rows": len(group),
                    "decline_count": int(group[TARGET].sum()),
                    "decline_rate": group[TARGET].mean(),
                    "mean_environment_value": group[variable].mean(),
                }
            )
    return pd.DataFrame(rows)


def plot_stratified_signal(stratified: pd.DataFrame, variable: str, output: Path) -> None:
    """Create grouped bar plot for one stratified environmental signal."""
    ensure_parent(output)
    plot_data = stratified.loc[stratified["environment_variable"] == variable].copy()
    if plot_data.empty:
        return
    pivot = plot_data.pivot(index="canopy_group", columns="stress_group", values="decline_rate")
    pivot = pivot.reindex(["low_canopy", "medium_canopy", "high_canopy"])
    pivot = pivot[[col for col in ["low_stress", "high_stress"] if col in pivot.columns]]
    ax = pivot.plot(kind="bar", figsize=(7, 5), color=["#4c78a8", "#f58518"])
    ax.set_ylabel("Next-year decline rate")
    ax.set_xlabel("Current canopy condition")
    ax.set_title(f"Decline Rate by Canopy Group and {variable}")
    ax.legend(title="Stress group")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close()


def best_canopy_model(test_metrics: pd.DataFrame) -> str:
    """Return the canopy-only test model with the highest PR-AUC."""
    canopy = test_metrics.loc[test_metrics["feature_set"] == "canopy_only"].copy()
    return canopy.sort_values(["pr_auc", "recall", "f1"], ascending=False).iloc[0]["model"]


def false_negative_environment_profile(
    data: pd.DataFrame, predictions_path: Path, test_metrics_path: Path
) -> tuple[pd.DataFrame, bool, str]:
    """Summarize environmental profile of canopy-only false negatives."""
    if not predictions_path.exists() or not test_metrics_path.exists():
        return pd.DataFrame(), False, "Prediction-level model files were not available."

    predictions = pd.read_csv(predictions_path)
    test_metrics = pd.read_csv(test_metrics_path)
    model = best_canopy_model(test_metrics)
    selected = predictions.loc[
        (predictions["feature_set"] == "canopy_only") & (predictions["model"] == model)
    ].copy()
    selected["prediction_group"] = np.select(
        [
            (selected["y_true"] == 1) & (selected["y_pred"] == 0),
            (selected["y_true"] == 1) & (selected["y_pred"] == 1),
            (selected["y_true"] == 0) & (selected["y_pred"] == 0),
            (selected["y_true"] == 0) & (selected["y_pred"] == 1),
        ],
        ["false_negative", "true_positive", "true_negative", "false_positive"],
        default="other",
    )
    merged = selected.merge(
        data[["cell_id", "year"] + [v for v in FN_PROFILE_VARIABLES if v in data.columns]],
        on=["cell_id", "year"],
        how="left",
    )
    rows = []
    group_order = ["all_test_rows", "false_negative", "true_positive", "true_negative", "false_positive"]
    for group_name in group_order:
        group = merged if group_name == "all_test_rows" else merged.loc[merged["prediction_group"] == group_name]
        for variable in FN_PROFILE_VARIABLES:
            if variable not in group.columns:
                continue
            rows.append(
                {
                    "canopy_only_model": model,
                    "comparison_group": group_name,
                    "variable": variable,
                    "n_rows": len(group),
                    "non_missing": group[variable].notna().sum(),
                    "mean": group[variable].mean(),
                    "median": group[variable].median(),
                    "std": safe_std(group[variable]),
                }
            )
    return pd.DataFrame(rows), True, f"Canopy-only false-negative profile used {model} predictions."


def plot_false_negative_profile(profile: pd.DataFrame) -> None:
    """Plot false-negative environmental profile against comparison groups."""
    if profile.empty:
        return
    ensure_parent(FIG_FN_ENV)
    selected_groups = ["false_negative", "true_positive", "true_negative"]
    baseline = profile.loc[profile["comparison_group"] == "all_test_rows"].set_index("variable")
    plot_data = profile.loc[profile["comparison_group"].isin(selected_groups)].copy()
    plot_data["standardized_mean_difference_vs_all_test"] = plot_data.apply(
        lambda row: (
            (row["mean"] - baseline.loc[row["variable"], "mean"]) / baseline.loc[row["variable"], "std"]
            if row["variable"] in baseline.index and baseline.loc[row["variable"], "std"] != 0
            else np.nan
        ),
        axis=1,
    )
    pivot = plot_data.pivot(
        index="variable", columns="comparison_group", values="standardized_mean_difference_vs_all_test"
    )
    pivot = pivot[[col for col in selected_groups if col in pivot.columns]]
    ax = pivot.plot(kind="bar", figsize=(9, 5), color=["#f58518", "#54a24b", "#4c78a8"])
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_ylabel("Standardized mean difference vs all test rows")
    ax.set_xlabel("Environmental variable")
    ax.set_title("Environmental Profile of Canopy-Only Test Predictions")
    ax.legend(title="Prediction group")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_FN_ENV, dpi=220)
    plt.close()


def feature_set_role_summary() -> pd.DataFrame:
    """Create a conceptual role table for feature sets."""
    rows = [
        {
            "feature_set": "canopy_only",
            "analytical_role": "biological state monitoring",
            "interpretation": "Direct satellite-derived canopy condition and recent canopy history.",
        },
        {
            "feature_set": "oisst_only",
            "analytical_role": "thermal exposure screening",
            "interpretation": "SST anomalies, hot-day counts, and annual thermal stress context.",
        },
        {
            "feature_set": "canopy_noaa",
            "analytical_role": "biological state + environmental context",
            "interpretation": "Combines current canopy state with thermal and upwelling context.",
        },
        {
            "feature_set": "CUTI/BEUTI",
            "analytical_role": "upwelling/nitrate-flux proxy context",
            "interpretation": "Latitude-bin proxies for coastal upwelling and biologically effective upwelling.",
        },
    ]
    return pd.DataFrame(rows)


def format_top_environmental(comparison: pd.DataFrame, n: int = 3) -> list[str]:
    """Return short bullet text for the strongest environmental differences."""
    bullets = []
    for _, row in comparison.head(n).iterrows():
        direction = "higher" if row["difference"] > 0 else "lower"
        bullets.append(
            f"- `{row['variable']}` is {direction} in decline rows "
            f"(standardized difference={row['standardized_difference']:.3f})."
        )
    return bullets


def stratified_highlights(stratified: pd.DataFrame) -> list[str]:
    """Summarize high-stress minus low-stress decline rates by canopy group."""
    bullets = []
    for variable in STRATIFIED_STRESS_VARIABLES:
        part = stratified.loc[stratified["environment_variable"] == variable]
        for canopy_group in ["low_canopy", "medium_canopy", "high_canopy"]:
            group = part.loc[part["canopy_group"].astype(str) == canopy_group]
            values = group.set_index("stress_group")["decline_rate"]
            if {"high_stress", "low_stress"}.issubset(values.index):
                delta = values.loc["high_stress"] - values.loc["low_stress"]
                bullets.append(
                    f"- `{variable}` in `{canopy_group}`: high-stress decline rate minus "
                    f"low-stress rate = {delta:.3f}."
                )
    return bullets[:8]


def false_negative_highlights(profile: pd.DataFrame) -> list[str]:
    """Summarize false-negative environmental profile against all test rows."""
    if profile.empty:
        return []
    baseline = profile.loc[profile["comparison_group"] == "all_test_rows"].set_index("variable")
    fn = profile.loc[profile["comparison_group"] == "false_negative"].set_index("variable")
    bullets = []
    for variable in FN_PROFILE_VARIABLES:
        if variable not in baseline.index or variable not in fn.index:
            continue
        base_std = baseline.loc[variable, "std"]
        if pd.isna(base_std) or base_std == 0:
            standardized = float("nan")
        else:
            standardized = (fn.loc[variable, "mean"] - baseline.loc[variable, "mean"]) / base_std
        direction = "higher" if fn.loc[variable, "mean"] > baseline.loc[variable, "mean"] else "lower"
        bullets.append(
            f"- Canopy-only false negatives have {direction} `{variable}` than all test rows "
            f"(standardized difference={standardized:.3f})."
        )
    return bullets


def write_report(
    data: pd.DataFrame,
    canopy_summary: pd.DataFrame,
    correlation: float,
    env_comparison: pd.DataFrame,
    stratified: pd.DataFrame,
    fn_profile: pd.DataFrame,
    fn_available: bool,
    fn_note: str,
) -> None:
    """Write the main Markdown report."""
    overall_decline_rate = data[TARGET].mean()
    lowest_quintile = canopy_summary.iloc[0]
    highest_quintile = canopy_summary.iloc[-1]
    lines = [
        "# Canopy Persistence and Environmental Context Analysis",
        "",
        "## Research Question",
        "",
        "How should the initial model finding be interpreted when canopy-only models outperform NOAA-enhanced models by aggregate PR-AUC?",
        "",
        "## Why Canopy-Only Can Be Strong",
        "",
        "Current canopy condition can be a strong short-term predictor because kelp canopy state is temporally persistent. The project target is next-year decline, so current-year canopy observations provide direct biological state information that may already integrate recent environmental stress, disturbance, and recovery history.",
        "",
        "## Canopy Persistence Results",
        "",
        f"- Rows analyzed: {len(data)} across {data['cell_id'].nunique()} cells from {int(data['year'].min())}-{int(data['year'].max())}.",
        f"- Overall next-year decline rate: {overall_decline_rate:.3f}.",
        f"- Correlation between current relative canopy and next-year relative canopy: {correlation:.3f}.",
        f"- Lowest current-canopy quintile decline rate: {lowest_quintile['decline_event_next_rate']:.3f}.",
        f"- Highest current-canopy quintile decline rate: {highest_quintile['decline_event_next_rate']:.3f}.",
        "",
        "## Decline vs Non-Decline NOAA Signal Results",
        "",
        *format_top_environmental(env_comparison),
        "",
        "These comparisons test whether NOAA variables show directional environmental differences even when they do not replace direct canopy observations as the highest-performing aggregate predictors.",
        "",
        "## Stratified Analysis by Canopy Condition",
        "",
        "Environmental stress indicators were compared within low, medium, and high current-canopy groups. This asks whether stress variables provide context beyond current biological state.",
        "",
        *stratified_highlights(stratified),
        "",
        "## Canopy-Only False-Negative Environmental Profile",
        "",
        fn_note,
        "",
        *false_negative_highlights(fn_profile),
        "",
        "## Feature-Set Role Interpretation",
        "",
        "- `canopy_only`: biological state monitoring.",
        "- `oisst_only`: thermal exposure screening.",
        "- `canopy_noaa`: biological state plus environmental context.",
        "- `CUTI/BEUTI`: upwelling and nitrate-flux proxy context.",
        "",
        "## Final Interpretation",
        "",
        "Current canopy condition was the strongest short-term predictor of next-year kelp decline, reflecting temporal persistence in canopy state. NOAA environmental indicators did not outperform canopy-only models in aggregate prediction performance, but SST stress and CUTI/BEUTI variables provide interpretable environmental context. Therefore, NOAA variables are better interpreted as environmental-risk context rather than replacements for direct canopy observations.",
    ]
    ensure_parent(REPORT)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_notebook() -> None:
    """Write a lightweight notebook that reruns and inspects the workflow."""
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Canopy Persistence and Environmental Context Analysis\n",
                "\n",
                "This notebook reruns the scripted analysis and previews the main output tables.\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": ["!python3 ../scripts/analyze_canopy_environment_context.py\n"],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "pd.read_csv('../outputs/metadata/canopy_persistence_summary.csv')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pd.read_csv('../outputs/metadata/environmental_signal_decline_vs_nondecline.csv')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pd.read_csv('../outputs/metadata/stratified_environmental_signal_by_canopy.csv')\n",
            ],
        },
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    ensure_parent(NOTEBOOK)
    NOTEBOOK.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Run the canopy persistence and environmental context analysis."""
    args = parse_args()
    data = load_main_dataset(args.dataset, args.year_min, args.year_max)

    canopy_summary, correlation = canopy_persistence(data)
    save_csv(canopy_summary, CANOPY_PERSISTENCE_SUMMARY)
    plot_canopy_persistence(data, canopy_summary)

    env_comparison = environmental_decline_comparison(data)
    save_csv(env_comparison, ENV_DECLINE_COMPARISON)
    plot_environmental_decline_comparison(env_comparison)

    stratified = stratified_environmental_signal(data)
    save_csv(stratified, STRATIFIED_ENV_SIGNAL)
    plot_stratified_signal(stratified, "hot_days_p90", FIG_STRAT_HOT)
    plot_stratified_signal(stratified, "beuti_anomaly", FIG_STRAT_BEUTI)

    fn_profile, fn_available, fn_note = false_negative_environment_profile(
        data, args.predictions, args.test_metrics
    )
    if fn_available:
        save_csv(fn_profile, FN_ENV_PROFILE)
        plot_false_negative_profile(fn_profile)

    roles = feature_set_role_summary()
    save_csv(roles, FEATURE_SET_ROLES)

    write_report(
        data,
        canopy_summary,
        correlation,
        env_comparison,
        stratified,
        fn_profile,
        fn_available,
        fn_note,
    )
    write_notebook()

    print("Canopy/environment context analysis complete.")
    print(f"Rows analyzed: {len(data)}")
    print(f"Current vs next-year relative canopy correlation: {correlation:.3f}")
    print(fn_note)


if __name__ == "__main__":
    main()
