"""Interpret kelp decline early-warning models with SHAP."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from train_model_comparison import (
    CUTI_BEUTI_FEATURES,
    LEAKAGE_VARIABLES,
    OISST_FEATURES,
    TARGET,
    feature_sets,
    load_dataset,
    main_subset,
    model_specs,
    predict_scores,
    preprocessor,
    split_data,
)


DATASET = Path("data/processed/modeling_dataset_ge500_noaa_v1.csv")
RESULTS = Path("outputs/metadata/model_comparison_results.csv")
TEST_METRICS = Path("outputs/metadata/model_comparison_test_metrics.csv")
MODEL_DIAGNOSTICS_REPORT = Path("outputs/metadata/model_diagnostics_report.md")
CANOPY_ENVIRONMENT_REPORT = Path("outputs/metadata/canopy_environment_context_report.md")

GROUPED_IMPORTANCE = Path("outputs/metadata/shap_grouped_importance.csv")
NOAA_IMPORTANCE = Path("outputs/metadata/shap_noaa_feature_importance.csv")
DEPENDENCE_DIRECTION = Path("outputs/metadata/shap_dependence_direction_summary.csv")
LOCAL_CASES = Path("outputs/metadata/shap_local_high_risk_cases.csv")
REPORT = Path("outputs/metadata/shap_interpretation_report.md")

GROUPED_FIGURE = Path("outputs/figures/shap_grouped_importance.png")
NOTEBOOK = Path("notebooks/07_shap_interpretation.ipynb")

TREE_MODELS = ["Random Forest", "XGBoost", "LightGBM"]
PRIMARY_MODELS = [
    ("canopy_only", "Random Forest"),
]
DEPENDENCE_FEATURES = [
    "annual_mean_sst_anomaly",
    "annual_max_sst_anomaly",
    "hot_days_p90",
    "hot_days_p95",
    "beuti_anomaly",
    "cuti_anomaly",
]
NOAA_FEATURES = set(OISST_FEATURES + CUTI_BEUTI_FEATURES)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Interpret selected kelp decline models with SHAP.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--test-metrics", type=Path, default=TEST_METRICS)
    return parser.parse_args()


def slug(text: str) -> str:
    """Create a compact filename-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def ensure_parent(path: Path) -> None:
    """Create output parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """Save a non-empty CSV."""
    if frame.empty:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    ensure_parent(path)
    frame.to_csv(path, index=False)


def load_inputs(dataset: Path, test_metrics: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Load data and validate the temporal split."""
    for path in [dataset, RESULTS, test_metrics, MODEL_DIAGNOSTICS_REPORT, CANOPY_ENVIRONMENT_REPORT]:
        if not path.exists():
            raise FileNotFoundError(path)
    data = main_subset(load_dataset(dataset))
    splits = split_data(data)
    if (splits["train"]["year"].min(), splits["train"]["year"].max()) != (1989, 2016):
        raise ValueError("Train split must be 1989-2016.")
    if (splits["validation"]["year"].min(), splits["validation"]["year"].max()) != (2017, 2020):
        raise ValueError("Validation split must be 2017-2020.")
    if (splits["test"]["year"].min(), splits["test"]["year"].max()) != (2021, 2024):
        raise ValueError("Test split must be 2021-2024.")
    metrics = pd.read_csv(test_metrics)
    return data, splits, metrics


def choose_tree_canopy_noaa_model(metrics: pd.DataFrame) -> str:
    """Select the best tree-based canopy+NOAA model by test PR-AUC."""
    candidates = metrics.loc[
        (metrics["feature_set"] == "canopy_noaa") & (metrics["model"].isin(TREE_MODELS))
    ].copy()
    if candidates.empty:
        raise ValueError("No tree-based canopy+NOAA metrics found.")
    return candidates.sort_values(["pr_auc", "recall", "f1"], ascending=False).iloc[0]["model"]


def fit_model(
    splits: dict[str, pd.DataFrame], feature_set_name: str, model_name: str
) -> tuple[Pipeline, list[str]]:
    """Fit one selected model using the same training definitions as model comparison."""
    sets = feature_sets(pd.concat(splits.values(), ignore_index=True))
    features = sets[feature_set_name]
    leaked = sorted(set(features) & LEAKAGE_VARIABLES)
    if leaked:
        raise ValueError(f"Leakage variables included in {feature_set_name}: {leaked}")
    if "cell_id" in features or "year" in features:
        raise ValueError("cell_id and year must not be ordinary SHAP predictors.")
    y_train = splits["train"][TARGET].astype(int)
    specs = model_specs(y_train)
    estimator, needs_scaling = specs[model_name]
    pipe = Pipeline(
        [
            ("preprocess", preprocessor(splits["train"], features, scale=needs_scaling)),
            ("model", estimator),
        ]
    )
    pipe.fit(splits["train"][features], y_train)
    return pipe, features


def transformed_frame(pipe: Pipeline, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Return transformed model matrix with readable feature names."""
    transformer = pipe.named_steps["preprocess"]
    transformed = transformer.transform(frame[features])
    feature_names = transformer.get_feature_names_out()
    clean_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]
    out = pd.DataFrame(transformed, columns=clean_names, index=frame.index)
    if out.shape[1] != len(clean_names):
        raise ValueError("Transformed SHAP matrix and feature names are misaligned.")
    return out


def positive_class_shap_values(explainer: shap.TreeExplainer, matrix: pd.DataFrame) -> np.ndarray:
    """Return positive-class SHAP values for binary classifiers."""
    values = explainer.shap_values(matrix)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        if values.shape[2] > 1:
            values = values[:, :, 1]
        else:
            values = values[:, :, 0]
    if values.shape != matrix.shape:
        raise ValueError(f"SHAP values {values.shape} do not align with matrix {matrix.shape}.")
    return values


def feature_group(feature: str) -> str:
    """Assign feature group labels for SHAP summaries."""
    raw = feature
    if raw.startswith("region_group"):
        return "region"
    if raw in {"center_lat", "center_lon"}:
        return "spatial"
    if raw in {
        "relative_canopy",
        "kelp_area_m2",
        "count_cells_kelp",
        "count_cells_no_clouds",
        "count_cells_historic_footprint",
        "historical_footprint_area_m2",
        "lag1_relative_canopy",
        "relative_canopy_change_lag1",
    }:
        return "canopy"
    if raw in OISST_FEATURES:
        return "OISST"
    if "cuti" in raw:
        return "CUTI"
    if "beuti" in raw:
        return "BEUTI"
    return "other"


def shap_importance(
    shap_values: np.ndarray, matrix: pd.DataFrame, feature_set_name: str, model_name: str
) -> pd.DataFrame:
    """Create feature-level SHAP importance table."""
    importance = pd.DataFrame(
        {
            "feature": matrix.columns,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    )
    importance["rank"] = importance["mean_abs_shap"].rank(method="first", ascending=False).astype(int)
    importance["feature_group"] = importance["feature"].map(feature_group)
    importance["feature_set"] = feature_set_name
    importance["model"] = model_name
    return importance.sort_values("rank")


def plot_shap_summary(
    shap_values: np.ndarray, matrix: pd.DataFrame, feature_set_name: str, model_name: str
) -> tuple[Path, Path]:
    """Create SHAP beeswarm and bar plots."""
    model_slug = slug(model_name)
    summary_path = Path(f"outputs/figures/shap_summary_{feature_set_name}_{model_slug}.png")
    bar_path = Path(f"outputs/figures/shap_bar_{feature_set_name}_{model_slug}.png")

    ensure_parent(summary_path)
    shap.summary_plot(shap_values, matrix, show=False, max_display=18)
    plt.title(f"SHAP Summary: {feature_set_name} / {model_name}")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=220, bbox_inches="tight")
    plt.close()

    ensure_parent(bar_path)
    shap.summary_plot(shap_values, matrix, show=False, plot_type="bar", max_display=18)
    plt.title(f"SHAP Importance: {feature_set_name} / {model_name}")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=220, bbox_inches="tight")
    plt.close()
    return summary_path, bar_path


def plot_grouped_importance(grouped: pd.DataFrame) -> None:
    """Plot grouped SHAP importance for interpreted models."""
    ensure_parent(GROUPED_FIGURE)
    plot_data = grouped.pivot_table(
        index="feature_group", columns="model_label", values="mean_abs_shap", aggfunc="sum"
    ).fillna(0)
    plot_data = plot_data.loc[plot_data.sum(axis=1).sort_values(ascending=False).index]
    ax = plot_data.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Mean absolute SHAP")
    ax.set_xlabel("Feature group")
    ax.set_title("Grouped SHAP Importance")
    ax.legend(title="Model")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(GROUPED_FIGURE, dpi=220)
    plt.close()


def plot_dependence(shap_values: np.ndarray, matrix: pd.DataFrame) -> list[str]:
    """Create dependence plots for key environmental features."""
    skipped = []
    for feature in DEPENDENCE_FEATURES:
        output = Path(f"outputs/figures/shap_dependence_{feature}.png")
        if feature not in matrix.columns:
            skipped.append(f"{feature}: unavailable after preprocessing")
            continue
        ensure_parent(output)
        shap.dependence_plot(feature, shap_values, matrix, show=False, interaction_index=None)
        plt.title(f"SHAP Dependence: {feature}")
        plt.tight_layout()
        plt.savefig(output, dpi=220, bbox_inches="tight")
        plt.close()
    return skipped


def dependence_direction_summary(shap_values: np.ndarray, matrix: pd.DataFrame) -> pd.DataFrame:
    """Summarize diagnostic SHAP associations for key dependence features."""
    shap_frame = pd.DataFrame(shap_values, columns=matrix.columns, index=matrix.index)
    rows = []
    for feature in DEPENDENCE_FEATURES:
        if feature not in matrix.columns:
            rows.append(
                {
                    "feature": feature,
                    "status": "unavailable",
                    "spearman_corr_feature_value_vs_shap": np.nan,
                    "median_feature_value": np.nan,
                    "mean_shap_low_feature_values": np.nan,
                    "mean_shap_high_feature_values": np.nan,
                    "high_minus_low_mean_shap": np.nan,
                    "interpretation": "Feature unavailable after preprocessing.",
                }
            )
            continue
        values = matrix[feature]
        shap_feature = shap_frame[feature]
        median = values.median()
        low = shap_feature.loc[values <= median]
        high = shap_feature.loc[values > median]
        high_minus_low = high.mean() - low.mean()
        corr = values.corr(shap_feature, method="spearman")
        if pd.isna(corr):
            interpretation = "Direction could not be estimated."
        elif corr > 0.05:
            interpretation = "In this fitted model, higher feature values have a positive SHAP association."
        elif corr < -0.05:
            interpretation = "In this fitted model, higher feature values have a negative SHAP association."
        else:
            interpretation = "In this fitted model, feature values have a weak monotonic SHAP association."
        rows.append(
            {
                "feature": feature,
                "status": "available",
                "spearman_corr_feature_value_vs_shap": corr,
                "median_feature_value": median,
                "mean_shap_low_feature_values": low.mean(),
                "mean_shap_high_feature_values": high.mean(),
                "high_minus_low_mean_shap": high_minus_low,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def noaa_importance(importance: pd.DataFrame) -> pd.DataFrame:
    """Create NOAA-only SHAP importance table."""
    noaa = importance.loc[importance["feature"].isin(NOAA_FEATURES)].copy()
    if noaa.empty:
        raise ValueError("NOAA-only SHAP importance table is empty.")
    noaa = noaa.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    noaa["rank_within_noaa"] = np.arange(1, len(noaa) + 1)
    return noaa[["feature", "mean_abs_shap", "rank_within_noaa", "feature_group"]]


def top_feature_text(row_values: pd.Series, positive: bool, n: int = 5) -> str:
    """Summarize top local SHAP features for one row."""
    values = row_values.sort_values(ascending=not positive)
    if positive:
        values = values.loc[values > 0]
    else:
        values = values.loc[values < 0].sort_values()
    parts = [f"{feature}={value:.4f}" for feature, value in values.head(n).items()]
    return "; ".join(parts)


def local_high_risk_cases(
    canopy_pipe: Pipeline,
    canopy_features: list[str],
    noaa_pipe: Pipeline,
    noaa_features: list[str],
    noaa_matrix: pd.DataFrame,
    noaa_shap_values: np.ndarray,
    splits: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create local SHAP summaries for selected high-risk test cases."""
    test = splits["test"].copy()
    noaa_scores = predict_scores(noaa_pipe, test[noaa_features])
    noaa_pred = (noaa_scores >= 0.5).astype(int)
    canopy_scores = predict_scores(canopy_pipe, test[canopy_features])
    canopy_pred = (canopy_scores >= 0.5).astype(int)
    test = test.assign(
        predicted_probability=noaa_scores,
        noaa_pred=noaa_pred,
        canopy_pred=canopy_pred,
    )
    top_tp = test.loc[(test[TARGET] == 1) & (test["noaa_pred"] == 1)].nlargest(
        5, "predicted_probability"
    )
    caught = test.loc[(test[TARGET] == 1) & (test["canopy_pred"] == 0) & (test["noaa_pred"] == 1)]
    fallback = test.loc[(test[TARGET] == 1) & (test["canopy_pred"] == 0)]
    selected = pd.concat([top_tp, caught, fallback], axis=0)
    selected = selected.loc[~selected.index.duplicated(keep="first")].head(12)
    shap_frame = pd.DataFrame(noaa_shap_values, columns=noaa_matrix.columns, index=test.index)

    rows = []
    for idx, row in selected.iterrows():
        values = shap_frame.loc[idx]
        rows.append(
            {
                "case_selection": (
                    "top_high_risk_true_positive"
                    if idx in set(top_tp.index)
                    else (
                        "canopy_false_negative_caught_by_canopy_noaa"
                        if idx in set(caught.index)
                        else "canopy_false_negative_environment_summary"
                    )
                ),
                "cell_id": row["cell_id"],
                "year": int(row["year"]),
                "region_group": row["region_group"],
                "y_true": int(row[TARGET]),
                "predicted_probability": row["predicted_probability"],
                "top_positive_shap_features": top_feature_text(values, positive=True),
                "top_negative_shap_features": top_feature_text(values, positive=False),
                "relative_canopy": row.get("relative_canopy", np.nan),
                "annual_mean_sst_anomaly": row.get("annual_mean_sst_anomaly", np.nan),
                "hot_days_p90": row.get("hot_days_p90", np.nan),
                "beuti_anomaly": row.get("beuti_anomaly", np.nan),
                "cuti_anomaly": row.get("cuti_anomaly", np.nan),
            }
        )
    return pd.DataFrame(rows)


def top_features_text(importance: pd.DataFrame, n: int = 8) -> list[str]:
    """Format top SHAP features for reports."""
    return [
        f"- `{row.feature}` ({row.feature_group}): mean |SHAP| = {row.mean_abs_shap:.4f}"
        for row in importance.head(n).itertuples(index=False)
    ]


def dependence_notes(direction: pd.DataFrame, skipped: list[str]) -> list[str]:
    """Create concise dependence plot notes."""
    notes = []
    available = direction.loc[direction["status"] == "available"]
    for row in available.itertuples(index=False):
        notes.append(
            f"- `{row.feature}`: {row.interpretation} "
            f"(Spearman r={row.spearman_corr_feature_value_vs_shap:.3f}; "
            f"high-minus-low mean SHAP={row.high_minus_low_mean_shap:.4f})."
        )
    for item in skipped:
        notes.append(f"- Skipped `{item}`.")
    return notes


def write_report(
    canopy_importance: pd.DataFrame,
    noaa_model_importance: pd.DataFrame,
    noaa_only: pd.DataFrame,
    grouped: pd.DataFrame,
    local_cases: pd.DataFrame,
    direction: pd.DataFrame,
    tree_noaa_model: str,
    svm_pr_auc: float,
    tree_pr_auc: float,
    dependence_skipped: list[str],
) -> None:
    """Write the SHAP interpretation report."""
    grouped_lines = [
        f"- {row.model_label} / {row.feature_group}: mean |SHAP| = {row.mean_abs_shap:.4f} "
        f"({row.share_of_model_importance:.1%})"
        for row in grouped.itertuples(index=False)
    ]
    lines = [
        "# SHAP Interpretation Report",
        "",
        "## Models Interpreted",
        "",
        "- Model A: `canopy_only / Random Forest`.",
        f"- Model B: `canopy_noaa / {tree_noaa_model}`.",
        "",
        "## Model Selection Rationale",
        "",
        "The canopy-only Random Forest was selected because it was the best overall model by test PR-AUC in the initial model comparison.",
        "",
        f"The best canopy+NOAA model by PR-AUC was SVM (test PR-AUC={svm_pr_auc:.4f}). For SHAP, `{tree_noaa_model}` was used among the tree-based canopy+NOAA models (test PR-AUC={tree_pr_auc:.4f}) because TreeExplainer is faster and more stable than full Kernel SHAP. SVM Kernel SHAP is left as an optional future refinement.",
        "",
        "## Top Canopy-Only Features",
        "",
        *top_features_text(canopy_importance),
        "",
        "## Top Canopy+NOAA Features",
        "",
        *top_features_text(noaa_model_importance),
        "",
        "## NOAA Environmental Feature Rankings",
        "",
        *top_features_text(noaa_only.rename(columns={"rank_within_noaa": "rank"}).assign(model="")),
        "",
        "## Grouped Importance",
        "",
        *grouped_lines,
        "",
        "## Dependence Plot Interpretation",
        "",
        "Dependence plots were used to inspect how the interpreted canopy+NOAA Random Forest used environmental variables when predicting decline risk. These plots summarize model behavior and should not be interpreted as causal ecological effect estimates.",
        "",
        "The SST anomaly variables showed relatively clear positive model associations with predicted decline risk. In contrast, hot-day exceedance counts and CUTI/BEUTI anomalies showed more mixed or context-dependent patterns. For example, some hot-day variables had negative SHAP associations in the interpreted Random Forest, and BEUTI/CUTI effects were not uniformly aligned with a simple monotonic ecological expectation.",
        "",
        "These results suggest that the canopy+NOAA model uses environmental variables in nonlinear and interaction-dependent ways. Therefore, NOAA variables are interpreted as environmental exposure context rather than direct causal drivers of kelp decline.",
        "",
        "Model-behavior diagnostics:",
        "",
        *dependence_notes(direction, dependence_skipped),
        "",
        "## Local High-Risk Case Examples",
        "",
        f"- Local explanation rows created: {len(local_cases)}.",
        "- The local table reports top positive and negative SHAP features for high-risk true positives and canopy-only false-negative cases where available.",
        "",
        "## Final Interpretation",
        "",
        "Current canopy condition dominates short-term prediction in the aggregate model comparison. The canopy-only SHAP results confirm that current and lagged canopy-condition variables are the main biological-state signals. In the canopy+NOAA model, OISST, CUTI, and BEUTI variables carry substantial internal SHAP importance, indicating that NOAA environmental exposure indicators provide useful stress-context information.",
        "",
        "However, because some SHAP dependence patterns are nonlinear or directionally mixed, these variables should be interpreted as contextual environmental indicators rather than direct causal drivers. The results support a two-layer interpretation: biological state monitoring provides the strongest short-term predictive signal, while NOAA environmental variables help characterize thermal and upwelling/nitrate-flux exposure context.",
        "",
        "Interpretation caution: SHAP values explain how the fitted model used features for prediction. They do not establish ecological causality. Directional patterns should be interpreted alongside known data limitations, including OISST grid resolution, CUTI/BEUTI latitude-bin assignment, missing biotic drivers such as grazing pressure, and the limited number of test years.",
        "",
        "## Limitations",
        "",
        "- SHAP explains model behavior, not causal mechanisms.",
        "- NOAA variables are exposure proxies.",
        "- OISST uses nearest valid ocean grid in Version 1.",
        "- CUTI is a coastal upwelling transport proxy, and BEUTI is a nitrate-flux proxy.",
        "- CUTI/BEUTI are latitude-bin environmental exposure proxies, not cell-specific in situ measurements.",
        "- No direct grazing or urchin variables are included.",
        "- The analysis has a small number of cells and limited test years.",
    ]
    ensure_parent(REPORT)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_notebook() -> None:
    """Create a lightweight notebook wrapper for the SHAP workflow."""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# SHAP Interpretation\n",
                    "\n",
                    "This notebook reruns the scripted SHAP workflow and previews the main output tables.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["!../.venv/bin/python ../scripts/interpret_models_shap.py\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "pd.read_csv('../outputs/metadata/shap_grouped_importance.csv')\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "pd.read_csv('../outputs/metadata/shap_noaa_feature_importance.csv')\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "pd.read_csv('../outputs/metadata/shap_local_high_risk_cases.csv')\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    ensure_parent(NOTEBOOK)
    NOTEBOOK.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Run SHAP interpretation for selected models."""
    args = parse_args()
    _, splits, metrics = load_inputs(args.dataset, args.test_metrics)
    tree_noaa_model = choose_tree_canopy_noaa_model(metrics)
    interpreted = PRIMARY_MODELS + [("canopy_noaa", tree_noaa_model)]
    results: dict[tuple[str, str], dict[str, object]] = {}
    importance_tables = []

    for feature_set_name, model_name in interpreted:
        pipe, features = fit_model(splits, feature_set_name, model_name)
        matrix = transformed_frame(pipe, splits["test"], features)
        explainer = shap.TreeExplainer(pipe.named_steps["model"])
        shap_values = positive_class_shap_values(explainer, matrix)
        importance = shap_importance(shap_values, matrix, feature_set_name, model_name)
        model_slug = slug(model_name)
        importance_path = Path(f"outputs/metadata/shap_feature_importance_{feature_set_name}_{model_slug}.csv")
        save_csv(importance[["feature", "mean_abs_shap", "rank", "feature_group"]], importance_path)
        plot_shap_summary(shap_values, matrix, feature_set_name, model_name)
        importance_tables.append(importance)
        results[(feature_set_name, model_name)] = {
            "pipe": pipe,
            "features": features,
            "matrix": matrix,
            "shap_values": shap_values,
            "importance": importance,
        }

    all_importance = pd.concat(importance_tables, ignore_index=True)
    grouped = (
        all_importance.groupby(["feature_set", "model", "feature_group"], as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values(["feature_set", "model", "mean_abs_shap"], ascending=[True, True, False])
    )
    grouped["model_label"] = grouped["feature_set"] + " / " + grouped["model"]
    totals = grouped.groupby(["feature_set", "model"])["mean_abs_shap"].transform("sum")
    grouped["share_of_model_importance"] = grouped["mean_abs_shap"] / totals
    save_csv(grouped, GROUPED_IMPORTANCE)
    plot_grouped_importance(grouped)

    noaa_key = ("canopy_noaa", tree_noaa_model)
    noaa_only = noaa_importance(results[noaa_key]["importance"])
    save_csv(noaa_only, NOAA_IMPORTANCE)
    dependence_skipped = plot_dependence(results[noaa_key]["shap_values"], results[noaa_key]["matrix"])
    direction = dependence_direction_summary(results[noaa_key]["shap_values"], results[noaa_key]["matrix"])
    save_csv(direction, DEPENDENCE_DIRECTION)

    canopy_key = ("canopy_only", "Random Forest")
    local_cases = local_high_risk_cases(
        results[canopy_key]["pipe"],
        results[canopy_key]["features"],
        results[noaa_key]["pipe"],
        results[noaa_key]["features"],
        results[noaa_key]["matrix"],
        results[noaa_key]["shap_values"],
        splits,
    )
    save_csv(local_cases, LOCAL_CASES)

    svm_pr_auc = float(
        metrics.loc[(metrics["feature_set"] == "canopy_noaa") & (metrics["model"] == "SVM"), "pr_auc"].iloc[0]
    )
    tree_pr_auc = float(
        metrics.loc[
            (metrics["feature_set"] == "canopy_noaa") & (metrics["model"] == tree_noaa_model), "pr_auc"
        ].iloc[0]
    )
    write_report(
        results[canopy_key]["importance"],
        results[noaa_key]["importance"],
        noaa_only,
        grouped,
        local_cases,
        direction,
        tree_noaa_model,
        svm_pr_auc,
        tree_pr_auc,
        dependence_skipped,
    )
    write_notebook()

    print("SHAP interpretation complete.")
    print("Interpreted models:")
    print("- canopy_only / Random Forest")
    print(f"- canopy_noaa / {tree_noaa_model}")
    print(f"SVM canopy+NOAA PR-AUC: {svm_pr_auc:.4f}")
    print(f"Tree canopy+NOAA PR-AUC: {tree_pr_auc:.4f}")


if __name__ == "__main__":
    main()
