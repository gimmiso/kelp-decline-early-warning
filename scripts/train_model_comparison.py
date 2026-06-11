"""Train and compare initial kelp canopy decline prediction models."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


warnings.filterwarnings("ignore", category=UserWarning)

TARGET = "decline_event_next"
INPUT_DATASET = Path("data/processed/modeling_dataset_ge500_noaa_v1.csv")
RESULTS_OUTPUT = Path("outputs/metadata/model_comparison_results.csv")
TEST_METRICS_OUTPUT = Path("outputs/metadata/model_comparison_test_metrics.csv")
CONFUSION_OUTPUT = Path("outputs/metadata/model_comparison_confusion_matrices.csv")
REPORT_OUTPUT = Path("outputs/metadata/model_comparison_report.md")
PERFORMANCE_FIGURE = Path("outputs/figures/model_performance_comparison.png")
PR_FIGURE = Path("outputs/figures/precision_recall_curves.png")
ROC_FIGURE = Path("outputs/figures/roc_curves.png")

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
IDENTIFIER_VARIABLES = {
    "cell_id",
    "year",
    "quarter",
    "source_csv_file",
    "oisst_assignment_method",
    "upwelling_assignment_method",
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
SPATIAL_CONTROL_FEATURES = ["center_lat", "center_lon"]
REGION_FEATURES = ["region_group"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Compare initial kelp decline prediction models.")
    parser.add_argument("--input", type=Path, default=INPUT_DATASET)
    parser.add_argument("--results-output", type=Path, default=RESULTS_OUTPUT)
    parser.add_argument("--test-metrics-output", type=Path, default=TEST_METRICS_OUTPUT)
    parser.add_argument("--confusion-output", type=Path, default=CONFUSION_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    parser.add_argument("--performance-figure", type=Path, default=PERFORMANCE_FIGURE)
    parser.add_argument("--pr-figure", type=Path, default=PR_FIGURE)
    parser.add_argument("--roc-figure", type=Path, default=ROC_FIGURE)
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the modeling dataset and add lagged canopy features."""
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path).sort_values(["cell_id", "year"]).reset_index(drop=True)
    data["lag1_relative_canopy"] = data.groupby("cell_id")["relative_canopy"].shift(1)
    data["relative_canopy_change_lag1"] = data["relative_canopy"] - data["lag1_relative_canopy"]
    return data


def main_subset(data: pd.DataFrame) -> pd.DataFrame:
    """Create and validate the complete-feature modeling period."""
    subset = data.loc[data["year"].between(1989, 2024)].copy()
    if subset["year"].min() != 1989 or subset["year"].max() != 2024:
        raise ValueError("Main subset years must be 1989-2024.")
    if len(subset) != 1800:
        raise ValueError(f"Expected 1,800 rows for 1989-2024, found {len(subset)}.")
    if subset["cell_id"].nunique() != 50:
        raise ValueError(f"Expected 50 unique cells, found {subset['cell_id'].nunique()}.")
    if subset.duplicated(["cell_id", "year"]).any():
        raise ValueError("Duplicate cell_id x year rows found.")
    if set(subset[TARGET].dropna().astype(int).unique()) != {0, 1}:
        raise ValueError("Target must contain both 0 and 1 classes.")
    return subset


def split_data(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Apply the temporal split."""
    splits = {
        "train": data.loc[data["year"].between(1989, 2016)].copy(),
        "validation": data.loc[data["year"].between(2017, 2020)].copy(),
        "test": data.loc[data["year"].between(2021, 2024)].copy(),
    }
    for name, split in splits.items():
        classes = set(split[TARGET].dropna().astype(int).unique())
        if classes != {0, 1}:
            raise ValueError(f"{name} split must contain both classes, found {classes}.")
    if splits["test"]["year"].min() != 2021 or splits["test"]["year"].max() != 2024:
        raise ValueError("Test metrics must be computed only on 2021-2024.")
    return splits


def available(columns: list[str], data: pd.DataFrame) -> list[str]:
    """Return features available in the dataset."""
    return [column for column in columns if column in data.columns]


def feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    """Define feature sets."""
    sets = {
        "canopy_only": available(CANOPY_FEATURES, data),
        "oisst_only": available(OISST_FEATURES, data),
        "canopy_noaa": available(
            CANOPY_FEATURES + OISST_FEATURES + CUTI_BEUTI_FEATURES + SPATIAL_CONTROL_FEATURES + REGION_FEATURES,
            data,
        ),
    }
    for name, features in sets.items():
        leaked = sorted((set(features) & LEAKAGE_VARIABLES) | (set(features) & IDENTIFIER_VARIABLES))
        if leaked:
            raise ValueError(f"Leakage or identifier variables included in {name}: {leaked}")
        all_null = [feature for feature in features if data[feature].isna().all()]
        if all_null:
            raise ValueError(f"All-null feature columns in {name}: {all_null}")
    return sets


def preprocessor(data: pd.DataFrame, features: list[str], scale: bool) -> ColumnTransformer:
    """Create a feature preprocessor fit only inside model pipelines."""
    categorical = [feature for feature in features if not is_numeric_dtype(data[feature])]
    numeric = [feature for feature in features if feature not in categorical]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        remainder="drop",
    )


def positive_class_weight(y: pd.Series) -> float:
    """Return scale_pos_weight for boosted tree models."""
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0:
        return 1.0
    return negatives / positives


def model_specs(y_train: pd.Series) -> dict[str, tuple[object, bool]]:
    """Return model constructors and whether scaling is needed."""
    scale_pos_weight = positive_class_weight(y_train)
    return {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs"),
            True,
        ),
        "SVM": (
            SVC(class_weight="balanced", probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=42),
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=3,
                n_jobs=-1,
            ),
            False,
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=2,
            ),
            False,
        ),
        "LightGBM": (
            LGBMClassifier(
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=15,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                verbose=-1,
                n_jobs=2,
            ),
            False,
        ),
    }


def predict_scores(model: Pipeline, x_data: pd.DataFrame) -> np.ndarray:
    """Return positive-class scores."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_data)[:, 1]
    return model.decision_function(x_data)


def metrics_row(y_true: pd.Series, scores: np.ndarray, feature_set: str, model: str, split: str) -> dict[str, object]:
    """Compute one metrics row."""
    predictions = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "feature_set": feature_set,
        "model": model,
        "split": split,
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "accuracy": accuracy_score(y_true, predictions),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_rows": len(y_true),
        "positive_rate": float(pd.Series(y_true).mean()),
    }


def train_and_evaluate(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, object]]]:
    """Train models for every feature set and return metrics plus fitted test scores."""
    splits = split_data(data)
    sets = feature_sets(data)
    y_train = splits["train"][TARGET].astype(int)
    specs = model_specs(y_train)
    rows: list[dict[str, object]] = []
    curves: dict[tuple[str, str], dict[str, object]] = {}

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
            for split_name in ["validation", "test"]:
                y_true = splits[split_name][TARGET].astype(int)
                scores = predict_scores(pipe, splits[split_name][features])
                rows.append(metrics_row(y_true, scores, feature_set_name, model_name, split_name))
                if split_name == "test":
                    curves[(feature_set_name, model_name)] = {"y_true": y_true.to_numpy(), "scores": scores}

    return pd.DataFrame(rows), curves


def plot_performance(test_metrics: pd.DataFrame, output: Path) -> None:
    """Plot PR-AUC, recall, and F1 for all model-feature combinations."""
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = test_metrics["feature_set"] + " / " + test_metrics["model"]
    x = np.arange(len(test_metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width, test_metrics["pr_auc"], width, label="PR-AUC")
    ax.bar(x, test_metrics["recall"], width, label="Recall")
    ax.bar(x + width, test_metrics["f1"], width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("NOAA V1 Kelp Decline Model Comparison, Test Period 2021-2024")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_curves(curves: dict[tuple[str, str], dict[str, object]], output_pr: Path, output_roc: Path) -> None:
    """Plot test PR and ROC curves for the combined feature set."""
    combined = {key: value for key, value in curves.items() if key[0] == "canopy_noaa"}
    for output, curve_type in [(output_pr, "pr"), (output_roc, "roc")]:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        for (_, model_name), values in combined.items():
            y_true = values["y_true"]
            scores = values["scores"]
            if curve_type == "pr":
                precision, recall, _ = precision_recall_curve(y_true, scores)
                ax.plot(recall, precision, label=model_name)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title("Precision-Recall Curves, Canopy + NOAA, Test 2021-2024")
            else:
                fpr, tpr, _ = roc_curve(y_true, scores)
                ax.plot(fpr, tpr, label=model_name)
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curves, Canopy + NOAA, Test 2021-2024")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output, dpi=200)
        plt.close(fig)


def write_report(metrics: pd.DataFrame, data: pd.DataFrame, output: Path) -> None:
    """Write the model comparison report."""
    test = metrics.loc[metrics["split"] == "test"].copy()
    best_pr = test.sort_values(["pr_auc", "recall", "f1"], ascending=False).iloc[0]
    best_recall = test.sort_values(["recall", "pr_auc", "f1"], ascending=False).iloc[0]
    canopy_best = test.loc[test["feature_set"] == "canopy_only", "pr_auc"].max()
    combined_best = test.loc[test["feature_set"] == "canopy_noaa", "pr_auc"].max()
    improves = combined_best > canopy_best

    lines = [
        "# Initial 5-Model Comparison Report",
        "",
        "## Dataset",
        "",
        "- Dataset used: `data/processed/modeling_dataset_ge500_noaa_v1.csv`",
        "- Main modeling years: 1989-2024",
        f"- Main modeling rows: {len(data)}",
        f"- Cells: {data['cell_id'].nunique()}",
        "- Target: `decline_event_next`, indicating whether next-year relative canopy falls below the cell-specific 1984-2013 baseline 25th percentile.",
        "",
        "## Temporal Split",
        "",
        "- Train: 1989-2016",
        "- Validation: 2017-2020",
        "- Test: 2021-2024",
        "",
        "## Feature Sets",
        "",
        "- `canopy_only`: current canopy/status variables plus lagged relative canopy and lagged canopy change.",
        "- `oisst_only`: NOAA OISST thermal stress variables.",
        "- `canopy_noaa`: canopy, OISST, CUTI/BEUTI, one-hot encoded `region_group`, and spatial controls `center_lat`, `center_lon`.",
        "",
        "## Leakage Variables Excluded",
        "",
    ]
    lines.extend([f"- `{variable}`" for variable in sorted(LEAKAGE_VARIABLES)])
    lines.extend(
        [
            "",
            "## Models Compared",
            "",
            "- Logistic Regression",
            "- SVM",
            "- Random Forest",
            "- XGBoost",
            "- LightGBM",
            "",
            "## Metrics",
            "",
            "Validation and test metrics include PR-AUC, ROC-AUC, recall, precision, F1, accuracy, and confusion matrix counts. Final model comparison prioritizes test PR-AUC, recall, and F1.",
            "",
            "## Best Test Models",
            "",
            f"- Best by test PR-AUC: {best_pr['feature_set']} / {best_pr['model']} (PR-AUC={best_pr['pr_auc']:.4f}, Recall={best_pr['recall']:.4f}, F1={best_pr['f1']:.4f})",
            f"- Best by test Recall: {best_recall['feature_set']} / {best_recall['model']} (Recall={best_recall['recall']:.4f}, PR-AUC={best_recall['pr_auc']:.4f}, F1={best_recall['f1']:.4f})",
            f"- Best Canopy + NOAA PR-AUC: {combined_best:.4f}",
            f"- Best Canopy-only PR-AUC: {canopy_best:.4f}",
            f"- Canopy + NOAA improves over Canopy-only by best test PR-AUC: {improves}",
            "",
            "## Limitations",
            "",
            "- The number of spatial cells is small for generalizable machine-learning inference.",
            "- The final test period is limited to four years, 2021-2024.",
            "- NOAA variables are environmental exposure proxies, not direct ecological mechanisms.",
            "- OISST uses nearest valid ocean grid assignment in Version 1.",
            "- CUTI/BEUTI are latitude-bin proxies.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def write_notebook(path: Path) -> None:
    """Create a lightweight notebook wrapper for the model comparison script."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Initial Model Comparison\\n",
    "\\n",
    "This notebook runs the initial five-model comparison workflow for next-year kelp canopy decline prediction using the validated NOAA V1 modeling dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!python ../scripts/train_model_comparison.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "results = pd.read_csv('../outputs/metadata/model_comparison_test_metrics.csv')\\n",
    "results.sort_values(['pr_auc', 'recall', 'f1'], ascending=False)"
   ]
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
    """Run the model comparison workflow."""
    args = parse_args()
    data = main_subset(load_dataset(args.input))
    metrics, curves = train_and_evaluate(data)
    test_metrics = metrics.loc[metrics["split"] == "test"].copy()
    confusion = metrics[
        ["feature_set", "model", "split", "tn", "fp", "fn", "tp", "n_rows", "positive_rate"]
    ].copy()

    for path in [args.results_output, args.test_metrics_output, args.confusion_output]:
        path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.results_output, index=False)
    test_metrics.to_csv(args.test_metrics_output, index=False)
    confusion.to_csv(args.confusion_output, index=False)
    plot_performance(test_metrics, args.performance_figure)
    plot_curves(curves, args.pr_figure, args.roc_figure)
    write_report(metrics, data, args.report_output)
    write_notebook(Path("notebooks/04_model_comparison.ipynb"))

    best = test_metrics.sort_values(["pr_auc", "recall", "f1"], ascending=False).iloc[0]
    print("Model comparison complete.")
    print(f"Rows used: {len(data)}")
    print("Temporal split: train 1989-2016, validation 2017-2020, test 2021-2024")
    print(f"Best test PR-AUC: {best['feature_set']} / {best['model']} = {best['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
