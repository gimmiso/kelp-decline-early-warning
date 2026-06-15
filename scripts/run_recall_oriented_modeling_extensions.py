"""Run recall-oriented modeling extensions for kelp decline screening.

This workflow extends the original model comparison without replacing it. It
adds actionable decline labels, trajectory and environmental stress features,
unweighted versus cost-sensitive model variants, feature-set ablations, and
validation-based threshold tuning for recall-oriented screening.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier

from train_model_comparison import (
    INPUT_DATASET,
    preprocessor,
    predict_scores,
)
from tune_decision_thresholds import select_thresholds, threshold_metrics


warnings.filterwarnings("ignore", category=UserWarning)

TARGET_ORIGINAL = "decline_event_next"
TARGET_ACTIONABLE_LOW = "actionable_decline_low_next"
TARGET_ACTIONABLE_DROP = "actionable_decline_drop_next"
CANOPY = "relative_canopy"
NEXT_CANOPY = "next_year_relative_canopy"
BASELINE_P25 = "baseline_p25_relative_canopy_1984_2013"
EPSILON = 1e-6
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)

MODEL_RESULTS_DIR = Path("outputs/model_results")
DIAGNOSTIC_DIR = Path("outputs/diagnostics")
METADATA_DIR = Path("outputs/metadata")
FIGURE_DIR = Path("outputs/figures")

ACTIONABLE_LABEL_SUMMARY = DIAGNOSTIC_DIR / "actionable_decline_label_summary.csv"
TRAJECTORY_FEATURE_SUMMARY = METADATA_DIR / "trajectory_feature_summary.csv"
FEATURE_AVAILABILITY = METADATA_DIR / "feature_availability_report.csv"
COST_SENSITIVE_PERFORMANCE = MODEL_RESULTS_DIR / "cost_sensitive_model_performance.csv"
COST_SENSITIVE_SUMMARY = MODEL_RESULTS_DIR / "cost_sensitive_model_summary.csv"
ACTIONABLE_PERFORMANCE = MODEL_RESULTS_DIR / "actionable_decline_model_performance.csv"
FEATURE_ABLATION_PERFORMANCE = MODEL_RESULTS_DIR / "feature_ablation_performance.csv"
EXTENDED_THRESHOLD_GRID = MODEL_RESULTS_DIR / "extended_threshold_tuning_results.csv"
EXTENDED_THRESHOLD_SUMMARY = MODEL_RESULTS_DIR / "extended_threshold_selection_summary.csv"
REPORT_OUTPUT = DIAGNOSTIC_DIR / "recall_oriented_modeling_report.md"
THRESHOLD_FIGURE = FIGURE_DIR / "extended_precision_recall_threshold_curve.png"


@dataclass(frozen=True)
class ModelSpec:
    """Model metadata for original and cost-sensitive variants."""

    model_family: str
    model_variant: str
    estimator: object
    needs_scaling: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run recall-oriented modeling extensions.")
    parser.add_argument("--input", type=Path, default=INPUT_DATASET)
    return parser.parse_args()


def load_modeling_data(path: Path) -> pd.DataFrame:
    """Load the modeling dataset and keep the established complete-feature period."""
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path).sort_values(["cell_id", "year"]).reset_index(drop=True)
    data = data.loc[data["year"].between(1989, 2024)].copy()
    if len(data) != 1800:
        raise ValueError(f"Expected 1,800 rows for 1989-2024, found {len(data)}.")
    return data


def add_actionable_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Add actionable decline labels and proportional next-year canopy drop."""
    required = {CANOPY, NEXT_CANOPY, BASELINE_P25}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns for actionable labels: {missing}")
    labeled = data.copy()
    labeled["relative_drop_next"] = (
        labeled[CANOPY] - labeled[NEXT_CANOPY]
    ) / np.maximum(labeled[CANOPY], EPSILON)
    labeled["actionable_decline_low_next"] = (
        (labeled[CANOPY] > 0.05) & (labeled[NEXT_CANOPY] < labeled[BASELINE_P25])
    ).astype(int)
    labeled["actionable_decline_drop_next"] = (
        (labeled[CANOPY] > 0.05) & (labeled["relative_drop_next"] >= 0.30)
    ).astype(int)
    return labeled


def rolling_slope(values: np.ndarray) -> float:
    """Return the slope across a three-year canopy window."""
    if len(values) < 3 or np.isnan(values).any():
        return np.nan
    return float(np.polyfit(np.arange(len(values)), values, 1)[0])


def years_since_high(values: pd.Series) -> pd.Series:
    """Compute years since high canopy using only past/current observations."""
    result = []
    last_high_index: int | None = None
    history: list[float] = []
    for index, value in enumerate(values):
        history.append(float(value))
        threshold = float(pd.Series(history).quantile(0.75))
        if value >= threshold:
            last_high_index = index
            result.append(0)
        elif last_high_index is None:
            result.append(np.nan)
        else:
            result.append(index - last_high_index)
    return pd.Series(result, index=values.index)


def add_trajectory_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add leakage-safe canopy trajectory features by cell."""
    enhanced = data.sort_values(["cell_id", "year"]).copy()
    grouped = enhanced.groupby("cell_id", group_keys=False)
    enhanced["canopy_lag1"] = grouped[CANOPY].shift(1)
    enhanced["canopy_lag2"] = grouped[CANOPY].shift(2)
    enhanced["canopy_lag3"] = grouped[CANOPY].shift(3)
    enhanced["canopy_2yr_change"] = enhanced[CANOPY] - enhanced["canopy_lag2"]
    enhanced["canopy_3yr_change"] = enhanced[CANOPY] - enhanced["canopy_lag3"]
    enhanced["canopy_3yr_mean"] = grouped[CANOPY].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    enhanced["canopy_3yr_std"] = grouped[CANOPY].rolling(3, min_periods=2).std().reset_index(level=0, drop=True)
    enhanced["canopy_3yr_max"] = grouped[CANOPY].rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
    enhanced["canopy_3yr_slope"] = grouped[CANOPY].rolling(3, min_periods=3).apply(
        rolling_slope, raw=True
    ).reset_index(level=0, drop=True)
    enhanced["canopy_3yr_cv"] = enhanced["canopy_3yr_std"] / np.maximum(enhanced["canopy_3yr_mean"], EPSILON)
    enhanced["canopy_drop_from_3yr_max"] = (
        enhanced["canopy_3yr_max"] - enhanced[CANOPY]
    ) / np.maximum(enhanced["canopy_3yr_max"], EPSILON)
    enhanced["years_since_last_high_canopy"] = grouped[CANOPY].apply(years_since_high).reset_index(level=0, drop=True)

    trajectory_features = [
        "canopy_lag1",
        "canopy_lag2",
        "canopy_2yr_change",
        "canopy_3yr_change",
        "canopy_3yr_mean",
        "canopy_3yr_slope",
        "canopy_3yr_cv",
        "canopy_drop_from_3yr_max",
        "years_since_last_high_canopy",
    ]
    summary = feature_summary(enhanced, trajectory_features, "canopy_trajectory")
    return enhanced, summary


def add_environment_extensions(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add available environmental lag, stress, and interaction features."""
    enhanced = data.sort_values(["cell_id", "year"]).copy()
    grouped = enhanced.groupby("cell_id", group_keys=False)
    rows: list[dict[str, object]] = []

    def add_available(feature: str, status: str, sources: str, notes: str) -> None:
        rows.append({"feature": feature, "status": status, "source_columns": sources, "notes": notes})

    if "lag1_annual_mean_sst_anomaly" in enhanced.columns:
        enhanced["sst_anomaly_lag1"] = enhanced["lag1_annual_mean_sst_anomaly"]
        add_available("sst_anomaly_lag1", "created", "lag1_annual_mean_sst_anomaly", "Alias for existing lagged OISST anomaly.")
    else:
        add_available("sst_anomaly_lag1", "skipped", "lag1_annual_mean_sst_anomaly", "Required source column unavailable.")

    if "annual_mean_sst_anomaly" in enhanced.columns:
        enhanced["sst_anomaly_2yr_mean"] = grouped["annual_mean_sst_anomaly"].rolling(
            2, min_periods=1
        ).mean().reset_index(level=0, drop=True)
        add_available("sst_anomaly_2yr_mean", "created", "annual_mean_sst_anomaly", "Rolling current plus lag-1 SST anomaly mean.")
    else:
        add_available("sst_anomaly_2yr_mean", "skipped", "annual_mean_sst_anomaly", "Required source column unavailable.")

    if "lag1_hot_days_p90" in enhanced.columns:
        enhanced["marine_heatwave_days_lag1"] = enhanced["lag1_hot_days_p90"]
        add_available("marine_heatwave_days_lag1", "created", "lag1_hot_days_p90", "Uses lagged hot-day proxy already present.")
    else:
        add_available("marine_heatwave_days_lag1", "skipped", "lag1_hot_days_p90", "Required source column unavailable.")

    add_available(
        "marine_heatwave_intensity_lag1",
        "skipped",
        "not available",
        "Dataset has hot-day counts but no marine heatwave intensity variable.",
    )

    for target, source in [("cuti_anomaly_lag1", "lag1_cuti_anomaly"), ("beuti_anomaly_lag1", "lag1_beuti_anomaly")]:
        if source in enhanced.columns:
            enhanced[target] = enhanced[source]
            add_available(target, "created", source, "Alias for existing lagged upwelling proxy anomaly.")
        else:
            add_available(target, "skipped", source, "Required source column unavailable.")

    if {"annual_mean_sst_anomaly", "cuti_anomaly"}.issubset(enhanced.columns):
        enhanced["high_sst_low_upwelling"] = enhanced["annual_mean_sst_anomaly"] * (-enhanced["cuti_anomaly"])
        add_available("high_sst_low_upwelling", "created", "annual_mean_sst_anomaly; cuti_anomaly", "Continuous interaction; positive when warm anomaly coincides with low CUTI.")
    else:
        add_available("high_sst_low_upwelling", "skipped", "annual_mean_sst_anomaly; cuti_anomaly", "Required source columns unavailable.")

    if {"annual_mean_sst_anomaly", "beuti_anomaly"}.issubset(enhanced.columns):
        enhanced["high_sst_low_beuti"] = enhanced["annual_mean_sst_anomaly"] * (-enhanced["beuti_anomaly"])
        add_available("high_sst_low_beuti", "created", "annual_mean_sst_anomaly; beuti_anomaly", "Continuous interaction; positive when warm anomaly coincides with low BEUTI.")
    else:
        add_available("high_sst_low_beuti", "skipped", "annual_mean_sst_anomaly; beuti_anomaly", "Required source columns unavailable.")

    if {"canopy_3yr_slope", "annual_mean_sst_anomaly"}.issubset(enhanced.columns):
        declining_slope = (-enhanced["canopy_3yr_slope"]).clip(lower=0)
        enhanced["declining_canopy_slope_x_sst_anomaly"] = declining_slope * enhanced["annual_mean_sst_anomaly"]
        add_available("declining_canopy_slope_x_sst_anomaly", "created", "canopy_3yr_slope; annual_mean_sst_anomaly", "Interaction between recent canopy decline and SST anomaly.")
    else:
        add_available("declining_canopy_slope_x_sst_anomaly", "skipped", "canopy_3yr_slope; annual_mean_sst_anomaly", "Required source columns unavailable.")

    if {"canopy_3yr_slope", "hot_days_p90"}.issubset(enhanced.columns):
        declining_slope = (-enhanced["canopy_3yr_slope"]).clip(lower=0)
        enhanced["declining_canopy_slope_x_mhw_days"] = declining_slope * enhanced["hot_days_p90"]
        add_available("declining_canopy_slope_x_mhw_days", "created", "canopy_3yr_slope; hot_days_p90", "Interaction between recent canopy decline and hot-day exposure.")
    else:
        add_available("declining_canopy_slope_x_mhw_days", "skipped", "canopy_3yr_slope; hot_days_p90", "Required source columns unavailable.")

    return enhanced, pd.DataFrame(rows)


def feature_summary(data: pd.DataFrame, features: list[str], group: str) -> pd.DataFrame:
    """Create a compact feature coverage summary."""
    rows = []
    for feature in features:
        if feature not in data.columns:
            rows.append({"feature_group": group, "feature": feature, "status": "missing", "missing_count": np.nan, "non_missing_count": np.nan})
        else:
            rows.append(
                {
                    "feature_group": group,
                    "feature": feature,
                    "status": "created",
                    "missing_count": int(data[feature].isna().sum()),
                    "non_missing_count": int(data[feature].notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def positive_class_weight(y: pd.Series) -> float:
    """Return negative-to-positive ratio for cost-sensitive boosted trees."""
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    return negatives / positives if positives else 1.0


def model_specs(y_train: pd.Series) -> list[ModelSpec]:
    """Return unweighted and cost-sensitive model specifications."""
    weight = positive_class_weight(y_train)
    return [
        ModelSpec("Logistic Regression", "unweighted", LogisticRegression(max_iter=2000, solver="lbfgs"), True),
        ModelSpec("Logistic Regression", "cost_sensitive", LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs"), True),
        ModelSpec("SVM", "unweighted", SVC(probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=42), True),
        ModelSpec("SVM", "cost_sensitive", SVC(class_weight="balanced", probability=True, kernel="rbf", C=1.0, gamma="scale", random_state=42), True),
        ModelSpec("Random Forest", "unweighted", RandomForestClassifier(n_estimators=300, random_state=42, min_samples_leaf=3, n_jobs=-1), False),
        ModelSpec("Random Forest", "cost_sensitive", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample", min_samples_leaf=3, n_jobs=-1), False),
        ModelSpec("XGBoost", "unweighted", XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", scale_pos_weight=1.0, random_state=42, n_jobs=2), False),
        ModelSpec("XGBoost", "cost_sensitive", XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", scale_pos_weight=weight, random_state=42, n_jobs=2), False),
        ModelSpec("LightGBM", "unweighted", LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=15, scale_pos_weight=1.0, random_state=42, verbose=-1, n_jobs=2), False),
        ModelSpec("LightGBM", "cost_sensitive", LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=15, scale_pos_weight=weight, random_state=42, verbose=-1, n_jobs=2), False),
    ]


def available(features: list[str], data: pd.DataFrame) -> list[str]:
    """Return feature columns that exist in the dataset."""
    return [feature for feature in features if feature in data.columns]


def define_feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    """Define current, trajectory, environmental, and combined ablation sets."""
    current = available(
        [
            "relative_canopy",
            "kelp_area_m2",
            "count_cells_kelp",
            "count_cells_no_clouds",
            "count_cells_historic_footprint",
            "historical_footprint_area_m2",
        ],
        data,
    )
    trajectory = available(
        [
            "canopy_lag1",
            "canopy_lag2",
            "canopy_2yr_change",
            "canopy_3yr_change",
            "canopy_3yr_mean",
            "canopy_3yr_slope",
            "canopy_3yr_cv",
            "canopy_drop_from_3yr_max",
            "years_since_last_high_canopy",
        ],
        data,
    )
    environment = available(
        [
            "annual_mean_sst",
            "annual_max_sst",
            "annual_min_sst",
            "annual_sst_std",
            "annual_mean_sst_anomaly",
            "annual_max_sst_anomaly",
            "hot_days_p90",
            "hot_days_p95",
            "sst_anomaly_lag1",
            "sst_anomaly_2yr_mean",
            "marine_heatwave_days_lag1",
            "annual_mean_cuti",
            "spring_mean_cuti",
            "summer_mean_cuti",
            "cuti_anomaly",
            "cuti_anomaly_lag1",
            "annual_mean_beuti",
            "spring_mean_beuti",
            "summer_mean_beuti",
            "beuti_anomaly",
            "beuti_anomaly_lag1",
            "high_sst_low_upwelling",
            "high_sst_low_beuti",
            "declining_canopy_slope_x_sst_anomaly",
            "declining_canopy_slope_x_mhw_days",
        ],
        data,
    )
    sets = {
        "canopy_current_only": current,
        "canopy_trajectory_only": trajectory,
        "canopy_current_plus_trajectory": current + trajectory,
        "environment_only": environment,
        "canopy_plus_environment": current + environment,
        "canopy_plus_trajectory_plus_environment": current + trajectory + environment,
    }
    return {name: list(dict.fromkeys(features)) for name, features in sets.items() if features}


def split_data(data: pd.DataFrame, target: str) -> dict[str, pd.DataFrame]:
    """Apply the established temporal split for one target."""
    splits = {
        "train": data.loc[data["year"].between(1989, 2016)].copy(),
        "validation": data.loc[data["year"].between(2017, 2020)].copy(),
        "test": data.loc[data["year"].between(2021, 2024)].copy(),
    }
    for split_name, split in splits.items():
        classes = set(split[target].dropna().astype(int).unique())
        if classes != {0, 1}:
            raise ValueError(f"{target} {split_name} split must contain both classes; found {classes}.")
    return splits


def score_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    """Compute thresholded and ranking metrics."""
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "f2": fbeta_score(y_true, predictions, beta=2, zero_division=0),
        "pr_auc": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores),
        "accuracy": accuracy_score(y_true, predictions),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def threshold_grid_rows(
    label: str,
    feature_set: str,
    spec: ModelSpec,
    y_validation: pd.Series,
    validation_scores: np.ndarray,
) -> pd.DataFrame:
    """Create validation threshold grid rows for one fitted model."""
    rows = []
    for threshold in THRESHOLDS:
        row = threshold_metrics(y_validation, validation_scores, float(threshold))
        row.update(
            {
                "target": label,
                "feature_set": feature_set,
                "model_family": spec.model_family,
                "model_variant": spec.model_variant,
                "model": f"{spec.model_family} ({spec.model_variant})",
                "selection_split": "validation",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def apply_selected_thresholds(
    selected: pd.DataFrame,
    test_lookup: dict[tuple[str, str, str, str], dict[str, object]],
) -> pd.DataFrame:
    """Apply validation-selected thresholds to held-out test scores."""
    rows = []
    for _, choice in selected.iterrows():
        key = (choice["target"], choice["feature_set"], choice["model_family"], choice["model_variant"])
        values = test_lookup[key]
        y_test = values["y_test"]
        scores = values["test_scores"]
        threshold = float(choice["selected_threshold"])
        row = threshold_metrics(y_test, scores, threshold)
        rows.append(
            {
                "target": choice["target"],
                "feature_set": choice["feature_set"],
                "model_family": choice["model_family"],
                "model_variant": choice["model_variant"],
                "model": choice["model"],
                "selection_rule": choice["selection_rule"],
                "threshold": threshold,
                "precision_floor_target": choice["precision_floor_target"],
                "precision_floor_used": choice["precision_floor_used"],
                "precision_floor_fallback_used": choice["precision_floor_fallback_used"],
                "validation_precision": choice["validation_precision"],
                "validation_recall": choice["validation_recall"],
                "validation_f1": choice["validation_f1"],
                "validation_f2": choice["validation_f2"],
                "test_precision": row["precision"],
                "test_recall": row["recall"],
                "test_f1": row["f1"],
                "test_f2": row["f2"],
                "test_pr_auc": average_precision_score(y_test, scores),
                "test_roc_auc": roc_auc_score(y_test, scores),
                "test_false_positives": row["false_positives"],
                "test_false_negatives": row["false_negatives"],
                "test_true_positives": row["true_positives"],
                "test_true_negatives": row["true_negatives"],
                "test_predicted_positive_rate": row["predicted_positive_rate"],
            }
        )
    return pd.DataFrame(rows)


def run_models(data: pd.DataFrame, targets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train all feature-set/model/target combinations and return metrics plus threshold tables."""
    feature_sets = define_feature_sets(data)
    performance_rows = []
    threshold_grids = []
    test_lookup: dict[tuple[str, str, str, str], dict[str, object]] = {}

    for target in targets:
        splits = split_data(data, target)
        y_train = splits["train"][target].astype(int)
        specs = model_specs(y_train)
        for feature_set_name, features in feature_sets.items():
            for spec in specs:
                pipe = Pipeline(
                    [
                        ("preprocess", preprocessor(splits["train"], features, scale=spec.needs_scaling)),
                        ("model", spec.estimator),
                    ]
                )
                pipe.fit(splits["train"][features], y_train)
                y_validation = splits["validation"][target].astype(int)
                validation_scores = predict_scores(pipe, splits["validation"][features])
                threshold_grids.append(
                    threshold_grid_rows(target, feature_set_name, spec, y_validation, validation_scores)
                )

                y_test = splits["test"][target].astype(int)
                test_scores = predict_scores(pipe, splits["test"][features])
                test_lookup[(target, feature_set_name, spec.model_family, spec.model_variant)] = {
                    "y_test": y_test,
                    "test_scores": test_scores,
                }
                metrics = score_metrics(y_test, test_scores)
                performance_rows.append(
                    {
                        "target": target,
                        "feature_set": feature_set_name,
                        "model_family": spec.model_family,
                        "model_variant": spec.model_variant,
                        "model": f"{spec.model_family} ({spec.model_variant})",
                        "split": "test",
                        "n_observations": len(y_test),
                        "n_positive_events": int(y_test.sum()),
                        "event_prevalence": float(y_test.mean()),
                        **metrics,
                    }
                )

    performance = pd.DataFrame(performance_rows)
    threshold_grid = pd.concat(threshold_grids, ignore_index=True)
    group_columns = ["target", "feature_set", "model_family", "model_variant", "model"]
    selected_rows = []
    for key, group in threshold_grid.groupby(group_columns, sort=False):
        target, feature_set, model_family, model_variant, model = key
        base = select_thresholds(group)
        base["target"] = target
        base["feature_set"] = feature_set
        base["model_family"] = model_family
        base["model_variant"] = model_variant
        base["model"] = model
        selected_rows.append(base)
    selected = pd.concat(selected_rows, ignore_index=True)
    threshold_summary = apply_selected_thresholds(selected, test_lookup)
    return performance, threshold_grid, threshold_summary


def summarize_labels(data: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Summarize target label distributions by split."""
    rows = []
    split_masks = {
        "train": data["year"].between(1989, 2016),
        "validation": data["year"].between(2017, 2020),
        "test": data["year"].between(2021, 2024),
        "all_modeling_years": data["year"].between(1989, 2024),
    }
    for target in targets:
        for split_name, mask in split_masks.items():
            subset = data.loc[mask, target].astype(int)
            rows.append(
                {
                    "target": target,
                    "split": split_name,
                    "n_observations": len(subset),
                    "n_positive_events": int(subset.sum()),
                    "event_prevalence": float(subset.mean()),
                }
            )
    return pd.DataFrame(rows)


def cost_sensitive_summary(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare unweighted and cost-sensitive variants for the original label."""
    original = performance.loc[performance["target"] == TARGET_ORIGINAL].copy()
    pairs = original.pivot_table(
        index=["target", "feature_set", "model_family"],
        columns="model_variant",
        values=["precision", "recall", "f1", "f2", "pr_auc", "roc_auc", "false_positives", "false_negatives"],
        aggfunc="first",
    )
    pairs.columns = [f"{metric}_{variant}" for metric, variant in pairs.columns]
    pairs = pairs.reset_index()
    for metric in ["precision", "recall", "f1", "f2", "pr_auc", "roc_auc"]:
        pairs[f"delta_{metric}_cost_sensitive_minus_unweighted"] = (
            pairs[f"{metric}_cost_sensitive"] - pairs[f"{metric}_unweighted"]
        )
    pairs["delta_false_negatives_cost_sensitive_minus_unweighted"] = (
        pairs["false_negatives_cost_sensitive"] - pairs["false_negatives_unweighted"]
    )
    pairs["delta_false_positives_cost_sensitive_minus_unweighted"] = (
        pairs["false_positives_cost_sensitive"] - pairs["false_positives_unweighted"]
    )
    return pairs


def plot_threshold_examples(summary: pd.DataFrame, output: Path) -> None:
    """Plot selected extended threshold results for headline models."""
    examples = summary.loc[
        (
            (summary["target"] == TARGET_ORIGINAL)
            & (summary["feature_set"].isin(["canopy_current_only", "canopy_plus_trajectory_plus_environment"]))
            & (summary["model_family"].isin(["Random Forest", "SVM"]))
            & (summary["model_variant"].isin(["unweighted", "cost_sensitive"]))
        )
    ].copy()
    examples = examples.loc[examples["selection_rule"].isin(["default_0.5", "max_f2", "max_recall_precision_ge_0.65"])]
    if examples.empty:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    examples["label"] = (
        examples["feature_set"]
        + " / "
        + examples["model_family"]
        + " / "
        + examples["model_variant"]
        + " / "
        + examples["selection_rule"]
    )
    examples = examples.sort_values(["test_f2", "test_recall"], ascending=False).head(16)
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(examples))
    width = 0.28
    ax.bar(x - width, examples["test_precision"], width, label="Precision")
    ax.bar(x, examples["test_recall"], width, label="Recall")
    ax.bar(x + width, examples["test_f2"], width, label="F2")
    ax.set_xticks(x)
    ax.set_xticklabels(examples["label"], rotation=75, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Held-out test score")
    ax.set_title("Extended threshold tuning examples, test period 2021-2024")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def dataframe_to_markdown(data: pd.DataFrame, float_digits: int = 3) -> str:
    """Convert a small DataFrame to Markdown without optional dependencies."""
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


def write_report(performance: pd.DataFrame, threshold_summary: pd.DataFrame, output: Path) -> None:
    """Write a concise interpretation report for the extension."""
    best_by_target = (
        performance.sort_values(["target", "f2", "recall", "precision", "pr_auc"], ascending=[True, False, False, False, False])
        .groupby("target", sort=False)
        .head(1)
    )
    best_ablation = (
        performance.loc[performance["target"] == TARGET_ORIGINAL]
        .sort_values(["feature_set", "f2", "recall", "precision"], ascending=[True, False, False, False])
        .groupby("feature_set", sort=False)
        .head(1)
    )
    best_threshold = threshold_summary.sort_values(
        ["test_f2", "test_recall", "test_precision", "test_pr_auc"], ascending=False
    ).iloc[0]
    balanced_candidates = threshold_summary.loc[
        (threshold_summary["selection_rule"] == "max_recall_precision_ge_0.65")
        & (threshold_summary["precision_floor_fallback_used"] == False)
    ].copy()
    balanced = balanced_candidates.sort_values(["test_f2", "test_recall", "test_precision"], ascending=False).iloc[0]
    actionable_drop_threshold = threshold_summary.loc[
        threshold_summary["target"] == TARGET_ACTIONABLE_DROP
    ].sort_values(["test_f2", "test_recall", "test_precision"], ascending=False).iloc[0]
    actionable_low_threshold = threshold_summary.loc[
        threshold_summary["target"] == TARGET_ACTIONABLE_LOW
    ].sort_values(["test_f2", "test_recall", "test_precision"], ascending=False).iloc[0]
    current_best = performance.loc[
        (performance["target"] == TARGET_ORIGINAL)
        & (performance["feature_set"] == "canopy_current_only")
    ].sort_values(["f2", "recall", "precision"], ascending=False).iloc[0]
    trajectory_best = performance.loc[
        (performance["target"] == TARGET_ORIGINAL)
        & (performance["feature_set"] == "canopy_trajectory_only")
    ].sort_values(["f2", "recall", "precision"], ascending=False).iloc[0]

    lines = [
        "# Recall-Oriented Modeling Extension Report",
        "",
        "## Purpose",
        "",
        "This extension tests whether recall-oriented early-warning screening can be strengthened beyond current canopy-state persistence by adding cost-sensitive models, actionable decline labels, canopy trajectory features, environmental stress interactions, feature-set ablations, and validation-based threshold tuning.",
        "",
        "## Best Test Models by Target",
        "",
        dataframe_to_markdown(
            best_by_target[
                [
                    "target",
                    "feature_set",
                    "model_family",
                    "model_variant",
                    "precision",
                    "recall",
                    "f1",
                    "f2",
                    "pr_auc",
                    "false_negatives",
                    "false_positives",
                ]
            ]
        ),
        "",
        "## Feature Ablation Snapshot",
        "",
        dataframe_to_markdown(
            best_ablation[
                [
                    "feature_set",
                    "model_family",
                    "model_variant",
                    "precision",
                    "recall",
                    "f1",
                    "f2",
                    "pr_auc",
                    "false_negatives",
                    "false_positives",
                ]
            ]
        ),
        "",
        "## Threshold-Tuning Recommendations",
        "",
        f"- Highest F2 screening result: `{best_threshold['target']} / {best_threshold['feature_set']} / {best_threshold['model_family']} ({best_threshold['model_variant']})` using `{best_threshold['selection_rule']}` at threshold {best_threshold['threshold']:.2f}.",
        f"- Recommended precision-floor balanced result: `{balanced['target']} / {balanced['feature_set']} / {balanced['model_family']} ({balanced['model_variant']})` using threshold {balanced['threshold']:.2f} without precision-floor fallback.",
        f"- Best actionable low-canopy threshold result: `{actionable_low_threshold['feature_set']} / {actionable_low_threshold['model_family']} ({actionable_low_threshold['model_variant']})` at threshold {actionable_low_threshold['threshold']:.2f}.",
        f"- Best actionable drop threshold result: `{actionable_drop_threshold['feature_set']} / {actionable_drop_threshold['model_family']} ({actionable_drop_threshold['model_variant']})` at threshold {actionable_drop_threshold['threshold']:.2f}.",
        "",
        "## Persistence-Adjusted Feature Interpretation",
        "",
        f"- Original-label current-canopy best F2: {current_best['f2']:.3f} (`{current_best['model_family']} / {current_best['model_variant']}`).",
        f"- Original-label trajectory-only best F2: {trajectory_best['f2']:.3f} (`{trajectory_best['model_family']} / {trajectory_best['model_variant']}`).",
        "- For the original decline label, trajectory-only and trajectory-augmented feature sets did not outperform current-canopy-only models, so full-sample performance still appears strongly tied to canopy-state persistence.",
        "- For `actionable_decline_drop_next`, trajectory features were more useful: the strongest default-threshold model used `canopy_trajectory_only`, and the strongest threshold-tuned model used `canopy_current_plus_trajectory`.",
        "",
        "## Interpretation",
        "",
        "Cost-sensitive learning and threshold tuning improve recall-oriented operating points, but they should be interpreted as screening configurations rather than operational proof. Actionable decline labels are more relevant for early warning because they focus on currently observable canopy entering a low or sharply declining state. The extension strengthens the early-warning story for actionable canopy-drop screening, but the original decline label still shows strong canopy-state persistence.",
        "",
        "## Output Files",
        "",
        "- `outputs/model_results/cost_sensitive_model_performance.csv`",
        "- `outputs/model_results/cost_sensitive_model_summary.csv`",
        "- `outputs/model_results/actionable_decline_model_performance.csv`",
        "- `outputs/model_results/feature_ablation_performance.csv`",
        "- `outputs/model_results/extended_threshold_tuning_results.csv`",
        "- `outputs/model_results/extended_threshold_selection_summary.csv`",
        "- `outputs/diagnostics/actionable_decline_label_summary.csv`",
        "- `outputs/metadata/trajectory_feature_summary.csv`",
        "- `outputs/metadata/feature_availability_report.csv`",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the recall-oriented modeling extension."""
    args = parse_args()
    for directory in [MODEL_RESULTS_DIR, DIAGNOSTIC_DIR, METADATA_DIR, FIGURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    data = load_modeling_data(args.input)
    data = add_actionable_labels(data)
    data, trajectory_summary = add_trajectory_features(data)
    data, availability_report = add_environment_extensions(data)
    targets = [TARGET_ORIGINAL, TARGET_ACTIONABLE_LOW, TARGET_ACTIONABLE_DROP]

    label_summary = summarize_labels(data, targets)
    performance, threshold_grid, threshold_summary = run_models(data, targets)
    cost_performance = performance.loc[performance["target"] == TARGET_ORIGINAL].copy()
    cost_summary = cost_sensitive_summary(performance)
    actionable_performance = performance.loc[performance["target"].isin([TARGET_ACTIONABLE_LOW, TARGET_ACTIONABLE_DROP])].copy()

    label_summary.to_csv(ACTIONABLE_LABEL_SUMMARY, index=False)
    trajectory_summary.to_csv(TRAJECTORY_FEATURE_SUMMARY, index=False)
    availability_report.to_csv(FEATURE_AVAILABILITY, index=False)
    cost_performance.to_csv(COST_SENSITIVE_PERFORMANCE, index=False)
    cost_summary.to_csv(COST_SENSITIVE_SUMMARY, index=False)
    actionable_performance.to_csv(ACTIONABLE_PERFORMANCE, index=False)
    performance.to_csv(FEATURE_ABLATION_PERFORMANCE, index=False)
    threshold_grid.to_csv(EXTENDED_THRESHOLD_GRID, index=False)
    threshold_summary.to_csv(EXTENDED_THRESHOLD_SUMMARY, index=False)
    plot_threshold_examples(threshold_summary, THRESHOLD_FIGURE)
    write_report(performance, threshold_summary, REPORT_OUTPUT)

    best = threshold_summary.sort_values(["test_f2", "test_recall", "test_precision"], ascending=False).iloc[0]
    print("Recall-oriented modeling extension complete.")
    print(f"Targets evaluated: {', '.join(targets)}")
    print(f"Feature sets evaluated: {performance['feature_set'].nunique()}")
    print(f"Model variants evaluated: {performance[['model_family', 'model_variant']].drop_duplicates().shape[0]}")
    print(
        "Best threshold-tuned F2 result: "
        f"{best['target']} / {best['feature_set']} / {best['model_family']} "
        f"({best['model_variant']}) at threshold {best['threshold']:.2f}, "
        f"recall {best['test_recall']:.3f}, precision {best['test_precision']:.3f}, F2 {best['test_f2']:.3f}"
    )


if __name__ == "__main__":
    main()
