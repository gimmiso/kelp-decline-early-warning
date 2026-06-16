"""Build leakage-safe canopy trajectory and instability proxy features.

This diagnostic extension treats recent canopy dynamics as a persistence and
time-series instability layer, not as an environmental driver layer. All
features for year t use only canopy observations from year t and earlier.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from run_recall_oriented_modeling_extensions import (
    BASELINE_P25,
    CANOPY,
    NEXT_CANOPY,
    TARGET_ACTIONABLE_DROP,
    add_actionable_labels,
)
from train_model_comparison import CANOPY_FEATURES, INPUT_DATASET, main_subset


DEFAULT_FEATURE_OUTPUT = Path("data/processed/canopy_trajectory_features.csv")
FEATURE_DIAGNOSTICS_OUTPUT = Path("results/tables/canopy_trajectory_feature_diagnostics.csv")
MODEL_COMPARISON_OUTPUT = Path("results/tables/canopy_trajectory_model_comparison.csv")
REPORT_OUTPUT = Path("outputs/diagnostics/canopy_trajectory_feature_report.md")

CRW_FEATURE_PATH = Path("data/processed/crw5km_composite_features.csv")
HABITAT_FEATURE_PATH = Path("data/processed/bathymetry_habitat_features.csv")

MODEL_START_YEAR = 1989
TRAIN_END_YEAR = 2016
VALIDATION_START_YEAR = 2017
VALIDATION_END_YEAR = 2020
TEST_START_YEAR = 2021
TEST_END_YEAR = 2024
BASELINE_START_YEAR = 1984
BASELINE_END_YEAR = 2013
EPSILON = 1e-6
TARGET_NEW_DECLINE = "new_decline_event_next"
TARGET_AT_RISK = "decline_event_next_at_risk_gt005"

TRAJECTORY_FEATURES = [
    "canopy_current_t",
    "canopy_lag1",
    "canopy_lag2",
    "canopy_3yr_mean_t",
    "canopy_5yr_mean_t",
    "canopy_3yr_slope_t",
    "canopy_5yr_slope_t",
    "canopy_cv_5yr_t",
    "years_since_last_high_canopy_t",
    "years_since_last_low_canopy_t",
    "recent_decline_rate_3yr_t",
    "recent_recovery_rate_3yr_t",
    "instability_score_5yr_t",
    "presence_frequency_5yr_t",
]

CRW_COMPOSITE_FEATURES = [
    "annual_mean_sst_crw5km",
    "spring_mean_sst_crw5km",
    "summer_mean_sst_crw5km",
    "warmest_month_mean_sst_crw5km",
    "annual_mean_ssta_crw5km",
    "spring_ssta_crw5km",
    "summer_ssta_crw5km",
    "annual_max_monthly_ssta_crw5km",
    "lag1_annual_mean_sst_crw5km",
    "lag1_annual_mean_ssta_crw5km",
]

HABITAT_FEATURES = [
    "mean_depth_m",
    "min_depth_m",
    "max_depth_m",
    "depth_range_m",
    "shallow_area_share_0_30m",
    "shallow_area_share_0_50m",
    "slope_mean",
    "slope_std",
    "n_bathymetry_pixels_used",
    "ocean_pixel_share",
    "bathymetry_missing_rate",
]


@dataclass(frozen=True)
class TargetSpec:
    """Target and optional row filter for model comparison."""

    name: str
    target: str
    filter_column: str | None = None


TARGETS = [
    TargetSpec("original_decline", "decline_event_next"),
    TargetSpec("at_risk_original_gt005", TARGET_AT_RISK, "at_risk_gt005"),
    TargetSpec("new_decline_transition", TARGET_NEW_DECLINE),
    TargetSpec("actionable_decline_drop", TARGET_ACTIONABLE_DROP),
    TargetSpec("high_canopy_original_decline", "decline_event_next", "high_canopy_t"),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build canopy trajectory and instability proxy features.")
    parser.add_argument("--input", type=Path, default=INPUT_DATASET)
    parser.add_argument("--feature-output", type=Path, default=DEFAULT_FEATURE_OUTPUT)
    parser.add_argument("--feature-diagnostics-output", type=Path, default=FEATURE_DIAGNOSTICS_OUTPUT)
    parser.add_argument("--model-comparison-output", type=Path, default=MODEL_COMPARISON_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    return parser.parse_args()


def load_full_modeling_data(input_path: Path) -> pd.DataFrame:
    """Load the full canopy panel and add labels needed for evaluation."""
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    data = pd.read_csv(input_path).sort_values(["cell_id", "year"]).reset_index(drop=True)
    data = add_actionable_labels(data)
    data[TARGET_NEW_DECLINE] = ((data[CANOPY] >= data[BASELINE_P25]) & (data[NEXT_CANOPY] < data[BASELINE_P25])).astype(int)
    data[TARGET_AT_RISK] = data["decline_event_next"].astype(int)
    data["at_risk_gt005"] = data[CANOPY] > 0.05

    baseline = data.loc[data["year"].between(BASELINE_START_YEAR, BASELINE_END_YEAR)]
    p50 = baseline.groupby("cell_id")[CANOPY].quantile(0.50)
    p75 = baseline.groupby("cell_id")[CANOPY].quantile(0.75)
    data["baseline_p50_relative_canopy_1984_2013"] = data["cell_id"].map(p50)
    data["baseline_p75_relative_canopy_1984_2013"] = data["cell_id"].map(p75)
    data["high_canopy_t"] = data[CANOPY] >= data["baseline_p75_relative_canopy_1984_2013"]
    return data


def slope(values: np.ndarray) -> float:
    """Return the linear slope across a complete rolling window."""
    if len(values) < 2 or np.isnan(values).any():
        return np.nan
    return float(np.polyfit(np.arange(len(values)), values, 1)[0])


def expanding_years_since(values: pd.Series, quantile: float, mode: str) -> pd.Series:
    """Compute years since last expanding high or low canopy event."""
    years = values.index.to_numpy()
    result: list[float] = []
    history: list[float] = []
    last_event_year: int | None = None
    for year, value in zip(years, values.to_numpy(dtype=float), strict=False):
        history.append(float(value))
        threshold = float(pd.Series(history).quantile(quantile))
        if mode == "high":
            event = value >= threshold
        elif mode == "low":
            event = value <= threshold
        else:
            raise ValueError(mode)
        if event:
            last_event_year = int(year)
            result.append(0.0)
        elif last_event_year is None:
            result.append(np.nan)
        else:
            result.append(float(int(year) - last_event_year))
    return pd.Series(result, index=values.index)


def add_canopy_trajectory_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create trajectory features using only current and past canopy observations."""
    output = data.sort_values(["cell_id", "year"]).copy()
    grouped = output.groupby("cell_id", group_keys=False)
    output["canopy_current_t"] = output[CANOPY]
    output["canopy_lag1"] = grouped[CANOPY].shift(1)
    output["canopy_lag2"] = grouped[CANOPY].shift(2)
    output["canopy_3yr_mean_t"] = grouped[CANOPY].rolling(3, min_periods=3).mean().reset_index(level=0, drop=True)
    output["canopy_5yr_mean_t"] = grouped[CANOPY].rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    output["canopy_3yr_slope_t"] = grouped[CANOPY].rolling(3, min_periods=3).apply(slope, raw=True).reset_index(level=0, drop=True)
    output["canopy_5yr_slope_t"] = grouped[CANOPY].rolling(5, min_periods=5).apply(slope, raw=True).reset_index(level=0, drop=True)
    canopy_5yr_std = grouped[CANOPY].rolling(5, min_periods=5).std().reset_index(level=0, drop=True)
    output["canopy_cv_5yr_t"] = canopy_5yr_std / np.maximum(output["canopy_5yr_mean_t"], EPSILON)

    year_indexed = output.set_index("year")
    high = year_indexed.groupby("cell_id", group_keys=False)[CANOPY].apply(lambda series: expanding_years_since(series, 0.75, "high"))
    low = year_indexed.groupby("cell_id", group_keys=False)[CANOPY].apply(lambda series: expanding_years_since(series, 0.25, "low"))
    output["years_since_last_high_canopy_t"] = high.to_numpy()
    output["years_since_last_low_canopy_t"] = low.to_numpy()

    start_3yr = grouped[CANOPY].shift(2)
    output["recent_decline_rate_3yr_t"] = ((start_3yr - output[CANOPY]) / np.maximum(start_3yr, EPSILON)).clip(lower=0)
    output["recent_recovery_rate_3yr_t"] = ((output[CANOPY] - start_3yr) / np.maximum(start_3yr, EPSILON)).clip(lower=0)
    output["instability_score_5yr_t"] = (
        grouped[CANOPY]
        .rolling(5, min_periods=5)
        .apply(lambda values: float(np.mean(np.abs(np.diff(values)))) if len(values) == 5 else np.nan, raw=True)
        .reset_index(level=0, drop=True)
    )
    output["presence_frequency_5yr_t"] = (
        grouped[CANOPY]
        .rolling(5, min_periods=5)
        .apply(lambda values: float(np.mean(values > 0.01)) if len(values) == 5 else np.nan, raw=True)
        .reset_index(level=0, drop=True)
    )

    # Source-year audit: all features are computed from windows ending at the row year.
    output["trajectory_max_source_year_used"] = output["year"]
    output["trajectory_leakage_flag"] = output["trajectory_max_source_year_used"] > output["year"]
    return output


def feature_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Return the model-period trajectory feature table."""
    columns = ["cell_id", "year", *TRAJECTORY_FEATURES, "trajectory_max_source_year_used", "trajectory_leakage_flag"]
    return data.loc[data["year"].between(MODEL_START_YEAR, TEST_END_YEAR), columns].reset_index(drop=True)


def add_optional_features(data: pd.DataFrame) -> pd.DataFrame:
    """Merge CRW composite and habitat features when local processed files exist."""
    output = data.copy()
    if CRW_FEATURE_PATH.exists():
        crw = pd.read_csv(CRW_FEATURE_PATH)
        output = output.merge(crw, on=["cell_id", "year"], how="left", validate="one_to_one")
    if HABITAT_FEATURE_PATH.exists():
        habitat = pd.read_csv(HABITAT_FEATURE_PATH)
        habitat = habitat.drop(columns=[column for column in ["feature_status"] if column in habitat.columns])
        output = output.merge(habitat, on="cell_id", how="left", validate="many_to_one")
    return output


def available(columns: list[str], data: pd.DataFrame) -> list[str]:
    """Return columns available in the merged dataset."""
    return [column for column in columns if column in data.columns]


def model_feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    """Define trajectory, existing, optional, and combined feature families."""
    sets = {
        "existing_canopy_only": available(CANOPY_FEATURES, data),
        "canopy_trajectory_only": available(TRAJECTORY_FEATURES, data),
    }
    crw = available(CRW_COMPOSITE_FEATURES, data)
    habitat = available(HABITAT_FEATURES, data)
    if crw:
        sets["crw_composite_only"] = crw
        sets["canopy_trajectory_plus_crw"] = available(TRAJECTORY_FEATURES + CRW_COMPOSITE_FEATURES, data)
    if habitat:
        sets["habitat_only"] = habitat
        sets["canopy_trajectory_plus_habitat"] = available(TRAJECTORY_FEATURES + HABITAT_FEATURES, data)
    if crw and habitat:
        sets["canopy_trajectory_plus_crw_plus_habitat"] = available(
            TRAJECTORY_FEATURES + CRW_COMPOSITE_FEATURES + HABITAT_FEATURES, data
        )
    return {name: features for name, features in sets.items() if features}


def has_two_classes(frame: pd.DataFrame, target: str) -> bool:
    """Return whether both classes are present."""
    return set(frame[target].dropna().astype(int).unique()) == {0, 1}


def preprocess(features: list[str]) -> ColumnTransformer:
    """Create numeric preprocessing for model pipelines."""
    return ColumnTransformer(
        transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), features)]
    )


def validation_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    """Select a validation threshold that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(y_true.astype(int), scores)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metric_row(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Compute metrics and confusion counts."""
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) == 2 else np.nan,
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def evaluate_naive_persistence(working: pd.DataFrame, target: TargetSpec) -> dict[str, object]:
    """Evaluate a current-low-canopy persistence risk score."""
    validation = working.loc[working["year"].between(VALIDATION_START_YEAR, VALIDATION_END_YEAR)].copy()
    test = working.loc[working["year"].between(TEST_START_YEAR, TEST_END_YEAR)].copy()
    train = working.loc[working["year"].between(MODEL_START_YEAR, TRAIN_END_YEAR)].copy()
    if validation.empty or test.empty or not has_two_classes(validation, target.target) or not has_two_classes(test, target.target):
        return {
            "target_definition": target.name,
            "feature_family": "naive_persistence_baseline",
            "model": "current_low_canopy_score",
            "status": "insufficient_target_classes",
            "n_train": len(train),
            "n_validation": len(validation),
            "n_test": len(test),
        }
    validation_scores = 1.0 - validation[CANOPY].to_numpy(dtype=float)
    test_scores = 1.0 - test[CANOPY].to_numpy(dtype=float)
    threshold = validation_threshold(validation[target.target], validation_scores)
    row = metric_row(test[target.target].astype(int), test_scores, threshold)
    row.update(
        {
            "target_definition": target.name,
            "feature_family": "naive_persistence_baseline",
            "model": "current_low_canopy_score",
            "n_train": len(train),
            "n_validation": len(validation),
            "n_test": len(test),
            "positive_events_test": int(test[target.target].sum()),
            "event_prevalence_test": float(test[target.target].mean()),
            "decision_threshold": threshold,
            "status": "computed",
        }
    )
    return row


def evaluate_models(data: pd.DataFrame) -> pd.DataFrame:
    """Evaluate naive, trajectory, and optional CRW/habitat feature families."""
    sets = model_feature_sets(data)
    estimators = {
        "Logistic Regression L2": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=3,
            n_jobs=-1,
        ),
    }
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        working = data.copy()
        if target.filter_column:
            working = working.loc[working[target.filter_column]].copy()
        rows.append(evaluate_naive_persistence(working, target))
        train = working.loc[working["year"].between(MODEL_START_YEAR, TRAIN_END_YEAR)].copy()
        validation = working.loc[working["year"].between(VALIDATION_START_YEAR, VALIDATION_END_YEAR)].copy()
        test = working.loc[working["year"].between(TEST_START_YEAR, TEST_END_YEAR)].copy()
        if train.empty or validation.empty or test.empty or not has_two_classes(train, target.target) or not has_two_classes(validation, target.target) or not has_two_classes(test, target.target):
            rows.append(
                {
                    "target_definition": target.name,
                    "feature_family": "all",
                    "model": "not_run",
                    "n_train": len(train),
                    "n_validation": len(validation),
                    "n_test": len(test),
                    "status": "insufficient_target_classes",
                }
            )
            continue
        for family, features in sets.items():
            for model_name, estimator in estimators.items():
                pipeline = Pipeline([("preprocess", preprocess(features)), ("model", estimator)])
                pipeline.fit(train[features], train[target.target].astype(int))
                validation_scores = pipeline.predict_proba(validation[features])[:, 1]
                test_scores = pipeline.predict_proba(test[features])[:, 1]
                threshold = validation_threshold(validation[target.target], validation_scores)
                row = metric_row(test[target.target].astype(int), test_scores, threshold)
                row.update(
                    {
                        "target_definition": target.name,
                        "feature_family": family,
                        "model": model_name,
                        "n_train": len(train),
                        "n_validation": len(validation),
                        "n_test": len(test),
                        "positive_events_test": int(test[target.target].sum()),
                        "event_prevalence_test": float(test[target.target].mean()),
                        "decision_threshold": threshold,
                        "status": "computed",
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def diagnostics_table(features: pd.DataFrame, full_features: pd.DataFrame) -> pd.DataFrame:
    """Create feature missingness, source-year, and leakage-audit diagnostics."""
    rows: list[dict[str, object]] = []
    for feature in TRAJECTORY_FEATURES:
        series = features[feature]
        rows.append(
            {
                "diagnostic": "feature_missingness",
                "feature": feature,
                "value": float(series.isna().mean()),
                "status": "computed",
                "notes": "Missingness across 1989-2024 model-period cell-year rows.",
            }
        )
        valid_years = features.loc[series.notna(), "year"]
        rows.append(
            {
                "diagnostic": "first_valid_year",
                "feature": feature,
                "value": int(valid_years.min()) if not valid_years.empty else np.nan,
                "status": "computed",
                "notes": "First model-period year with a non-missing value for this feature.",
            }
        )
    leakage_count = int(features["trajectory_leakage_flag"].sum())
    rows.extend(
        [
            {
                "diagnostic": "leakage_audit",
                "feature": "trajectory_max_source_year_used",
                "value": leakage_count,
                "status": "passed" if leakage_count == 0 else "failed",
                "notes": "Rows where max source year used exceeded the row year.",
            },
            {
                "diagnostic": "feature_rows",
                "feature": "rows",
                "value": int(len(features)),
                "status": "computed",
                "notes": "Canopy trajectory feature rows for 1989-2024.",
            },
            {
                "diagnostic": "threshold_note",
                "feature": "decline_event_next",
                "value": np.nan,
                "status": "documented",
                "notes": "Default labels use the existing 1984-2013 p25 baseline column; full-history p25 is present in the source data but not used here.",
            },
            {
                "diagnostic": "window_definition",
                "feature": "rolling_windows",
                "value": np.nan,
                "status": "documented",
                "notes": "3-year windows use t,t-1,t-2; 5-year windows use t,t-1,t-2,t-3,t-4.",
            },
            {
                "diagnostic": "proxy_note",
                "feature": "instability_score_5yr_t",
                "value": np.nan,
                "status": "documented",
                "notes": "This is a time-series instability proxy based on mean absolute annual canopy change, not true spatial fragmentation.",
            },
        ]
    )
    first_all_valid = full_features.loc[full_features[TRAJECTORY_FEATURES].notna().all(axis=1), "year"]
    rows.append(
        {
            "diagnostic": "first_year_all_trajectory_features_valid",
            "feature": "all_trajectory_features",
            "value": int(first_all_valid.min()) if not first_all_valid.empty else np.nan,
            "status": "computed",
            "notes": "Computed across the full 1984-2024 panel before model-period filtering.",
        }
    )
    return pd.DataFrame(rows)


def small_markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact markdown table."""
    if frame.empty:
        return "No rows."
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
        else:
            display[column] = display[column].astype(str)
    columns = list(display.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(row[column] for column in columns) + " |" for _, row in display.iterrows()]
    return "\n".join([header, divider, *rows])


def best_by_target_family(results: pd.DataFrame) -> pd.DataFrame:
    """Return best PR-AUC row by target and feature family."""
    computed = results.loc[results["status"] == "computed"].copy()
    if computed.empty:
        return computed
    return (
        computed.sort_values(["target_definition", "feature_family", "pr_auc"], ascending=[True, True, False])
        .groupby(["target_definition", "feature_family"])
        .head(1)
        .reset_index(drop=True)
    )


def feature_family_best(results: pd.DataFrame, target_name: str, family: str) -> pd.Series | None:
    """Return best row for a target/family if present."""
    subset = results.loc[
        (results["status"] == "computed")
        & (results["target_definition"] == target_name)
        & (results["feature_family"] == family)
    ].sort_values("pr_auc", ascending=False)
    return None if subset.empty else subset.iloc[0]


def comparison_sentence(results: pd.DataFrame, target_name: str, base_family: str, contender_family: str) -> str:
    """Return a concise PR-AUC comparison sentence."""
    base = feature_family_best(results, target_name, base_family)
    contender = feature_family_best(results, target_name, contender_family)
    if base is None or contender is None:
        return f"`{contender_family}` versus `{base_family}` could not be compared for `{target_name}`."
    delta = float(contender["pr_auc"] - base["pr_auc"])
    direction = "improved" if delta > 0 else "did not improve"
    return (
        f"For `{target_name}`, `{contender_family}` {direction} over `{base_family}` "
        f"by PR-AUC ({contender['pr_auc']:.3f} vs {base['pr_auc']:.3f}; delta {delta:+.3f})."
    )


def write_report(output: Path, features: pd.DataFrame, diagnostics: pd.DataFrame, results: pd.DataFrame) -> None:
    """Write the canopy trajectory diagnostic report."""
    computed = results.loc[results["status"] == "computed"].copy()
    best = best_by_target_family(results)
    top = computed.sort_values(["target_definition", "pr_auc"], ascending=[True, False]).groupby("target_definition").head(1)
    leakage = diagnostics.loc[diagnostics["diagnostic"] == "leakage_audit"].iloc[0]
    first_all_valid = diagnostics.loc[diagnostics["diagnostic"] == "first_year_all_trajectory_features_valid", "value"].iloc[0]
    actionable_traj = feature_family_best(results, "actionable_decline_drop", "canopy_trajectory_only")
    actionable_naive = feature_family_best(results, "actionable_decline_drop", "naive_persistence_baseline")
    fn_sentence = "Actionable false-negative comparison was unavailable."
    if actionable_traj is not None and actionable_naive is not None:
        fn_sentence = (
            "For actionable decline, canopy trajectory features had "
            f"{int(actionable_traj['false_negatives'])} false negatives versus "
            f"{int(actionable_naive['false_negatives'])} for the naive persistence score."
        )

    lines = [
        "# Canopy Trajectory and Instability Proxy Feature Report",
        "",
        "## Purpose",
        "",
        "This report adds leakage-safe canopy trajectory and time-series instability proxy features.",
        "The features are a diagnostic extension of the persistence baseline, not an environmental driver layer.",
        "",
        "## Leakage Audit",
        "",
        "- Every generated feature row stores `trajectory_max_source_year_used`.",
        "- The audit verifies `trajectory_max_source_year_used <= year` for every model-period row.",
        f"- Leakage audit status: `{leakage['status']}`; violating rows: `{int(leakage['value'])}`.",
        f"- First year where all trajectory features are available in the full panel: `{int(first_all_valid)}`.",
        "",
        "## Feature Definitions",
        "",
        "- 3-year windows use year `t`, `t-1`, and `t-2` only.",
        "- 5-year windows use year `t`, `t-1`, `t-2`, `t-3`, and `t-4` only.",
        "- `instability_score_5yr_t` is mean absolute annual canopy change over the past/current 5-year window.",
        "- `presence_frequency_5yr_t` is the share of years in the 5-year window with `relative_canopy > 0.01`.",
        "- These are time-series instability proxies, not true spatial fragmentation metrics because patch geometry is not used.",
        "",
        "## Label and Threshold Notes",
        "",
        "- The default `decline_event_next` label uses the existing `baseline_p25_relative_canopy_1984_2013` threshold.",
        "- The source dataset also contains full-history p25 fields, but this script does not use them for the default model comparison.",
        "- The optional high-canopy subgroup uses the 1984-2013 cell-specific p75 threshold.",
        "",
        "## Feature Construction Summary",
        "",
        f"- Feature rows built: `{len(features)}`",
        f"- Cells represented: `{features['cell_id'].nunique()}`",
        f"- Model-period years: `{features['year'].min()}-{features['year'].max()}`",
        f"- Maximum feature missingness: `{diagnostics.loc[diagnostics['diagnostic'] == 'feature_missingness', 'value'].max():.4f}`",
        "",
        "## Model Comparison",
        "",
        f"- Computed model-comparison rows: `{len(computed)}`",
        "",
        "Best result per target:",
        "",
        small_markdown_table(top[["target_definition", "feature_family", "model", "pr_auc", "recall", "precision", "f1", "false_negatives"]])
        if not top.empty
        else "No computed model rows.",
        "",
        "Best row by target and feature family:",
        "",
        small_markdown_table(best[["target_definition", "feature_family", "model", "pr_auc", "recall", "precision", "f1", "false_negatives"]])
        if not best.empty
        else "No computed model rows.",
        "",
        "## Diagnostic Answers",
        "",
        "- Do trajectory features improve original broad decline prediction?",
        f"  {comparison_sentence(results, 'original_decline', 'existing_canopy_only', 'canopy_trajectory_only')}",
        "- Do trajectory features improve at-risk or transition-oriented prediction?",
        f"  {comparison_sentence(results, 'at_risk_original_gt005', 'existing_canopy_only', 'canopy_trajectory_only')}",
        f"  {comparison_sentence(results, 'new_decline_transition', 'existing_canopy_only', 'canopy_trajectory_only')}",
        f"  {comparison_sentence(results, 'actionable_decline_drop', 'existing_canopy_only', 'canopy_trajectory_only')}",
        "- Do improvements mainly strengthen persistence-based risk-state screening?",
        "  Interpret gains cautiously: trajectory features use current and recent canopy states, so improvements mostly refine persistence/risk-state screening unless they hold for stricter transition and actionable labels.",
        "- Do trajectory features reduce false negatives for actionable decline?",
        f"  {fn_sentence}",
        "- Do CRW and/or habitat features still add value after trajectory features are included?",
        f"  {comparison_sentence(results, 'at_risk_original_gt005', 'canopy_trajectory_only', 'canopy_trajectory_plus_crw')}",
        f"  {comparison_sentence(results, 'at_risk_original_gt005', 'canopy_trajectory_only', 'canopy_trajectory_plus_habitat')}",
        f"  {comparison_sentence(results, 'actionable_decline_drop', 'canopy_trajectory_only', 'canopy_trajectory_plus_crw_plus_habitat')}",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run canopy trajectory feature construction and model comparison."""
    args = parse_args()
    for path in [args.feature_output, args.feature_diagnostics_output, args.model_comparison_output, args.report_output]:
        path.parent.mkdir(parents=True, exist_ok=True)

    full_data = load_full_modeling_data(args.input)
    full_with_features = add_canopy_trajectory_features(full_data)
    features = feature_rows(full_with_features)
    diagnostics = diagnostics_table(features, full_with_features)
    features.to_csv(args.feature_output, index=False)
    diagnostics.to_csv(args.feature_diagnostics_output, index=False)

    modeling = main_subset(full_with_features.sort_values(["cell_id", "year"]).reset_index(drop=True))
    modeling = add_optional_features(modeling)
    results = evaluate_models(modeling)
    results.to_csv(args.model_comparison_output, index=False)
    write_report(args.report_output, features, diagnostics, results)

    first_all_valid = diagnostics.loc[diagnostics["diagnostic"] == "first_year_all_trajectory_features_valid", "value"].iloc[0]
    leakage = diagnostics.loc[diagnostics["diagnostic"] == "leakage_audit"].iloc[0]
    max_missingness = diagnostics.loc[diagnostics["diagnostic"] == "feature_missingness", "value"].max()
    print(f"Canopy trajectory feature rows built: {len(features)}")
    print(f"First year all trajectory features valid: {int(first_all_valid)}")
    print(f"Maximum feature missingness: {max_missingness:.4f}")
    print(f"Leakage audit status: {leakage['status']} ({int(leakage['value'])} violating rows)")
    print(f"Computed model-comparison rows: {(results['status'] == 'computed').sum()}")
    print(f"Wrote features: {args.feature_output}")
    print(f"Wrote diagnostics: {args.feature_diagnostics_output}")
    print(f"Wrote model comparison: {args.model_comparison_output}")
    print(f"Wrote report: {args.report_output}")


if __name__ == "__main__":
    main()
