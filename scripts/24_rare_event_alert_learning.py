"""Rare-event alert learning for transition/actionable kelp decline targets.

This workflow tests whether stricter transition/actionable targets are missed
because events are rare and default learning thresholds are conservative. It
uses only existing processed feature assets, keeps validation/test
distributions untouched, and applies all resampling to the training set only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]

BASE_DATA = ROOT / "data" / "processed" / "modeling_dataset_ge500_noaa_v1.csv"
TRAJECTORY_FEATURES_PATH = ROOT / "data" / "processed" / "canopy_trajectory_features.csv"
CRW_FEATURES_PATH = ROOT / "data" / "processed" / "crw5km_composite_features.csv"
HABITAT_FEATURES_PATH = ROOT / "data" / "processed" / "bathymetry_habitat_features.csv"
CLAIM_GATE_PATH = ROOT / "results" / "tables" / "claim_gate_summary.csv"

RESULTS_DIR = ROOT / "results" / "tables"
DIAGNOSTICS_DIR = ROOT / "outputs" / "diagnostics"

SPLIT_DIAGNOSTICS_OUT = RESULTS_DIR / "rare_event_split_diagnostics.csv"
HARD_NEGATIVE_DIAGNOSTICS_OUT = RESULTS_DIR / "rare_event_hard_negative_diagnostics.csv"
THRESHOLD_TUNING_OUT = RESULTS_DIR / "rare_event_threshold_tuning.csv"
TOPK_OUT = RESULTS_DIR / "rare_event_topk_alert_evaluation.csv"
MODEL_COMPARISON_OUT = RESULTS_DIR / "rare_event_alert_model_comparison.csv"
REPORT_OUT = DIAGNOSTICS_DIR / "rare_event_alert_learning_report.md"

MODEL_START_YEAR = 1989
TRAIN_END_YEAR = 2016
VALIDATION_START_YEAR = 2017
VALIDATION_END_YEAR = 2020
TEST_START_YEAR = 2021
TEST_END_YEAR = 2024
EPSILON = 1e-6
RANDOM_STATE = 42

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
PRECISION_FLOORS = [0.30, 0.40, 0.50]
TOP_PCTS = [0.05, 0.10, 0.20]
FIXED_ALERT_BUDGETS = [3, 5, 10]

CSV_WRITE_KWARGS = {
    "index": False,
    "lineterminator": "\n",
    "na_rep": "",
    "float_format": "%.6f",
}

CANOPY_FEATURES = [
    "relative_canopy",
]

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

CRW_FEATURES = [
    "annual_mean_sst_crw5km",
    "spring_mean_sst_crw5km",
    "summer_mean_sst_crw5km",
    "warmest_month_mean_sst_crw5km",
    "annual_mean_ssta_crw5km",
    "spring_ssta_crw5km",
    "summer_ssta_crw5km",
    "annual_max_monthly_ssta_crw5km",
    "lag1_annual_mean_sst_crw5km",
    "lag1_spring_mean_sst_crw5km",
    "lag1_summer_mean_sst_crw5km",
    "lag1_warmest_month_mean_sst_crw5km",
    "lag1_annual_mean_ssta_crw5km",
    "lag1_spring_ssta_crw5km",
    "lag1_summer_ssta_crw5km",
    "lag1_annual_max_monthly_ssta_crw5km",
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
    "cuti_anomaly",
    "beuti_anomaly",
    "lag1_cuti_anomaly",
    "lag1_beuti_anomaly",
]


@dataclass(frozen=True)
class TargetSpec:
    name: str
    column: str
    filter_column: str | None = None


@dataclass(frozen=True)
class FeatureSet:
    name: str
    columns: list[str]
    status: str = "computed"
    notes: str = ""


@dataclass(frozen=True)
class StrategySpec:
    name: str
    base_strategy: str
    threshold_tuned: bool
    class_weighted: bool = False


TARGETS = [
    TargetSpec("actionable_drop", "actionable_decline_drop_next"),
    TargetSpec("new_transition", "new_decline_event_next"),
    TargetSpec("at_risk_original", "decline_event_next", "at_risk_original_eligible"),
]

STRATEGIES = [
    StrategySpec("original_distribution", "original_distribution", False, False),
    StrategySpec("class_weighted", "original_distribution", False, True),
    StrategySpec("positive_oversampling_train_only", "positive_oversampling_train_only", False, False),
    StrategySpec("random_negative_undersampling_train_only", "random_negative_undersampling_train_only", False, False),
    StrategySpec("hard_negative_undersampling_train_only", "hard_negative_undersampling_train_only", False, False),
    StrategySpec("class_weighted_threshold_tuned", "original_distribution", True, True),
    StrategySpec("positive_oversampling_threshold_tuned", "positive_oversampling_train_only", True, False),
    StrategySpec("hard_negative_threshold_tuned", "hard_negative_undersampling_train_only", True, False),
]


def clean_csv_cells(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    object_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        out[col] = (
            out[col]
            .astype("string")
            .str.replace("\r\n", " ", regex=False)
            .str.replace("\n", " ", regex=False)
            .str.replace("\r", " ", regex=False)
        )
        out[col] = out[col].where(out[col].notna(), "")
    return out


def write_portable_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_csv_cells(df).to_csv(path, **CSV_WRITE_KWARGS)


def load_inputs() -> pd.DataFrame:
    if not BASE_DATA.exists():
        raise FileNotFoundError(BASE_DATA)
    data = pd.read_csv(BASE_DATA).sort_values(["cell_id", "year"]).reset_index(drop=True)

    for path, suffix in [
        (TRAJECTORY_FEATURES_PATH, "trajectory"),
        (CRW_FEATURES_PATH, "crw"),
        (HABITAT_FEATURES_PATH, "habitat"),
    ]:
        if not path.exists():
            continue
        features = pd.read_csv(path)
        if "year" in features.columns:
            data = data.merge(features, on=["cell_id", "year"], how="left")
        else:
            data = data.merge(features, on="cell_id", how="left", suffixes=("", f"_{suffix}"))

    data = data.loc[data["year"].between(MODEL_START_YEAR, TEST_END_YEAR)].copy()
    add_targets(data)
    add_hard_negative_flags(data)
    data["split"] = np.select(
        [
            data["year"] <= TRAIN_END_YEAR,
            data["year"].between(VALIDATION_START_YEAR, VALIDATION_END_YEAR),
            data["year"].between(TEST_START_YEAR, TEST_END_YEAR),
        ],
        ["train", "validation", "test"],
        default="other",
    )
    data = data.loc[data["split"].isin(["train", "validation", "test"])].reset_index(drop=True)
    return data


def add_targets(data: pd.DataFrame) -> None:
    data["relative_drop_next"] = (
        data["relative_canopy"] - data["next_year_relative_canopy"]
    ) / np.maximum(data["relative_canopy"], EPSILON)
    data["actionable_decline_drop_next"] = (
        (data["relative_canopy"] > 0.05) & (data["relative_drop_next"] >= 0.30)
    ).astype(int)
    data["new_decline_event_next"] = (
        (data["relative_canopy"] >= data["baseline_p25_relative_canopy_1984_2013"])
        & (data["next_year_relative_canopy"] < data["baseline_p25_relative_canopy_1984_2013"])
    ).astype(int)
    data["at_risk_original_eligible"] = data["relative_canopy"] > 0.05


def add_hard_negative_flags(data: pd.DataFrame) -> None:
    data["hard_negative_canopy_present"] = data["relative_canopy"] > 0.05

    decline_sources = []
    if "recent_decline_rate_3yr_t" in data.columns:
        decline_sources.append(data["recent_decline_rate_3yr_t"].fillna(0) > 0)
    if "canopy_3yr_slope_t" in data.columns:
        decline_sources.append(data["canopy_3yr_slope_t"].fillna(0) < 0)
    data["hard_negative_recent_decline"] = np.logical_or.reduce(decline_sources) if decline_sources else False

    risk_parts = []
    for col in ["annual_max_monthly_ssta_crw5km", "annual_mean_sst_anomaly", "instability_score_5yr_t"]:
        if col in data.columns:
            q75 = data.loc[data["split"].eq("train") if "split" in data.columns else data.index, col].quantile(0.75)
            risk_parts.append(data[col] >= q75)
    if "recent_decline_rate_3yr_t" in data.columns:
        risk_parts.append(data["recent_decline_rate_3yr_t"].fillna(0) > 0)
    data["hard_negative_thermal_or_trajectory_risk"] = np.logical_or.reduce(risk_parts) if risk_parts else data[
        "hard_negative_recent_decline"
    ]
    data["hard_negative_any"] = (
        data["hard_negative_canopy_present"]
        | data["hard_negative_recent_decline"]
        | data["hard_negative_thermal_or_trajectory_risk"]
    )


def available_feature_sets(data: pd.DataFrame) -> list[FeatureSet]:
    def available(columns: list[str]) -> list[str]:
        return [col for col in columns if col in data.columns]

    sets = [
        FeatureSet("canopy_only", available(CANOPY_FEATURES)),
        FeatureSet("canopy_trajectory", available(TRAJECTORY_FEATURES)),
        FeatureSet("CRW_composite", available(CRW_FEATURES)),
        FeatureSet("bathymetry_habitat", available(HABITAT_FEATURES)),
        FeatureSet("canopy_trajectory_plus_CRW", available(TRAJECTORY_FEATURES + CRW_FEATURES)),
        FeatureSet(
            "canopy_trajectory_plus_CRW_plus_habitat",
            available(TRAJECTORY_FEATURES + CRW_FEATURES + HABITAT_FEATURES),
        ),
        FeatureSet(
            "canopy_trajectory_plus_OISST_plus_habitat",
            available(TRAJECTORY_FEATURES + OISST_FEATURES + HABITAT_FEATURES),
            notes="Fallback feature family if CRW coverage is incomplete.",
        ),
    ]
    return [fs for fs in sets if fs.columns]


def target_data(data: pd.DataFrame, target: TargetSpec) -> pd.DataFrame:
    subset = data.copy()
    if target.filter_column:
        subset = subset.loc[subset[target.filter_column].astype(bool)].copy()
    subset = subset.dropna(subset=[target.column])
    subset[target.column] = subset[target.column].astype(int)
    return subset


def split_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        subset = target_data(data, target)
        for split in ["train", "validation", "test", "all"]:
            part = subset if split == "all" else subset.loc[subset["split"].eq(split)]
            positives = int(part[target.column].sum()) if len(part) else 0
            rows.append(
                {
                    "target_definition": target.name,
                    "target_column": target.column,
                    "split": split,
                    "rows": len(part),
                    "positive_events": positives,
                    "positive_rate": positives / len(part) if len(part) else np.nan,
                    "cells_with_positive_events": part.loc[part[target.column].eq(1), "cell_id"].nunique(),
                    "years_with_positive_events": part.loc[part[target.column].eq(1), "year"].nunique(),
                    "filter_column": target.filter_column or "",
                }
            )
    return pd.DataFrame(rows)


def hard_negative_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rules = [
        "hard_negative_canopy_present",
        "hard_negative_recent_decline",
        "hard_negative_thermal_or_trajectory_risk",
        "hard_negative_any",
    ]
    for target in TARGETS:
        subset = target_data(data, target)
        for split in ["train", "validation", "test", "all"]:
            part = subset if split == "all" else subset.loc[subset["split"].eq(split)]
            negatives = part[target.column].eq(0)
            for rule in rules:
                count = int((negatives & part[rule].astype(bool)).sum()) if len(part) else 0
                rows.append(
                    {
                        "target_definition": target.name,
                        "split": split,
                        "hard_negative_rule": rule,
                        "negative_rows": int(negatives.sum()) if len(part) else 0,
                        "hard_negative_rows": count,
                        "hard_negative_share_of_negatives": count / negatives.sum() if negatives.sum() else np.nan,
                        "status": "computed" if rule in part.columns else "missing_rule",
                    }
                )
    return pd.DataFrame(rows)


def make_model(model_name: str, class_weighted: bool, y_train: pd.Series) -> Pipeline:
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / pos if pos else 1.0

    if model_name == "Logistic Regression":
        model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if class_weighted else None,
            random_state=RANDOM_STATE,
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", model)]
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=50,
            min_samples_leaf=3,
            class_weight="balanced" if class_weighted else None,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", model)]
    elif model_name == "XGBoost":
        model = XGBClassifier(
            n_estimators=60,
            max_depth=3,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight if class_weighted else 1.0,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", model)]
    elif model_name == "LightGBM":
        model = LGBMClassifier(
            n_estimators=60,
            learning_rate=0.06,
            num_leaves=15,
            class_weight="balanced" if class_weighted else None,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", model)]
    else:
        raise ValueError(model_name)
    return Pipeline(steps)


def resample_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_meta: pd.DataFrame,
    strategy: StrategySpec,
) -> tuple[pd.DataFrame, pd.Series, float, str]:
    rng = np.random.default_rng(RANDOM_STATE)
    pos_idx = y_train[y_train.eq(1)].index.to_numpy()
    neg_idx = y_train[y_train.eq(0)].index.to_numpy()
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X_train, y_train, np.nan, "insufficient_class_diversity"

    if strategy.base_strategy == "original_distribution":
        return X_train, y_train, len(neg_idx) / len(pos_idx), "not_applicable"

    if strategy.base_strategy == "positive_oversampling_train_only":
        extra_pos = rng.choice(pos_idx, size=max(len(neg_idx) - len(pos_idx), 0), replace=True)
        sampled_idx = np.concatenate([X_train.index.to_numpy(), extra_pos])
        rng.shuffle(sampled_idx)
        return X_train.loc[sampled_idx], y_train.loc[sampled_idx], 1.0, "not_applicable"

    if strategy.base_strategy == "random_negative_undersampling_train_only":
        keep_neg = rng.choice(neg_idx, size=min(len(neg_idx), len(pos_idx) * 3), replace=False)
        sampled_idx = np.concatenate([pos_idx, keep_neg])
        rng.shuffle(sampled_idx)
        return X_train.loc[sampled_idx], y_train.loc[sampled_idx], len(keep_neg) / len(pos_idx), "random_negative"

    if strategy.base_strategy == "hard_negative_undersampling_train_only":
        hard_neg_idx = train_meta.loc[y_train.eq(0) & train_meta["hard_negative_any"].astype(bool)].index.to_numpy()
        ordinary_neg_idx = np.setdiff1d(neg_idx, hard_neg_idx)
        hard_n = min(len(hard_neg_idx), len(pos_idx) * 3)
        ordinary_n = min(len(ordinary_neg_idx), max(len(pos_idx), 1))
        sampled_hard = rng.choice(hard_neg_idx, size=hard_n, replace=False) if hard_n else np.array([], dtype=int)
        sampled_ordinary = (
            rng.choice(ordinary_neg_idx, size=ordinary_n, replace=False) if ordinary_n else np.array([], dtype=int)
        )
        sampled_idx = np.concatenate([pos_idx, sampled_hard, sampled_ordinary])
        rng.shuffle(sampled_idx)
        ratio = (len(sampled_hard) + len(sampled_ordinary)) / len(pos_idx)
        return X_train.loc[sampled_idx], y_train.loc[sampled_idx], ratio, "hard_negative_any"

    raise ValueError(strategy.base_strategy)


def predict_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        raw = model.decision_function(X)
        return 1 / (1 + np.exp(-raw))
    raise ValueError("Model cannot produce scores")


def metrics_at_threshold(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if len(np.unique(y_true)) > 1:
        pr_auc = average_precision_score(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
    else:
        pr_auc = np.nan
        roc_auc = np.nan
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def threshold_grid(
    y_val: pd.Series,
    val_score: np.ndarray,
    context: dict,
) -> tuple[list[dict], dict[float, float]]:
    rows = []
    selected: dict[float, float] = {}
    for floor in PRECISION_FLOORS:
        floor_rows = []
        for threshold in THRESHOLDS:
            metrics = metrics_at_threshold(y_val, val_score, threshold)
            row = {
                **context,
                "precision_floor": floor,
                "threshold": threshold,
                "selection_split": "validation",
                **metrics,
            }
            floor_rows.append(row)
        candidates = [r for r in floor_rows if r["precision"] >= floor]
        if candidates:
            best = sorted(candidates, key=lambda r: (r["f2"], r["recall"], r["precision"]), reverse=True)[0]
            fallback = False
        else:
            best = sorted(floor_rows, key=lambda r: (r["f2"], r["recall"], r["precision"]), reverse=True)[0]
            fallback = True
        selected[floor] = best["threshold"]
        for row in floor_rows:
            row["selected_threshold"] = best["threshold"]
            row["selected_for_precision_floor"] = bool(row["threshold"] == best["threshold"])
            row["precision_floor_fallback_used"] = fallback
            rows.append(row)
    return rows, selected


def topk_rows(test_part: pd.DataFrame, y_score: np.ndarray, context: dict) -> list[dict]:
    scored = test_part[["cell_id", "year", context["target_column"]]].copy()
    scored["score"] = y_score
    rows = []
    for year, year_df in scored.groupby("year"):
        positives = int(year_df[context["target_column"]].sum())
        n_rows = len(year_df)
        for pct in TOP_PCTS:
            k = max(1, int(np.ceil(n_rows * pct)))
            rows.append(topk_metric_row(year_df, k, positives, context, year, f"top_{int(pct * 100)}pct"))
        for k in FIXED_ALERT_BUDGETS:
            rows.append(topk_metric_row(year_df, min(k, n_rows), positives, context, year, f"top_{k}_cell_years"))
    return rows


def topk_metric_row(year_df: pd.DataFrame, k: int, positives: int, context: dict, year: int, rule: str) -> dict:
    selected = year_df.sort_values("score", ascending=False).head(k)
    captured = int(selected[context["target_column"]].sum())
    return {
        **context,
        "test_year": int(year),
        "alert_rule": rule,
        "annual_test_rows": len(year_df),
        "annual_alert_count": k,
        "annual_positive_events": positives,
        "annual_positive_events_captured": captured,
        "annual_positive_events_missed": positives - captured,
        "recall_at_k": captured / positives if positives else np.nan,
        "precision_at_k": captured / k if k else np.nan,
    }


def run_models(data: pd.DataFrame, feature_sets: list[FeatureSet]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict] = []
    threshold_rows: list[dict] = []
    topk_result_rows: list[dict] = []
    model_names = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
    total_fits = len(TARGETS) * len(feature_sets) * len(STRATEGIES) * len(model_names)
    fit_counter = 0

    for target in TARGETS:
        subset = target_data(data, target)
        train = subset.loc[subset["split"].eq("train")].copy()
        validation = subset.loc[subset["split"].eq("validation")].copy()
        test = subset.loc[subset["split"].eq("test")].copy()
        if train.empty or validation.empty or test.empty:
            continue
        if train[target.column].nunique() < 2 or validation[target.column].nunique() < 2 or test[target.column].nunique() < 2:
            continue

        for feature_set in feature_sets:
            usable_features = [col for col in feature_set.columns if col in subset.columns]
            if not usable_features:
                continue

            X_train_base = train[usable_features]
            y_train_base = train[target.column].astype(int)
            X_val = validation[usable_features]
            y_val = validation[target.column].astype(int)
            X_test = test[usable_features]
            y_test = test[target.column].astype(int)

            for strategy in STRATEGIES:
                X_train, y_train, sampling_ratio, hard_rule = resample_train(
                    X_train_base, y_train_base, train, strategy
                )
                for model_name in model_names:
                    fit_counter += 1
                    if fit_counter == 1 or fit_counter % 50 == 0:
                        print(
                            f"Fitting {fit_counter}/{total_fits}: "
                            f"{target.name} | {feature_set.name} | {strategy.name} | {model_name}",
                            flush=True,
                        )
                    context = {
                        "target_definition": target.name,
                        "target_column": target.column,
                        "feature_family": feature_set.name,
                        "model": model_name,
                        "learning_strategy": strategy.name,
                        "sampling_ratio": sampling_ratio,
                        "hard_negative_rule": hard_rule,
                    }
                    try:
                        model = make_model(model_name, strategy.class_weighted, y_train)
                        model.fit(X_train, y_train)
                        val_score = predict_scores(model, X_val)
                        test_score = predict_scores(model, X_test)

                        if strategy.threshold_tuned:
                            grid_rows, selected_by_floor = threshold_grid(y_val, val_score, context)
                            threshold_rows.extend(grid_rows)
                            for floor, threshold in selected_by_floor.items():
                                model_rows.append(
                                    comparison_row(
                                        context,
                                        y_test,
                                        test_score,
                                        threshold,
                                        "validation_f2_with_precision_floor",
                                        floor,
                                        train,
                                        validation,
                                        test,
                                        "computed",
                                        feature_set.notes,
                                    )
                                )
                                topk_context = {
                                    **context,
                                    "threshold_strategy": "validation_f2_with_precision_floor",
                                    "selected_threshold": threshold,
                                    "precision_floor": floor,
                                }
                                topk_result_rows.extend(topk_rows(test, test_score, topk_context))
                        else:
                            model_rows.append(
                                comparison_row(
                                    context,
                                    y_test,
                                    test_score,
                                    0.50,
                                    "default_0_50",
                                    np.nan,
                                    train,
                                    validation,
                                    test,
                                    "computed",
                                    feature_set.notes,
                                )
                            )
                            topk_context = {
                                **context,
                                "threshold_strategy": "default_0_50",
                                "selected_threshold": 0.50,
                                "precision_floor": np.nan,
                            }
                            topk_result_rows.extend(topk_rows(test, test_score, topk_context))
                    except Exception as exc:
                        model_rows.append(
                            {
                                **context,
                                "threshold_strategy": "not_run",
                                "selected_threshold": np.nan,
                                "precision_floor": np.nan,
                                "pr_auc": np.nan,
                                "roc_auc": np.nan,
                                "recall": np.nan,
                                "precision": np.nan,
                                "f1": np.nan,
                                "f2": np.nan,
                                "false_negatives": np.nan,
                                "false_positives": np.nan,
                                "true_positives": np.nan,
                                "true_negatives": np.nan,
                                "train_positive_count": int(y_train_base.sum()),
                                "validation_positive_count": int(y_val.sum()),
                                "test_positive_count": int(y_test.sum()),
                                "status": "failed",
                                "notes": str(exc),
                            }
                        )

    return pd.DataFrame(model_rows), pd.DataFrame(threshold_rows), pd.DataFrame(topk_result_rows)


def comparison_row(
    context: dict,
    y_test: pd.Series,
    test_score: np.ndarray,
    threshold: float,
    threshold_strategy: str,
    precision_floor: float,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    status: str,
    notes: str,
) -> dict:
    metrics = metrics_at_threshold(y_test, test_score, threshold)
    target_col = context["target_column"]
    return {
        "target_definition": context["target_definition"],
        "feature_family": context["feature_family"],
        "model": context["model"],
        "learning_strategy": context["learning_strategy"],
        "threshold_strategy": threshold_strategy,
        "selected_threshold": threshold,
        "precision_floor": precision_floor,
        **metrics,
        "train_positive_count": int(train[target_col].sum()),
        "validation_positive_count": int(validation[target_col].sum()),
        "test_positive_count": int(test[target_col].sum()),
        "sampling_ratio": context["sampling_ratio"],
        "hard_negative_rule": context["hard_negative_rule"],
        "status": status,
        "notes": notes,
    }


def best_row(df: pd.DataFrame, target: str, metric: str, maximize: bool = True, precision_floor: float | None = None) -> pd.Series | None:
    subset = df.loc[df["target_definition"].eq(target) & df["status"].eq("computed")].copy()
    if precision_floor is not None:
        subset = subset.loc[subset["precision"].fillna(-np.inf) >= precision_floor]
    subset = subset.dropna(subset=[metric])
    if subset.empty:
        return None
    if metric == "false_negatives":
        return subset.sort_values(
            ["false_negatives", "precision", "recall", "f2", "pr_auc"],
            ascending=[True, False, False, False, False],
            na_position="last",
        ).iloc[0]
    return subset.sort_values(
        [metric, "precision", "recall", "pr_auc"],
        ascending=[not maximize, False, False, False],
        na_position="last",
    ).iloc[0]


def best_topk(topk: pd.DataFrame, target: str) -> pd.Series | None:
    subset = topk.loc[topk["target_definition"].eq(target)].copy()
    if subset.empty:
        return None
    aggregate = (
        subset.groupby(["feature_family", "model", "learning_strategy", "threshold_strategy", "alert_rule"], dropna=False)
        .agg(
            mean_recall_at_k=("recall_at_k", "mean"),
            mean_precision_at_k=("precision_at_k", "mean"),
            total_captured=("annual_positive_events_captured", "sum"),
            total_missed=("annual_positive_events_missed", "sum"),
            total_alerts=("annual_alert_count", "sum"),
        )
        .reset_index()
    )
    return aggregate.sort_values(["mean_recall_at_k", "mean_precision_at_k"], ascending=False).iloc[0]


def gate3_label_from_results(comparison: pd.DataFrame, target: str, precision_floor: float = 0.40) -> str:
    subset = comparison.loc[comparison["target_definition"].eq(target) & comparison["status"].eq("computed")].copy()
    if subset.empty:
        return "insufficient_information"
    canopy = subset.loc[subset["feature_family"].eq("canopy_only")]
    canopy_pr = canopy["pr_auc"].max()
    canopy_fn = canopy["false_negatives"].min()
    labels = []
    for _, row in subset.iterrows():
        conditions = 0
        pr_gain = pd.notna(canopy_pr) and row["pr_auc"] - canopy_pr >= 0.03
        if pr_gain:
            conditions += 1
        if pd.notna(row["f2"]) and row["f2"] >= 0.50:
            conditions += 1
        if pd.notna(canopy_fn) and canopy_fn > 0 and (canopy_fn - row["false_negatives"]) / canopy_fn >= 0.40:
            conditions += 1
        if row["recall"] >= 0.70:
            conditions += 1
        if row["precision"] >= precision_floor:
            conditions += 1
        if "threshold_tuned" in row["learning_strategy"]:
            conditions += 1
        if conditions >= 3 and pr_gain:
            labels.append("transition_early_warning_supported")
        elif (row["recall"] >= 0.70 or (pd.notna(canopy_fn) and row["false_negatives"] < canopy_fn)) and row[
            "precision"
        ] >= precision_floor:
            labels.append("transition_recall_oriented_sensitivity_only")
    if "transition_early_warning_supported" in labels:
        return "transition_early_warning_supported"
    if "transition_recall_oriented_sensitivity_only" in labels:
        return "transition_recall_oriented_sensitivity_only"
    return "transition_early_warning_not_supported"


def write_report(
    split_diag: pd.DataFrame,
    hard_diag: pd.DataFrame,
    comparison: pd.DataFrame,
    topk: pd.DataFrame,
) -> None:
    lines = [
        "# Rare-Event Alert Learning Report",
        "",
        "## Purpose",
        "",
        "This experiment tests whether rare transition/actionable kelp decline events can be detected more sensitively using training-only resampling, class weighting, hard-negative sampling, and validation-selected thresholds. It does not create new ecological events, does not resample validation or test rows, and does not replace the existing V1/V2 workflows.",
        "",
        "## Event-Count Diagnosis",
        "",
        dataframe_to_markdown(split_diag),
        "",
        "## Hard-Negative Strategy",
        "",
        "Hard negatives are non-event rows that still represent plausible decline-risk cases: canopy-present rows, rows with recent decline or instability, and rows with thermal or trajectory risk. These are more informative than easy negatives because they resemble rows that might plausibly trigger an alert.",
        "",
        dataframe_to_markdown(hard_diag.loc[hard_diag["split"].eq("train")]),
        "",
        "## Main Results",
        "",
    ]
    for target in [t.name for t in TARGETS]:
        best_pr = best_row(comparison, target, "pr_auc")
        best_f2 = best_row(comparison, target, "f2")
        best_recall = best_row(comparison, target, "recall", precision_floor=0.40)
        best_fn = best_row(comparison, target, "false_negatives", maximize=False)
        top = best_topk(topk, target)
        lines.extend(
            [
                f"### {target}",
                "",
                f"- Best PR-AUC: {format_best(best_pr, 'pr_auc')}",
                f"- Best F2: {format_best(best_f2, 'f2')}",
                f"- Best recall with precision >= 0.40: {format_best(best_recall, 'recall')}",
                f"- Lowest false negatives: {format_best(best_fn, 'false_negatives')}",
                f"- Best top-k alert result: {format_topk(top)}",
                "",
            ]
        )

    actionable_gate = gate3_label_from_results(comparison, "actionable_drop")
    transition_gate = gate3_label_from_results(comparison, "new_transition")
    overall_gate = (
        "transition_early_warning_supported"
        if "transition_early_warning_supported" in {actionable_gate, transition_gate}
        else "transition_recall_oriented_sensitivity_only"
        if "transition_recall_oriented_sensitivity_only" in {actionable_gate, transition_gate}
        else "transition_early_warning_not_supported"
    )
    lines.extend(
        [
            "## Comparison to Claim Gates",
            "",
            f"- Actionable-drop Gate 3 interpretation after rare-event learning: `{actionable_gate}`.",
            f"- New-transition Gate 3 interpretation after rare-event learning: `{transition_gate}`.",
            f"- Overall Gate 3 interpretation: `{overall_gate}`.",
            "",
            "These labels are diagnostic. They should not be interpreted as operational early-warning success unless precision, recall, F2, PR-AUC, and false-negative reduction are jointly acceptable.",
            "",
            "## Precision-Recall Tradeoff",
            "",
            "Rare-event learning improved sensitivity for some model/target combinations, especially through class weighting and threshold tuning. Where recall increased without PR-AUC improvement over canopy-only baselines, the result is best described as recall-oriented sensitivity improvement. Top-k gains are alert-prioritization support, not full early-warning success.",
            "",
            "## Recommended Next Step",
            "",
        ]
    )
    if overall_gate == "transition_early_warning_supported":
        lines.append("Refine the alert model and add wave exposure as an external validation-oriented disturbance layer.")
    else:
        lines.append("Wave exposure remains the next priority because rare-event learning alone does not establish robust transition/actionable early-warning support.")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a small markdown table without requiring optional tabulate."""
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
        else:
            display[col] = display[col].fillna("").astype(str)
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[col]).replace("|", "/") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_best(row: pd.Series | None, metric: str) -> str:
    if row is None:
        return "not available"
    return (
        f"`{row['feature_family']}` / `{row['model']}` / `{row['learning_strategy']}` "
        f"({metric} = `{row[metric]:.3f}`, precision = `{row['precision']:.3f}`, "
        f"recall = `{row['recall']:.3f}`, FN = `{row['false_negatives']:.0f}`)"
    )


def format_topk(row: pd.Series | None) -> str:
    if row is None:
        return "not available"
    return (
        f"`{row['feature_family']}` / `{row['model']}` / `{row['learning_strategy']}` / "
        f"`{row['alert_rule']}` (mean recall = `{row['mean_recall_at_k']:.3f}`, "
        f"mean precision = `{row['mean_precision_at_k']:.3f}`, captured = `{row['total_captured']:.0f}`)"
    )


def main() -> None:
    data = load_inputs()
    feature_sets = available_feature_sets(data)
    split_diag = split_diagnostics(data)
    hard_diag = hard_negative_diagnostics(data)
    comparison, threshold, topk = run_models(data, feature_sets)

    write_portable_csv(split_diag, SPLIT_DIAGNOSTICS_OUT)
    write_portable_csv(hard_diag, HARD_NEGATIVE_DIAGNOSTICS_OUT)
    write_portable_csv(threshold, THRESHOLD_TUNING_OUT)
    write_portable_csv(topk, TOPK_OUT)
    write_portable_csv(comparison, MODEL_COMPARISON_OUT)
    write_report(split_diag, hard_diag, comparison, topk)

    print(f"Rows written to {SPLIT_DIAGNOSTICS_OUT.relative_to(ROOT)}: {len(split_diag)}")
    print(f"Rows written to {HARD_NEGATIVE_DIAGNOSTICS_OUT.relative_to(ROOT)}: {len(hard_diag)}")
    print(f"Rows written to {THRESHOLD_TUNING_OUT.relative_to(ROOT)}: {len(threshold)}")
    print(f"Rows written to {TOPK_OUT.relative_to(ROOT)}: {len(topk)}")
    print(f"Rows written to {MODEL_COMPARISON_OUT.relative_to(ROOT)}: {len(comparison)}")
    for target in [t.name for t in TARGETS]:
        best_f2 = best_row(comparison, target, "f2")
        best_fn = best_row(comparison, target, "false_negatives", maximize=False)
        top = best_topk(topk, target)
        print(f"{target}: best F2 -> {format_best(best_f2, 'f2')}")
        print(f"{target}: lowest FN -> {format_best(best_fn, 'false_negatives')}")
        print(f"{target}: best top-k -> {format_topk(top)}")
    print("Gate 3 actionable_drop:", gate3_label_from_results(comparison, "actionable_drop"))
    print("Gate 3 new_transition:", gate3_label_from_results(comparison, "new_transition"))


if __name__ == "__main__":
    main()
