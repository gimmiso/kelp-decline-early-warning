"""Multi-horizon actionable kelp decline warning experiment.

This workflow compares actionable canopy-drop screening at two warning
horizons: next-year drop and within-two-year drop. It reuses existing processed
feature layers only and keeps future canopy values out of predictor columns.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
BASE_DATA = ROOT / "data" / "processed" / "modeling_dataset_ge500_noaa_v1.csv"
TRAJECTORY_FEATURES_PATH = ROOT / "data" / "processed" / "canopy_trajectory_features.csv"
CRW_FEATURES_PATH = ROOT / "data" / "processed" / "crw5km_composite_features.csv"
HABITAT_FEATURES_PATH = ROOT / "data" / "processed" / "bathymetry_habitat_features.csv"
WAVE_FEATURES_PATH = ROOT / "data" / "processed" / "wave_exposure_features.csv"

RESULTS_DIR = ROOT / "results" / "tables"
DIAGNOSTICS_DIR = ROOT / "outputs" / "diagnostics"

MODEL_COMPARISON_OUT = RESULTS_DIR / "multihorizon_actionable_model_comparison.csv"
SUMMARY_OUT = RESULTS_DIR / "multihorizon_actionable_summary.csv"
COMMON_YEAR_OUT = RESULTS_DIR / "multihorizon_actionable_common_year_comparison.csv"
REPORT_OUT = DIAGNOSTICS_DIR / "multihorizon_actionable_warning_report.md"

MODEL_START_YEAR = 1989
TRAIN_END_YEAR = 2016
VALIDATION_START_YEAR = 2017
VALIDATION_END_YEAR = 2020
TEST_START_YEAR = 2021
TEST_END_YEAR = 2024
EPSILON = 1e-6
RANDOM_STATE = 42

CSV_WRITE_KWARGS = {
    "index": False,
    "lineterminator": "\n",
    "na_rep": "",
    "float_format": "%.6f",
}

TARGET_1YEAR = "actionable_decline_drop_next_1year"
TARGET_2YEAR = "actionable_decline_drop_next_2year"

CANOPY_CURRENT_FEATURES = [
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

WAVE_FEATURES = [
    "winter_max_wave_height_cdip_model",
    "winter_mean_wave_height_cdip_model",
    "annual_max_wave_height_cdip_model",
    "annual_mean_wave_height_cdip_model",
    "lag1_winter_max_wave_height_cdip_model",
    "lag1_winter_mean_wave_height_cdip_model",
    "wave_height_anomaly_cdip_model",
    "storm_month_count_cdip_model",
    "distance_to_cdip_model_point_km",
]

LEAKAGE_PATTERNS = [
    "next_year",
    "t_plus",
    "future",
    "decline_event",
    "actionable_decline",
    "relative_drop",
    "drop_next",
    "target",
]


@dataclass(frozen=True)
class HorizonSpec:
    """Target metadata for one warning horizon."""

    name: str
    target: str
    valid_column: str


HORIZONS = [
    HorizonSpec("1year", TARGET_1YEAR, "label_valid_1year"),
    HorizonSpec("2year", TARGET_2YEAR, "label_valid_2year"),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run multi-horizon actionable warning experiment.")
    parser.add_argument("--base-data", type=Path, default=BASE_DATA)
    parser.add_argument("--model-comparison-output", type=Path, default=MODEL_COMPARISON_OUT)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_OUT)
    parser.add_argument("--common-year-output", type=Path, default=COMMON_YEAR_OUT)
    parser.add_argument("--report-output", type=Path, default=REPORT_OUT)
    return parser.parse_args()


def clean_csv_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten embedded newlines for GitHub-friendly CSV rendering."""
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
    """Write stable LF CSV files with a final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = clean_csv_cells(df).to_csv(**CSV_WRITE_KWARGS)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def available(columns: list[str], data: pd.DataFrame) -> list[str]:
    """Return nonempty columns available in the dataset."""
    return [col for col in columns if col in data.columns and not data[col].isna().all()]


def merge_feature_table(data: pd.DataFrame, path: Path, label: str) -> pd.DataFrame:
    """Merge an optional feature table by cell-year or cell ID."""
    if not path.exists():
        return data
    features = pd.read_csv(path)
    if "cell_id" not in features.columns:
        return data
    merge_keys = ["cell_id", "year"] if "year" in features.columns else ["cell_id"]
    drop_cols = [col for col in features.columns if col not in merge_keys and col in data.columns]
    features = features.drop(columns=drop_cols)
    return data.merge(features, on=merge_keys, how="left", suffixes=("", f"_{label}"))


def add_multihorizon_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Add one-year and within-two-year actionable drop labels."""
    out = data.sort_values(["cell_id", "year"]).copy()
    grouped = out.groupby("cell_id", group_keys=False)
    out["canopy_t"] = out["relative_canopy"]
    out["canopy_t_plus_1"] = grouped["relative_canopy"].shift(-1)
    out["canopy_t_plus_2"] = grouped["relative_canopy"].shift(-2)
    out["label_valid_1year"] = out["canopy_t"].notna() & out["canopy_t_plus_1"].notna()
    out["label_valid_2year"] = out["canopy_t"].notna() & out["canopy_t_plus_1"].notna() & out["canopy_t_plus_2"].notna()
    out["relative_drop_next_1year"] = (out["canopy_t"] - out["canopy_t_plus_1"]) / np.maximum(out["canopy_t"], EPSILON)
    out["relative_drop_next_2year"] = (
        out["canopy_t"] - out[["canopy_t_plus_1", "canopy_t_plus_2"]].min(axis=1)
    ) / np.maximum(out["canopy_t"], EPSILON)
    out[TARGET_1YEAR] = np.where(
        out["label_valid_1year"],
        ((out["canopy_t"] > 0.05) & (out["relative_drop_next_1year"] >= 0.30)).astype(int),
        np.nan,
    )
    out[TARGET_2YEAR] = np.where(
        out["label_valid_2year"],
        ((out["canopy_t"] > 0.05) & (out["relative_drop_next_2year"] >= 0.30)).astype(int),
        np.nan,
    )
    return out


def load_inputs(base_path: Path) -> pd.DataFrame:
    """Load base and optional feature layers."""
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    data = pd.read_csv(base_path).sort_values(["cell_id", "year"]).reset_index(drop=True)
    data = merge_feature_table(data, TRAJECTORY_FEATURES_PATH, "trajectory")
    data = merge_feature_table(data, CRW_FEATURES_PATH, "crw")
    data = merge_feature_table(data, HABITAT_FEATURES_PATH, "habitat")
    data = merge_feature_table(data, WAVE_FEATURES_PATH, "wave")
    data = data.loc[data["year"].between(MODEL_START_YEAR, TEST_END_YEAR)].copy()
    data = add_multihorizon_labels(data)
    data["split"] = np.select(
        [
            data["year"].between(MODEL_START_YEAR, TRAIN_END_YEAR),
            data["year"].between(VALIDATION_START_YEAR, VALIDATION_END_YEAR),
            data["year"].between(TEST_START_YEAR, TEST_END_YEAR),
        ],
        ["train", "validation", "test"],
        default="other",
    )
    return data.loc[data["split"].isin(["train", "validation", "test"])].reset_index(drop=True)


def feature_families(data: pd.DataFrame) -> dict[str, list[str]]:
    """Define the requested multi-horizon feature families."""
    environment = available(OISST_FEATURES + CUTI_BEUTI_FEATURES + CRW_FEATURES + HABITAT_FEATURES + WAVE_FEATURES, data)
    families = {
        "canopy_current_only": available(CANOPY_CURRENT_FEATURES, data),
        "canopy_current_plus_trajectory": available(CANOPY_CURRENT_FEATURES + TRAJECTORY_FEATURES, data),
        "canopy_current_plus_environment": available(CANOPY_CURRENT_FEATURES + environment, data),
        "canopy_current_plus_trajectory_plus_environment": available(
            CANOPY_CURRENT_FEATURES + TRAJECTORY_FEATURES + environment,
            data,
        ),
    }
    for family, features in families.items():
        if not features:
            raise ValueError(f"No available features for {family}.")
        leaked = [
            feature
            for feature in features
            if any(pattern in feature.lower() for pattern in LEAKAGE_PATTERNS)
            or feature in {"canopy_t_plus_1", "canopy_t_plus_2"}
        ]
        if leaked:
            raise ValueError(f"Future/target leakage features in {family}: {leaked}")
    return families


def preprocessor(data: pd.DataFrame, features: list[str], scale: bool) -> ColumnTransformer:
    """Create a preprocessing transformer."""
    categorical = [feature for feature in features if not is_numeric_dtype(data[feature])]
    numeric = [feature for feature in features if feature not in categorical]
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, object, list[str]]] = [
        ("num", Pipeline(numeric_steps), numeric),
    ]
    if categorical:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def positive_class_weight(y: pd.Series) -> float:
    """Compute negative-to-positive ratio for boosted models."""
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    return float(negatives / positives) if positives else 1.0


def model_specs(y_train: pd.Series) -> dict[str, tuple[object, bool]]:
    """Return available classifiers and whether scaling is needed."""
    scale_pos_weight = positive_class_weight(y_train)
    specs: dict[str, tuple[object, bool]] = {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
            True,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=250,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                min_samples_leaf=3,
                n_jobs=1,
            ),
            False,
        ),
    }
    if XGBClassifier is not None:
        specs["XGBoost"] = (
            XGBClassifier(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            False,
        )
    if LGBMClassifier is not None:
        specs["LightGBM"] = (
            LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=15,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=1,
            ),
            False,
        )
    return specs


def f2_at_threshold(y_true: pd.Series, scores: np.ndarray, threshold: float) -> float:
    """Compute F2 for threshold selection."""
    predictions = (scores >= threshold).astype(int)
    return float(fbeta_score(y_true.astype(int), predictions, beta=2, zero_division=0))


def select_f2_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    """Select threshold on validation scores by F2."""
    precision, recall, thresholds = precision_recall_curve(y_true.astype(int), scores)
    if len(thresholds) == 0:
        return 0.5
    candidates = pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1], "threshold": thresholds})
    candidates["f2"] = (5 * candidates["precision"] * candidates["recall"]) / (
        4 * candidates["precision"] + candidates["recall"]
    ).replace(0, np.nan)
    candidates["f2"] = candidates["f2"].fillna(0)
    return float(candidates.sort_values(["f2", "recall", "precision"], ascending=[False, False, False]).iloc[0]["threshold"])


def score_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, object]:
    """Score one evaluation set."""
    y_true_int = y_true.astype(int)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_int, predictions, labels=[0, 1]).ravel()
    classes = set(y_true_int.unique())
    return {
        "pr_auc": float(average_precision_score(y_true_int, scores)) if len(classes) > 1 else np.nan,
        "roc_auc": float(roc_auc_score(y_true_int, scores)) if len(classes) > 1 else np.nan,
        "recall": float(recall_score(y_true_int, predictions, zero_division=0)),
        "precision": float(precision_score(y_true_int, predictions, zero_division=0)),
        "f1": float(f1_score(y_true_int, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true_int, predictions, beta=2, zero_division=0)),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "positive_count": int(y_true_int.sum()),
        "total_evaluated_rows": int(len(y_true_int)),
        "event_rate": float(y_true_int.mean()) if len(y_true_int) else np.nan,
    }


def subset_for_horizon(data: pd.DataFrame, horizon: HorizonSpec) -> pd.DataFrame:
    """Return rows with computable label for a horizon."""
    subset = data.loc[data[horizon.valid_column]].copy()
    subset[horizon.target] = subset[horizon.target].astype(int)
    return subset


def validate_split_classes(data: pd.DataFrame, target: str) -> None:
    """Ensure train, validation, and test splits are usable."""
    for split in ["train", "validation", "test"]:
        values = set(data.loc[data["split"].eq(split), target].dropna().astype(int).unique())
        if values != {0, 1}:
            raise ValueError(f"{target} {split} split must contain both classes, found {values}")


def train_and_evaluate(data: pd.DataFrame) -> pd.DataFrame:
    """Train horizon-specific classifiers and evaluate valid/common-year test rows."""
    families = feature_families(data)
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        horizon_data = subset_for_horizon(data, horizon)
        validate_split_classes(horizon_data, horizon.target)
        splits = {name: horizon_data.loc[horizon_data["split"].eq(name)].copy() for name in ["train", "validation", "test"]}
        y_train = splits["train"][horizon.target].astype(int)
        specs = model_specs(y_train)

        for family_name, features in families.items():
            for model_name, (estimator, needs_scaling) in specs.items():
                pipeline = Pipeline(
                    [
                        ("preprocess", preprocessor(splits["train"], features, needs_scaling)),
                        ("model", estimator),
                    ]
                )
                try:
                    pipeline.fit(splits["train"][features], y_train)
                    validation_scores = pipeline.predict_proba(splits["validation"][features])[:, 1]
                    threshold = select_f2_threshold(splits["validation"][horizon.target].astype(int), validation_scores)
                    for scope, test_subset in [
                        ("horizon_valid", splits["test"]),
                        ("common_years", splits["test"].loc[splits["test"]["label_valid_2year"]].copy()),
                    ]:
                        if test_subset.empty:
                            continue
                        scores = pipeline.predict_proba(test_subset[features])[:, 1]
                        metrics = score_metrics(test_subset[horizon.target].astype(int), scores, threshold)
                        rows.append(
                            {
                                "horizon": horizon.name,
                                "target": horizon.target,
                                "evaluation_scope": scope,
                                "feature_family": family_name,
                                "model": model_name,
                                "decision_threshold": threshold,
                                "train_rows": len(splits["train"]),
                                "validation_rows": len(splits["validation"]),
                                "test_year_min": int(test_subset["year"].min()),
                                "test_year_max": int(test_subset["year"].max()),
                                "feature_count": len(features),
                                "status": "computed",
                                **metrics,
                            }
                        )
                except Exception as exc:  # pragma: no cover
                    rows.append(
                        {
                            "horizon": horizon.name,
                            "target": horizon.target,
                            "evaluation_scope": "not_evaluated",
                            "feature_family": family_name,
                            "model": model_name,
                            "status": "failed",
                            "notes": str(exc),
                        }
                    )
    return pd.DataFrame(rows)


def label_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize label counts and event rates by horizon and split."""
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for scope_name, scope_filter in [
            ("horizon_valid", data[horizon.valid_column]),
            ("common_years", data["label_valid_2year"]),
        ]:
            scoped = data.loc[scope_filter].copy()
            for split_name in ["all", "train", "validation", "test"]:
                split_df = scoped if split_name == "all" else scoped.loc[scoped["split"].eq(split_name)]
                valid = split_df[horizon.target].dropna().astype(int)
                rows.append(
                    {
                        "horizon": horizon.name,
                        "target": horizon.target,
                        "evaluation_scope": scope_name,
                        "split": split_name,
                        "rows": int(len(valid)),
                        "positive_count": int(valid.sum()) if len(valid) else 0,
                        "event_rate": float(valid.mean()) if len(valid) else np.nan,
                        "year_min": int(split_df["year"].min()) if len(split_df) else np.nan,
                        "year_max": int(split_df["year"].max()) if len(split_df) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_summary(data: pd.DataFrame, model_results: pd.DataFrame) -> pd.DataFrame:
    """Build compact horizon-level summary."""
    labels = label_summary(data)
    test_labels = labels.loc[(labels["split"].eq("test")) & (labels["evaluation_scope"].eq("horizon_valid"))]
    computed = model_results.loc[model_results["status"].eq("computed")].copy()
    rows: list[dict[str, object]] = []
    for horizon in HORIZONS:
        for scope in ["horizon_valid", "common_years"]:
            subset = computed.loc[(computed["horizon"].eq(horizon.name)) & (computed["evaluation_scope"].eq(scope))]
            if subset.empty:
                continue
            best = subset.sort_values(["pr_auc", "f2", "recall"], ascending=[False, False, False]).iloc[0]
            current_best = subset.loc[subset["feature_family"].eq("canopy_current_only")].sort_values(
                ["pr_auc", "f2"], ascending=[False, False]
            ).iloc[0]
            trajectory_best = subset.loc[subset["feature_family"].eq("canopy_current_plus_trajectory")].sort_values(
                ["pr_auc", "f2"], ascending=[False, False]
            ).iloc[0]
            env_best = subset.loc[subset["feature_family"].eq("canopy_current_plus_environment")].sort_values(
                ["pr_auc", "f2"], ascending=[False, False]
            ).iloc[0]
            combined_best = subset.loc[
                subset["feature_family"].eq("canopy_current_plus_trajectory_plus_environment")
            ].sort_values(["pr_auc", "f2"], ascending=[False, False]).iloc[0]
            test_label = test_labels.loc[test_labels["horizon"].eq(horizon.name)].iloc[0]
            rows.append(
                {
                    "horizon": horizon.name,
                    "target": horizon.target,
                    "evaluation_scope": scope,
                    "test_year_min": best["test_year_min"],
                    "test_year_max": best["test_year_max"],
                    "test_rows": int(best["total_evaluated_rows"]),
                    "test_positive_count": int(best["positive_count"]),
                    "test_event_rate": float(best["event_rate"]),
                    "horizon_valid_test_positive_count": int(test_label["positive_count"]),
                    "horizon_valid_test_event_rate": float(test_label["event_rate"]),
                    "best_feature_family": best["feature_family"],
                    "best_model": best["model"],
                    "best_pr_auc": float(best["pr_auc"]),
                    "best_recall": float(best["recall"]),
                    "best_precision": float(best["precision"]),
                    "best_f2": float(best["f2"]),
                    "best_false_negatives": int(best["false_negatives"]),
                    "current_only_best_pr_auc": float(current_best["pr_auc"]),
                    "trajectory_best_pr_auc": float(trajectory_best["pr_auc"]),
                    "environment_best_pr_auc": float(env_best["pr_auc"]),
                    "combined_best_pr_auc": float(combined_best["pr_auc"]),
                    "trajectory_minus_current_pr_auc": float(trajectory_best["pr_auc"] - current_best["pr_auc"]),
                    "combined_minus_current_pr_auc": float(combined_best["pr_auc"] - current_best["pr_auc"]),
                    "trajectory_helps_pr_auc": bool(trajectory_best["pr_auc"] > current_best["pr_auc"]),
                }
            )
    return pd.DataFrame(rows)


def build_common_year_comparison(model_results: pd.DataFrame) -> pd.DataFrame:
    """Compare one-year and two-year results on common test years."""
    common = model_results.loc[
        model_results["status"].eq("computed") & model_results["evaluation_scope"].eq("common_years")
    ].copy()
    one = common.loc[common["horizon"].eq("1year")].copy()
    two = common.loc[common["horizon"].eq("2year")].copy()
    compare_cols = [
        "feature_family",
        "model",
        "pr_auc",
        "recall",
        "precision",
        "f1",
        "f2",
        "false_negatives",
        "false_positives",
        "positive_count",
        "total_evaluated_rows",
        "event_rate",
    ]
    merged = one[compare_cols].merge(two[compare_cols], on=["feature_family", "model"], suffixes=("_1year", "_2year"))
    for metric in ["pr_auc", "recall", "f2", "false_negatives", "event_rate"]:
        merged[f"{metric}_2year_minus_1year"] = merged[f"{metric}_2year"] - merged[f"{metric}_1year"]
    return merged.sort_values(["pr_auc_2year", "f2_2year"], ascending=[False, False])


def compact_table(df: pd.DataFrame) -> str:
    """Render a compact Markdown table."""
    if df.empty:
        return "No rows."
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
        else:
            display[col] = display[col].astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row[col] for col in display.columns) + " |" for _, row in display.iterrows()]
    return "\n".join([header, divider, *rows])


def write_report(data: pd.DataFrame, summary: pd.DataFrame, common: pd.DataFrame, output: Path) -> None:
    """Write the multi-horizon diagnostic report."""
    label_counts = label_summary(data)
    split_counts = label_counts.loc[label_counts["evaluation_scope"].eq("horizon_valid")][
        ["horizon", "split", "rows", "positive_count", "event_rate", "year_min", "year_max"]
    ]
    test_counts = label_counts.loc[
        label_counts["split"].eq("test") & label_counts["evaluation_scope"].eq("horizon_valid")
    ][["horizon", "rows", "positive_count", "event_rate", "year_min", "year_max"]]
    best = summary.loc[summary["evaluation_scope"].eq("horizon_valid")][
        [
            "horizon",
            "test_year_min",
            "test_year_max",
            "test_positive_count",
            "test_event_rate",
            "best_feature_family",
            "best_model",
            "best_pr_auc",
            "best_recall",
            "best_precision",
            "best_f2",
            "best_false_negatives",
            "trajectory_minus_current_pr_auc",
        ]
    ]
    common_top = common[
        [
            "feature_family",
            "model",
            "pr_auc_1year",
            "pr_auc_2year",
            "pr_auc_2year_minus_1year",
            "recall_1year",
            "recall_2year",
            "f2_1year",
            "f2_2year",
            "false_negatives_1year",
            "false_negatives_2year",
        ]
    ].head(8)
    two_year_rows = summary.loc[summary["horizon"].eq("2year") & summary["evaluation_scope"].eq("horizon_valid")]
    one_year_rows = summary.loc[summary["horizon"].eq("1year") & summary["evaluation_scope"].eq("horizon_valid")]
    two_vs_one = ""
    if not one_year_rows.empty and not two_year_rows.empty:
        one_best = one_year_rows.iloc[0]
        two_best = two_year_rows.iloc[0]
        if two_best["best_pr_auc"] > one_best["best_pr_auc"]:
            two_vs_one = "The two-year label produced higher best PR-AUC in the horizon-valid test comparison."
        else:
            two_vs_one = "The two-year label did not produce higher best PR-AUC than the one-year label in the horizon-valid test comparison."

    lines = [
        "# Multi-Horizon Actionable Warning Report",
        "",
        "## Purpose",
        "",
        "The project mostly used next-year decline labels. This experiment tests whether actionable sharp canopy-drop risk can be screened at two warning horizons: next year and within the next two years.",
        "The goal is risk-screening evidence for the main machine-learning framing, not a claim of operational early warning.",
        "",
        "## Label Definitions",
        "",
        "- `actionable_decline_drop_next_1year`: current relative canopy > 0.05 and proportional drop from year `t` to `t+1` is at least 30%.",
        "- `actionable_decline_drop_next_2year`: current relative canopy > 0.05 and proportional drop from year `t` to the minimum canopy in `t+1` or `t+2` is at least 30%.",
        "- Rows without required future canopy observations are excluded from that horizon's evaluation.",
        "- Future canopy values are used only to define labels, not as predictors.",
        "",
        "## Test Event Counts",
        "",
        compact_table(test_counts),
        "",
        "Full horizon-valid label counts by split:",
        "",
        compact_table(split_counts),
        "",
        "## Best Horizon-Valid Test Results",
        "",
        compact_table(best),
        "",
        "## Common-Year Comparison",
        "",
        "The common-year comparison uses years where both one-year and two-year labels are available.",
        "",
        compact_table(common_top),
        "",
        "## Interpretation",
        "",
        two_vs_one,
        "The two-year target has a wider event window and a higher test event rate, so higher apparent performance should be read as broader horizon risk screening rather than a cleaner operational warning system.",
        "Trajectory features are considered helpful when `canopy_current_plus_trajectory` improves PR-AUC over `canopy_current_only`; this varies by horizon and should be interpreted as persistence-aware screening rather than ecological mechanism discovery.",
        "This experiment remains an actionable risk-screening diagnostic. It does not prove operational early warning, causal drivers, or spatial transferability.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the multi-horizon actionable warning workflow."""
    args = parse_args()
    data = load_inputs(args.base_data)
    model_results = train_and_evaluate(data)
    summary = build_summary(data, model_results)
    common = build_common_year_comparison(model_results)

    write_portable_csv(model_results, args.model_comparison_output)
    write_portable_csv(summary, args.summary_output)
    write_portable_csv(common, args.common_year_output)
    write_report(data, summary, common, args.report_output)

    print("Multi-horizon actionable warning experiment complete.")
    print(f"Model rows: {len(model_results)}")
    print("Horizon-valid summary:")
    print(
        summary.loc[summary["evaluation_scope"].eq("horizon_valid")][
            [
                "horizon",
                "test_positive_count",
                "test_event_rate",
                "best_feature_family",
                "best_model",
                "best_pr_auc",
                "best_recall",
                "best_precision",
                "best_f2",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote model comparison: {args.model_comparison_output}")
    print(f"Wrote summary: {args.summary_output}")
    print(f"Wrote common-year comparison: {args.common_year_output}")
    print(f"Wrote report: {args.report_output}")


if __name__ == "__main__":
    main()
