"""Diagnose and repair spatial-validation failure modes.

This script treats spatial validation failure as a debugging signal. It uses
only existing Kelpwatch, CRW, habitat, canopy-trajectory, and CDIP wave layers
to build cell-relative, anomaly-based, and percentile-style features, then
tests whether those features improve latitude-band spatial transfer.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "tables"
DIAGNOSTICS_DIR = ROOT / "outputs" / "diagnostics"
SPATIAL_VALIDATION_SCRIPT = ROOT / "scripts" / "26_spatial_validation_diagnostics.py"

SPATIAL_FOLD_RESULTS = RESULTS_DIR / "spatial_validation_fold_results.csv"
SPATIAL_SUMMARY = RESULTS_DIR / "spatial_validation_summary.csv"
SPATIAL_GAP = RESULTS_DIR / "spatial_vs_temporal_validation_gap.csv"
INTEGRATED_MASTER = RESULTS_DIR / "integrated_model_comparison_master.csv"
CLAIM_GATE_SUMMARY = RESULTS_DIR / "claim_gate_summary.csv"

ANATOMY_OUTPUT = RESULTS_DIR / "spatial_failure_anatomy.csv"
SHIFT_OUTPUT = RESULTS_DIR / "spatial_feature_shift_diagnostics.csv"
FEATURE_OUTPUT = ROOT / "data" / "processed" / "spatial_repair_features.csv"
FEATURE_DIAGNOSTICS_OUTPUT = RESULTS_DIR / "spatial_repair_feature_diagnostics.csv"
MODEL_COMPARISON_OUTPUT = RESULTS_DIR / "spatial_repair_model_comparison.csv"
STABILITY_OUTPUT = RESULTS_DIR / "spatial_repair_stability_ranking.csv"
THRESHOLD_OUTPUT = RESULTS_DIR / "spatial_repair_threshold_sensitivity.csv"
REPORT_OUTPUT = DIAGNOSTICS_DIR / "spatial_failure_repair_report.md"

CSV_WRITE_KWARGS = {
    "index": False,
    "lineterminator": "\n",
    "na_rep": "",
    "float_format": "%.6f",
}

CANOPY_RELATIVE_FEATURES = [
    "canopy_current_relative_to_cell_median",
    "canopy_current_relative_to_cell_max",
    "canopy_current_percentile_within_cell",
    "canopy_drop_from_recent_peak_5yr",
    "canopy_drop_from_historical_peak",
    "canopy_3yr_slope_relative_to_cell_median",
    "canopy_5yr_slope_relative_to_cell_median",
]

THERMAL_RELATIVE_FEATURES = [
    "spring_ssta_crw5km_local_z",
    "summer_ssta_crw5km_local_z",
    "annual_mean_ssta_crw5km_local_z",
    "warmest_month_mean_sst_crw5km_percentile_within_cell",
    "spring_ssta_crw5km_percentile_within_cell",
    "summer_ssta_crw5km_percentile_within_cell",
]

WAVE_RELATIVE_FEATURES = [
    "winter_max_wave_height_local_z",
    "winter_mean_wave_height_local_z",
    "annual_max_wave_height_percentile_within_wave_source",
    "lag1_winter_max_wave_height_local_z",
]

REPAIR_INTERACTION_FEATURES = [
    "spring_ssta_local_z_x_shallow_area_share_0_30m",
    "summer_ssta_local_z_x_shallow_area_share_0_30m",
    "winter_wave_local_z_x_shallow_area_share_0_30m",
    "canopy_drop_from_recent_peak_5yr_x_spring_ssta_local_z",
    "canopy_3yr_slope_relative_x_winter_wave_local_z",
]

KEY_SHIFT_FEATURES = [
    "relative_canopy",
    "canopy_3yr_slope_t",
    "canopy_5yr_slope_t",
    "instability_score_5yr_t",
    "presence_frequency_5yr_t",
    "annual_mean_sst_crw5km",
    "spring_ssta_crw5km",
    "summer_ssta_crw5km",
    "mean_depth_m",
    "shallow_area_share_0_30m",
    "slope_mean",
    "winter_max_wave_height_cdip_model",
    "distance_to_cdip_model_point_km",
]

MIN_TEST_POSITIVES = 5
MIN_TRAIN_POSITIVES = 10


def load_spatial_validation_module():
    """Load script 26 as a helper module despite its numeric filename."""
    spec = importlib.util.spec_from_file_location("spatial_validation_diagnostics", SPATIAL_VALIDATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {SPATIAL_VALIDATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sv = load_spatial_validation_module()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run spatial failure repair diagnostics.")
    parser.add_argument("--input", type=Path, default=sv.INPUT_DATASET)
    parser.add_argument("--anatomy-output", type=Path, default=ANATOMY_OUTPUT)
    parser.add_argument("--shift-output", type=Path, default=SHIFT_OUTPUT)
    parser.add_argument("--feature-output", type=Path, default=FEATURE_OUTPUT)
    parser.add_argument("--feature-diagnostics-output", type=Path, default=FEATURE_DIAGNOSTICS_OUTPUT)
    parser.add_argument("--model-comparison-output", type=Path, default=MODEL_COMPARISON_OUTPUT)
    parser.add_argument("--stability-output", type=Path, default=STABILITY_OUTPUT)
    parser.add_argument("--threshold-output", type=Path, default=THRESHOLD_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    parser.add_argument(
        "--include-boosting",
        action="store_true",
        help="Also run XGBoost and LightGBM. Default uses Logistic Regression and Random Forest for runtime stability.",
    )
    return parser.parse_args()


def clean_csv_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten embedded line breaks for GitHub and pandas-friendly CSVs."""
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
    """Write stable LF CSV files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = clean_csv_cells(df).to_csv(**CSV_WRITE_KWARGS)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def expanding_percentile(values: pd.Series) -> pd.Series:
    """Percentile rank of current value within same-cell history through year t."""
    history: list[float] = []
    out: list[float] = []
    for value in values.to_numpy(dtype=float):
        if np.isnan(value):
            out.append(np.nan)
            history.append(value)
            continue
        hist = np.asarray([v for v in history if not np.isnan(v)] + [value], dtype=float)
        out.append(float(np.mean(hist <= value)))
        history.append(value)
    return pd.Series(out, index=values.index)


def training_reference_zscore(data: pd.DataFrame, group_col: str, value_col: str, out_col: str) -> pd.Series:
    """Compute z-score against 1989-2016 within-group reference values."""
    reference = data.loc[data["year"].between(1989, 2016)].groupby(group_col)[value_col].agg(["mean", "std"])
    means = data[group_col].map(reference["mean"])
    stds = data[group_col].map(reference["std"]).replace(0, np.nan)
    return (data[value_col] - means) / stds


def percentile_against_training_reference(data: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    """Percentile of each value against 1989-2016 same-group reference values."""
    references = {
        group: values.dropna().to_numpy(dtype=float)
        for group, values in data.loc[data["year"].between(1989, 2016)].groupby(group_col)[value_col]
    }
    output: list[float] = []
    for group, value in zip(data[group_col], data[value_col], strict=False):
        ref = references.get(group)
        if ref is None or len(ref) == 0 or pd.isna(value):
            output.append(np.nan)
        else:
            output.append(float(np.mean(ref <= float(value))))
    return pd.Series(output, index=data.index)


def add_spatial_repair_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add leakage-aware cell-relative and anomaly features."""
    out = data.sort_values(["cell_id", "year"]).copy()
    grouped = out.groupby("cell_id", group_keys=False)
    eps = 1e-6

    expanding_median = grouped[sv.CANOPY].expanding(min_periods=1).median().reset_index(level=0, drop=True)
    expanding_max = grouped[sv.CANOPY].expanding(min_periods=1).max().reset_index(level=0, drop=True)
    expanding_mean = grouped[sv.CANOPY].expanding(min_periods=2).mean().reset_index(level=0, drop=True)
    expanding_std = grouped[sv.CANOPY].expanding(min_periods=3).std().reset_index(level=0, drop=True)
    recent_peak_5yr = grouped[sv.CANOPY].rolling(5, min_periods=1).max().reset_index(level=0, drop=True)

    out["canopy_current_relative_to_cell_median"] = out[sv.CANOPY] / np.maximum(expanding_median, eps)
    out["canopy_current_relative_to_cell_max"] = out[sv.CANOPY] / np.maximum(expanding_max, eps)
    out["canopy_current_percentile_within_cell"] = grouped[sv.CANOPY].apply(expanding_percentile).reset_index(level=0, drop=True)
    out["canopy_drop_from_recent_peak_5yr"] = (recent_peak_5yr - out[sv.CANOPY]) / np.maximum(recent_peak_5yr, eps)
    out["canopy_drop_from_historical_peak"] = (expanding_max - out[sv.CANOPY]) / np.maximum(expanding_max, eps)
    out["canopy_anomaly_from_expanding_median"] = out[sv.CANOPY] - expanding_median
    out["canopy_zscore_expanding"] = (out[sv.CANOPY] - expanding_mean) / expanding_std.replace(0, np.nan)

    if "canopy_3yr_slope_t" in out.columns:
        out["canopy_3yr_slope_relative_to_cell_median"] = out["canopy_3yr_slope_t"] / np.maximum(expanding_median, eps)
        out["canopy_3yr_slope_relative"] = out["canopy_3yr_slope_relative_to_cell_median"]
    if "canopy_5yr_slope_t" in out.columns:
        out["canopy_5yr_slope_relative_to_cell_median"] = out["canopy_5yr_slope_t"] / np.maximum(expanding_median, eps)

    thermal_map = {
        "spring_ssta_crw5km_local_z": "spring_ssta_crw5km",
        "summer_ssta_crw5km_local_z": "summer_ssta_crw5km",
        "annual_mean_ssta_crw5km_local_z": "annual_mean_ssta_crw5km",
    }
    for out_col, source_col in thermal_map.items():
        if source_col in out.columns:
            out[out_col] = training_reference_zscore(out, "cell_id", source_col, out_col)
    percentile_map = {
        "warmest_month_mean_sst_crw5km_percentile_within_cell": "warmest_month_mean_sst_crw5km",
        "spring_ssta_crw5km_percentile_within_cell": "spring_ssta_crw5km",
        "summer_ssta_crw5km_percentile_within_cell": "summer_ssta_crw5km",
    }
    for out_col, source_col in percentile_map.items():
        if source_col in out.columns:
            out[out_col] = percentile_against_training_reference(out, "cell_id", source_col)

    wave_map = {
        "winter_max_wave_height_local_z": "winter_max_wave_height_cdip_model",
        "winter_mean_wave_height_local_z": "winter_mean_wave_height_cdip_model",
        "lag1_winter_max_wave_height_local_z": "lag1_winter_max_wave_height_cdip_model",
    }
    for out_col, source_col in wave_map.items():
        if source_col in out.columns:
            group_col = "cdip_mop_id" if "cdip_mop_id" in out.columns else "cell_id"
            out[out_col] = training_reference_zscore(out, group_col, source_col, out_col)
    if "annual_max_wave_height_cdip_model" in out.columns:
        group_col = "cdip_mop_id" if "cdip_mop_id" in out.columns else "cell_id"
        out["annual_max_wave_height_percentile_within_wave_source"] = percentile_against_training_reference(
            out, group_col, "annual_max_wave_height_cdip_model"
        )

    if {"spring_ssta_crw5km_local_z", "shallow_area_share_0_30m"}.issubset(out.columns):
        out["spring_ssta_local_z_x_shallow_area_share_0_30m"] = out["spring_ssta_crw5km_local_z"] * out["shallow_area_share_0_30m"]
    if {"summer_ssta_crw5km_local_z", "shallow_area_share_0_30m"}.issubset(out.columns):
        out["summer_ssta_local_z_x_shallow_area_share_0_30m"] = out["summer_ssta_crw5km_local_z"] * out["shallow_area_share_0_30m"]
    if {"winter_max_wave_height_local_z", "shallow_area_share_0_30m"}.issubset(out.columns):
        out["winter_wave_local_z_x_shallow_area_share_0_30m"] = out["winter_max_wave_height_local_z"] * out["shallow_area_share_0_30m"]
    if {"canopy_drop_from_recent_peak_5yr", "spring_ssta_crw5km_local_z"}.issubset(out.columns):
        out["canopy_drop_from_recent_peak_5yr_x_spring_ssta_local_z"] = (
            out["canopy_drop_from_recent_peak_5yr"] * out["spring_ssta_crw5km_local_z"]
        )
    if {"canopy_3yr_slope_relative", "winter_max_wave_height_local_z"}.issubset(out.columns):
        out["canopy_3yr_slope_relative_x_winter_wave_local_z"] = (
            out["canopy_3yr_slope_relative"] * out["winter_max_wave_height_local_z"]
        )

    repair_features = (
        [feature for feature in CANOPY_RELATIVE_FEATURES if feature in out.columns]
        + [feature for feature in THERMAL_RELATIVE_FEATURES if feature in out.columns]
        + [feature for feature in WAVE_RELATIVE_FEATURES if feature in out.columns]
        + [feature for feature in REPAIR_INTERACTION_FEATURES if feature in out.columns]
    )
    return out, repair_features


def spatial_failure_anatomy() -> pd.DataFrame:
    """Diagnose target and band-level spatial validation failure modes."""
    fold = pd.read_csv(SPATIAL_FOLD_RESULTS)
    gap = pd.read_csv(SPATIAL_GAP)
    computed = fold.loc[fold["status"].eq("computed")].copy()
    best_band = (
        computed.sort_values(["validation_scheme", "heldout_band", "target_definition", "pr_auc"], ascending=[True, True, True, False])
        .groupby(["validation_scheme", "heldout_band", "target_definition"])
        .head(1)
        .reset_index(drop=True)
    )
    best_band["positive_rate"] = best_band["test_positive_count"] / best_band["test_rows"].replace(0, np.nan)
    best_band["ranking_failure_flag"] = best_band["pr_auc"] < 0.40
    best_band["threshold_failure_flag"] = (best_band["recall"] < 0.50) | (best_band["f2"] < 0.50)
    best_band["failure_mode"] = np.select(
        [
            best_band["ranking_failure_flag"] & best_band["threshold_failure_flag"],
            best_band["ranking_failure_flag"],
            best_band["threshold_failure_flag"],
        ],
        ["ranking_and_threshold_failure", "ranking_failure", "threshold_failure"],
        default="no_major_failure_in_best_row",
    )

    target_loss = (
        gap.groupby("target_definition", as_index=False)
        .agg(
            worst_spatial_minus_temporal_pr_auc=("spatial_minus_temporal_pr_auc", "min"),
            worst_spatial_minus_temporal_recall=("spatial_minus_temporal_recall", "min"),
        )
        .sort_values("worst_spatial_minus_temporal_pr_auc")
    )
    pr_loss_target = target_loss.iloc[0]["target_definition"] if not target_loss.empty else ""
    recall_loss_target = target_loss.sort_values("worst_spatial_minus_temporal_recall").iloc[0]["target_definition"] if not target_loss.empty else ""

    best_band["target_with_largest_pr_auc_loss"] = pr_loss_target
    best_band["target_with_largest_recall_loss"] = recall_loss_target
    worst_band = (
        best_band.loc[best_band["validation_scheme"].eq("three_band_holdout")]
        .groupby("heldout_band")["pr_auc"]
        .mean()
        .sort_values()
    )
    best_band["worst_three_band_by_mean_pr_auc"] = worst_band.index[0] if not worst_band.empty else ""
    return best_band[
        [
            "validation_scheme",
            "heldout_band",
            "target_definition",
            "feature_family",
            "model",
            "positive_rate",
            "recall",
            "precision",
            "false_negatives",
            "false_positives",
            "pr_auc",
            "roc_auc",
            "ranking_failure_flag",
            "threshold_failure_flag",
            "failure_mode",
            "target_with_largest_pr_auc_loss",
            "target_with_largest_recall_loss",
            "worst_three_band_by_mean_pr_auc",
        ]
    ]


def iqr(series: pd.Series) -> float:
    """Return interquartile range."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    return float(numeric.quantile(0.75) - numeric.quantile(0.25))


def feature_shift_diagnostics(data: pd.DataFrame, repair_features: list[str]) -> pd.DataFrame:
    """Compare train-band and held-out-band feature distributions."""
    candidate_features = [f for f in KEY_SHIFT_FEATURES + repair_features if f in data.columns]
    rows: list[dict[str, object]] = []
    for scheme, band_col in [("three_band_holdout", "3_band"), ("five_band_holdout", "5_band")]:
        for band in sorted(data[band_col].astype(str).unique()):
            train = data.loc[data[band_col].astype(str) != band]
            heldout = data.loc[data[band_col].astype(str) == band]
            for feature in candidate_features:
                train_values = pd.to_numeric(train[feature], errors="coerce")
                heldout_values = pd.to_numeric(heldout[feature], errors="coerce")
                train_std = float(train_values.std())
                heldout_std = float(heldout_values.std())
                pooled_std = np.sqrt(np.nanmean([train_std**2, heldout_std**2]))
                mean_diff = float(heldout_values.mean() - train_values.mean())
                rows.append(
                    {
                        "validation_scheme": scheme,
                        "heldout_band": band,
                        "feature": feature,
                        "train_mean": float(train_values.mean()),
                        "heldout_mean": float(heldout_values.mean()),
                        "mean_difference": mean_diff,
                        "standardized_mean_difference": mean_diff / pooled_std if pooled_std and not np.isnan(pooled_std) else np.nan,
                        "train_median": float(train_values.median()),
                        "heldout_median": float(heldout_values.median()),
                        "median_difference": float(heldout_values.median() - train_values.median()),
                        "train_iqr": iqr(train_values),
                        "heldout_iqr": iqr(heldout_values),
                        "iqr_difference": iqr(heldout_values) - iqr(train_values),
                        "train_missingness": float(train_values.isna().mean()),
                        "heldout_missingness": float(heldout_values.isna().mean()),
                        "missingness_difference": float(heldout_values.isna().mean() - train_values.isna().mean()),
                        "abs_standardized_mean_difference": abs(mean_diff / pooled_std) if pooled_std and not np.isnan(pooled_std) else np.nan,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["validation_scheme", "abs_standardized_mean_difference"],
        ascending=[True, False],
    )


def repair_feature_sets(data: pd.DataFrame, repair_features: list[str]) -> dict[str, list[str]]:
    """Define spatial repair feature families."""
    canopy_relative = [feature for feature in CANOPY_RELATIVE_FEATURES if feature in data.columns]
    thermal_relative = [feature for feature in THERMAL_RELATIVE_FEATURES if feature in data.columns]
    wave_relative = [feature for feature in WAVE_RELATIVE_FEATURES if feature in data.columns]
    interactions = [feature for feature in REPAIR_INTERACTION_FEATURES if feature in data.columns]
    dynamic_relative = canopy_relative + thermal_relative + wave_relative
    sets = {
        "canopy_absolute_baseline": sv.available(sv.CANOPY_FEATURES, data),
        "canopy_relative": canopy_relative,
        "thermal_absolute": sv.available(sv.CRW_COMPOSITE_FEATURES, data),
        "thermal_relative": thermal_relative,
        "trajectory_absolute": sv.available(sv.TRAJECTORY_FEATURES, data),
        "trajectory_relative": [
            feature
            for feature in [
                "canopy_3yr_slope_relative_to_cell_median",
                "canopy_5yr_slope_relative_to_cell_median",
                "canopy_drop_from_recent_peak_5yr",
                "canopy_drop_from_historical_peak",
            ]
            if feature in data.columns
        ],
        "relative_dynamic_combined": dynamic_relative,
        "relative_dynamic_plus_habitat_context": dynamic_relative
        + sv.available(sv.HABITAT_FEATURES, data)
        + interactions,
    }
    return {name: features for name, features in sets.items() if features}


def repair_estimators(include_boosting: bool = False) -> dict[str, object]:
    """Return compact estimators.

    Boosting models are optional because the spatial-repair diagnostic fits many
    held-out folds and threshold-sensitivity variants. The default keeps the
    command-line run reproducible on a laptop while still testing a linear and a
    nonlinear tree-based model.
    """
    models: dict[str, object] = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=3,
            n_jobs=1,
        ),
    }
    if include_boosting and XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
    if include_boosting and LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
    return models


def evaluate_repair_fold(
    data: pd.DataFrame,
    validation_scheme: str,
    band_column: str,
    heldout_band: str,
    target,
    feature_families: dict[str, list[str]],
    include_boosting: bool = False,
) -> list[dict[str, object]]:
    """Evaluate repair feature families for one spatial fold."""
    working = data.copy()
    if target.filter_column:
        working = working.loc[working[target.filter_column]].copy()
    train = working.loc[working[band_column].astype(str) != heldout_band].copy()
    test = working.loc[working[band_column].astype(str) == heldout_band].copy()
    base = {
        "validation_scheme": validation_scheme,
        "heldout_band": heldout_band,
        "target_definition": target.name,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_positive_count": int(train[target.target].sum()) if len(train) else 0,
        "test_positive_count": int(test[target.target].sum()) if len(test) else 0,
    }
    if base["train_positive_count"] < MIN_TRAIN_POSITIVES or base["test_positive_count"] < MIN_TEST_POSITIVES:
        return [
            {
                **base,
                "feature_family": family,
                "model": "not_run",
                "status": "underpowered_fold",
                "notes": "Insufficient train or held-out positives.",
            }
            for family in feature_families
        ]
    rows: list[dict[str, object]] = []
    for family, features in feature_families.items():
        for model_name, estimator in repair_estimators(include_boosting).items():
            pipeline = Pipeline([("preprocess", sv.preprocess(features)), ("model", estimator)])
            try:
                pipeline.fit(train[features], train[target.target].astype(int))
                train_scores = pipeline.predict_proba(train[features])[:, 1]
                test_scores = pipeline.predict_proba(test[features])[:, 1]
                threshold = sv.select_threshold(train[target.target], train_scores)
                metrics = sv.score_metrics(test[target.target].astype(int), test_scores, threshold)
                status = "computed"
                notes = "Cell-relative/anomaly repair features; threshold selected on non-held-out bands."
            except Exception as exc:  # pragma: no cover
                metrics = {}
                threshold = np.nan
                status = "failed"
                notes = str(exc)
            rows.append(
                {
                    **base,
                    **metrics,
                    "feature_family": family,
                    "model": model_name,
                    "decision_threshold": threshold,
                    "status": status,
                    "notes": notes,
                }
            )
    return rows


def evaluate_repair_models(data: pd.DataFrame, repair_features: list[str], include_boosting: bool = False) -> pd.DataFrame:
    """Run spatial holdout comparison for repair feature families."""
    families = repair_feature_sets(data, repair_features)
    rows: list[dict[str, object]] = []
    for scheme, band_column in [("three_band_holdout", "3_band"), ("five_band_holdout", "5_band")]:
        for target in sv.TARGETS:
            for band in sorted(data[band_column].astype(str).unique()):
                rows.extend(evaluate_repair_fold(data, scheme, band_column, band, target, families, include_boosting))
    return pd.DataFrame(rows)


def summarize_repair_results(repair_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize repair model results using best PR-AUC row per held-out fold."""
    computed = repair_results.loc[repair_results["status"].eq("computed")].copy()
    if computed.empty:
        return pd.DataFrame()
    best = (
        computed.sort_values(["validation_scheme", "heldout_band", "target_definition", "feature_family", "pr_auc"], ascending=[True, True, True, True, False])
        .groupby(["validation_scheme", "heldout_band", "target_definition", "feature_family"])
        .head(1)
    )
    rows: list[dict[str, object]] = []
    for keys, group in best.groupby(["validation_scheme", "target_definition", "feature_family"], dropna=False):
        rows.append(
            {
                "validation_scheme": keys[0],
                "target_definition": keys[1],
                "feature_family": keys[2],
                "mean_pr_auc": float(group["pr_auc"].mean()),
                "median_pr_auc": float(group["pr_auc"].median()),
                "min_pr_auc": float(group["pr_auc"].min()),
                "max_pr_auc": float(group["pr_auc"].max()),
                "mean_recall": float(group["recall"].mean()),
                "mean_precision": float(group["precision"].mean()),
                "mean_f2": float(group["f2"].mean()),
                "total_false_negatives": int(group["false_negatives"].sum()),
                "total_positives_across_heldout_folds": int(group["test_positive_count"].sum()),
                "valid_folds": int(group["heldout_band"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def stability_ranking(repair_results: pd.DataFrame) -> pd.DataFrame:
    """Create stability-aware ranking by target, feature family, and model."""
    computed = repair_results.loc[repair_results["status"].eq("computed")].copy()
    rows: list[dict[str, object]] = []
    for keys, group in computed.groupby(["validation_scheme", "target_definition", "feature_family", "model"], dropna=False):
        mean_pr = float(group["pr_auc"].mean())
        min_pr = float(group["pr_auc"].min())
        mean_recall = float(group["recall"].mean())
        min_recall = float(group["recall"].min())
        mean_precision = float(group["precision"].mean())
        min_precision = float(group["precision"].min())
        mean_f2 = float(group["f2"].mean())
        min_f2 = float(group["f2"].min())
        std_pr = float(group["pr_auc"].std(ddof=0))
        std_recall = float(group["recall"].std(ddof=0))
        rows.append(
            {
                "validation_scheme": keys[0],
                "target_definition": keys[1],
                "feature_family": keys[2],
                "model": keys[3],
                "mean_pr_auc": mean_pr,
                "min_pr_auc": min_pr,
                "mean_recall": mean_recall,
                "min_recall": min_recall,
                "mean_precision": mean_precision,
                "min_precision": min_precision,
                "mean_f2": mean_f2,
                "min_f2": min_f2,
                "total_false_negatives": int(group["false_negatives"].sum()),
                "total_positives_across_heldout_folds": int(group["test_positive_count"].sum()),
                "pr_auc_std": std_pr,
                "recall_std": std_recall,
                "valid_folds": int(group["heldout_band"].nunique()),
                "stability_score": min_pr + 0.25 * min_f2 - 0.10 * std_pr,
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return ranking
    return ranking.sort_values(
        ["validation_scheme", "target_definition", "stability_score", "min_pr_auc", "mean_pr_auc"],
        ascending=[True, True, False, False, False],
    )


def threshold_for_precision_floor(y_true: pd.Series, scores: np.ndarray, precision_floor: float) -> float | None:
    """Select train threshold maximizing recall while meeting a precision floor."""
    precision, recall, thresholds = sv.precision_recall_curve(y_true.astype(int), scores)
    if len(thresholds) == 0:
        return None
    candidates = pd.DataFrame({"precision": precision[:-1], "recall": recall[:-1], "threshold": thresholds})
    candidates = candidates.loc[candidates["precision"] >= precision_floor]
    if candidates.empty:
        return None
    candidates = candidates.sort_values(["recall", "precision", "threshold"], ascending=[False, False, True])
    return float(candidates.iloc[0]["threshold"])


def threshold_sensitivity(data: pd.DataFrame, repair_features: list[str], include_boosting: bool = False) -> pd.DataFrame:
    """Evaluate spatial-training threshold recalibration for primary targets."""
    families = repair_feature_sets(data, repair_features)
    floors = [0.30, 0.40, 0.50]
    rows: list[dict[str, object]] = []
    for scheme, band_column in [("three_band_holdout", "3_band"), ("five_band_holdout", "5_band")]:
        for target in [t for t in sv.TARGETS if t.name in {"at_risk_original", "actionable_drop"}]:
            for heldout_band in sorted(data[band_column].astype(str).unique()):
                working = data.loc[data[target.filter_column]].copy() if target.filter_column else data.copy()
                train = working.loc[working[band_column].astype(str) != heldout_band].copy()
                test = working.loc[working[band_column].astype(str) == heldout_band].copy()
                if int(train[target.target].sum()) < MIN_TRAIN_POSITIVES or int(test[target.target].sum()) < MIN_TEST_POSITIVES:
                    continue
                for family, features in families.items():
                    for model_name, estimator in repair_estimators(include_boosting).items():
                        pipeline = Pipeline([("preprocess", sv.preprocess(features)), ("model", estimator)])
                        try:
                            pipeline.fit(train[features], train[target.target].astype(int))
                            train_scores = pipeline.predict_proba(train[features])[:, 1]
                            test_scores = pipeline.predict_proba(test[features])[:, 1]
                            for floor in floors:
                                threshold = threshold_for_precision_floor(train[target.target], train_scores, floor)
                                if threshold is None:
                                    rows.append(
                                        {
                                            "validation_scheme": scheme,
                                            "heldout_band": heldout_band,
                                            "target_definition": target.name,
                                            "feature_family": family,
                                            "model": model_name,
                                            "precision_floor": floor,
                                            "status": "no_training_threshold_met_precision_floor",
                                            "train_rows": len(train),
                                            "test_rows": len(test),
                                            "test_positive_count": int(test[target.target].sum()),
                                        }
                                    )
                                    continue
                                metrics = sv.score_metrics(test[target.target].astype(int), test_scores, threshold)
                                rows.append(
                                    {
                                        "validation_scheme": scheme,
                                        "heldout_band": heldout_band,
                                        "target_definition": target.name,
                                        "feature_family": family,
                                        "model": model_name,
                                        "precision_floor": floor,
                                        "decision_threshold": threshold,
                                        "train_rows": len(train),
                                        "test_rows": len(test),
                                        "test_positive_count": int(test[target.target].sum()),
                                        **metrics,
                                        "status": "computed",
                                    }
                                )
                        except Exception as exc:  # pragma: no cover
                            rows.append(
                                {
                                    "validation_scheme": scheme,
                                    "heldout_band": heldout_band,
                                    "target_definition": target.name,
                                    "feature_family": family,
                                    "model": model_name,
                                    "precision_floor": np.nan,
                                    "status": "failed",
                                    "notes": str(exc),
                                }
                            )
    return pd.DataFrame(rows)


def feature_diagnostics(data: pd.DataFrame, repair_features: list[str]) -> pd.DataFrame:
    """Summarize repair feature missingness and coverage."""
    rows: list[dict[str, object]] = []
    for feature in repair_features:
        values = pd.to_numeric(data[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "missingness": float(values.isna().mean()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "feature_type": (
                    "canopy_relative"
                    if feature in CANOPY_RELATIVE_FEATURES
                    else "thermal_relative"
                    if feature in THERMAL_RELATIVE_FEATURES
                    else "wave_relative"
                    if feature in WAVE_RELATIVE_FEATURES
                    else "interaction"
                ),
            }
        )
    return pd.DataFrame(rows)


def compact_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table."""
    if frame.empty:
        return "No rows."
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
        else:
            display[column] = display[column].astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row[col] for col in display.columns) + " |" for _, row in display.iterrows()]
    return "\n".join([header, divider, *rows])


def write_report(
    output: Path,
    anatomy: pd.DataFrame,
    shift: pd.DataFrame,
    feature_diag: pd.DataFrame,
    stability: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> None:
    """Write spatial failure repair report."""
    primary_targets = stability.loc[
        stability["validation_scheme"].eq("three_band_holdout")
        & stability["target_definition"].isin(["at_risk_original", "actionable_drop"])
    ].copy()
    best_mean = primary_targets.sort_values(["target_definition", "mean_pr_auc"], ascending=[True, False]).groupby("target_definition").head(1)
    best_worst = primary_targets.sort_values(["target_definition", "min_pr_auc"], ascending=[True, False]).groupby("target_definition").head(1)
    best_stable = primary_targets.sort_values(["target_definition", "stability_score"], ascending=[True, False]).groupby("target_definition").head(1)
    threshold_primary = thresholds.loc[
        thresholds["status"].eq("computed")
        & thresholds["validation_scheme"].eq("three_band_holdout")
        & thresholds["target_definition"].isin(["at_risk_original", "actionable_drop"])
    ].copy()
    threshold_agg = pd.DataFrame()
    if not threshold_primary.empty:
        threshold_agg = (
            threshold_primary.groupby(
                ["target_definition", "precision_floor", "feature_family", "model"],
                as_index=False,
                dropna=False,
            )
            .agg(
                mean_recall=("recall", "mean"),
                mean_precision=("precision", "mean"),
                min_precision=("precision", "min"),
                mean_f2=("f2", "mean"),
                total_false_negatives=("false_negatives", "sum"),
                total_false_positives=("false_positives", "sum"),
                valid_folds=("heldout_band", "nunique"),
            )
            .sort_values(["target_definition", "precision_floor", "mean_f2"], ascending=[True, True, False])
        )
    threshold_best = threshold_agg.groupby(["target_definition", "precision_floor"]).head(1) if not threshold_agg.empty else pd.DataFrame()
    worst_pr = anatomy.loc[anatomy["validation_scheme"].eq("three_band_holdout"), "target_with_largest_pr_auc_loss"].mode()
    worst_recall = anatomy.loc[anatomy["validation_scheme"].eq("three_band_holdout"), "target_with_largest_recall_loss"].mode()
    worst_band = anatomy.loc[anatomy["validation_scheme"].eq("three_band_holdout"), "worst_three_band_by_mean_pr_auc"].mode()
    top_shift = shift.loc[shift["validation_scheme"].eq("three_band_holdout")].head(15)

    lines = [
        "# Spatial Failure Repair Report",
        "",
        "## Purpose",
        "",
        "This diagnostic treats spatial-validation weakness as a model-debugging signal rather than only a limitation.",
        "It tests whether replacing or supplementing absolute predictors with cell-relative, anomaly-based, and percentile-style features improves latitude-band spatial transfer.",
        "",
        "## Spatial Failure Anatomy",
        "",
        f"- Target with largest PR-AUC loss under existing spatial validation: `{worst_pr.iloc[0] if not worst_pr.empty else 'unknown'}`",
        f"- Target with largest recall loss under existing spatial validation: `{worst_recall.iloc[0] if not worst_recall.empty else 'unknown'}`",
        f"- Worst three-band held-out band by mean best-row PR-AUC: `{worst_band.iloc[0] if not worst_band.empty else 'unknown'}`",
        "",
        "Best existing model row by target and held-out band:",
        "",
        compact_table(
            anatomy.loc[anatomy["validation_scheme"].eq("three_band_holdout")][
                [
                    "heldout_band",
                    "target_definition",
                    "feature_family",
                    "model",
                    "positive_rate",
                    "pr_auc",
                    "recall",
                    "precision",
                    "false_negatives",
                    "failure_mode",
                ]
            ]
        ),
        "",
        "## Feature Distribution Shift",
        "",
        "Largest three-band standardized mean shifts:",
        "",
        compact_table(
            top_shift[
                [
                    "heldout_band",
                    "feature",
                    "standardized_mean_difference",
                    "median_difference",
                    "missingness_difference",
                ]
            ]
        ),
        "",
        "## Repair Features",
        "",
        "- Canopy repair features are expanding or rolling within-cell summaries available up to year `t`.",
        "- Environmental repair features are training-period local z-scores or percentiles from existing CRW and CDIP wave variables.",
        "- These features do not add new external data sources.",
        "- The default run uses Logistic Regression and Random Forest for runtime stability; optional boosting models can be enabled with `--include-boosting`.",
        "",
        f"- Repair features built: `{len(feature_diag)}`",
        f"- Maximum repair-feature missingness: `{feature_diag['missingness'].max():.3f}`",
        "",
        "## Spatial Repair Results",
        "",
        "Best three-band repair model by mean spatial PR-AUC:",
        "",
        compact_table(
            best_mean[
                [
                    "target_definition",
                    "feature_family",
                    "model",
                    "mean_pr_auc",
                    "min_pr_auc",
                    "mean_recall",
                    "min_recall",
                    "mean_precision",
                    "mean_f2",
                    "total_false_negatives",
                    "valid_folds",
                ]
            ]
        ),
        "",
        "Best three-band repair model by worst-band PR-AUC:",
        "",
        compact_table(
            best_worst[
                [
                    "target_definition",
                    "feature_family",
                    "model",
                    "mean_pr_auc",
                    "min_pr_auc",
                    "mean_recall",
                    "min_recall",
                    "mean_f2",
                    "total_false_negatives",
                    "valid_folds",
                ]
            ]
        ),
        "",
        "Top stability-ranked three-band repair models for primary targets:",
        "",
        compact_table(
            best_stable[
                [
                    "target_definition",
                    "feature_family",
                    "model",
                    "stability_score",
                    "mean_pr_auc",
                    "min_pr_auc",
                    "mean_recall",
                    "min_recall",
                    "pr_auc_std",
                    "recall_std",
                ]
            ]
        ),
        "",
        "Precision-floor threshold recalibration highlights aggregated across three-band holdouts:",
        "",
        "Precision floors are enforced during spatial-training threshold selection; held-out precision can still fall below the requested floor.",
        "",
        compact_table(
            threshold_best[
                [
                    "target_definition",
                    "precision_floor",
                    "feature_family",
                    "model",
                    "mean_recall",
                    "mean_precision",
                    "min_precision",
                    "mean_f2",
                    "total_false_negatives",
                    "total_false_positives",
                    "valid_folds",
                ]
            ].head(18)
        ),
        "",
        "## Interpretation",
        "",
        "Spatial repair features should be interpreted as internal robustness diagnostics, not as proof of external spatial generalization.",
        "If repair features improve PR-AUC but recall remains low, the result supports better ranking under spatial transfer but not operational early warning.",
        "If recall improves only with low precision, the result supports recall-oriented alert sensitivity rather than robust spatially transferable prediction.",
        "",
        "## Next Step",
        "",
        "The project can stop adding feature layers for the current V1 portfolio and frame external region validation as the next methodological step.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run spatial failure repair diagnostics."""
    args = parse_args()
    data = sv.add_spatial_bands(sv.load_data(args.input))
    if sv.WAVE_FEATURE_PATH.exists() and "cdip_mop_id" not in data.columns:
        wave_source = pd.read_csv(sv.WAVE_FEATURE_PATH)[["cell_id", "year", "cdip_mop_id"]].drop_duplicates()
        data = data.merge(wave_source, on=["cell_id", "year"], how="left", validate="one_to_one")
    data, repair_features = add_spatial_repair_features(data)

    anatomy = spatial_failure_anatomy()
    shift = feature_shift_diagnostics(data, repair_features)
    feature_diag = feature_diagnostics(data, repair_features)
    repair_results = evaluate_repair_models(data, repair_features, args.include_boosting)
    stability = stability_ranking(repair_results)
    thresholds = threshold_sensitivity(data, repair_features, args.include_boosting)

    write_portable_csv(anatomy, args.anatomy_output)
    write_portable_csv(shift, args.shift_output)
    write_portable_csv(data[["cell_id", "year", *repair_features]], args.feature_output)
    write_portable_csv(feature_diag, args.feature_diagnostics_output)
    write_portable_csv(repair_results, args.model_comparison_output)
    write_portable_csv(stability, args.stability_output)
    write_portable_csv(thresholds, args.threshold_output)
    write_report(args.report_output, anatomy, shift, feature_diag, stability, thresholds)

    primary = stability.loc[
        (stability["validation_scheme"].eq("three_band_holdout"))
        & (stability["target_definition"].isin(["at_risk_original", "actionable_drop"]))
    ]
    best_primary = primary.sort_values(["target_definition", "mean_pr_auc"], ascending=[True, False]).groupby("target_definition").head(1)
    print(f"Repair features built: {len(repair_features)}")
    print(f"Repair model rows: {len(repair_results)}")
    print("Best three-band repair rows for primary targets:")
    print(best_primary[["target_definition", "feature_family", "model", "mean_pr_auc", "min_pr_auc", "mean_recall", "mean_precision", "mean_f2"]].to_string(index=False))
    print(f"Wrote anatomy: {args.anatomy_output}")
    print(f"Wrote feature shift diagnostics: {args.shift_output}")
    print(f"Wrote repair features: {args.feature_output}")
    print(f"Wrote repair feature diagnostics: {args.feature_diagnostics_output}")
    print(f"Wrote repair model comparison: {args.model_comparison_output}")
    print(f"Wrote stability ranking: {args.stability_output}")
    print(f"Wrote threshold sensitivity: {args.threshold_output}")
    print(f"Wrote report: {args.report_output}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
