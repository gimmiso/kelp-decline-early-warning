"""Run spatial holdout validation diagnostics for kelp decline screening.

This script does not add new feature layers. It tests whether existing model
performance is sensitive to coastal spatial autocorrelation among neighboring
10 km Kelpwatch cells by holding out latitude bands.
"""

from __future__ import annotations

import argparse
import warnings
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

from run_recall_oriented_modeling_extensions import (
    BASELINE_P25,
    CANOPY,
    NEXT_CANOPY,
    TARGET_ACTIONABLE_DROP,
    add_actionable_labels,
)
from train_model_comparison import CANOPY_FEATURES, INPUT_DATASET, main_subset


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "tables"
DIAGNOSTICS_DIR = ROOT / "outputs" / "diagnostics"

CRW_FEATURE_PATH = PROCESSED_DIR / "crw5km_composite_features.csv"
HABITAT_FEATURE_PATH = PROCESSED_DIR / "bathymetry_habitat_features.csv"
TRAJECTORY_FEATURE_PATH = PROCESSED_DIR / "canopy_trajectory_features.csv"
WAVE_FEATURE_PATH = PROCESSED_DIR / "wave_exposure_features.csv"
INTEGRATED_MASTER_PATH = RESULTS_DIR / "integrated_model_comparison_master.csv"

FOLD_RESULTS_OUTPUT = RESULTS_DIR / "spatial_validation_fold_results.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "spatial_validation_summary.csv"
GAP_OUTPUT = RESULTS_DIR / "spatial_vs_temporal_validation_gap.csv"
REPORT_OUTPUT = DIAGNOSTICS_DIR / "spatial_validation_diagnostics_report.md"

MODEL_START_YEAR = 1989
TEST_END_YEAR = 2024
TARGET_NEW_DECLINE = "new_decline_event_next"
TARGET_AT_RISK = "decline_event_next_at_risk_gt005"
MIN_TEST_POSITIVES_PRIMARY = 5
MIN_TRAIN_POSITIVES = 10

CSV_WRITE_KWARGS = {
    "index": False,
    "lineterminator": "\n",
    "na_rep": "",
    "float_format": "%.6f",
}

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


@dataclass(frozen=True)
class TargetSpec:
    """Target and optional row filter."""

    name: str
    target: str
    filter_column: str | None = None
    priority: str = "secondary"


TARGETS = [
    TargetSpec("at_risk_original", TARGET_AT_RISK, "at_risk_gt005", "primary"),
    TargetSpec("actionable_drop", TARGET_ACTIONABLE_DROP, None, "primary"),
    TargetSpec("original_decline", "decline_event_next", None, "secondary"),
    TargetSpec("new_transition", TARGET_NEW_DECLINE, None, "secondary"),
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run spatial validation diagnostics.")
    parser.add_argument("--input", type=Path, default=INPUT_DATASET)
    parser.add_argument("--fold-results-output", type=Path, default=FOLD_RESULTS_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=SUMMARY_OUTPUT)
    parser.add_argument("--gap-output", type=Path, default=GAP_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    return parser.parse_args()


def clean_csv_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten embedded line breaks in object cells."""
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
    clean_csv_cells(df).to_csv(path, **CSV_WRITE_KWARGS)


def load_data(input_path: Path) -> pd.DataFrame:
    """Load and merge existing feature layers."""
    data = pd.read_csv(input_path).sort_values(["cell_id", "year"]).reset_index(drop=True)
    data = main_subset(data)
    data = add_actionable_labels(data)
    data[TARGET_NEW_DECLINE] = ((data[CANOPY] >= data[BASELINE_P25]) & (data[NEXT_CANOPY] < data[BASELINE_P25])).astype(int)
    data[TARGET_AT_RISK] = data["decline_event_next"].astype(int)
    data["at_risk_gt005"] = data[CANOPY] > 0.05

    for path, keys, drop_columns in [
        (CRW_FEATURE_PATH, ["cell_id", "year"], []),
        (HABITAT_FEATURE_PATH, ["cell_id"], ["feature_status"]),
        (TRAJECTORY_FEATURE_PATH, ["cell_id", "year"], ["trajectory_max_source_year_used", "trajectory_leakage_flag"]),
        (
            WAVE_FEATURE_PATH,
            ["cell_id", "year"],
            ["cdip_mop_id", "cdip_mop_lat", "cdip_mop_lon", "cdip_mop_water_depth_m", "wave_source_type", "winter_definition"],
        ),
    ]:
        if path.exists():
            layer = pd.read_csv(path).drop(columns=drop_columns, errors="ignore")
            data = data.merge(layer, on=keys, how="left", validate="many_to_one" if keys == ["cell_id"] else "one_to_one")
    return data


def available(columns: list[str], data: pd.DataFrame) -> list[str]:
    """Return feature columns present in data."""
    return [column for column in columns if column in data.columns]


def feature_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    """Return the limited spatial-validation feature families."""
    sets = {
        "canopy_only": available(CANOPY_FEATURES, data),
        "canopy_trajectory": available(TRAJECTORY_FEATURES, data),
        "trajectory_crw_habitat": available(TRAJECTORY_FEATURES + CRW_COMPOSITE_FEATURES + HABITAT_FEATURES, data),
    }
    wave = available(WAVE_FEATURES, data)
    if wave:
        sets["trajectory_crw_habitat_wave"] = available(
            TRAJECTORY_FEATURES + CRW_COMPOSITE_FEATURES + HABITAT_FEATURES + WAVE_FEATURES,
            data,
        )
    return {name: features for name, features in sets.items() if features}


def assign_latitude_bands(cells: pd.DataFrame, n_bands: int) -> pd.DataFrame:
    """Assign retained cells to ordered latitude bands."""
    labels = [f"band_{idx + 1}_{name}" for idx, name in enumerate(["south", "south_central", "central", "north_central", "north"][:n_bands])]
    if n_bands == 3:
        labels = ["band_1_south", "band_2_central", "band_3_north"]
    cells = cells.sort_values("center_lat").copy()
    cells[f"{n_bands}_band"] = pd.qcut(cells["center_lat"].rank(method="first"), q=n_bands, labels=labels)
    return cells[["cell_id", f"{n_bands}_band"]]


def add_spatial_bands(data: pd.DataFrame) -> pd.DataFrame:
    """Add three-band and five-band latitude fold labels."""
    cells = data[["cell_id", "center_lat", "center_lon"]].drop_duplicates()
    out = data.copy()
    out = out.merge(assign_latitude_bands(cells, 3), on="cell_id", how="left", validate="many_to_one")
    out = out.merge(assign_latitude_bands(cells, 5), on="cell_id", how="left", validate="many_to_one")
    return out


def has_two_classes(frame: pd.DataFrame, target: str) -> bool:
    """Return whether the frame has both target classes."""
    return set(frame[target].dropna().astype(int).unique()) == {0, 1}


def preprocess(features: list[str]) -> ColumnTransformer:
    """Create numeric preprocessing pipeline."""
    return ColumnTransformer(
        transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), features)]
    )


def estimators() -> dict[str, object]:
    """Return model estimators used in the spatial diagnostic."""
    models: dict[str, object] = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=3,
            n_jobs=-1,
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=120,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )
    return models


def select_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    """Select a training-only threshold maximizing F1."""
    precision, recall, thresholds = precision_recall_curve(y_true.astype(int), scores)
    if len(thresholds) == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1_values))])


def score_metrics(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, object]:
    """Compute test metrics from scores and a training-selected threshold."""
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) == 2 else np.nan,
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def evaluate_naive(train: pd.DataFrame, test: pd.DataFrame, target: TargetSpec) -> dict[str, object]:
    """Evaluate current-low-canopy naive persistence in a spatial fold."""
    train_scores = 1.0 - train[CANOPY].to_numpy(dtype=float)
    test_scores = 1.0 - test[CANOPY].to_numpy(dtype=float)
    threshold = select_threshold(train[target.target], train_scores)
    row = score_metrics(test[target.target].astype(int), test_scores, threshold)
    row.update({"feature_family": "naive_persistence", "model": "current_low_canopy_score", "decision_threshold": threshold})
    return row


def evaluate_fold(
    data: pd.DataFrame,
    validation_scheme: str,
    band_column: str,
    heldout_band: str,
    target: TargetSpec,
    features_by_family: dict[str, list[str]],
) -> list[dict[str, object]]:
    """Evaluate all feature families for one held-out spatial band and target."""
    working = data.copy()
    if target.filter_column:
        working = working.loc[working[target.filter_column]].copy()
    train = working.loc[working[band_column] != heldout_band].copy()
    test = working.loc[working[band_column] == heldout_band].copy()
    train_pos = int(train[target.target].sum()) if target.target in train else 0
    test_pos = int(test[target.target].sum()) if target.target in test else 0

    base = {
        "validation_scheme": validation_scheme,
        "heldout_band": heldout_band,
        "target_definition": target.name,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_positive_count": train_pos,
        "test_positive_count": test_pos,
    }
    if train.empty or test.empty or train_pos < MIN_TRAIN_POSITIVES or test_pos < MIN_TEST_POSITIVES_PRIMARY:
        return [
            {
                **base,
                "feature_family": family,
                "model": "not_run",
                "status": "underpowered_fold",
                "notes": f"Requires at least {MIN_TRAIN_POSITIVES} train positives and {MIN_TEST_POSITIVES_PRIMARY} test positives.",
            }
            for family in ["naive_persistence", *features_by_family.keys()]
        ]
    if not has_two_classes(train, target.target) or not has_two_classes(test, target.target):
        return [
            {
                **base,
                "feature_family": family,
                "model": "not_run",
                "status": "single_class_fold",
                "notes": "Train or held-out band lacks both target classes.",
            }
            for family in ["naive_persistence", *features_by_family.keys()]
        ]

    rows: list[dict[str, object]] = []
    naive = evaluate_naive(train, test, target)
    rows.append({**base, **naive, "status": "computed", "notes": "Training-band threshold selected from non-held-out cells."})

    for family, features in features_by_family.items():
        for model_name, estimator in estimators().items():
            pipeline = Pipeline([("preprocess", preprocess(features)), ("model", estimator)])
            try:
                pipeline.fit(train[features], train[target.target].astype(int))
                train_scores = pipeline.predict_proba(train[features])[:, 1]
                test_scores = pipeline.predict_proba(test[features])[:, 1]
                threshold = select_threshold(train[target.target], train_scores)
                metrics = score_metrics(test[target.target].astype(int), test_scores, threshold)
                status = "computed"
                notes = "Training-band threshold selected from non-held-out cells."
            except Exception as exc:  # pragma: no cover - model-specific failure
                metrics = {
                    "pr_auc": np.nan,
                    "roc_auc": np.nan,
                    "recall": np.nan,
                    "precision": np.nan,
                    "f1": np.nan,
                    "f2": np.nan,
                    "false_negatives": np.nan,
                    "false_positives": np.nan,
                    "decision_threshold": np.nan,
                }
                status = "failed"
                notes = str(exc)
            rows.append(
                {
                    **base,
                    **metrics,
                    "feature_family": family,
                    "model": model_name,
                    "decision_threshold": threshold if status == "computed" else np.nan,
                    "status": status,
                    "notes": notes,
                }
            )
    return rows


def run_spatial_validation(data: pd.DataFrame) -> pd.DataFrame:
    """Run three-band and five-band latitude holdout validation."""
    features_by_family = feature_sets(data)
    rows: list[dict[str, object]] = []
    schemes = [
        ("three_band_holdout", "3_band"),
        ("five_band_holdout", "5_band"),
    ]
    for scheme, band_column in schemes:
        for target in TARGETS:
            for band in sorted(data[band_column].dropna().astype(str).unique()):
                rows.extend(evaluate_fold(data, scheme, band_column, band, target, features_by_family))
    return pd.DataFrame(rows)


def summarize_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize fold-level results by target and feature family."""
    computed_all = fold_results.loc[fold_results["status"].eq("computed")].copy()
    best_per_fold = (
        computed_all.sort_values(
            ["validation_scheme", "heldout_band", "target_definition", "feature_family", "pr_auc"],
            ascending=[True, True, True, True, False],
        )
        .groupby(["validation_scheme", "heldout_band", "target_definition", "feature_family"])
        .head(1)
        .reset_index(drop=True)
    )
    rows: list[dict[str, object]] = []
    group_cols = ["validation_scheme", "target_definition", "feature_family"]
    for keys, group in fold_results.groupby(group_cols, dropna=False):
        computed = best_per_fold
        for col, key in zip(group_cols, keys, strict=False):
            computed = computed.loc[computed[col].eq(key)]
        underpowered = group.loc[group["status"].isin(["underpowered_fold", "single_class_fold"])]
        row = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "mean_pr_auc": float(computed["pr_auc"].mean()) if not computed.empty else np.nan,
                "median_pr_auc": float(computed["pr_auc"].median()) if not computed.empty else np.nan,
                "min_pr_auc": float(computed["pr_auc"].min()) if not computed.empty else np.nan,
                "max_pr_auc": float(computed["pr_auc"].max()) if not computed.empty else np.nan,
                "mean_recall": float(computed["recall"].mean()) if not computed.empty else np.nan,
                "mean_precision": float(computed["precision"].mean()) if not computed.empty else np.nan,
                "mean_f2": float(computed["f2"].mean()) if not computed.empty else np.nan,
                "total_false_negatives": int(computed["false_negatives"].sum()) if not computed.empty else np.nan,
                "total_positives_across_heldout_folds": int(computed["test_positive_count"].sum()) if not computed.empty else 0,
                "valid_folds": int(computed[["validation_scheme", "heldout_band"]].drop_duplicates().shape[0]),
                "underpowered_folds": int(underpowered[["validation_scheme", "heldout_band"]].drop_duplicates().shape[0]),
                "best_model_counted": "best PR-AUC model per held-out fold is used before summarizing across folds.",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def best_spatial_summary(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Collapse to best model per fold before target/family summaries."""
    computed = fold_results.loc[fold_results["status"].eq("computed")].copy()
    if computed.empty:
        return computed
    best_per_fold = (
        computed.sort_values(["validation_scheme", "heldout_band", "target_definition", "feature_family", "pr_auc"], ascending=[True, True, True, True, False])
        .groupby(["validation_scheme", "heldout_band", "target_definition", "feature_family"])
        .head(1)
    )
    rows: list[dict[str, object]] = []
    for keys, group in best_per_fold.groupby(["validation_scheme", "target_definition", "feature_family"], dropna=False):
        rows.append(
            {
                "validation_scheme": keys[0],
                "target_definition": keys[1],
                "feature_family": keys[2],
                "spatial_mean_pr_auc": float(group["pr_auc"].mean()),
                "spatial_mean_recall": float(group["recall"].mean()),
                "spatial_mean_precision": float(group["precision"].mean()),
                "spatial_mean_f2": float(group["f2"].mean()),
                "spatial_total_false_negatives": int(group["false_negatives"].sum()),
                "spatial_total_positives": int(group["test_positive_count"].sum()),
                "valid_folds": int(group["heldout_band"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def family_to_temporal_names(family: str) -> list[str]:
    """Map spatial diagnostic families to integrated temporal family names."""
    mapping = {
        "naive_persistence": ["naive_persistence"],
        "canopy_only": ["canopy_only"],
        "canopy_trajectory": ["canopy_trajectory", "canopy_plus_trajectory"],
        "trajectory_crw_habitat": ["canopy_plus_CRW_plus_habitat"],
        "trajectory_crw_habitat_wave": ["canopy_plus_CRW_plus_habitat_plus_wave"],
    }
    return mapping.get(family, [family])


def compare_to_temporal(best_spatial: pd.DataFrame) -> pd.DataFrame:
    """Compare spatial holdout summary with existing temporal/integrated results."""
    if not INTEGRATED_MASTER_PATH.exists() or best_spatial.empty:
        return pd.DataFrame()
    temporal = pd.read_csv(INTEGRATED_MASTER_PATH)
    temporal = temporal.loc[temporal["status"].eq("computed")].copy()
    rows: list[dict[str, object]] = []
    primary = best_spatial.loc[best_spatial["validation_scheme"].eq("three_band_holdout")].copy()
    for row in primary.itertuples(index=False):
        temporal_names = family_to_temporal_names(row.feature_family)
        subset = temporal.loc[
            temporal["normalized_target"].eq(row.target_definition)
            & temporal["normalized_feature_family"].isin(temporal_names)
        ].copy()
        if subset.empty:
            continue
        best_pr = subset.sort_values("pr_auc", ascending=False).iloc[0]
        best_recall = subset.sort_values("recall", ascending=False).iloc[0]
        pr_gap = float(row.spatial_mean_pr_auc - best_pr.pr_auc)
        recall_gap = float(row.spatial_mean_recall - best_recall.recall)
        if pr_gap >= -0.05:
            interpretation = "spatial_holdout_similar_to_temporal"
        elif pr_gap >= -0.15:
            interpretation = "spatial_holdout_moderately_lower"
        else:
            interpretation = "spatial_holdout_substantially_lower"
        rows.append(
            {
                "target_definition": row.target_definition,
                "feature_family": row.feature_family,
                "best_temporal_pr_auc": float(best_pr.pr_auc),
                "spatial_mean_pr_auc": float(row.spatial_mean_pr_auc),
                "spatial_minus_temporal_pr_auc": pr_gap,
                "best_temporal_recall": float(best_recall.recall),
                "spatial_mean_recall": float(row.spatial_mean_recall),
                "spatial_minus_temporal_recall": recall_gap,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def compact_table(frame: pd.DataFrame) -> str:
    """Render a compact markdown table."""
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


def event_feasibility_table(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize fold target event counts."""
    rows: list[dict[str, object]] = []
    for scheme, band_column in [("three_band_holdout", "3_band"), ("five_band_holdout", "5_band")]:
        for band in sorted(data[band_column].astype(str).unique()):
            heldout = data.loc[data[band_column].astype(str).eq(band)]
            for target in TARGETS:
                working = heldout.loc[heldout[target.filter_column]].copy() if target.filter_column else heldout
                rows.append(
                    {
                        "validation_scheme": scheme,
                        "heldout_band": band,
                        "target_definition": target.name,
                        "test_rows": int(len(working)),
                        "positive_events": int(working[target.target].sum()),
                        "event_prevalence": float(working[target.target].mean()) if len(working) else np.nan,
                        "feasibility": "usable" if int(working[target.target].sum()) >= MIN_TEST_POSITIVES_PRIMARY else "underpowered",
                    }
                )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    data: pd.DataFrame,
    event_counts: pd.DataFrame,
    fold_results: pd.DataFrame,
    summary: pd.DataFrame,
    gap: pd.DataFrame,
) -> None:
    """Write spatial validation diagnostic report."""
    cells = data[["cell_id", "center_lat", "center_lon"]].drop_duplicates()
    computed = fold_results.loc[fold_results["status"].eq("computed")]
    primary_summary = summary.loc[summary["validation_scheme"].eq("three_band_holdout")].copy()
    primary_best = (
        primary_summary.sort_values(["target_definition", "mean_pr_auc"], ascending=[True, False])
        .groupby("target_definition")
        .head(1)
    )
    primary_events = event_counts.loc[event_counts["validation_scheme"].eq("three_band_holdout")]
    five_events = event_counts.loc[event_counts["validation_scheme"].eq("five_band_holdout")]
    five_underpowered = int((five_events["feasibility"] == "underpowered").sum())
    five_total = int(len(five_events))
    five_status = "usable_with_caution" if five_underpowered == 0 else "underpowered_for_some_targets"

    lines = [
        "# Spatial Validation Diagnostics Report",
        "",
        "## Purpose",
        "",
        "This diagnostic checks whether kelp decline risk-screening performance is dependent on spatial autocorrelation among neighboring 10 km retained Kelpwatch cells.",
        "The existing workflow already uses temporal train/validation/test splits; this layer asks whether models transfer across latitude-defined coastal bands within the same study domain.",
        "",
        "## Retained Cell and Target Summary",
        "",
        f"- Retained cells: `{cells['cell_id'].nunique()}`",
        f"- Latitude range: `{cells['center_lat'].min():.3f}` to `{cells['center_lat'].max():.3f}`",
        f"- Longitude range: `{cells['center_lon'].min():.3f}` to `{cells['center_lon'].max():.3f}`",
        f"- Year range: `{int(data['year'].min())}` to `{int(data['year'].max())}`",
        "",
        "Target event counts across the full model period:",
        "",
        compact_table(
            pd.DataFrame(
                [
                    {"target_definition": target.name, "positive_events": int((data.loc[data[target.filter_column]] if target.filter_column else data)[target.target].sum())}
                    for target in TARGETS
                ]
            )
        ),
        "",
        "## Spatial Fold Design",
        "",
        "- `three_band_holdout`: retained cells are split into south, central, and north latitude bands; train on two bands and test on the held-out band.",
        "- `five_band_holdout`: retained cells are split into five latitude bands; leave one band out at a time.",
        "- Classification thresholds are selected using training bands only.",
        "- The primary diagnostic is `three_band_holdout`; five-band results are reported as underpowered when held-out positives are too sparse.",
        "",
        "## Event-Count Feasibility",
        "",
        f"- Five-band feasibility status: `{five_status}` (`{five_underpowered}` underpowered target-band combinations out of `{five_total}`).",
        "",
        compact_table(primary_events[["heldout_band", "target_definition", "test_rows", "positive_events", "event_prevalence", "feasibility"]]),
        "",
        "## Results",
        "",
        f"- Computed fold-model rows: `{len(computed)}`",
        "",
        "Best three-band summary row per target:",
        "",
        compact_table(
            primary_best[
                [
                    "target_definition",
                    "feature_family",
                    "mean_pr_auc",
                    "mean_recall",
                    "mean_precision",
                    "mean_f2",
                    "total_false_negatives",
                    "valid_folds",
                    "underpowered_folds",
                ]
            ]
        ),
        "",
        "Primary target summaries:",
        "",
        compact_table(
            primary_summary.loc[
                primary_summary["target_definition"].isin(["at_risk_original", "actionable_drop"]),
                [
                    "target_definition",
                    "feature_family",
                    "mean_pr_auc",
                    "mean_recall",
                    "mean_precision",
                    "mean_f2",
                    "total_false_negatives",
                    "valid_folds",
                    "underpowered_folds",
                ],
            ].sort_values(["target_definition", "mean_pr_auc"], ascending=[True, False])
        ),
        "",
        "## Comparison to Temporal Split",
        "",
        compact_table(gap.sort_values(["target_definition", "feature_family"]) if not gap.empty else gap),
        "",
        "## Interpretation",
        "",
        "If spatial holdout performance is stable, the workflow shows some internal spatial transferability within the retained California cells.",
        "If spatial performance drops or is unstable, the model is better interpreted as regional risk screening within the studied domain rather than robust spatially transferable early warning.",
        "In this repository, spatial validation should be treated as a robustness diagnostic and not as external region validation.",
        "",
        "## Limitations",
        "",
        "- Only 50 retained cells are available.",
        "- Spatial folds can have low positive event counts, especially for stricter transition/actionable labels.",
        "- Latitude bands are an approximate coastal spatial blocking strategy.",
        "- External region validation remains future work.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run spatial validation diagnostics."""
    args = parse_args()
    data = add_spatial_bands(load_data(args.input))
    event_counts = event_feasibility_table(data)
    fold_results = run_spatial_validation(data)
    summary = summarize_results(fold_results)
    best_spatial = best_spatial_summary(fold_results)
    gap = compare_to_temporal(best_spatial)

    write_portable_csv(fold_results, args.fold_results_output)
    write_portable_csv(summary, args.summary_output)
    write_portable_csv(gap, args.gap_output)
    write_report(args.report_output, data, event_counts, fold_results, summary, gap)

    primary_events = event_counts.loc[event_counts["validation_scheme"].eq("three_band_holdout")]
    print(f"Retained cells: {data['cell_id'].nunique()}")
    print(f"Spatial validation fold-model rows: {len(fold_results)}")
    print(f"Computed rows: {(fold_results['status'] == 'computed').sum()}")
    print("Three-band positive event counts:")
    print(primary_events.pivot(index="heldout_band", columns="target_definition", values="positive_events").to_string())
    five_underpowered = int((event_counts.loc[event_counts["validation_scheme"].eq("five_band_holdout"), "feasibility"] == "underpowered").sum())
    print(f"Five-band underpowered target-band combinations: {five_underpowered}")
    print(f"Wrote fold results: {args.fold_results_output}")
    print(f"Wrote summary: {args.summary_output}")
    print(f"Wrote temporal gap: {args.gap_output}")
    print(f"Wrote report: {args.report_output}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
