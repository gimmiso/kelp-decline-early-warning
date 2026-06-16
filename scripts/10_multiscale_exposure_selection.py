"""Compare V2 multi-scale environmental exposure models.

This script evaluates whether OISST exposure summaries at different spatial
supports provide useful transition-oriented screening signal. It does not
overwrite Version 1 models; it writes compact V2 result tables under
``results/tables``.
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
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
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
from train_model_comparison import INPUT_DATASET, main_subset


DEFAULT_MULTISCALE_FEATURES = Path("data/processed/multiscale_environmental_features.csv")
MODEL_COMPARISON_OUTPUT = Path("results/tables/multiscale_model_comparison.csv")
SELECTED_SCALE_OUTPUT = Path("results/tables/selected_scale_by_predictor.csv")
COLLINEARITY_OUTPUT = Path("results/tables/feature_collinearity_v2.csv")
REPORT_OUTPUT = Path("outputs/diagnostics/multiscale_exposure_selection_report.md")

TARGET_ORIGINAL = "decline_event_next"
TARGET_NEW_DECLINE = "new_decline_event_next"
TARGET_AT_RISK = "decline_event_next_at_risk_gt005"
SCALES = ["nearest", "10km", "25km", "30km", "50km", "75km"]
UPWELLING_FEATURES = ["cuti_anomaly", "beuti_anomaly"]
HIGH_CORRELATION_THRESHOLD = 0.80

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass(frozen=True)
class FeatureSet:
    """Feature set metadata."""

    comparison: str
    scale: str
    features: list[str]
    model_family: str
    estimator: object


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run V2 multi-scale exposure model comparison.")
    parser.add_argument("--input", type=Path, default=INPUT_DATASET)
    parser.add_argument("--multiscale-features", type=Path, default=DEFAULT_MULTISCALE_FEATURES)
    parser.add_argument("--model-comparison-output", type=Path, default=MODEL_COMPARISON_OUTPUT)
    parser.add_argument("--selected-scale-output", type=Path, default=SELECTED_SCALE_OUTPUT)
    parser.add_argument("--collinearity-output", type=Path, default=COLLINEARITY_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=REPORT_OUTPUT)
    parser.add_argument("--include-current-canopy", action="store_true")
    return parser.parse_args()


def load_data(input_path: Path, multiscale_path: Path, include_current_canopy: bool) -> pd.DataFrame:
    """Load V1 modeling rows and merge V2 multi-scale features."""
    if not multiscale_path.exists():
        raise FileNotFoundError(
            f"{multiscale_path} does not exist. Run scripts/09_build_multiscale_environmental_features.py first."
        )
    base = main_subset(pd.read_csv(input_path).sort_values(["cell_id", "year"]).reset_index(drop=True))
    multiscale = pd.read_csv(multiscale_path)
    data = base.merge(multiscale, on=["cell_id", "year"], how="left", validate="one_to_one")
    data = add_actionable_labels(data)
    data[TARGET_NEW_DECLINE] = (
        (data[CANOPY] >= data[BASELINE_P25]) & (data[NEXT_CANOPY] < data[BASELINE_P25])
    ).astype(int)
    data[TARGET_AT_RISK] = data[TARGET_ORIGINAL]
    data["at_risk_gt005"] = data[CANOPY] > 0.05
    data = add_multiscale_lags(data)
    if include_current_canopy:
        data["current_canopy_allowed"] = data[CANOPY]
    return data


def add_multiscale_lags(data: pd.DataFrame) -> pd.DataFrame:
    """Add lagged multi-scale thermal exposure summaries by cell."""
    output = data.sort_values(["cell_id", "year"]).copy()
    grouped = output.groupby("cell_id", group_keys=False)
    for column in list(output.columns):
        if column.startswith("oisst_") and (
            column.endswith("_annual_mean_sst_anomaly_mean") or column.endswith("_hot_days_p90_mean")
        ):
            output[f"lag1_{column}"] = grouped[column].shift(1)
    return output


def scale_feature_names(scale: str) -> list[str]:
    """Return domain-guided reduced OISST features for one spatial scale."""
    prefix = f"oisst_{scale}"
    return [
        f"{prefix}_annual_mean_sst_anomaly_mean",
        f"{prefix}_hot_days_p90_mean",
        f"lag1_{prefix}_hot_days_p90_mean",
    ]


def available(features: list[str], data: pd.DataFrame) -> list[str]:
    """Keep features available in the current dataset."""
    return [feature for feature in features if feature in data.columns]


def build_feature_sets(data: pd.DataFrame, include_current_canopy: bool) -> list[FeatureSet]:
    """Create M0-M3 feature sets with compact, domain-guided predictors."""
    sets: list[FeatureSet] = []
    current = ["current_canopy_allowed"] if include_current_canopy else []
    upwelling = available(UPWELLING_FEATURES, data)
    for scale in SCALES:
        comparison = "M0_nearest_grid_baseline" if scale == "nearest" else f"M2_single_scale_{scale}"
        if scale == "30km":
            comparison = "M1_fixed_30km_buffer"
        features = available(scale_feature_names(scale), data) + upwelling + current
        if features:
            sets.extend(model_variants(comparison, scale, features))

    multi_features = []
    for scale in SCALES:
        multi_features.extend(available(scale_feature_names(scale), data))
    multi_features = list(dict.fromkeys(multi_features + upwelling + current))
    if multi_features:
        sets.append(
            FeatureSet(
                comparison="M3_multiscale_l1_regularized",
                scale="multi",
                features=multi_features,
                model_family="Logistic Regression L1",
                estimator=LogisticRegression(
                    class_weight="balanced",
                    penalty="l1",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=42,
                ),
            )
        )
        sets.append(
            FeatureSet(
                comparison="M3_multiscale_random_forest",
                scale="multi",
                features=multi_features,
                model_family="Random Forest",
                estimator=RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight="balanced",
                    min_samples_leaf=3,
                    n_jobs=-1,
                ),
            )
        )
    return sets


def model_variants(comparison: str, scale: str, features: list[str]) -> list[FeatureSet]:
    """Return model variants for one scale."""
    return [
        FeatureSet(
            comparison=comparison,
            scale=scale,
            features=features,
            model_family="Logistic Regression L2",
            estimator=LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        ),
        FeatureSet(
            comparison=comparison,
            scale=scale,
            features=features,
            model_family="Random Forest",
            estimator=RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=3,
                n_jobs=-1,
            ),
        ),
    ]


def preprocessor(features: list[str]) -> ColumnTransformer:
    """Create a numeric preprocessing pipeline."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                features,
            )
        ],
        remainder="drop",
    )


def targets() -> list[tuple[str, str, str | None]]:
    """Return target definitions and optional subset column."""
    return [
        ("A_original_decline_state", TARGET_ORIGINAL, None),
        ("B_at_risk_original_decline_gt005", TARGET_AT_RISK, "at_risk_gt005"),
        ("C_new_decline_transition", TARGET_NEW_DECLINE, None),
        ("D_actionable_decline_drop", TARGET_ACTIONABLE_DROP, None),
    ]


def split_rows(data: pd.DataFrame, validation_design: str, held_out_region: str | None = None) -> dict[str, pd.DataFrame]:
    """Create temporal or region-holdout splits."""
    if validation_design == "temporal_holdout":
        return {
            "train": data.loc[data["year"].between(1989, 2016)].copy(),
            "validation": data.loc[data["year"].between(2017, 2020)].copy(),
            "test": data.loc[data["year"].between(2021, 2024)].copy(),
        }
    if validation_design == "region_holdout":
        if held_out_region is None:
            raise ValueError("held_out_region is required for region_holdout")
        return {
            "train": data.loc[(data["region_group"] != held_out_region) & data["year"].between(1989, 2020)].copy(),
            "validation": data.loc[(data["region_group"] != held_out_region) & data["year"].between(2017, 2020)].copy(),
            "test": data.loc[(data["region_group"] == held_out_region) & data["year"].between(2021, 2024)].copy(),
        }
    raise ValueError(f"Unknown validation design: {validation_design}")


def has_two_classes(frame: pd.DataFrame, target: str) -> bool:
    """Return whether a split contains both target classes."""
    return set(frame[target].dropna().astype(int).unique()) == {0, 1}


def predict_scores(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    """Return positive-class scores from a fitted model."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(x)


def evaluate_split(
    split: dict[str, pd.DataFrame],
    target: str,
    feature_set: FeatureSet,
    target_name: str,
    validation_design: str,
    held_out_region: str,
) -> dict[str, object] | None:
    """Fit one model and evaluate it on the held-out test split."""
    train = split["train"].dropna(subset=[target]).copy()
    test = split["test"].dropna(subset=[target]).copy()
    if not has_two_classes(train, target) or not has_two_classes(test, target):
        return None
    model = Pipeline(
        [
            ("preprocessor", preprocessor(feature_set.features)),
            ("model", feature_set.estimator),
        ]
    )
    x_train = train[feature_set.features]
    y_train = train[target].astype(int)
    x_test = test[feature_set.features]
    y_test = test[target].astype(int)
    model.fit(x_train, y_train)
    scores = predict_scores(model, x_test)
    predictions = (scores >= 0.5).astype(int)
    return {
        "target_definition": target_name,
        "target": target,
        "comparison": feature_set.comparison,
        "scale": feature_set.scale,
        "model_family": feature_set.model_family,
        "validation_design": validation_design,
        "held_out_region": held_out_region,
        "n_train": len(train),
        "n_test": len(test),
        "test_positive_events": int(y_test.sum()),
        "test_event_prevalence": float(y_test.mean()),
        "pr_auc": safe_metric(average_precision_score, y_test, scores),
        "roc_auc": safe_metric(roc_auc_score, y_test, scores),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "brier_score": brier_score_loss(y_test, np.clip(scores, 0, 1)),
        "feature_count": len(feature_set.features),
        "features": ";".join(feature_set.features),
    }


def safe_metric(metric, y_true: pd.Series, scores: np.ndarray) -> float:
    """Return ranking metric or NaN when undefined."""
    try:
        return float(metric(y_true, scores))
    except ValueError:
        return np.nan


def run_comparison(data: pd.DataFrame, include_current_canopy: bool) -> pd.DataFrame:
    """Run temporal and region-holdout model comparisons."""
    feature_sets = build_feature_sets(data, include_current_canopy)
    rows = []
    designs: list[tuple[str, str | None]] = [("temporal_holdout", None)]
    for region in sorted(data["region_group"].dropna().unique()):
        designs.append(("region_holdout", region))
    for target_name, target, subset in targets():
        target_data = data.loc[data[subset]].copy() if subset else data.copy()
        for validation_design, held_out_region in designs:
            split = split_rows(target_data, validation_design, held_out_region)
            for feature_set in feature_sets:
                result = evaluate_split(
                    split,
                    target,
                    feature_set,
                    target_name,
                    validation_design,
                    held_out_region or "none",
                )
                if result is not None:
                    rows.append(result)
    return pd.DataFrame(rows)


def selected_scale_table(results: pd.DataFrame) -> pd.DataFrame:
    """Select candidate scales using PR-AUC, recall, variance, and interpretability."""
    single = results.loc[
        results["validation_design"].eq("temporal_holdout")
        & results["comparison"].str.startswith(("M0_", "M1_", "M2_"))
    ].copy()
    if single.empty:
        return pd.DataFrame()
    grouped = (
        single.groupby(["target_definition", "scale"], as_index=False)
        .agg(
            mean_pr_auc=("pr_auc", "mean"),
            mean_recall=("recall", "mean"),
            pr_auc_std=("pr_auc", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_brier_score=("brier_score", "mean"),
        )
        .fillna({"pr_auc_std": 0.0})
    )
    grouped["decision_score"] = (
        grouped["mean_pr_auc"]
        + 0.25 * grouped["mean_recall"]
        + 0.10 * grouped["mean_balanced_accuracy"]
        - 0.10 * grouped["pr_auc_std"]
        - 0.05 * grouped["mean_brier_score"]
    )
    selected = grouped.sort_values(
        ["target_definition", "decision_score", "mean_pr_auc", "mean_recall"],
        ascending=[True, False, False, False],
    ).groupby("target_definition", as_index=False).head(1)
    selected["selection_rule"] = "combined_pr_auc_recall_stability_calibration"
    selected["interpretation"] = "Selected as a candidate exposure support, not a universal optimal resolution."
    return selected.reset_index(drop=True)


def collinearity_table(data: pd.DataFrame) -> pd.DataFrame:
    """Create high-correlation table among V2 reduced environmental features."""
    features = []
    for scale in SCALES:
        features.extend(scale_feature_names(scale))
    features.extend(UPWELLING_FEATURES)
    features = available(list(dict.fromkeys(features)), data)
    corr = data[features].corr()
    rows = []
    for i, left in enumerate(features):
        for right in features[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and abs(value) >= HIGH_CORRELATION_THRESHOLD:
                rows.append({"feature_1": left, "feature_2": right, "correlation": value, "abs_correlation": abs(value)})
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def write_report(output: Path, results: pd.DataFrame, selected: pd.DataFrame, collinearity: pd.DataFrame) -> None:
    """Write a compact V2 model-selection report."""
    best_transition = results.loc[
        results["target_definition"].eq("C_new_decline_transition")
        & results["validation_design"].eq("temporal_holdout")
    ].sort_values("pr_auc", ascending=False).head(1)
    best_actionable = results.loc[
        results["target_definition"].eq("D_actionable_decline_drop")
        & results["validation_design"].eq("temporal_holdout")
    ].sort_values("pr_auc", ascending=False).head(1)
    lines = [
        "# V2 Multi-Scale Exposure Selection Report",
        "",
        "## Purpose",
        "",
        "This report evaluates OISST exposure variables at multiple spatial supports using transition-oriented kelp decline targets. It treats spatial resolution as a modeling choice rather than a fixed preprocessing assumption.",
        "",
        "## Outputs",
        "",
        "- `results/tables/multiscale_model_comparison.csv`",
        "- `results/tables/selected_scale_by_predictor.csv`",
        "- `results/tables/feature_collinearity_v2.csv`",
        "",
        "## Main Findings",
        "",
        f"- Model-result rows: `{len(results)}`.",
        f"- High-correlation V2 feature pairs: `{len(collinearity)}`.",
    ]
    if not best_transition.empty:
        row = best_transition.iloc[0]
        lines.append(
            f"- Best temporal new-decline transition PR-AUC: `{row.pr_auc:.3f}` from `{row.comparison} / {row.model_family}` at scale `{row.scale}`."
        )
    if not best_actionable.empty:
        row = best_actionable.iloc[0]
        lines.append(
            f"- Best temporal actionable-drop PR-AUC: `{row.pr_auc:.3f}` from `{row.comparison} / {row.model_family}` at scale `{row.scale}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The transition and actionable-drop labels are harder than the original decline-state label because they reduce the influence of already-low canopy persistence. Lower performance under these labels is scientifically meaningful and should not be framed as failure.",
            "",
            "Scale selection is reported as multi-scale exposure selection, not as discovery of one universal optimal resolution. Thermal stress, upwelling proxies, and local biological processes may operate at different spatial supports.",
            "",
        ]
    )
    if not selected.empty:
        lines.extend(["## Selected Candidate Scales", "", selected.to_string(index=False), ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run V2 multi-scale model comparison."""
    args = parse_args()
    data = load_data(args.input, args.multiscale_features, args.include_current_canopy)
    results = run_comparison(data, args.include_current_canopy)
    selected = selected_scale_table(results)
    collinearity = collinearity_table(data)

    args.model_comparison_output.parent.mkdir(parents=True, exist_ok=True)
    args.selected_scale_output.parent.mkdir(parents=True, exist_ok=True)
    args.collinearity_output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.model_comparison_output, index=False)
    selected.to_csv(args.selected_scale_output, index=False)
    collinearity.to_csv(args.collinearity_output, index=False)
    write_report(args.report_output, results, selected, collinearity)

    print(f"Wrote model comparison: {args.model_comparison_output}")
    print(f"Wrote selected scales: {args.selected_scale_output}")
    print(f"Wrote V2 collinearity table: {args.collinearity_output}")
    print(f"Wrote report: {args.report_output}")


if __name__ == "__main__":
    main()
