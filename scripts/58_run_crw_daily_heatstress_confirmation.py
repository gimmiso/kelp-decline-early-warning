"""Evaluate locked daily Q1 CRW heat-stress features on matched Q2-Q3 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


SEED = 20260816
BOOTSTRAP_REPLICATES = 2000
FORECAST_YEARS = list(range(2005, 2025))
TRAJECTORY = [
    "relative_canopy",
    "canopy_lag1",
    "canopy_lag2",
    "canopy_2yr_change",
    "canopy_3yr_change",
    "canopy_3yr_mean",
    "canopy_3yr_std",
    "canopy_3yr_slope",
    "canopy_drop_from_3yr_max",
]
MONTHLY = [
    "q1_mean_sst_anomaly_crw5km",
    "q1_max_monthly_sst_anomaly_crw5km",
]
DAILY = [
    "q1_max_7day_mean_anomaly_crw5km",
    "q1_positive_anomaly_degree_days_crw5km",
    "q1_days_above_local_p90_crw5km",
    "q1_max_consecutive_hot_days_crw5km",
]
FEATURE_SETS = {
    "prior_trajectory": TRAJECTORY,
    "monthly_crw": [*TRAJECTORY, *MONTHLY],
    "daily_stress": [*TRAJECTORY, *DAILY],
    "monthly_and_daily_crw": [*TRAJECTORY, *MONTHLY, *DAILY],
}
COMPARISONS = {
    "daily_stress_minus_prior_trajectory": ("daily_stress", "prior_trajectory"),
    "daily_stress_minus_monthly_crw": ("daily_stress", "monthly_crw"),
    "monthly_and_daily_crw_minus_monthly_crw": ("monthly_and_daily_crw", "monthly_crw"),
    "monthly_crw_minus_prior_trajectory": ("monthly_crw", "prior_trajectory"),
}
BUDGETS = [0.05, 0.10, 0.20, 0.30]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-analysis-rows", type=Path, required=True)
    parser.add_argument("--daily-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> tuple[str, bool]:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    return sha, dirty


def estimator(family: str) -> Pipeline:
    if family == "logistic":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=5000, random_state=SEED)),
            ]
        )
    if family == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=5,
                        min_samples_leaf=10,
                        max_features="sqrt",
                        random_state=SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    if family == "xgboost":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.03,
                        min_child_weight=5,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        reg_alpha=0.0,
                        eval_metric="logloss",
                        random_state=SEED,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    raise KeyError(family)


def prepare_rows(base_path: Path, daily_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(base_path)
    daily = pd.read_csv(daily_path).rename(columns={"year": "outcome_year"})
    if base.duplicated(["cell_id", "outcome_year"]).any():
        raise ValueError("Base analysis rows are not unique")
    if daily.duplicated(["cell_id", "outcome_year"]).any():
        raise ValueError("Daily features are not unique")
    keep_daily = [
        "cell_id",
        "outcome_year",
        "source_point_id",
        "q1_daily_coverage",
        "q1_daily_coverage_pass",
        "local_q1_p90_c",
        *DAILY,
    ]
    merged = base.merge(daily[keep_daily], on=["cell_id", "outcome_year"], how="left", validate="one_to_one")
    candidate = merged.loc[merged["outcome_year"].between(1990, 2024)].copy()
    all_features = sorted({item for values in FEATURE_SETS.values() for item in values})
    common = candidate.dropna(subset=all_features).copy()
    common["target"] = common["target"].astype(int)
    exclusion = pd.DataFrame(
        [
            {"stage": "base_rows_1990_2024", "rows": len(candidate), "cells": candidate["cell_id"].nunique()},
            {"stage": "complete_matched_information_blocks", "rows": len(common), "cells": common["cell_id"].nunique()},
            {"stage": "forecast_rows_2005_2024", "rows": int(common["outcome_year"].isin(FORECAST_YEARS).sum()), "cells": common.loc[common["outcome_year"].isin(FORECAST_YEARS), "cell_id"].nunique()},
        ]
    )
    return common, exclusion


def score_models(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    for year in FORECAST_YEARS:
        train = data.loc[data["outcome_year"].between(1990, year - 1)]
        test = data.loc[data["outcome_year"].eq(year)]
        if train["target"].nunique() != 2 or test["target"].nunique() != 2:
            raise ValueError(f"Fold {year} does not contain two classes")
        folds.append(
            {
                "forecast_year": year,
                "train_start": int(train["outcome_year"].min()),
                "train_end": int(train["outcome_year"].max()),
                "n_train": len(train),
                "events_train": int(train["target"].sum()),
                "n_test": len(test),
                "events_test": int(test["target"].sum()),
                "test_cells": test["cell_id"].nunique(),
            }
        )
        for family in ["logistic", "random_forest", "xgboost"]:
            for feature_set, features in FEATURE_SETS.items():
                model = estimator(family)
                model.fit(train[features], train["target"])
                out = test[["cell_id", "region_group", "outcome_year", "target"]].copy()
                out["score"] = model.predict_proba(test[features])[:, 1]
                out["model_family"] = family
                out["feature_set"] = feature_set
                prediction_frames.append(out)
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(folds)


def calculate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (family, feature_set, year), group in predictions.groupby(
        ["model_family", "feature_set", "outcome_year"], sort=True
    ):
        prevalence = float(group["target"].mean())
        ap = float(average_precision_score(group["target"], group["score"]))
        ranked = group.sort_values(["score", "cell_id"], ascending=[False, True])
        row: dict[str, object] = {
            "model_family": family,
            "feature_set": feature_set,
            "forecast_year": int(year),
            "n": len(group),
            "events": int(group["target"].sum()),
            "prevalence": prevalence,
            "average_precision": ap,
            "ap_lift": ap - prevalence,
        }
        for budget in BUDGETS:
            k = max(1, int(math.ceil(budget * len(group))))
            row[f"recall_top_{int(budget * 100)}pct"] = float(ranked.head(k)["target"].sum() / group["target"].sum())
            row[f"selected_top_{int(budget * 100)}pct"] = k
        rows.append(row)
    return pd.DataFrame(rows)


def moving_block_draws(difference: pd.Series) -> np.ndarray:
    difference = difference.sort_index()
    years = difference.index.to_numpy()
    starts = years[:-1]
    rng = np.random.default_rng(SEED)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled: list[float] = []
        while len(sampled) < len(years):
            start = int(rng.choice(starts))
            sampled.extend([float(difference.loc[start]), float(difference.loc[start + 1])])
        draws.append(float(np.mean(sampled[: len(years)])))
    return np.asarray(draws)


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_columns = ["ap_lift", *[f"recall_top_{int(b * 100)}pct" for b in BUDGETS]]
    summary = metrics.groupby(["model_family", "feature_set"], as_index=False)[metric_columns].mean()
    summary = summary.rename(columns={column: f"mean_{column}" for column in metric_columns})
    increments: list[dict[str, object]] = []
    draws_rows: list[dict[str, object]] = []
    for family, group in metrics.groupby("model_family"):
        for comparison, (richer, base) in COMPARISONS.items():
            for metric in metric_columns:
                wide = group.pivot(index="forecast_year", columns="feature_set", values=metric)
                difference = wide[richer] - wide[base]
                draws = moving_block_draws(difference)
                increments.append(
                    {
                        "model_family": family,
                        "comparison": comparison,
                        "metric": metric,
                        "mean_difference": float(difference.mean()),
                        "ci_low": float(np.quantile(draws, 0.025)),
                        "ci_high": float(np.quantile(draws, 0.975)),
                        "positive_years": int(difference.gt(0).sum()),
                        "negative_years": int(difference.lt(0).sum()),
                        "forecast_years": len(difference),
                    }
                )
                draws_rows.extend(
                    {
                        "model_family": family,
                        "comparison": comparison,
                        "metric": metric,
                        "replicate": index,
                        "mean_difference": float(value),
                    }
                    for index, value in enumerate(draws)
                )
    return summary, pd.DataFrame(increments), pd.DataFrame(draws_rows)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, exclusion = prepare_rows(args.base_analysis_rows, args.daily_features)
    predictions, folds = score_models(data)
    metrics = calculate_metrics(predictions)
    summary, increments, draws = summarize(metrics)

    analysis_path = args.output_dir / "analysis_rows_q23_daily_heatstress.csv"
    prediction_path = args.output_dir / "predictions.csv"
    data.to_csv(analysis_path, index=False)
    predictions.to_csv(prediction_path, index=False)
    folds.to_csv(args.output_dir / "fold_audit.csv", index=False)
    metrics.to_csv(args.output_dir / "model_family_year_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "model_family_summary.csv", index=False)
    increments.to_csv(args.output_dir / "increment_summary.csv", index=False)
    draws.to_csv(args.output_dir / "bootstrap_differences.csv", index=False)
    exclusion.to_csv(args.output_dir / "matched_row_flow.csv", index=False)

    profile_columns = [*MONTHLY, *DAILY]
    data.loc[data["outcome_year"].isin(FORECAST_YEARS), profile_columns].describe().T.to_csv(
        args.output_dir / "feature_distribution.csv"
    )
    data.loc[data["outcome_year"].isin(FORECAST_YEARS), profile_columns].corr(method="spearman").to_csv(
        args.output_dir / "feature_spearman_correlations.csv"
    )
    region_profile = (
        data.loc[data["outcome_year"].isin(FORECAST_YEARS)]
        .groupby("region_group", as_index=False)
        .agg(rows=("cell_id", "size"), cells=("cell_id", "nunique"), events=("target", "sum"), mean_daily_coverage=("q1_daily_coverage", "mean"))
    )
    region_profile.to_csv(args.output_dir / "region_data_profile.csv", index=False)

    primary = increments.loc[
        increments["model_family"].eq("logistic")
        & increments["comparison"].eq("daily_stress_minus_prior_trajectory")
        & increments["metric"].eq("ap_lift")
    ].iloc[0]
    beyond_monthly = increments.loc[
        increments["model_family"].eq("logistic")
        & increments["comparison"].eq("monthly_and_daily_crw_minus_monthly_crw")
        & increments["metric"].eq("ap_lift")
    ].iloc[0]
    decision = {
        "primary_claim_gate": "supported_positive_increment" if primary.ci_low > 0 else "not_supported",
        "daily_minus_trajectory_mean_ap_lift_difference": float(primary.mean_difference),
        "daily_minus_trajectory_ci": [float(primary.ci_low), float(primary.ci_high)],
        "daily_beyond_monthly_claim_gate": "supported_positive_increment" if beyond_monthly.ci_low > 0 else "not_supported",
        "daily_beyond_monthly_mean_ap_lift_difference": float(beyond_monthly.mean_difference),
        "daily_beyond_monthly_ci": [float(beyond_monthly.ci_low), float(beyond_monthly.ci_high)],
        "interpretation_rule": "No mechanism claim; incremental predictive value only.",
    }
    (args.output_dir / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    git_sha, git_dirty = git_state()
    manifest = {
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "classification": config["classification"],
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "base_analysis_rows": str(args.base_analysis_rows),
        "base_analysis_rows_sha256": sha256(args.base_analysis_rows),
        "daily_features": str(args.daily_features),
        "daily_features_sha256": sha256(args.daily_features),
        "analysis_rows_sha256": sha256(analysis_path),
        "predictions_sha256": sha256(prediction_path),
        "analysis_rows": len(data),
        "forecast_rows": int(data["outcome_year"].isin(FORECAST_YEARS).sum()),
        "forecast_events": int(data.loc[data["outcome_year"].isin(FORECAST_YEARS), "target"].sum()),
        "forecast_years": FORECAST_YEARS,
        "feature_sets": FEATURE_SETS,
        "comparisons": COMPARISONS,
        "seed": SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "git_sha_at_run": git_sha,
        "git_dirty_at_run": git_dirty,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "decision": decision,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
