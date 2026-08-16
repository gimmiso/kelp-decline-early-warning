"""Run fixed model-family robustness for the March forecast-clock experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


SEED = 20260816
BOOTSTRAP_REPLICATES = 2000
TRAJECTORY = [
    "canopy_lag1",
    "canopy_lag2",
    "canopy_2yr_change",
    "canopy_3yr_change",
    "canopy_3yr_mean",
    "canopy_3yr_std",
    "canopy_3yr_slope",
    "canopy_drop_from_3yr_max",
]
FEATURE_SETS = {
    "prior_trajectory": ["relative_canopy", *TRAJECTORY],
    "march_current_oisst": ["relative_canopy", *TRAJECTORY, "winter_oisst_y"],
    "march_current_crw5km": [
        "relative_canopy",
        *TRAJECTORY,
        "q1_mean_sst_anomaly_crw5km",
        "q1_max_monthly_sst_anomaly_crw5km",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pipeline(family: str) -> Pipeline:
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


def moving_block_draws(difference: pd.Series) -> np.ndarray:
    difference = difference.sort_index()
    years = difference.index.to_numpy()
    starts = years[:-1]
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled: list[float] = []
        while len(sampled) < len(years):
            start = int(rng.choice(starts))
            sampled.extend([float(difference.loc[start]), float(difference.loc[start + 1])])
        draws.append(float(np.mean(sampled[: len(years)])))
    return np.asarray(draws)


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.analysis_rows)
    rows: list[dict[str, object]] = []
    for family in ["logistic", "random_forest", "xgboost"]:
        for year in range(2005, 2025):
            train = data.loc[data["outcome_year"].between(1990, year - 1)]
            test = data.loc[data["outcome_year"].eq(year)]
            prevalence = float(test["target"].mean())
            for feature_set, features in FEATURE_SETS.items():
                estimator = pipeline(family)
                estimator.fit(train[features], train["target"])
                score = estimator.predict_proba(test[features])[:, 1]
                ap = float(average_precision_score(test["target"], score))
                rows.append(
                    {
                        "model_family": family,
                        "feature_set": feature_set,
                        "forecast_year": year,
                        "n": len(test),
                        "events": int(test["target"].sum()),
                        "average_precision": ap,
                        "prevalence": prevalence,
                        "ap_lift": ap - prevalence,
                    }
                )
    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby(["model_family", "feature_set"])
        .agg(
            forecast_years=("forecast_year", "nunique"),
            mean_ap=("average_precision", "mean"),
            macro_within_year_ap_lift=("ap_lift", "mean"),
        )
        .reset_index()
    )
    increment_rows: list[dict[str, object]] = []
    for family, group in metrics.groupby("model_family"):
        wide = group.pivot(index="forecast_year", columns="feature_set", values="ap_lift")
        for feature_set, label in [
            ("march_current_crw5km", "crw_increment"),
            ("march_current_oisst", "oisst_increment"),
        ]:
            difference = wide[feature_set] - wide["prior_trajectory"]
            draws = moving_block_draws(difference)
            increment_rows.append(
                {
                    "model_family": family,
                    "comparison": label,
                    "mean_ap_lift_difference": float(difference.mean()),
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                    "positive_years": int(difference.gt(0).sum()),
                    "forecast_years": len(difference),
                }
            )
    increments = pd.DataFrame(increment_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "model_family_year_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "model_family_summary.csv", index=False)
    increments.to_csv(args.output_dir / "model_family_increment_summary.csv", index=False)
    metadata = {
        "classification": "exploratory_fixed_model_family_robustness",
        "analysis_rows": str(args.analysis_rows),
        "analysis_rows_sha256": sha256(args.analysis_rows),
        "seed": SEED,
        "bootstrap": {"type": "two_year_moving_block", "replicates": BOOTSTRAP_REPLICATES},
    }
    (args.output_dir / "model_family_robustness.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(increments.to_string(index=False))


if __name__ == "__main__":
    main()
