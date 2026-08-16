"""Independently reconstruct daily heat-stress metrics from raw predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


SEED = 20260816
REPLICATES = 2000
BUDGETS = [5, 10, 20, 30]
COMPARISONS = {
    "daily_stress_minus_prior_trajectory": ("daily_stress", "prior_trajectory"),
    "daily_stress_minus_monthly_crw": ("daily_stress", "monthly_crw"),
    "monthly_and_daily_crw_minus_monthly_crw": ("monthly_and_daily_crw", "monthly_crw"),
    "monthly_crw_minus_prior_trajectory": ("monthly_crw", "prior_trajectory"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def moving_block_draws(difference: pd.Series) -> np.ndarray:
    difference = difference.sort_index()
    years = difference.index.to_numpy()
    starts = years[:-1]
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(REPLICATES):
        sampled = []
        while len(sampled) < len(years):
            start = int(rng.choice(starts))
            sampled.extend([float(difference.loc[start]), float(difference.loc[start + 1])])
        draws.append(float(np.mean(sampled[: len(years)])))
    return np.asarray(draws)


def main() -> None:
    args = parse_args()
    pred = pd.read_csv(args.output_dir / "predictions.csv")
    saved_metrics = pd.read_csv(args.output_dir / "model_family_year_metrics.csv")
    saved_increments = pd.read_csv(args.output_dir / "increment_summary.csv")
    folds = pd.read_csv(args.output_dir / "fold_audit.csv")
    source_qc = pd.read_csv(args.output_dir / "source_quality_checks.csv")
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    add("source_quality_checks_all_pass", source_qc["passed"].astype(str).str.lower().eq("true").all())
    add("prediction_scores_finite", np.isfinite(pred["score"]).all())
    add("prediction_scores_in_unit_interval", pred["score"].between(0, 1).all())
    add("all_20_forecast_years", set(pred["outcome_year"].unique()) == set(range(2005, 2025)))
    add("three_model_families", set(pred["model_family"].unique()) == {"logistic", "random_forest", "xgboost"})
    add("four_feature_sets", set(pred["feature_set"].unique()) == {"prior_trajectory", "monthly_crw", "daily_stress", "monthly_and_daily_crw"})
    add("fold_training_precedes_test", (folds["train_end"] == folds["forecast_year"] - 1).all())
    add("fold_training_starts_1990", folds["train_start"].eq(1990).all())
    add("fold_test_two_classes", folds["events_test"].gt(0).all() and (folds["events_test"] < folds["n_test"]).all())

    key_counts = pred.groupby(["model_family", "feature_set"])[["cell_id", "outcome_year"]].apply(lambda x: len(x.drop_duplicates()))
    add("identical_prediction_row_count", key_counts.nunique() == 1, str(key_counts.to_dict()))
    base_keys = None
    identical_keys = True
    for _, group in pred.groupby(["model_family", "feature_set"]):
        keys = set(zip(group["cell_id"], group["outcome_year"]))
        if base_keys is None:
            base_keys = keys
        identical_keys &= keys == base_keys
    add("identical_prediction_keys", identical_keys)

    reconstructed = []
    for (family, feature_set, year), group in pred.groupby(["model_family", "feature_set", "outcome_year"], sort=True):
        prevalence = float(group["target"].mean())
        row = {
            "model_family": family,
            "feature_set": feature_set,
            "forecast_year": int(year),
            "n": len(group),
            "events": int(group["target"].sum()),
            "prevalence": prevalence,
            "average_precision": float(average_precision_score(group["target"], group["score"])),
        }
        row["ap_lift"] = row["average_precision"] - prevalence
        ranked = group.sort_values(["score", "cell_id"], ascending=[False, True])
        for budget in BUDGETS:
            k = max(1, int(math.ceil((budget / 100) * len(group))))
            row[f"recall_top_{budget}pct"] = float(ranked.head(k)["target"].sum() / group["target"].sum())
            row[f"selected_top_{budget}pct"] = k
        reconstructed.append(row)
    reconstructed = pd.DataFrame(reconstructed)
    merged = saved_metrics.merge(
        reconstructed,
        on=["model_family", "feature_set", "forecast_year"],
        suffixes=("_saved", "_recomputed"),
        validate="one_to_one",
    )
    numeric = ["prevalence", "average_precision", "ap_lift", *[f"recall_top_{b}pct" for b in BUDGETS]]
    for column in numeric:
        difference = (merged[f"{column}_saved"] - merged[f"{column}_recomputed"]).abs().max()
        add(f"recompute_{column}", difference <= 1e-12, f"max_abs_diff={difference}")

    increment_ok = True
    max_increment_diff = 0.0
    metric_columns = ["ap_lift", *[f"recall_top_{b}pct" for b in BUDGETS]]
    for family, group in reconstructed.groupby("model_family"):
        for comparison, (richer, base) in COMPARISONS.items():
            for metric in metric_columns:
                wide = group.pivot(index="forecast_year", columns="feature_set", values=metric)
                difference = wide[richer] - wide[base]
                draws = moving_block_draws(difference)
                expected = saved_increments.loc[
                    saved_increments["model_family"].eq(family)
                    & saved_increments["comparison"].eq(comparison)
                    & saved_increments["metric"].eq(metric)
                ].iloc[0]
                values = [float(difference.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
                saved = [float(expected.mean_difference), float(expected.ci_low), float(expected.ci_high)]
                local = max(abs(a - b) for a, b in zip(values, saved))
                max_increment_diff = max(max_increment_diff, local)
                increment_ok &= local <= 1e-12
    add("recompute_all_increment_estimates_and_intervals", increment_ok, f"max_abs_diff={max_increment_diff}")

    quality = pd.DataFrame(checks)
    quality.to_csv(args.output_dir / "independent_validation_checks.csv", index=False)
    result = {
        "checks": len(quality),
        "passed": int(quality["passed"].sum()),
        "failed": int((~quality["passed"]).sum()),
        "all_passed": bool(quality["passed"].all()),
    }
    (args.output_dir / "independent_validation_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise SystemExit("Independent validation failed")


if __name__ == "__main__":
    main()
