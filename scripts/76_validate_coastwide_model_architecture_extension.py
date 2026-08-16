"""Independently validate coastwide model-architecture extension outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add(rows: list[dict[str, object]], check: str, value: object, passed: bool) -> None:
    rows.append({"check": check, "value": value, "passed": bool(passed)})


def main() -> None:
    args = parse_args(); config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(args.run_dir / "predictions.csv")
    folds = pd.read_csv(args.run_dir / "fold_audit.csv")
    tuning = pd.read_csv(args.run_dir / "inner_tuning_audit.csv")
    annual = pd.read_csv(args.run_dir / "annual_metrics.csv")
    summary = pd.read_csv(args.run_dir / "model_summary.csv")
    increments = pd.read_csv(args.run_dir / "increment_summary.csv")
    budgets = pd.read_csv(args.run_dir / "budget_summary.csv")
    holdout = pd.read_csv(args.run_dir / "spatiotemporal_region_holdout_predictions.csv")
    holdout_audit = pd.read_csv(args.run_dir / "spatiotemporal_region_holdout_audit.csv")
    rows = []
    add(rows, "manifest_complete", manifest.get("status"), manifest.get("status") == "complete")
    hash_failures = [name for name, expected in manifest["outputs"].items() if not (args.run_dir / name).exists() or sha256(args.run_dir / name) != expected]
    add(rows, "manifest_hashes", len(hash_failures), not hash_failures)
    key = ["domain", "cell_id", "year", "model_family", "feature_set"]
    add(rows, "prediction_keys_unique", int(predictions.duplicated(key).sum()), not predictions.duplicated(key).any())
    add(rows, "scores_finite_bounded", int((~predictions["score"].between(0, 1)).sum()), predictions["score"].between(0, 1).all() and np.isfinite(predictions["score"]).all())
    sample_key = ["domain", "cell_id", "year"]
    model_counts = predictions.groupby(sample_key)["model_family"].nunique()
    add(rows, "three_models_same_samples", int(model_counts.ne(3).sum()), model_counts.eq(3).all())
    add(rows, "forward_only", int(folds["train_end"].ge(folds["year"]).sum()), folds["train_end"].lt(folds["year"]).all())

    tuning_failures = 0
    for keys, group in tuning.groupby(["domain", "year", "model_family", "feature_set"], sort=True):
        best = group.sort_values(["inner_macro_ap_lift", "C"], ascending=[False, True]).iloc[0]
        fold = folds.loc[folds["domain"].eq(keys[0]) & folds["year"].eq(keys[1]) & folds["model_family"].eq(keys[2]) & folds["feature_set"].eq(keys[3])].iloc[0]
        chosen = json.loads(fold["chosen"])
        if not np.isclose(chosen["C"], best["C"]) or ("l1_ratio" in chosen and not np.isclose(chosen["l1_ratio"], best["l1_ratio"])):
            tuning_failures += 1
    add(rows, "inner_tuning_argmax_recomputed", tuning_failures, tuning_failures == 0)

    recomputed = []
    for keys, group in predictions.groupby(["domain", "model_family", "feature_set", "year"], sort=True):
        ap = average_precision_score(group["event"], group["score"])
        recomputed.append({"domain": keys[0], "model_family": keys[1], "feature_set": keys[2], "year": int(keys[3]),
                           "ap_lift_check": float(ap - group["event"].mean())})
    annual_check = annual.merge(pd.DataFrame(recomputed), on=["domain", "model_family", "feature_set", "year"], validate="one_to_one")
    annual_error = float((annual_check["ap_lift"] - annual_check["ap_lift_check"]).abs().max())
    add(rows, "annual_ap_recomputed", annual_error, annual_error < 1e-12)
    summary_errors = []
    brier_errors = []
    for row in summary.itertuples(index=False):
        part = annual.loc[annual["domain"].eq(row.domain) & annual["model_family"].eq(row.model_family) & annual["feature_set"].eq(row.feature_set)]
        summary_errors.append(abs(part["ap_lift"].mean() - row.macro_within_year_ap_lift))
        pred = predictions.loc[predictions["domain"].eq(row.domain) & predictions["model_family"].eq(row.model_family) & predictions["feature_set"].eq(row.feature_set)]
        brier_errors.append(abs(brier_score_loss(pred["event"], pred["score"]) - row.brier))
    add(rows, "summary_metrics_recomputed", max(summary_errors + brier_errors), max(summary_errors + brier_errors) < 1e-12)

    increment_errors = []
    for row in increments.itertuples(index=False):
        part = annual.loc[annual["domain"].eq(row.domain) & annual["model_family"].eq(row.model_family)]
        pivot = part.pivot(index="year", columns="feature_set", values="ap_lift")
        increment_errors.append(abs((pivot[row.left_feature_set] - pivot[row.right_feature_set]).mean() - row.estimate))
    add(rows, "increments_recomputed", max(increment_errors), max(increment_errors) < 1e-12)

    budget_failures = 0
    for row in budgets.itertuples(index=False):
        group = predictions.loc[predictions["domain"].eq(row.domain) & predictions["model_family"].eq(row.model_family) & predictions["feature_set"].eq(row.feature_set)]
        selected = []
        for _, year_data in group.groupby("year", sort=True):
            count = max(1, int(np.ceil(len(year_data) * row.budget_fraction)))
            selected.append(year_data.sort_values(["score", "cell_id"], ascending=[False, True]).head(count))
        selected = pd.concat(selected)
        recall = selected["event"].sum() / group["event"].sum()
        if not np.isclose(recall, row.micro_recall): budget_failures += 1
    add(rows, "budget_recall_recomputed", budget_failures, budget_failures == 0)
    add(rows, "holdout_test_regions_match_label", int((holdout["region_group"] != holdout["heldout_region"]).sum()), (holdout["region_group"] == holdout["heldout_region"]).all())
    add(rows, "holdout_period_locked", f"{holdout.year.min()}-{holdout.year.max()}", holdout["year"].between(2015, 2024).all())
    holdout_key = ["heldout_region", "cell_id", "year", "model_family", "feature_set"]
    add(rows, "holdout_keys_unique", int(holdout.duplicated(holdout_key).sum()), not holdout.duplicated(holdout_key).any())
    holdout_sample_counts = holdout.groupby(["heldout_region", "cell_id", "year"])["model_family"].nunique()
    add(rows, "holdout_three_models_same_samples", int(holdout_sample_counts.ne(3).sum()), holdout_sample_counts.eq(3).all())
    region_overlap = sum(row.heldout_region in json.loads(row.train_regions) for row in holdout_audit.itertuples(index=False))
    add(rows, "holdout_train_region_excluded", int(region_overlap), region_overlap == 0)
    temporal_overlap = int(holdout_audit["train_year_max"].ge(holdout_audit["test_year_min"]).sum())
    add(rows, "holdout_train_precedes_test", temporal_overlap, temporal_overlap == 0)

    report = pd.DataFrame(rows); report.to_csv(args.run_dir / "independent_validation.csv", index=False)
    failures = report.loc[~report["passed"]]
    (args.run_dir / "independent_validation_ko.md").write_text(
        "# 전 해안 모형구조 확장 독립검증\n\n" + f"{len(report)}개 검사 중 실패 {len(failures)}개.\n", encoding="utf-8"
    )
    result = {"status": "passed" if failures.empty else "failed", "checks": len(report), "failures": len(failures),
              "report_sha256": sha256(args.run_dir / "independent_validation.csv")}
    (args.run_dir / "validation_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not failures.empty: raise AssertionError(failures.to_string(index=False))


if __name__ == "__main__": main()
