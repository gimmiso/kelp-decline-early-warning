"""Independently validate pooled California hierarchical-additive outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-dir", type=Path, required=True); return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args(); manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    pred = pd.read_csv(args.run_dir / "predictions.csv"); folds = pd.read_csv(args.run_dir / "fold_audit.csv")
    annual = pd.read_csv(args.run_dir / "annual_metrics.csv"); summary = pd.read_csv(args.run_dir / "model_summary.csv")
    increments = pd.read_csv(args.run_dir / "increment_summary.csv"); tuning = pd.read_csv(args.run_dir / "inner_tuning_audit.csv")
    matched = pd.read_csv(args.run_dir / "matched_old_new_annual.csv")
    rows = []
    def add(check: str, value: object, passed: bool) -> None: rows.append({"check": check, "value": value, "passed": bool(passed)})
    add("manifest_complete", manifest.get("status"), manifest.get("status") == "complete")
    bad_hash = [n for n, h in manifest["outputs"].items() if not (args.run_dir / n).exists() or sha256(args.run_dir / n) != h]
    add("manifest_hashes", len(bad_hash), not bad_hash)
    key = ["support_m", "site_id", "year", "feature_set"]
    add("prediction_keys_unique", int(pred.duplicated(key).sum()), not pred.duplicated(key).any())
    add("scores_bounded", int((~pred.score.between(0, 1)).sum()), pred.score.between(0, 1).all())
    add("forward_only", int(folds.train_end.ge(folds.year).sum()), folds.train_end.lt(folds.year).all())
    add("five_blocks_same_samples", int(pred.groupby(["support_m", "site_id", "year"]).feature_set.nunique().ne(5).sum()), pred.groupby(["support_m", "site_id", "year"]).feature_set.nunique().eq(5).all())
    tune_failures = 0
    for keys, group in tuning.groupby(["support_m", "year", "feature_set"]):
        best = group.sort_values(["inner_macro_ap_lift", "C"], ascending=[False, True]).iloc[0]
        chosen = folds.loc[folds.support_m.eq(keys[0]) & folds.year.eq(keys[1]) & folds.feature_set.eq(keys[2]), "chosen_C"].iloc[0]
        tune_failures += int(not np.isclose(best.C, chosen))
    add("inner_tuning_argmax", tune_failures, tune_failures == 0)
    recalculated = []
    for keys, group in pred.groupby(["support_m", "feature_set", "year"]):
        ap = average_precision_score(group.event, group.score)
        recalculated.append({"support_m": keys[0], "feature_set": keys[1], "year": keys[2], "check": ap - group.event.mean()})
    merged = annual.merge(pd.DataFrame(recalculated), on=["support_m", "feature_set", "year"], validate="one_to_one")
    add("annual_ap_recomputed", float((merged.ap_lift - merged.check).abs().max()), np.allclose(merged.ap_lift, merged.check))
    errors = []
    for row in summary.itertuples():
        part = pred.loc[pred.support_m.eq(row.support_m) & pred.feature_set.eq(row.feature_set)]
        errors.append(abs(brier_score_loss(part.event, part.score) - row.brier))
    add("brier_recomputed", max(errors), max(errors) < 1e-12)
    inc_errors = []
    for row in increments.itertuples():
        part = annual.loc[annual.support_m.eq(row.support_m)].pivot(index="year", columns="feature_set", values="ap_lift")
        inc_errors.append(abs((part[row.left_feature_set] - part[row.right_feature_set]).mean() - row.estimate))
    add("increments_recomputed", max(inc_errors), max(inc_errors) < 1e-12)
    add("old_new_matched_nonempty", len(matched), len(matched) > 0)
    report = pd.DataFrame(rows); report.to_csv(args.run_dir / "independent_validation.csv", index=False); failures = report.loc[~report.passed]
    (args.run_dir / "independent_validation_ko.md").write_text("# California pooled 계층모형 독립검증\n\n" + f"{len(report)}개 검사 중 실패 {len(failures)}개.\n", encoding="utf-8")
    result = {"status": "passed" if failures.empty else "failed", "checks": len(report), "failures": len(failures), "report_sha256": sha256(args.run_dir / "independent_validation.csv")}
    (args.run_dir / "validation_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not failures.empty: raise AssertionError(failures.to_string(index=False))


if __name__ == "__main__": main()
