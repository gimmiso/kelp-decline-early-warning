"""Independently validate the California ecological method-audit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runner() -> object:
    path = Path(__file__).with_name("72_run_california_local_ecology_method_audit.py")
    spec = importlib.util.spec_from_file_location("method_audit_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runner at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(rows: list[dict[str, object]], check: str, value: object, passed: bool, severity: str = "error") -> None:
    rows.append({"check": check, "value": value, "passed": bool(passed), "severity": severity})


def main() -> None:
    args = parse_args()
    runner = load_runner()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assignment = pd.read_csv(args.run_dir / "context_site_assignment.csv")
    field = pd.read_csv(args.run_dir / "field_site_year_features.csv")
    analysis = pd.read_csv(args.run_dir / "analysis_rows.csv")
    folds = pd.read_csv(args.run_dir / "fold_audit.csv")
    predictions = pd.read_csv(args.run_dir / "predictions.csv")
    annual = pd.read_csv(args.run_dir / "annual_metrics.csv")
    increments = pd.read_csv(args.run_dir / "increment_summary.csv")
    decisions = pd.read_csv(args.run_dir / "decision_table.csv")
    missingness = pd.read_csv(args.run_dir / "feature_missingness.csv")
    panel = pd.read_csv(args.panel)
    rows: list[dict[str, object]] = []

    record(rows, "manifest_complete", manifest.get("status"), manifest.get("status") == "complete")
    hash_failures = []
    for name, expected in manifest["outputs"].items():
        path = args.run_dir / name
        if not path.exists() or sha256(path) != expected:
            hash_failures.append(name)
    record(rows, "manifest_output_hashes", len(hash_failures), not hash_failures)
    expected_context_counts = {"bull_north": 10, "giant_central_southwest": 88, "giant_southeast": 26}
    actual_context_counts = assignment.groupby("context")["site_id"].nunique().to_dict()
    record(rows, "locked_context_site_counts", json.dumps(actual_context_counts, sort_keys=True), actual_context_counts == expected_context_counts)
    record(rows, "context_sites_nonoverlapping", int(assignment["site_id"].duplicated().sum()), not assignment["site_id"].duplicated().any())
    record(rows, "field_site_year_unique", int(field.duplicated(["site_id", "year"]).sum()), not field.duplicated(["site_id", "year"]).any())
    record(rows, "analysis_support_site_year_unique", int(analysis.duplicated(["support_m", "site_id", "year"]).sum()), not analysis.duplicated(["support_m", "site_id", "year"]).any())

    panel_events = panel[["support_m", "site_id", "year", "event"]]
    event_check = analysis[["support_m", "site_id", "year", "event"]].merge(
        panel_events, on=["support_m", "site_id", "year"], suffixes=("_analysis", "_panel"), validate="one_to_one"
    )
    event_mismatches = int(event_check["event_analysis"].ne(event_check["event_panel"]).sum())
    record(rows, "event_labels_match_locked_panel", event_mismatches, event_mismatches == 0)

    fit_folds = folds.loc[folds["status"].eq("fit")]
    leakage = int(fit_folds["train_end"].ge(fit_folds["year"]).sum())
    record(rows, "expanding_folds_have_no_future_rows", leakage, leakage == 0)
    one_class_fit = int((fit_folds["events_train"].eq(0) | fit_folds["events_train"].eq(fit_folds["n_train"])).sum())
    record(rows, "fitted_training_folds_have_two_classes", one_class_fit, one_class_fit == 0)

    prediction_key = ["support_m", "site_id", "context", "year", "model_family", "feature_set"]
    record(rows, "prediction_keys_unique", int(predictions.duplicated(prediction_key).sum()), not predictions.duplicated(prediction_key).any())
    record(rows, "prediction_scores_finite", int((~np.isfinite(predictions["score"])).sum()), np.isfinite(predictions["score"]).all())
    record(rows, "prediction_scores_bounded", int((~predictions["score"].between(0, 1)).sum()), predictions["score"].between(0, 1).all())
    sample_key = ["support_m", "site_id", "context", "year", "model_family"]
    feature_set_count = predictions.groupby(sample_key)["feature_set"].nunique()
    record(rows, "identical_prediction_samples_across_blocks", int(feature_set_count.ne(5).sum()), feature_set_count.eq(5).all())

    all_missing_used = []
    for (support_m, context), group in analysis.groupby(["support_m", "context"], sort=True):
        used = set()
        for feature_set in runner.FEATURE_SET_ORDER:
            used.update(runner.features_for(context, feature_set))
        for feature in used:
            if group[feature].isna().all():
                all_missing_used.append(f"{support_m}:{context}:{feature}")
    record(rows, "no_context_used_feature_is_all_missing", len(all_missing_used), not all_missing_used)
    north = analysis.loc[analysis["context"].eq("bull_north")]
    record(rows, "north_depth_fallback_is_explicit", float(north["depth_mapped_fallback"].mean()), north["depth_mapped_fallback"].eq(1).all(), severity="warning")

    recalculated = []
    for keys, group in predictions.groupby(["support_m", "model_family", "feature_set", "year"], sort=True):
        if group["event"].nunique() < 2:
            continue
        ap = average_precision_score(group["event"], group["score"])
        recalculated.append(
            {
                "support_m": int(keys[0]),
                "model_family": keys[1],
                "feature_set": keys[2],
                "year": int(keys[3]),
                "ap_lift_recalc": float(ap - group["event"].mean()),
                "brier_recalc": float(brier_score_loss(group["event"], group["score"])),
            }
        )
    recalculated = pd.DataFrame(recalculated)
    metric_check = annual.merge(recalculated, on=["support_m", "model_family", "feature_set", "year"], validate="one_to_one")
    ap_error = float((metric_check["ap_lift"] - metric_check["ap_lift_recalc"]).abs().max())
    brier_error = float((metric_check["brier"] - metric_check["brier_recalc"]).abs().max())
    record(rows, "annual_ap_lift_recomputed", ap_error, ap_error < 1e-12)
    record(rows, "annual_brier_recomputed", brier_error, brier_error < 1e-12)

    increment_errors = []
    for row in increments.itertuples(index=False):
        group = annual.loc[annual["support_m"].eq(row.support_m) & annual["model_family"].eq(row.model_family)]
        pivot = group.pivot(index="year", columns="feature_set", values="ap_lift")
        estimate = float((pivot[row.left_feature_set] - pivot[row.right_feature_set]).dropna().mean())
        increment_errors.append(abs(estimate - row.estimate))
    max_increment_error = float(max(increment_errors))
    record(rows, "increment_estimates_recomputed", max_increment_error, max_increment_error < 1e-12)

    primary = decisions.loc[decisions["comparison"].eq(config["interpretation"]["primary_comparison"])].iloc[0]
    source = increments.loc[
        increments["support_m"].eq(config["interpretation"]["primary_support_m"])
        & increments["model_family"].eq(config["interpretation"]["primary_family"])
        & increments["comparison"].eq(config["interpretation"]["primary_comparison"])
    ].iloc[0]
    primary_match = bool(
        np.isclose(primary["primary_estimate"], source["estimate"])
        and np.isclose(primary["primary_ci_low"], source["hierarchical_ci_low"])
        and np.isclose(primary["primary_ci_high"], source["hierarchical_ci_high"])
    )
    record(rows, "primary_decision_matches_increment", primary["primary_estimate"], primary_match)
    recomputed_stable = bool(
        source["estimate"] > 0
        and source["hierarchical_ci_low"] > 0
        and primary["same_direction_families_300m"] >= config["interpretation"]["requires_same_direction_model_families"]
        and primary["logistic_spline_estimate_1000m"] > 0
    )
    record(rows, "primary_stability_rule_recomputed", recomputed_stable, recomputed_stable == bool(primary["stable_positive_increment"]))

    report = pd.DataFrame(rows)
    report.to_csv(args.run_dir / "independent_validation.csv", index=False)
    error_failures = report.loc[report["severity"].eq("error") & ~report["passed"]]
    lines = [
        "# California Giraldo 방법론 감사 독립 검증",
        "",
        f"오류 등급 검사는 {len(report.loc[report['severity'].eq('error')])}개 중 {len(error_failures)}개가 실패했다.",
        "",
        "- 원본 30% 급감 라벨과 분석 라벨을 다시 대조했다.",
        "- 모든 expanding-window가 예측연도 이전 자료만 사용하는지 확인했다.",
        "- 동일 셀-연도 표본에서 다섯 정보블록이 비교되는지 확인했다.",
        "- AP lift, Brier score, 정보블록 증분 및 주 판정식을 원 예측치에서 독립 재계산했다.",
        "- 북부 10개 Reef Check 지점의 지도형 수심 대체가 숨겨지지 않았는지 확인했다.",
    ]
    (args.run_dir / "independent_validation_ko.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    validation_manifest = {
        "status": "passed" if error_failures.empty else "failed",
        "checks": len(report),
        "error_failures": int(len(error_failures)),
        "report_sha256": sha256(args.run_dir / "independent_validation.csv"),
        "summary_sha256": sha256(args.run_dir / "independent_validation_ko.md"),
    }
    (args.run_dir / "validation_manifest.json").write_text(
        json.dumps(validation_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation_manifest, ensure_ascii=False, indent=2))
    if not error_failures.empty:
        raise AssertionError(error_failures.to_string(index=False))


if __name__ == "__main__":
    main()
