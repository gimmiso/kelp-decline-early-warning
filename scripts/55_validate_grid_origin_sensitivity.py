"""Independently recheck the grid-size/origin sensitivity outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


KEY = ["spec_id", "domain", "model", "year"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_check(rows: list[dict[str, object]], check: str, value: object, passed: bool) -> None:
    rows.append({"check": check, "value": value, "passed": bool(passed)})


def close(left: pd.Series, right: pd.Series, tolerance: float = 1e-12) -> bool:
    return bool(np.allclose(left, right, rtol=0, atol=tolerance, equal_nan=True))


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output_dir
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    specifications = pd.read_csv(output / "grid_specifications.csv")
    predictions = pd.read_csv(output / "predictions.csv")
    year_metrics = pd.read_csv(output / "year_metrics.csv")
    summaries = pd.read_csv(output / "model_summary.csv")
    estimates = pd.read_csv(output / "bootstrap_estimates.csv")
    differences = pd.read_csv(output / "bootstrap_differences.csv")
    draws = pd.read_csv(output / "bootstrap_draws.csv")
    agreement = pd.read_csv(output / "pixel_spatial_agreement.csv")
    results = pd.read_csv(output / "grid_sensitivity_results.csv")
    folds = pd.read_csv(output / "fold_audit.csv")
    quality = pd.read_csv(output / "quality_checks.csv")
    reproduction = pd.read_csv(output / "base_reproduction_checks.csv")

    checks: list[dict[str, object]] = []
    expected_thresholds = {5: 125, 10: 500, 20: 2000}
    scaled = specifications.loc[specifications["threshold_policy"].eq("area_scaled")]
    observed_thresholds = scaled.groupby("grid_size_km")["minimum_footprint_pixels"].unique()
    threshold_ok = all(
        len(observed_thresholds.loc[size]) == 1
        and int(observed_thresholds.loc[size][0]) == threshold
        for size, threshold in expected_thresholds.items()
    )
    add_check(checks, "twenty_locked_specifications", len(specifications), len(specifications) == 20)
    add_check(checks, "twelve_area_scaled_specifications", len(scaled), len(scaled) == 12)
    add_check(checks, "area_scaled_pixel_thresholds", str(expected_thresholds), threshold_ok)
    fixed = specifications.loc[specifications["threshold_policy"].eq("fixed_500")]
    add_check(
        checks,
        "fixed_500_is_secondary_5_and_20km_only",
        f"rows={len(fixed)}",
        len(fixed) == 8
        and set(fixed["grid_size_km"]) == {5, 20}
        and fixed["minimum_footprint_pixels"].eq(500).all(),
    )

    add_check(
        checks,
        "prediction_keys_unique",
        int(predictions.duplicated([*KEY, "cell_id"]).sum()),
        not predictions.duplicated([*KEY, "cell_id"]).any(),
    )
    add_check(
        checks,
        "prediction_probabilities_bounded_complete",
        int(predictions["score"].isna().sum()),
        predictions["score"].notna().all() and predictions["score"].between(0, 1).all(),
    )
    add_check(
        checks,
        "strict_forward_folds",
        int((folds["train_end"] >= folds["year"]).sum()),
        (folds["train_end"] < folds["year"]).all(),
    )
    add_check(
        checks,
        "twenty_test_years_per_model",
        int(predictions.groupby(["spec_id", "domain", "model"])["year"].nunique().min()),
        predictions.groupby(["spec_id", "domain", "model"])["year"].nunique().eq(20).all(),
    )

    recomputed_rows = []
    for keys, group in predictions.groupby(KEY, sort=True):
        ranked = group.sort_values(["score", "cell_id"], ascending=[False, True])
        n = len(ranked)
        events = int(ranked["event"].sum())
        prevalence = float(ranked["event"].mean())
        ap = float(average_precision_score(ranked["event"], ranked["score"]))
        k = max(1, math.ceil(n * 0.20))
        recomputed_rows.append(
            dict(
                zip(KEY, keys, strict=True),
                n=n,
                events=events,
                prevalence=prevalence,
                ap=ap,
                ap_lift=ap - prevalence,
                top20_k=k,
                top20_tp=int(ranked.head(k)["event"].sum()),
            )
        )
    recomputed = pd.DataFrame(recomputed_rows)
    compared = year_metrics.merge(recomputed, on=KEY, suffixes=("_saved", "_recomputed"), validate="one_to_one")
    metric_columns = ["n", "events", "prevalence", "ap", "ap_lift", "top20_k", "top20_tp"]
    metric_ok = all(
        close(compared[f"{column}_saved"], compared[f"{column}_recomputed"])
        for column in metric_columns
    )
    add_check(checks, "year_metrics_recomputed_from_predictions", len(compared), metric_ok)

    macro = (
        year_metrics.loc[year_metrics["estimable"]]
        .groupby(["spec_id", "domain", "model"])["ap_lift"]
        .mean()
        .rename("recomputed")
        .reset_index()
    )
    macro_compare = summaries.merge(macro, on=["spec_id", "domain", "model"], validate="one_to_one")
    add_check(
        checks,
        "macro_year_lifts_recomputed",
        len(macro_compare),
        close(macro_compare["macro_year_ap_lift"], macro_compare["recomputed"]),
    )
    recall = (
        year_metrics.groupby(["spec_id", "domain", "model"])[["top20_tp", "events"]]
        .sum()
        .assign(recomputed=lambda frame: frame["top20_tp"] / frame["events"])
        .reset_index()
    )
    recall_compare = summaries.merge(
        recall[["spec_id", "domain", "model", "recomputed"]],
        on=["spec_id", "domain", "model"],
        validate="one_to_one",
    )
    add_check(
        checks,
        "top20_recall_recomputed",
        len(recall_compare),
        close(recall_compare["top20_recall_micro"], recall_compare["recomputed"]),
    )

    model_draws = draws.loc[draws["kind"].eq("model")]
    model_quantiles = (
        model_draws.groupby(["spec_id", "domain", "name"])["value"]
        .quantile([0.025, 0.975])
        .unstack()
        .rename(columns={0.025: "draw_ci_low", 0.975: "draw_ci_high"})
        .reset_index()
        .rename(columns={"name": "model"})
    )
    estimate_compare = estimates.merge(
        model_quantiles, on=["spec_id", "domain", "model"], validate="one_to_one"
    )
    add_check(
        checks,
        "model_bootstrap_quantiles_recomputed",
        len(estimate_compare),
        close(estimate_compare["ci_low"], estimate_compare["draw_ci_low"])
        and close(estimate_compare["ci_high"], estimate_compare["draw_ci_high"]),
    )
    difference_draws = draws.loc[draws["kind"].eq("difference")]
    difference_quantiles = (
        difference_draws.groupby(["spec_id", "domain", "name"])["value"]
        .quantile([0.025, 0.975])
        .unstack()
        .rename(columns={0.025: "draw_ci_low", 0.975: "draw_ci_high"})
        .reset_index()
        .rename(columns={"name": "comparison"})
    )
    difference_compare = differences.merge(
        difference_quantiles,
        on=["spec_id", "domain", "comparison"],
        validate="one_to_one",
    )
    add_check(
        checks,
        "difference_bootstrap_quantiles_recomputed",
        len(difference_compare),
        close(difference_compare["ci_low"], difference_compare["draw_ci_low"])
        and close(difference_compare["ci_high"], difference_compare["draw_ci_high"]),
    )
    add_check(
        checks,
        "two_thousand_draws_each",
        int(draws.groupby(["spec_id", "domain", "kind", "name"])["replicate"].nunique().min()),
        draws.groupby(["spec_id", "domain", "kind", "name"])["replicate"].nunique().eq(2000).all(),
    )

    reconstructed_jaccard = agreement["label_tp_pixels"] / (
        agreement["label_tp_pixels"]
        + agreement["label_fp_pixels"]
        + agreement["label_fn_pixels"]
    )
    add_check(
        checks,
        "label_jaccard_recomputed_from_pixel_counts",
        len(agreement),
        close(agreement["label_positive_jaccard"], reconstructed_jaccard),
    )
    identity = agreement.loc[
        agreement["spec_id"].eq(agreement["reference_spec"])
        & agreement["reference_scope"].eq("same_size_policy_o00")
    ]
    add_check(
        checks,
        "same_grid_spatial_identity",
        len(identity),
        len(identity) == 5
        and identity["label_positive_jaccard"].eq(1).all()
        and identity["top20_spatial_jaccard"].eq(1).all(),
    )
    global_rows = agreement.loc[agreement["reference_scope"].eq("global_10km_o00")]
    add_check(
        checks,
        "global_agreement_rows_complete",
        len(global_rows),
        len(global_rows) == 20 and global_rows["common_pixel_years"].gt(0).all(),
    )

    scaled_results = results.loc[results["threshold_policy"].eq("area_scaled")]
    add_check(
        checks,
        "current_lift_supported_all_scaled_grids",
        int(scaled_results["current_ci_low"].gt(0).sum()),
        len(scaled_results) == 12 and scaled_results["current_ci_low"].gt(0).all(),
    )
    add_check(
        checks,
        "trajectory_support_count_is_eight",
        int(scaled_results["trajectory_ci_low"].gt(0).sum()),
        int(scaled_results["trajectory_ci_low"].gt(0).sum()) == 8,
    )
    add_check(
        checks,
        "oisst_support_count_is_zero",
        int(scaled_results["oisst_ci_low"].gt(0).sum()),
        int(scaled_results["oisst_ci_low"].gt(0).sum()) == 0,
    )
    primary = results.loc[results["spec_id"].eq(config["primary_reference_spec"])].iloc[0]
    add_check(
        checks,
        "primary_reference_values",
        f"cells={int(primary.stable_cells)}, lift={primary.current_lift:.12f}",
        int(primary.stable_cells) == 167
        and np.isclose(primary.current_lift, 0.17643625413190192, rtol=0, atol=1e-12),
    )
    add_check(
        checks,
        "base_panel_reproduction_passed",
        int(reproduction["mismatches"].sum()),
        reproduction["passed"].astype(bool).all() and reproduction["mismatches"].sum() == 0,
    )
    add_check(
        checks,
        "run_quality_checks_passed",
        int(quality["passed"].astype(bool).sum()),
        quality["passed"].astype(bool).all(),
    )

    hash_failures = []
    for filename, expected in manifest["output_sha256"].items():
        observed = sha256(output / filename)
        if observed != expected:
            hash_failures.append(filename)
    add_check(checks, "manifest_output_hashes", len(hash_failures), not hash_failures)
    add_check(
        checks,
        "manifest_complete_and_classified",
        manifest.get("status"),
        manifest.get("status") == "complete"
        and manifest.get("classification") == "pilot_informed_locked_robustness",
    )

    checks_frame = pd.DataFrame(checks)
    checks_frame.to_csv(output / "validation_recheck.csv", index=False)
    passed = int(checks_frame["passed"].sum())
    total = len(checks_frame)
    report = [
        "# 격자 크기·원점 민감도 독립 검증",
        "",
        f"- 결과: **{passed}/{total} checks passed**",
        "- 저장된 예측확률에서 연도별 AP lift와 top-20% 포착률을 재계산했다.",
        "- 저장된 2,000회 bootstrap draws에서 95% 분위수를 재계산했다.",
        "- 픽셀 혼동행렬에서 라벨 Jaccard를 재계산하고, 동일 격자의 공간 일치도 1.0을 확인했다.",
        "- 실행 manifest의 모든 원래 산출물 SHA-256을 다시 확인했다.",
        "",
        "## 검증 결과",
        "",
        *[
            f"- {'PASS' if row.passed else 'FAIL'} — `{row.check}`: {row.value}"
            for row in checks_frame.itertuples(index=False)
        ],
    ]
    (output / "validation_report_ko.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if passed != total:
        raise AssertionError(checks_frame.loc[~checks_frame["passed"]].to_string(index=False))
    print(json.dumps({"passed": passed, "total": total}, ensure_ascii=False))


if __name__ == "__main__":
    main()
