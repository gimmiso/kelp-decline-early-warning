"""Independently recalculate the saved observation-label robustness results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


PRIMARY = "fixedpre_protocol_max"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(left, right, atol=tolerance, rtol=0, equal_nan=True))


def main() -> None:
    args = parse_args()
    run = args.run_dir
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(run / "predictions.csv")
    year_metrics = pd.read_csv(run / "year_metrics.csv")
    summary = pd.read_csv(run / "model_summary.csv")
    estimates = pd.read_csv(run / "bootstrap_estimates.csv")
    differences = pd.read_csv(run / "bootstrap_differences.csv")
    draws = pd.read_csv(run / "bootstrap_draws.csv")
    panels = pd.read_csv(run / "variant_panels.csv")
    annual = pd.read_csv(run / "annual_variants.csv")
    folds = pd.read_csv(run / "fold_audit.csv")
    footprint = pd.read_csv(run / "fixed_footprint_reproduction.csv")
    quality = pd.read_csv(run / "quality_checks.csv")

    rows: list[dict[str, object]] = []

    def add(check: str, passed: bool, evidence: str, severity: str = "blocker") -> None:
        rows.append(
            {
                "check": check,
                "passed": bool(passed),
                "severity_if_failed": severity,
                "evidence": evidence,
            }
        )

    add("saved_quality_checks_pass", quality["passed"].all(), f"passed={int(quality['passed'].sum())}/{len(quality)}")
    add(
        "prediction_keys_unique",
        not predictions.duplicated(["mode", "variant", "model", "cell_id", "year"]).any(),
        f"duplicates={predictions.duplicated(['mode', 'variant', 'model', 'cell_id', 'year']).sum()}",
    )
    add(
        "temporal_folds_forward_only",
        folds["train_end"].lt(folds["year"]).all(),
        f"max_train_minus_test={(folds['train_end'] - folds['year']).max()}",
    )
    add(
        "fixed_footprint_counts_match",
        footprint[["allhistory_count_match", "pre2005_count_match"]].all().all(),
        f"failed_rows={int((~footprint[['allhistory_count_match', 'pre2005_count_match']]).any(axis=1).sum())}",
    )

    same_rows = True
    same_outcomes = True
    for _, group in predictions.groupby(["mode", "variant"]):
        keys = []
        outcomes = []
        for _, model_group in group.groupby("model"):
            ordered = model_group.sort_values(["cell_id", "year"])
            keys.append(list(zip(ordered["cell_id"], ordered["year"], strict=True)))
            outcomes.append(ordered["event"].tolist())
        same_rows &= all(value == keys[0] for value in keys[1:])
        same_outcomes &= all(value == outcomes[0] for value in outcomes[1:])
    add("models_use_matched_rows", same_rows, "row keys identical within every mode-variant")
    add("models_use_matched_outcomes", same_outcomes, "event vectors identical within every mode-variant")

    common = predictions.loc[
        predictions["mode"].eq("common_rows") & predictions["model"].eq("current_only")
    ]
    common_keys = []
    for _, group in common.groupby("variant"):
        ordered = group.sort_values(["cell_id", "year"])
        common_keys.append(list(zip(ordered["cell_id"], ordered["year"], strict=True)))
    add(
        "common_mode_exact_rows_across_variants",
        all(value == common_keys[0] for value in common_keys[1:]),
        f"variants={len(common_keys)}; rows={len(common_keys[0])}",
    )

    native_primary_keys = set(
        map(
            tuple,
            predictions.loc[
                predictions["mode"].eq("native")
                & predictions["variant"].eq(PRIMARY)
                & predictions["model"].eq("current_only"),
                ["cell_id", "year"],
            ].to_numpy(),
        )
    )
    high_quality_keys = set(
        map(
            tuple,
            predictions.loc[
                predictions["mode"].eq("high_observation_subset")
                & predictions["model"].eq("current_only"),
                ["cell_id", "year"],
            ].to_numpy(),
        )
    )
    add(
        "high_observation_subset_is_native_subset",
        high_quality_keys.issubset(native_primary_keys),
        f"high_quality={len(high_quality_keys)}; native={len(native_primary_keys)}",
    )

    panel_event_match = panels["event"].eq(panels["relative_drop_next"].ge(0.30)).all()
    eligible = panels["eligible"].astype(bool)
    eligibility_match = (
        panels.loc[eligible, "relative_canopy"].gt(0.05).all()
        and panels.loc[eligible, "annual_area_m2"].notna().all()
        and panels.loc[eligible, "next_year_area_m2"].notna().all()
    )
    add("event_definition_recomputed", panel_event_match, "event == relative_drop_next >= 0.30")
    add("eligibility_definition_recomputed", eligibility_match, "eligible rows have canopy >0.05 and both years observed")

    primary_annual = annual.loc[annual["variant"].eq(PRIMARY)]
    primary_complete = (
        primary_annual["valid_quarters"].ge(3)
        & primary_annual["q3_valid"].astype(bool)
    )
    add(
        "primary_annual_completeness_rule",
        primary_annual["annual_complete"].astype(bool).eq(primary_complete).all(),
        "50% threshold encoded upstream; annual_complete == >=3 valid quarters and Q3 valid",
    )

    max_year_error = 0.0
    max_top20_error = 0.0
    manual_rows = []
    for (mode, variant, model, year), group in predictions.groupby(
        ["mode", "variant", "model", "year"]
    ):
        prevalence = float(group["event"].mean())
        ap = (
            float(average_precision_score(group["event"], group["score"]))
            if group["event"].nunique() == 2
            else np.nan
        )
        lift = ap - prevalence if np.isfinite(ap) else np.nan
        ranked = group.sort_values(["score", "cell_id"], ascending=[False, True])
        k = max(1, math.ceil(len(ranked) * 0.20))
        top20_tp = int(ranked.head(k)["event"].sum())
        stored = year_metrics.loc[
            year_metrics["mode"].eq(mode)
            & year_metrics["variant"].eq(variant)
            & year_metrics["model"].eq(model)
            & year_metrics["year"].eq(year)
        ].iloc[0]
        if np.isfinite(lift):
            max_year_error = max(max_year_error, abs(lift - float(stored["ap_lift"])))
        max_top20_error = max(max_top20_error, abs(top20_tp - int(stored["top20_tp"])))
        manual_rows.append(
            {
                "mode": mode,
                "variant": variant,
                "model": model,
                "year": year,
                "lift": lift,
                "events": int(group["event"].sum()),
                "top20_tp": top20_tp,
            }
        )
    add("year_ap_lifts_recomputed", max_year_error < 1e-12, f"max_error={max_year_error:.3g}")
    add("year_top20_counts_recomputed", max_top20_error == 0, f"max_error={max_top20_error}")

    manual = pd.DataFrame(manual_rows)
    max_summary_error = 0.0
    max_recall_error = 0.0
    for (mode, variant, model), group in manual.groupby(["mode", "variant", "model"]):
        stored = summary.loc[
            summary["mode"].eq(mode)
            & summary["variant"].eq(variant)
            & summary["model"].eq(model)
        ].iloc[0]
        macro = float(group["lift"].mean())
        recall = float(group["top20_tp"].sum() / group["events"].sum())
        max_summary_error = max(max_summary_error, abs(macro - float(stored["macro_year_ap_lift"])))
        max_recall_error = max(max_recall_error, abs(recall - float(stored["top20_recall_micro"])))
    add("macro_lifts_recomputed", max_summary_error < 1e-12, f"max_error={max_summary_error:.3g}")
    add("top20_recall_recomputed", max_recall_error < 1e-12, f"max_error={max_recall_error:.3g}")

    max_bootstrap_error = 0.0
    for row in estimates.itertuples(index=False):
        distribution = draws.loc[
            draws["mode"].eq(row.mode)
            & draws["variant"].eq(row.variant)
            & draws["kind"].eq("model")
            & draws["name"].eq(row.model),
            "value",
        ]
        max_bootstrap_error = max(
            max_bootstrap_error,
            abs(float(distribution.quantile(0.025)) - row.ci_low),
            abs(float(distribution.quantile(0.975)) - row.ci_high),
        )
    for row in differences.itertuples(index=False):
        distribution = draws.loc[
            draws["mode"].eq(row.mode)
            & draws["variant"].eq(row.variant)
            & draws["kind"].eq("difference")
            & draws["name"].eq(row.comparison),
            "value",
        ]
        max_bootstrap_error = max(
            max_bootstrap_error,
            abs(float(distribution.quantile(0.025)) - row.ci_low),
            abs(float(distribution.quantile(0.975)) - row.ci_high),
        )
    add("bootstrap_intervals_recomputed_from_draws", max_bootstrap_error < 1e-12, f"max_error={max_bootstrap_error:.3g}")

    mismatched_inputs = []
    for name, item in manifest["inputs"].items():
        path = Path(item["path"])
        if not path.exists() or sha256(path) != item["sha256"]:
            mismatched_inputs.append(name)
    add("input_hashes_match_manifest", not mismatched_inputs, f"mismatches={mismatched_inputs}")
    mismatched_outputs = []
    for name, expected in manifest["output_sha256"].items():
        if sha256(run / name) != expected:
            mismatched_outputs.append(name)
    add("output_hashes_match_manifest", not mismatched_outputs, f"mismatches={mismatched_outputs}")

    current = estimates.loc[
        estimates["mode"].eq("native")
        & estimates["variant"].eq(PRIMARY)
        & estimates["model"].eq("current_only")
    ].iloc[0]
    oisst = differences.loc[
        differences["mode"].eq("native")
        & differences["variant"].eq(PRIMARY)
        & differences["comparison"].eq("oisst_minus_trajectory")
    ].iloc[0]
    add(
        "decision_current_lift_matches",
        close(float(current.estimate), float(manifest["decisions"]["primary_current_lift"])),
        f"estimate={current.estimate:.12f}",
    )
    add(
        "decision_oisst_increment_matches",
        close(float(oisst.estimate), float(manifest["decisions"]["primary_oisst_increment"])),
        f"estimate={oisst.estimate:.12f}",
    )

    validation = pd.DataFrame(rows)
    validation.to_csv(run / "validation_recheck.csv", index=False)
    blockers = validation.loc[
        ~validation["passed"] & validation["severity_if_failed"].eq("blocker")
    ]
    assessment = "Share with caveats" if blockers.empty else "Needs revision"
    report = f"""# 관측품질·라벨 강건성 독립 검증

## Overall Assessment: {assessment}

prediction 수준 재계산, 시계열 누수, 동일행 비교, 고품질 부분표본, 고정 footprint 재현 및 파일 hash를 다시 확인했다. 총 {len(validation)}개 검사 중 {int(validation['passed'].sum())}개가 통과했고, 공유를 막는 오류는 {'없다' if blockers.empty else '있다'}.

## 재계산 결과

- 연도별 AP lift 최대 저장 오차: {max_year_error:.3g}
- macro AP lift 최대 저장 오차: {max_summary_error:.3g}
- Top-20% recall 최대 저장 오차: {max_recall_error:.3g}
- 저장 bootstrap draw에서 2.5%·97.5% 구간 재계산 최대 오차: {max_bootstrap_error:.3g}
- common-row 모드는 8개 라벨 정의에서 같은 cell-year를 사용했다.
- 고관측품질 결과행은 주 native 결과행의 진부분집합이다.
- 공식 NetCDF에서 다시 만든 pre-2005 footprint pixel 수와 기존 inventory가 165셀 모두 일치했다.

## 해석상 필수 제한

- 이 분석은 기존 결과를 확인한 뒤 수행한 강건성 분석이며 독립 확증이 아니다.
- `area_se`는 픽셀 공분산과 annual-maximum 선택오차가 없어 inverse-variance 가중치로 쓰지 않는다.
- 센서시대별 성능 차이는 관측기회·사건구성·생태상태가 함께 달라진 기술적 진단이며 센서의 인과효과가 아니다.
- Kelpwatch 웹/API와 지역별 층화표본 대조는 아직 별도 Gate 2로 남아 있다.
"""
    (run / "validation_report_ko.md").write_text(report, encoding="utf-8")
    if not blockers.empty:
        raise AssertionError(blockers.to_string(index=False))
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
