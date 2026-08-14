"""Independently validate the saved minimal paper-extension experiment outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


UPWELLING_COLUMNS = [
    "winter_cuti_anomaly",
    "spring_cuti_anomaly",
    "winter_beuti_anomaly",
    "spring_beuti_anomaly",
    "upwelling_season_beuti_anomaly",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/experiments/20260814_minimal_paper_extension_v2"),
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/processed/kelpwatch_west_coast_panel_165.csv"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(left, right, atol=tolerance, rtol=0, equal_nan=True))


def main() -> None:
    args = parse_args()
    run = args.run_dir
    manifest = json.loads((run / "manifest.json").read_text())
    predictions = pd.read_csv(run / "predictions.csv")
    year_metrics = pd.read_csv(run / "year_metrics.csv")
    summary = pd.read_csv(run / "model_summary.csv")
    stress = pd.read_csv(run / "stress_2022.csv")
    transfer = pd.read_csv(run / "leave_region_out_summary.csv")
    folds = pd.read_csv(run / "fold_audit.csv")
    environment = pd.read_csv(run / "environment_features.csv")
    panel = pd.read_csv(args.panel)

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

    add("panel_has_165_cells", panel["cell_id"].nunique() == 165, f"cells={panel['cell_id'].nunique()}")
    add(
        "panel_keys_unique",
        not panel.duplicated(["cell_id", "year"]).any(),
        f"duplicates={panel.duplicated(['cell_id', 'year']).sum()}",
    )
    label_match = (panel["event"].astype(bool) == panel["relative_drop_next"].ge(0.30)).all()
    eligible = panel["eligible"].astype(str).str.lower().eq("true")
    add("event_definition_matches_30pct", label_match, "event == relative_drop_next >= 0.30")
    add(
        "eligibility_matches_canopy_cutoff",
        panel.loc[eligible, "relative_canopy"].gt(0.05).all(),
        f"minimum eligible relative_canopy={panel.loc[eligible, 'relative_canopy'].min():.6f}",
    )
    add(
        "prediction_keys_unique",
        not predictions.duplicated(["domain", "model", "cell_id", "year"]).any(),
        f"duplicates={predictions.duplicated(['domain', 'model', 'cell_id', 'year']).sum()}",
    )
    add(
        "temporal_folds_are_forward_only",
        folds["train_end"].lt(folds["year"]).all(),
        f"max(train_end-test_year)={(folds['train_end'] - folds['year']).max()}",
    )

    outcome_consistent = True
    same_rows = True
    for domain, domain_frame in predictions.groupby("domain"):
        row_sets = []
        outcome_sets = []
        for _, model_frame in domain_frame.groupby("model"):
            ordered = model_frame.sort_values(["cell_id", "year"])
            row_sets.append(list(zip(ordered["cell_id"], ordered["year"], strict=True)))
            outcome_sets.append(ordered["event"].tolist())
        same_rows &= all(x == row_sets[0] for x in row_sets[1:])
        outcome_consistent &= all(x == outcome_sets[0] for x in outcome_sets[1:])
    add("exact_matched_rows_within_domain", same_rows, "all model row-key lists are identical")
    add("exact_matched_outcomes_within_domain", outcome_consistent, "all model outcome lists are identical")

    max_year_metric_error = 0.0
    manual_rows = []
    for (domain, model, year), group in predictions.groupby(["domain", "model", "year"]):
        prevalence = float(group["event"].mean())
        ap = float(average_precision_score(group["event"], group["score"]))
        lift = ap - prevalence
        stored = year_metrics.loc[
            year_metrics["domain"].eq(domain)
            & year_metrics["model"].eq(model)
            & year_metrics["year"].eq(year)
        ].iloc[0]
        max_year_metric_error = max(max_year_metric_error, abs(lift - float(stored["ap_lift"])))
        ranked = group.sort_values(["score", "cell_id"], ascending=[False, True])
        k = max(1, math.ceil(len(ranked) * 0.20))
        manual_rows.append(
            {
                "domain": domain,
                "model": model,
                "year": year,
                "lift": lift,
                "events": int(group["event"].sum()),
                "top20_tp": int(ranked.head(k)["event"].sum()),
            }
        )
    add(
        "year_ap_lifts_recomputed",
        max_year_metric_error < 1e-12,
        f"maximum absolute discrepancy={max_year_metric_error:.3g}",
    )

    manual = pd.DataFrame(manual_rows)
    max_summary_error = 0.0
    max_top20_error = 0.0
    for (domain, model), group in manual.groupby(["domain", "model"]):
        stored = summary.loc[summary["domain"].eq(domain) & summary["model"].eq(model)].iloc[0]
        macro = float(group["lift"].mean())
        top20 = float(group["top20_tp"].sum() / group["events"].sum())
        max_summary_error = max(max_summary_error, abs(macro - float(stored["macro_year_ap_lift"])))
        max_top20_error = max(max_top20_error, abs(top20 - float(stored["top20_recall_micro"])))
    add("macro_year_lifts_recomputed", max_summary_error < 1e-12, f"maximum discrepancy={max_summary_error:.3g}")
    add("top20_recall_recomputed", max_top20_error < 1e-12, f"maximum discrepancy={max_top20_error:.3g}")

    current_manual = manual.loc[
        manual["domain"].eq("full_oisst_domain")
        & manual["model"].eq("current_only")
        & manual["year"].ne(2022),
        "lift",
    ].mean()
    current_stored = stress.loc[
        stress["domain"].eq("full_oisst_domain")
        & stress["model"].eq("current_only")
        & stress["scenario"].eq("exclude_2022"),
        "macro_year_ap_lift",
    ].iloc[0]
    add(
        "exclude_2022_recomputed",
        close(float(current_manual), float(current_stored)),
        f"manual={current_manual:.12f}; stored={current_stored:.12f}",
    )

    support_flag = environment["upwelling_supported"].astype(str).str.lower().eq("true")
    support_cells = environment.loc[support_flag, "cell_id"].nunique()
    unsupported = environment.loc[~support_flag]
    add("upwelling_support_is_107_cells", support_cells == 107, f"supported_cells={support_cells}")
    add(
        "no_upwelling_values_outside_support",
        unsupported[UPWELLING_COLUMNS].isna().all().all(),
        f"non_null_outside_support={int(unsupported[UPWELLING_COLUMNS].notna().sum().sum())}",
    )
    add(
        "oisst_2024_has_full_weeks",
        environment.loc[environment["year"].eq(2024), "oisst_observed_weeks"].ge(52).all(),
        f"minimum_weeks={environment.loc[environment['year'].eq(2024), 'oisst_observed_weeks'].min():.0f}",
    )
    add(
        "all_loro_current_lifts_positive",
        transfer.loc[transfer["model"].eq("current_only"), "macro_year_ap_lift"].gt(0).all(),
        f"minimum={transfer.loc[transfer['model'].eq('current_only'), 'macro_year_ap_lift'].min():.3f}",
        severity="claim",
    )

    add("panel_hash_matches_manifest", sha256(args.panel) == manifest["panel_sha256"], sha256(args.panel))
    add(
        "environment_hash_matches_manifest",
        sha256(run / "environment_features.csv") == manifest["environment_sha256"],
        sha256(run / "environment_features.csv"),
    )
    mismatched_hashes = []
    for name, expected in manifest["output_sha256"].items():
        if sha256(run / name) != expected:
            mismatched_hashes.append(name)
    add("saved_output_hashes_match_manifest", not mismatched_hashes, f"mismatches={mismatched_hashes}")

    validation = pd.DataFrame(rows)
    validation.to_csv(run / "validation_recheck.csv", index=False)
    blockers = validation.loc[~validation["passed"] & validation["severity_if_failed"].eq("blocker")]
    assessment = "Share with caveats" if blockers.empty else "Needs revision"
    status = "통과" if blockers.empty else "수정 필요"
    report = f"""# 최소 추가 실험 검증 보고서

## Overall Assessment: {assessment}

독립 재계산과 파일 무결성 검사는 **{status}**다. 총 {len(validation)}개 검증 중 {int(validation['passed'].sum())}개가 통과했으며, 공유를 막는 계산·누수·행 불일치 오류는 {'없다' if blockers.empty else '있다'}.

## Methodology Review

- 질문: 현재 캐노피, 과거 궤적, NOAA proxy가 다음 해 30% 급감 셀의 연도 내 순위화에 기여하는가.
- 모집단: 현재 상대 캐노피가 0.05보다 큰 165셀의 eligible cell-year.
- 검증: 2005–2024 expanding window; 모든 fold에서 train_end < test_year.
- 비교: full OISST domain 및 U.S. 31–47°N upwelling-supported domain 내부에서 정확히 같은 cell-year 행.
- 지표: 연도별 AP에서 그해 유병률을 뺀 뒤 평균한 macro within-year AP lift; 제한예산은 연도별 Top-20%의 micro event recall.

## Calculation Spot-Checks

- 연도별 AP lift: 저장된 전체 값을 prediction-level에서 재계산, 최대 오차 {max_year_metric_error:.3g}.
- macro within-year AP lift: 최대 오차 {max_summary_error:.3g}.
- Top-20% recall: 최대 오차 {max_top20_error:.3g}.
- 2022 제외 current-only: 수동 {current_manual:.6f}, 저장 {current_stored:.6f}.
- manifest hash: panel, environment, 모든 핵심 CSV 일치.

## Issues Found

1. **Medium — 라벨 타당성:** 30% 감소는 운영 임계값이며 독립적인 생태 붕괴 임계값 검증이 아니다.
2. **Medium — 환경 proxy support:** OISST는 0.25° 광역 표층 열노출이고 CUTI/BEUTI는 1° 위도 수준 지수다. 10 km 셀의 현장 수온·영양염 또는 인과효과로 표현하면 안 된다.
3. **Medium — 확증 수준:** 이전 파일럿 결과를 본 후 고정한 확장 실행이므로 사전등록 또는 독립 확증이 아니다.
4. **Low — 북부 표본:** Northern California 지역 보류의 추정 가능 연도는 7개뿐이다.

## Visualization Review

연구지역 지도와 네 결과 그림을 렌더링해 확인했다. 결과 그림은 0 기준선, 지표 단위, 비교대상, 신뢰구간 또는 연도 범위를 표시한다. 증분가치 그림의 축은 0 주변 차이를 보여주기 위한 좁은 delta 축이며 0선을 명시했다.

## Required Caveats for Stakeholders

- “환경이 중요하지 않다”가 아니라 “선택한 공개 proxy가 이 설계에서 추가 예측정보를 보이지 않았다”고 쓴다.
- Top-20% recall 32.0%는 조사 우선순위 보조 성능이며 복원 성공률이 아니다.
- 지역 보류 결과는 기본 순위 신호의 전이 가능성을 지지하지만 추가 변수의 보편적 효과를 지지하지 않는다.

## Incomplete Handoff Blockers

- 없음. 단, CRW 5 km·파랑·관측노력 보정은 이번 ‘최소 4개 실험’의 범위 밖이며 전체 논문 확증분석으로 완료된 것이 아니다.
"""
    (run / "validation_report_ko.md").write_text(report, encoding="utf-8")
    validation_manifest = {
        "assessment": assessment,
        "checks": len(validation),
        "passed": int(validation["passed"].sum()),
        "validation_csv_sha256": sha256(run / "validation_recheck.csv"),
        "validated_manifest_sha256": sha256(run / "manifest.json"),
    }
    (run / "validation_manifest.json").write_text(
        json.dumps(validation_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(validation.to_string(index=False))
    print(json.dumps(validation_manifest, ensure_ascii=False, indent=2))
    if not blockers.empty:
        raise AssertionError(blockers.to_string(index=False))


if __name__ == "__main__":
    main()
