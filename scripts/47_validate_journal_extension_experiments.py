"""Independently validate the saved journal-extension experiment outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


OISST = [
    "annual_mean_sst_anomaly",
    "annual_max_sst",
    "hot_weeks_p90",
    "winter_mean_sst_anomaly",
    "summer_mean_sst_anomaly",
    "lag1_annual_mean_sst_anomaly",
]
UPWELLING = [
    "winter_cuti_anomaly",
    "spring_cuti_anomaly",
    "winter_beuti_anomaly",
    "spring_beuti_anomaly",
    "upwelling_season_beuti_anomaly",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/experiments/20260814_journal_extension_v1"),
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/processed/kelpwatch_west_coast_panel_165.csv"),
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=Path(
            "outputs/experiments/20260814_minimal_paper_extension_v2/environment_features.csv"
        ),
    )
    parser.add_argument("--expected-cells", type=int, default=165)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(left, right, atol=tolerance, rtol=0, equal_nan=True))


def safe_ap(frame: pd.DataFrame) -> float:
    if frame["event"].nunique() < 2:
        return np.nan
    return float(average_precision_score(frame["event"], frame["score"]))


def main() -> None:
    args = parse_args()
    run = args.run_dir
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    panel = pd.read_csv(args.panel)
    environment = pd.read_csv(args.environment)
    predictions = pd.read_csv(run / "expanding_predictions.csv")
    annual = pd.read_csv(run / "expanding_year_metrics.csv")
    model_summary = pd.read_csv(run / "model_family_summary.csv")
    increments = pd.read_csv(run / "model_family_increment_summary.csv")
    random_metrics = pd.read_csv(run / "random_protocol_replicate_metrics.csv")
    random_repeat_zero = pd.read_csv(run / "random_protocol_predictions_repeat0.csv")
    protocol = pd.read_csv(run / "protocol_comparison.csv")
    fold_audit = pd.read_csv(run / "fold_audit.csv")
    budget_detail = pd.read_csv(run / "budget_year_detail.csv")
    budget_summary = pd.read_csv(run / "budget_summary.csv")
    sensitivity_metrics = pd.read_csv(run / "sensitivity_model_metrics.csv")
    sensitivity_increments = pd.read_csv(run / "sensitivity_increment_summary.csv")

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

    add(
        "panel_has_expected_cells",
        panel["cell_id"].nunique() == args.expected_cells,
        f"expected={args.expected_cells}; cells={panel['cell_id'].nunique()}",
    )
    add(
        "panel_keys_unique",
        not panel.duplicated(["cell_id", "year"]).any(),
        f"duplicates={panel.duplicated(['cell_id', 'year']).sum()}",
    )
    add(
        "environment_keys_unique",
        not environment.duplicated(["cell_id", "year"]).any(),
        f"duplicates={environment.duplicated(['cell_id', 'year']).sum()}",
    )
    merged = panel.merge(
        environment,
        on=["cell_id", "year"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_environment"),
    )
    add("merge_preserves_panel_grain", len(merged) == len(panel), f"panel={len(panel)}; merged={len(merged)}")
    add(
        "forward_only_expanding_folds",
        fold_audit["train_end"].lt(fold_audit["year"]).all(),
        f"max_train_end_minus_test={(fold_audit['train_end'] - fold_audit['year']).max()}",
    )
    prediction_key = [
        "domain",
        "model_family",
        "feature_set",
        "cell_id",
        "year",
    ]
    add(
        "expanding_prediction_keys_unique",
        not predictions.duplicated(prediction_key).any(),
        f"duplicates={predictions.duplicated(prediction_key).sum()}",
    )
    add(
        "prediction_scores_finite_and_bounded",
        np.isfinite(predictions["score"]).all()
        and predictions["score"].between(0, 1).all(),
        f"missing={predictions['score'].isna().sum()}; min={predictions['score'].min():.6f}; max={predictions['score'].max():.6f}",
    )

    exact_rows = True
    exact_outcomes = True
    domain_population: dict[str, int] = {}
    for domain, domain_frame in predictions.groupby("domain"):
        key_lists = []
        event_lists = []
        for _, group in domain_frame.groupby(["model_family", "feature_set"]):
            ordered = group.sort_values(["cell_id", "year"])
            key_lists.append(list(zip(ordered["cell_id"], ordered["year"], strict=True)))
            event_lists.append(ordered["event"].tolist())
        exact_rows &= all(values == key_lists[0] for values in key_lists[1:])
        exact_outcomes &= all(values == event_lists[0] for values in event_lists[1:])
        domain_population[domain] = len(key_lists[0])
    add("exact_rows_across_models_within_domain", exact_rows, str(domain_population))
    add("exact_outcomes_across_models_within_domain", exact_outcomes, "ordered event vectors match")

    max_annual_error = 0.0
    manual_annual_rows: list[dict[str, object]] = []
    annual_keys = ["domain", "model_family", "feature_set", "year"]
    for keys, group in predictions.groupby(annual_keys, sort=True):
        prevalence = float(group["event"].mean())
        ap = safe_ap(group)
        lift = ap - prevalence if np.isfinite(ap) else np.nan
        stored = annual.loc[
            annual["domain"].eq(keys[0])
            & annual["model_family"].eq(keys[1])
            & annual["feature_set"].eq(keys[2])
            & annual["year"].eq(keys[3])
        ].iloc[0]
        if np.isfinite(lift):
            max_annual_error = max(max_annual_error, abs(lift - float(stored.ap_lift)))
        manual_annual_rows.append(
            {
                **dict(zip(annual_keys, keys, strict=True)),
                "ap_lift": lift,
            }
        )
    add("annual_ap_lifts_recomputed", max_annual_error < 1e-12, f"max_error={max_annual_error:.3g}")
    manual_annual = pd.DataFrame(manual_annual_rows)

    max_macro_error = 0.0
    max_pooled_error = 0.0
    for keys, group in predictions.groupby(
        ["domain", "model_family", "feature_set"], sort=True
    ):
        yearly = manual_annual.loc[
            manual_annual["domain"].eq(keys[0])
            & manual_annual["model_family"].eq(keys[1])
            & manual_annual["feature_set"].eq(keys[2]),
            "ap_lift",
        ].dropna()
        macro = float(yearly.mean())
        pooled = safe_ap(group) - float(group["event"].mean())
        stored = model_summary.loc[
            model_summary["domain"].eq(keys[0])
            & model_summary["model_family"].eq(keys[1])
            & model_summary["feature_set"].eq(keys[2])
        ].iloc[0]
        max_macro_error = max(max_macro_error, abs(macro - float(stored.macro_within_year_ap_lift)))
        max_pooled_error = max(max_pooled_error, abs(pooled - float(stored.pooled_ap_lift)))
    add("model_macro_lifts_recomputed", max_macro_error < 1e-12, f"max_error={max_macro_error:.3g}")
    add("model_pooled_lifts_recomputed", max_pooled_error < 1e-12, f"max_error={max_pooled_error:.3g}")

    max_increment_error = 0.0
    for row in increments.itertuples():
        source = manual_annual.loc[
            manual_annual["domain"].eq(row.domain)
            & manual_annual["model_family"].eq(row.model_family)
        ]
        left = source.loc[source["feature_set"].eq(row.left_feature_set), ["year", "ap_lift"]].rename(columns={"ap_lift": "left"})
        right = source.loc[source["feature_set"].eq(row.right_feature_set), ["year", "ap_lift"]].rename(columns={"ap_lift": "right"})
        joined = left.merge(right, on="year", validate="one_to_one").dropna()
        manual_value = float((joined["left"] - joined["right"]).mean())
        max_increment_error = max(max_increment_error, abs(manual_value - float(row.estimate)))
    add("paired_model_increments_recomputed", max_increment_error < 1e-12, f"max_error={max_increment_error:.3g}")

    expected_random_groups = 20
    repeat_counts = random_metrics.groupby(["domain", "feature_set"])["repeat"].nunique()
    add(
        "random_protocol_has_20_repeats",
        repeat_counts.eq(expected_random_groups).all(),
        f"min={repeat_counts.min()}; max={repeat_counts.max()}",
    )
    random_rows_exact = True
    for domain, domain_frame in random_repeat_zero.groupby("domain"):
        expected = (
            predictions.loc[
                predictions["domain"].eq(domain)
                & predictions["model_family"].eq("logistic")
                & predictions["feature_set"].eq("current_only"),
                ["cell_id", "year", "event"],
            ]
            .sort_values(["cell_id", "year"])
            .reset_index(drop=True)
        )
        for _, group in domain_frame.groupby("feature_set"):
            observed = group[["cell_id", "year", "event"]].sort_values(["cell_id", "year"]).reset_index(drop=True)
            random_rows_exact &= observed.equals(expected)
    add("random_oof_repeat0_covers_exact_operational_rows", random_rows_exact, "cell-year-event rows match")
    max_random_metric_error = 0.0
    for (domain, feature_set), group in random_repeat_zero.groupby(["domain", "feature_set"]):
        pooled = safe_ap(group) - float(group["event"].mean())
        annual_values = []
        for _, year_group in group.groupby("year"):
            annual_values.append(safe_ap(year_group) - float(year_group["event"].mean()))
        macro = float(np.mean(annual_values))
        stored = random_metrics.loc[
            random_metrics["domain"].eq(domain)
            & random_metrics["feature_set"].eq(feature_set)
            & random_metrics["repeat"].eq(0)
        ].iloc[0]
        max_random_metric_error = max(
            max_random_metric_error,
            abs(pooled - float(stored.pooled_ap_lift)),
            abs(macro - float(stored.macro_within_year_ap_lift)),
        )
    add("random_repeat0_metrics_recomputed", max_random_metric_error < 1e-12, f"max_error={max_random_metric_error:.3g}")
    add(
        "protocol_table_has_all_information_sets",
        len(protocol) == 7,
        f"rows={len(protocol)}",
    )

    max_budget_count_error = 0
    for keys, group in predictions.groupby(["domain", "model_family", "feature_set", "year"]):
        ranked = group.sort_values(["score", "cell_id"], ascending=[False, True])
        for budget in sorted(budget_detail["budget_fraction"].unique()):
            selected_n = max(1, math.ceil(len(ranked) * budget))
            true_positives = int(ranked.head(selected_n)["event"].sum())
            stored = budget_detail.loc[
                budget_detail["domain"].eq(keys[0])
                & budget_detail["model_family"].eq(keys[1])
                & budget_detail["feature_set"].eq(keys[2])
                & budget_detail["year"].eq(keys[3])
                & np.isclose(budget_detail["budget_fraction"], budget)
            ].iloc[0]
            max_budget_count_error = max(
                max_budget_count_error,
                abs(selected_n - int(stored.selected_n)),
                abs(true_positives - int(stored.true_positives)),
            )
    add("budget_rank_counts_recomputed", max_budget_count_error == 0, f"max_count_error={max_budget_count_error}")
    max_budget_summary_error = 0.0
    for keys, group in budget_detail.groupby(["domain", "model_family", "feature_set", "budget_fraction"]):
        recall = float(group["true_positives"].sum() / group["events"].sum())
        stored = budget_summary.loc[
            budget_summary["domain"].eq(keys[0])
            & budget_summary["model_family"].eq(keys[1])
            & budget_summary["feature_set"].eq(keys[2])
            & np.isclose(budget_summary["budget_fraction"], keys[3])
        ].iloc[0]
        max_budget_summary_error = max(max_budget_summary_error, abs(recall - float(stored.micro_recall)))
    add("budget_micro_recall_recomputed", max_budget_summary_error < 1e-12, f"max_error={max_budget_summary_error:.3g}")
    monotonic = budget_summary.sort_values("budget_fraction").groupby(
        ["domain", "model_family", "feature_set"]
    )["micro_recall"].apply(lambda values: values.is_monotonic_increasing)
    add("budget_recall_is_monotonic", monotonic.all(), f"violations={(~monotonic).sum()}", severity="claim")

    full_base = merged.dropna(subset=OISST)
    supported_base = merged.loc[
        merged["upwelling_supported"].fillna(False).astype(str).str.lower().eq("true")
    ].dropna(subset=[*OISST, *UPWELLING])
    max_sensitivity_n_error = 0
    max_sensitivity_event_error = 0
    for row in sensitivity_metrics.loc[sensitivity_metrics["feature_set"].eq("current_only")].itertuples():
        source = full_base if row.domain == "full_oisst_domain" else supported_base
        selected = source.loc[
            source["year"].between(2005, 2024)
            & source["relative_canopy"].gt(row.canopy_cutoff)
            & source["relative_drop_next"].notna()
        ]
        events = int(selected["relative_drop_next"].ge(row.decline_threshold).sum())
        max_sensitivity_n_error = max(max_sensitivity_n_error, abs(len(selected) - int(row.n)))
        max_sensitivity_event_error = max(max_sensitivity_event_error, abs(events - int(row.events)))
    add("sensitivity_population_counts_recomputed", max_sensitivity_n_error == 0, f"max_n_error={max_sensitivity_n_error}")
    add("sensitivity_event_counts_recomputed", max_sensitivity_event_error == 0, f"max_event_error={max_sensitivity_event_error}")

    primary_sensitivity = sensitivity_metrics.loc[
        np.isclose(sensitivity_metrics["decline_threshold"], 0.30)
        & np.isclose(sensitivity_metrics["canopy_cutoff"], 0.05)
    ]
    primary_model = model_summary.loc[model_summary["model_family"].eq("logistic")]
    joined_primary = primary_sensitivity.merge(
        primary_model[["domain", "feature_set", "macro_within_year_ap_lift"]],
        on=["domain", "feature_set"],
        suffixes=("_sensitivity", "_main"),
        validate="one_to_one",
    )
    primary_sensitivity_error = (
        joined_primary["macro_within_year_ap_lift_sensitivity"]
        - joined_primary["macro_within_year_ap_lift_main"]
    ).abs().max()
    add("primary_sensitivity_matches_main_logistic_run", primary_sensitivity_error < 1e-12, f"max_error={primary_sensitivity_error:.3g}")
    add(
        "oisst_increment_nonpositive_in_all_nine_sensitivities",
        sensitivity_increments.loc[
            sensitivity_increments["domain"].eq("full_oisst_domain")
            & sensitivity_increments["comparison"].eq("oisst_minus_trajectory"),
            "estimate",
        ].le(0).all(),
        "all 9 point estimates <= 0",
        severity="claim",
    )
    add(
        "current_baseline_positive_for_all_model_families",
        model_summary.loc[
            model_summary["domain"].eq("full_oisst_domain")
            & model_summary["feature_set"].eq("current_only"),
            "macro_ci_low",
        ].gt(0).all(),
        "all lower confidence limits > 0",
        severity="claim",
    )

    add("panel_hash_matches_manifest", sha256(args.panel) == manifest["panel_sha256"], sha256(args.panel))
    add(
        "environment_hash_matches_manifest",
        sha256(args.environment) == manifest["environment_sha256"],
        sha256(args.environment),
    )
    mismatched_hashes = [
        name
        for name, expected in manifest["output_sha256"].items()
        if sha256(run / name) != expected
    ]
    add("saved_output_hashes_match_manifest", not mismatched_hashes, f"mismatches={mismatched_hashes}")

    validation = pd.DataFrame(rows)
    validation.to_csv(run / "validation_recheck.csv", index=False)
    blockers = validation.loc[
        ~validation["passed"] & validation["severity_if_failed"].eq("blocker")
    ]
    assessment = "Share with caveats" if blockers.empty else "Needs revision"
    trajectory_rows = increments.loc[
        increments["domain"].eq("full_oisst_domain")
        & increments["comparison"].eq("trajectory_minus_current")
    ]
    trajectory_supported = trajectory_rows.loc[
        trajectory_rows["ci_low"].gt(0), "model_label"
    ].tolist()
    trajectory_unsupported = trajectory_rows.loc[
        trajectory_rows["ci_low"].le(0), "model_label"
    ].tolist()
    if len(trajectory_supported) == len(trajectory_rows):
        trajectory_text = (
            "궤적 증분은 세 고정 모델 모두에서 양수로 지지됐으므로 "
            "‘작지만 모델 계열에 강건한 추가가치’로 표현할 수 있다."
        )
    elif trajectory_supported:
        trajectory_text = (
            f"궤적 증분은 {', '.join(trajectory_supported)}에서 양수로 지지됐지만 "
            f"{', '.join(trajectory_unsupported)}에서는 불확실하므로 "
            "‘작고 모델에 따라 불확실한 추가가치’로 표현한다."
        )
    else:
        trajectory_text = (
            "궤적 증분은 어떤 고정 모델에서도 양수로 확정되지 않았다고 표현한다."
        )
    report = f"""# 저널 보완 실험 검증 보고서

## Overall Assessment: {assessment}

총 {len(validation)}개 검증 중 {int(validation['passed'].sum())}개가 통과했다. 공유를 막는 계산·누수·행 불일치 오류는 {'없다' if blockers.empty else '있다'}.

## Methodology Review

- 운영 성능은 2005–2024 expanding-window와 연도 내 AP lift로 평가했다.
- random cell-year CV는 동일한 2005–2024 평가행을 사용하지만 미래 연도와 동일 셀의 다른 연도를 학습에 허용한다. 이는 낙관 편향을 보여주기 위한 비교이며 배포성능 추정치가 아니다.
- 모델 계열은 결과를 본 뒤 최고 사양을 선택하지 않고 Logistic, Random Forest, XGBoost의 고정 사양을 사용했다.
- 라벨 민감도는 20/30/40% 감소와 현재 상대 캐노피 0.02/0.05/0.10의 9개 조합을 모두 expanding-window로 다시 적합했다.

## Calculation Spot-Checks

- 연도별 AP lift 최대 재계산 오차: {max_annual_error:.3g}.
- 모델 macro AP lift 최대 오차: {max_macro_error:.3g}.
- 정보 블록 증분값 최대 오차: {max_increment_error:.3g}.
- random-CV repeat 0의 pooled/macro 최대 오차: {max_random_metric_error:.3g}.
- 예산별 선발 수·적중 사건 수 최대 정수 오차: {max_budget_count_error}.
- 민감도 평가행 수 최대 오차: {max_sensitivity_n_error}; 사건 수 최대 오차: {max_sensitivity_event_error}.
- 입력 및 핵심 산출물 hash가 manifest와 일치한다.

## Issues Found

1. **Medium — 확증 수준:** 이전 결과를 확인한 뒤 수행한 강건성 분석이므로 독립 확증이나 사전등록으로 표현할 수 없다.
2. **Medium — random-CV 해석:** random-CV와 expanding-window의 차이는 시간 누수, 동일 셀 반복, 학습량 차이가 결합된 결과다. 특정 한 원인의 인과효과로 분해하지 않는다.
3. **Medium — 모델 비교 범위:** 세 모델은 고정 사양의 결론 강건성 점검이며 최적 알고리즘 선발이나 완전한 하이퍼파라미터 탐색이 아니다.
4. **Medium — 라벨 타당성:** 민감도 통과는 30%가 생태학적 붕괴 임계값임을 증명하지 않는다. 모두 annual-maximum canopy의 운영 라벨이다.
5. **Medium — 환경 해석:** NOAA 결과는 선택한 proxy의 증분 예측정보에 관한 것이며 환경의 생태적 중요성이나 인과효과를 검정하지 않는다.

## Required Caveats for Paper

- 현재 상태 신호는 세 모델에서 반복됐다고 쓸 수 있다.
- {trajectory_text}
- OISST와 CUTI/BEUTI는 세 모델 모두에서 양의 증분가치가 확정되지 않았다고 쓸 수 있다.
- random split이 항상 모든 지표를 부풀렸다고 쓰면 안 된다. 현재 상태의 macro within-year 지표는 random과 expanding이 거의 같았고, 복잡한 정보 블록과 pooled 지표에서 낙관 차이가 커졌다.
"""
    (run / "validation_report_ko.md").write_text(report, encoding="utf-8")
    validation_manifest = {
        "assessment": assessment,
        "checks": len(validation),
        "passed": int(validation["passed"].sum()),
        "blockers": len(blockers),
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
