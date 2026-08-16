"""Independently validate the saved Giraldo error-diagnostic case study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "outputs/experiments/20260816_giraldo_error_case_study_v1"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ap_lift(group: pd.DataFrame) -> float:
    if group.empty or group["event"].nunique() < 2:
        return np.nan
    return float(average_precision_score(group["event"], group["score"])) - float(
        group["event"].mean()
    )


def main() -> None:
    args = parse_args()
    run = args.run_dir
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    decision = json.loads((run / "decision.json").read_text(encoding="utf-8"))
    config_path = Path(manifest["inputs"]["config"])
    config = json.loads(config_path.read_text(encoding="utf-8"))
    field_path = Path(manifest["inputs"]["field"])
    panel_path = Path(manifest["inputs"]["panel"])
    prediction_path = Path(manifest["inputs"]["predictions"])
    field = pd.read_csv(field_path, low_memory=False)
    panel = pd.read_csv(panel_path)
    predictions = pd.read_csv(prediction_path)
    field_cell_years = pd.read_csv(run / "field_cell_years.csv")
    analysis = pd.read_csv(run / "analysis_dataset.csv")
    associations = pd.read_csv(run / "ecological_axis_associations.csv")
    support = pd.read_csv(run / "condition_support_summary.csv")
    performance_increments = pd.read_csv(run / "performance_increment_summary.csv")
    repeated_errors = pd.read_csv(run / "repeated_error_cells.csv")

    rows: list[dict[str, object]] = []

    def add(
        check: str, passed: bool, evidence: str, severity: str = "blocker"
    ) -> None:
        rows.append(
            {
                "check": check,
                "passed": bool(passed),
                "severity_if_failed": severity,
                "evidence": evidence,
            }
        )

    add(
        "manifest_complete",
        manifest.get("status") == "complete",
        f"status={manifest.get('status')}",
    )
    add(
        "classification_is_exploratory",
        "exploratory" in manifest.get("classification", ""),
        manifest.get("classification", ""),
    )
    add(
        "field_input_hash",
        sha256(field_path) == manifest["inputs"]["field_sha256"],
        sha256(field_path),
    )
    add(
        "panel_input_hash",
        sha256(panel_path) == manifest["inputs"]["panel_sha256"],
        sha256(panel_path),
    )
    add(
        "prediction_input_hash",
        sha256(prediction_path) == manifest["inputs"]["predictions_sha256"],
        sha256(prediction_path),
    )
    add(
        "config_input_hash",
        sha256(config_path) == manifest["inputs"]["config_sha256"],
        sha256(config_path),
    )
    output_hash_failures = [
        filename
        for filename, expected in manifest["output_sha256"].items()
        if sha256(run / filename) != expected
    ]
    add(
        "saved_output_hashes",
        not output_hash_failures,
        f"mismatches={output_hash_failures}",
    )

    add("raw_field_shape", field.shape == (9728, 144), f"shape={field.shape}")
    add(
        "raw_field_sites_and_years",
        field["site_campus_unique_ID"].nunique() == 191
        and field["survey_year"].min() == 1999
        and field["survey_year"].max() == 2021,
        f"sites={field['site_campus_unique_ID'].nunique()}; years={field['survey_year'].min()}-{field['survey_year'].max()}",
    )
    add(
        "panel_grain_and_cells",
        panel["cell_id"].nunique() == 165
        and not panel.duplicated(["cell_id", "year"]).any(),
        f"cells={panel['cell_id'].nunique()}; duplicate_keys={panel.duplicated(['cell_id', 'year']).sum()}",
    )
    add(
        "field_cell_year_keys_unique",
        not field_cell_years.duplicated(["cell_id", "year"]).any(),
        f"rows={len(field_cell_years)}; duplicate_keys={field_cell_years.duplicated(['cell_id', 'year']).sum()}",
    )
    analysis_key = ["cell_id", "year", "model_family"]
    add(
        "analysis_keys_unique",
        not analysis.duplicated(analysis_key).any(),
        f"rows={len(analysis)}; duplicate_keys={analysis.duplicated(analysis_key).sum()}",
    )
    matched_keys = analysis[["cell_id", "year"]].drop_duplicates()
    add(
        "locked_primary_population",
        len(matched_keys) == 424
        and matched_keys["cell_id"].nunique() == 52
        and analysis.loc[analysis["model_family"].eq("logistic"), "event"].sum()
        == 151,
        f"cell_years={len(matched_keys)}; cells={matched_keys['cell_id'].nunique()}; events={analysis.loc[analysis['model_family'].eq('logistic'), 'event'].sum()}",
    )
    family_counts = analysis.groupby("model_family").size().to_dict()
    add(
        "three_model_families_share_rows",
        set(family_counts) == {"logistic", "random_forest", "xgboost"}
        and set(family_counts.values()) == {424},
        str(family_counts),
    )
    outcome_agreement = (
        analysis.pivot(
            index=["cell_id", "year"], columns="model_family", values="event"
        )
        .nunique(axis=1)
        .eq(1)
        .all()
    )
    add("events_agree_across_families", outcome_agreement, "one event label per cell-year")
    add(
        "no_missing_primary_axes",
        not analysis[
            [
                "grazing_log1p",
                "rock_probability",
                "depth_m",
                "kelp_species_contrast",
            ]
        ].isna().any().any(),
        str(analysis[["grazing_log1p", "rock_probability", "depth_m", "kelp_species_contrast"]].isna().sum().to_dict()),
    )

    full_primary_predictions = predictions.loc[
        predictions["domain"].eq(config["population"]["prediction_domain"])
        & predictions["year"].between(
            config["population"]["first_forecast_year"],
            config["population"]["last_field_year"],
        )
        & predictions["model_family"].eq("logistic")
        & predictions["feature_set"].isin(
            ["current_plus_trajectory", "trajectory_plus_oisst"]
        )
    ].copy()
    primary_predictions = full_primary_predictions.merge(
        matched_keys, on=["cell_id", "year"], how="inner"
    )
    annual_rows = []
    for (feature_set, year), group in primary_predictions.groupby(
        ["feature_set", "year"], sort=True
    ):
        annual_rows.append(
            {
                "feature_set": feature_set,
                "year": year,
                "ap_lift": safe_ap_lift(group),
            }
        )
    annual = pd.DataFrame(annual_rows).pivot(
        index="year", columns="feature_set", values="ap_lift"
    )
    paired_increment = (
        annual["trajectory_plus_oisst"]
        - annual["current_plus_trajectory"]
    ).dropna()
    stored_increment = performance_increments.loc[
        performance_increments["scope"].eq("giraldo_matched")
        & performance_increments["model_family"].eq("logistic")
        & performance_increments["comparison"].eq("oisst_minus_trajectory")
    ].iloc[0]
    add(
        "logistic_matched_oisst_increment_recomputed",
        np.isclose(
            paired_increment.mean(), stored_increment["estimate"], atol=1e-12, rtol=0
        ),
        f"recomputed={paired_increment.mean():.15f}; stored={stored_increment['estimate']:.15f}; years={len(paired_increment)}",
    )

    logistic_analysis = analysis.loc[analysis["model_family"].eq("logistic")].copy()
    regenerated = full_primary_predictions.copy()
    regenerated["risk_percentile"] = regenerated.groupby(
        ["feature_set", "year"]
    )["score"].rank(method="average", pct=True)
    regenerated = regenerated.merge(
        matched_keys, on=["cell_id", "year"], how="inner"
    )
    rank_wide = regenerated.pivot(
        index=["cell_id", "year", "event"],
        columns="feature_set",
        values=["score", "risk_percentile"],
    ).reset_index()
    rank_wide.columns = [
        "_".join(str(value) for value in column if str(value))
        if isinstance(column, tuple)
        else str(column)
        for column in rank_wide.columns
    ]
    signed = 2 * rank_wide["event"] - 1
    rank_wide["rank_gain_recomputed"] = signed * (
        rank_wide["risk_percentile_trajectory_plus_oisst"]
        - rank_wide["risk_percentile_current_plus_trajectory"]
    ) * 100
    rank_wide["brier_gain_recomputed"] = (
        rank_wide["event"] - rank_wide["score_current_plus_trajectory"]
    ) ** 2 - (
        rank_wide["event"] - rank_wide["score_trajectory_plus_oisst"]
    ) ** 2
    diagnostic_compare = logistic_analysis.merge(
        rank_wide[
            [
                "cell_id",
                "year",
                "rank_gain_recomputed",
                "brier_gain_recomputed",
            ]
        ],
        on=["cell_id", "year"],
        validate="one_to_one",
    )
    max_rank_error = (
        diagnostic_compare["oisst_signed_rank_gain_pp"]
        - diagnostic_compare["rank_gain_recomputed"]
    ).abs().max()
    max_brier_error = (
        diagnostic_compare["oisst_brier_gain"]
        - diagnostic_compare["brier_gain_recomputed"]
    ).abs().max()
    add(
        "row_level_rank_gain_recomputed",
        max_rank_error < 1e-12,
        f"max_abs_error={max_rank_error:.3g}",
    )
    add(
        "row_level_brier_gain_recomputed",
        max_brier_error < 1e-12,
        f"max_abs_error={max_brier_error:.3g}",
    )

    add(
        "association_table_complete",
        len(associations) == 180
        and associations["scope"].nunique() == 3
        and associations["model_family"].nunique() == 3
        and associations["outcome"].nunique() == 5
        and associations["axis"].nunique() == 4,
        f"rows={len(associations)}; scopes={associations['scope'].nunique()}; families={associations['model_family'].nunique()}; outcomes={associations['outcome'].nunique()}; axes={associations['axis'].nunique()}",
    )
    add(
        "association_intervals_and_q_values_valid",
        associations["ci_low"].le(associations["estimate_per_sd"]).all()
        and associations["ci_high"].ge(associations["estimate_per_sd"]).all()
        and associations["q_value"].between(0, 1).all(),
        f"invalid_intervals={int((associations['ci_low'].gt(associations['estimate_per_sd']) | associations['ci_high'].lt(associations['estimate_per_sd'])).sum())}; invalid_q={int((~associations['q_value'].between(0, 1)).sum())}",
    )
    add(
        "condition_support_table_complete",
        len(support) == 20,
        f"rows={len(support)}",
    )
    add(
        "no_stable_condition_matches_saved_decision",
        not support["stable_condition_supported"].astype(bool).any()
        and decision["stable_condition_count"] == 0
        and decision["decision"] == "no_stable_ecological_failure_condition_detected",
        f"stable_rows={support['stable_condition_supported'].astype(bool).sum()}; decision={decision['decision']}",
        severity="claim",
    )
    add(
        "repeated_error_table_one_row_per_cell",
        len(repeated_errors) == 52 and repeated_errors["cell_id"].is_unique,
        f"rows={len(repeated_errors)}; unique_cells={repeated_errors['cell_id'].nunique()}",
    )

    figures = sorted(run.glob("figure_*.png"))
    figure_failures = []
    for figure in figures:
        image = mpimg.imread(figure)
        if image.ndim < 2 or min(image.shape[:2]) < 500:
            figure_failures.append(f"{figure.name}:{image.shape}")
    add(
        "four_figures_rendered_at_usable_size",
        len(figures) == 4 and not figure_failures,
        f"figures={len(figures)}; failures={figure_failures}",
    )

    validation = pd.DataFrame(rows)
    validation.to_csv(run / "validation_recheck.csv", index=False)
    blockers = validation.loc[
        ~validation["passed"] & validation["severity_if_failed"].eq("blocker")
    ]
    assessment = "Share with caveats" if blockers.empty else "Needs revision"
    report = f"""# Giraldo 보조 사례연구 독립 검증 보고서

## Overall Assessment: {assessment}

총 {len(validation)}개 검증 중 {int(validation['passed'].sum())}개가 통과했다. 계산·키·입력 hash·산출물 hash와 그림 렌더링에 관한 공유 차단 오류는 {'없다' if blockers.empty else '있다'}.

## Methodology Review

- 보조 사례연구는 주 165셀 모형을 재적합하지 않고 저장된 expanding-window out-of-fold 예측을 사용했다.
- 분석대상은 Giraldo 자료가 같은 셀·연도에 존재하는 California 52셀·424 cell-year로 제한됐다.
- 네 생태축은 분석 전에 고정됐고 144개 열 전수 탐색을 하지 않았다.
- 계수는 지역·연도 고정효과와 셀·연도 two-way cluster 표준오차를 사용했다.
- 이 분석은 pilot-informed exploratory diagnostic이며 독립 확증이 아니다.

## Calculation Spot-Checks

- 원자료 9,728행×144열, 191현장, 1999–2021년을 재확인했다.
- 52셀·424 cell-year·151사건 및 세 모형군의 동일 평가행을 재확인했다.
- Logistic matched 표본의 OISST AP-lift 증분을 원 예측확률에서 재계산했다: {paired_increment.mean():+.6f}.
- 행별 OISST signed rank gain 최대 오차: {max_rank_error:.3g}.
- 행별 OISST Brier gain 최대 오차: {max_brier_error:.3g}.
- 실행 manifest의 입력과 원 산출물 SHA-256을 재확인했다.

## Issues Found

1. **High — 외적 타당성:** 현장자료는 California에만 존재하고 Oregon·Washington·Baja는 포함하지 않는다.
2. **High — 공간 support:** 60㎡ transect와 10 km 의사결정 셀은 같은 생태적 평균을 측정하지 않는다.
3. **Medium — 표본선택:** 현장조사는 암반초와 기존 모니터링 지점에 편중되어 52셀 표본이 165셀의 무작위 부분표본이 아니다.
4. **Medium — 시간 복제:** 424행이 독립표본 424개가 아니며 52셀과 17개 예측연도의 반복관측이다.
5. **Medium — 음성결과 해석:** 안정적인 조건을 찾지 못한 것은 생태축의 무관성을 증명하지 않는다.

## Required Caveats for Paper

- “현장 생태조건에서도 안정적인 조건부 가치가 확인되지 않았다”고 쓸 수 있다.
- “성게·서식처가 급감에 중요하지 않다” 또는 “현장자료는 가치가 없다”고 쓰면 안 된다.
- Giraldo 자료는 B5 주 정보블록이 아니라 California 보조 오류진단 자료로 표시해야 한다.
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
