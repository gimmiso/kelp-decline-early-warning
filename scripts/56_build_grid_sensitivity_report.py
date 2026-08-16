"""Build the canonical artifact for the grid-size/origin sensitivity report."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


ORIGINS = {
    "o00": "기본 원점",
    "ox50": "동쪽 반 셀",
    "oy50": "북쪽 반 셀",
    "oxy50": "동·북 반 셀",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def rounded(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def main() -> None:
    args = parse_args()
    output = args.output_dir
    results = pd.read_csv(output / "grid_sensitivity_results.csv")
    validation = pd.read_csv(output / "validation_recheck.csv")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    generated_at = datetime.fromtimestamp(
        manifest["finished_at_utc_epoch"], tz=UTC
    ).isoformat().replace("+00:00", "Z")

    scaled_sql = """SELECT * FROM grid_sensitivity_results
WHERE threshold_policy = 'area_scaled'
ORDER BY grid_size_km,
CASE origin_name WHEN 'o00' THEN 1 WHEN 'ox50' THEN 2 WHEN 'oy50' THEN 3 ELSE 4 END"""
    policy_sql = """SELECT * FROM grid_sensitivity_results
WHERE grid_size_km IN (5, 20)
ORDER BY grid_size_km, threshold_policy,
CASE origin_name WHEN 'o00' THEN 1 WHEN 'ox50' THEN 2 WHEN 'oy50' THEN 3 ELSE 4 END"""
    headline_sql = """WITH scaled AS (
  SELECT * FROM grid_sensitivity_results WHERE threshold_policy = 'area_scaled'
), origin_ranges AS (
  SELECT grid_size_km, MAX(current_lift) - MIN(current_lift) AS origin_range
  FROM scaled GROUP BY grid_size_km
)
SELECT
  SUM(CASE WHEN current_ci_low > 0 THEN 1 ELSE 0 END) AS current_supported,
  (SELECT MAX(origin_range) FROM origin_ranges) AS max_origin_range,
  MIN(CASE WHEN spec_id <> 'g10_area_scaled_o00' THEN global_label_jaccard END) AS min_label,
  MIN(CASE WHEN spec_id <> 'g10_area_scaled_o00' THEN global_top20_spatial_jaccard END) AS min_top20,
  SUM(CASE WHEN trajectory_ci_low > 0 THEN 1 ELSE 0 END) AS trajectory_supported,
  SUM(CASE WHEN oisst_ci_low > 0 THEN 1 ELSE 0 END) AS oisst_supported
FROM scaled"""
    validation_sql = 'SELECT "check", value, passed FROM validation_recheck ORDER BY "check"'
    connection = sqlite3.connect(":memory:")
    results.to_sql("grid_sensitivity_results", connection, index=False)
    validation.to_sql("validation_recheck", connection, index=False)
    scaled = pd.read_sql_query(scaled_sql, connection)
    policy = pd.read_sql_query(policy_sql, connection)
    headline = pd.read_sql_query(headline_sql, connection).iloc[0]
    pd.read_sql_query(validation_sql, connection)
    connection.close()

    scaled["grid_size"] = scaled["grid_size_km"].astype(str) + " km"
    scaled["origin"] = scaled["origin_name"].map(ORIGINS)
    scaled["spec_label"] = scaled["grid_size"] + " · " + scaled["origin"]
    scaled_rows = [
        {
            "spec_label": row.spec_label,
            "grid_size": row.grid_size,
            "origin": row.origin,
            "stable_cells": int(row.stable_cells),
            "current_lift": rounded(row.current_lift),
            "current_ci_low": rounded(row.current_ci_low),
            "current_ci_high": rounded(row.current_ci_high),
            "trajectory_increment": rounded(row.trajectory_increment),
            "trajectory_ci_low": rounded(row.trajectory_ci_low),
            "trajectory_ci_high": rounded(row.trajectory_ci_high),
            "oisst_increment": rounded(row.oisst_increment),
            "oisst_ci_low": rounded(row.oisst_ci_low),
            "oisst_ci_high": rounded(row.oisst_ci_high),
            "label_jaccard": rounded(row.global_label_jaccard),
            "top20_jaccard": rounded(row.global_top20_spatial_jaccard),
            "test_events": int(row.test_events),
            "test_rows": int(row.eligible_test_rows),
        }
        for row in scaled.itertuples(index=False)
    ]

    policy["grid_size"] = policy["grid_size_km"].astype(str) + " km"
    policy["origin"] = policy["origin_name"].map(ORIGINS)
    policy["threshold"] = policy["threshold_policy"].map(
        {"area_scaled": "면적비례", "fixed_500": "500픽셀 고정"}
    )
    policy["spec_label"] = policy["grid_size"] + " · " + policy["origin"]
    policy_rows = [
        {
            "spec_label": row.spec_label,
            "grid_size": row.grid_size,
            "origin": row.origin,
            "threshold": row.threshold,
            "minimum_pixels": int(row.minimum_footprint_pixels),
            "stable_cells": int(row.stable_cells),
            "current_lift": rounded(row.current_lift),
        }
        for row in policy.itertuples(index=False)
    ]

    primary = scaled.loc[scaled["spec_id"].eq("g10_area_scaled_o00")].iloc[0]
    max_origin_range = float(headline.max_origin_range)
    min_label = float(headline.min_label)
    min_top20 = float(headline.min_top20)
    trajectory_supported = int(headline.trajectory_supported)
    oisst_supported = int(headline.oisst_supported)
    validation_passed = int(validation["passed"].astype(bool).sum())

    results_path = "outputs/experiments/20260816_grid_origin_sensitivity_v1/grid_sensitivity_results.csv"
    agreement_path = "outputs/experiments/20260816_grid_origin_sensitivity_v1/pixel_spatial_agreement.csv"
    validation_path = "outputs/experiments/20260816_grid_origin_sensitivity_v1/validation_recheck.csv"
    common_filters = [
        "공식 Kelpwatch 2026 Q2 NetCDF",
        "1984–2004 양성 픽셀 고정 footprint",
        "관측 유효면적 ≥50%, 유효 분기 ≥3, Q3 필수",
        "2005–2024 expanding-window 평가",
        "다음 해 연간 최대 캐노피 30% 이상 감소",
    ]
    sources = [
        {
            "id": "grid_results",
            "label": "격자 민감도 결과표",
            "path": results_path,
            "query": {
                "language": "python",
                "description": "20개 격자 사양의 미래예측 성능, 부트스트랩 구간, 코호트 및 공간 일치도를 결합한 검증 결과표.",
                "tables_used": [results_path],
                "filters": common_filters,
                "metric_definitions": {
                    "current_lift": "연도별 AP에서 해당 연도 사건률을 뺀 값의 20개 연도 macro 평균.",
                    "trajectory_increment": "동일 표본에서 current+trajectory AP lift와 current-only AP lift의 연도별 차이 평균.",
                    "oisst_increment": "OISST 지원 동일 표본에서 trajectory+OISST와 current+trajectory AP lift의 연도별 차이 평균.",
                },
            },
        },
        {
            "id": "spatial_agreement",
            "label": "픽셀 가중 공간 일치도",
            "path": agreement_path,
            "query": {
                "language": "python",
                "description": "공통 pre-2005 양성 30m 픽셀-연도에서 급감 라벨과 상위 20% 조사영역의 Jaccard를 계산.",
                "tables_used": [agreement_path],
                "filters": common_filters,
                "metric_definitions": {
                    "label_jaccard": "두 격자 모두 급감으로 분류한 픽셀-연도 / 적어도 한 격자가 급감으로 분류한 픽셀-연도.",
                    "top20_jaccard": "두 격자의 상위 20% 조사 셀에 동시에 포함된 공통 픽셀-연도 / 어느 한쪽에 포함된 공통 픽셀-연도.",
                },
            },
        },
        {
            "id": "validation",
            "label": "독립 재검산 결과",
            "path": validation_path,
            "query": {
                "language": "python",
                "description": "예측확률·bootstrap draw·픽셀 혼동행렬·manifest hash에서 결과를 독립 재검산한 25개 검사.",
                "tables_used": [validation_path],
                "filters": ["25개 사전 정의 정합성 검사"],
            },
        },
    ]
    source_sql = {
        "grid_results": (scaled_sql, ["grid_sensitivity_results"]),
        "spatial_agreement": (scaled_sql, ["grid_sensitivity_results"]),
        "validation": (validation_sql, ["validation_recheck"]),
    }
    for source in sources:
        sql, tables_used = source_sql[source["id"]]
        source["query"].update(
            {
                "language": "sql",
                "engine": "sqlite",
                "sql": sql,
                "tables_used": tables_used,
            }
        )
    sources[1]["path"] = results_path
    sources.extend(
        [
            {
                "id": "headline_metrics",
                "label": "격자 강건성 핵심 지표",
                "path": results_path,
                "query": {
                    "language": "sql",
                    "engine": "sqlite",
                    "sql": headline_sql,
                    "description": "면적비례 12개 격자의 지지 횟수, 원점 범위 및 최소 공간 Jaccard를 집계.",
                    "tables_used": ["grid_sensitivity_results"],
                    "filters": common_filters,
                },
            },
            {
                "id": "policy_results",
                "label": "footprint 기준 비교",
                "path": results_path,
                "query": {
                    "language": "sql",
                    "engine": "sqlite",
                    "sql": policy_sql,
                    "description": "5km와 20km의 면적비례 및 500픽셀 고정 코호트를 선택.",
                    "tables_used": ["grid_sensitivity_results"],
                    "filters": [*common_filters, "grid_size_km IN (5, 20)"],
                },
            },
        ]
    )

    cards = [
        {
            "id": "current_support_card",
            "description": "현재상태 AP lift의 95% CI 하한이 0보다 큰 면적비례 격자 수.",
            "dataset": "headline",
            "sourceId": "headline_metrics",
            "metrics": [{"label": "현재상태 지지", "field": "current_supported", "format": "number", "unit": "/ 12 격자"}],
        },
        {
            "id": "origin_range_card",
            "description": "동일 크기에서 네 원점 간 현재상태 AP lift 최대 범위.",
            "dataset": "headline",
            "sourceId": "headline_metrics",
            "metrics": [{"label": "원점 이동 최대 범위", "field": "max_origin_range", "format": "number"}],
        },
        {
            "id": "top20_card",
            "description": "10km 기본 격자 대비 상위 20% 조사영역의 최소 픽셀 가중 Jaccard.",
            "dataset": "headline",
            "sourceId": "headline_metrics",
            "metrics": [{"label": "최소 조사영역 Jaccard", "field": "min_top20", "format": "number"}],
        },
        {
            "id": "trajectory_card",
            "description": "궤적 증분의 95% CI 하한이 0보다 큰 면적비례 격자 수.",
            "dataset": "headline",
            "sourceId": "headline_metrics",
            "metrics": [{"label": "궤적 증분 지지", "field": "trajectory_supported", "format": "number", "unit": "/ 12 격자"}],
        },
        {
            "id": "oisst_card",
            "description": "OISST 증분의 95% CI 하한이 0보다 큰 면적비례 격자 수.",
            "dataset": "headline",
            "sourceId": "headline_metrics",
            "metrics": [{"label": "OISST 증분 지지", "field": "oisst_supported", "format": "number", "unit": "/ 12 격자"}],
        },
    ]

    charts = [
        {
            "id": "current_lift_chart",
            "title": "현재상태 AP lift: 격자 크기·원점별",
            "subtitle": "면적비례 footprint 기준의 12개 격자; 정확한 95% 구간은 아래 표에 제시.",
            "type": "bar",
            "dataset": "scaled_results",
            "sourceId": "grid_results",
            "encodings": {
                "x": {"field": "origin", "type": "nominal", "label": "원점"},
                "y": {"field": "current_lift", "type": "quantitative", "label": "Macro within-year AP lift"},
                "color": {"field": "grid_size", "type": "nominal", "label": "격자 크기"},
                "tooltip": [
                    {"field": "stable_cells", "type": "quantitative", "label": "안정 코호트 셀"},
                    {"field": "current_ci_low", "type": "quantitative", "label": "95% CI 하한"},
                    {"field": "current_ci_high", "type": "quantitative", "label": "95% CI 상한"},
                ],
            },
        },
        {
            "id": "spatial_chart",
            "title": "10km 기본 격자 대비 공간 일치도",
            "subtitle": "점 하나는 격자 크기×원점 사양 하나; 셀 성능과 실제 조사대상 안정성은 별개다.",
            "type": "scatter",
            "dataset": "scaled_results",
            "sourceId": "spatial_agreement",
            "encodings": {
                "x": {"field": "label_jaccard", "type": "quantitative", "label": "급감 라벨 Jaccard"},
                "y": {"field": "top20_jaccard", "type": "quantitative", "label": "상위 20% 조사영역 Jaccard"},
                "color": {"field": "grid_size", "type": "nominal", "label": "격자 크기"},
                "tooltip": [
                    {"field": "origin", "type": "nominal", "label": "원점"},
                    {"field": "stable_cells", "type": "quantitative", "label": "안정 코호트 셀"},
                    {"field": "test_events", "type": "quantitative", "label": "평가 사건 수"},
                ],
            },
        },
        {
            "id": "cohort_policy_chart",
            "title": "footprint 기준에 따른 코호트 셀 수",
            "subtitle": "5km에서 500픽셀 고정은 더 엄격하고, 20km에서는 더 느슨해진다.",
            "type": "bar",
            "dataset": "policy_results",
            "sourceId": "policy_results",
            "encodings": {
                "x": {"field": "spec_label", "type": "nominal", "label": "격자 사양"},
                "y": {"field": "stable_cells", "type": "quantitative", "label": "안정 코호트 셀"},
                "color": {"field": "threshold", "type": "nominal", "label": "footprint 기준"},
                "tooltip": [
                    {"field": "minimum_pixels", "type": "quantitative", "label": "최소 footprint 픽셀"},
                    {"field": "current_lift", "type": "quantitative", "label": "현재상태 AP lift"},
                ],
            },
        },
    ]

    tables = [
        {
            "id": "scaled_table",
            "title": "면적비례 격자별 결과",
            "subtitle": "AP lift·증분가치·공간 일치도의 정확한 점추정과 95% 구간.",
            "dataset": "scaled_results",
            "sourceId": "grid_results",
            "defaultSort": {"field": "current_lift", "direction": "desc"},
            "columns": [
                {"field": "spec_label", "label": "격자 사양", "type": "text"},
                {"field": "stable_cells", "label": "셀", "format": "number"},
                {"field": "current_lift", "label": "현재 lift", "format": "number"},
                {"field": "current_ci_low", "label": "현재 CI 하한", "format": "number"},
                {"field": "current_ci_high", "label": "현재 CI 상한", "format": "number"},
                {"field": "trajectory_increment", "label": "궤적 증분", "format": "number"},
                {"field": "oisst_increment", "label": "OISST 증분", "format": "number"},
                {"field": "label_jaccard", "label": "라벨 Jaccard", "format": "number"},
                {"field": "top20_jaccard", "label": "Top-20 Jaccard", "format": "number"},
            ],
        }
    ]

    title = "켈프 급감 위험순위화의 격자 크기·원점 민감도"
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {title}\n\n공식 Kelpwatch 원자료로 재구축한 5·10·20km 격자와 반 셀 원점 이동 강건성 분석."},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "grid_results",
            "body": (
                "## 기술 요약\n\n"
                f"면적비례 footprint 기준의 **12/12개 격자**에서 현재상태 AP lift의 95% CI가 0보다 컸다. "
                f"기본 10km 격자는 {int(primary.stable_cells)}셀, lift {primary.current_lift:.3f} "
                f"(95% CI {primary.current_ci_low:.3f}–{primary.current_ci_high:.3f})였다. "
                f"같은 크기 안에서 원점 이동에 따른 lift 최대 범위는 {max_origin_range:.3f}이었다."
            ),
        },
        {"id": "metrics", "type": "metric-strip", "cardIds": [card["id"] for card in cards]},
        {
            "id": "key_findings",
            "type": "markdown",
            "sourceId": "grid_results",
            "body": (
                "## 핵심 발견\n\n"
                "- **RQ1은 강해졌다.** 현재 캐노피만으로 만든 순위 신호는 5·10·20km와 네 원점에서 반복됐다.\n"
                f"- **궤적은 조건부다.** 증분가치의 95% CI가 0보다 큰 사양은 {trajectory_supported}/12개였다.\n"
                f"- **OISST 추가가치는 지지되지 않았다.** 양의 증분 CI를 보인 사양은 {oisst_supported}/12개였다.\n"
                "- **성능 안정성과 지도 안정성은 다르다.** 아래 공간 일치도에서 실제 상위 20% 조사영역은 격자에 따라 더 크게 바뀐다."
            ),
        },
        {"id": "current_chart_block", "type": "chart", "chartId": "current_lift_chart"},
        {"id": "spatial_chart_block", "type": "chart", "chartId": "spatial_chart"},
        {
            "id": "scope",
            "type": "markdown",
            "body": (
                "## 범위와 정의\n\n"
                "- 공간단위: EPSG:6933 equal-area 5·10·20km 격자, 크기별 기본·동쪽 반 셀·북쪽 반 셀·동북 반 셀 원점.\n"
                "- 고정 footprint: 1984–2004년 중 양의 켈프 면적을 가진 30m station pixels.\n"
                "- 주 코호트 기준: 셀 면적에 비례한 최소 footprint 125·500·2,000픽셀.\n"
                "- 라벨: 현재 상대 캐노피가 0.05보다 큰 셀 중 다음 해 연간 최대 캐노피가 30% 이상 감소.\n"
                "- 평가지표: 연도 내 AP lift, top-20% recall, 2년 블록 bootstrap 95% CI, 픽셀 가중 공간 Jaccard."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "grid_results",
            "body": (
                "## 방법론과 모델\n\n"
                "모든 격자는 공식 NetCDF에서 독립 재집계했다. 1989년부터 예측연도 직전까지를 학습하는 expanding-window로 "
                "2005–2024년을 평가했다. 주분석은 전체 켈프 도메인의 로지스틱 회귀 current-only와 current+trajectory이며, "
                "OISST는 유효 source-grid 81개에서 40km 이내 최근접 매칭이 가능한 동일 표본 보조분석이다."
            ),
        },
        {"id": "policy_chart_block", "type": "chart", "chartId": "cohort_policy_chart"},
        {
            "id": "robustness",
            "type": "markdown",
            "sourceId": "validation",
            "body": (
                "## 강건성·한계\n\n"
                f"독립 재검산은 **{validation_passed}/{len(validation)}개 검사**를 통과했다. 이전 165셀 패널도 4/4 항목에서 완전히 재현됐고, "
                "고정 프로토콜 적용 시 Southern California 2셀이 추가되어 기본 코호트는 167셀이 됐다. "
                "다만 이 실험은 공간 support 강건성이지 새로운 지역·센서·현장자료에 대한 생태적 외부검증은 아니다. "
                "또한 OISST 비교는 40km 지원범위 내 동일 표본으로 한정되므로 미지원 연안을 대표하지 않는다."
            ),
        },
        {"id": "detail_table_block", "type": "table", "tableId": "scaled_table"},
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## 논문 반영과 다음 단계\n\n"
                "1. 본문 주결과에는 10km 기본 사양을 두고, 이 분석은 ‘공간단위 강건성’ 절에 배치한다.\n"
                "2. ‘현재상태 신호가 유지된다’와 ‘우선조사 위치가 완전히 고정되지는 않는다’를 함께 보고한다.\n"
                "3. 주분석은 면적비례 코호트만 사용하고, 500픽셀 고정 결과는 코호트 선택 민감도로 부록에 둔다.\n"
                "4. 운영 배포 시에는 한 격자 지도를 절대적 정답으로 쓰지 말고, 여러 원점에서 반복 선택되는 픽셀을 합의 우선지역으로 제시하는 후속 분석이 필요하다."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 더 확인할 질문\n\n"
                "- 여러 원점에서 반복 선택된 공간만 조사하면 단일 격자 top-20%보다 사건 포착률이 안정적인가?\n"
                "- 5km의 세밀한 공간단위가 실제 관리비용 증가를 정당화할 만큼 위치 정확도를 높이는가?\n"
                "- 지역 holdout에서 격자 ensemble 우선순위가 단일 격자보다 calibration과 포착률을 개선하는가?"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "켈프 급감 위험순위화의 공간 support 강건성과 우선조사 위치 안정성을 평가한 기술 보고서.",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": [
                    {
                        "current_supported": int(headline.current_supported),
                        "max_origin_range": rounded(max_origin_range),
                        "min_label": rounded(min_label),
                        "min_top20": rounded(min_top20),
                        "trajectory_supported": trajectory_supported,
                        "oisst_supported": oisst_supported,
                    }
                ],
                "scaled_results": scaled_rows,
                "policy_results": policy_rows,
            },
        },
        "sources": sources,
        "package_info": {
            "sourceKind": "local-reproducible-experiment",
            "controls": {"edit": False, "refresh": False},
        },
    }
    (output / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"artifact": str(output / "artifact.json"), "rows": len(scaled_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
