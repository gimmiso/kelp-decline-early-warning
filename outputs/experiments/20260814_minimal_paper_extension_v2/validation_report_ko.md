# 최소 추가 실험 검증 보고서

## Overall Assessment: Share with caveats

독립 재계산과 파일 무결성 검사는 **통과**다. 총 19개 검증 중 19개가 통과했으며, 공유를 막는 계산·누수·행 불일치 오류는 없다.

## Methodology Review

- 질문: 현재 캐노피, 과거 궤적, NOAA proxy가 다음 해 30% 급감 셀의 연도 내 순위화에 기여하는가.
- 모집단: 현재 상대 캐노피가 0.05보다 큰 165셀의 eligible cell-year.
- 검증: 2005–2024 expanding window; 모든 fold에서 train_end < test_year.
- 비교: full OISST domain 및 U.S. 31–47°N upwelling-supported domain 내부에서 정확히 같은 cell-year 행.
- 지표: 연도별 AP에서 그해 유병률을 뺀 뒤 평균한 macro within-year AP lift; 제한예산은 연도별 Top-20%의 micro event recall.

## Calculation Spot-Checks

- 연도별 AP lift: 저장된 전체 값을 prediction-level에서 재계산, 최대 오차 9.71e-17.
- macro within-year AP lift: 최대 오차 8.33e-17.
- Top-20% recall: 최대 오차 5.55e-17.
- 2022 제외 current-only: 수동 0.176753, 저장 0.176753.
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
