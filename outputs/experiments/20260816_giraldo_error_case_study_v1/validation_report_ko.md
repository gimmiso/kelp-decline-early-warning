# Giraldo 보조 사례연구 독립 검증 보고서

## Overall Assessment: Share with caveats

총 25개 검증 중 25개가 통과했다. 계산·키·입력 hash·산출물 hash와 그림 렌더링에 관한 공유 차단 오류는 없다.

## Methodology Review

- 보조 사례연구는 주 165셀 모형을 재적합하지 않고 저장된 expanding-window out-of-fold 예측을 사용했다.
- 분석대상은 Giraldo 자료가 같은 셀·연도에 존재하는 California 52셀·424 cell-year로 제한됐다.
- 네 생태축은 분석 전에 고정됐고 144개 열 전수 탐색을 하지 않았다.
- 계수는 지역·연도 고정효과와 셀·연도 two-way cluster 표준오차를 사용했다.
- 이 분석은 pilot-informed exploratory diagnostic이며 독립 확증이 아니다.

## Calculation Spot-Checks

- 원자료 9,728행×144열, 191현장, 1999–2021년을 재확인했다.
- 52셀·424 cell-year·151사건 및 세 모형군의 동일 평가행을 재확인했다.
- Logistic matched 표본의 OISST AP-lift 증분을 원 예측확률에서 재계산했다: +0.011526.
- 행별 OISST signed rank gain 최대 오차: 7.11e-15.
- 행별 OISST Brier gain 최대 오차: 9.97e-17.
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
