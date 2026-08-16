# Giraldo 현장자료 기반 예측오류 보조 사례연구 프로토콜 v1

작성일: 2026-08-16
분류: pilot-informed exploratory diagnostic
설정 파일: `configs/giraldo_error_case_study_v1.json`

## 목적

본 분석은 Giraldo-Ospina 자료를 미국 서부 해안 165개 셀의 다섯 번째 예측 정보블록으로 사용하지 않는다. 현장자료가 실제로 존재하는 California 부분표본에서 기존 expanding-window 모형의 반복 오류와 OISST의 조건부 증분가치가 제한된 생태조건과 관련되는지를 탐색한다.

탐색질문은 다음과 같다.

> California 현장자료가 존재하는 부분표본에서 반복적인 누락·과대경보와 OISST의 순위변화는 성게 방목압, 물리적 서식처 및 켈프 군집 구성과 체계적으로 관련되는가?

## 분석대상과 제한

- 주 예측값: `20260814_journal_extension_v1`의 2005–2021년 expanding-window out-of-fold 예측
- 주 모형: Logistic Regression
- 강건성 모형: Random Forest, XGBoost
- 공간범위: Giraldo 현장좌표와 10 km 셀이 겹치는 California 부분표본
- 현장자료가 없는 Oregon, Washington, Baja로 일반화하지 않는다.
- 미관측 셀·연도에 성게나 서식처 값을 대체하지 않는다.
- 다른 연도의 최근접 현장관측을 이월하지 않는다.

## 공간·시간 결합

각 site-year의 좌표 중앙값을 165셀 중심점과 비교한다. 주 결합은 가장 가까운 셀 중심까지의 Haversine 거리가 7.2 km 이하인 경우이며, 10 km 정사각형의 반대각선에 해당하는 근사 규칙이다. 5 km 이하 결합을 엄격한 민감도 분석으로 사용한다.

현장 관측은 다음 순서로 집계한다.

1. 60㎡ transect → site-year 중앙값
2. site-year → 10 km cell-year 중앙값

각 cell-year에는 IQR, 현장 수, transect 수, 최대 결합거리를 함께 보존한다. 이 값은 10 km 셀 전체의 무편향 생태평균이 아니라 조사된 암반초의 상태를 나타낸다.

## 사전 고정 생태축

144개 열을 전수 탐색하지 않고 다음 네 축만 사용한다.

1. 성게 방목압: `log1p(보라성게 밀도 + 자주성게 밀도)`
2. 암반 서식처: 조사 site polygon의 평균 암반확률
3. 수심: `mean_depth`의 양의 수심 크기
4. 켈프 종 구성: `log1p(bull kelp 밀도) - log1p(giant kelp 밀도)`

## 오류와 조건부 가치 정의

- 매년 전체 full-OISST 평가표에서 위험점수 상위 20%를 조사대상으로 선택한다.
- false negative는 실제 급감 셀이 상위 20%에 들지 못한 경우다.
- false positive는 비급감 셀이 상위 20%에 포함된 경우다.
- OISST signed rank gain은 OISST 추가가 사건 셀을 위로, 비사건 셀을 아래로 이동시켰을 때 양수가 되도록 정의한다.
- OISST Brier gain은 trajectory 모형 대비 OISST 모형의 행별 제곱오차 감소량이며 양수가 개선이다.

## 추정과 판정

각 생태축을 한 번에 하나씩 사용하고 지역·연도 고정효과를 포함한다. 계수는 분석표 내부 1표준편차 변화당 효과로 표현하며, 셀과 연도를 동시에 고려한 two-way cluster-robust 표준오차를 사용한다. 같은 결과변수·모형군 안의 네 생태축에는 Benjamini–Hochberg 보정을 적용한다.

안정적인 조건으로 해석하려면 다음을 모두 만족해야 한다.

1. 95% 신뢰구간이 0을 제외
2. 보정 q-value < 0.10
3. 세 모형 중 둘 이상에서 같은 방향
4. 엄격한 5 km 결합에서도 같은 방향

이 기준을 만족하지 않으면 상관의 부재가 증명된 것으로 표현하지 않고, 현재의 부분표본과 공간 support에서 안정적인 조건을 확인하지 못했다고 결론내린다.

## 결과별 대응

- 안정적인 조건 없음: 현장자료도 공개 프록시 실패의 단일 생태조건을 식별하지 못했으며, 표본·공간 support의 한계를 강조한다.
- 일부 조건만 Logistic에서 관찰: 모형 의존적 탐색신호로만 보고한다.
- 둘 이상 모형과 공간 민감도에서 반복: 후속 표준화 현장조사의 우선 가설로 제안한다.
- matched 표본에서만 OISST 증가: 표본선택에 따른 조건부 결과로 보고하며 165셀 전체 가치로 일반화하지 않는다.
