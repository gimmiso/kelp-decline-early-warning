# 전 해안·California 모형구조 확장 프로토콜 v1

## 지위

이 분석은 기존 외부예측 결과를 본 뒤 추가하는 `post hoc methodological robustness`이다. 기존 Logistic·Random Forest·XGBoost 결과를 교체하거나 외부시험 성능이 가장 높은 모형을 새 주모형으로 선언하지 않는다.

## 전 해안 5 km

동일한 30% 다음 해 급감 라벨과 5% 최소 canopy 조건, 2005–2024 expanding-window를 유지한다.

1. Elastic-net logistic: 상관된 trajectory·환경변수의 계수를 축소한다.
2. Penalized hierarchical additive logistic: 현재 canopy와 공개환경변수의 4-knot spline, region factor, L2-축소 cell intercept를 하나의 모형에서 적합한다. 이는 Python에서 실행 가능한 GAMM형 MAP 근사이며 완전한 Bayesian GAMM으로 부르지 않는다.
3. LambdaMART ranker: 학습연도를 query group으로 두어 연도 내 순위를 직접 학습한다. 마지막 과거 3년의 out-of-time score로 Platt calibration한 확률을 별도로 저장한다.

Elastic-net과 hierarchical additive의 규제강도는 각 outer forecast fold 안에서 마지막 과거 3년을 inner validation으로 사용해 선택한다. outer test year는 튜닝에 사용하지 않는다.

2015–2024년에는 지역 하나를 완전히 제외하고 1989–2014년 다른 지역만 학습하는 시공간 이중 holdout을 추가한다.

## California Giraldo 부분표본

v3의 고정 site-year와 종·지역별 변수정의를 그대로 사용한다. 별도 context 모형의 확률을 합치는 대신 하나의 pooled hierarchical additive logistic을 적합한다.

- context factor와 L2-축소 site intercept
- context 내부 학습자료로 표준화한 생태·환경변수
- 현재 canopy·purple urchin·수온·질산염·파랑의 제한된 spline
- expanding-window와 300 m/1 km 분리

이는 Giraldo의 동시점 Tweedie GAM을 재현하는 분석이 아니라 다음 해 이진 canopy 위험에 맞춘 구조적 민감도이다.

## 판정

모든 정보블록은 동일한 cell/site-year 표본에서 비교한다. AP만 좋아지는 결과는 채택하지 않는다. 다음을 함께 보고한다.

- macro within-year AP lift와 정보블록 증분
- 5/10/20/30% budget recall
- Brier, log loss, calibration intercept/slope
- 2021·2022 제외와 leave-one-year-out
- 전 해안 시공간 region holdout

증분 95% CI가 0을 제외하고, budget recall이 악화되지 않으며, calibration이 실질적으로 악화되지 않을 때만 안정적인 개선으로 기술한다.
