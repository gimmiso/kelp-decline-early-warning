# Giraldo 보조 사례연구 방법론 감사 프로토콜 v3

## 목적

기존 v2의 `안정 조건 0/20`을 생태적 관계의 부재로 해석하지 않는다. v3는 Giraldo-Ospina et al. (2025)의 본문, Appendix S1-S5, 공식 Zenodo 재현코드와 2024-2026 후속연구를 기준으로, v2의 음성결과가 다음 설계 선택에 민감했는지 평가한다.

1. 논문 분석대상이 아닌 site를 포함한 코호트 희석
2. bull kelp와 giant kelp 및 생태지역의 통합
3. 비선형 임계반응을 단일 선형계수로 축약
4. purple urchin 대신 red+purple urchin 합을 사용
5. PISCO에서는 현장 transect 수심 `depth_mean` 대신 격자형 평균수심 `mean_depth` 사용
6. 파랑, 영양염, NPP, 포자량 및 교란체제 상호작용 누락

## 고정 코호트

- 저자 재현코드의 `No_survey_years_per_site.csv`에서 2014년 이전 조사연도가 3개 이상인 site만 사용한다.
- 북부는 bull kelp 문맥, 중앙-남서부와 남동부는 각각 저자 giant kelp 문맥으로 분리한다.
- 같은 site가 저자의 bull/giant 모델에 동시에 포함될 수 있는 중앙부에서는 위성 canopy 비교에 사용한 giant kelp 문맥을 우선한다.
- 현장관측이 없는 site-year를 이월하거나 대체하지 않는다.
- 북부 Reef Check 10개 site는 공개자료의 `depth_mean`이 전부 결측이므로 이 문맥에서만 지도형 `-mean_depth`를 사용하고 `depth_mapped_fallback=1`로 표시한다. 이 예외를 숨기지 않고 별도 민감도 결과로 보고한다.

## 특징 블록

모든 비교는 동일한 현장관측 site-year 표본에서 수행한다.

1. `trajectory`: KelpWatch 현재 상태와 과거 궤적
2. `trajectory_plus_field_state`: 종별 현장 켈프 밀도, purple urchin, 종별 성게 임계초과 비율, giant kelp 기반 능동방목 proxy, 현장 수심, 암반확률, VRM, 과거 현장 켈프 평균·변동성
3. `trajectory_plus_paper_environment`: 저자 최종모형이 지역별로 선택한 수온·질산염·파랑·orbital velocity·NPP와 물리서식처
4. `trajectory_plus_paper_full`: 현장 생태상태, 환경, giant kelp 전년도 포자량을 결합
5. `trajectory_plus_full_regime`: 2014년 이후 및 2019년 이후 방목·수온 관계 변화 항을 추가

성게 밀도는 `den_STRPURAD`만 사용한다. giant kelp의 능동방목 proxy는 Smith et al. (2024)의 공개 재현코드에 제시된 결정론적 관계를 사용하고 무작위 잔차는 추가하지 않는다.

## 모형과 평가

- Logistic linear: v2의 선형성 가정 확인
- Logistic spline: 각 추가 연속변수에 4-knot cubic spline 적용
- Random Forest, XGBoost: 비선형성과 상호작용에 대한 알고리즘 강건성
- 모든 모형은 2004년부터 예측 직전 연도까지만 학습하는 expanding-window로 2008-2021년을 평가한다.
- 주 지표는 연도 내 AP lift의 macro 평균이며, Brier score와 상위 20% recall을 함께 보고한다.
- 증분 신뢰구간은 연도와 overlap cluster를 함께 재표집하는 계층 bootstrap으로 산출한다.

## 판정

주 판정은 300 m Logistic spline의 `paper full - trajectory` 증분이다. 95% 계층 bootstrap 신뢰구간이 0을 제외하고, 4개 모형 중 3개 이상이 같은 방향이며, 1 km Logistic spline도 같은 방향일 때에만 "방법론 보완 후 생태블록의 안정적 추가가치"로 기술한다.

통과하지 못하면 “현재의 현장부분표본과 다음 해 canopy 급감 목표에서는 논문 정렬·비선형·교란체제 보완 후에도 안정적인 추가가치를 확인하지 못했다”고 결론낸다. 이는 Giraldo 논문의 동시점 kelp 밀도 설명결과를 반박하지 않는다.
