# 다중공간척도 본 분석·California 사례연구 프로토콜 v1

## 1. 목적과 역할 분리

본 연구는 서로 다른 질문을 하나의 공간격자에 억지로 결합하지 않는다.

1. **전 해안 5 km 본 분석**은 제한된 조사예산에서 다음 해 canopy 급감 위험을 순위화하고 정보블록의 증분적 의사결정 가치를 평가한다.
2. **California 300 m·1 km 보조 사례연구**는 Giraldo 현장자료가 관측된 부분표본에서 원격자료 기반 모형의 오류가 국지 생태조건과 체계적으로 관련되는지 탐색한다.

본 분석의 성능은 전 해안 의사결정 성능으로, 사례연구의 결과는 California 부분표본의 오류진단으로 각각 보고한다. 두 분모를 합치거나 사례연구의 연관성을 전 해안 인과기작으로 일반화하지 않는다.

## 2. 전 해안 5 km 본 분석

### 2.1 모집단과 공간지지

- KelpWatch 공식 분기 자료의 고정된 pre-forecast footprint를 사용한다.
- 5 km area-scaled 격자, 원점 o00을 본 규격으로 사용한다.
- 안정 footprint를 충족한 368개 셀을 추적한다.
- 격자 크기와 반 셀 원점 이동 결과는 별도 강건성 실험으로 보고하여 단일 경계의 절대성을 주장하지 않는다.

### 2.2 예측질문과 정보블록

- 대상: 현재 canopy가 최소 기준을 넘는 셀의 다음 해 30% 이상 상대감소.
- 정보블록 순서: 현재 canopy → 과거 canopy 궤적 → OISST → CUTI/BEUTI.
- 각 증분 비교는 같은 셀-연도와 같은 평가연도에서 수행한다.
- 공개 환경자료는 운영 가능한 proxy로 정의하며 실제 생태기작의 직접 측정치로 해석하지 않는다.

### 2.3 평가

- 주 평가: 2005–2024 expanding-window 미래예측.
- 주 지표: 연도별 AP에서 해당 연도 유병률을 뺀 AP lift의 macro 평균.
- 보조 지표: pooled AP lift, Brier score, log loss.
- 의사결정 지표: 연도별 상위 5%, 10%, 20%, 30% 조사 시 micro recall과 연도 블록 bootstrap 신뢰구간.
- 모델 강건성: 고정된 Logistic, Random Forest, XGBoost.
- 라벨 강건성: 20%, 30%, 40% 급감 × 현재 상대 canopy 0.02, 0.05, 0.10.

### 2.4 해석규칙

- 현재 상태 가치는 macro AP-lift 신뢰구간 하한이 0보다 클 때 지지한다.
- 추가 블록은 동일 연도 paired AP-lift 차이의 95% 연도-block bootstrap 신뢰구간 하한이 0보다 클 때 양의 증분가치가 있다고 판정한다.
- 환경블록 음성결과는 해당 공간지지·예측시점·특징정의에서 추가 예측정보가 확인되지 않았다는 의미이며 환경의 생태적 무관성을 뜻하지 않는다.

## 3. California 300 m·1 km 사례연구

### 3.1 공간지지와 고정 footprint

- Giraldo 현장 site 좌표 주변의 KelpWatch 양의 pre-2005 canopy pixel로 site별 footprint를 고정한다.
- 300 m와 1 km를 사전에 함께 분석한다. 결과를 본 뒤 유리한 척도만 선택하지 않는다.
- 최소 footprint pixel은 300 m에서 5개, 1 km에서 20개이며 pre-2005 양의 canopy 연도가 10년 이상인 site만 유지한다.
- footprint 중심 간 거리가 공간지지의 두 배 이내인 site는 동일 overlap cluster로 묶어 독립 공간복제로 세지 않는다.

### 3.2 연간 canopy 관측 규칙

- 분기별 유효 footprint pixel 비율이 50% 이상이어야 해당 분기가 유효하다.
- 연간값은 최소 3개 유효 분기와 Q3 유효를 요구하고, 유효 분기 중 최대 canopy를 사용한다.
- 이 규칙은 구름·센서 관측노력 차이를 완전히 제거하지 않지만 단일 저품질 최대값에 대한 의존을 줄인다.

### 3.3 MUR 1 km 일별 열스트레스

- 자료: NASA JPL MUR v4.1, NOAA CoastWatch ERDDAP `jplMURSST41`.
- 각 site에서 유효한 최근접 해양 pixel을 고정한다.
- 특징: 연평균 SST, 연최대 SST, 7일 평균 anomaly 최대, 양의 anomaly 누적 degree-days, 과거자료 expanding 90백분위 초과일수, 최대 연속 고온일수.
- 각 연도의 climatology와 90백분위는 해당 연도 1월 1일 이전의 동일 pixel 자료만 사용한다.
- MUR가 2002-06-01에 시작하고 최소 365일의 과거기준을 요구하므로 완전한 열스트레스 특징은 2004년부터 구성한다. 단일 학습연도 모형을 피하기 위해 최소 4개 완전 학습연도를 확보한 2008년부터 예측한다.

### 3.4 생태축과 오류진단

사전 고정 생태축은 다음 네 가지다.

- 성게 방목압: red+purple urchin density의 `log1p` 합성축.
- 암반 서식처 확률.
- 양의 수심(m).
- bull kelp 대 giant kelp 밀도의 로그 대비.

오류진단은 trajectory 모형의 순위품질·false negative·false positive와 MUR 추가 전후의 signed rank gain·Brier gain이다. 각 생태축은 region과 year 고정효과를 포함한 선형 연관모형으로 평가하며 overlap cluster와 year의 two-way cluster 분산을 사용한다.

### 3.5 다중검정과 안정성

각 support×모델×오류결과 안에서 네 생태축의 Benjamini–Hochberg q-value를 계산한다. 논문에서 안정된 조건으로 강조하려면 다음을 모두 만족해야 한다.

1. 탐색적 오류진단에 사전 고정한 기준에 따라 300 m Logistic 주 분석에서 95% CI가 0을 제외하고 q<0.10.
2. 300 m의 세 모델 중 최소 두 모델이 같은 방향.
3. 1 km Logistic 추정치도 같은 방향.

이를 통과하지 못한 결과는 탐색적 추정치로만 제시한다.

## 4. 예상 결과별 대응

- **두 support에서 같은 생태조건이 안정적으로 지지됨:** 원격자료 모형이 실패하는 관측가능한 국지 조건에 대한 가설로 제시하되 인과효과로 쓰지 않는다.
- **한 support에서만 지지됨:** 공간지지 의존 결과로 보고하고 주 결론에서 제외한다.
- **안정된 조건이 없음:** 현장부분표본·시점정렬·생태축만으로 오류의 체계적 조건을 확인하지 못한 음성결과로 보고한다.
- **MUR가 예측을 개선함:** 전 해안의 거친 OISST 음성결과와 대비하여 공간지지 또는 열스트레스 요약의 중요성이라는 제한된 해석을 제시한다.
- **MUR도 개선하지 못함:** 평균화와 공간해상도를 바꿔도 공개 SST의 canopy 이력 이후 추가가치를 확인하지 못했다고 결론내리되, 표층 proxy의 한계를 명시한다.

## 5. 재현성과 보고

- 두 분석은 별도 config, 입력 hash, 예측표, fold audit, 결과표, 그림, manifest를 저장한다.
- headline 성능과 증분값은 저장된 예측에서 독립 재계산한다.
- 데이터 grain, 결측, 중복키, merge cardinality, 공간·연도 coverage를 결과와 함께 공개한다.
- 모든 실험은 `experiments/registry.csv`에 상태, commit, run id, 판정과 핵심 수치를 기록한다.
