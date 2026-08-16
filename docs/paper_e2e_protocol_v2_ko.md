# 논문용 End-to-End 분석계획 v2

## 0. 문서 상태와 원칙

- Protocol ID: `KELP-EWS-E2E-v2.0`
- 작성일: 2026-08-13
- 상태: **실험 실행 전 검토용 초안**
- 성격: 기존 50-cell 결과와 165-cell Kelpwatch-only 파일럿을 확인한 뒤 작성한 **pilot-informed locked extension plan**이다. 사전등록으로 표현하지 않는다.
- 잠금 대상: 아래 규칙이 승인된 뒤 처음 생성되는 전 서해안 NOAA 결합 결과다.
- 최우선 원칙: 결과가 어느 방향으로 나오든 같은 규칙으로 분석하고, 결과에 맞춰 변수·표본·기간·평가지표를 다시 고르지 않는다.
- 발표자료 제작은 본 계획과 후속 실험 범위에서 제외한다.
- Baja California부터 Washington까지의 연구지역 확대는 표본 수를 늘리기 위한 사후 변경이 아니라, 사용자가 의도한 **목표 모집단의 확대**다. 따라서 California 연구를 그대로 외삽하지 않고 생태권역·우점종·자료지원범위의 차이를 연구질문과 검증설계에 포함한다.

현재까지 이미 확인한 파일럿 결과는 가설 생성과 설계 점검에만 사용한다. 특히 165-cell Kelpwatch-only 백테스트의 궤적 효과는 같은 자료에서 다시 계산해도 독립적 확증이 아니므로 `replication/robustness`로만 분류한다. 아직 실행하지 않은 전 서해안 NOAA 증분가치 비교가 이번 잠금 확장의 중심이다.

## 1. 논문의 결정 문제와 중심 주장

### 1.1 실제 결정 문제

연말 `t` 시점에 그해까지 완전히 관측된 Kelpwatch 캐노피와 NOAA 환경자료를 이용하여, 다음 해 `t+1`에 캐노피 면적이 30% 이상 감소할 가능성이 높은 10 km 셀을 제한된 현장조사 예산 안에서 우선순위화할 수 있는지 평가한다.

이 연구는 다음 세 가지를 구분한다.

1. 해안 전체가 나쁜 **연도**를 식별하는 능력
2. 같은 연도 안에서 더 위험한 **셀**을 식별하는 능력
3. 현재 캐노피와 과거 변화가 이미 알려진 뒤에도 NOAA가 주는 **추가 정보**

### 1.2 논문의 중심 질문

> 의도적으로 확대한 북동태평양 서해안 범위에서, 예측시점 이전의 열 노출과 용승 지표가 현재 켈프 캐노피 및 최근 궤적을 넘어 다음 해 급감 셀의 **연도 내 우선순위**에 재현 가능한 증분가치를 제공하며, 그 가치가 생태권역 사이에서도 이전되는가?

이 질문의 새로움은 `환경이 kelp에 영향을 주는가`가 아니다. 그 관계는 선행연구에서 이미 충분히 알려져 있다. 본 연구의 빈틈은 환경자료가 강한 캐노피 이력 기준선 이후에도, 미래 연도만으로 검증한 제한예산 공간순위에서 실제 추가가치를 갖는지와 그 가치가 확대 지역에서 이전되는지를 검증하는 데 있다. 근거 종합은 `docs/literature_rationale_environmental_predictors_v2_ko.md`에 고정한다.

### 1.3 허용되는 주장

- 예측적 연관성, 증분 예측가치, 순위화 성능, 확률 보정, 공간·시간 이전 가능성
- 지정한 자료·공간범위·예측시점에서의 후향적 백테스트 결과
- 환경자료가 광역 연도 위험과 연도 내 공간 위험 중 어디에 기여하는지에 대한 평가
- 확대 지역에서의 평균효과와 생태권역별 이질성, 그리고 미관측 권역으로의 이전 가능성

### 1.4 금지되는 주장

- NOAA 변수의 계수나 중요도를 켈프 감소의 인과효과로 해석
- 한두 해 또는 일부 지역의 높은 AP를 전 서해안 일반화로 표현
- Landsat가 구분하지 못하는 giant kelp와 bull kelp의 종별 효과 주장
- 5 km 또는 0.25도 SST를 10 km 셀의 현장 수온으로 표현
- 결과를 본 뒤 가장 좋은 임계값·격자·환경제품·모형만 선택해 최종모형으로 제시
- 확률 보정이 실패했는데 위험확률로 표현하거나, 공간 이전이 실패했는데 운영 가능한 조기경보로 표현
- 파랑·영양염 현장관측·성게 등 미관측 요인이 있는데 열·용승 결과를 `환경 전체의 효과`로 표현

## 2. 연구질문, 가설, 반증조건

### RQ1. 캐노피 이력만으로 다음 해 급감의 연도 내 우선순위가 가능한가?

#### H1a: 현재 상태 기준선

- 대립가설: `current_only`가 훈련기간 사건률 및 단순 지속성 기준선보다 평균 연도 내 AP lift가 높다.
- 반증조건: 차이가 0 이하이거나 불확실성 구간이 큰 폭으로 양·음을 모두 포함한다.
- 위치: 기반 성능 확인. 이미 파일럿을 보았으므로 새 확증가설이 아니라 재현성 기준이다.

#### H1b: 궤적 증분가치

- 대립가설: `current_plus_trajectory - current_only`의 평균 연도 내 AP lift 차이가 양수다.
- 반증조건: 차이가 0 이하이거나, 양수여도 지역-연도 평가에서 방향이 유지되지 않는다.
- 위치: NOAA 비교를 위한 강한 캐노피 기준선. 동일 165-cell 파일럿 결과를 이미 보았으므로 확증이 아니라 잠금 재분석이다.

### RQ2. NOAA 열 노출은 캐노피 이력을 넘어 도움이 되는가? — 주 연구질문

#### H2a: 5 km 열 노출의 증분가치 — 주 가설

- 비교: `current_plus_trajectory_plus_crw_thermal - current_plus_trajectory`
- 주 평가량: 2005–2024 expanding-window 예측의 **macro within-year AP lift 차이**
- 대립가설: 차이가 양수이고, 20% 조사예산의 재현율을 악화시키지 않는다.
- 강한 지지 기준:
  1. 점추정치가 양수다.
  2. 95% year-block bootstrap 구간의 하한이 0보다 크다.
  3. 평가 가능한 연도의 60% 이상에서 paired AP lift 차이가 양수다.
  4. 연도별 Top-20% recall 차이가 0 이상이다.
- 부분 지지: 점추정치는 양수지만 구간이 0을 포함하거나 Top-20%가 개선되지 않는다.
- 반증: 점추정치가 0 이하이거나, 효과가 한 광역 급감연도에만 의존한다.

#### H2b: 제품 해상도 가설

- 비교: 동일한 열 특징을 만든 `CRW CoralTemp 5 km` 대 `OISST v2.1 0.25도`
- 대립가설: CRW 기반 열 특징의 연도 내 증분가치가 OISST보다 크다.
- 해석 제한: CRW가 우세해도 ‘해상도가 원인’이라고 단정하지 않는다. 제품의 입력자료·climatology·해안 마스크 차이도 함께 달라지므로 scale-mismatch와 source sensitivity로 표현한다.

#### H2c: 예측시점–생태시기 정렬 — 잠금 보조가설

- 이유: 선행연구의 가장 직접적인 예측근거는 같은 해 겨울 해양조건으로 그해 여름 bull-kelp canopy를 예측한 것이다. 12월 31일 `t`에서 `t+1` annual maximum을 예측하는 주 질문은 더 긴 lead time을 요구하므로 같은 효과를 당연시할 수 없다.
- 보조 clock: 3월 31일 `t+1`에 `t`까지의 canopy 이력과 `t+1` 1–3월 환경자료로 `t+1` Q2–Q3 maximum의 30% 감소를 예측한다.
- 판정: 주 clock은 실패하고 pre-season clock만 성공하면 `환경자료는 next-year 조기경보가 아니라 짧은 pre-season 업데이트에 유용`하다고 결론낸다. 보조 clock이 주 결과를 대체하지 않는다.
- 누수방지: 4월 이후 환경자료와 `t+1` canopy 관측은 어떤 형태로도 predictor에 포함하지 않는다.

### RQ3. 열 이외의 환경 신호가 도움이 되는가? — 보조가설

#### H3a: CUTI/BEUTI 증분가치

- 분석 모집단: NOAA가 설명한 31–47°N의 U.S. West Coast 유효범위에 포함되고, 연구 지역 정의상 미국 연안으로 분류되는 셀만 사용한다. 범위 밖 값을 31° 또는 47° 경계값으로 강제 대입하지 않는다.
- 비교: `current_plus_trajectory_plus_crw_plus_upwelling - current_plus_trajectory_plus_crw`
- 대립가설: macro within-year AP lift 차이가 양수다.
- 반증조건: 차이가 0 이하, 지역별 부호가 심하게 반대, 또는 유효 표본이 아래 표본성 기준을 충족하지 못한다.
- 주장 범위: 지지되더라도 U.S. West Coast 지원범위 안에서만 주장한다. Baja와 47°N 북쪽으로 외삽하지 않는다.

#### H3b: 파랑 교란의 증분가치 — 자료적합성 통과 시 잠금 보조분석

- 생태 근거: California의 giant kelp 연구에서는 겨울 파랑이 캐노피 손실의 주요 설명변수였고, 최근 광역 연구에서도 파랑효과의 방향과 크기가 종·지역에 따라 달랐다.
- 자료결정: 결과를 보기 전에 coverage, 기간, 근해 격자거리, 섬/해안 마스크를 감사한다. 1984–2024 전역을 일관되게 지원하는 자료가 품질문턱을 통과하면 `wave_increment`를 잠금 보조분석으로 실행한다.
- 비교: `current_plus_trajectory_plus_crw_plus_wave - current_plus_trajectory_plus_crw`
- 예상가설: 극한 겨울 파랑 노출은 특히 giant-kelp 우점권역에서 다음 캐노피 감소 위험의 공간순위를 개선하지만, bull-kelp 우점권역에서는 효과가 약하거나 달라질 수 있다.
- 실패/제외: 근해 support가 불충분하면 결과를 만들지 않고 `failed_qc`로 남긴다. NOAA 열·용승의 음성 결과를 `환경은 쓸모없다`고 확대해석하지 않는다.

### RQ4. 높은 pooled 점수는 실제 셀 우선순위 성능을 과대평가하는가?

#### H4: 연도 성분과 공간 성분의 분리

- 각 예측모형에 대해 원점수, 연도 평균점수, 연도 내 순위점수를 각각 평가한다.
- 예상가설: pooled AP의 상당한 부분은 광역 event-burden이 높은 연도 구분에서 발생하고, macro within-year AP lift는 더 작다.
- 주의: AP는 가법 분해되지 않으므로 ‘pooled AP의 몇 %가 연도효과’라고 계산하지 않는다. 세 점수의 성능을 병렬 비교한다.
- 반증조건: 연도 내 순위 성능이 pooled 성능과 유사하게 높고 여러 해·지역에서 반복된다.

### RQ5. 개선이 실제 조사예산과 공간 이전에서 유지되는가?

#### H5a: 조사예산 효용

- 대립가설: NOAA 포함 모형이 동일 연도·동일 후보에서 현재+궤적 모형보다 Top-20% recall을 높인다.
- 보조예산: Top-10%, 고정 5개 셀, 고정 10개 셀.
- 무사건 연도는 recall 평균에서 제외하되, 불필요한 조사 건수는 별도로 보고한다.

#### H5b: 공간 이전

- 대립가설: leave-one-region-out과 위도 기반 공간블록에서 NOAA 증분가치가 양수다.
- 반증조건: 보유지역 내부에서는 개선되지만 미관측 지역에서 0 이하가 된다.
- 해석: 실패하면 ‘지역 내 후향적 우선순위’로 주장을 제한하고 범용 서해안 모델을 주장하지 않는다.

### RQ6. 결과가 데이터·라벨·공간단위 선택에 강건한가?

#### H6: 설계 강건성

- 대립가설: 효과 방향이 급감 20/30/40/50%, 적격기준 0.02/0.05/0.10, 고정 pre-2005/rolling-available cohort, 5/10/20 km 격자, 관측완전성 규칙, 열 제품 대안에서 일관된다.
- 반증조건: 주 결론의 부호가 임계값이나 격자 원점에 따라 반복적으로 바뀐다.
- 해석: 불안정하면 ‘효과 존재’ 대신 ‘설계 의존성’을 논문의 주 결과로 보고하고 운영 적용 주장을 차단한다.

## 3. 데이터 선택 감사와 분석 모집단

### 3.1 Kelpwatch 주 자료

- 출처: EDI package `knb-lter-sbc.74.34`, revision 34
- 로컬 snapshot: `LandsatKelpBiomass_2026_Q2_withmetadata.nc`
- SHA-256: `b27caf8e4d08c0d2816d044e254a6339c854f2933e14a3fa2a139b37832f884f`
- 범위: 1984 Q1–2026 Q2, Baja California에서 U.S.–Canada border까지
- 분석 사용기간: 1984–2025의 완결된 연도만 사용한다. 2026은 부분연도이므로 제외한다.
- 반응변수: `area`를 canopy area로 사용한다. 연구질문이 canopy **면적 감소**이므로 biomass가 아니라 area가 주 변수다.
- 종 해석: Landsat는 giant/bull kelp를 구분하지 못하므로 전 범위에서는 `canopy-forming kelp`라고 쓴다.

현재 파일 메타데이터의 `title/summary`는 central/southern California라고 쓰지만 실제 지리경계는 27.01–48.40°N이며, Kelpwatch 공식 방법론은 Baja–U.S./Canada border 범위를 설명한다. 이 불일치는 데이터 담당자 문서 또는 EDI XML과 대조해 해소하고, 해소 전에는 데이터 범위 문장을 파일의 좌표경계와 Kelpwatch 공식 문서에 한정한다.

### 3.2 왜 50개가 아니라 전 서해안인가

기존 50개는 NOAA 완전자료 때문에 잘린 것이 아니라, Northern/Central California의 285개 후보 격자 중 all-history footprint ≥500을 적용한 결과다. 따라서 이 50개는 전 서해안 모집단의 확률표본이 아니며 북부·남부·Baja·Oregon·Washington 일반화에 사용할 수 없다. 이번 지역 확대는 의도된 설계변경이므로 `California 표본 확대`가 아니라 **새로운 West Coast target population**으로 기술한다.

새 주 모집단은 다음 기준을 **예측기간 시작 전 자료만으로** 정의한다.

1. 전 서해안 공식 NetCDF 내 10 km equal-area grid
2. 1984–2004년에 관측된 kelp footprint pixel ≥500
3. 1984–2004년 완결연도 중 canopy-positive 연도 ≥10
4. 중복 좌표 없음, 유효 중심좌표, 연간 관측완전성 통과

현재 파일럿에서 이 기준은 165개 셀을 만들지만, 165는 품질검사 전 잠정치다. 표본 수를 유지하기 위해 기준을 낮추지 않는다.

이 고정 cohort의 estimand는 `1984–2004년에 이미 확립된 표층 캐노피 서식지`다. 북부의 변동성 큰 bull kelp나 이후 새로 관측된 서식지를 과도하게 제외할 가능성이 있으므로, 지역별로 전체 후보 대비 포함률과 제외사유를 반드시 공개한다. 보조 `rolling-available cohort`는 각 예측시점까지의 자료만으로 footprint를 다시 정의해 운영대상 민감도를 평가하되, 미래연도의 all-history footprint는 사용하지 않는다.

### 3.3 분석 모집단을 질문별로 분리

| 모집단 | 잠정 셀 수 | 사용 자료 | 답하는 질문 |
|---|---:|---|---|
| Full thermal domain | 165 | Kelpwatch + CRW, OISST sensitivity | RQ1, RQ2, RQ4–RQ6 |
| U.S. upwelling domain | 107 | Kelpwatch + CRW + CUTI/BEUTI | RQ3 및 보조 비교 |
| Recent California measurement audit | 별도 산정 | Landsat + Planet 3 m, 2017–2024 | 결과 측정오차 탐색만 |

`CUTI/BEUTI`의 단순 위도수치 범위에 들어오더라도 공식 설명의 U.S. West Coast 적용범위를 벗어난 멕시코 셀은 주 용승 분석에서 제외한다. Washington의 47°N 북쪽 셀도 경계값을 복제하지 않는다.

### 3.4 보류 또는 보조 자료

- Planet 3 m: 2017–2024 California만 제공되므로 20년 주 백테스트를 대체할 수 없다. 최근 Landsat 결과 측정의 일치도 점검에만 사용한다.
- bathymetry/habitat: 정적 공간맥락이며 다음 해 변화 신호가 아니다. 공간 이전 민감도에서만 사용한다.
- wave exposure: 선행연구상 중요한 누락변수다. ERA5 ocean-wave reanalysis를 전역 후보로 먼저 감사하고, California의 CDIP 근해 자료는 공간 support 점검에 사용한다. 전역 범위·기간·근해 정합성 문턱을 통과하면 잠금 보조분석, 실패하면 `failed_qc`로 남기고 열·용승에 한정해 결론낸다.
- CRW coral HotSpot/DHW: 임계값은 coral bleaching용이므로 kelp의 생태 임계값으로 직접 사용하지 않는다. 주 분석은 원 SST로부터 kelp-agnostic anomaly와 percentile exposure를 만든다.
- quarterly decline: 계절적 소실을 생태적 붕괴로 오인할 위험이 있어 주 논문 결과에서 제외한다.

## 4. 공간·시간 단위와 예측시점

### 4.1 공간단위

- 주 단위: EPSG:6933에 전 지구적으로 고정된 10,000 m × 10,000 m equal-area grid
- 셀 원점은 코드와 설정파일에 고정한다.
- 해안선과 격자원점에 따른 MAUP를 점검하기 위해 5 km, 20 km 및 10 km 반셀 이동격자를 민감도로 평가한다.
- 인접 셀 중복은 없지만 공간 자기상관은 존재할 수 있으므로 무작위 행 분할을 사용하지 않는다.

### 4.2 예측시점과 horizon

- 주 protocol: 12월 31일 `t` 이후 실행하는 연간 우선순위. 입력은 `t`년까지, 결과는 `t+1` annual maximum이다.
- 이것은 정확히 365일 앞 특정 날짜의 예측이 아니라 **next-season annual monitoring prioritization**이다.
- 보조 operational sensitivity: 9월 30일 cutoff를 두고 Q1–Q3와 9월까지 환경자료만 사용하는 조기 실행 가능성을 평가한다. 주 결과와 섞지 않는다.

### 4.3 백테스트

- 예측연도: 2005–2024, 총 20개 outer test year
- 각 fold 학습: 1989년부터 `t-1`까지 expanding window
- 라벨 결과연도: 2006–2025
- 전처리·결측대치·표준화·하이퍼파라미터 선택은 각 outer fold의 과거자료에서만 수행한다.
- 같은 연도의 모든 비교모형은 정확히 같은 적격 셀-연도를 평가한다.

## 5. Kelpwatch 관측과 결과변수

### 5.1 연간 canopy 구성

주 규칙은 Kelpwatch Report Card 선례와 정합되게 다음을 요구한다.

1. historical habitat의 50% 이상이 유효 관측된 quarter가 최소 3개
2. 그중 Q3가 반드시 포함
3. 통과한 quarter 중 최대 canopy area를 연간 값으로 사용

민감도 규칙:

- quarter valid fraction 75%, 최소 3 quarter
- Q3 필수조건 없음
- annual maximum 대신 valid-quarter mean 및 Q2–Q3 maximum

결과값 결측은 0으로 대치하지 않는다. 해당 셀-연도를 분석에서 제외하고 결측 원인과 지역·연도·센서별 패턴을 보고한다.

### 5.2 미래정보가 없는 reference

- all-history `count_cells_historic_footprint`는 셀 선택과 상대 canopy denominator에 사용하지 않는다.
- 고정 reference는 1984–2004 자료만으로 만든다.
- `relative_canopy_t = annual_area_t / pre2005_p95_annual_area`
- pre-2005 p95가 0이거나 안정적으로 추정되지 않는 셀은 제외한다.
- all-history footprint 기반 결과는 과거 연구와 비교하기 위한 민감도일 뿐이다.

### 5.3 주 결과

- 적격: `relative_canopy_t > 0.05`, current와 next annual area가 모두 관측됨
- 급감: `(area_t - area_t+1) / area_t >= 0.30`
- 적격하지 않은 행을 음성으로 재분류하지 않는다.

### 5.4 보조 결과와 민감도

- 20%, 40%, 50% 상대 감소
- `relative_canopy_t > 0.02`, `>0.10`
- 연속 결과: `log((area_t+1 + epsilon) / (area_t + epsilon))`
- pre-2005 cell-specific p25 아래로의 새로운 진입
- absolute-area loss 결과는 셀 크기·기존 bed size 영향이 크므로 보조로만 제시

### 5.5 관측오차 감사

- `passes`, `passes5`, `passes7`, `passes8`, `area_se`를 연도·지역·센서시대별로 요약
- 관측노력 변수를 주 예측변수에 넣지 않는다. 모델이 센서변화나 구름을 생태신호로 학습할 수 있기 때문이다.
- 대신 높은 관측품질 subset과 센서시대별 결과를 민감도로 평가한다.
- `area_se`는 픽셀 간 공분산과 annual-maximum 선택오차가 제공될 때만 inverse-variance weighting에 사용한다. 이 정보가 없으면 RSS(독립 가정)와 단순합(완전상관 상계)을 관측오차 진단으로만 보고하고 임의의 단일 가중치는 만들지 않는다.
- 각 지역에서 최대 5개, 최소 3개 셀을 층화추출하여 공식 Kelpwatch API 집계와 NetCDF 집계를 quarter별로 재검증한다.

실행 메모(2026-08-16): 기존 165셀 패널은 all-history habitat footprint, 75% 기준, 연 3분기, Q3 비필수 규칙으로 만들어져 위 주 규칙과 달랐다. 공식 NetCDF에서 pre-2005 양성 픽셀을 고정 footprint로 다시 집계한 `20260816_observation_label_robustness_v1`을 이후 주 분석의 canopy·label 입력으로 사용한다. 기존 구현은 전 행 정확 재현 후 민감도로만 유지한다. 웹/API 층화표본 대조는 아직 Gate 2 잔여 작업이다.

## 6. NOAA 환경자료와 특징

### 6.1 Full-domain 주 열 자료: CRW CoralTemp 5 km

선택 이유:

- 1985년부터 현재까지 일별·전 지구·0.05도 gap-free SST
- 기존 OISST보다 근해 10 km 셀과 공간 support가 가깝다.
- Kelpwatch report-card 계열에서도 5 km SST가 사용된다.

공간 정합:

- 셀 중심점 하나가 아니라, 10 km 셀 안 또는 pre-2005 kelp habitat 주변의 유효 ocean pixels를 집계한다.
- 각 셀-연도에 source pixel 수, 최근접거리, land/ocean mask 상태를 기록한다.
- 셀 안 유효점이 없을 때만 최대 15 km 이내 최근접 ocean pixel을 허용하고 `fallback=true`로 기록한다.
- 15 km를 넘으면 결측으로 두며 자동 외삽하지 않는다.

주 열 특징은 생태적 시기와 지속성을 함께 표현하도록 아래 5개로 제한한다. 모두 forecast year `t` 안의 자료이며 `t+1` 자료를 사용하지 않는다.

1. 1–3월 평균 SST anomaly
2. 4–6월 평균 SST anomaly
3. 7-day rolling anomaly의 연최대값
4. 양의 SST anomaly 누적 degree-days
5. pre-2005 local day-of-year 90th percentile 초과일수

climatology는 1985–2004로 고정한다. leap day, day-of-year smoothing, 최소 유효일수 규칙을 코드와 설정에 고정한다. 공식 CRW SSTA와 1991–2020 climatology는 민감도다.

### 6.2 OISST v2.1 0.25도

- 동일한 4개 열 특징을 가능한 한 같은 정의로 만든다.
- 주 역할은 기존 연구와의 연결 및 제품·해상도 민감도다.
- 최근접 유효 ocean point와 3×3 ocean-only neighborhood를 모두 보존한다.
- 매칭거리와 coastal support를 보고하며 OISST를 10 km 현장값으로 표현하지 않는다.

### 6.3 CUTI/BEUTI

- 공식 범위: daily, 1988–present, 31–47°N, 1° latitude bins, U.S. West Coast proxy
- 범위 안에서만 가장 가까운 위도 bin을 연결한다.
- 범위 밖 값 clamp, 선형 외삽, Baja/북부 Washington 복제는 금지한다.
- 주 특징은 April–September CUTI anomaly와 BEUTI anomaly, 그리고 각 lag-1로 제한한다.
- anomaly reference는 1988–2004로 고정한다.
- CUTI/BEUTI는 local in-situ nutrient 또는 cell-specific upwelling로 표현하지 않는다.

### 6.4 파랑 자료 적합성 gate

- 전역 후보: ECMWF ERA5 ocean-wave reanalysis의 significant wave height. 전역·장기 일관성이 장점이지만 근해 10 km 셀보다 거친 offshore proxy다.
- California 대조: CDIP MOP/hindcast의 근해 significant wave height. 더 지역적이지만 전체 West Coast/Baja의 동일기간 주 자료로 사용하지 않는다.
- 사전 특징: 직전 10–12월 및 직전 12개월의 maximum 7-day mean significant wave height, 상위 95-percentile 초과일수. 데이터 coverage 감사가 끝나기 전에는 특징을 추가하지 않는다.
- 통과문턱: 1984–2024 분석연도의 95% 이상, 최종 full-domain 셀의 90% 이상, 셀별 ocean-grid 거리 기록, 지역별 결측률 15% 미만. 문턱은 outcome 및 모델 성능을 보기 전에 판정한다.
- ERA5 값을 local breaking-wave height나 reef orbital velocity로 표현하지 않는다. CDIP 대조가 큰 체계적 불일치를 보이면 H3b는 탐색 또는 제외로 강등한다.

### 6.5 환경자료 품질문턱

- 날짜 중복 0건, 시간축 단조증가, 단위·결측코드 검증
- 완결연도 일수 365/366 확인
- 셀-연도 coverage ≥95%가 원칙; 미달은 원인별 표기
- CRW full-domain 셀의 90% 미만이 분석 가능하면 주 NOAA 실험을 중단하고 공간정합을 재설계
- CUTI/BEUTI domain에 60개 미만 또는 평가 가능한 연도 10개 미만이면 RQ3를 탐색 분석으로 강등
- 파랑 후보가 6.4 문턱을 통과하지 못하면 H3b를 실행하지 않으며, 사후에 더 유리한 파랑제품으로 교체하지 않는다.

## 7. 비교할 정보세트와 모형

### 7.1 모든 모집단의 공통 기준선

1. `train_prevalence`: 훈련자료 사건률
2. `region_prevalence`: 훈련자료 내 지역별 사건률; unseen region에는 전체 사건률
3. `recent_decline_rule`: 직전 1년 감소량
4. `current_only`: 현재 relative canopy
5. `current_plus_trajectory`: 현재 + 사전 고정 8개 궤적 특징

### 7.2 열 자료 비교

6. `crw_only`
7. `current_plus_trajectory_plus_crw`
8. `oisst_only`
9. `current_plus_trajectory_plus_oisst`

### 7.3 U.S. upwelling domain

10. `upwelling_only`
11. `current_plus_trajectory_plus_crw_plus_upwelling`

### 7.4 파랑 자료적합성 통과 시 full domain 보조세트

12. `wave_only`
13. `current_plus_trajectory_plus_crw_plus_wave`

모든 주 비교는 같은 적격행에서 paired comparison으로 수행한다. 환경자료 결측 때문에 비교 모형마다 표본이 달라지게 두지 않는다. full-domain과 upwelling-domain 결과는 서로 다른 estimand로 명시한다.

### 7.5 주 모형

- standardized ridge logistic regression
- 각 outer fold 안에서 median imputation과 scaling을 적합
- `C ∈ {0.01, 0.1, 1, 10}`을 inner expanding-window로 선택
- inner 목적함수는 macro within-year AP lift
- 동률이면 더 강한 regularization을 선택
- 클래스 가중치와 resampling은 사용하지 않는다.

랜덤 포리스트·XGBoost·LightGBM은 주 결과를 만드는 모델경쟁에 사용하지 않는다. 필요하면 방향성 강건성 보조표로만 두고, 테스트 성능을 보고 최적 알고리즘을 선택하지 않는다.

### 7.6 지리정보 처리

- 좌표와 region label은 주 feature에서 제외한다.
- region은 층화평가, bootstrap, 공간 holdout에 사용한다.
- region fixed effect를 포함한 모형은 within-domain calibration 민감도로만 평가한다.

## 8. 평가량과 통계추론

### 8.1 주 평가량

`macro within-year AP lift = mean_year(AP_year - prevalence_year)`

- 양 클래스가 모두 존재하는 연도만 AP를 계산한다.
- 제외된 연도도 N, 사건수, 사건률, Top-k 결과를 모두 보고한다.
- 모델 차이는 동일 셀-연도 예측을 짝지어 계산한다.

### 8.2 의사결정 평가량

- 매년 Top-20% recall: 주 예산지표
- Top-10%, 고정 5/10-cell 예산: 보조
- precision, false alerts, missed events
- oracle upper bound: 해당 연도 사건수와 예산으로 가능한 최대 recall
- 모든 동점은 score 내림차순 후 cell ID 오름차순으로 결정

### 8.3 보조 평가량

- pooled AP와 AP lift
- macro region-year AP lift
- Brier score, log loss
- calibration intercept/slope, reliability curve
- AUROC는 보충자료에만 배치
- 연속 결과에 대한 Spearman rank correlation 및 MAE

### 8.4 불확실성

- 주: 2-year moving-block bootstrap으로 연도를 재표집한 paired difference 95% interval
- 보조: year × cell two-way cluster bootstrap
- 2,000회, seed `20260813`
- bootstrap 유효반복 <95%이면 실패 원인을 기록하고 결과를 확증에 사용하지 않는다.
- p-value 하나로 결론내지 않고 효과크기, 구간, 연도별 방향, 예산효용을 함께 판정한다.

### 8.5 다중비교 통제

확증적 비교 순서는 다음처럼 고정한다.

1. H2a: CRW thermal increment — 주 비교
2. H3a: upwelling increment — 별도 지원범위 보조 비교
3. H5a: 각 비교의 Top-20% 효용 — decision co-check

H3b는 자료적합성 gate를 통과할 때만 잠금 보조비교다. H1, H2b, H4, H5b, H6은 재현·기전분해·강건성으로 분류한다. threshold × grid × product 조합에서 가장 좋은 값을 골라 확증 결과로 올리지 않는다. 필요하면 실행된 증분 비교의 명목 p-value에 Holm 보정을 병기하지만, 최종 claim gate는 위의 다면 기준을 따른다.

## 9. 누수·편향 방지 체크리스트

다음 중 하나라도 실패하면 모델 결과를 생성하지 않거나 `INVALID`로 표시한다.

- 셀 선정에 2005년 이후 canopy presence/footprint 사용 금지
- 상대 canopy reference에 2005년 이후 값 사용 금지
- outcome `t+1` 또는 그 변환값이 feature에 포함되지 않음
- 환경 feature는 12월 31일 `t` 이후 날짜를 포함하지 않음
- imputer/scaler/hyperparameter가 outer test year를 보지 않음
- 행 무작위 train/test split 없음
- test 결과로 feature·C·threshold·예산을 선택하지 않음
- 결측 outcome을 0 또는 non-event로 대치하지 않음
- NOAA 범위 밖 clamp/extrapolation 없음
- 비교모형 간 평가행 완전 동일
- 파일럿과 확증 결과를 같은 표에서 구분 없이 합치지 않음

추가 편향 감사를 수행한다.

- survivorship/selection bias: pre-2005 안정서식지에 한정된 target population 명시
- regression-to-the-mean: current-only와 연속 결과 비교
- observation effort: sensor era·passes·cloud coverage 민감도
- spatial autocorrelation: 지역 holdout과 위도 block
- species/province heterogeneity: 지역별 결과와 부호 보고
- omitted biotic regime: 2014–2016 이후 북부 California를 별도 시기/권역 진단으로 보고하되 주모형에서 임의 제외하지 않음
- label prevalence shift: 연도별 사건률과 training-to-test shift 보고

## 10. End-to-End 실행 단계와 통과 기준

### Gate 0 — 계획 잠금

산출물:

- 본 문서
- `configs/paper_e2e_v2.yaml`
- `experiments/registry.csv`
- 데이터 출처 registry

통과 기준: 연구질문, 비교, 모집단, 결과, 지표, claim gate가 승인되고 Git commit이 생성됨.

### Gate 1 — 원자료 provenance

작업:

- Kelpwatch, CRW, OISST, CUTI, BEUTI의 URL, dataset ID, snapshot date, version, license, checksum 기록
- EDI metadata 범위 불일치 확인
- raw 파일은 Git에 넣지 않고 manifest만 추적

통과 기준: 모든 입력이 고유 version/hash로 재식별 가능.

### Gate 2 — Kelpwatch 품질·모집단

작업:

- 좌표·중복·단위·fill value·분기·센서별 passes 감사
- pre-2005 기준만으로 cohort 재구성
- 지역별 셀 수, 관측연도, 결측률, event count 산출
- API 층화 대조

통과 기준: 치명적 불일치 0건, API 대조 허용오차 이내, 모집단 규칙이 미래정보를 사용하지 않음.

### Gate 3 — NOAA 정합·coverage

작업:

- CRW 165-cell 예상 전 범위 수집 및 coastal matching
- OISST 동일 feature 정의
- CUTI/BEUTI 지원범위 subset 구성
- ERA5 파랑 전역 적합성과 California CDIP 대조 가능성은 outcome과 모델 성능을 보기 전에 판정
- 매칭거리·source pixel 수·결측률·climatology 검증

통과 기준: 6.4의 품질문턱 통과. 실패하면 모델링 전에 수정하고 수정 이유를 log에 남김.

### Gate 4 — 동결 데이터셋

작업:

- cell-year feature table 생성
- feature dictionary와 단위 작성
- 입력 manifest 및 config hash 생성
- outcome 분포는 품질확인 목적으로만 보고, 모델별 결과는 아직 보지 않음

통과 기준: duplicate key 0, 금지 feature 0, 비교행 일치, 모든 시간 cutoff test 통과.

### Gate 5 — 백테스트

작업:

- 20개 expanding-window fold 실행
- fold별 train/test 기간·N·사건수·선택 C 기록
- 예측 원본을 append-only로 저장

통과 기준: 모든 fold의 `max(train_year) < test_year`, 예측 중복·결측 0.

### Gate 6 — 평가·불확실성

작업:

- pooled, macro-year, macro-region-year, Top-k, calibration
- paired year-block bootstrap
- year-score와 within-year-score 병렬 평가

통과 기준: metric recomputation test와 bootstrap validity 통과.

### Gate 7 — 강건성·이전성

작업:

- threshold, eligibility, fixed/rolling-available cohort, grid scale/origin, annual aggregation, product, climatology
- leave-one-region-out 및 latitude blocks
- observation-quality subset
- 최근 California Landsat–Planet 측정 일치도 탐색

통과 기준: 모든 사전 지정 조합을 결과에 상관없이 표에 남김.

### Gate 8 — claim gate

작업:

- 11절 시나리오 매트릭스로 결론 선택
- 허용/금지 문구 자동 생성
- 결과가 약하면 후속 모델을 임의 추가하지 않고 결론을 제한

통과 기준: 초록·결론의 모든 수치가 잠금 출력과 일치하고 과장 문구 없음.

### Gate 9 — 재현 패키지

작업:

- fresh environment에서 end-to-end 재실행
- notebook top-to-bottom 실행
- 표·그림을 원시 prediction에서 재생성
- Git branch, commit, 원격 push, draft PR

통과 기준: 동일 config/input hash에서 metric 허용오차 내 일치.

## 11. 가능한 결과 시나리오와 사전 대응

| 시나리오 | 관찰 패턴 | 허용되는 결론 | 다음 대응 | 금지되는 대응 |
|---|---|---|---|---|
| A. NOAA가 공간순위와 예산 모두 개선 | H2a CI 하한>0, 다수 연도 양수, Top-20% 비악화/개선 | 열 노출이 캐노피 이력 이후에도 decision-relevant value 제공 | 외부지역/새 연도 검증, calibration 강화 | 최고 AP만 골라 인과 주장 |
| B. pooled만 개선 | pooled ΔAP>0, macro within-year≈0 | NOAA는 bad-year detection에는 도움, 같은 해 hotspot 선택에는 제한 | year-level alert와 cell ranking의 2단계 시스템 제안 | hotspot 예측 성공이라고 표현 |
| C. within-year만 개선 | macro within-year>0, year-only 약함 | 광역 위험보다 같은 해 상대순위에 유용 | 지역별 운영예산 검증 | 전체 해안 위험예측으로 확대 |
| D. NOAA 완전 무효 | H2a≤0, Top-k 개선 없음 | 시험한 NOAA 열·용승 proxy는 canopy history를 넘어 안정적 증분가치 없음 | null result를 benchmark/evaluation 기여로 보고 | 미측정 파랑·영양염·성게까지 포함한 `환경 전체`가 무효라고 표현하거나 사후 양수 찾기 |
| E. CRW 양수, OISST 무효 | 5 km만 개선 | 근해 support/product choice가 중요할 가능성 | CRW 주, OISST sensitivity; source 차이 논의 | 해상도만이 원인이라고 단정 |
| F. OISST 양수, CRW 무효 | coarse 제품만 개선 | 제품·climatology 의존 결과 | 코드·매칭·climatology 재감사 후 독립 복제 | OISST가 생태적으로 우월하다고 즉시 주장 |
| G. CUTI/BEUTI만 양수 | U.S. subset H3a 양수 | 지원범위 내 용승 proxy의 추가 정보 | U.S. domain 제한 결론, 지역별 검증 | Baja/북부 Washington 외삽 |
| H. 지역별 부호 반대 | 평균은 작고 지역 이질성 큼 | 보편모형보다 생태권역별 차이가 핵심 | hierarchical/region-specific model을 별도 후속연구로 등록 | 결과 좋은 지역만 본문 선택 |
| I. 궤적만 개선 | H1b 양수, H2/H3a/H3b 무효 | canopy monitoring history가 주 신호 | 단순모형 중심, NOAA는 contextual | NOAA가 중요하다는 원래 기대에 맞춰 해석 왜곡 |
| J. current-only가 최상 | current가 trajectory/env 이상 | 성능 대부분 상태 지속성·regression-to-mean 가능 | continuous outcome/transition 결과로 진짜 경보 여부 점검 | 조기경보 완성 주장 |
| K. 어떤 모델도 baseline을 못 이김 | macro lift≈0, Top-k≈oracle 대비 낮음 | 현재 자료로 현장 우선순위는 지원되지 않음 | 실패조건과 필요한 새 자료를 명확히 제시 | 모델만 복잡하게 바꿔 최고점 탐색 |
| L. 한두 해 의존 | leave-one-year-out에서 효과 소실 | 사건 clustering에 민감한 후향적 신호 | 주장 축소, 추가 연도 축적 | pooled N을 독립표본 수처럼 제시 |
| M. threshold/grid에 민감 | 부호 반복 변경 | 결론이 운영정의·공간단위 의존 | multiverse 결과 전체 보고, 강한 주장 차단 | 가장 좋은 threshold/grid만 채택 |
| N. 공간 holdout 실패 | 내부 양수, unseen region≤0 | 관측지역 내부 ranking에 한정 | 새 지역 데이터로 재훈련/보정 | West Coast transferable 표현 |
| O. ranking 양호, calibration 불량 | AP 양수, Brier/slope 불량 | 순위 보조도구 가능, 절대확률 불가 | training-only recalibration 후 독립 검증 | risk probability라고 표기 |
| P. Landsat–Planet 불일치 | 최근 CA outcome 일치 낮음 | 결과 측정오차가 성능 한계의 일부 | measurement-error model/고해상도 검증 | Landsat label을 ground truth로 표현 |
| Q. 열은 시기별 유효, 체제전환 후 붕괴 | 이전기간/일부 권역 양수, 2014년 이후 북부 CA에서 실패 | 환경 신호의 작동범위가 생물학적 체제에 의존 | pre/post-2014와 권역별 오차를 진단하고 상호작용은 후속등록 | 실패기간을 제외해 평균성능을 부풀림 |
| R. pre-season만 유효 | 12월 next-year H2a 무효, 3월 H2c 양수 | 환경 갱신은 짧은 lead time의 여름 조사계획에만 유용 | 운영시점을 3월로 제한해 별도 검증 | 더 긴 next-year 경보가 성공한 것처럼 표현 |

결과가 기대와 반대여도 해당 시나리오의 결론을 사용한다. 새로운 분석을 추가하려면 별도 experiment ID와 `exploratory` 상태를 먼저 registry에 등록하고, 확증 결과와 분리한다.

## 12. 중단·수정 규칙

다음은 즉시 중단 사유다.

1. 원자료 hash/version을 재식별할 수 없음
2. NetCDF와 공식 API 층화대조에서 설명되지 않는 체계적 불일치
3. 미래 footprint/reference 사용이 남아 있음
4. NOAA 범위 밖 값의 clamp 또는 암묵적 외삽
5. outer test가 전처리나 tuning에 사용됨
6. 동일 비교의 평가행이 다름
7. metric 계산을 원시 prediction에서 재현할 수 없음

수정은 허용하되 다음을 지킨다.

- QC 실패를 고치는 수정은 가능하며 commit과 change log를 남긴다.
- 결과 개선을 목적으로 feature·모형·표본을 변경하지 않는다.
- 잠금 이후 설계 변경은 protocol version을 올리고 기존 결과를 폐기하지 않는다.
- 심각한 measurement mismatch가 있으면 분석을 계속하지 않고 데이터 단계로 되돌아간다.

## 13. 실험 로그와 Git 규칙

### 13.1 실험 ID

- 형식: `E2E-v2-E##`
- 각 실험은 실행 전에 registry에 한 줄을 만든다.
- 상태: `planned`, `running`, `passed_qc`, `failed_qc`, `complete`, `exploratory`, `invalidated`

### 13.2 각 run에 반드시 남길 항목

- run ID, experiment ID, UTC/KST start/end
- Git commit SHA와 dirty 여부
- config path와 SHA-256
- 모든 입력의 path/URL/version/SHA-256
- 실행 명령과 Python/package versions
- random seed와 fold 정의
- stdout/stderr log
- QC 결과
- row/cell/year/event counts
- raw predictions, fold metrics, aggregate metrics, bootstrap draws
- 성공/실패/중단 이유 및 claim-gate 판정

### 13.3 파일구조

```text
configs/paper_e2e_v2.yaml
experiments/registry.csv
outputs/experiments/<run_id>/
  manifest.json
  command.txt
  run.log
  quality_checks.csv
  fold_audit.csv
  predictions.parquet
  metrics.csv
  bootstrap_differences.csv
  decision.json
```

run 디렉터리는 덮어쓰지 않는다. 재실행은 새 run ID를 만든다. raw NetCDF는 `.gitignore`에 두고 checksum manifest만 Git에 포함한다. 큰 prediction 파일이 Git 크기정책을 넘으면 release artifact 또는 별도 데이터 저장소에 두고 immutable checksum을 commit한다.

### 13.4 Git 순서

1. 계획 브랜치에서 protocol/config/registry만 먼저 commit
2. 사용자 검토 후 plan tag 생성
3. 데이터 단계, feature 단계, model 단계, evaluation 단계를 분리 commit
4. 각 commit에서 관련 테스트 실행
5. 원격 branch push
6. 최종에는 draft PR에서 protocol deviation과 모든 실패실험을 함께 공개

## 14. 논문 본문과 보충자료의 사전 구조

### 본문

1. 연구문제와 target population
2. 데이터 범위와 질문별 모집단 분리
3. expanding-window와 주 NOAA 증분가치 비교
4. pooled 대 within-year 평가
5. Top-20% 조사예산 결과
6. claim-gate에 따른 결론

### 본문 표·그림

- Figure 1: 전 서해안 cohort 및 NOAA 지원범위
- Figure 2: 연도별 적격 N·사건률·관측완전성
- Table 1: 데이터 출처·공간/시간 support·역할
- Table 2: paired incremental-value 결과
- Figure 3: pooled/year-only/within-year 성능 비교
- Figure 4: Top-k budget curve와 oracle bound
- Figure 5: 지역 holdout 및 이질성

### 보충자료

- 모든 연도·지역 fold 결과
- grid/threshold/climatology/aggregation multiverse
- 관측노력 및 sensor-era audit
- OISST/CRW matching distance와 coverage
- CUTI/BEUTI 지원범위 및 제외셀
- ERA5 파랑 coverage gate와 CDIP 근해 support 대조 결과
- year-end next-year와 March pre-season clock의 분리 결과
- calibration과 raw prediction 재현정보
- 실패·무효화 실험 registry

## 15. 실행 전 최종 승인 체크

다음 질문에 모두 `예`일 때만 NOAA 수집과 후속 모델링을 시작한다.

- RQ2의 NOAA 증분가치가 주 연구질문이라는 데 동의하는가?
- West Coast full thermal과 U.S. upwelling을 별도 모집단으로 보고하는가?
- 165개는 잠정치이며 품질검사 후 줄어들 수 있음을 수용하는가?
- CRW 5 km를 주 열 자료, OISST를 sensitivity로 두는가?
- 파랑은 중요한 누락축으로 인정하되 outcome 이전의 coverage gate를 통과할 때만 잠금 보조분석으로 실행하는가?
- year-end next-year 질문과 March pre-season 민감도를 서로 다른 estimand로 유지하는가?
- 30% 급감과 0.05 적격기준을 주 규칙으로 유지하고 전체 민감도를 공개하는가?
- null/negative/heterogeneous 결과도 11절에 따라 그대로 논문화하는가?
- 계획 잠금 이후 결과를 보고 변수·모형을 바꾸지 않는가?

승인 전에는 후속 NOAA 다운로드와 확증 모델 실행을 하지 않는다.
