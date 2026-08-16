# 환경변수와 연구질문 타당성 검토 v2

## 0. 결론

**환경변수를 넣는 것은 타당하고, 연구질문도 논문화할 수 있다.** 단, 논문의 질문은 이미 알려진 `수온·영양염·파랑이 kelp에 영향을 주는가`가 되어서는 안 된다. 그 질문에는 기존 연구가 상당 부분 답했다.

이 연구가 실제로 검증할 질문은 다음과 같다.

> 의도적으로 확대한 북동태평양 서해안 범위에서, 예측시점 이전의 열 노출과 용승 지표가 현재 캐노피와 최근 궤적을 넘어 다음 해 급감 셀의 **연도 내 우선순위**에 추가가치를 제공하며, 그 가치가 미래 연도와 생태권역으로 이전되는가?

타당성 판정은 다음과 같다.

| 타당성 차원 | 판정 | 이유 |
|---|---|---|
| 생태학적 전제 | 높음 | 수온, nutrient/upwelling, 파랑, 성게가 kelp 변동의 반복된 설명변수다. |
| 예측 질문 | 높음 | 기존 설명·복원우선순위 연구와 달리, canopy-history 이후 증분가치와 미래연도 순위성능을 묻는다. |
| 환경변수 구성타당도 | 중간 | SST와 CUTI/BEUTI는 광역 proxy이며 현장 수온·질산염·미세 용승 자체가 아니다. |
| 확대지역 일반화 | 검증대상 | giant/bull kelp, 생태권역, 파랑 적응, 생물압이 달라 평균효과를 전제할 수 없다. |
| 인과 해석 | 낮음/금지 | 관측자료 예측설계이며 SST는 열 스트레스와 nutrient availability를 동시에 대리할 수 있다. |
| 운영 타당도 | 아직 미확인 | within-year ranking, Top-20% budget, calibration, spatial holdout을 통과해야 한다. |

이 문서는 2026-08-13까지 확인한 핵심 1차 연구와 공식 자료문서를 대상으로 한 **표적 선행연구 검토**다. 체계적 문헌고찰이나 메타분석으로 표현하지 않는다.

## 1. 선행연구가 이미 답한 것

### 1.1 수온과 nutrient/upwelling은 넣을 근거가 충분하다

[García-Reyes et al. (2022)](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0267737)은 북부 California의 1991–2020 bull-kelp canopy를 분석했다. 겨울 해양조건으로 여름 canopy를 예측했고, 1991–2013 구간에서는 겨울 SST가 가장 강한 예측변수였으며 Point Arena 북쪽 변동의 최대 87%, 남쪽의 57%를 설명했다. 일부 남부 구간에서는 봄 BEUTI도 유의했다. 따라서 SST와 BEUTI를 후보에서 빼는 것보다, 시기와 공간 support를 명시해 넣는 편이 문헌에 부합한다.

그러나 이 논문은 동시에 중요한 경고를 준다. 모형은 2014년 붕괴를 예측했지만 2015년 이후 canopy를 지속적으로 과대예측했다. 해양조건만으로는 성게 grazing이 지배하는 새로운 생태상태를 설명하지 못했기 때문이다. 또한 관측된 겨울 SST는 직접 열사 임계값에 도달하지 않아, SST가 nutrient availability의 proxy일 가능성도 논의됐다. 따라서 본 연구의 SST 계수는 열의 인과효과로 해석할 수 없다.

[Cavanaugh et al. (2011)](https://www.int-res.com/abstracts/meps/v429/meps09141)은 Santa Barbara Channel giant kelp에서 겨울 손실은 significant wave height와, 봄 회복은 SST와 연관됨을 보였다. SST는 이 지역에서 nutrient availability의 proxy로 해석됐다. 동일한 SST라도 `열 손상`과 `저영양 상태`를 분리하지 못한다는 점을 다시 확인한다.

[Leichter et al. (2023)](https://par.nsf.gov/biblio/10480055-persistence-southern-california-giant-kelp-beds-alongshore-variation-nutrient-exposure-driven-seasonal-upwelling-internal-waves)은 southern California의 nutrient exposure가 계절 용승뿐 아니라 internal wave에 의해 국지적으로 달라짐을 보였다. 따라서 1° 위도 bin의 CUTI/BEUTI는 regional forcing proxy로는 타당하지만 10 km 셀의 현장 nutrient 측정값으로 부를 수 없다.

### 1.2 파랑을 무시한 채 `환경 전체`를 말하면 안 된다

[Bell et al. (2015)](https://www.tomwbell.net/uploads/5/6/9/7/56976837/bell_etal_2015_jbg.pdf)은 약 1,500 km California 해안, 723개 500 m 구간, 25년 자료에서 giant-kelp biomass의 지리적 제어요인을 비교했다. 이전 점유상태를 포함한 뒤에도 파랑 교란의 상대적 중요도가 가장 컸고, nitrate availability와 광역 기후지수가 뒤를 이었다. 효과는 공간적으로 달랐다.

[Giraldo-Ospina et al. (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11831097/)은 1,350 km California 현장자료에서 bull/giant kelp 밀도를 분석했다. 중요 변수는 지역과 종마다 달랐지만 nitrate, wave energy/exposure, purple urchin density, temperature의 조합이 반복해서 선택됐다. bull kelp와 giant kelp의 지역별 모형 설명력도 달랐다. 이것은 파랑을 보조분석에 포함할 이유와, 확대지역에서 단일 평균효과를 전제하지 말아야 할 이유를 동시에 제공한다.

[Cavanaugh et al. (2026)](https://www.nature.com/articles/s43247-025-03134-y)은 2017–2024 Planet 3 m California canopy persistence를 역사적 지속성, 지형, 봄 CRW 5 km SST anomaly, 겨울 파랑과 함께 분석했다. CRW 5 km 사용은 직접적인 선례가 있지만, 열·파랑 효과가 위도 및 우점종에 따라 달랐다. bull kelp는 강한 파랑환경에 적응해 파랑효과가 상대적으로 약할 수 있다고 설명했다.

따라서 본 연구의 주 질문은 `열·용승 proxy의 추가가치`로 정확히 제한한다. 파랑은 전역 장기자료의 근해 support가 사전 품질문턱을 통과할 때만 잠금 보조분석으로 넣는다. 파랑을 넣지 못했거나 음성이라면 `환경변수가 무효`라는 문장은 금지한다.

### 1.3 생물학적 체제전환은 예측 실패의 경쟁 설명이다

[McPherson et al. (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7935997/)은 북부 California의 350 km 해안에서 2014년 이후 bull-kelp 90% 이상 손실이 해양열파와 포식자 손실, 성게 증가가 결합된 지속적 상태전환이었음을 보였다. 단일 해양물리 변수로는 이 과정을 충분히 설명할 수 없다.

[Cavanaugh et al. (2025)](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/1365-2745.70072)은 북부 California refugia에서 초기에는 국지적 냉수조건이 중요했지만 2019년 이후에는 성게 포식을 피하는 능력이 더 중요해졌다고 보고했다. 변수의 중요도가 교란 순서와 생태상태에 따라 바뀔 수 있다는 직접 근거다.

본 연구에는 West Coast 전역·20년을 동일하게 덮는 성게 또는 포식자 자료가 없다. 이 변수를 억지로 대치하지 않는다. 대신 2014–2016 이후 북부 California의 오차와 성능붕괴를 사전 진단으로 고정하고, 실패하면 `미관측 생물압이 존재하는 체제에서 환경모형이 이전되지 않았다`고 결론낸다.

## 2. 확대지역이 연구질문을 강하게 만드는 조건

[Bell et al. (2023), Kelpwatch](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0271477)은 Landsat 기반 canopy 기록을 10 × 10 km 단위로 집계해 2014–2016 해양열파 이후 회복이 위도·지역에 따라 크게 달랐음을 보였다. [Kelpwatch 공식 방법론](https://kelpwatch.org/methodology)도 현재 자료가 North America West Coast의 광범위한 지역을 포함하고, Landsat canopy가 giant kelp와 bull kelp를 구분하지 못함을 명시한다.

따라서 지역 확대 자체는 타당하다. 오히려 기존 50개 Northern/Central California 셀보다 외적 타당도를 직접 시험할 수 있다. 다만 다음 조건을 지켜야 한다.

1. `표본을 50개에서 165개로 늘렸다`가 아니라 목표 모집단을 의도적으로 바꿨다고 쓴다.
2. 전체 평균 외에 생태권역별 N, 사건률, AP lift, Top-k, calibration을 모두 보고한다.
3. Landsat가 종을 구분하지 못하므로 종별 효과를 주장하지 않는다. 지역/문헌상 우점종 차이는 이질성의 가능한 설명으로만 사용한다.
4. CUTI/BEUTI는 공식 U.S. West Coast 31–47°N 범위 안에서만 쓴다. Baja와 47°N 북쪽에는 경계값을 복제하지 않는다.
5. full-domain 결론은 CRW/OISST처럼 실제 전역 coverage가 있는 자료에만 적용한다.
6. leave-one-region-out이 실패하면 `West Coast transferable model` 주장을 철회한다.

확대지역은 단순히 검정력을 높이는 장치가 아니다. **환경효과가 지역·종·교란체제에 따라 달라진다는 선행연구를 정면으로 시험하는 설계**여야 한다.

## 3. 기존 연구와 겹치지 않는 지점

[A site selection decision framework for effective kelp restoration (2025)](https://www.sciencedirect.com/science/article/pii/S0006320725000175)은 California에서 현장 생물·환경모형, 역사적 안정성, 해양열파 후 손실, 최근상태를 결합해 복원 대상지를 분류했다. 그러므로 `환경과 canopy 자료를 합쳐 복원 우선순위를 만든다`만으로는 새롭지 않다.

본 연구가 유지해야 할 차별점은 아래 네 가지다.

1. **예측시점 고정:** 과거 전체를 설명하는 것이 아니라 각 연도에 당시 사용 가능했던 자료만 쓴다.
2. **증분가치:** 환경모형의 절대점수가 아니라 current canopy + trajectory라는 강한 기준선 이후의 추가가치를 묻는다.
3. **결정지표:** pooled discrimination뿐 아니라 같은 해 Top-20% 조사예산에서 놓치지 않는 급감 셀을 측정한다.
4. **이전성:** 2005–2024 미래연도 backtest와 공간 holdout으로 새 연도·새 권역 성능을 분리한다.

즉 novelty는 새 환경변수나 복잡한 알고리즘이 아니라 **누수 없는 prospective-style evaluation과 decision relevance**다.

## 4. 변수별 최종 판정

| 변수군 | 넣는가 | 역할 | 해석 제한 |
|---|---|---|---|
| current canopy | 예 | 강한 기준선 | 낮은 현재값에서 생기는 regression-to-the-mean 점검 |
| recent trajectory | 예 | 강한 기준선 | 이미 본 165-cell pilot은 독립 확증이 아님 |
| CRW CoralTemp 5 km SST | 예, 주 환경자료 | full-domain 열 노출 | coastal proxy; coral HotSpot/DHW 임계값은 쓰지 않음 |
| OISST 0.25° | 예, 민감도 | 제품·공간 support 비교 | 10 km local SST로 표현 금지 |
| CUTI/BEUTI | 예, 지원범위 보조 | regional upwelling/nitrate-flux proxy | Baja/북부 Washington 외삽 금지 |
| 파랑 | 조건부 예 | 중요한 누락 환경축 | ERA5는 offshore coarse proxy; QC 실패 시 제외 |
| bathymetry/reef habitat | 보조 | 정적 공간문맥·이전성 진단 | 다음 해 조기신호가 아님 |
| 성게/포식자 | 주모형에는 불가 | 실패·체제전환 해석 | 균일한 전역 장기자료가 없으면 임의 proxy 금지 |
| 위도/region | 주 feature에서 제외 | 층화·holdout·민감도 | 위치를 외워 성능을 내는 모형 방지 |
| Landsat passes/cloud | 예측변수 제외 | 측정품질 감사 | 생태신호로 학습시키지 않음 |

### 4.1 CRW 특징의 시기

연평균 하나만 넣으면 문헌의 겨울·봄 기전을 희석할 수 있다. 주 clock에서는 예측연도 `t`의 다음 다섯 특징을 고정한다.

- 1–3월 평균 local SST anomaly
- 4–6월 평균 local SST anomaly
- 연중 maximum 7-day SST anomaly
- 연중 positive anomaly degree-days
- pre-2005 day-of-year 90th percentile 초과일수

이 특징은 `t+1` 캐노피를 예측하는 **선행 노출**이며, 직접적인 생리기전을 식별하지 않는다. García-Reyes et al.의 같은 해 winter→summer 관계와 본 연구의 year-end→next-year 관계는 lead time이 다르다. 그래서 3월 말까지의 환경자료로 그해 Q2–Q3 canopy를 예측하는 pre-season clock을 별도 잠금 민감도로 둔다. 주 next-year 결과가 실패하고 짧은 clock만 성공하면 운영 주장을 짧은 horizon으로 제한한다.

### 4.2 파랑 자료 선택

[ECMWF 공식 ERA5 문서](https://www.ecmwf.int/en/forecasts/datasets/browse-reanalysis-datasets)는 1940년부터 현재까지 전역 ocean-wave reanalysis를 제공한다. 장기 전역 일관성 때문에 확대지역 후보가 될 수 있지만 10 km 근해·섬의 실제 breaking wave 또는 reef orbital velocity가 아니다.

[NOAA NCEI WAVEWATCH III hindcast metadata](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc%3ANCEP-WAVEWATCH)는 확인 가능한 archive 범위가 2005–2019이며, coarse grid가 local coastal application에 제한적일 수 있음을 명시한다. 1984–2024 주 분석의 단일 원자료로는 부적합하다. [CDIP California wave model](https://www.cdip.ucsd.edu/m/documents/models.html)은 근해 대조에는 유용하지만 West Coast/Baja 전역의 동질적 주 자료가 아니다.

따라서 outcome이나 모델성능을 보기 전에 ERA5 coverage를 검사하고, 통과할 때만 파랑 보조가설을 실행한다. 이 결정은 데이터 support에 의한 것이며 결과선택이 아니다.

## 5. 연구질문과 가설의 최종 구조

### 주 질문

`H2a`: full-domain에서 `current + trajectory + CRW thermal`이 `current + trajectory`보다 macro within-year AP lift와 Top-20% recall을 개선하는가?

### 보조 질문

- `H2b`: CRW 결과가 OISST 제품 대안에서도 방향상 유지되는가?
- `H2c`: year-end next-year signal이 약할 때, 문헌과 시기가 맞는 March pre-season forecast에서는 열 정보가 유효한가?
- `H3a`: 공식 U.S. 31–47°N subset에서 CUTI/BEUTI가 CRW 이후에도 추가가치를 주는가?
- `H3b`: 사전 자료품질 gate를 통과한 파랑 proxy가 CRW 이후 추가가치를 주며, 그 효과가 권역에 따라 다른가?
- `H4`: 개선이 bad year 탐지인가, 같은 해 위험 셀의 공간순위 개선인가?
- `H5`: 제한예산과 미관측 권역에서도 개선이 유지되는가?
- `H6`: 결론이 라벨, 격자, 관측완전성, 제품 선택에 의존하는가?

### 예상 가능한 결과와 과학적으로 유효한 결론

- 열·용승이 양수: canopy history에 없는 pre-forecast 정보가 있음을 지지한다. 인과효과는 아니다.
- pooled만 양수: 광역 bad-year alert에는 유용하지만 같은 해 현장대상 선택에는 유용하지 않다.
- 일부 권역만 양수: 보편모형이 아니라 생태권역별 조건부 신호가 결론이다.
- 2014년 이후 북부 California에서 실패: 미관측 grazing regime이 환경-only 이전성을 제한한다는 문헌과 정합된다.
- year-end는 무효, March pre-season만 양수: next-year 경보가 아니라 짧은 계절예보로 범위를 좁힌다.
- 전부 무효: 시험한 원격 열·용승 proxy가 canopy history를 넘어 운영상 추가가치를 주지 않는다는 정직한 benchmark 결과다. 연구질문 자체가 무효였다는 뜻은 아니다.

상세 claim gate와 각 결과에 대한 대응은 `docs/paper_e2e_protocol_v2_ko.md` 11절에 고정한다.

## 6. 논문에서 사용할 수 있는 한 문장

> Previous studies establish temperature, nutrient supply, wave disturbance, and grazing as important but regionally variable controls of canopy-forming kelps; we therefore test not whether these drivers matter ecologically, but whether remotely sensed pre-forecast thermal and upwelling exposures provide incremental, out-of-time spatial prioritization value beyond canopy history across an intentionally expanded West Coast domain.

이 문장은 환경변수 사용의 근거, 기존 연구와의 차이, 확대지역의 역할을 동시에 정확히 표현한다.

## 7. 최종 판단

환경변수를 빼면 연구는 `캐노피 자기회귀/궤적 benchmark`로 축소되고, 원래 질문인 NOAA의 추가가치를 답하지 못한다. 반대로 환경변수를 무차별적으로 많이 넣으면 표본 수에 비해 feature가 늘고, 지역·종 차이와 누락 생물압을 환경효과로 오해할 위험이 커진다.

따라서 최종 설계는 다음 원칙으로 충분히 방어 가능하다.

1. CRW 열 노출을 full-domain 주 환경가설로 둔다.
2. OISST는 제품 민감도, CUTI/BEUTI는 공식 지원범위의 별도 estimand로 둔다.
3. 파랑은 중요한 누락축으로 인정하고, outcome을 보기 전 자료품질 gate를 통과할 때만 잠금 보조분석으로 실행한다.
4. 성게 등 생물압은 없는 자료를 꾸며 넣지 않고 체제전환 실패분석과 limitation으로 다룬다.
5. 결론은 pooled AP가 아니라 macro within-year AP lift, Top-20% recall, 미래연도·공간 holdout으로 판정한다.
6. 확대지역의 평균효과보다 권역별 부호와 이전 실패를 동등하게 보고한다.

이 조건을 지키면 연구질문은 유효하다. 더 정확히 말하면, **환경의 생태학적 중요성은 전제가 아니라 선행근거이고, 환경자료의 추가적인 운영 예측가치는 아직 답이 없는 검증대상**이다.
