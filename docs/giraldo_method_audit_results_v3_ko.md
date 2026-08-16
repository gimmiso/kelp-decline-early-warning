# Giraldo 보조 사례연구 방법론 감사 및 재분석 결과 v3

## 한 줄 판정

기존 v2의 `안정 조건 0/20`은 종·지역 통합, 선형성, 성게 변수 정의, 수심 변수 및 논문 선택 변수의 불일치 때문에 지나치게 강한 음성결론이었다. Giraldo 논문에 맞춰 다시 분석하자 AP 증분은 대체로 양의 방향으로 바뀌었지만, 신뢰구간·2021년 제외·top-20% recall·Brier score를 함께 보면 아직 안정적이고 운영 가능한 추가가치로 판정할 수 없다.

## 1. 논문과 기존 보조설계의 차이

Giraldo-Ospina et al. (2025)은 다음 해 이진 급감위험을 예측한 논문이 아니다. 현장 transect의 같은 해 bull/giant kelp 밀도를 Tweedie GAM으로 설명하고, site-zone과 survey year의 random effect를 포함했으며, 생태지역과 종별로 모형을 분리했다. 연속변수는 `k=4` smooth로 적합했고 AIC 기반 full-subset selection을 수행했다.

기존 v2와의 중요한 불일치는 다음과 같다.

1. bull kelp, giant kelp 및 생태지역을 하나의 평균관계로 통합했다.
2. 논문이 사용한 purple urchin 대신 red+purple urchin 합을 사용했다.
3. giant/bull kelp에서 서로 다른 성게 비선형 임계를 단일 선형계수로 검사했다.
4. PISCO 현장 수심 `depth_mean` 대신 지도형 `mean_depth`를 사용했다.
5. 논문 최종모형의 지역별 수온·질산염·파랑·orbital velocity·NPP·전년도 포자량을 반영하지 않았다.
6. 2014년 해양열파 이후와 2019년 이후 grazer-dominated 상태의 관계 변화를 검사하지 않았다.
7. Giraldo의 설명대상과 달리 우리 반응변수는 다음 해 위성 캐노피 30% 급감이다. 그러므로 논문에 맞춘 뒤에도 무효일 수 있으며 이는 원 논문을 반박하지 않는다.

공개 재현코드에도 주의점이 있다. 공개 스크립트는 75/25 무작위 분할로 `train.gam`과 `test.gam`을 만들지만, 확인한 full-subset 스크립트 범위에서는 `test.gam`이 후속 성능평가에 연결되는 부분이 보이지 않았다. 본 재분석은 이 방식을 따르지 않고 미래연도 expanding-window를 유지했다.

## 2. 데이터 품질 감사

- 공식 원자료: 9,728 transect 행, 191 sites, 1999–2021년, 144개 변수.
- 저자 메타데이터의 `preMHW >= 3` 조건을 통과한 곳: 126 sites.
- 위성 canopy 종 문맥과 중복을 제거한 고정 코호트: 북부 bull 10, central-southwest giant 88, southeast giant 26 sites.
- 300 m 분석 패널: 865 site-years, 111 sites. 실제 expanding prediction에 들어간 동일표본은 641 site-years.
- 1 km 분석 패널: 1,109 site-years, 123 sites. 실제 expanding prediction에 들어간 동일표본은 831 site-years.
- `(site, year, zone, transect)` 중복키: 175개 키, 350행. 이 중 68개 키는 `prev_year_spores` 값이 달랐다. 중복행이 site-year 평균에 이중 가중되지 않도록 먼저 고유 transect 단위 median으로 축약했다.
- 북부 Reef Check 10개 sites는 공개 CSV에서 `depth_mean`이 100% 결측이다. 북부에만 `-mean_depth`를 지도형 대체값으로 사용하고 `depth_mapped_fallback=1`로 기록했다.

마지막 두 항목 때문에 Giraldo 공개 코드와 데이터만으로 “완전한 원 분석 재현”이라고 표현해서는 안 된다.

## 3. 재분석 설계

다섯 정보블록을 같은 site-year에서 순차 비교했다.

1. KelpWatch 현재 canopy와 과거 궤적
2. 현장 종별 kelp, purple urchin, 종별 성게 임계초과 비율, active-grazing proxy, 현장/지도 수심, 암반, VRM, 과거 현장 kelp
3. 논문이 지역별로 선택한 수온·질산염·파랑·orbital velocity·NPP
4. 현장 생태상태+논문 환경+giant kelp 전년도 포자량
5. 2014/2019 체제변화 상호작용

모형은 linear logistic, 4-knot spline logistic, Random Forest, XGBoost를 사용했다. 2004년부터 예측 직전 연도까지만 학습하고 2008–2021년을 순차 예측했다. 주 평가는 300 m spline logistic의 `paper full - trajectory` macro within-year AP-lift 증분이며, 연도와 겹침 공간군집을 함께 재표집한 2,000회 계층 bootstrap을 사용했다.

## 4. 결과

### 4.1 논문 정렬 후 신호는 양의 방향으로 이동했다

`paper full - trajectory` AP-lift 증분은 다음과 같았다.

| support | linear logistic | spline logistic | Random Forest | XGBoost |
|---|---:|---:|---:|---:|
| 300 m | +0.0476 | +0.0378 | +0.0419 | +0.0716 |
| 1 km | +0.0346 | +0.0195 | +0.0146 | +0.0404 |

300 m에서 네 모형이 모두 양의 방향이었고 1 km도 모두 양의 방향이었다. 따라서 기존 v2의 선형·통합 설계가 신호를 일부 희석했을 가능성은 실제 결과로 확인됐다.

### 4.2 그러나 안정성 기준은 통과하지 못했다

주모형 300 m spline logistic의 증분은 `+0.0378`, 계층 bootstrap 95% CI는 `[-0.0572, +0.1459]`였다. 신뢰구간이 0을 포함하므로 안정적인 추가가치로 판정하지 않았다.

환경블록만 추가한 300 m 결과도 `+0.0333`, 95% CI `[-0.0687, +0.1510]`였다. 현장 생태상태를 환경블록에 더한 추가 증분은 `+0.0045`에 불과했다. 2014/2019 체제항을 더하면 spline logistic은 300 m `-0.0380`, 1 km `-0.0485`로 악화됐다.

### 4.3 2021년 의존성이 컸다

주모형의 연도별 AP 증분은 -0.1653에서 +0.5287까지 변했다. 2021년을 제외하면 평균 증분은 `+0.00004`, 계층 bootstrap 95% CI `[-0.0757, +0.0824]`로 사실상 0이었다.

다만 2021년 제외 후에도 300 m Random Forest `+0.0433`, XGBoost `+0.0671`이었고 1 km에서도 각각 `+0.0198`, `+0.0584`였다. 이는 복잡한 비선형·상호작용 가능성을 남기지만, spline/linear 및 의사결정 지표와 일치하지 않으므로 확증결과가 아니라 후속 가설이다.

### 4.4 종·지역 내부에서는 더 약했다

300 m spline logistic의 within-context AP 증분은 북부 bull `-0.2500`(평가 가능한 context-year 5개), central-southwest giant `+0.0022`(13개), southeast giant `+0.0298`(12개)였다. 전체 양의 증분 일부는 서로 다른 문맥의 확률을 같은 연도에 함께 순위화하면서 생긴 지역 간 분리에서 왔을 가능성이 있다.

### 4.5 AP 개선은 운영지표로 이어지지 않았다

주모형에서 paper-full 블록의 top-20% recall 변화는 `-0.0070`, Brier 개선은 `-0.0421`이었다. 즉 평균 AP 방향이 양수여도 실제 상위 20% 조사대상 포착률과 확률 calibration은 개선되지 않았다. Random Forest와 XGBoost에서는 300 m top-20% recall이 각각 `+0.0315`, `+0.0385`였지만 1 km에서는 `+0.0028`, `+0.0057`로 작았다.

## 5. 최신 문헌과 함께 본 해석

- Giraldo-Ospina et al. (2025)의 결과는 생태지역·종별 비선형 반응과 동시 현장자료의 중요성을 지지하지만, 다음 해 이진 급감 위험에 대한 외부검증 증거는 아니다.
- Smith et al. (2024)은 urchin density만이 아니라 kelp가 남아 있을 때 urchin의 active grazing 행동이 달라지는 비선형 관계를 강조한다. 이번 분석에서 단순 purple urchin보다 active-grazing proxy와 비선형 모형을 포함한 이유다.
- Cavanaugh et al. (2025)은 연속 교란 뒤에는 초기의 기후 refugia보다 이후 grazing protection이 더 중요해질 수 있음을 보여준다. 그러나 이번 데이터에서는 단순한 2014/2019 상호작용만으로 그 체제변화를 안정적으로 포착하지 못했다.
- Cavanaugh et al. (2026)은 3 m Planet 자료에서 사전 persistence, 냉수, 얕은 서식처, 낮은 fragmentation의 효과가 위도에 따라 달라짐을 보였다. 이는 300 m/1 km 평균 환경값을 더 복잡한 모델에 넣는 것만으로는 국지 persistence 기작을 복원하기 어렵다는 근거다.
- 시공간 생태자료의 무작위 교차검증은 공간상관 때문에 낙관적일 수 있다. 따라서 Giraldo의 내부 설명모형보다 본 연구의 expanding-window와 겹침군집 bootstrap을 유지하는 것이 연구질문에 더 적합하다.

## 6. 다음 발전 단계

### 현재 논문에 바로 사용할 것

이 v3는 California 보조 사례연구로 유지한다. 결론은 “방법론을 논문에 맞추면 양의 순위신호가 나타나지만 단일연도, calibration, 예산 포착률 및 지역 내부 강건성을 통과하지 못했다”이다. 기존 `0/20`보다 더 정확하고 논문에 방어 가능한 결론이다.

현장변수는 전체 해안에서 운영 가능한 입력으로 표현하지 않는다. 관측된 site-year에서 원격자료 실패조건을 설명하거나 가능한 성능의 상한을 진단하는 부분표본 자료다.

### 별도 생태모형으로 발전시킬 것

1. 2017–2024 Planet 3 m canopy polygon으로 species/region-specific persistence와 fragmentation label을 만든다.
2. giant와 bull kelp를 분리하고, 다음 해 30% 이진급감 하나가 아니라 persistence·재형성·fragmentation을 연속 또는 count 반응으로 분석한다.
3. 하나의 계층모형에서 region/species별 intercept와 nonlinear slope를 부분풀링한다. 별도 모형의 확률을 그대로 전 해안 순위에 섞는 calibration 문제를 피한다.
4. 공간차단 범위를 Appendix S1의 약 1.9–2.4 km 공간상관 범위 이상으로 고정하고, 연도 holdout과 region holdout을 함께 사용한다.
5. 2014/2019 binary interaction 대신 disturbance history, kelp baseline stability, active-grazing proxy, fragmentation의 사전 고정 상호작용만 검사한다.

이 단계는 이미 받은 표형 데이터의 모형만 바꾸는 작업보다 크다. 기존 v3 재실행은 약 6분이었지만, Planet polygon으로 새로운 label과 공간특징을 만드는 작업은 별도 데이터 엔지니어링과 검증이 필요하다.

## 참고한 원문·재현자료

- Giraldo-Ospina et al. (2025), *Ecological Applications*, DOI: [10.1002/eap.3092](https://doi.org/10.1002/eap.3092)
- Giraldo-Ospina et al. official reproducibility release: [Zenodo 14590515](https://zenodo.org/records/14590515)
- Smith et al. (2024), *Proceedings of the Royal Society B*, DOI: [10.1098/rspb.2023.2749](https://doi.org/10.1098/rspb.2023.2749)
- Cavanaugh et al. (2025), *Journal of Ecology*, DOI: [10.1111/1365-2745.70072](https://doi.org/10.1111/1365-2745.70072)
- Cavanaugh et al. (2026), *Communications Earth & Environment*: [High-resolution Planet Dove data identify local drivers of kelp canopy persistence](https://www.nature.com/articles/s43247-025-03134-y)
- Ecological Informatics (2025), DOI: [10.1016/j.ecoinf.2025.103521](https://doi.org/10.1016/j.ecoinf.2025.103521)
