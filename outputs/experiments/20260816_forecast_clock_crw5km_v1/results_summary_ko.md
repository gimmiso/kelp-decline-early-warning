# 예측시점·CRW 5 km 실험 결과

## 핵심 판정

- March CRW 5 km의 과거 캐노피 이후 AP 증분: `-0.0056` (2-year block 95% CI `-0.0222`–`0.0079`).
- March OISST의 동일 증분: `-0.0012` (95% CI `-0.0170`–`0.0107`).
- 주 결과는 같은 165셀 후보군, 같은 forecast year, 같은 적격 행에서 비교했다.
- March 결과는 Q2-Q3 canopy만 사용하므로 3월 31일 이후의 결과창과 겹치지 않는다.
- CRW 증분은 Logistic `-0.0056`, Random Forest `-0.0107`, XGBoost `-0.0001`로 세 모형군 모두 양수가 아니었고, 모든 95% 신뢰구간이 0을 포함했다.

## 모델 요약

| outcome_definition | model | forecast_years | mean_ap | macro_within_year_ap_lift | mean_recall_top_5pct | mean_recall_top_10pct | mean_recall_top_20pct | mean_recall_top_30pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| annual_max_decline_sensitivity | december_plus_prior_winter_oisst | 20 | 0.5834 | 0.1969 | 0.1042 | 0.2102 | 0.3743 | 0.4785 |
| annual_max_decline_sensitivity | march_plus_current_crw5km | 20 | 0.5754 | 0.1889 | 0.1018 | 0.2024 | 0.3580 | 0.4690 |
| annual_max_decline_sensitivity | march_plus_current_crw5km_oisst | 20 | 0.5761 | 0.1895 | 0.1042 | 0.1944 | 0.3644 | 0.4768 |
| annual_max_decline_sensitivity | march_plus_current_winter_oisst | 20 | 0.5810 | 0.1945 | 0.1018 | 0.2043 | 0.3603 | 0.4832 |
| annual_max_decline_sensitivity | prior_current | 20 | 0.5652 | 0.1786 | 0.0964 | 0.2006 | 0.3584 | 0.4612 |
| annual_max_decline_sensitivity | prior_trajectory | 20 | 0.5827 | 0.1962 | 0.1047 | 0.2087 | 0.3727 | 0.4796 |
| q23_year_over_year_decline | december_plus_prior_winter_oisst | 20 | 0.5792 | 0.1800 | 0.0948 | 0.1959 | 0.3463 | 0.4608 |
| q23_year_over_year_decline | march_plus_current_crw5km | 20 | 0.5721 | 0.1729 | 0.0973 | 0.1957 | 0.3322 | 0.4498 |
| q23_year_over_year_decline | march_plus_current_crw5km_oisst | 20 | 0.5743 | 0.1751 | 0.0993 | 0.1844 | 0.3401 | 0.4558 |
| q23_year_over_year_decline | march_plus_current_winter_oisst | 20 | 0.5765 | 0.1772 | 0.0972 | 0.1899 | 0.3376 | 0.4545 |
| q23_year_over_year_decline | prior_current | 20 | 0.5610 | 0.1618 | 0.0894 | 0.1899 | 0.3383 | 0.4390 |
| q23_year_over_year_decline | prior_trajectory | 20 | 0.5777 | 0.1785 | 0.0967 | 0.1898 | 0.3486 | 0.4549 |

## 증분 비교

| outcome_definition | comparison | augmented_model | baseline_model | mean_ap_lift_difference | ci_low | ci_high | positive_years | forecast_years | mean_top20_recall_difference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| annual_max_decline_sensitivity | primary_crw_increment | march_plus_current_crw5km | prior_trajectory | -0.0073 | -0.0294 | 0.0080 | 11 | 20 | -0.0147 |
| annual_max_decline_sensitivity | march_oisst_increment | march_plus_current_winter_oisst | prior_trajectory | -0.0017 | -0.0188 | 0.0100 | 10 | 20 | -0.0124 |
| annual_max_decline_sensitivity | december_oisst_increment | december_plus_prior_winter_oisst | prior_trajectory | 0.0007 | -0.0040 | 0.0049 | 10 | 20 | 0.0016 |
| annual_max_decline_sensitivity | crw_minus_oisst | march_plus_current_crw5km | march_plus_current_winter_oisst | -0.0056 | -0.0142 | 0.0030 | 7 | 20 | -0.0023 |
| annual_max_decline_sensitivity | oisst_after_crw | march_plus_current_crw5km_oisst | march_plus_current_crw5km | 0.0007 | -0.0088 | 0.0097 | 11 | 20 | 0.0064 |
| q23_year_over_year_decline | primary_crw_increment | march_plus_current_crw5km | prior_trajectory | -0.0056 | -0.0222 | 0.0079 | 11 | 20 | -0.0164 |
| q23_year_over_year_decline | march_oisst_increment | march_plus_current_winter_oisst | prior_trajectory | -0.0012 | -0.0170 | 0.0107 | 12 | 20 | -0.0110 |
| q23_year_over_year_decline | december_oisst_increment | december_plus_prior_winter_oisst | prior_trajectory | 0.0015 | -0.0034 | 0.0059 | 11 | 20 | -0.0023 |
| q23_year_over_year_decline | crw_minus_oisst | march_plus_current_crw5km | march_plus_current_winter_oisst | -0.0043 | -0.0130 | 0.0036 | 8 | 20 | -0.0054 |
| q23_year_over_year_decline | oisst_after_crw | march_plus_current_crw5km_oisst | march_plus_current_crw5km | 0.0022 | -0.0097 | 0.0136 | 11 | 20 | 0.0079 |

## 데이터 품질

| check | value | expected | passed |
| --- | --- | --- | --- |
| locked_cells | 165.0000 | 165 | True |
| forecast_years | 20.0000 | 20 | True |
| q23_evaluation_rows | 2478.0000 | >=2000 | True |
| q23_events | 1009.0000 | >=200 | True |
| all_years_two_classes | 20.0000 | 20 | True |
| crw_mapping_max_km | 7.3040 | <=15 | True |
| same_rows_all_models_within_outcome | 1.0000 | 1 | True |

## 모형군 민감도

| model_family | CRW AP 증분 | 95% CI | OISST AP 증분 | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Logistic | -0.0056 | -0.0222–0.0079 | -0.0012 | -0.0170–0.0107 |
| Random Forest | -0.0107 | -0.0422–0.0067 | -0.0036 | -0.0347–0.0176 |
| XGBoost | -0.0001 | -0.0310–0.0229 | 0.0086 | -0.0283–0.0379 |

## 해석 제한

- CRW와 OISST는 근해 표층수온 프록시이며 현장 수온이 아니다.
- CRW 특징은 월평균 3개만 사용하므로 일별 극값·지속시간 효과를 검정하지 않는다.
- 이 실험은 예측시점과 제품 support의 민감도이며 인과효과 분석이 아니다.
