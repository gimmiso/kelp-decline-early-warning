# California 국지 사례연구 독립 검증

- 판정: **Share with caveats**
- 통과: 26/26
- 차단 실패: 0

이 검증은 입력 grain, 누수 없는 미래분할, 동일표본, 연도별 AP, 순위·예산 선택, 현장자료 결합, 해시를 독립 재계산한다.
MUR climatology의 과거일 전용 규칙은 소스 코드와 저장된 provenance 필드로 확인했으며 원격 일자료 전체를 재다운로드하지는 않았다.

## 점검표

| check                                            | passed   | severity_if_failed   | evidence                                                                     |
|:-------------------------------------------------|:---------|:---------------------|:-----------------------------------------------------------------------------|
| panel_keys_unique                                | True     | blocker              | 0                                                                            |
| field_keys_unique                                | True     | blocker              | 0                                                                            |
| mur_keys_unique                                  | True     | blocker              | 0                                                                            |
| prediction_keys_unique                           | True     | blocker              | 0                                                                            |
| forward_only_expanding_folds                     | True     | blocker              | max_train_end_minus_test=-1                                                  |
| at_least_four_complete_training_years            | True     | blocker              | minimum=4                                                                    |
| scores_finite_and_bounded                        | True     | blocker              | missing=0; min=0.001923; max=1.000000                                        |
| exact_rows_across_models_within_support_year     | True     | blocker              | groups=28                                                                    |
| exact_outcomes_across_models_within_support_year | True     | blocker              | ordered site-event vectors match                                             |
| risk_percentiles_recomputed                      | True     | blocker              | max_error=1.11e-16                                                           |
| top20_selection_recomputed                       | True     | blocker              | max_symmetric_difference=0                                                   |
| annual_ap_lifts_recomputed                       | True     | blocker              | max_error=9.89e-17                                                           |
| paired_increments_recomputed                     | True     | blocker              | max_error=1.08e-16                                                           |
| diagnostic_keys_unique                           | True     | blocker              | 0                                                                            |
| ecology_join_keys_recomputed                     | True     | blocker              | expected=5151; actual=5151                                                   |
| both_spatial_supports_present                    | True     | blocker              | [300, 1000]                                                                  |
| overlap_not_counted_as_independent_sites         | True     | claim                | {300: {'sites': 152, 'clusters': 120}, 1000: {'sites': 169, 'clusters': 73}} |
| leave_overlap_cluster_out_audit_complete         | True     | blocker              | rows=20                                                                      |
| source_quality_checks_pass                       | True     | blocker              | []                                                                           |
| mur_quality_gate_passed                          | True     | blocker              | 1.0                                                                          |
| mur_climatology_rule_recorded_as_past_only       | True     | claim                | ['strictly prior MUR days at the same source pixel']                         |
| future_sst_cannot_change_prior_year_features     | True     | blocker              | max_error=0                                                                  |
| panel_hash_matches_manifest                      | True     | blocker              | 106bc06eac131b45273fccff63581fa49f1adc4e74e711d01d686dd5808901aa             |
| field_hash_matches_manifest                      | True     | blocker              | a9bee03d5acae023465ef1f2f2e5e86122962e97758a3200877bf2a8f4d193de             |
| mur_hash_matches_manifest                        | True     | blocker              | 06547c2b369af079dda6cf8a83593b2ac87cb16d9d367990e1dcfbdbc22b9f8d             |
| saved_output_hashes_match_manifest               | True     | blocker              | []                                                                           |
