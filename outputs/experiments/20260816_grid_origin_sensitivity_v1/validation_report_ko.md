# 격자 크기·원점 민감도 독립 검증

- 결과: **25/25 checks passed**
- 저장된 예측확률에서 연도별 AP lift와 top-20% 포착률을 재계산했다.
- 저장된 2,000회 bootstrap draws에서 95% 분위수를 재계산했다.
- 픽셀 혼동행렬에서 라벨 Jaccard를 재계산하고, 동일 격자의 공간 일치도 1.0을 확인했다.
- 실행 manifest의 모든 원래 산출물 SHA-256을 다시 확인했다.

## 검증 결과

- PASS — `twenty_locked_specifications`: 20
- PASS — `twelve_area_scaled_specifications`: 12
- PASS — `area_scaled_pixel_thresholds`: {5: 125, 10: 500, 20: 2000}
- PASS — `fixed_500_is_secondary_5_and_20km_only`: rows=8
- PASS — `prediction_keys_unique`: 0
- PASS — `prediction_probabilities_bounded_complete`: 0
- PASS — `strict_forward_folds`: 0
- PASS — `twenty_test_years_per_model`: 20
- PASS — `year_metrics_recomputed_from_predictions`: 2000
- PASS — `macro_year_lifts_recomputed`: 100
- PASS — `top20_recall_recomputed`: 100
- PASS — `model_bootstrap_quantiles_recomputed`: 100
- PASS — `difference_bootstrap_quantiles_recomputed`: 60
- PASS — `two_thousand_draws_each`: 2000
- PASS — `label_jaccard_recomputed_from_pixel_counts`: 40
- PASS — `same_grid_spatial_identity`: 5
- PASS — `global_agreement_rows_complete`: 20
- PASS — `current_lift_supported_all_scaled_grids`: 12
- PASS — `trajectory_support_count_is_eight`: 8
- PASS — `oisst_support_count_is_zero`: 0
- PASS — `primary_reference_values`: cells=167, lift=0.176436254132
- PASS — `base_panel_reproduction_passed`: 0
- PASS — `run_quality_checks_passed`: 11
- PASS — `manifest_output_hashes`: 0
- PASS — `manifest_complete_and_classified`: complete
