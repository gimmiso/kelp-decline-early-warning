# Integrated Model Results Report

## Overview

This report consolidates existing result tables across V1 model comparison, threshold tuning, recall-oriented extensions, OISST sensitivity, CRW 5 km composites, bathymetry/habitat, multiscale environmental exposure, naive persistence, and canopy trajectory diagnostics.

No heavy feature construction or model training was rerun. Missing expected files are recorded below rather than fabricated.

Integrated rows:

```text
Source result files found: 23
Missing expected files: 1
Rows integrated: 3872
Rankable computed rows: 3868
Targets detected: 5
Detected target groups: original_decline, at_risk_original, new_transition, actionable_drop, high_canopy_subgroup
```

Status counts:

- `computed`: 3868
- `dry_run_no_local_crw_cache`: 4

Found result files:

- `outputs/diagnostics/at_risk_subset_model_performance.csv`
- `outputs/diagnostics/new_decline_transition_model_performance.csv`
- `outputs/metadata/model_comparison_results.csv`
- `outputs/metadata/model_comparison_test_metrics.csv`
- `outputs/metadata/model_diagnostics_same_model_comparison.csv`
- `outputs/metadata/threshold_tuning_test_results.csv`
- `outputs/model_results/actionable_decline_model_performance.csv`
- `outputs/model_results/cost_sensitive_model_performance.csv`
- `outputs/model_results/environment_incremental_value_performance.csv`
- `outputs/model_results/extended_threshold_selection_summary.csv`
- `outputs/model_results/feature_ablation_performance.csv`
- `outputs/model_results/oisst_matching_sensitivity_model_performance.csv`
- `outputs/model_results/threshold_selection_summary.csv`
- `results/tables/bathymetry_habitat_model_comparison.csv`
- `results/tables/canopy_trajectory_model_comparison.csv`
- `results/tables/crw5km_composite_model_comparison.csv`
- `results/tables/crw5km_model_comparison.csv`
- `results/tables/high_canopy_subgroup_performance.csv`
- `results/tables/ml_vs_naive_baseline_gap.csv`
- `results/tables/multiscale_model_comparison.csv`
- `results/tables/naive_persistence_baseline_comparison.csv`
- `results/tables/rare_event_alert_model_comparison.csv`
- `results/tables/wave_exposure_model_comparison.csv`

Missing expected result files:

- `results/tables/model_comparison_results.csv`

## Target Hierarchy

- **Original broad decline-state screening (`original_decline`)** identifies whether next-year canopy falls below a cell-specific low-canopy threshold. Strong performance here can reflect current canopy state, persistent low canopy, or spatial risk structure.
- **At-risk original target (`at_risk_original`)** removes already-low or near-zero current canopy observations. This is a stronger test of whether models help before visible low-state persistence dominates.
- **New transition target (`new_transition`)** focuses on transition into a low-canopy state from a not-yet-low state. This is closer to true early-warning than broad original-label prediction.
- **Actionable decline/drop target (`actionable_drop`)** emphasizes meaningful next-year decline or low-canopy warning sensitivity. Recall and false negatives matter, but precision must remain interpretable.
- **High-canopy subgroup (`high_canopy_subgroup`)** asks whether decline from healthier canopy states can be predicted; this is also transition-oriented.

## Main Findings by Target

### original_decline

- Best PR-AUC: `multiscale_environment` / `multiscale_environment` / `Random Forest` = `0.971`.
- Best recall: `threshold_tuned` / `threshold_tuned` / `Random Forest` = `1.000`.
- Best F2: `threshold_tuned` / `threshold_tuned` / `LightGBM (unweighted)` = `0.913`.
- Best false-negative count: `Random Forest` = `0`.
- Interpretation: Broad decline-state screening; high PR-AUC should be interpreted as risk-state prediction, especially when led by multiscale_environment.

### at_risk_original

- Best PR-AUC: `validity_diagnostics` / `canopy_only` / `Random Forest` = `0.897`.
- Best recall: `V2_extension` / `bathymetry_habitat_only` / `Logistic Regression` = `1.000`.
- Best F2: `V2_extension` / `bathymetry_habitat_only` / `Random Forest` = `0.837`.
- Best false-negative count: `Logistic Regression` = `0`.
- Interpretation: At-risk evaluation removes already-low canopy states and is a stronger early-warning stress test, but still may reflect persistence and spatial risk.

### new_transition

- Best PR-AUC: `V2_extension` / `canopy_trajectory` / `E2_logistic_current_lag_slope` = `0.610`.
- Best recall: `V2_extension` / `canopy_plus_trajectory` / `LightGBM` = `1.000`.
- Best F2: `V2_extension` / `canopy_plus_trajectory` / `XGBoost` = `0.676`.
- Best false-negative count: `LightGBM` = `0`.
- Interpretation: Strict transition-oriented target; useful performance here would be stronger evidence for early-warning skill than original-label performance.

### actionable_drop

- Best PR-AUC: `multiscale_environment` / `multiscale_environment` / `Random Forest` = `0.803`.
- Best recall: `threshold_tuned` / `threshold_tuned` / `Random Forest (cost_sensitive)` = `1.000`.
- Best F2: `threshold_tuned` / `threshold_tuned` / `Random Forest (cost_sensitive)` = `0.824`.
- Best false-negative count: `Random Forest (cost_sensitive)` = `0`.
- Interpretation: Actionable drop target emphasizes warning sensitivity; false-negative reduction must be balanced against precision.

### high_canopy_subgroup

- Best PR-AUC: `trajectory` / `canopy_plus_trajectory` / `Logistic Regression L2` = `0.657`.
- Best recall: `V2_extension` / `canopy_only` / `canopy_only / SVM` = `1.000`.
- Best F2: `` / `` / `` = `NA`.
- Best false-negative count: `canopy_only / SVM` = `0`.
- Interpretation: High-canopy subgroup asks whether models detect decline from healthier states; this is closer to transition monitoring than broad low-state detection.


## Baseline Gap Diagnosis

- `original_decline`: best-minus-naive `0.077`, best-minus-canopy `0.073`, CRW-minus-OISST `0.006`, trajectory-minus-canopy `-0.056`, false-negative reduction vs canopy `18`.
- `at_risk_original`: best-minus-naive `0.090`, best-minus-canopy `0.000`, CRW-minus-OISST `-0.294`, trajectory-minus-canopy `-0.226`, false-negative reduction vs canopy `0`.
- `new_transition`: best-minus-naive `0.319`, best-minus-canopy `0.028`, CRW-minus-OISST `-0.284`, trajectory-minus-canopy `0.028`, false-negative reduction vs canopy `0`.
- `actionable_drop`: best-minus-naive `0.406`, best-minus-canopy `0.153`, CRW-minus-OISST `-0.505`, trajectory-minus-canopy `-0.075`, false-negative reduction vs canopy `0`.
- `high_canopy_subgroup`: best-minus-naive `0.213`, best-minus-canopy `0.018`, CRW-minus-OISST `-0.422`, trajectory-minus-canopy `0.018`, false-negative reduction vs canopy `0`.

Interpretation rule: improvements on the original broad decline target support **risk-state screening** unless the same feature family also improves at-risk, new-transition, or actionable-drop targets. False-negative reductions are useful for screening, but a recall gain with low precision is a sensitivity trade-off rather than operational early-warning success.

## Feature-Layer Interpretation

- **CRW 5 km composite:** best PR-AUC 0.852 on `original_decline` using `bathymetry_habitat` / `Logistic Regression L2`.
- **bathymetry/habitat:** best PR-AUC 0.903 on `original_decline` using `wave_exposure` / `XGBoost`.
- **canopy trajectory:** best PR-AUC 0.809 on `original_decline` using `trajectory` / `Logistic Regression L2`.
- **threshold-tuned:** best PR-AUC 0.897 on `original_decline` using `threshold_tuned` / `Random Forest`.
- **rare-event/cost-sensitive:** best PR-AUC 0.888 on `original_decline` using `rare_event_learning` / `SVM (cost_sensitive)`.

The integrated tables distinguish source-aware environmental exposure, static habitat context, and canopy-state or trajectory features. Habitat-only and environmental-only performance should be read as spatial or exposure-risk screening, not as proof of a causal mechanism. Trajectory features are leakage-audited time-series instability proxies; they are not true patch fragmentation unless patch geometry is added.

## Current Bottlenecks

- **Target definition:** the original decline target is partly a low-state/risk-state label, so it can overstate early-warning skill.
- **Rare-event class imbalance:** stricter transition and actionable targets have fewer positive events, making recall and precision unstable.
- **Persistence bias:** current canopy and low-canopy persistence remain strong predictors, especially for the original target.
- **Missing disturbance variables:** wave exposure, storm disturbance, grazing pressure, disease context, and restoration/intervention history are not yet represented.
- **Missing biological drivers:** urchin/grazer pressure and predator/community data are likely needed for a stronger ecological transition case study.
- **Limited spatial support / coarse covariates:** OISST and latitude-bin proxies remain broad exposure layers; CRW 5 km improves spatial resolution but is still satellite SST exposure, not in situ kelp-bed temperature.

## Recommended Next Steps

1. **Define claim gates:** decide which targets and minimum precision/recall criteria are required before using the phrase early-warning.
2. **Consolidate rare-event learning:** focus on transition and actionable targets with calibrated threshold selection, class weighting, and precision floors.
3. **Add wave exposure:** test whether disturbance exposure explains actionable drops better than SST alone.
4. **Assess true fragmentation feasibility:** only add fragmentation claims if patch geometry or within-cell spatial structure can be measured.
5. **Generate final figures and tables:** use the integrated master and best-by-target tables to build a concise portfolio or manuscript result section.
