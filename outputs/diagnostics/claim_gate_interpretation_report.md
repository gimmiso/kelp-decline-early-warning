# Claim-Gate Interpretation Report

## Purpose

Claim gates are conservative interpretation rules used to prevent overclaiming from integrated model-result tables. They are not formal statistical significance tests, confidence intervals, or causal tests. They translate existing model metrics into defensible claim levels.

No new models were trained and no heavy feature construction was rerun for this report.

## Gate Definitions

### Gate 1: Broad Risk-State Screening Support

Relevant target: `original_decline`.

This gate passes if at least one broad screening condition is met: best PR-AUC is at least `0.75`, environment/combined models improve over OISST-only by at least `0.03` PR-AUC, or the best model improves over naive persistence by at least `0.03` PR-AUC. Passing this gate supports broad decline-risk state screening only.

### Gate 2: At-Risk Screening Support

Relevant target: `at_risk_original`.

This gate evaluates whether a model improves over canopy-only baselines among observations that are more relevant to early warning. It checks PR-AUC gain, F2 gain, false-negative reduction, recall gain, and whether precision meets the selected floor. Passing this gate supports at-risk screening, not robust transition early warning.

### Gate 3: Transition/Actionable Early-Warning Support

Relevant targets: `new_transition` and `actionable_drop`.

This is the strict early-warning-oriented gate. It checks PR-AUC gain over canopy-only, F2 gain, false-negative reduction, recall, precision floor, and whether threshold information indicates validation-selected thresholding. A target must meet at least three conditions and include PR-AUC improvement over canopy-only to be labeled as supported. If recall or false negatives improve without PR-AUC support, the result is labeled as recall-oriented warning sensitivity only.

## Results by Target

### original_decline

- Gate: `G1` (Broad risk-state screening support)
- Result: `risk_state_screening_supported`; pass = `True`
- Selected model: `multiscale_environment` / `Random Forest`
- Metrics: PR-AUC `0.971`, recall `1.000`, precision `NA`, F2 `0.913`, false negatives `0`
- Baseline: `naive_persistence / OISST_only` PR-AUC `0.893`, recall `NA`, false negatives `NA`
- Supporting conditions: best_model_pr_auc_ge_0_75; environment_or_combined_minus_OISST_pr_auc_ge_0_03; best_model_minus_naive_pr_auc_ge_0_03

### at_risk_original

- Gate: `G2` (At-risk screening support)
- Result: `at_risk_screening_partially_supported`; pass = `True`
- Selected model: `naive_persistence` / `current_low_canopy_score`
- Metrics: PR-AUC `0.573`, recall `0.978`, precision `0.484`, F2 `NA`, false negatives `1`
- Baseline: `canopy_only` PR-AUC `0.897`, recall `0.800`, false negatives `4`
- Supporting conditions: false_negatives_decrease_ge_30pct_vs_canopy; recall_improves_ge_0_10_vs_canopy; precision_ge_floor

### actionable_drop

- Gate: `G3` (Transition/actionable early-warning support)
- Result: `transition_recall_oriented_sensitivity_only`; pass = `False`
- Selected model: `threshold_tuned` / `SVM (cost_sensitive)`
- Metrics: PR-AUC `0.577`, recall `0.941`, precision `0.516`, F2 `0.808`, false negatives `2`
- Baseline: `canopy_only` PR-AUC `0.650`, recall `0.765`, false negatives `8`
- Supporting conditions: best_model_minus_canopy_f2_ge_0_10; false_negatives_decrease_ge_40pct_vs_canopy; recall_ge_0_70; precision_ge_floor; threshold_validation_selected

### new_transition

- Gate: `G3` (Transition/actionable early-warning support)
- Result: `transition_early_warning_not_supported`; pass = `False`
- Selected model: `multiscale_environment` / `M2_idw_k8_sensitivity / Logistic Regression L2`
- Metrics: PR-AUC `0.206`, recall `1.000`, precision `0.130`, F2 `NA`, false negatives `0`
- Baseline: `canopy_only` PR-AUC `0.582`, recall `0.654`, false negatives `6`
- Supporting conditions: false_negatives_decrease_ge_40pct_vs_canopy; recall_ge_0_70


## Sensitivity Analysis

The same gates were evaluated with precision floors `0.30`, `0.40`, and `0.50`.

- `G1` / `original_decline` -> 0.30: `risk_state_screening_supported`, 0.40: `risk_state_screening_supported`, 0.50: `risk_state_screening_supported`
- `G2` / `at_risk_original` -> 0.30: `at_risk_screening_partially_supported`, 0.40: `at_risk_screening_partially_supported`, 0.50: `at_risk_screening_partially_supported`
- `G3` / `actionable_drop` -> 0.30: `transition_recall_oriented_sensitivity_only`, 0.40: `transition_recall_oriented_sensitivity_only`, 0.50: `transition_recall_oriented_sensitivity_only`
- `G3` / `new_transition` -> 0.30: `transition_recall_oriented_sensitivity_only`, 0.40: `transition_early_warning_not_supported`, 0.50: `transition_early_warning_not_supported`

## Final Claim Level

Current results support recall-oriented warning sensitivity, not robust operational early warning.

Gate 1: `risk_state_screening_supported`. Gate 2: `at_risk_screening_partially_supported`. Gate 3: `transition_recall_oriented_sensitivity_only`.

The safest current claim is that this repository supports **broad decline-risk state screening** and provides a structured robustness framework for testing stricter early-warning claims. At-risk screening evidence is partial because gains over canopy-only are limited under the default precision floor. Gate 3 overall is `transition_recall_oriented_sensitivity_only`.

## What Can Be Claimed Safely

- Broad risk-state screening is supported for the original decline-state target.
- The integrated workflow can distinguish broad screening, at-risk screening, and stricter transition/actionable targets.
- Recall-oriented threshold and cost-sensitive settings can reduce false negatives in some settings, but these are sensitivity trade-offs.

## What Can Be Claimed Only Partially

- At-risk screening support is partial unless future runs show clearer improvement over canopy-only baselines while maintaining acceptable precision.
- Transition/actionable warning sensitivity can be discussed only as a diagnostic result when recall or false negatives improve without PR-AUC support.

## What Cannot Yet Be Claimed

- The results do not yet establish an operational kelp decline early-warning system.
- Strong original-label PR-AUC should not be used as evidence of true transition early-warning.
- Environmental or habitat covariate gains should not be interpreted as causal mechanisms.

## Recommended Next Steps

1. **Rare-event learning:** prioritize transition/actionable labels if Gate 3 failures are driven by low recall or false negatives.
2. **Wave exposure:** add disturbance exposure if environment, habitat, and trajectory features still fail transition targets.
3. **True fragmentation feasibility:** add patch geometry or within-cell spatial structure before making fragmentation claims.
4. **External/spatial validation:** if future gates become strong, test leave-region-out or external-region validation before stronger claims.
