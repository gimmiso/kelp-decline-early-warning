# V2 Multi-Scale Exposure Selection Report

## Purpose

This report evaluates OISST exposure variables at multiple spatial supports using transition-oriented kelp decline targets. It treats spatial resolution as a modeling choice rather than a fixed preprocessing assumption.

## Outputs

- `results/tables/multiscale_model_comparison.csv`
- `results/tables/selected_scale_by_predictor.csv`
- `results/tables/feature_collinearity_v2.csv`

## Main Findings

- Model-result rows: `154`.
- High-correlation V2 feature pairs: `81`.
- Best temporal new-decline transition PR-AUC: `0.282` from `M2_single_scale_75km / Random Forest` at scale `75km`.
- Best temporal actionable-drop PR-AUC: `0.387` from `M2_single_scale_50km / Logistic Regression L2` at scale `50km`.

## Interpretation

The transition and actionable-drop labels are harder than the original decline-state label because they reduce the influence of already-low canopy persistence. Lower performance under these labels is scientifically meaningful and should not be framed as failure.

Scale selection is reported as multi-scale exposure selection, not as discovery of one universal optimal resolution. Thermal stress, upwelling proxies, and local biological processes may operate at different spatial supports.

## Selected Candidate Scales

               target_definition   scale  mean_pr_auc  mean_recall  pr_auc_std  mean_balanced_accuracy  mean_brier_score  decision_score                               selection_rule                                                                interpretation
        A_original_decline_state    30km     0.796095     0.522388    0.014107                0.549073          0.255843        0.967397 combined_pr_auc_recall_stability_calibration Selected as a candidate exposure support, not a universal optimal resolution.
B_at_risk_original_decline_gt005 nearest     0.569101     0.333333    0.242734                0.468750          0.249530        0.662559 combined_pr_auc_recall_stability_calibration Selected as a candidate exposure support, not a universal optimal resolution.
        C_new_decline_transition    10km     0.262298     0.615385    0.015872                0.540451          0.221775        0.457513 combined_pr_auc_recall_stability_calibration Selected as a candidate exposure support, not a universal optimal resolution.
       D_actionable_decline_drop    10km     0.285999     0.882353    0.005228                0.594791          0.310019        0.550043 combined_pr_auc_recall_stability_calibration Selected as a candidate exposure support, not a universal optimal resolution.
