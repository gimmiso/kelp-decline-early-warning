# Recall-Oriented Modeling Extension Report

## Purpose

This extension tests whether recall-oriented early-warning screening can be strengthened beyond current canopy-state persistence by adding cost-sensitive models, actionable decline labels, canopy trajectory features, environmental stress interactions, feature-set ablations, and validation-based threshold tuning.

## Best Test Models by Target

| target | feature_set | model_family | model_variant | precision | recall | f1 | f2 | pr_auc | false_negatives | false_positives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actionable_decline_drop_next | canopy_trajectory_only | XGBoost | cost_sensitive | 0.520 | 0.765 | 0.619 | 0.699 | 0.496 | 8 | 24 |
| actionable_decline_low_next | environment_only | Logistic Regression | cost_sensitive | 0.301 | 0.889 | 0.449 | 0.639 | 0.266 | 5 | 93 |
| decline_event_next | canopy_current_only | Logistic Regression | cost_sensitive | 0.732 | 0.896 | 0.805 | 0.857 | 0.857 | 14 | 44 |

## Feature Ablation Snapshot

| feature_set | model_family | model_variant | precision | recall | f1 | f2 | pr_auc | false_negatives | false_positives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canopy_current_only | Logistic Regression | cost_sensitive | 0.732 | 0.896 | 0.805 | 0.857 | 0.857 | 14 | 44 |
| canopy_current_plus_trajectory | Logistic Regression | cost_sensitive | 0.648 | 0.522 | 0.579 | 0.543 | 0.693 | 64 | 38 |
| canopy_plus_environment | Logistic Regression | cost_sensitive | 0.692 | 0.687 | 0.689 | 0.688 | 0.802 | 42 | 41 |
| canopy_plus_trajectory_plus_environment | Logistic Regression | cost_sensitive | 0.683 | 0.627 | 0.654 | 0.637 | 0.721 | 50 | 39 |
| canopy_trajectory_only | Logistic Regression | cost_sensitive | 0.639 | 0.515 | 0.570 | 0.536 | 0.697 | 65 | 39 |
| environment_only | Logistic Regression | cost_sensitive | 0.667 | 0.627 | 0.646 | 0.634 | 0.783 | 50 | 42 |

## Threshold-Tuning Recommendations

- Highest F2 screening result: `decline_event_next / canopy_current_only / LightGBM (unweighted)` using `max_f2` at threshold 0.05.
- Recommended precision-floor balanced result: `decline_event_next / canopy_current_only / XGBoost (unweighted)` using threshold 0.25 without precision-floor fallback.
- Best actionable low-canopy threshold result: `canopy_current_only / Random Forest (cost_sensitive)` at threshold 0.05.
- Best actionable drop threshold result: `canopy_current_plus_trajectory / SVM (cost_sensitive)` at threshold 0.20.

## Persistence-Adjusted Feature Interpretation

- Original-label current-canopy best F2: 0.857 (`Logistic Regression / cost_sensitive`).
- Original-label trajectory-only best F2: 0.536 (`Logistic Regression / cost_sensitive`).
- For the original decline label, trajectory-only and trajectory-augmented feature sets did not outperform current-canopy-only models, so full-sample performance still appears strongly tied to canopy-state persistence.
- For `actionable_decline_drop_next`, trajectory features were more useful: the strongest default-threshold model used `canopy_trajectory_only`, and the strongest threshold-tuned model used `canopy_current_plus_trajectory`.

## Interpretation

Cost-sensitive learning and threshold tuning improve recall-oriented operating points, but they should be interpreted as screening configurations rather than operational proof. Actionable decline labels are more relevant for early warning because they focus on currently observable canopy entering a low or sharply declining state. The extension strengthens the early-warning story for actionable canopy-drop screening, but the original decline label still shows strong canopy-state persistence.

## Output Files

- `outputs/model_results/cost_sensitive_model_performance.csv`
- `outputs/model_results/cost_sensitive_model_summary.csv`
- `outputs/model_results/actionable_decline_model_performance.csv`
- `outputs/model_results/feature_ablation_performance.csv`
- `outputs/model_results/extended_threshold_tuning_results.csv`
- `outputs/model_results/extended_threshold_selection_summary.csv`
- `outputs/diagnostics/actionable_decline_label_summary.csv`
- `outputs/metadata/trajectory_feature_summary.csv`
- `outputs/metadata/feature_availability_report.csv`
