# Recall-Oriented Threshold Tuning Report

## Purpose

This diagnostic evaluates whether validation-selected decision thresholds can reduce false negatives for kelp decline early-warning screening. The original model comparison at the default 0.5 threshold remains unchanged.

Threshold tuning changes the operating point of the classifier, trading precision for recall. It does not improve or change ranking metrics such as PR-AUC or ROC-AUC.

Thresholds were selected using the validation period only and then fixed for the held-out test period to avoid test-set leakage.

## Data and Split

- Dataset: `data/processed/modeling_dataset_ge500_noaa_v1.csv`
- Train: 1989-2016
- Validation: 2017-2020
- Test: 2021-2024
- Candidate thresholds: 0.05 to 0.95 in 0.05 increments
- Primary rule: highest validation F1 among thresholds with validation recall >= 0.70; if none reach 0.70, highest recall with F1 as tie-breaker
- Secondary rule: highest validation F1 regardless of recall

## Main Findings

- Largest test recall gain: canopy_noaa / SVM (0.246 to 1.000; false negatives 101 to 0).
- Largest false-negative reduction: canopy_noaa / SVM (101 fewer false negatives).
- Highest recall-oriented test recall: canopy_noaa / SVM (threshold=0.05, recall=1.000, precision=0.670, F1=0.802).
- Best recall-oriented F1: canopy_only / Random Forest (threshold=0.30, recall=0.910, precision=0.753, F1=0.824).

## Specific Model Checks

- `canopy_only / Random Forest` remained strong after threshold tuning: recall changed from 0.590 to 0.910, false negatives from 55 to 12, precision from 0.908 to 0.753, and F1 from 0.715 to 0.824.
- `canopy_noaa / SVM` became more useful as an early-warning screen after threshold tuning: recall changed from 0.246 to 1.000, false negatives from 101 to 0, precision from 0.868 to 0.670, and F1 from 0.384 to 0.802.

## Threshold-Tuned Test Comparison

| feature_set | model | threshold_recall | test_recall_default | test_recall_recall | recall_gain | test_false_negatives_default | test_false_negatives_recall | false_negative_reduction | test_precision_default | test_precision_recall | precision_change | test_f1_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canopy_noaa | SVM | 0.050 | 0.246 | 1.000 | 0.754 | 101 | 0 | 101 | 0.868 | 0.670 | -0.198 | 0.802 |
| canopy_noaa | Random Forest | 0.050 | 0.276 | 1.000 | 0.724 | 97 | 0 | 97 | 0.925 | 0.670 | -0.255 | 0.802 |
| canopy_only | SVM | 0.300 | 0.000 | 0.709 | 0.709 | 134 | 39 | 95 | 0.000 | 0.798 | 0.798 | 0.751 |
| canopy_noaa | LightGBM | 0.050 | 0.381 | 0.948 | 0.567 | 83 | 7 | 76 | 0.895 | 0.665 | -0.230 | 0.782 |
| oisst_only | SVM | 0.250 | 0.097 | 0.537 | 0.440 | 121 | 62 | 59 | 0.929 | 0.649 | -0.280 | 0.588 |
| oisst_only | Random Forest | 0.300 | 0.291 | 0.642 | 0.351 | 95 | 48 | 47 | 0.709 | 0.667 | -0.042 | 0.654 |
| oisst_only | LightGBM | 0.200 | 0.321 | 0.664 | 0.343 | 91 | 45 | 46 | 0.683 | 0.622 | -0.060 | 0.643 |
| canopy_only | Random Forest | 0.300 | 0.590 | 0.910 | 0.321 | 55 | 12 | 43 | 0.908 | 0.753 | -0.155 | 0.824 |
| canopy_noaa | XGBoost | 0.250 | 0.522 | 0.754 | 0.231 | 64 | 33 | 31 | 0.875 | 0.711 | -0.164 | 0.732 |
| canopy_only | XGBoost | 0.400 | 0.590 | 0.799 | 0.209 | 55 | 27 | 28 | 0.868 | 0.748 | -0.120 | 0.773 |
| canopy_only | Logistic Regression | 0.400 | 0.799 | 1.000 | 0.201 | 27 | 0 | 27 | 0.699 | 0.670 | -0.029 | 0.802 |
| oisst_only | Logistic Regression | 0.450 | 0.261 | 0.366 | 0.104 | 99 | 85 | 14 | 0.714 | 0.681 | -0.034 | 0.476 |
| oisst_only | XGBoost | 0.400 | 0.448 | 0.537 | 0.090 | 74 | 62 | 12 | 0.723 | 0.649 | -0.074 | 0.588 |
| canopy_only | LightGBM | 0.400 | 0.590 | 0.672 | 0.082 | 55 | 44 | 11 | 0.859 | 0.833 | -0.025 | 0.744 |
| canopy_noaa | Logistic Regression | 0.500 | 0.627 | 0.627 | 0.000 | 50 | 50 | 0 | 0.764 | 0.764 | 0.000 | 0.689 |

## Interpretation

For early-warning use, a recall-oriented operating point can be appropriate when false negatives are more costly than false positives. The preferred threshold-tuned model depends on whether the priority is maximum recall or a more balanced precision-recall trade-off.

- If the goal is maximum screening sensitivity, the strongest recall-oriented option is `canopy_noaa / SVM`.
- If the goal is a stronger balance between recall and precision, the strongest recall-oriented F1 option is `canopy_only / Random Forest`.

These results should be interpreted as operating-point diagnostics, not as evidence that threshold tuning improves the underlying model ranking quality.

## Output Files

- `outputs/metadata/threshold_tuning_validation_grid.csv`
- `outputs/metadata/threshold_tuning_selected_thresholds.csv`
- `outputs/metadata/threshold_tuning_test_results.csv`
- `outputs/figures/threshold_tuning_recall_precision_tradeoff.png`
- `outputs/figures/threshold_tuning_false_negatives.png`

## Validation Grid Summary

- Validation grid rows: 285
- Selected threshold rows: 45
- Test result rows: 45
- F1-optimal threshold rows: 15
