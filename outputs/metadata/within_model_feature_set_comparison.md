# Within-Model Feature-Set Comparison

## Existing Diagnostic Files Checked

- `outputs/metadata/model_diagnostics_report.md`: **Present**. Narrative diagnostic summary, including best model, same-model NOAA improvement for SVM, leakage audit, and limitations.
- `outputs/metadata/model_comparison_test_metrics.csv`: **Present**. Test-period PR-AUC, ROC-AUC, recall, precision, F1, accuracy, confusion counts, row counts, and positive rate for each model-feature-set combination.
- `outputs/metadata/model_comparison_results.csv`: **Present**. Validation and test metrics for all model-feature-set combinations.
- `outputs/metadata/model_comparison_confusion_matrices.csv`: **Present**. Validation and test confusion-matrix counts for each model-feature-set combination.
- `outputs/figures/model_diagnostics_feature_set_pr_auc.png`: **Present**. Feature-set PR-AUC diagnostic figure.
- `outputs/figures/model_diagnostics_feature_set_recall.png`: **Present**. Feature-set recall diagnostic figure.

## Why This Table Was Added

The existing metric files already contain the necessary test metrics and false-negative counts. However, they do not present all pairwise feature-set changes within each algorithm in a compact seminar-paper format. This table holds the algorithm fixed and compares `canopy_noaa`, `oisst_only`, and `canopy_only` feature sets directly.

## Key Interpretation

- The best aggregate test PR-AUC remains `canopy_only / Random Forest`.
- The best canopy+NOAA PR-AUC remains `canopy_noaa / SVM`.
- Within-model comparison shows that adding NOAA context did not improve PR-AUC over canopy-only for most algorithms, but it changed recall/F1 and false-negative behavior for some models.
- The clearest same-model improvement from adding NOAA context occurred for SVM relative to canopy-only at the default threshold: recall increased, F1 increased, and false negatives decreased, although PR-AUC decreased relative to canopy-only SVM.
- These results support interpreting NOAA variables as environmental exposure context rather than replacements for direct canopy monitoring.

## Comparison Table

| model | comparison | delta_pr_auc | delta_recall | delta_f1 | delta_precision | delta_false_negatives | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | canopy_noaa - canopy_only | -0.0311 | -0.1716 | -0.0571 | 0.0643 | 23 | decreased PR-AUC by -0.031; decreased recall by -0.172; decreased F1 by -0.057; increased false negatives by 23. Adding NOAA context did not imply causal effects and should be read as feature-set augmentation. Aggregate PR-AUC did not improve within this algorithm. |
| Logistic Regression | oisst_only - canopy_only | -0.0994 | -0.5373 | -0.3631 | 0.0149 | 72 | decreased PR-AUC by -0.099; decreased recall by -0.537; decreased F1 by -0.363; increased false negatives by 72. This contrasts environmental exposure variables alone with direct canopy-state monitoring. |
| Logistic Regression | canopy_noaa - oisst_only | 0.0683 | 0.3657 | 0.3060 | 0.0494 | -49 | increased PR-AUC by +0.068; increased recall by +0.366; increased F1 by +0.306; reduced false negatives by 49. This isolates the added value of combining canopy-state variables with NOAA exposure variables relative to NOAA-only inputs. |
| SVM | canopy_noaa - canopy_only | -0.0407 | 0.2463 | 0.3837 | 0.8684 | -33 | decreased PR-AUC by -0.041; increased recall by +0.246; increased F1 by +0.384; reduced false negatives by 33. Adding NOAA context did not imply causal effects and should be read as feature-set augmentation. Aggregate PR-AUC did not improve within this algorithm. It reduced missed decline events at the default threshold. |
| SVM | oisst_only - canopy_only | -0.1454 | 0.0970 | 0.1757 | 0.9286 | -13 | decreased PR-AUC by -0.145; increased recall by +0.097; increased F1 by +0.176; reduced false negatives by 13. This contrasts environmental exposure variables alone with direct canopy-state monitoring. |
| SVM | canopy_noaa - oisst_only | 0.1047 | 0.1493 | 0.2080 | -0.0602 | -20 | increased PR-AUC by +0.105; increased recall by +0.149; increased F1 by +0.208; reduced false negatives by 20. This isolates the added value of combining canopy-state variables with NOAA exposure variables relative to NOAA-only inputs. |
| Random Forest | canopy_noaa - canopy_only | -0.0667 | -0.3134 | -0.2896 | 0.0170 | 42 | decreased PR-AUC by -0.067; decreased recall by -0.313; decreased F1 by -0.290; increased false negatives by 42. Adding NOAA context did not imply causal effects and should be read as feature-set augmentation. Aggregate PR-AUC did not improve within this algorithm. |
| Random Forest | oisst_only - canopy_only | -0.1930 | -0.2985 | -0.3022 | -0.1990 | 40 | decreased PR-AUC by -0.193; decreased recall by -0.299; decreased F1 by -0.302; increased false negatives by 40. This contrasts environmental exposure variables alone with direct canopy-state monitoring. |
| Random Forest | canopy_noaa - oisst_only | 0.1263 | -0.0149 | 0.0126 | 0.2159 | 2 | increased PR-AUC by +0.126; decreased recall by -0.015; increased F1 by +0.013; increased false negatives by 2. This isolates the added value of combining canopy-state variables with NOAA exposure variables relative to NOAA-only inputs. |
| XGBoost | canopy_noaa - canopy_only | -0.0508 | -0.0672 | -0.0480 | 0.0069 | 9 | decreased PR-AUC by -0.051; decreased recall by -0.067; decreased F1 by -0.048; increased false negatives by 9. Adding NOAA context did not imply causal effects and should be read as feature-set augmentation. Aggregate PR-AUC did not improve within this algorithm. |
| XGBoost | oisst_only - canopy_only | -0.1543 | -0.1418 | -0.1492 | -0.1452 | 19 | decreased PR-AUC by -0.154; decreased recall by -0.142; decreased F1 by -0.149; increased false negatives by 19. This contrasts environmental exposure variables alone with direct canopy-state monitoring. |
| XGBoost | canopy_noaa - oisst_only | 0.1036 | 0.0746 | 0.1012 | 0.1521 | -10 | increased PR-AUC by +0.104; increased recall by +0.075; increased F1 by +0.101; reduced false negatives by 10. This isolates the added value of combining canopy-state variables with NOAA exposure variables relative to NOAA-only inputs. |
| LightGBM | canopy_noaa - canopy_only | -0.0680 | -0.2090 | -0.1651 | 0.0360 | 28 | decreased PR-AUC by -0.068; decreased recall by -0.209; decreased F1 by -0.165; increased false negatives by 28. Adding NOAA context did not imply causal effects and should be read as feature-set augmentation. Aggregate PR-AUC did not improve within this algorithm. |
| LightGBM | oisst_only - canopy_only | -0.2099 | -0.2687 | -0.2626 | -0.1762 | 36 | decreased PR-AUC by -0.210; decreased recall by -0.269; decreased F1 by -0.263; increased false negatives by 36. This contrasts environmental exposure variables alone with direct canopy-state monitoring. |
| LightGBM | canopy_noaa - oisst_only | 0.1419 | 0.0597 | 0.0975 | 0.2122 | -8 | increased PR-AUC by +0.142; increased recall by +0.060; increased F1 by +0.097; reduced false negatives by 8. This isolates the added value of combining canopy-state variables with NOAA exposure variables relative to NOAA-only inputs. |
