# Seminar Paper Results Summary

## Dataset and Target Summary

This project constructed a cell-year early-warning dataset for kelp canopy decline using Kelpwatch satellite-derived canopy observations and NOAA environmental exposure indicators. The spatial design used a regular 10 km x 10 km fishnet across the Northern and Central California coastal corridor. The initial design contained 285 candidate fishnet cells. After Kelpwatch historical-footprint filtering, 74 exploratory cells had `count_cells_historic_footprint > 0`, and 50 main modeling cells met the stricter `count_cells_historic_footprint >= 500` threshold.

The annual response panel used Kelpwatch `quarter = max` rows, interpreted as growing-season maximum canopy observations. The main target, `decline_event_next`, was defined as whether the next-year growing-season maximum relative canopy fell below the cell-specific historical 25th percentile from the 1984-2013 baseline period. The complete-feature modeling period was 1989-2024, yielding 1,800 cell-year rows across 50 cells. Models were evaluated with a temporal split: training on 1989-2016, validation on 2017-2020, and final test evaluation on 2021-2024.

## Model Comparison Summary

Five supervised classification algorithms were compared to evaluate whether kelp canopy decline was better captured by linear additive effects, nonlinear decision boundaries, or tree-based feature interactions:

- **Logistic Regression:** transparent linear baseline.
- **SVM:** nonlinear margin-based classifier for a relatively small tabular dataset.
- **Random Forest:** robust nonlinear tree baseline and SHAP-compatible model.
- **XGBoost:** boosted-tree benchmark.
- **LightGBM:** efficient boosted-tree benchmark.

The models were evaluated across three feature sets:

- `canopy_only`: direct biological state variables from Kelpwatch canopy observations.
- `oisst_only`: NOAA OISST thermal exposure variables alone.
- `canopy_noaa`: canopy variables combined with OISST, CUTI, BEUTI, and spatial/context variables.

The temporal split was used to reduce look-ahead bias and approximate forward prediction. Because the task is framed as early-warning screening, PR-AUC, recall, F1, and false negatives were emphasized rather than accuracy alone.

### Compact Final Result Table

| Item | Result |
|---|---|
| Best overall model | `canopy_only / Random Forest` |
| Best overall test PR-AUC | `0.8974` |
| Best overall model recall / F1 / false negatives | Recall `0.5896`; F1 `0.7149`; false negatives `55` |
| Best canopy+NOAA model | `canopy_noaa / SVM` |
| Best canopy+NOAA test PR-AUC | `0.8459` |
| Best canopy+NOAA recall / F1 / false negatives | Recall `0.2463`; F1 `0.3837`; false negatives `101` |
| SHAP-interpreted canopy-only model | `canopy_only / Random Forest` |
| SHAP-interpreted canopy+NOAA model | `canopy_noaa / Random Forest` |
| Main within-model feature-set result | `canopy_noaa` did not improve PR-AUC over `canopy_only` within the tested algorithms. |
| Main screening nuance | For SVM, `canopy_noaa` improved recall and F1 and reduced false negatives by `33` relative to `canopy_only` at the default threshold. |
| Main NOAA-context finding | NOAA variables provide environmental exposure context rather than replacing direct canopy monitoring. |

The best aggregate predictive performance came from `canopy_only / Random Forest` with test PR-AUC `0.8974`. The best canopy+NOAA model was `canopy_noaa / SVM` with test PR-AUC `0.8459`. Therefore, canopy+NOAA did not outperform canopy-only in aggregate PR-AUC. This distinction is important: NOAA variables contributed interpretive environmental context, but the strongest short-term predictive signal came from current and lagged canopy-state variables.

## Within-Model Feature-Set Comparison

Within-model feature-set comparisons were used to isolate the effect of changing feature sets while holding the algorithm fixed. This comparison matters because an across-model leaderboard alone can confound algorithm choice with feature-set choice.

Across all five tested algorithms, adding NOAA environmental variables to canopy variables did not improve PR-AUC over canopy-only. The deltas for `canopy_noaa - canopy_only` were negative for Logistic Regression, SVM, Random Forest, XGBoost, and LightGBM. However, performance changes varied by metric. The most notable screening-oriented nuance was observed for SVM: adding canopy+NOAA features relative to canopy-only decreased PR-AUC by `0.0407`, but increased recall by `0.2463`, increased F1 by `0.3837`, and reduced false negatives by `33` at the default threshold.

OISST-only models were generally weaker than canopy-only models, indicating that thermal exposure variables alone did not replace direct canopy-state monitoring. In contrast, canopy+NOAA models generally improved over OISST-only models, showing that NOAA exposure variables were more informative when interpreted alongside canopy state than when used alone.

## Canopy Persistence Findings

Canopy persistence analysis supported the interpretation that current canopy condition is a strong short-term signal. Current relative canopy and next-year relative canopy had a correlation of `0.610`. The lowest current-canopy quintile had a next-year decline rate of `0.633`, compared with `0.303` for the highest current-canopy quintile. This pattern helps explain why canopy-only models performed strongly: current canopy state already integrates recent ecological history, disturbance, recovery, and biological condition.

This result does not imply that environmental variables are irrelevant. Rather, it suggests that direct canopy observations are the strongest short-term predictors for next-year canopy decline in the current dataset and temporal split.

## NOAA Environmental Context Findings

NOAA OISST, CUTI, and BEUTI variables provided environmental exposure context. Decline rows showed directional differences in selected environmental variables, including higher `annual_mean_sst_anomaly` and lower `cuti_anomaly` relative to non-decline rows. BEUTI-related signals also differed, but these should be interpreted cautiously as proxy-based and context-dependent.

The environmental variables should not be interpreted as causal proof. OISST is assigned using a nearest valid ocean-grid approach in Version 1. CUTI is interpreted as a coastal upwelling transport proxy, and BEUTI is interpreted as a nitrate-flux proxy. CUTI/BEUTI are assigned by nearest latitude bin and are not direct cell-level nutrient measurements.

## SHAP Interpretation Findings

SHAP interpretation was performed for two Random Forest models: `canopy_only / Random Forest` and `canopy_noaa / Random Forest`. Although SVM achieved the best canopy+NOAA PR-AUC, full Kernel SHAP for SVM was not used because it can be slower and less stable. Random Forest models were selected for TreeExplainer-based interpretation.

The canopy-only Random Forest relied entirely on canopy-state variables by construction. Grouped SHAP importance for `canopy_only / Random Forest` assigned `100%` of grouped importance to canopy variables.

For `canopy_noaa / Random Forest`, grouped SHAP importance showed substantial internal model use of environmental exposure variables:

```text
OISST: 38.6%
BEUTI: 25.4%
CUTI: 18.0%
canopy: 15.6%
spatial: 2.2%
region: 0.3%
```

These SHAP results explain fitted model behavior, not ecological causality. Some SHAP dependence patterns were nonlinear or directionally mixed, so NOAA variables are best interpreted as environmental-context indicators rather than simple monotonic causal drivers.

## Final Interpretation

Current canopy condition was the strongest short-term predictor of next-year kelp decline, reflecting temporal persistence in canopy state. NOAA OISST, CUTI, and BEUTI variables did not replace direct canopy monitoring and did not improve aggregate PR-AUC over canopy-only models within the tested algorithms. However, they provided interpretable environmental exposure context related to thermal exposure and upwelling/nitrate-flux proxy conditions.

The results support a two-layer interpretation. First, biological state monitoring through Kelpwatch canopy observations provides the strongest short-term predictive signal. Second, NOAA environmental variables help characterize the environmental context in which canopy decline risk is modeled, especially when interpreted through diagnostics and SHAP rather than as standalone causal drivers.

## Limitations

- The main modeling subset contains a small number of retained cells (`50` cells).
- Final test evaluation is limited to 2021-2024.
- The temporal split does not fully test spatial generalization.
- OISST features use nearest valid ocean-grid assignment in Version 1.
- CUTI/BEUTI features use nearest latitude-bin assignment.
- The workflow does not include direct cell-level nutrient measurements.
- Grazing pressure, urchin observations, sea star wasting disease, and other biotic disturbance variables are not included.
- Environmental interpretation is proxy-based.
- The workflow supports early-warning screening, not causal attribution.

Future work should evaluate spatial or grouped cross-validation, compare nearest-grid OISST assignment with coastal-buffer averages, add ecological covariates such as grazing, wave disturbance, and disease context where available, and estimate uncertainty using bootstrap confidence intervals for model metrics.
