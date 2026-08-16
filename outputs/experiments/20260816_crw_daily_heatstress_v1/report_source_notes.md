# Report source notes and chart map

Audience: technical. Delivery surface: validated Data Analytics MCP report artifact. The report was validated before one successful render; no parallel HTML report was created.

## Required report roles

- Title: `일별 CRW 열스트레스 증분가치 확증`
- Technical summary: primary and beyond-monthly claim gates
- Key findings: algorithm robustness, annual heterogeneity, monitoring-budget value, feature redundancy
- Scope and definitions: cohort, forecast clock, outcome, sample, AP lift
- Methods: four locked features, expanding window, three fixed algorithms, moving-block bootstrap
- Limitations and validation: source coverage, official overlap, 19 independent checks, proxy and timing limits
- Recommended next steps: stop further same-support CRW feature search; test genuinely different information support
- Further questions: nowcast timing, high-resolution/subsurface SST, regional heterogeneity

## Chart map

| Report section | Analytical question | Form | Dataset | Supported takeaway |
|---|---|---|---|---|
| Algorithm robustness | Does any model family gain from richer CRW blocks? | Grouped bar | `model_family_summary.csv` | No model family shows stable daily or monthly increment |
| Annual heterogeneity | Is the Logistic daily increment direction consistent by year? | Signed annual bar | `model_family_year_metrics.csv` | Positive and negative years mix; mean remains near zero |
| Monitoring budget | Does daily CRW improve recall at 5–30% survey budgets? | Grouped bar | `model_family_summary.csv` | Recall curves nearly overlap and paired CIs cross zero |
| Feature redundancy | Are daily summaries independent from monthly CRW? | Ranked bar | `feature_spearman_correlations.csv` | Correlations are uniformly high, consistent with signal redundancy |

Palette policy: hard two-root cap plus neutrals for grouped comparisons; single-root plus a neutral zero reference for signed annual differences and ranked correlations. Group identity is also carried by direct legend labels, not color alone.

The artifact snapshot contains 76 aggregate rows across six datasets, below the 2,000-row-per-dataset and 3 MB limits. Exact audit detail remains in the run directory rather than the reader-facing report.

All six SQLite source queries recorded in the artifact were executed against the shaped report tables after a targeted alias correction. `report_query_validation.csv` records the expected 1, 12, 20, 16, 15 and 12 rows respectively, with zero null cells. The complete corrected artifact passed validation and rendering.
