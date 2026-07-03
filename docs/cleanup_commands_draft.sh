#!/usr/bin/env bash

# DO NOT RUN WITHOUT HUMAN REVIEW.
# This file is a draft cleanup checklist only.
# Every command below is intentionally commented out.
# Review docs/repo_cleanup_audit.md and docs/main_paper_scope.md before using any command.

# -----------------------------------------------------------------------------
# 1. Create future archive/supplementary folders
# -----------------------------------------------------------------------------

# DO NOT RUN WITHOUT HUMAN REVIEW:
# mkdir -p archive/exploratory
# mkdir -p archive/legacy_notebooks
# mkdir -p archive/legacy_src
# mkdir -p docs/supplementary

# -----------------------------------------------------------------------------
# 2. Possible README split
# -----------------------------------------------------------------------------

# DO NOT RUN WITHOUT HUMAN REVIEW:
# git mv README.md docs/full_workflow_log.md
# cp docs/main_paper_readme_template.md README.md

# Alternative safer approach:
# DO NOT RUN WITHOUT HUMAN REVIEW:
# cp README.md docs/full_workflow_log.md
# Then manually rewrite README.md as a concise paper-facing overview.

# -----------------------------------------------------------------------------
# 3. Possible notebook archive
# -----------------------------------------------------------------------------

# DO NOT RUN WITHOUT HUMAN REVIEW:
# git mv notebooks/01_Kelpwatch_Panel_Construction archive/legacy_notebooks/
# git mv notebooks/01_kelpwatch_panel_construction.ipynb archive/legacy_notebooks/
# git mv notebooks/02_decline_label_construction.ipynb archive/legacy_notebooks/
# git mv notebooks/02_oisst_sst_feature_engineering.ipynb archive/legacy_notebooks/
# git mv notebooks/03_decline_label_construction.ipynb archive/legacy_notebooks/
# git mv notebooks/04_model_comparison.ipynb archive/legacy_notebooks/
# git mv notebooks/04_modeling_xgboost_shap.ipynb archive/legacy_notebooks/
# git mv notebooks/05_model_diagnostics.ipynb archive/legacy_notebooks/
# git mv notebooks/06_canopy_environment_context_analysis.ipynb archive/legacy_notebooks/
# git mv notebooks/07_shap_interpretation.ipynb archive/legacy_notebooks/

# -----------------------------------------------------------------------------
# 4. Possible src archive if confirmed unused
# -----------------------------------------------------------------------------

# DO NOT RUN WITHOUT HUMAN REVIEW:
# git mv src/data_loader.py archive/legacy_src/
# git mv src/feature_engineering.py archive/legacy_src/
# git mv src/labeling.py archive/legacy_src/
# git mv src/modeling.py archive/legacy_src/
# git mv src/visualization.py archive/legacy_src/

# -----------------------------------------------------------------------------
# 5. Possible exploratory script grouping
# -----------------------------------------------------------------------------

# DO NOT RUN WITHOUT HUMAN REVIEW:
# mkdir -p scripts/exploratory
# git mv scripts/09_build_multiscale_environmental_features.py scripts/exploratory/
# git mv scripts/10_multiscale_exposure_selection.py scripts/exploratory/
# git mv scripts/14_ecological_data_feasibility_scan.py scripts/exploratory/
# git mv scripts/16_build_crw_5km_sst_features.py scripts/exploratory/
# git mv scripts/16a_download_crw5km_point_cache.py scripts/exploratory/
# git mv scripts/16b_build_crw5km_composite_features.py scripts/exploratory/
# git mv scripts/17_build_bathymetry_habitat_features.py scripts/exploratory/
# git mv scripts/24_rare_event_alert_learning.py scripts/exploratory/
# git mv scripts/25_build_wave_exposure_features.py scripts/exploratory/
# git mv scripts/27_spatial_failure_repair.py scripts/exploratory/
# git mv scripts/28_multihorizon_actionable_warning.py scripts/exploratory/
# git mv scripts/29_quarterly_actionable_warning_feasibility.py scripts/exploratory/

# Warning: moving scripts will break README commands and imports unless paths are updated.

# -----------------------------------------------------------------------------
# 6. Possible supplementary output grouping
# -----------------------------------------------------------------------------

# DO NOT RUN WITHOUT HUMAN REVIEW:
# mkdir -p docs/supplementary/tables
# mkdir -p docs/supplementary/figures
# git mv results/tables/crw5km_* docs/supplementary/tables/
# git mv results/tables/bathymetry_* docs/supplementary/tables/
# git mv results/tables/wave_* docs/supplementary/tables/
# git mv results/tables/quarterly_* docs/supplementary/tables/
# git mv results/tables/multihorizon_* docs/supplementary/tables/

# Warning: moving outputs will break existing report links unless README/docs are updated.

# -----------------------------------------------------------------------------
# 7. Possible local references ignore rule
# -----------------------------------------------------------------------------

# DO NOT RUN WITHOUT HUMAN REVIEW:
# printf '\n# Local literature PDFs\nreferences/\nreferences.zip\n' >> .gitignore

# If selected citation metadata should remain tracked, prefer:
# DO NOT RUN WITHOUT HUMAN REVIEW:
# printf '\n# Local literature PDFs\nreferences/**/*.pdf\nreferences.zip\n' >> .gitignore

# -----------------------------------------------------------------------------
# 8. Possible cleanup of accidentally tracked generated detail tables
# -----------------------------------------------------------------------------

# DO NOT RUN WITHOUT HUMAN REVIEW:
# git rm --cached results/tables/rare_event_topk_alert_evaluation.csv
# git rm --cached results/tables/rare_event_threshold_tuning.csv

# Warning: this removes files from Git tracking. Only do this after deciding
# whether large detail tables should live in a release artifact or supplement.

# -----------------------------------------------------------------------------
# 9. Paper-freeze manifest placeholder
# -----------------------------------------------------------------------------

# DO NOT RUN WITHOUT HUMAN REVIEW:
# touch docs/paper_freeze_manifest.md
# git add docs/paper_freeze_manifest.md

# -----------------------------------------------------------------------------
# 10. Safety checks before any real cleanup commit
# -----------------------------------------------------------------------------

# DO NOT RUN WITHOUT HUMAN REVIEW:
# git status --short
# git diff --stat
# git diff --cached --stat
# python3 -m py_compile scripts/*.py

