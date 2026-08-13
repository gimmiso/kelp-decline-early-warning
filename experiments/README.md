# Experiment logging

All `KELP-EWS-E2E-v2.0` experiments must be registered before execution.

- Never overwrite a run directory.
- A failed or invalidated run remains in the registry.
- Raw data stay outside Git; immutable source manifests and checksums stay in Git.
- Every result must be traceable to a Git SHA, config SHA-256, input SHA-256, exact command, environment, and seed.
- Pilot results already inspected before protocol lock are labeled `pilot_complete_not_confirmatory`.
- An unplanned analysis must be registered as `exploratory` before it runs and cannot replace a confirmatory result.

Expected run directory:

```text
outputs/experiments/<run_id>/
  manifest.json
  command.txt
  run.log
  quality_checks.csv
  fold_audit.csv
  predictions.parquet
  metrics.csv
  bootstrap_differences.csv
  decision.json
```

The `decision.json` file records which pre-specified scenario in the protocol applies and the exact allowed/blocked claims.
