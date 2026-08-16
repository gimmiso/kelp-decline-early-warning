#!/usr/bin/env python3
"""Rebuild only the region holdout after fixing within-year rank normalization."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_analysis_module():
    source = Path(__file__).with_name("75_run_coastwide_model_architecture_extension.py")
    spec = importlib.util.spec_from_file_location("coastwide_architecture", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    module = load_analysis_module()
    manifest_path = args.output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing completed output manifest: {manifest_path}")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    domains = module.prepare_data(args.panel, args.environment, config)
    predictions, audit = module.spatiotemporal_region_holdout(domains, config)
    annual, increments = module.summarize_holdout(predictions, config)
    rebuilt = {
        "spatiotemporal_region_holdout_predictions.csv": predictions,
        "spatiotemporal_region_holdout_audit.csv": audit,
        "spatiotemporal_region_holdout_annual.csv": annual,
        "spatiotemporal_region_holdout_increments.csv": increments,
    }
    for name, frame in rebuilt.items():
        frame.to_csv(args.output_dir / name, index=False)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in rebuilt:
        manifest["outputs"][name] = module.sha256(args.output_dir / name)
    manifest.setdefault("repairs", []).append(
        {
            "reason": "Ranker scores in the multi-year region holdout must be normalized within deployment year.",
            "scope": sorted(rebuilt),
            "method": "Recomputed every region-holdout family and feature set from source data; no outer-test model selection.",
        }
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(increments.to_string(index=False))


if __name__ == "__main__":
    main()
