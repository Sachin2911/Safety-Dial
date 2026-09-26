#!/usr/bin/env python3
"""Finalize a completed local S2 bundle after its final upload failed.

Verifies every export against optimizer.pt on CPU, retries the private upload, and
writes the missing final receipt/report. Never runs optimization or regenerates mixed
exports. Run only after the original training process has exited.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
from helpers.hfStore import HFStore
from helpers.walkerTrainingFinalize import finalize_completed_run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    report_path = REPO_ROOT / "docs/mainPlan/results/s2" / f"{args.run_dir.name}.json"
    report = finalize_completed_run(args.run_dir, report_path, HFStore())
    print(json.dumps({"run_id": report["run_id"], "steps": report["steps"],
                      "optimizer_steps_executed": 0, "revision": report["checkpoint"]["revision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
