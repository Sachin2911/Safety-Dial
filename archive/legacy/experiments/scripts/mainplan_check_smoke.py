#!/usr/bin/env python3
"""Evaluate the declared LeWM smoke checks before scheduling full training."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def assess_smoke(report):
    history = report.get("history", [])
    reasons = []
    if report.get("steps", 0) < 1000:
        reasons.append("fewer than 1,000 optimizer steps")
    if len(history) < 10:
        reasons.append("too few logged loss measurements")
    if not report.get("render_validation", {}).get("passed"):
        reasons.append("missing successful nonblack render check")
    roles = report.get("episode_splits", {})
    training, validation = set(roles.get("training", [])), set(roles.get("validation", []))
    if not training or not validation or training & validation:
        reasons.append("training/validation source episodes are not disjoint")
    trends = {}
    for name in ("pred_loss", "sigreg_loss"):
        values = np.asarray([row.get(name, math.nan) for row in history])
        if len(values) < 10 or not np.isfinite(values).all():
            reasons.append(f"missing/nonfinite {name}")
            continue
        width = max(3, len(values) // 5)
        first, last = float(values[:width].mean()), float(values[-width:].mean())
        trends[name] = {"first_window_mean": first, "last_window_mean": last, "window_measurements": width}
        if last >= first:
            reasons.append(f"{name} did not fall over the smoke run")
    speeds = [row.get("it_per_s", 0) for row in history[-10:]]
    speed = float(np.median(speeds)) if speeds else 0
    estimate = (10 * report["steps_per_epoch"] / speed) if speed > 0 and report.get("steps_per_epoch") else None
    return {"gate": {"passes": not reasons, "reasons": reasons}, "loss_trends": trends,
            "observed_iterations_per_second": speed,
            "estimated_ten_epoch_seconds_at_observed_rate": estimate,
            "estimate_note": "Measured smoke throughput; full-run data loading and contention may differ"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("--out must not already exist")
    source = Path(__file__).resolve().parents[2] / "docs/mainPlan/results/s2" / f"{args.run_dir.name}.json"
    report = json.loads(source.read_text())
    result = {"run_dir": str(args.run_dir), "source_report": str(source), **assess_smoke(report)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["gate"]["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
