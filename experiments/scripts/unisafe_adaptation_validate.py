"""Validate training finalists, then freeze candidates before reserved test episodes."""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe_source import ROOT, save_implementation, sha256


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", type=Path, nargs="+", required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/unisafe/adaptation-study.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    save_implementation(args.output)
    write(args.output / "protocol.json", protocol)
    candidates, seen = [], set()
    for root in args.search:
        assert json.loads((root / "summary.json").read_text())["status"] == "complete"
        for path in sorted(root.glob("*/finalists.json")):
            state = json.loads(path.read_text())
            method = state["method"]
            for rank, adapter in enumerate(state["finalists"]):
                identity = method, json.dumps(adapter, sort_keys=True)
                if identity in seen:
                    continue
                seen.add(identity)
                candidates.append({
                    "name": f"{path.parent.name}-rank{rank}", "method": method,
                    "adapter": adapter, "source": str(path), "source_sha256": sha256(path),
                })
    for i, adapter in enumerate(protocol["fixed_candidates"]):
        candidates.append({"name": f"fixed-rank{i}", "method": "fixed", "adapter": adapter})
    candidates.append({
        "name": "baseline", "method": "baseline",
        "adapter": {"hold_steps": 0, "min_close_steps": 3, "min_aperture": .005},
    })
    rng = np.random.default_rng(412071)
    episodes = []
    for seed in protocol["validation_seeds"]:
        for i in rng.permutation(len(candidates)):
            candidate = candidates[i]
            episodes.append({"seed": seed, "name": candidate["name"], "adapter": candidate["adapter"]})
    plan = {
        "purpose": "Validation only; parameter selection precedes reserved tests",
        "adapter_kind": protocol["adapter_kind"], "record_cameras": False, "episodes": episodes,
    }
    write(args.output / "candidates.json", candidates)
    write(args.output / "plan.json", plan)
    started = time.perf_counter()
    with (args.output / "worker.log").open("w") as log:
        subprocess.run([
            sys.executable, "-u", str(ROOT / "experiments/scripts/unisafe_adaptation_worker.py"),
            "--plan", str((args.output / "plan.json").resolve()),
            "--output", str((args.output / "rollouts").resolve()),
        ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    measured = json.loads((args.output / "rollouts/summary.json").read_text())
    for candidate in candidates:
        trials = [e for e in measured["episodes"] if e["job"]["name"] == candidate["name"]]
        assert len(trials) == len(protocol["validation_seeds"])
        candidate["outcomes"] = dict(Counter(e["outcome"] for e in trials))
        candidate["violations"] = sum(e["physical_failure"] or e["boundary_exit"] for e in trials)
        candidate["mean_utility"] = float(np.mean([e["utility"] for e in trials]))
        candidate["environment_steps"] = sum(e["steps"] for e in trials)
    baseline = next(c for c in candidates if c["method"] == "baseline")
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["method"]].append(candidate)
    selected = []
    for method, options in grouped.items():
        feasible = [c for c in options if c["violations"] <= baseline["violations"]]
        pool = feasible or [c for c in options if c["violations"] == min(x["violations"] for x in options)]
        best = min(pool, key=lambda c: (-c["mean_utility"], json.dumps(c["adapter"], sort_keys=True)))
        selected.append({**best, "meets_observed_validation_safety_screen": bool(feasible)})
    result = {
        "status": "complete", "candidates": candidates, "selected": selected,
        "selection_rule": protocol["validation_selection_rule"],
        "validation_seeds": protocol["validation_seeds"],
        "environment_steps": measured["environment_steps"], "physics_steps": measured["physics_steps"],
        "query_counts": measured["query_counts"], "episodes": len(measured["episodes"]),
        "pretrained_weight_updates": 0, "elapsed_seconds": time.perf_counter() - started,
        "worker_elapsed_seconds_overlapping": measured["elapsed_seconds"],
        "reserved_test_observations_used": 0,
        "limits": "Observed validation failure screening is not a safety guarantee. If no candidate meets it, the least-violating candidate is retained and explicitly flagged for diagnostic evaluation.",
    }
    write(args.output / "summary.json", result)
    for candidate in selected:
        print(candidate["method"], candidate["adapter"], candidate["outcomes"], flush=True)


if __name__ == "__main__":
    main()
