"""Freeze selected candidates into a reproducible fresh-process test plan."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe_source import ROOT, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/unisafe/adaptation-study.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    validation_path = args.validation / "summary.json"
    validation = json.loads(validation_path.read_text())
    assert validation["status"] == "complete" and validation["reserved_test_observations_used"] == 0
    interval = protocol["reserved_test_seeds"]
    seeds = list(range(interval["start"], interval["stop_exclusive"]))
    assert not set(seeds) & set(protocol["training_seeds"] + protocol["validation_seeds"] + protocol["development_seeds"])
    assert {c["method"] for c in validation["selected"]} == {
        "baseline", "fixed", "evolution", "random", "policy_gradient"
    }, "All declared comparison methods must be selected before test"
    jobs = []
    for candidate in validation["selected"]:
        method = candidate["method"]
        for mode in ("filtered", "unfiltered"):
            if mode == "unfiltered" and method not in ("baseline", "evolution"):
                continue
            selected_seeds = seeds if mode == "filtered" else seeds[:protocol["unfiltered_test_seed_count"]]
            name = method if mode == "filtered" else method + "-unfiltered"
            for seed in selected_seeds:
                jobs.append({
                    "name": f"{name}-seed{seed:04d}",
                    "plan": {
                        "purpose": "Reserved evaluation; one episode in a fresh simulator process",
                        "adapter_kind": protocol["adapter_kind"], "record_cameras": True,
                        "episodes": [{
                            "name": name, "seed": seed, "mode": mode,
                            "adapter": candidate["adapter"],
                        }],
                    },
                })
    rng = np.random.default_rng(6505)
    rng.shuffle(jobs)
    files = [
        "experiments/scripts/unisafe_adaptation_worker.py",
        "experiments/scripts/unisafe_adaptation_batch.py",
        "experiments/helpers/unisafe_adaptation.py",
        "experiments/helpers/unisafe_controller.py",
        "experiments/helpers/unisafe_model.py",
        "experiments/helpers/unisafe_environment.py",
        "experiments/helpers/unisafe_source.py",
        "configs/unisafe/baseline.json",
    ]
    result = {
        "workers": 1, "scope": "64 reserved seeds per selected filtered controller; 32 unfiltered ablation seeds",
        "validation": str(validation_path), "validation_sha256": sha256(validation_path),
        "selected": validation["selected"], "reserved_seeds": seeds,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "frozen_sources_sha256": {p: sha256(ROOT / p) for p in files},
        "environment_step_cap": len(jobs) * 300, "jobs": jobs,
        "limits": "Held out from this adaptation and selection study; overlap with the released checkpoint's original training data is unknown. Source hashes are checked before each job. Compare observed prefixes rather than assume simulator snapshot restoration.",
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print("Frozen test jobs:", len(jobs), "maximum control steps:", result["environment_step_cap"])


if __name__ == "__main__":
    main()
