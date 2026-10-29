"""Write the frozen task 15 live design from the stage 0 summary.

Fixes K by the stage 0 rule, the arm settings, the stage A engineering jobs, the stage B
dial screen in a seeded interleaved order (each block of five arms shares one task and
seed, blocks and arms within blocks shuffled with seed 15), the stage C tasks and seeds,
and the run ceilings from progress/15-regrasp-dial-plan.md.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TASKS = (0, 3, 6, 9)
STAGE_B_ARMS = ("native", "select-0.10", "select-0.06", "project-0.10", "always-0.10")


def arms(k):
    return {
        "native": {"mode": "native", "gate": "closure"},
        "select-0.10": {"mode": "select", "speed_limit_mps": 0.10, "candidates": k,
                        "gate": "closure"},
        "select-0.06": {"mode": "select", "speed_limit_mps": 0.06, "candidates": k,
                        "gate": "closure"},
        "project-0.10": {"mode": "project", "speed_limit_mps": 0.10, "gate": "closure"},
        "always-0.10": {"mode": "select", "speed_limit_mps": 0.10, "candidates": k,
                        "gate": "always"},
        "project-always-0.10": {"mode": "project", "speed_limit_mps": 0.10, "gate": "always"},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0", type=Path,
                        default=ROOT / "outputs/regrasp15/stage0/summary.json")
    parser.add_argument("--out", type=Path, default=ROOT / "configs/regrasp15/design.json")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite the frozen design: {args.out}")
    stage0 = json.loads(args.stage0.read_text())
    if not stage0.get("complete") or stage0["k"] != 16:
        raise SystemExit("Stage 0 must be complete with K = 16 decodes per record")
    rescue = stage0["limits"]["0.10"]["rescue_rate"]
    k = 16 if rescue is not None and rescue >= 0.5 else 32
    if k != stage0["live_k"]:
        raise SystemExit("Stage 0 summary disagrees with the frozen K rule")
    stage_a = [
        {"job": 0, "stage": "stage_a", "arm": "native", "task": 1, "seed": 150000},
        {"job": 1, "stage": "stage_a", "arm": "always-0.10", "task": 1, "seed": 150001},
        {"job": 2, "stage": "stage_a", "arm": "project-always-0.10", "task": 1, "seed": 150002},
        {"job": 3, "stage": "stage_a", "arm": "select-0.10", "task": 1, "seed": 150003},
    ]
    rng = np.random.default_rng(15)
    blocks = [(task, seed) for seed in range(151000, 151004) for task in TASKS]
    stage_b, number = [], 100
    for i in rng.permutation(len(blocks)):
        task, seed = blocks[i]
        for j in rng.permutation(len(STAGE_B_ARMS)):
            stage_b.append({"job": number, "stage": "stage_b", "arm": STAGE_B_ARMS[j],
                            "task": task, "seed": seed})
            number += 1
    design = {
        "schema": "regrasp15-design-v1",
        "k": k,
        "stage0": {"summary_sha256": hashlib.sha256(args.stage0.read_bytes()).hexdigest(),
                   "rescue_rate_0.10": rescue, "live_k": stage0["live_k"]},
        "arms": arms(k),
        "stage_a_jobs": stage_a,
        "stage_b_jobs": stage_b,
        "stage_c_tasks": list(TASKS),
        "stage_c_seeds": list(range(152000, 152012)),
        "episode_timeout_seconds": 2700,
        "ceilings": {"simulator_starts": 200, "replacements": 20, "attempts_per_job": 2,
                     "policy_controls": 420_000, "native_queries": 60_000,
                     "candidate_decodes": 120_000},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(design, indent=2) + "\n")
    print(json.dumps({"k": k, "stage_b_jobs": len(stage_b), "out": str(args.out)}))


if __name__ == "__main__":
    main()
