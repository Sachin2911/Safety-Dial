"""Audit the complete frozen comparison and describe every matched initial observation.

This reads existing evidence only. It does not select settings, change the sample,
or replay simulator physics. Incomplete evidence raises instead of imputing outcomes.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def build(run, audit_dir):
    run, audit_dir = Path(run), Path(audit_dir)
    plan = read(run / "comparison-plan.json")
    finalist_path = run / "finalist.json"
    finalist = read(finalist_path)
    require(
        hashlib.sha256(finalist_path.read_bytes()).hexdigest() == plan["finalist_sha256"],
        "Finalist changed after comparison was sealed",
    )
    rows = [
        r
        for r in read(audit_dir / "attempt-ledger.json")["attempts"]
        if r.get("purpose") == "validation"
    ]
    n = len(plan["jobs"])
    require(n in (12, 24), "Unexpected comparison allocation")
    require(len(rows) == n, "Missing or extra comparison attempts")
    require(all(r["valid"] and r["full_episode"] for r in rows), "Invalid comparison evidence")
    lookup = {r["episode_id"]: r for r in rows}
    require(len(lookup) == n, "Duplicate comparison episode IDs")
    require(set(lookup) == {j["episode_id"] for j in plan["jobs"]}, "Unplanned comparison")
    actual_order = sorted(rows, key=lambda r: r["process"]["started_at"])
    require(
        [r["episode_id"] for r in actual_order] == [j["episode_id"] for j in plan["jobs"]],
        "Comparison order differs from frozen schedule",
    )
    require(
        finalist["frozen_at"] < actual_order[0]["process"]["started_at"],
        "Finalist was not frozen before comparison",
    )
    blocks, permutations = [], []
    for i in range(0, n, 3):
        jobs = plan["jobs"][i : i + 3]
        permutation = tuple(j["dial"] for j in jobs)
        permutations.append(permutation)
        require(set(permutation) == {0, 0.5, 1}, "Incomplete three-setting block")
        require(len({(j["scene"], j["seed"], j["block"]) for j in jobs}) == 1, "Unmatched block")
        selected = {}
        for job in jobs:
            row = lookup[job["episode_id"]]
            stored_job = read(Path(row["output"]) / "job.json")
            require(all(stored_job[k] == v for k, v in job.items()), "Job differs from plan")
            require(row["genome"] == finalist["genome"], "Different comparison controller")
            horizon = {0: 8, 0.5: 12, 1: 16}[row["dial"]]
            # This pilot selected a horizon-16 finalist; refuse a misleading general audit.
            require(finalist["decoded"]["horizon"] == 16, "Different anchor horizon")
            require(
                row["query_count"] == (row["steps"] + horizon - 1) // horizon,
                "Unexpected query schedule",
            )
            if row["dial"] == 0:
                require(row["changed_commands"] == 0, "Zero dial altered an action")
            with np.load(Path(row["output"]) / "initial_observation.npz") as initial:
                selected[row["dial"]] = {k: initial[k].copy() for k in initial.files}
        differences = []
        for dial in (0.5, 1):
            metrics = {}
            for key in ("state", "overview", "wrist"):
                zero, other = selected[0][key], selected[dial][key]
                delta = np.abs(zero.astype(float) - other.astype(float))
                metrics[key] = {
                    "exact_equal": bool(np.array_equal(zero, other)),
                    "mean_absolute_difference": float(delta.mean()),
                    "max_absolute_difference": float(delta.max()),
                }
            differences.append({"dial_against_zero": dial, "metrics": metrics})
        blocks.append(
            {
                "block": jobs[0]["block"],
                "scene": jobs[0]["scene"],
                "seed": jobs[0]["seed"],
                "order": list(permutation),
                "safe_completion_by_dial": {
                    str(j["dial"]): lookup[j["episode_id"]]["S_req"] for j in jobs
                },
                "violation_by_dial": {
                    str(j["dial"]): lookup[j["episode_id"]]["V"] for j in jobs
                },
                "initial_observation_differences": differences,
            }
        )
    require(len(set(permutations[:6])) == min(6, len(permutations)), "Permutation cycle changed")
    result = {
        "valid": True,
        "comparison_episodes": n,
        "matched_blocks": len(blocks),
        "finalist_sha256": plan["finalist_sha256"],
        "frozen_before_comparison": True,
        "job_and_execution_order_match": True,
        "same_anchor_all_settings": True,
        "zero_changed_commands": 0,
        "position_counts_by_dial": {
            str(d): dict(Counter(p.index(d) for p in permutations)) for d in (0, 0.5, 1)
        },
        "blocks": blocks,
        "interpretation": "Matched scene/seeds; initial rendering equality is measured, not assumed.",
        "physics_replayed": False,
    }
    target = audit_dir / "matched-independent-audit.json"
    require(not target.exists(), "Refusing to overwrite independent audit")
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "blocks"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("audit_dir", type=Path)
    args = parser.parse_args()
    build(args.run, args.audit_dir)
