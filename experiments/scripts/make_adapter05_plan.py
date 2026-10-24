"""Freeze fresh jobs and controller settings before any adapter05 simulator start."""

import collections
import datetime as dt
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.execution_adapter import AdapterConfig


def make_plan():
    tasks, seeds = [0, 3, 6, 9, 15, 17], [400, 401]
    arms = ["native", "fixed", "adaptive"]
    jobs = []
    for seed_index, seed in enumerate(seeds):
        for task_index, task in enumerate(tasks):
            block = seed_index * len(tasks) + task_index
            for offset in range(3):
                arm_index = (block + offset) % 3
                jobs.append(
                    {
                        "job": f"p{len(jobs):02d}",
                        "stage": "pilot",
                        "task": task,
                        "seed": seed,
                        "arm": arms[arm_index],
                        "preferred_lane": (block + arm_index) % 2,
                    }
                )
    assert collections.Counter((j["arm"], j["preferred_lane"]) for j in jobs) == {
        (arm, lane): 6 for arm in arms for lane in range(2)
    }
    return {
        "schema": "adapter05-v1",
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "progress/05-execution-adapter.md",
        "tasks": tasks,
        "policy_seeds": seeds,
        "arms": arms,
        "family": "jar",
        "condition": "base",
        "max_episode_steps": 2000,
        "action_frequency": 20,
        "execute_horizon": 8,
        "chunk_length": 16,
        "camera_resolution": 256,
        "source_profile": "legacy",
        "goal_hold_steps": 10,
        "max_total_starts": 40,
        "max_replacements_per_job": 1,
        "max_total_replacements": 2,
        "max_policy_controls": 80000,
        "max_explicit_warmup_controls": 400,
        "episode_timeout_seconds": 1500,
        "job_admission_rental_hours": 4.5,
        "max_rental_hours": 6,
        "offer_rate_ceiling_usd_per_hour": 0.75,
        "maniguard_revision": "6bf4c61a1f168cc46e02eb22c41bcb51815e9726",
        "openpi_revision": "215abfb217dbac7d5f1273282331b9b1866c0479",
        "checkpoint_revision": "1d84eda070313a202a595e449fcf41a1a1e8a546",
        "dataset": "Sachioster/maniguard-jar-replication-20261022",
        "adapter_configs": {arm: asdict(AdapterConfig(mode=arm)) for arm in arms},
        "engineering_jobs": [
            {
                "job": "e0",
                "stage": "engineering",
                "task": 1,
                "seed": 390,
                "arm": "native",
                "preferred_lane": 0,
            },
            {
                "job": "e1",
                "stage": "engineering",
                "task": 10,
                "seed": 391,
                "arm": "adaptive",
                "preferred_lane": 1,
            },
        ],
        "jobs": jobs,
    }


def main():
    plan = make_plan()
    paths = [
        "experiments/helpers/execution_adapter.py",
        "experiments/helpers/adapter05_runtime.py",
        "experiments/helpers/adapter05_validation.py",
        "experiments/helpers/tracking.py",
        "experiments/helpers/accounted_recorder.py",
        "scripts/todo2/record_rollout.py",
        "experiments/scripts/run_adapter05.py",
        "experiments/scripts/patch_adapter05.py",
        "experiments/scripts/launch_adapter05.sh",
        "experiments/scripts/analyze_adapter05.py",
        "experiments/scripts/analyze_tracking.py",
        "scripts/todo2/timeline.py",
        "scripts/todo2/frames.py",
        "progress/05-execution-adapter.md",
        "progress/receipts/01-jar-eval.yaml",
        "scripts/todo1b/serve_jar_lane.sh",
        "scripts/todo1/probe_server.py",
    ]
    plan["source_sha256"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}
    path = ROOT / "progress/receipts/05-adapter-plan.json"
    with path.open("x") as stream:
        json.dump(plan, stream, indent=2)
        stream.write("\n")
    print("Frozen", path, "with", len(plan["jobs"]), "scored jobs")


if __name__ == "__main__":
    main()
