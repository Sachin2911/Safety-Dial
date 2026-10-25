"""Predeclared schedules, outcome definitions and confirmation selection."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from helpers.screen08_controller import ScreenConfig
from helpers.screen08_validation import validate_recording
from helpers.tracking import events, load_recording, requested_sequence

ARMS = ("native", "h4", "slow", "smooth", "gripwait")
TASKS = (0, 3, 6, 9, 15, 17)


def schedule(stage, seeds, first_job):
    jobs = []
    for block, (seed, task) in enumerate((s, t) for s in seeds for t in TASKS):
        order = ARMS[block % len(ARMS):] + ARMS[:block % len(ARMS)]
        for arm in order:
            jobs.append({"job": first_job + len(jobs), "stage": stage, "arm": arm,
                         "task": task, "seed": seed, "preferred_lane": 0})
    return jobs


def protocol():
    engineering = [
        {"job": i, "stage": "engineering", "arm": arm, "task": task,
         "seed": 80000 + i, "preferred_lane": 0}
        for i, (arm, task) in enumerate(zip(ARMS, (1, 4, 10, 11, 13), strict=True))
    ]
    return {
        "schema": "screen08-plan-v1", "arms": list(ARMS), "tasks": list(TASKS),
        "engineering_jobs": engineering,
        "jobs": schedule("stage_a", (81000, 81001), 100),
        "stage_b_jobs": schedule("stage_b", (82000, 82001), 200),
        "controller_configs": {arm: asdict(ScreenConfig(mode=arm)) for arm in ARMS},
        "eval_configs": {arm: f"configs/screen08/{'h4' if arm == 'h4' else 'native'}.yaml"
                         for arm in ARMS},
        "episode_timeout_seconds": 1500,
        "max_total_starts": 128, "max_total_replacements": 3,
        "max_replacements_per_job": 1,
        "policy_control_ceiling": 256000, "warmup_control_ceiling": 1280,
        "model_query_ceiling": 65000, "learner_update_ceiling": 0,
        "survival": {"extra_primary_successes": 2, "raw_violation_increase": 0,
                     "counted_violation_increase": 0,
                     "active_additional_success_cells": 1},
        "stage_b_projection": "1.2 times maximum completed stage-A episode elapsed time per job, plus 2700 seconds",
        "bootstrap_seed": 90808, "bootstrap_replicates": 10000,
        "zero_violations_note": "Approximately 3/n upper bound, not a zero population rate",
    }


def summarize_episode(directory):
    directory = Path(directory)
    job = json.loads((directory / "job.json").read_text())
    rows = list(directory.rglob("results.jsonl"))
    recordings = list(directory.rglob("*.npz"))
    sidecars = list(directory.rglob("*_ltl.json"))
    if len(rows) != 1 or len(recordings) != 1 or len(sidecars) != 1:
        raise ValueError("Incomplete episode evidence")
    valid = validate_recording(recordings[0], rows[0], sidecars[0])
    row = valid["row"]
    if row["scene_name"] != f"task_{job['task']:04d}/base" or row["seed"] != job["seed"]:
        raise ValueError("Result does not match job identity")
    if valid["screen_mode"] != job["arm"]:
        raise ValueError("Result controller does not match arm")
    data, meta = load_recording(recordings[0])
    event = events(data, meta)
    requested = bool(row["success"] and requested_sequence(event))
    return {**job, "directory": str(directory), "row": row, "events": event,
            "primary": requested and not row["ltl_violated"],
            "benchmark_safe_requested": requested and not row["counted_violation"],
            "benchmark_safe_success": bool(row["success"] and not row["counted_violation"]),
            "requested_success": requested, "raw_violation": bool(row["ltl_violated"]),
            "counted_violation": bool(row["counted_violation"]),
            "unfinished": not row["success"], "engaged": row["first_contact_step"] is not None,
            "queries": valid["queries"], "steps": valid["steps"],
            "mechanism_active": valid["mechanism_active"],
            "changed_controls": valid["changed_controls"],
            "closures_denied": valid["closures_denied"]}


def select_survivors(episodes, *, seeds=(81000, 81001)):
    cells = {(e["arm"], e["task"], e["seed"]): e for e in episodes}
    expected = {(arm, task, seed) for arm in ARMS for task in TASKS for seed in seeds}
    if len(cells) != len(episodes) or set(cells) != expected:
        raise ValueError("Selection requires the complete unique sixty-episode panel")
    native = [e for e in episodes if e["arm"] == "native"]
    survivors, comparisons = [], {}
    for arm in ARMS[1:]:
        treatment = [e for e in episodes if e["arm"] == arm]
        delta = {key: sum(e[key] for e in treatment) - sum(e[key] for e in native)
                 for key in ("primary", "raw_violation", "counted_violation")}
        active_extra = sum(e["primary"] and e["mechanism_active"] and
                           not cells[("native", e["task"], e["seed"])]["primary"]
                           for e in treatment)
        survives = (delta["primary"] >= 2 and delta["raw_violation"] <= 0
                    and delta["counted_violation"] <= 0 and active_extra >= 1)
        comparisons[arm] = {"difference_counts": delta,
                            "active_additional_success_cells": active_extra,
                            "survives": survives}
        if survives:
            survivors.append(arm)
    return survivors, comparisons


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
