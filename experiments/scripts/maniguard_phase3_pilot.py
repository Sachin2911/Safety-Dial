"""Execute only the prospective Part 1 pilot, with independent allocation supervision.

Each attempt has a new immutable directory. No automatic infrastructure retry or
outcome-dependent expansion is supported. Re-entry preserves the original allocation.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_phase3_budget import atomic, clock, register, run_owned
from helpers.maniguard_phase3_protocol import CONFIG, draft, freeze, sha, validate_frozen
from helpers.maniguard_phase3_results import audit
from helpers.maniguard_phase3_search import (
    ENGINEERING,
    VARIANTS,
    Evolution,
    comparison_jobs,
    select_allocation,
    training_jobs,
)

SIM = ROOT / "data/maniguard/envs/sim/bin/python"
EPISODE = ROOT / "experiments/scripts/maniguard_phase3_episode.py"


def save_new(path, value):
    if Path(path).exists():
        raise FileExistsError("Immutable pilot artifact already exists: " + str(path))
    atomic(path, value)


def ensure_guard():
    status = subprocess.check_output(
        ["supervisorctl", "status", "maniguard_part1_guard"], text=True
    )
    if "RUNNING" not in status:
        raise RuntimeError("Independent allocation guard is not running")


def ensure_policy(directory, attempt=1):
    from helpers.maniguard_phase2_services import PolicyService

    ensure_guard()
    if clock()["physical_remaining"] < 1800:
        raise RuntimeError("Too little physical budget for service startup and engineering")
    service_dir = "service" if attempt == 1 else f"service-{attempt}"
    receipt = PolicyService().ensure_family("clutter", directory / service_dir, startup_timeout=600)
    pid = int(subprocess.check_output(["supervisorctl", "pid", "maniguard_policy"], text=True))
    register(directory, SimpleNamespace(pid=pid), "policy")
    return receipt


def launch(directory, job, protocol=None):
    if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,100}", job["episode_id"]):
        raise ValueError("Episode ID must be a short path-free identifier")
    ensure_guard()
    if clock()["physical_remaining"] < 1230:
        raise RuntimeError("Not enough time for declared timeout plus cleanup")
    attempt = directory / "attempts" / (job["episode_id"] + "-a1")
    planned = directory / "jobs" / (job["episode_id"] + ".json")
    save_new(planned, job)
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        str(SIM),
        str(EPISODE),
        "--job",
        str(planned),
        "--output",
        str(attempt / "data"),
    ]
    if protocol is not None:
        command += ["--protocol", str(protocol)]
    env = dict(
        os.environ,
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        CUDA_VISIBLE_DEVICES="0",
    )
    began = time.monotonic()
    process = run_owned(command, attempt, "episode", registry=directory, env=env)
    return attempt, process, began


def audited(attempt, process, began):
    audit_started = time.monotonic()
    result = audit(attempt / "data")
    result["process"] = process
    result["spawn_through_audit_seconds"] = process["elapsed_seconds"] + (
        time.monotonic() - audit_started
    )
    if process["status"] != "completed" or process.get("returncode") != 0:
        result.update(valid=False, engineering_valid=False, U=None)
        result["errors"].append("Process did not exit successfully")
    save_new(attempt / "audit.json", result)
    return result


def engineering(directory, attempt_number=1):
    ensure_policy(directory, attempt_number)
    checks = []
    for i, dial in enumerate((0, 0.5, 1)):
        job = {
            "episode_id": f"engineering-{i}"
            if attempt_number == 1
            else f"engineering-r{attempt_number}-{i}",
            "purpose": "engineering",
            "scene": ENGINEERING,
            "seed": 3000,
            "genome": [3, 3, 2, 0],
            "dial": dial,
            "action_horizon": 32,
        }
        attempt, process, began = launch(directory, job)
        result = audited(attempt, process, began)
        checks.append(result)
        if not result["engineering_valid"]:
            save_new(directory / f"engineering-incomplete-{attempt_number}.json", checks)
            raise RuntimeError(
                "Engineering attempt invalid; preserve costs and diagnose " + str(attempt)
            )
    save_new(directory / "engineering.json", checks)
    return checks


def stage_fits(episodes, t_plan):
    if episodes * t_plan + 300 > clock()["physical_remaining"]:
        raise RuntimeError("Remaining planned work no longer fits physical deadline")


def run(directory, protocol):
    validate_frozen(protocol)
    ensure_guard()
    engine = Evolution()
    schedule = np.random.default_rng(20261019)
    variant = None
    remaining = 48
    t_plan = 417
    for generation in range(3):
        stage_fits(8 if generation == 0 else remaining, t_plan)
        population = engine.ask()
        jobs = training_jobs(generation, population, schedule)
        save_new(
            directory / f"population-{generation}-plan.json",
            {
                "population": population,
                "jobs": jobs,
                "evolution_rng": engine.rng.bit_generator.state,
                "schedule_rng": schedule.bit_generator.state,
                "protocol_sha256": sha(protocol),
            },
        )
        records = []
        timings = []
        for job in jobs:
            attempt, process, began = launch(directory, job, protocol)
            result = audited(attempt, process, began)
            records.append((job, result))
            if not result["valid"]:
                raise RuntimeError("Invalid/incomplete physical evidence stops the population")
            # The auditor derives and seals labels, but neither the selector nor the
            # optimizer consumes them until the timing-only decision is saved.
            timings.append(result["spawn_through_audit_seconds"])
        if generation == 0:
            decision = select_allocation(timings, clock()["physical_remaining"])
            save_new(directory / "allocation-selection.json", decision)
            variant, t_plan = decision["variant"], decision["t_plan"]
            if variant is None:
                # Preserve exploratory population evidence without starting another.
                return {"status": "insufficient_time_after_population_zero"}
            remaining = variant
        outcomes = [[] for _ in range(4)]
        for job, result in records:
            outcomes[job["slot"]].append(result)
        ranking = engine.tell(population, outcomes)
        remaining -= 8
        save_new(
            directory / f"population-{generation}-result.json",
            {
                "population": population,
                "ranking": ranking,
                "outcomes": outcomes,
                "rng_state": engine.rng.bit_generator.state,
            },
        )
        if generation + 1 == VARIANTS[variant][0]:
            break
    finalist = engine.finalist(population, ranking)
    finalist.update(
        protocol_sha256=sha(protocol),
        frozen_at=time.time(),
        sources=json.loads(Path(protocol).read_text())["sources"],
    )
    save_new(directory / "finalist.json", finalist)
    if finalist["identity_selected"]:
        return {"status": "identity_selected", "comparison": "predeclared stop"}
    comparisons = comparison_jobs(variant, finalist["genome"], schedule)
    save_new(
        directory / "comparison-plan.json",
        {
            "jobs": comparisons,
            "schedule_rng": schedule.bit_generator.state,
            "finalist_sha256": sha(directory / "finalist.json"),
        },
    )
    for i in range(0, len(comparisons), 3):
        stage_fits(len(comparisons) - i, t_plan)
        for job in comparisons[i : i + 3]:
            attempt, process, began = launch(directory, job, protocol)
            result = audited(attempt, process, began)
            if not result["valid"]:
                raise RuntimeError("Invalid comparison evidence; keep incomplete block")
    return {"status": "completed", "variant": variant}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("action", choices=("draft", "engineering", "freeze", "run"))
    parser.add_argument("--engineering-attempt", type=int, default=1)
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.action == "draft":
        save_new(CONFIG, draft())
    elif args.action == "engineering":
        engineering(directory, args.engineering_attempt)
    elif args.action == "freeze":
        checks = json.loads((directory / "engineering.json").read_text())
        freeze(directory / "protocol.json", checks)
    else:
        try:
            result = run(directory, directory / "protocol.json")
        except (OSError, ValueError, RuntimeError) as error:
            atomic(
                directory / "terminal.json",
                {"status": "incomplete", "reason": str(error), "clock": clock()},
            )
            raise
        else:
            atomic(directory / "terminal.json", {**result, "clock": clock()})


if __name__ == "__main__":
    main()
