"""Bounded serial Part 2 execution. No full-episode retries or sample expansion."""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_followup_budget import atomic, clock, register, run_owned
from helpers.maniguard_part2_controller import conditions
from helpers.maniguard_part2_protocol import ENGINEERING, RUN, draft, freeze, plan, validate_frozen
from helpers.maniguard_part2_results import audit
from helpers.maniguard_phase3_protocol import sha


def save_new(path, value):
    if path.exists():
        raise FileExistsError("Immutable output already exists: " + str(path))
    atomic(path, value)


def ensure_policy():
    from helpers.maniguard_phase2_services import PolicyService, probe_policy_metadata

    require_guard()
    service_dir = RUN / "service"
    i = 1
    while service_dir.exists():
        i += 1
        service_dir = RUN / f"service-{i}"
    probe = lambda timeout: probe_policy_metadata(
        18000, timeout, python=ROOT / "data/maniguard/envs/sim/bin/python",
    )
    service = PolicyService(probe=probe).ensure_family(
        "clutter", service_dir, startup_timeout=600,
    )
    pid = int(subprocess.check_output(["supervisorctl", "pid", "maniguard_policy"], text=True))
    register(RUN, SimpleNamespace(pid=pid), "policy")
    return service


def require_guard():
    status = subprocess.check_output(
        ["supervisorctl", "status", "maniguard_followup_guard"], text=True,
    )
    if "RUNNING" not in status:
        raise RuntimeError("Independent deadline supervision required")


def launch(directory, job, protocol=None):
    require_guard()
    if clock()["physical_remaining"] < 630:
        raise RuntimeError("No allowance for a full bounded attempt and cleanup")
    attempt = directory / "attempts" / (job["episode_id"] + "-a1")
    planned = directory / "jobs" / (job["episode_id"] + ".json")
    save_new(planned, job)
    command = [
        "uv", "run", "--no-project", "--python",
        str(ROOT / "data/maniguard/envs/sim/bin/python"),
        str(ROOT / "experiments/scripts/maniguard_part2_episode.py"),
        "--job", str(planned), "--output", str(attempt / "data"),
    ]
    if protocol:
        command += ["--protocol", str(protocol)]
    env = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4",
               CUDA_VISIBLE_DEVICES="0")
    process = run_owned(command, attempt, "episode", timeout=600, registry=RUN, env=env)
    record = audit(attempt / "data")
    record["process"] = process
    if process["status"] != "completed" or process.get("returncode") != 0:
        record.update(valid=False, engineering_valid=False, U=None)
        record["errors"].append("Episode process failed or incomplete")
    save_new(attempt / "audit.json", record)
    print(json.dumps({"episode_id": job["episode_id"], "valid": record["valid"],
                      "steps": record.get("steps"), "errors": record["errors"]}), flush=True)
    return record


def engineering(directory):
    if (directory / "draft.json").exists():
        if json.loads((directory / "draft.json").read_text()) != draft():
            raise ValueError("Prospective configuration changed")
    else:
        save_new(directory / "draft.json", draft())
    ensure_policy()
    records = []
    for i, condition in enumerate(("A", "C", "D")):
        job = {"episode_id": f"part2-engineering-{condition}-checked", "part": 2,
               "purpose": "engineering", "condition": condition, "scene": ENGINEERING,
               "seed": 3900, "action_horizon": 32,
               "controller": asdict(conditions()[condition])}
        record = launch(directory, job)
        records.append(record)
        if not record["engineering_valid"]:
            save_new(directory / "engineering-incomplete.json", records)
            raise RuntimeError("Bounded engineering failed; preserve and diagnose")
    save_new(directory / "engineering.json", records)
    # Capture is equal across arms. Reject expensive capture before full episodes.
    if any(r["costs"]["passive_capture_seconds"] / 33 > 0.15 for r in records):
        raise RuntimeError("Passive capture overhead exceeds prospective .15 second/observation cap")
    freeze(directory / "protocol.json", records)
    schedule = plan()
    schedule["protocol_sha256"] = sha(directory / "protocol.json")
    save_new(directory / "jobs-plan.json", schedule)
    print("ENGINEERING_AND_FREEZE_COMPLETE", flush=True)


def comparison(directory):
    protocol = directory / "protocol.json"
    validate_frozen(protocol)
    schedule = json.loads((directory / "jobs-plan.json").read_text())
    if len(schedule["jobs"]) != 16:
        raise ValueError("The Part 2 allocation is exactly sixteen full episodes")
    records = []
    started = time.time()
    for job in schedule["jobs"]:
        # A completed immutable audit may be read after coordinator interruption;
        # an attempted incomplete slot is never silently retried.
        path = directory / "attempts" / (job["episode_id"] + "-a1") / "audit.json"
        if path.exists():
            record = json.loads(path.read_text())
        else:
            record = launch(directory, job, protocol)
        records.append(record)
        atomic(directory / "progress.json", {"records": records, "clock": clock()})
        if not record["valid"]:
            raise RuntimeError("Invalid full slot stops dependent comparison")
    save_new(directory / "results.json", {"records": records, "clock": clock(),
                                           "coordinator_seconds": time.time() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("engineering", "comparison"))
    args = parser.parse_args()
    directory = RUN / "part2"
    directory.mkdir(parents=True, exist_ok=True)
    (engineering if args.stage == "engineering" else comparison)(directory)


if __name__ == "__main__":
    main()
