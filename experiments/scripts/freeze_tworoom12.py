"""Freeze task-12 source and schedules; never overwrite an earlier freeze.

Stage A freezes the engineering panel. Stage B checks that the dial sweep fits the remaining
rental window at the measured stage A throughput. Stage C picks the largest ladder rung that
fits. All are deterministic functions of inputs recorded in the freeze.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.tworoom12_protocol import design, digest, schedule_A, schedule_B


def sources():
    paths = set()
    for directory, pattern in (
        ("experiments/helpers", "tworoom12*.py"),
        ("experiments/scripts", "prepare_tworoom12.py"),
        ("experiments/scripts", "worker_tworoom12.py"),
        ("experiments/scripts", "run_tworoom12.py"),
        ("experiments/scripts", "freeze_tworoom12.py"),
        ("experiments/scripts", "admit_tworoom12.py"),
        ("experiments/scripts", "tworoom12_watchdog.py"),
        ("experiments/scripts", "bootstrap_tworoom12.sh"),
        ("experiments/scripts", "setup_tworoom12.sh"),
        ("experiments/tests", "test_tworoom12.py"),
        ("configs/tworoom12", "*"),
    ):
        paths.update(p for p in (ROOT / directory).glob(pattern) if p.is_file())
    # Shared code the task-12 runtime imports; frozen here so a later edit invalidates it.
    paths.add(ROOT / "experiments/helpers/safecem10_planner.py")
    paths.add(ROOT / "experiments/helpers/dsrl06_infra.py")
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def remaining_seconds(rental_path, spec):
    rental = json.loads(Path(rental_path).read_text())
    deadline = dt.datetime.fromisoformat(rental["allocation_deadline_utc"])
    margin = spec["money"]["preservation_margin_minutes"] * 60
    return (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds() - margin


def size_fixed(engineering, available, jobs, stage):
    seconds = engineering["system_seconds_per_episode"] * len(jobs)
    if seconds > available * 0.85:
        raise ValueError(f"Stage {stage} does not fit the remaining window; do not freeze it")
    return {"jobs": len(jobs), "seconds_estimate": seconds, "available_seconds": available}


def size_c(spec, engineering, available):
    stage = spec["stages"]["C"]
    seconds = engineering["system_seconds_per_episode"]
    P = stage["population"]
    for rung in stage["ladder"]:
        search = len(stage["methods"]) * rung["replicates"] * P * rung["generations"]
        search *= stage["eval_seeds_per_candidate"]
        finals = (
            len(stage["methods"])
            * rung["replicates"]
            * stage["front_cap"]
            * len(stage["final_seeds"])
            * 2
        )
        total = (search + finals) * seconds
        if total <= available * 0.85:
            return {**rung, "episodes": search + finals, "seconds_estimate": total}
    raise ValueError("No ladder rung fits the remaining window; do not freeze C")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["preparation", "A", "B", "C"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engineering", type=Path)
    parser.add_argument(
        "--rental", type=Path, default=ROOT / "progress/rental-tworoom12-20261025.json"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Freeze already exists; choose a fresh version path")
    spec = design()
    stage = "A" if args.stage == "preparation" else args.stage
    engineering = None
    sizing = None
    jobs = []
    if stage == "A":
        jobs = schedule_A(spec)
    else:
        if not args.engineering:
            raise ValueError("Scored stages need the engineering acceptance record")
        engineering = json.loads(args.engineering.read_text())
        if not engineering["passed"]:
            raise ValueError("Engineering did not pass")
        available = remaining_seconds(args.rental, spec)
        if stage == "B":
            jobs = schedule_B(spec)
            sizing = size_fixed(engineering, available, jobs, "B")
        else:
            rung = size_c(spec, engineering, available)
            sizing = {**rung, "available_seconds": available}
            sizing["plan"] = {
                "utc": utc(),
                "generations": rung["generations"],
                "replicates": rung["replicates"],
                "seconds_per_episode": engineering["seconds_per_episode_per_worker"],
                "workers_measured": engineering["workers"],
                "episodes": rung["episodes"],
            }
    record = {
        "schema": "tworoom12-freeze-v1",
        "utc": utc(),
        "stage": args.stage,
        "authorization": {"user_launch": True, "persistent_engineering_repairs": True},
        "money": spec["money"],
        "source_sha256": sources(),
        "design": spec,
        "jobs": jobs,
        "sizing": sizing,
        "engineering_acceptance_sha256": digest(args.engineering) if args.engineering else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, record)
    print(
        json.dumps(
            {
                "freeze": str(args.output),
                "sha256": digest(args.output),
                "source_files": len(record["source_sha256"]),
                "jobs": len(jobs),
                "sizing": sizing,
            }
        )
    )


if __name__ == "__main__":
    main()
