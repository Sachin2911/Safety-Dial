"""Freeze task-10 source and schedules; never overwrite an earlier freeze.

Stage A freezes the engineering panel. Stage B sizes its seed panel from the measured
stage A throughput and the remaining rental window. Stage C picks the largest ladder rung
that fits. All three are deterministic functions of inputs recorded in the freeze.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.safecem10_protocol import design, digest, schedule_A, schedule_B, schedule_B2


def sources():
    paths = set()
    for directory, pattern in (
        ("experiments/helpers", "safecem10*.py"),
        ("experiments/scripts", "prepare_safecem10.py"),
        ("experiments/scripts", "worker_safecem10.py"),
        ("experiments/scripts", "run_safecem10.py"),
        ("experiments/scripts", "freeze_safecem10.py"),
        ("experiments/scripts", "admit_safecem10.py"),
        ("experiments/scripts", "safecem10_watchdog.py"),
        ("experiments/scripts", "retrain_probe_safecem10.py"),
        ("experiments/scripts", "bootstrap_safecem10.sh"),
        ("experiments/scripts", "setup_safecem10.sh"),
        ("experiments/tests", "test_safecem10.py"),
        ("configs/safecem10", "*"),
    ):
        paths.update(p for p in (ROOT / directory).glob(pattern) if p.is_file())
    paths.add(ROOT / "experiments/helpers/dsrl06_infra.py")
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def remaining_seconds(rental_path, spec):
    rental = json.loads(Path(rental_path).read_text())
    deadline = dt.datetime.fromisoformat(rental["allocation_deadline_utc"])
    margin = spec["money"]["preservation_margin_minutes"] * 60
    return (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds() - margin


def size_b(spec, engineering, available, imagination):
    stage = spec["stages"]["B"]
    quantile_count = len(
        [
            q
            for q in spec["imagination"]["quantiles"]
            if not any(
                abs(round(imagination["radial_quantiles_px"][str(q)]) - g) < 2
                for g in stage["dials"]
            )
            and round(imagination["radial_quantiles_px"][str(q)]) <= 80
        ]
    )
    per_seed = len(stage["layouts"]) * (
        len(stage["dials"]) + quantile_count + len(stage["penalty_lams"])
    )
    seconds = engineering["system_seconds_per_episode"]
    fits = int(available * 0.85 / (per_seed * seconds))
    seeds = min(len(stage["seeds"]), fits)
    if seeds < 5:
        raise ValueError("Fewer than five seeds fit the remaining window; do not freeze B")
    return stage["seeds"][:seeds], per_seed


def size_b2(spec, engineering, available, jobs):
    seconds = engineering["system_seconds_per_episode"] * len(jobs)
    if seconds > available * 0.85:
        raise ValueError("Stage B2 does not fit the remaining window; do not freeze it")
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
    parser.add_argument(
        "--stage", choices=["preparation", "A", "B", "C", "B2", "C2"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engineering", type=Path)
    parser.add_argument("--imagination", type=Path)
    parser.add_argument(
        "--probe-receipt",
        type=Path,
        help="probe-v2.json from the retraining step; its sha256 is pinned into the design",
    )
    parser.add_argument(
        "--rental", type=Path, default=ROOT / "progress/rental-safecem10-20261025.json"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Freeze already exists; choose a fresh version path")
    spec = design()
    stage = "A" if args.stage == "preparation" else args.stage
    engineering = None
    imagination = None
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
        if stage in ("B2", "C2"):
            if not args.probe_receipt:
                raise ValueError("Grounded stages need the probe v2 receipt")
            receipt = json.loads(args.probe_receipt.read_text())
            spec["probe_v2"]["sha256"] = receipt["probe_sha256"]
            spec["probe_v2"]["receipt_sha256"] = digest(args.probe_receipt)
        if stage == "B2":
            imagination = json.loads(args.imagination.read_text())
            jobs = schedule_B2(spec, imagination)
            sizing = size_b2(spec, engineering, available, jobs)
        elif stage == "B":
            imagination = json.loads(
                (args.imagination or ROOT / "outputs/safecem10/remote/imagination.json").read_text()
            )
            seeds, per_seed = size_b(spec, engineering, available, imagination)
            spec["stages"]["B"]["seeds"] = seeds
            jobs = schedule_B(spec, imagination)
            sizing = {"seeds": seeds, "jobs_per_seed": per_seed, "available_seconds": available}
        else:
            rung = size_c(spec, engineering, available)
            sizing = {**rung, "available_seconds": available}
            plan = {
                "utc": utc(),
                "generations": rung["generations"],
                "replicates": rung["replicates"],
                "seconds_per_episode": engineering["seconds_per_episode_per_worker"],
                "workers_measured": engineering["workers"],
                "episodes": rung["episodes"],
            }
            sizing["plan"] = plan
    record = {
        "schema": "safecem10-freeze-v1",
        "utc": utc(),
        "stage": args.stage,
        "authorization": {"user_launch": True, "persistent_engineering_repairs": True},
        "money": spec["money"],
        "source_sha256": sources(),
        "design": spec,
        "jobs": jobs,
        "sizing": sizing,
        "engineering_acceptance_sha256": digest(args.engineering) if args.engineering else None,
        "imagination": imagination,
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
