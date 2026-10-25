"""Bind the task-7 learning schedule to completed numerical and live validation."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl07_protocol import CAPS, frozen_jobs


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_b_plan(root, engineering_freeze, summary_path, rental_path, *, now=None):
    root = Path(root)
    plan = json.loads(Path(engineering_freeze).read_text())
    if (
        plan.get("authorization", {}).get("user_launch") is not True
        or plan["jobs"] != frozen_jobs()
        or plan["caps"] != dict(CAPS)
        or plan["money"]["total_usd"] != 15
    ):
        raise ValueError("Task-7 authorization, schedule or cap differs")
    for name, expected in plan["source_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    summary = json.loads(Path(summary_path).read_text())
    rental = json.loads(Path(rental_path).read_text())
    numerical_path = Path(summary_path).with_name("numerical-summary.json")
    numerical = json.loads(numerical_path.read_text())
    if not (
        summary.get("admit") is True
        and summary.get("source_parity_valid") is True
        and summary.get("cpu_actor_parity", {}).get("valid") is True
        and summary["instance_id"] == rental["instance_id"]
        and summary["plan_sha256"] == digest(engineering_freeze)
        and summary["numerical_summary_sha256"] == digest(numerical_path)
        and numerical.get("admit_live_integration") is True
        and numerical.get("same_process_pass") is True
        and numerical.get("native_draw_passthrough_pass") is True
        and numerical.get("influence_present") is True
        and numerical["learner_mode"] == summary["learner_mode"]
        and numerical["learner_mode"] in ("deterministic", "deterministic-no-autotune")
    ):
        raise ValueError("Completed engineering does not admit the learning comparison")
    now = now or dt.datetime.now(dt.timezone.utc)
    remaining = (dt.datetime.fromisoformat(rental["allocation_deadline_utc"]) - now).total_seconds()
    projection = summary["projected_stage_b_seconds_including_preservation"]
    if not 0 < projection <= remaining:
        raise ValueError("Measured pilot projection no longer fits the remaining allowance")
    for name in (
        "experiments/scripts/freeze_dsrl07_pilot.py",
        "experiments/tests/test_dsrl07_pilot_freeze.py",
    ):
        plan["source_sha256"][name] = digest(root / name)
    plan.update(
        {
            "schema": "dsrl07-stage-b-v1",
            "stage": "B",
            "frozen_utc": now.isoformat(),
            "stage_a_freeze_sha256": digest(engineering_freeze),
            "stage_a_summary_sha256": digest(summary_path),
            "numerical_summary_sha256": digest(numerical_path),
            "learner_mode": numerical["learner_mode"],
            "engineering_projection": projection,
            "remaining_allocation_seconds_at_freeze": remaining,
            "admission": "Frozen 24 training episodes followed by final-checkpoint 48 evaluations",
        }
    )
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engineering-freeze", type=Path, default=ROOT / "progress/receipts/07-dsrl-stage-a.json"
    )
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "outputs/dsrl07/stage-a-summary.json"
    )
    parser.add_argument(
        "--rental", type=Path, default=ROOT / "progress/rental-dsrl07-20261025.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "progress/receipts/07-dsrl-stage-b.json"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Stage-B freezes are immutable")
    plan = stage_b_plan(ROOT, args.engineering_freeze, args.summary, args.rental)
    payload = (json.dumps(plan, indent=2, allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"freeze": str(args.output), "sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
