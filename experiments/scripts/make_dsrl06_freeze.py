"""Freeze reviewed task-6 source and build its credential-free bootstrap archive."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import FinancialBook, atomic_json, utc
from helpers.dsrl06_learner import LearnerConfig
from helpers.dsrl06_protocol import CAPS, frozen_jobs

FREEZE = ROOT / "progress/receipts/06-dsrl-stage-a.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files():
    names = {
        "AGENTS.md",
        "firstPrinciples.md",
        "progress/06-dsrl-plan.md",
        "progress/panels.json",
        "progress/receipts/01-sources.json",
        "progress/receipts/01-jar-eval.yaml",
        "progress/receipts/01b-env-behavior-freeze.txt",
        "scripts/stage2/prepare_vast_host.sh",
    }
    for directory in (
        "experiments",
        "configs/dsrl06",
        "scripts/todo1",
        "scripts/todo1b",
        "scripts/todo2",
    ):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".sh", ".json", ".txt", ".yaml"}:
                names.add(str(path.relative_to(ROOT)))
    for path in (ROOT / "progress/receipts").glob("06-dsrl-*"):
        if path.is_file() and path != FREEZE and "stage-b" not in path.name:
            names.add(str(path.relative_to(ROOT)))
    for directory in (
        "outputs/dsrl06-preflight/metadata",
        "outputs/dsrl06-preflight/learner",
        "outputs/dsrl06-preflight/actor",
    ):
        names.update(str(p.relative_to(ROOT)) for p in (ROOT / directory).glob("*") if p.is_file())
    names.add("outputs/dsrl06-preflight/decoder-probe-vectors.npy")
    return sorted(names)


def create_freeze(args):
    if FREEZE.exists():
        raise ValueError(
            "Stage A is immutable; record an explicit amendment instead of overwriting"
        )
    tests = json.loads(args.verification.read_text())
    if tests.get("passed") is not True:
        raise ValueError("Passing final source checks are required")
    names = source_files()
    hashes = {name: digest(ROOT / name) for name in names}
    expected_receipt = "669c5e16bf6a67c80350510a4d658323748ff9033682f148166c1a1a85efe079"
    if hashes["outputs/dsrl06-preflight/learner/cpu-source-parity.json"] != expected_receipt:
        raise ValueError("Prior engineering receipt changed")
    freeze = {
        "schema": "dsrl06-stage-a-v1",
        "frozen_utc": utc(),
        "authorization": {
            "user_launch": True,
            "instruction": "Implement progress/06-dsrl-plan.md; Vast rental and SSH authorized",
            "hard_total_usd": 15,
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": hashes,
        "verification": tests,
        "money": FinancialBook.limits(),
        "caps": dict(CAPS),
        "jobs": frozen_jobs(),
        "learner_config": LearnerConfig().resolved(),
        "prior_engineering_updates": 2,
        "prior_engineering_receipt": "sha256:" + expected_receipt,
        "actor_evaluation_mode": "cpu_published_actor",
        "actor_cpu_parity": {"atol": 1e-5, "rtol": 1e-5, "max_observations": 8},
        "engineering_profile": {
            "timed_updates": 128,
            "source_parity_update_calls": 2,
            "source_parity_maximum_state_difference": 1e-6,
            "topology": "Entire fixed-batch profile interval overlaps all three rollout processes",
            "learner_compile_barrier": "E3 ended and E0-E2 each returned their first policy query",
        },
        "shape_audit": "progress/receipts/06-dsrl-shape-audit.json",
        "decoder_probe_spec": "progress/receipts/06-dsrl-decoder-probe.json",
        "stop_initiation_before_nominal_seconds": 300,
        "engineering_deadline_minutes_from_working_host_creation": 90,
        "first_episode_wall_minutes": 40,
        "later_episode_wall_minutes": 25,
        "warm_start_reserve_minutes": 45,
        "cold_start_reserve_minutes": 60,
        "preservation_reserve_minutes": 20,
        "runtime_shape_backstop": [16, 32],
        "information_boundary": ["overview_rgb64", "native_robot_state8"],
        "terminal_bootstrap": "mask0 and potential0 on success and timeout; deliberate adaptation",
        "simulator_reset_seed": None,
        "simulator_seed_note": "Benchmark seeds NumPy and policy using CRC32; no explicit reset seed",
        "prior_simulator_starts": 0,
        "prior_model_queries": 0,
        "admission": "Only engineering; stage B needs positive authority and its own frozen projection",
    }
    atomic_json(FREEZE, freeze)
    print(
        json.dumps(
            {
                "freeze": str(FREEZE.relative_to(ROOT)),
                "sha256": digest(FREEZE),
                "source_files": len(hashes),
            }
        )
    )


def bundle(args):
    freeze = json.loads(FREEZE.read_text())
    for name, expected in freeze["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    rental = ROOT / "progress/rental-dsrl06-20261025.json"
    allocation = json.loads(rental.read_text())
    if not isinstance(allocation.get("instance_id"), int):
        raise ValueError("Resolve the allocated instance before bundling its bootstrap")
    names = sorted(
        {*freeze["source_sha256"], str(FREEZE.relative_to(ROOT)), str(rental.relative_to(ROOT))}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError("Do not overwrite a transferred bootstrap bundle")
    with tarfile.open(args.output, "w:gz") as archive:
        for name in names:
            path = ROOT / name
            if path.is_symlink():
                raise ValueError("Source archive must contain regular files")
            archive.add(path, arcname=name, recursive=False)
    print(
        json.dumps(
            {
                "bundle": str(args.output),
                "sha256": digest(args.output),
                "files": len(names),
                "bytes": args.output.stat().st_size,
            }
        )
    )


def pilot_freeze(args):
    if args.output.exists():
        raise ValueError("Stage B cannot overwrite an existing freeze")
    plan = json.loads(args.engineering_freeze.read_text())
    for name, expected in plan["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    summary = json.loads(args.summary.read_text())
    rental = json.loads((ROOT / "progress/rental-dsrl06-20261025.json").read_text())
    if not (
        summary.get("admit") is True
        and summary.get("authority_admit") is True
        and summary.get("topology_three_rollouts") is True
        and summary["instance_id"] == rental["instance_id"]
        and summary["plan_sha256"] == digest(args.engineering_freeze)
    ):
        raise ValueError("Engineering does not admit the development pilot")
    import datetime as dt

    remaining = (
        dt.datetime.fromisoformat(rental["allocation_deadline_utc"])
        - dt.datetime.now(dt.timezone.utc)
    ).total_seconds()
    if summary["projected_stage_b_seconds_including_preservation"] > remaining:
        raise ValueError("Measured pilot projection no longer fits the remaining allowance")
    plan.update(
        {
            "schema": "dsrl06-stage-b-v1",
            "stage": "B",
            "frozen_utc": utc(),
            "stage_a_freeze_sha256": digest(args.engineering_freeze),
            "stage_a_summary_sha256": digest(args.summary),
            "engineering_projection": summary["projected_stage_b_seconds_including_preservation"],
            "remaining_allocation_seconds_at_freeze": remaining,
            "admission": "Frozen 24 training episodes followed by final-checkpoint 48 evaluations",
        }
    )
    atomic_json(args.output, plan)
    print(json.dumps({"freeze": str(args.output), "sha256": digest(args.output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("freeze")
    prepare.add_argument("--verification", type=Path, required=True)
    archive = sub.add_parser("bundle")
    archive.add_argument(
        "--output", type=Path, default=ROOT / "outputs/dsrl06/control/bootstrap-source.tar.gz"
    )
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--engineering-freeze", type=Path, default=FREEZE)
    pilot.add_argument("--summary", type=Path, required=True)
    pilot.add_argument(
        "--output", type=Path, default=ROOT / "progress/receipts/06-dsrl-stage-b.json"
    )
    args = parser.parse_args()
    if args.action == "freeze":
        create_freeze(args)
    elif args.action == "pilot":
        pilot_freeze(args)
    else:
        bundle(args)


if __name__ == "__main__":
    main()
