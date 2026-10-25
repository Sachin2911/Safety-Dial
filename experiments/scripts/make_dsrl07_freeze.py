"""Freeze task-7 inputs, source, accounting and the fresh development schedule."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.dsrl06_learner import LearnerConfig
from helpers.dsrl07_infra import FinancialBook
from helpers.dsrl07_protocol import CAPS, frozen_jobs
from make_dsrl06_freeze import source_files

FREEZE = ROOT / "progress/receipts/07-dsrl-stage-a.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_inputs():
    target = ROOT / "outputs/dsrl07/inputs"
    manifest = target / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        for row in [*data["snapshots"], data["profile"]]:
            assert digest(ROOT / row["path"]) == row["sha256"]
        return
    target.mkdir(parents=True, exist_ok=True)
    audit = json.loads((ROOT / "progress/receipts/06-dsrl-decoder-audit.json").read_text())
    rows = []
    for row in audit["snapshots"]:
        relative = Path(row["factual_artifact_path"]).relative_to(
            "/workspace/Safety-Dial/outputs/dsrl06"
        )
        source = ROOT / "outputs/dsrl06/remote" / relative
        if digest(source) != row["factual_artifact_sha256"]:
            raise ValueError("Original factual observation hash differs")
        destination = target / (row["snapshot_id"].replace(":", "-") + ".npz")
        shutil.copyfile(source, destination)
        rows.append(
            {
                "path": str(destination.relative_to(ROOT)),
                "sha256": digest(destination),
                "snapshot_id": row["snapshot_id"],
                "task": row["task"],
                "historical_lane": row["factual_policy_gpu_lane"],
            }
        )
    source = ROOT / "outputs/dsrl06/remote/engineering-inputs.npz"
    destination = target / "learner-profile.npz"
    shutil.copyfile(source, destination)
    atomic_json(
        manifest,
        {
            "schema": "dsrl07-saved-inputs-v1",
            "created_utc": utc(),
            "snapshots": rows,
            "profile": {"path": str(destination.relative_to(ROOT)), "sha256": digest(destination)},
            "source_audit_sha256": digest(ROOT / "progress/receipts/06-dsrl-decoder-audit.json"),
            "role": "Numerical diagnosis only; these observations never enter training replay",
        },
    )


def freeze(verification):
    if FREEZE.exists():
        raise ValueError("Source freezes are immutable; record a repair amendment")
    prepare_inputs()
    checks = json.loads(verification.read_text())
    if checks.get("passed") is not True:
        raise ValueError("Passing checks required before freeze")
    names = set(source_files())
    names.update({"progress/07-dsrl-repair.md", "progress/receipts/07-dsrl-financial-history.json"})
    names.update(str(p.relative_to(ROOT)) for p in (ROOT / "outputs/dsrl07/inputs").glob("*"))
    hashes = {name: digest(ROOT / name) for name in sorted(names)}
    atomic_json(
        FREEZE,
        {
            "schema": "dsrl07-stage-a-v1",
            "frozen_utc": utc(),
            "authorization": {
                "user_launch": True,
                "instruction": "Implement next step, four GPUs authorized, persist through recoverable failures",
                "hard_cumulative_usd": 15,
            },
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": hashes,
            "verification": checks,
            "money": FinancialBook.limits(),
            "caps": dict(CAPS),
            "jobs": frozen_jobs(),
            "learner_config": LearnerConfig().resolved(),
            "prior_engineering_updates": 0,
            "prior_engineering_receipt": "task7-no-workstation-gradients",
            "actor_evaluation_mode": "cpu_published_actor",
            "actor_cpu_parity": {"atol": 1e-5, "rtol": 1e-5, "max_observations": 8},
            "input_manifest_sha256": digest(ROOT / "outputs/dsrl07/inputs/manifest.json"),
            "diagnostic_query_cap": 1000,
            "first_battery_queries": 216,
            "learner_mode_order": [
                "default-diagnostic-only",
                "deterministic",
                "deterministic-no-autotune",
            ],
            "admission": "Numerical diagnosis then live integration; separate stage-B freeze before development",
        },
    )
    print(json.dumps({"freeze": str(FREEZE), "sha256": digest(FREEZE), "files": len(hashes)}))


def bundle(destination):
    plan = json.loads(FREEZE.read_text())
    for name, sha in plan["source_sha256"].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Frozen source changed: {name}")
    rental = ROOT / "progress/rental-dsrl07-20261025.json"
    if not isinstance(json.loads(rental.read_text())["instance_id"], int):
        raise ValueError("Resolve rental before bundling")
    names = sorted(
        {*plan["source_sha256"], str(FREEZE.relative_to(ROOT)), str(rental.relative_to(ROOT))}
    )
    if destination.exists():
        raise ValueError("Bootstrap bundles are immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for name in names:
            if (ROOT / name).is_symlink():
                raise ValueError("Source bundle cannot contain symlinks")
            archive.add(ROOT / name, arcname=name, recursive=False)
    print(
        json.dumps({"bundle": str(destination), "sha256": digest(destination), "files": len(names)})
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inputs", "freeze", "bundle"))
    parser.add_argument("--verification", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/dsrl07/control/bootstrap-source.tar.gz"
    )
    args = parser.parse_args()
    if args.command == "inputs":
        prepare_inputs()
    elif args.command == "freeze":
        freeze(args.verification)
    else:
        bundle(args.output)


if __name__ == "__main__":
    main()
