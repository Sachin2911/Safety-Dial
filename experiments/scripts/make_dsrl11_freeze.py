"""Freeze task-11 sources, inputs and authorizations; build immutable bootstrap bundles.

The inventory is traced from the task-11 entry points instead of hashing all of
``experiments/``: other agents edit their own experiment files concurrently, and a
freeze must not depend on, or be invalidated by, work outside this task.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.dsrl11_infra import LIMITS
from helpers.dsrl11_learner import Dsrl11LearnerConfig
from helpers.dsrl11_protocol import (
    CALIBRATION_FRACTION,
    CALIBRATION_INTERVAL,
    CALIBRATION_RANGE,
    CAPS,
    DIAGNOSTIC_SEEDS,
    PAUSE_SUCCESSIVE,
    PRIMARY_CONDITION,
    condition_order,
    frozen_jobs,
    geometry_specification,
)

SETUP = ROOT / "progress/receipts/11-dsrl-setup.json"
FREEZE = ROOT / "progress/receipts/11-dsrl-stage-a.json"
RENTAL = ROOT / "progress/rental-dsrl11-20261025.json"
READINESS = "progress/receipts/11-dsrl-launch-readiness.md"
TASK7_FREEZE = ROOT / "progress/receipts/07-dsrl-stage-a.json"
TASK7_FINAL = ROOT / "outputs/dsrl07/remote-final"
TASK7_LOCAL_COPY = ROOT / "progress/receipts/07-dsrl-completion/local-copy.json"
INPUTS = ROOT / "outputs/dsrl11/inputs"
PREFLIGHT = ROOT / "outputs/dsrl11/preflight/receipt.json"
# Episode subprocesses import from these directories (see run_dsrl11.start_episode).
IMPORT_ROOTS = ("experiments", "experiments/helpers", "scripts/todo2")
ENTRY_POINTS = (
    "experiments/scripts/run_dsrl11.py",
    "experiments/scripts/run_dsrl06.py",
    "experiments/scripts/probe_dsrl11.py",
    "experiments/scripts/diagnose_dsrl11.py",
    "experiments/scripts/dsrl11_learner_service.py",
    "experiments/scripts/serve_dsrl11.py",
    "experiments/scripts/dsrl11_watchdog.py",
    "experiments/scripts/sync_dsrl11.py",
    "experiments/scripts/manage_dsrl11.py",
    "experiments/scripts/make_dsrl11_freeze.py",
    "experiments/scripts/patch_dsrl06.py",
    "experiments/helpers/dsrl11_recording.py",
)
OPTIONAL_ENTRY_POINTS = (
    "experiments/scripts/analyze_dsrl11.py",
    "experiments/scripts/preserve_dsrl11.py",
)
SHELL = ("experiments/scripts/bootstrap_dsrl11.sh", "experiments/scripts/prepare_dsrl11.sh")
# Preparation inputs proven by the task-7 bootstrap; unchanged since its freeze.
PREPARATION_PREFIXES = (
    "configs/dsrl06/",
    "progress/receipts/01",
    "scripts/stage2/",
    "scripts/todo1/",
    "scripts/todo1b/",
    "scripts/todo2/",
)
TASK_DOCUMENTS = ("progress/11-dsrl-improvement-plan.md", "progress/receipts/11-dsrl-review.md")
# Shared documents edited by other sessions: hashed for context, never gated.
CONTEXT = ("AGENTS.md", "firstPrinciples.md")
LITERAL = re.compile(
    r"^(experiments|scripts|configs|progress/receipts|outputs/dsrl06-preflight)/[\w./-]+$"
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _module_paths(name):
    parts = name.split(".")
    for base in IMPORT_ROOTS:
        stem = ROOT / base / Path(*parts)
        for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
            if candidate.is_file():
                yield candidate


def _references(path):
    tree = ast.parse(path.read_text(), str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield from _module_paths(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield from _module_paths(node.module)
            for alias in node.names:
                yield from _module_paths(f"{node.module}.{alias.name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value.split(":", 1)[0]
            if LITERAL.match(value) and (ROOT / value).is_file():
                yield ROOT / value
            # Observer factories are named as module:function strings.
            if re.fullmatch(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*", node.value):
                yield from _module_paths(value)


def traced_sources():
    pending = [ROOT / name for name in ENTRY_POINTS]
    pending += [ROOT / name for name in OPTIONAL_ENTRY_POINTS if (ROOT / name).is_file()]
    seen = set()
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        seen.add(path)
        if path.suffix == ".py":
            pending.extend(_references(path))
    return {str(path.relative_to(ROOT)) for path in seen}


def preparation_sources():
    task7 = json.loads(TASK7_FREEZE.read_text())["source_sha256"]
    names = {name for name in task7 if name.startswith(PREPARATION_PREFIXES)}
    for name in names:
        if digest(ROOT / name) != task7[name]:
            raise ValueError(f"Preparation input changed since the task-7 bootstrap: {name}")
    return names


def source_files():
    names = traced_sources() | preparation_sources() | set(SHELL) | set(TASK_DOCUMENTS)
    names |= {
        str(p.relative_to(ROOT))
        for p in (ROOT / "experiments/tests").glob("*.py")
        if p.name == "conftest.py" or p.name.startswith("test_dsrl11_")
    }
    if any(
        name.startswith(("experiments/helpers/unisafe", "experiments/helpers/safecem"))
        for name in names
    ):
        raise ValueError("Another task's sources leaked into the task-11 inventory")
    return sorted(names)


def input_files():
    root = ROOT / "outputs/dsrl11/inputs"
    return sorted(str(p.relative_to(ROOT)) for p in root.rglob("*") if p.is_file())


def copy_verified(source, destination, expected):
    if digest(source) != expected:
        raise ValueError(f"Source input hash differs: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if digest(destination) != expected:
        raise ValueError(f"Copied input hash differs: {destination}")


def inputs():
    """Copy the verified task-7 inputs: the final replay (A1) and saved pool views (A2)."""
    manifest_path = INPUTS / "manifest.json"
    if manifest_path.exists():
        raise ValueError("Task-11 inputs are copied once")
    local = json.loads(TASK7_LOCAL_COPY.read_text())
    if local.get("exact_inventory_verified") is not True or local["instance_id"] != 57632188:
        raise ValueError("Only the verified task-7 remote-final snapshot is an admissible source")
    checkpoint = TASK7_FINAL / "learner/checkpoints/episode24"
    saved = json.loads((checkpoint / "manifest.json").read_text())
    target = INPUTS / "task7/learner/checkpoints/episode24"
    for name in ("replay.npz", "state.json"):
        copy_verified(checkpoint / name, target / name, saved[name])
    # The original manifest is kept byte for byte; its learner.msgpack entry is not needed.
    copy_verified(
        checkpoint / "manifest.json", target / "manifest.json", digest(checkpoint / "manifest.json")
    )
    frozen = json.loads(TASK7_FREEZE.read_text())["source_sha256"]
    pool = json.loads((ROOT / "outputs/dsrl07/inputs/manifest.json").read_text())
    rows = []
    for row in pool["snapshots"]:
        if frozen.get(row["path"]) != row["sha256"]:
            raise ValueError("Saved pool view differs from the task-7 freeze")
        name = Path(row["path"]).name
        copy_verified(ROOT / row["path"], INPUTS / "pool" / name, row["sha256"])
        rows.append({"file": name, **{k: row[k] for k in ("sha256", "snapshot_id", "task")}})
    atomic_json(
        INPUTS / "pool/manifest.json",
        {
            "schema": "dsrl11-pool-inputs-v1",
            "created_utc": utc(),
            "snapshots": rows,
            "source": "task-7 frozen saved inputs (task-6 factual engineering observations)",
            "role": "A2 decoder diagnosis only; never training or calibration replay",
        },
    )
    atomic_json(
        manifest_path,
        {
            "schema": "dsrl11-inputs-v1",
            "created_utc": utc(),
            "task7_local_copy_sha256": digest(TASK7_LOCAL_COPY),
            "task7_checkpoint": str(checkpoint.relative_to(ROOT)),
            "task7_checkpoint_manifest_sha256": digest(checkpoint / "manifest.json"),
            "pool_manifest_sha256": digest(INPUTS / "pool/manifest.json"),
            "role": "A1 refits the task-7 replay; held-out tasks are not inputs",
        },
    )
    print(json.dumps({"inputs": input_files()}))


def freeze(args):
    if FREEZE.exists():
        raise ValueError("The stage-A freeze is immutable; record a repair amendment")
    verification = json.loads(args.verification.read_text())
    if verification.get("passed") is not True:
        raise ValueError("Passing workstation checks are required before the freeze")
    preflight = json.loads(PREFLIGHT.read_text())
    setup_plan = json.loads(SETUP.read_text())
    names, inputs_ = source_files(), input_files()
    if not inputs_:
        raise ValueError("Copy the verified inputs before freezing")
    atomic_json(
        FREEZE,
        {
            "schema": "dsrl11-freeze-v1",
            "frozen_utc": utc(),
            "authorization": {**setup_plan["authorization"], "stage_a": args.authorization},
            "setup_sha256": digest(SETUP),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {name: digest(ROOT / name) for name in names},
            "input_sha256": {name: digest(ROOT / name) for name in inputs_},
            "context_sha256": {name: digest(ROOT / name) for name in CONTEXT},
            "verification": verification,
            "money": dict(LIMITS),
            "caps": dict(CAPS),
            "jobs": frozen_jobs(),
            "condition_order": condition_order(),
            "primary_condition": PRIMARY_CONDITION,
            "final_checkpoint": "each condition's batch-B publication (actor version 2), "
            "exported before any evaluation episode",
            "learner_defaults": Dsrl11LearnerConfig().resolved(),
            "a1": {
                "seeds": list(DIAGNOSTIC_SEEDS),
                "cells": [[u, r] for u in (20, 5) for r in ("min", "mean")],
                "admission": "finite and the stage-C pause rule never fires, for both seeds "
                "and both action sets",
                "preference": [[20, "min"], [20, "mean"], [5, "min"], [5, "mean"]],
            },
            "calibration_gate": {
                "quantity": "the learner's configured reduction (mean or min) of its ten "
                "critic heads, for recorded and actor-sampled actions",
                "range": list(CALIBRATION_RANGE),
                "fraction": CALIBRATION_FRACTION,
                "interval_updates": CALIBRATION_INTERVAL,
                "successive_failures": PAUSE_SUCCESSIVE,
                "initialization": "descriptive baseline, excluded from the pause series",
                "rationale": "a random ensemble scatters single heads (task 7: 26% outside) "
                "while its reduced estimate lay inside; decided before any A1 run",
            },
            "geometry": geometry_specification(),
            "prior_engineering_updates": int(preflight["workstation_updates"]),
            "prior_engineering_receipt": str(PREFLIGHT.relative_to(ROOT)),
            "prior_engineering_receipt_sha256": digest(PREFLIGHT),
            "launch_readiness_receipt": READINESS,
            "heldout": "held-out tasks are untouched: no job, input or audit uses them",
        },
    )
    print(json.dumps({"freeze": str(FREEZE), "sha256": digest(FREEZE), "files": len(names)}))


def setup(args):
    if SETUP.exists():
        raise ValueError("The setup authorization is immutable")
    if not (ROOT / READINESS).is_file():
        raise ValueError("Record the launch-readiness review first")
    names = source_files()
    atomic_json(
        SETUP,
        {
            "schema": "dsrl11-setup-v1",
            "created_utc": utc(),
            "authorization": {
                "user_launch": True,
                "rental": "one fresh four-RTX-4090 Vast instance",
                "instruction": args.instruction,
                "decision": args.decision,
                "total_cap_usd": LIMITS["total_usd"],
                "scope": "host verification, remote financial watchdog and pinned runtime "
                "preparation while workstation tests and the stage-A freeze finish; no "
                "frozen stage starts before the stage-A freeze is deployed",
            },
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {name: digest(ROOT / name) for name in names},
            "input_sha256": {},
            "context_sha256": {name: digest(ROOT / name) for name in CONTEXT},
            "money": dict(LIMITS),
            "launch_readiness_receipt": READINESS,
        },
    )
    print(json.dumps({"setup": str(SETUP), "sha256": digest(SETUP), "files": len(names)}))


def verify(plan):
    for name, expected in {**plan["source_sha256"], **plan["input_sha256"]}.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen file changed: {name}")


def bundle(args):
    plan_path = SETUP if args.plan == "setup" else FREEZE
    plan = json.loads(plan_path.read_text())
    verify(plan)
    rental = json.loads(RENTAL.read_text())
    if not isinstance(rental.get("instance_id"), int):
        raise ValueError("Resolve the rental before bundling")
    names = sorted(
        {
            *plan["source_sha256"],
            *plan["input_sha256"],
            plan["launch_readiness_receipt"],
            str(plan_path.relative_to(ROOT)),
            str(RENTAL.relative_to(ROOT)),
            *([str(SETUP.relative_to(ROOT))] if args.plan == "stage-a" else []),
        }
    )
    destination = args.output or ROOT / f"outputs/dsrl11/control/{args.plan}-source.tar.gz"
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
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--inputs", action="store_true")
    authorization = sub.add_parser("setup")
    authorization.add_argument("--instruction", required=True)
    authorization.add_argument("--decision", required=True)
    sub.add_parser("inputs")
    freezing = sub.add_parser("freeze")
    freezing.add_argument("--verification", type=Path, required=True)
    freezing.add_argument("--authorization", required=True)
    packing = sub.add_parser("bundle")
    packing.add_argument("--plan", choices=("setup", "stage-a"), required=True)
    packing.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "list":
        print("\n".join(input_files() if args.inputs else source_files()))
    elif args.command == "setup":
        setup(args)
    elif args.command == "inputs":
        inputs()
    elif args.command == "freeze":
        freeze(args)
    else:
        bundle(args)


if __name__ == "__main__":
    main()
