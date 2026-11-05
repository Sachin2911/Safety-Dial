#!/usr/bin/env python3
"""Retire one verified public HDF5 after its complete AP task and backups finish.

Rented-host maintenance only. The shared attempt ledger, weights, retrieval cache,
dataset receipts and all original run evidence are preserved.
"""

import argparse
import math
import os
from pathlib import Path
import re
import stat
import time

from common import (PROTOCOL_SHA256, TASKS, configuration, publish, read, sha)
from accounting import validate_boundary_pause

ASSET_ROOT = Path("/workspace/LeWM_Experiment1/assets/ap_reproduction")
RUNS_ROOT = Path("/workspace/LeWM_Experiment1/runs")
REPOSITORY = "Sachioster/LeWMExp1_AP_Reproduction"
INSTANCE_ID = 59238554
ARMS = ("gaussian_final", "gaussian_observed")


def regular(path, root):
    """Require a real file beneath the exact root without following any symlink."""
    path, root = Path(path), Path(root)
    if not path.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Input path is outside its exact authorized root")
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError("Symlinks are not allowed in retirement inputs")
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Retirement inputs must be regular files")
    return info


def fingerprint(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def checksum(value, length=64):
    if not isinstance(value, str) or not re.fullmatch(rf"[0-9a-f]{{{length}}}", value):
        raise ValueError("Invalid checksum or pinned revision in backup evidence")
    return value


def run_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise ValueError("Invalid source run identity")
    return value


def verify_file(path, root, expected):
    before = regular(path, root)
    if type(expected.get("bytes")) is not int or before.st_size != expected["bytes"]:
        raise ValueError("Retirement evidence byte count differs")
    if sha(path) != checksum(expected.get("sha256")):
        raise ValueError("Retirement evidence checksum differs")
    if fingerprint(regular(path, root)) != fingerprint(before):
        raise ValueError("A retirement input changed while it was verified")
    return before


def backup_proofs(config):
    entries = config.get("retire_backups")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Fresh, verified controller backup proofs are required")
    proofs = {}
    for entry in entries:
        identity = run_id(entry["run_id"])
        if identity in proofs:
            raise ValueError("Duplicate source-run backup proof")
        if entry.get("repo_id") != REPOSITORY or entry.get("verified") is not True:
            raise ValueError("Retirement requires verified private AP dataset backups")
        proofs[identity] = {"run_id": identity, "repo_id": REPOSITORY, "verified": True,
                            "revision": checksum(entry["revision"], 40),
                            "manifest_sha256": checksum(entry["manifest_sha256"])}
    return proofs


def source_manifest(identity, proof, current, ledger, boundary_pauses):
    root = RUNS_ROOT / identity
    for name in ("run.json", "result.json", "manifest.json", ".control/upload.json"):
        regular(root / name, RUNS_ROOT)
    metadata = read(root / "run.json")
    if (metadata.get("run_id") != identity or metadata.get("session_id") != current["session_id"]
            or metadata.get("instance_id") != current["instance_id"]):
        raise ValueError("Source evidence belongs to a different run, instance or session")
    storage = metadata.get("storage", {})
    if storage.get("repo_id") != REPOSITORY or storage.get("private") is not True:
        raise ValueError("Source run was not backed up to the private AP dataset")
    result = read(root / "result.json")
    upload = read(root / ".control/upload.json")
    if upload.get("verified") is not True or upload.get("error"):
        raise ValueError("Source run's local backup receipt is not verified")
    for key in ("revision", "manifest_sha256"):
        if upload.get(key) != proof[key]:
            raise ValueError("Fresh controller proof differs from the source backup receipt")
    if sha(root / "manifest.json") != proof["manifest_sha256"]:
        raise ValueError("Local source manifest differs from the verified backup")
    manifest = read(root / "manifest.json")
    if manifest.get("run_id") != identity or manifest.get("result") != result:
        raise ValueError("Source manifest identity or outcome differs")
    completed = (result.get("status") == "completed"
                 and type(result.get("exit_code")) is int and result["exit_code"] == 0)
    pause = validate_boundary_pause(root, ledger=ledger, backup_proof=proof)
    if pause is not None:
        boundary_pauses.append(pause)
    if not completed and pause is None:
        raise ValueError("Source run did not complete or prove the exact recorded boundary")
    return manifest


def retire(task, config):
    root = Path(config["asset_root"])
    if root != ASSET_ROOT or root.resolve() != ASSET_ROOT:
        raise ValueError("Retirement is restricted to this AP session's exact asset root")
    if config.get("conditions") != ["clean"] or config.get("arms") != list(ARMS):
        raise ValueError("Retirement configuration differs from the frozen paired clean study")
    current_root = Path(os.environ["LEWM_RUN_DIR"])
    regular(current_root / "run.json", RUNS_ROOT)
    current = read(current_root / "run.json")
    run_id(current["session_id"])
    expected_instance, expected_session, tasks = INSTANCE_ID, None, list(TASKS)
    binding = config.get("maintenance_binding")
    if binding is not None:
        partition = config.get("execution_partition")
        assigned = {"gpu1_existing": ["tworoom", "pusht", "reacher"], "gpu2_cube": ["cube"]}
        if partition not in assigned or binding.get("tasks") != assigned[partition]:
            raise ValueError("Maintenance binding differs from the explicit host task assignment")
        expected_instance, expected_session, tasks = (
            binding.get("instance_id"), binding.get("session_id"), binding["tasks"])
        if type(expected_instance) is not int or expected_instance <= 0 or not expected_session:
            raise ValueError("Maintenance host binding has not been completed")
        run_id(expected_session)
        if partition == "gpu1_existing" and expected_instance != INSTANCE_ID:
            raise ValueError("The existing host cannot be rebound to another instance")
    if task not in tasks:
        raise ValueError("Task retirement is assigned to a different host")
    if expected_session is not None and current.get("session_id") != expected_session:
        raise ValueError("Maintenance is running in a different bound session")
    if (current.get("instance_id") != expected_instance or current.get("storage", {}).get("repo_id")
            != REPOSITORY or current.get("storage", {}).get("private") is not True):
        raise ValueError("Maintenance is not running in the authorized private AP session")
    if current_root != RUNS_ROOT / run_id(current["run_id"]):
        raise ValueError("Maintenance run path differs from its recorded identity")

    dataset = root / "datasets" / (task + ".h5")
    receipt_path = dataset.with_suffix(".receipt.json")
    regular(receipt_path, ASSET_ROOT)
    dataset_receipt = read(receipt_path)
    spec = config["task_specs"][task]
    if dataset_receipt.get("task") != task:
        raise ValueError("Dataset receipt task differs")
    for key in ("dataset_revision", "archive_sha256", "archive_bytes"):
        if dataset_receipt.get(key) != spec[key]:
            raise ValueError("Dataset receipt differs from the pinned public source")
    dataset_stat = verify_file(dataset, ASSET_ROOT, dataset_receipt)

    cache_path = root / "latents" / (task + ".npy")
    cache_receipt_path = cache_path.with_suffix(".receipt.json")
    regular(cache_receipt_path, ASSET_ROOT)
    cache = read(cache_receipt_path)
    if cache.get("task") != task or cache.get("dataset") != dataset_receipt:
        raise ValueError("Cache receipt does not identify this exact verified dataset")
    if cache.get("checkpoint_sha256") != spec["weights_sha256"]:
        raise ValueError("Cache checkpoint differs from the frozen released weights")
    cache_stat = verify_file(cache_path, ASSET_ROOT, cache)
    cache_source = run_id(cache["source_run"])

    protocol_path = Path(config.get("ap_root", "/opt/ap-reference/code/ap")) / "protocols/main.json"
    if sha(protocol_path) != PROTOCOL_SHA256:
        raise ValueError("Original AP query protocol checksum differs")
    rows = read(protocol_path)["studies"][task]["confirm"]
    expected = {f"scientific:{task}:{row['record_id']}_clean_{arm}"
                for row in rows for arm in ARMS}
    if len(rows) != 128 or len(expected) != 256:
        raise ValueError("Original paired scientific cohort is not exactly 256 cases")
    ledger_path = root / "attempt-ledger.json"
    regular(ledger_path, ASSET_ROOT)
    ledger_hash = sha(ledger_path)
    ledger = read(ledger_path)
    actual = {identity for identity in ledger["attempts"]
              if identity.startswith(f"scientific:{task}:")}
    if actual != expected:
        raise ValueError("Scientific ledger has missing or unknown cases for this task")
    sources, cases_by_source = {cache_source}, {}
    for identity in sorted(expected):
        record = ledger["attempts"][identity]
        finished = record.get("finished_epoch")
        if (record.get("completed") is not True or record.get("phase") != "scientific"
                or type(finished) not in (int, float) or not math.isfinite(finished)
                or finished <= 0 or finished > current["created_epoch"]):
            raise ValueError("A scientific case was not completed before maintenance")
        source = run_id(record["run"])
        sources.add(source)
        cases_by_source.setdefault(source, []).append(identity.split(":", 2)[2])
    proofs = backup_proofs(config)
    if not sources.issubset(proofs):
        raise ValueError("Fresh backup proofs do not cover every scientific and cache source run")
    boundary_pauses = []
    manifests = {source: source_manifest(source, proofs[source], current, ledger, boundary_pauses)
                 for source in sorted(sources)}
    for source, cases in cases_by_source.items():
        published = manifests[source]["files"]
        for case in cases:
            journals = [f"metrics/journals/{case}.{suffix}" for suffix in ("jsonl", "jsonl.gz")
                        if f"metrics/journals/{case}.{suffix}" in published]
            if len(journals) != 1:
                raise ValueError("Each scientific case needs exactly one backed-up raw journal")
            required = [f"metrics/cases/{case}.json", f"metrics/trajectories/{case}.json.gz",
                        f"artifacts/videos/{case}.mp4", journals[0]]
            for relative in required:
                entry = published.get(relative)
                if (not isinstance(entry, dict) or type(entry.get("bytes")) is not int
                        or entry["bytes"] <= 0):
                    raise ValueError("Missing or empty verified case evidence: " + relative)
                checksum(entry.get("sha256"))
    cache_relative = f"artifacts/retrieval-cache/{task}.npy"
    published_cache = manifests[cache_source]["files"].get(cache_relative)
    if published_cache != {"bytes": cache["bytes"], "sha256": cache["sha256"]}:
        raise ValueError("The full cache file is absent from the verified source manifest")
    cache_metadata = RUNS_ROOT / cache_source / "artifacts/cache.json"
    expected_metadata = manifests[cache_source]["files"].get("artifacts/cache.json")
    if not expected_metadata:
        raise ValueError("Published cache provenance is absent from its verified source manifest")
    verify_file(cache_metadata, RUNS_ROOT, expected_metadata)
    if read(cache_metadata) != cache:
        raise ValueError("Published and working cache provenance differ")

    receipt = {"task": task, "path": str(dataset), "sha256": dataset_receipt["sha256"],
               "bytes": dataset_receipt["bytes"], "public_asset_spec": spec,
               "dataset_receipt": dataset_receipt, "scientific_cases": len(expected),
               "ledger_sha256_before": ledger_hash,
               "cache": {"path": str(cache_path), "sha256": cache["sha256"],
                         "bytes": cache["bytes"], "source_run": cache_source},
               "source_backups": [proofs[source] for source in sorted(sources)],
               "validated_boundary_pauses": boundary_pauses,
               "instance_id": current["instance_id"], "session_id": current["session_id"],
               "execution_partition": config.get("execution_partition"),
               "maintenance_run": current["run_id"], "simulator_actions": 0, "model_queries": 0}
    publish("artifacts/input-retirement-intent.json", dict(receipt, deleted=False,
                                                          guards_passed=True))
    if fingerprint(regular(dataset, ASSET_ROOT)) != fingerprint(dataset_stat):
        raise ValueError("Dataset changed after verification; retirement refused")
    if fingerprint(regular(cache_path, ASSET_ROOT)) != fingerprint(cache_stat):
        raise ValueError("Preserved cache changed after verification; retirement refused")
    if sha(ledger_path) != ledger_hash:
        raise ValueError("Shared ledger changed after verification; retirement refused")
    # This is the utility's sole deletion operation. No directory or glob is accepted.
    dataset.unlink()
    return dict(receipt, deleted=True, deleted_epoch=time.time())


def main(task):
    started = time.monotonic()
    completed, receipt = False, None
    try:
        config = configuration()
        receipt = retire(task, config)
        publish("artifacts/retirement.json", receipt)
        completed = True
        print(f"Retired verified public {task} HDF5; preserved caches, weights and evidence.",
              flush=True)
    finally:
        publish("metrics/accounting.json", {"phase": "maintenance", "task": task,
                "counts": {}, "completed": completed, "deleted": bool(receipt),
                "simulator_actions": 0, "model_queries": 0,
                "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    main(parser.parse_args().task)
