"""Completion and immutable-copy checks; independent of the frozen experiment code."""

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def completion(root):
    root = Path(root)
    output = root / "outputs/screen08"
    if not (output / "pipeline-ended.txt").exists():
        return {"status": "pending"}
    if (output / "pipeline-exit-code.txt").read_text().strip() != "0":
        return {"status": "needs_attention", "reason": "pipeline_failed"}
    for name in ("study-complete.txt", "engineering-complete.txt", "stage_a-complete.txt"):
        if not (output / name).is_file():
            raise ValueError(f"Missing completion evidence: {name}")
    plan_path = root / "progress/receipts/08-screen-plan.json"
    plan = json.loads(plan_path.read_text())
    decision = json.loads((output / "stage-b-decision.json").read_text())
    if decision["plan_sha256"] != sha256(plan_path):
        raise ValueError("Confirmation decision is not bound to the source freeze")
    survivors = decision["survivors"]
    if len(set(survivors)) != len(survivors) or not set(survivors) <= set(plan["arms"][1:]):
        raise ValueError("Invalid confirmation survivors")
    if decision["admit"]:
        if not survivors or decision["reason"] != "admitted":
            raise ValueError("Invalid confirmation admission")
        if not (output / "stage_b-complete.txt").is_file():
            raise ValueError("Admitted confirmation has not finished")
        confirmation = [j for j in plan["stage_b_jobs"]
                        if j["arm"] in ("native", *survivors)]
    else:
        reason = "insufficient_time" if survivors else "no_survivors"
        if decision["reason"] != reason:
            raise ValueError("Invalid confirmation non-admission")
        confirmation = []
    expected = {j["job"]: j for j in plan["engineering_jobs"] + plan["jobs"] + confirmation}
    observed, stage_a_evidence = {}, {}
    for validated in sorted((output / "episodes").glob("*/validation.json")):
        job = json.loads((validated.parent / "job.json").read_text())
        identity = job["job"]
        if identity in observed or identity not in expected:
            raise ValueError("Duplicate or unplanned validated job")
        if any(job.get(k) != v for k, v in expected[identity].items()):
            raise ValueError("Validated job identity differs from its frozen schedule")
        validation = json.loads(validated.read_text())
        row = validation["row"]
        if (row["status"] != "completed" or row["seed"] != job["seed"]
                or row["scene_name"] != f"task_{job['task']:04d}/base"
                or validation["screen_mode"] != job["arm"]):
            raise ValueError("Validation does not match the scheduled episode")
        observed[identity] = job
        if job["stage"] == "stage_a":
            stage_a_evidence[str(validated.relative_to(output))] = sha256(validated)
    if set(observed) != set(expected):
        raise ValueError("The complete scheduled panel is not preserved")
    if stage_a_evidence != decision["evidence_sha256"]:
        raise ValueError("Confirmation evidence changed after the selection decision")
    for name, expected_hash in plan["source_sha256"].items():
        if sha256(root / name) != expected_hash:
            raise ValueError(f"Frozen source differs: {name}")
    return {"status": "complete", "plan_sha256": sha256(plan_path),
            "validated_jobs": sorted(observed), "confirmation_jobs": len(confirmation),
            "confirmation_reason": decision["reason"],
            "source_hashes_checked": len(plan["source_sha256"])}


def active_experiment_processes(proc=Path("/proc")):
    active = []
    for path in proc.glob("[0-9]*/cmdline"):
        try:
            args = path.read_bytes().decode(errors="replace").split("\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        scripts = {Path(arg).name for arg in args}
        modules = {args[i + 1] for i, arg in enumerate(args[:-1]) if arg == "-m"}
        if ("run_screen08.py" in scripts
                or modules & {"maniguard.eval.benchmark", "maniguard.serve.openpi_native"}):
            active.append(int(path.parent.name))
    return active


def inventory(directory):
    result = {}
    for path in sorted(Path(directory).rglob("*")):
        if path.is_symlink():
            raise ValueError("Preservation evidence must not contain symlinks")
        if path.is_file():
            result[str(path.relative_to(directory))] = {
                "bytes": path.stat().st_size, "sha256": sha256(path),
            }
    return result


def snapshot(root, instance_id):
    root = Path(root)
    state = completion(root)
    if state["status"] != "complete":
        return state
    active = active_experiment_processes()
    if active:
        return {"status": "pending", "reason": "waiting_for_process_exit", "pids": active}
    if os.environ.get("CONTAINER_ID") != str(instance_id):
        raise ValueError("Snapshot is not running on the recorded task-8 allocation")
    destination = root / "outputs/screen08-completed-snapshot"
    if destination.exists():
        receipt = verify_snapshot(destination)
        if receipt["instance_id"] != instance_id or receipt["completion"] != state:
            raise ValueError("Existing snapshot is not for this completed experiment")
        return {"status": "snapshot_ready", "path": str(destination),
                "files": len(receipt["files"]), "instance_id": instance_id}
    temporary = Path(tempfile.mkdtemp(prefix=".screen08-copy-", dir=root / "outputs"))
    try:
        source = root / "outputs/screen08"
        before = {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()}
        for path in source.rglob("*"):
            if path.is_symlink():
                raise ValueError("Unexpected evidence symlink")
        shutil.copytree(source, temporary / "raw")
        files = inventory(temporary / "raw")
        after = inventory(source)
        if before != set(files) or set(after) != set(files):
            raise ValueError("Evidence inventory changed during the terminal snapshot")
        auxiliary = {"network-cost.json", "logs/remote-watchdog.log"}
        if any(files[name] != after[name] for name in files if name not in auxiliary):
            raise ValueError("Completed experiment bytes changed during the snapshot")
        if active_experiment_processes():
            raise ValueError("An experiment process appeared during the terminal snapshot")
        receipt = {"schema": "screen08-terminal-snapshot-v1", "instance_id": instance_id,
                   "created_utc": datetime.now(timezone.utc).isoformat(),
                   "completion": state, "files": files,
                   "source_hashes_match_except_live_auxiliary": True,
                   "live_auxiliary_snapshot": sorted(auxiliary)}
        (temporary / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"status": "snapshot_ready", "path": str(destination),
            "files": len(files), "instance_id": instance_id}


def verify_snapshot(directory):
    directory = Path(directory)
    if {p.name for p in directory.iterdir()} != {"raw", "receipt.json"}:
        raise ValueError("Unexpected snapshot top-level inventory")
    receipt = json.loads((directory / "receipt.json").read_text())
    if not receipt["files"] or inventory(directory / "raw") != receipt["files"]:
        raise ValueError("Snapshot file inventory or bytes differ from remote receipt")
    return receipt
