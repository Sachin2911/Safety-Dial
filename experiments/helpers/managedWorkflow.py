"""Finite, resumable experiment stages with immutable outputs and explicit research gates.

Commands are argv lists passed directly to subprocess.Popen(shell=False). A resume
never infers success from the presence of files: completed stages require their saved
configuration identity and every declared output hash to match. Scientific gate stops
are terminal results, distinct from execution failures. Run this process under supervisor.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import yaml


class WorkflowError(RuntimeError):
    pass


def atomic_json(path: Path, data: dict):
    """Replace only the workflow's own state file; result files are never modified."""
    fd, temp = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def file_hash(path: Path):
    if path.is_symlink() or not path.is_file():
        raise WorkflowError(f"Expected a regular output file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise WorkflowError(f"Output changed while hashing: {path}")
    return digest.hexdigest()


def output_hash(path: Path):
    if path.is_symlink():
        raise WorkflowError(f"Output symlinks are not accepted: {path}")
    if path.is_file():
        return {"kind": "file", "sha256": file_hash(path)}
    if not path.is_dir():
        raise WorkflowError(f"Required output is missing: {path}")
    paths = sorted(path.rglob("*"))
    files = {}
    for child in paths:
        if child.is_symlink():
            raise WorkflowError(f"Output tree contains a symlink: {child}")
        if child.is_file():
            files[str(child.relative_to(path))] = file_hash(child)
    if paths != sorted(path.rglob("*")):
        raise WorkflowError(f"Output tree changed while hashing: {path}")
    return {"kind": "directory", "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "n_files": len(files)}


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise WorkflowError(f"{label} must be a finite positive number")
    return float(value)


def load_workflow(config_path: Path, repo_root: Path):
    raw = Path(config_path).read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict) or config.get("version") != 1:
        raise WorkflowError("Workflow configuration requires version: 1")
    workflow_id = config.get("workflow_id", "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", workflow_id):
        raise WorkflowError("workflow_id must be a simple unique identifier")
    cwd = (Path(repo_root) / config.get("working_directory", ".")).resolve()
    if not cwd.is_dir():
        raise WorkflowError(f"Working directory does not exist: {cwd}")
    if not isinstance(config.get("state_dir"), str) or not config["state_dir"]:
        raise WorkflowError("An explicit fresh state_dir is required")
    state_dir = (cwd / config["state_dir"]).resolve()
    stages = config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise WorkflowError("At least one finite stage is required")
    ids = set()
    for stage in stages:
        if not isinstance(stage, dict):
            raise WorkflowError("Each stage must be a mapping")
        sid = stage.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", sid) or sid in ids:
            raise WorkflowError("Stage IDs must be unique simple identifiers")
        ids.add(sid)
        argv = stage.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or not arg for arg in argv):
            raise WorkflowError(f"{sid}: argv must be a nonempty list of nonempty strings, never a shell command")
        _positive(stage.get("timeout_seconds"), f"{sid}.timeout_seconds")
        _positive(stage.get("wait_timeout_seconds", config.get("wait_timeout_seconds", 86400)), "wait_timeout_seconds")
        for key in ("wait_for", "fresh_outputs", "outputs"):
            values = stage.get(key, [])
            if not isinstance(values, list) or any(not isinstance(value, str) or not value for value in values):
                raise WorkflowError(f"{sid}.{key} must be a list of paths")
        if not stage.get("outputs") or not stage.get("fresh_outputs"):
            raise WorkflowError(f"{sid}: declare both outputs and fresh_outputs")
        for key in ("requires_gates", "watch_gates"):
            if not isinstance(stage.get(key, []), list):
                raise WorkflowError(f"{sid}.{key} must be a list of gate specifications")
        for gate in stage.get("requires_gates", []) + stage.get("watch_gates", []) + ([stage["gate"]] if "gate" in stage else []):
            if (not isinstance(gate, dict) or not isinstance(gate.get("path"), str)
                    or not isinstance(gate.get("key"), str) or not gate["key"]):
                raise WorkflowError(f"{sid}: gates require a JSON path and dotted boolean key")
            codes = gate.get("stop_exit_codes", [0, 2])
            if not isinstance(codes, list) or any(type(code) is not int for code in codes):
                raise WorkflowError("Gate stop_exit_codes must be integer exit codes")
    _positive(config.get("poll_interval_seconds", 15), "poll_interval_seconds")
    return config, {"config_sha256": hashlib.sha256(raw).hexdigest(), "cwd": str(cwd), "state_dir": str(state_dir)}


def read_gate(spec: dict, cwd: Path):
    path = (cwd / spec["path"]).resolve()
    try:
        value = json.loads(path.read_text())
        for key in spec["key"].split("."):
            value = value[key]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise WorkflowError(f"Cannot resolve gate {path}:{spec['key']}") from exc
    if type(value) is not bool:
        raise WorkflowError(f"Gate must be a JSON boolean, not a truthy value: {path}:{spec['key']}")
    return {"path": str(path), "key": spec["key"], "passes": value, "sha256": file_hash(path)}


def _record_outputs(paths, cwd):
    return {str((cwd / path).resolve()): output_hash((cwd / path).resolve()) for path in paths}


def _verify_recorded(stage_state):
    outputs = stage_state.get("output_hashes")
    if not outputs:
        raise WorkflowError("A completed stage has no verified output hashes")
    for path, expected in outputs.items():
        if output_hash(Path(path)) != expected:
            raise WorkflowError(f"Recorded output hash changed: {path}")


def _stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def run_workflow(config_path: Path, repo_root: Path, *, resume=False, dry_run=False):
    config, identity = load_workflow(config_path, repo_root)
    cwd, state_dir = Path(identity["cwd"]), Path(identity["state_dir"])
    if dry_run:
        return {"status": "validated", "workflow_id": config["workflow_id"], **identity,
                "stages": [stage["id"] for stage in config["stages"]], "subprocesses_started": 0}
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    with (state_dir / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkflowError("This workflow already has an active runner") from exc
        if state_path.exists():
            if not resume:
                raise WorkflowError("Workflow state already exists; use --resume to verify completed stages")
            state = json.loads(state_path.read_text())
            if any(state.get(key) != value for key, value in identity.items()):
                raise WorkflowError("Workflow configuration or working path changed; use a new state_dir")
        else:
            state = {"workflow_id": config["workflow_id"], **identity, "status": "pending", "stages": {}}
        logs = state_dir / "logs"
        logs.mkdir(exist_ok=True)
        current = None

        def save():
            state["updated_unix"] = time.time()
            atomic_json(state_path, state)

        try:
            for stage in config["stages"]:
                sid = stage["id"]
                current = state["stages"].setdefault(sid, {"status": "pending", "attempts": []})
                state["current_stage"] = sid
                if current["status"] in {"completed", "gate_stopped"}:
                    _verify_recorded(current)
                    if current["status"] == "gate_stopped":
                        state["status"] = "gate_stopped"
                        save()
                        return state
                    print(f"[workflow] verified completed stage {sid}", flush=True)
                    continue
                if current["status"] == "running" or current.get("interrupted_unknown_child"):
                    current["interrupted_unknown_child"] = True
                    raise WorkflowError(f"Stage {sid} was interrupted while running; its child/output state is unverified, so automatic rerun is refused")
                current.pop("error", None)
                state.pop("error", None)
                current["status"] = "waiting"
                state["status"] = "waiting"
                deadline = time.monotonic() + stage.get("wait_timeout_seconds", config.get("wait_timeout_seconds", 86400))
                while True:
                    watched, pending_gate_json = [], []
                    for spec in stage.get("watch_gates", []):
                        if not (cwd / spec["path"]).is_file():
                            continue
                        try:
                            watched.append(read_gate(spec, cwd))
                        except WorkflowError as exc:
                            # Upstream finite scripts may publish JSON with write_text.
                            # A syntax-incomplete write is pending, never a passing gate.
                            # Completed wrong schemas/non-booleans still fail immediately.
                            if not isinstance(exc.__cause__, json.JSONDecodeError):
                                raise
                            pending_gate_json.append(spec["path"])
                    current["watched_gates"] = watched
                    current["pending_gate_json"] = pending_gate_json
                    if any(not gate["passes"] for gate in watched):
                        current["output_hashes"] = _record_outputs([gate["path"] for gate in watched], cwd)
                        current["status"] = state["status"] = "gate_stopped"
                        save()
                        print(f"[workflow] upstream scientific gate stopped waiting stage {sid}", flush=True)
                        return state
                    missing = [path for path in stage.get("wait_for", []) if not (cwd / path).is_file()]
                    current["waiting_for"] = missing
                    save()
                    if not missing and not pending_gate_json:
                        break
                    if time.monotonic() >= deadline:
                        raise WorkflowError(f"Timed out waiting for input files for stage {sid}")
                    time.sleep(min(config.get("poll_interval_seconds", 15), max(0.001, deadline - time.monotonic())))
                before_gates = [read_gate(spec, cwd) for spec in stage.get("requires_gates", [])]
                current["required_gates"] = before_gates
                if any(not gate["passes"] for gate in before_gates):
                    current["output_hashes"] = _record_outputs([gate["path"] for gate in before_gates], cwd)
                    current["status"] = state["status"] = "gate_stopped"
                    save()
                    return state
                for output in stage["fresh_outputs"] + stage["outputs"]:
                    if (cwd / output).exists():
                        raise WorkflowError(f"Refusing to overwrite stage {sid} output: {cwd / output}")
                attempt = len(current["attempts"]) + 1
                log_path = logs / f"{sid}.attempt-{attempt:03d}.log"
                record = {"number": attempt, "started_unix": time.time(), "log": str(log_path)}
                current["attempts"].append(record)
                current["status"] = state["status"] = "running"
                save()
                print(f"[workflow] starting {sid}; log {log_path}", flush=True)
                with log_path.open("xb") as log:
                    process = subprocess.Popen(stage["argv"], cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                               shell=False, start_new_session=True)
                    record["pid"] = process.pid
                    save()
                    try:
                        record["returncode"] = process.wait(timeout=stage["timeout_seconds"])
                    except subprocess.TimeoutExpired as exc:
                        record["timed_out"] = True
                        _stop_process(process)
                        raise WorkflowError(f"Stage {sid} exceeded its finite subprocess timeout") from exc
                    except BaseException:
                        _stop_process(process)
                        raise
                    finally:
                        record["finished_unix"] = time.time()
                gate = read_gate(stage["gate"], cwd) if "gate" in stage else None
                current["gate"] = gate
                if (gate is not None and not gate["passes"]
                        and record["returncode"] in stage["gate"].get("stop_exit_codes", [0, 2])):
                    existing = [path for path in stage["outputs"] if (cwd / path).exists()]
                    current["output_hashes"] = _record_outputs([*existing, gate["path"]], cwd)
                    current["status"] = state["status"] = "gate_stopped"
                    save()
                    print(f"[workflow] scientific gate stopped at {sid}", flush=True)
                    return state
                if record["returncode"] != 0:
                    raise WorkflowError(f"Stage {sid} failed with exit code {record['returncode']}")
                current["output_hashes"] = _record_outputs(stage["outputs"], cwd)
                if gate:
                    current["output_hashes"].update(_record_outputs([gate["path"]], cwd))
                current["status"] = "completed"
                save()
                print(f"[workflow] completed {sid}", flush=True)
            state["status"] = "completed"
            save()
            return state
        except BaseException as exc:
            if current is not None:
                current["status"] = "failed"
                current["error"] = str(exc)
            state["status"] = "failed"
            state["error"] = str(exc)
            save()
            raise
