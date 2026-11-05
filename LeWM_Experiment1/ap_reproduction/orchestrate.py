#!/usr/bin/env python3
"""Local, finite AP queue controller. Only compute.py status and run are invoked.

The separately written JSON plan pins the rental, session, deadline, local input
files and exact commands. This program never retries a launch, synchronizes a
failed backup, stops an experiment, or destroys a rental.

Plans may select the separately authorized controller_session="ap-cube". That
route uses its own state/journal/lock and the pinned Cube source snapshot; plans
without controller_session retain the original rental and canonical source.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
COMPUTE = PROJECT / "LeWM_Experiment1/computeConfigs/compute.py"
STATE = PROJECT / "LeWM_Experiment1/computeConfigs/.state/instance.json"
LOG = PROJECT / "LeWM_Experiment1/computeConfigs/.state/ap-orchestrator.jsonl"
POLICY = PROJECT / "LeWM_Experiment1/computeConfigs/config/compute-policy.json"
SOURCE = PROJECT / "LeWM_Experiment1/ap_reproduction"
REPOSITORY = "Sachioster/LeWMExp1_AP_Reproduction"
INSTANCE_ID = 59238554
CUBE_SOURCE = PROJECT / "LeWM_Experiment1/runs/AP/cube-source-20261105"
CUBE_SOURCE_MANIFEST = (
    PROJECT / "LeWM_Experiment1/computeConfigs/records/ap_reproduction/cube-source-snapshot.json")
CUBE_INSTANCE_ID = 59256965
CUBE_SESSION_ID = "lewm-exp1-d88cfc7fb578"
CUBE_DEADLINE = 1793885983.2140357
GATE_FILES = {"common.py", "instrumentation.py", "engineering.py", "evaluate.py"}
ENTRYPOINTS = {"prepare.py", "encode_cache.py", "engineering.py", "evaluate.py"}


class StopQueue(RuntimeError):
    pass


class ControllerUnavailable(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value):
        raise StopQueue("Plan contains an invalid run or session identifier")
    return value


def positive(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StopQueue("Plan contains an invalid finite numeric bound")
    if not math.isfinite(value) or value <= 0:
        raise StopQueue("Plan contains an invalid finite numeric bound")
    return value


def local_path(value, *, directory=False):
    if not isinstance(value, str) or any(ord(char) < 32 for char in value):
        raise StopQueue("Plan contains an invalid local path")
    candidate = PROJECT / value
    if candidate.is_symlink():
        raise StopQueue("Pinned input paths cannot be symlinks")
    path = candidate.resolve()
    if not path.is_relative_to(PROJECT):
        raise StopQueue("Pinned input paths must remain inside this repository")
    if not (path.is_dir() if directory else path.is_file()):
        raise StopQueue("A pinned local input is missing")
    return path


def routing(plan):
    name = plan.get("controller_session")
    if name is None:
        return {"controller_session": None, "instance_id": INSTANCE_ID,
                "state": STATE, "log": LOG, "source": SOURCE}
    if name != "ap-cube":
        raise StopQueue("Only the explicitly authorized ap-cube named controller is permitted")
    if plan.get("session_id") != CUBE_SESSION_ID:
        raise StopQueue("Named Cube plan does not match its authorized session identity")
    if positive(plan["deadline_epoch"]) > CUBE_DEADLINE:
        raise StopQueue("Named Cube plan exceeds its authorized session deadline")
    parent = STATE.parent / "named"
    directory = parent / name
    if parent.is_symlink() or directory.is_symlink() or not directory.resolve().is_relative_to(
            STATE.parent.resolve() / "named"):
        raise StopQueue("Named controller state must not alias another session directory")
    return {"controller_session": name, "instance_id": CUBE_INSTANCE_ID,
            "state": directory / STATE.name, "log": directory / LOG.name,
            "source": CUBE_SOURCE}


def validate_plan(plan):
    route = routing(plan)
    if plan.get("schema_version") != 1 or plan.get("instance_id") != route["instance_id"]:
        raise StopQueue("Plan does not identify the authorized AP instance")
    if plan.get("storage_repo") != REPOSITORY:
        raise StopQueue("Plan storage is not the AP reproduction dataset")
    identifier(plan["session_id"])
    positive(plan["deadline_epoch"])
    source = local_path(plan["source"], directory=True)
    if source != route["source"]:
        raise StopQueue("Plan source differs from the exact authorized session source directory")
    local_path(plan["config"])
    local_path(plan["protocol"])
    if route["controller_session"]:
        config = read(local_path(plan["config"]))
        binding = config.get("maintenance_binding", {})
        if (config.get("execution_partition") != "gpu2_cube"
                or config.get("hosting_tasks") != ["cube"]
                or binding.get("instance_id") != CUBE_INSTANCE_ID
                or binding.get("session_id") != CUBE_SESSION_ID
                or binding.get("tasks") != ["cube"]):
            raise StopQueue("Named Cube configuration differs from its exact task and host binding")
    pins = plan.get("gate_hashes")
    if not isinstance(pins, dict) or set(pins) != GATE_FILES:
        raise StopQueue("Plan must pin exactly the four engineering gate source files")
    hashes = [*pins.values(), plan["config_sha256"], plan["protocol_sha256"]]
    source_manifest = plan.get("source_manifest")
    if source_manifest is not None:
        manifest_path = local_path(source_manifest)
        hashes.append(plan["source_manifest_sha256"])
        if route["controller_session"] and manifest_path != CUBE_SOURCE_MANIFEST:
            raise StopQueue("Named Cube plan uses an unexpected source-snapshot manifest")
    elif route["controller_session"] or "source_manifest_sha256" in plan:
        raise StopQueue("Named Cube plans require a pinned explicit source-snapshot manifest")
    for value in hashes:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise StopQueue("Plan contains an invalid input checksum")
    prior = plan.get("prior_run_ids", [])
    failures = plan.get("allowed_prior_failed_run_ids", [])
    if not isinstance(prior, list) or not isinstance(failures, list):
        raise StopQueue("Prior run identities must be explicit lists")
    if len(set(prior)) != len(prior) or len(set(failures)) != len(failures):
        raise StopQueue("Duplicate prior run identities")
    for run_id in prior + failures:
        identifier(run_id)
    if not set(failures).issubset(prior):
        raise StopQueue("Failure exceptions must name explicit prior runs")
    stages = plan.get("stages")
    if not isinstance(stages, list) or not stages:
        raise StopQueue("Plan requires a finite, nonempty stage list")
    ids = []
    for stage in stages:
        ids.append(identifier(stage["run_id"]))
        if positive(stage["max_seconds"]) > 21600:
            raise StopQueue("A stage exceeds the authorized six-hour session bound")
        seeds = stage.get("seeds")
        if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
            raise StopQueue("Every stage requires explicit integer seeds")
        command = stage.get("command")
        if not isinstance(command, list) or len(command) < 2 or any(
                not isinstance(arg, str) or not arg or any(ord(c) < 32 for c in arg)
                for arg in command):
            raise StopQueue("Every stage requires a nonempty argv list")
        if command[0] not in {"python", "python3", "/venv/main/bin/python"}:
            raise StopQueue("Unexpected stage interpreter")
        if command[1] not in ENTRYPOINTS:
            raise StopQueue("Unexpected AP stage entrypoint")
        if route["controller_session"] and (len(command) < 3 or command[2] != "cube"):
            raise StopQueue("The named Cube controller is authorized for Cube stages only")
        if any(re.match(r"--?(?:token|password|api[-_]key)(?:=|$)", arg) for arg in command):
            raise StopQueue("Credentials cannot appear in a stage command")
    if len(set(ids)) != len(ids) or set(ids) & set(failures):
        raise StopQueue("Stage IDs must be unique and cannot excuse a previous failed stage")
    return route


class Queue:
    def __init__(self, path, poll_seconds):
        self.path = Path(path).resolve()
        self.plan_hash = digest(self.path)
        self.plan = read(self.path)
        route = validate_plan(self.plan)
        self.controller_session = route["controller_session"]
        self.instance_id = route["instance_id"]
        self.state, self.log, self.source = route["state"], route["log"], route["source"]
        self.poll_seconds = positive(poll_seconds)
        if not 1 <= self.poll_seconds <= 60:
            raise StopQueue("Poll intervals must be between one and sixty seconds")
        self.stages = self.plan["stages"]
        self.stage_ids = {stage["run_id"] for stage in self.stages}
        self.known = self.stage_ids | set(self.plan.get("prior_run_ids", []))
        self.allowed_failures = set(self.plan.get("allowed_prior_failed_run_ids", []))
        self.intents = set()
        self.last_status, self.last_emitted = None, 0
        if self.log.exists():
            if self.log.is_symlink():
                raise StopQueue("Orchestrator journal cannot be a symlink")
            with self.log.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get("session_id") == self.plan["session_id"] and row.get(
                            "event") == "launch_intent":
                        self.intents.add(identifier(row["run_id"]))

    def emit(self, event, **fields):
        record = {"epoch": time.time(), "event": event, "instance_id": self.instance_id,
                  "session_id": self.plan["session_id"], "plan_sha256": self.plan_hash, **fields}
        if self.controller_session:
            record["controller_session"] = self.controller_session
        encoded = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        with self.log.open("a") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        print(encoded, end="", flush=True)

    def deadline(self):
        if time.time() >= self.plan["deadline_epoch"]:
            raise StopQueue("Plan deadline reached; no further launch is permitted")
        if digest(self.path) != self.plan_hash:
            raise StopQueue("Plan changed while the queue was running")

    def validate_inputs(self):
        self.deadline()
        for name, expected in self.plan["gate_hashes"].items():
            path = local_path(str(self.source / name))
            if digest(path) != expected:
                raise StopQueue("Engineering gate source checksum changed: " + name)
        for name in ("config", "protocol"):
            if digest(local_path(self.plan[name])) != self.plan[name + "_sha256"]:
                raise StopQueue("Pinned " + name + " checksum changed")
        if self.plan.get("source_manifest") is not None:
            if digest(local_path(self.plan["source_manifest"])) != self.plan[
                    "source_manifest_sha256"]:
                raise StopQueue("Pinned source-snapshot manifest checksum changed")

    def local_identity(self):
        self.deadline()
        state = read(self.state)
        if state.get("instance_id") != self.instance_id or state.get("session_id") != self.plan[
                "session_id"]:
            raise StopQueue("Local compute session or instance identity changed")
        if state.get("storage", {}).get("repo_id") != REPOSITORY:
            raise StopQueue("Local compute storage identity changed")
        if state.get("phase") != "ready":
            raise StopQueue("Local compute session is not ready")
        actual_deadline = positive(state["deadline_epoch"])
        if self.plan["deadline_epoch"] > actual_deadline:
            raise StopQueue("Plan deadline exceeds the rental's authorized deadline")
        submitted = state.get("runs")
        if not isinstance(submitted, list) or set(submitted) - self.known:
            raise StopQueue("Local compute state contains an unexpected run")
        return state

    def pause(self):
        self.deadline()
        time.sleep(min(self.poll_seconds, self.plan["deadline_epoch"] - time.time()))

    def call(self, action, arguments=()):
        if action not in {"status", "run"}:
            raise StopQueue("Only compute status and run are permitted")
        self.deadline()
        timeout = min(60.0, self.plan["deadline_epoch"] - time.time())
        command = [sys.executable, str(COMPUTE)]
        if self.controller_session:
            command += ["--session", self.controller_session]
        command += [action, *arguments]
        process = subprocess.Popen(command,
                                   cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            stdout, _stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise ControllerUnavailable("controller_timeout") from None
        if process.returncode:
            raise ControllerUnavailable("controller_nonzero_exit")
        if len(stdout) > 8 * 1024 * 1024:
            raise ControllerUnavailable("controller_response_too_large")
        try:
            result = json.loads(stdout)
        except ValueError:
            raise ControllerUnavailable("controller_invalid_json") from None
        if not isinstance(result, dict):
            raise ControllerUnavailable("controller_invalid_json")
        return result

    def status(self):
        for attempt in range(1, 4):
            local = self.local_identity()
            try:
                report = self.call("status")
                if report.get("instance_id") != self.instance_id or report.get("storage_repo") != (
                        REPOSITORY):
                    raise StopQueue("Reported rental or storage identity changed")
                if report.get("deadline_epoch") != local["deadline_epoch"]:
                    raise StopQueue("Reported session deadline changed")
                if report.get("phase") != "ready" or report.get("provider_status") != "running":
                    raise StopQueue("Managed instance is no longer ready and running")
                if report.get("deadline_passed"):
                    raise StopQueue("Compute wrapper reports the session deadline has passed")
                if report.get("connection_error"):
                    raise ControllerUnavailable("status_connection_unavailable")
                return report, self.local_identity()
            except ControllerUnavailable as error:
                self.emit("status_retry", attempt=attempt, reason=str(error))
                if attempt == 3:
                    raise StopQueue("Three consecutive status attempts failed") from None
                self.pause()
        raise AssertionError("Unreachable status retry state")

    def inspect(self, report):
        rows = report.get("runs")
        if not isinstance(rows, list):
            raise StopQueue("Compute status has no valid run inventory")
        runs, active = {}, []
        for row in rows:
            run_id = identifier(row["run_id"])
            if run_id not in self.known or run_id in runs:
                raise StopQueue("Unexpected or duplicated run in remote compute status")
            runs[run_id] = row
            if row.get("phase") == "backup_failed" or row.get("upload", {}).get("error"):
                raise StopQueue("Backup failure requires manual review: " + run_id)
            result = row.get("result")
            success = bool(result and result.get("status") == "completed"
                           and type(result.get("exit_code")) is int and result["exit_code"] == 0)
            if result and not success and run_id not in self.allowed_failures:
                raise StopQueue("A new or unapproved failed run was observed: " + run_id)
            if row.get("supervisor_alive") or row.get("job_alive"):
                if row.get("job_alive") and not row.get("supervisor_alive"):
                    raise StopQueue("A job lost its supervisor: " + run_id)
                active.append(run_id)
            elif not result:
                raise StopQueue("Run has neither a live supervisor nor a result: " + run_id)
            elif not self.backed_up(row):
                raise StopQueue("Idle run lacks a verified backup: " + run_id)
        if len(active) > 1:
            raise StopQueue("More than one run is active")
        return runs, active

    @staticmethod
    def backed_up(row):
        upload = row.get("upload", {})
        return bool(row.get("ready_to_destroy") and upload.get("verified") is True
                    and re.fullmatch(r"[0-9a-f]{64}", str(upload.get("manifest_sha256", "")))
                    and re.fullmatch(r"[0-9a-f]{40}", str(upload.get("revision", ""))))

    def compact(self, phase, run_id=None, row=None):
        signature = (phase, run_id, row.get("phase") if row else None)
        now = time.monotonic()
        if signature != self.last_status or now - self.last_emitted >= 60:
            fields = {"phase": phase, "run_id": run_id}
            progress = (row or {}).get("progress")
            if isinstance(progress, dict):
                fields["progress"] = {key: progress[key] for key in ("completed", "total")
                                      if type(progress.get(key)) is int}
            self.emit("queue_status", **fields)
            self.last_status, self.last_emitted = signature, now

    def launch(self, stage, local):
        run_id = stage["run_id"]
        if run_id in self.intents or run_id in local["runs"]:
            raise StopQueue("A launch was already attempted; refusing to relaunch: " + run_id)
        self.validate_inputs()
        reserve = positive(read(POLICY)["final_sync_reserve_seconds"])
        if time.time() + stage["max_seconds"] + reserve > self.plan["deadline_epoch"]:
            raise StopQueue("Next stage does not fit before the deadline and backup reserve")
        argv = ["--source", str(local_path(self.plan["source"], directory=True)),
                "--config", str(local_path(self.plan["config"])),
                "--protocol", str(local_path(self.plan["protocol"])),
                "--max-seconds", str(stage["max_seconds"]), "--run-id", run_id]
        if self.plan.get("source_manifest") is not None:
            argv += ["--source-manifest", str(local_path(self.plan["source_manifest"]))]
        for seed in stage["seeds"]:
            argv += ["--seed", str(seed)]
        argv += ["--", *stage["command"]]
        self.emit("launch_intent", run_id=run_id, max_seconds=stage["max_seconds"])
        self.intents.add(run_id)
        try:
            reply = self.call("run", argv)
            if reply.get("run_id") != run_id:
                raise ControllerUnavailable("launch_reply_identity_uncertain")
            self.emit("launch_reply_received", run_id=run_id)
        except ControllerUnavailable as error:
            self.emit("launch_reply_uncertain", run_id=run_id, reason=str(error))
        # The reply is never sufficient for another launch. Re-read authoritative
        # wrapper status and local state even when the launch reply looked normal.
        report, local = self.status()
        runs, _active = self.inspect(report)
        if run_id not in runs:
            raise StopQueue("Launch remains ambiguous after status reconciliation: " + run_id)
        self.emit("launch_reconciled", run_id=run_id, phase=runs[run_id]["phase"])

    def run(self):
        self.validate_inputs()
        self.emit("queue_started", stages=len(self.stages))
        while True:
            self.validate_inputs()
            report, local = self.status()
            runs, active = self.inspect(report)
            absent_prior = set(self.plan.get("prior_run_ids", [])) - set(runs)
            if absent_prior:
                raise StopQueue("A declared prior run is absent from remote status")
            absent_submitted = set(local["runs"]) - set(runs)
            if absent_submitted:
                raise StopQueue("A submitted run is absent; launch state is ambiguous")
            pending = [stage for stage in self.stages
                       if stage["run_id"] not in runs or not self.backed_up(runs[stage["run_id"]])]
            if not pending:
                if active or any(not self.backed_up(row) for row in runs.values()):
                    raise StopQueue("Queue finished but another known run is still unfinished")
                self.emit("queue_complete", stages=len(self.stages))
                return
            stage = pending[0]
            if active:
                self.compact("waiting", active[0], runs[active[0]])
                self.pause()
                continue
            if any(not self.backed_up(row) for row in runs.values()):
                raise StopQueue("Prior backups have not all been verified")
            self.compact("ready_to_launch", stage["run_id"])
            self.launch(stage, local)
            self.pause()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    queue = None
    try:
        queue = Queue(args.plan, args.poll_seconds)
        queue.log.parent.mkdir(parents=True, exist_ok=True)
        lock_path = queue.log.with_suffix(".lock")
        if lock_path.is_symlink():
            raise StopQueue("Orchestrator lock cannot be a symlink")
        with lock_path.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise StopQueue("Another AP orchestrator holds the local queue lock") from None
            queue.run()
        return 0
    except KeyboardInterrupt:
        if queue:
            queue.emit("queue_stopped", reason="User interrupted the local orchestrator")
        return 130
    except Exception as error:
        # Never print wrapper stdout/stderr, environment values, or exception
        # payloads from arbitrary libraries. Only our deliberate messages are safe.
        reason = str(error) if isinstance(error, StopQueue) else type(error).__name__
        if queue:
            queue.emit("queue_stopped", reason=reason)
        else:
            print(json.dumps({"event": "queue_stopped", "reason": reason}), flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
