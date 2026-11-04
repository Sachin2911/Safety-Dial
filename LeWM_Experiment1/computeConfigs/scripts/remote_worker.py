"""SSH-dispatched run supervisor. Runs on the rented instance, never imports a simulator."""

import base64
import collections
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import threading
import time

from compute_common import (
    ComputeError,
    atomic_json,
    error_summary,
    locked,
    process_alive,
    process_identity,
    read_json,
    relative_path,
    safe_path,
    valid_id,
)
from hf_store import sync_run
from run_io import mark_ready, publish_artifact


def run_status(root):
    state = read_json(root / ".control/status.json", {})
    upload = read_json(root / ".control/upload.json", {})
    supervisor = process_alive(state.get("supervisor"))
    child = process_alive(read_json(root / ".control/child.json"))
    result = read_json(root / "result.json")
    return {
        "run_id": root.name,
        "phase": state.get("phase", "launch_interrupted"),
        "supervisor_alive": supervisor,
        "job_alive": child,
        "heartbeat_epoch": state.get("heartbeat_epoch"),
        "result": result,
        "progress": read_json(root / ".control/progress.json"),
        "upload": {k: v for k, v in upload.items() if k != "uploaded"},
        "error": state.get("error"),
        "ready_to_destroy": bool(
            result and upload.get("verified") and not supervisor and not child
        ),
    }


def all_status(root):
    runs = root / "runs"
    return (
        [run_status(path) for path in sorted(runs.iterdir()) if path.is_dir()]
        if runs.exists()
        else []
    )


def stop_group(identity):
    if not process_alive(identity):
        return
    pid = identity["pid"]
    if os.getpgid(pid) != pid:
        raise ComputeError("Unexpected process group; refusing to signal it")
    # Keep identities of existing descendants, so an exiting group leader does not
    # make us mistake a surviving worker for a completed job.
    members = [identity]
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pid:
                member = process_identity(int(path.parent.name))
                if member:
                    members.append(member)
        except (OSError, ValueError, IndexError):
            continue
    os.killpg(pid, signal.SIGTERM)
    until = time.monotonic() + 10
    while any(process_alive(member) for member in members) and time.monotonic() < until:
        time.sleep(0.1)
    if any(process_alive(member) for member in members):
        os.killpg(pid, signal.SIGKILL)
        until = time.monotonic() + 5
        while any(process_alive(member) for member in members) and time.monotonic() < until:
            time.sleep(0.1)
        if any(process_alive(member) for member in members):
            raise ComputeError("Job processes did not exit; rental must be retained")


def extract_source(data, destination, limit):
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        total = 0
        for member in archive.getmembers():
            relative_path(member.name)
            if not member.isfile():
                raise ComputeError("Source archive must contain only regular files")
            total += member.size
            if total > limit:
                raise ComputeError("Source archive exceeds its declared limit")
            path = safe_path(destination, member.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output, archive.extractfile(member) as source:
                while block := source.read(1024 * 1024):
                    output.write(block)
            path.chmod(member.mode & 0o755)


def launch(root, request):
    spec = request["spec"]
    run_id = valid_id(spec["run_id"])
    directory = root / "runs" / run_id
    with locked(root / ".control/launch.lock"):
        if directory.exists():
            existing = read_json(directory / "run.json")
            if existing != spec:
                raise ComputeError("Run ID already belongs to a different launch")
            return run_status(directory)  # A lost SSH reply must never execute the job twice.
        if any(item["supervisor_alive"] or item["job_alive"] for item in all_status(root)):
            raise ComputeError("An active job already occupies this instance")
        if time.time() >= spec["job_deadline_epoch"]:
            raise ComputeError("Job deadline has already passed")
        directory.mkdir(parents=True, exist_ok=False)
        (directory / ".control").mkdir()
        for name in ("logs", "metrics", "artifacts"):
            (directory / name).mkdir()
        atomic_json(directory / "run.json", spec)
        atomic_json(directory / "config.json", request["experiment_config"])
        (directory / "protocol.md").write_text(request["protocol"])
        source = base64.b64decode(request["source"], validate=True)
        (directory / "source.tar.gz").write_bytes(source)
        from compute_common import sha256

        if sha256(directory / "source.tar.gz") != spec["source_sha256"]:
            raise ComputeError("Source bundle hash mismatch")
        if sha256(directory / "config.json") != spec["config_sha256"]:
            raise ComputeError("Experiment configuration hash mismatch")
        if sha256(directory / "protocol.md") != spec["protocol_sha256"]:
            raise ComputeError("Protocol hash mismatch")
        extract_source(source, directory / ".control/code", spec["source_limit_bytes"])
        for name in ("run.json", "config.json", "protocol.md", "source.tar.gz"):
            mark_ready(directory, name)
        atomic_json(directory / ".control/status.json", {"phase": "starting"})
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "supervise", str(directory)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        process.stdin.write(json.dumps({"token": request["token"]}).encode())
        process.stdin.close()
        # Wait for supervisor identity, not for the experiment or the initial upload.
        until = time.monotonic() + 10
        while time.monotonic() < until and process.poll() is None:
            state = read_json(directory / ".control/status.json", {})
            if state.get("supervisor"):
                break
            time.sleep(0.1)
        return run_status(directory)


class RunLog:
    def __init__(self, root):
        self.root = root
        self.path = root / ".control/live.log"
        self.lock = threading.Lock()
        self.number = 0
        self.path.write_bytes(b"")

    def write(self, data):
        with self.lock:
            with self.path.open("ab") as stream:
                stream.write(data)
            if self.path.stat().st_size >= 1024 * 1024:
                self._rotate()

    def _rotate(self):
        if self.path.stat().st_size:
            publish_artifact(
                f"logs/stdout-{self.number:06d}.log", source=self.path, run_dir=self.root
            )
            self.number += 1
            self.path.write_bytes(b"")

    def rotate(self):
        with self.lock:
            self._rotate()


def child_entry(root):
    spec = read_json(root / "run.json")
    state = read_json(root / ".control/status.json", {})
    atomic_json(root / ".control/child.json", process_identity(os.getpid()))
    remaining = spec["job_deadline_epoch"] - time.time()
    if remaining <= 0 or not process_alive(state.get("supervisor")):
        raise ComputeError("Supervisor disappeared or the job deadline passed before launch")
    env = {
        key: value
        for key, value in os.environ.items()
        if not any(word in key.upper() for word in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"))
    }
    env.update(
        LEWM_RUN_DIR=str(root),
        LEWM_CONFIG=str(root / "config.json"),
        LEWM_PROTOCOL=str(root / "protocol.md"),
        LEWM_SEEDS=json.dumps(spec["seeds"]),
        PYTHONPATH=str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", ""),
        PYTHONUNBUFFERED="1",
    )
    os.chdir(root / ".control/code")
    command = [
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=10s",
        f"{remaining:.1f}s",
        *spec["command"],
    ]
    os.execve(command[0], command, env)


def supervise(root, token):
    spec = read_json(root / "run.json")
    control = root / ".control"
    state = {
        "phase": "preflight",
        "supervisor": process_identity(os.getpid()),
        "started_epoch": time.time(),
        "heartbeat_epoch": time.time(),
        "error": None,
    }
    atomic_json(control / "status.json", state)
    process, reader, uploader, log = None, None, None, RunLog(root)
    upload_errors = []
    log_errors = []
    stop_reason = None

    def upload():
        try:
            sync_run(root, token)
            upload_errors.clear()
        except Exception as error:
            upload_errors[:] = [error_summary(error)]
            journal = read_json(control / "upload.json", {"uploaded": {}})
            journal["error"] = error_summary(error)
            atomic_json(control / "upload.json", journal)

    try:
        # Prove write access and preserve launch metadata before any experiment executes.
        sync_run(root, token)
        if (control / "stop.request").exists() or time.time() >= spec["job_deadline_epoch"]:
            raise ComputeError("Stopped or expired during upload preflight")
        state.update(phase="running", heartbeat_epoch=time.time())
        atomic_json(control / "status.json", state)
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "child", str(root)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        def read_output():
            try:
                while block := process.stdout.read1(65536):
                    log.write(block)
            except Exception as error:
                log_errors.append(error_summary(error))
            finally:
                process.stdout.close()

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        last_sync = time.monotonic()
        while process.poll() is None:
            if log_errors:
                raise ComputeError("Job log capture failed: " + log_errors[0])
            if (control / "stop.request").exists():
                stop_reason = "stopped"
            elif time.time() >= spec["job_deadline_epoch"]:
                stop_reason = "timed_out"
            if stop_reason:
                stop_group(read_json(control / "child.json") or process_identity(process.pid))
                break
            requested = (control / "sync.request").exists()
            if (requested or time.monotonic() - last_sync >= spec["sync_interval_seconds"]) and (
                uploader is None or not uploader.is_alive()
            ):
                (control / "sync.request").unlink(missing_ok=True)
                log.rotate()
                uploader = threading.Thread(target=upload, daemon=True)
                uploader.start()
                last_sync = time.monotonic()
            state.update(heartbeat_epoch=time.time(), error=upload_errors[-1:] or None)
            atomic_json(control / "status.json", state)
            time.sleep(1)
        returncode = process.wait(timeout=15)
        # Clean up descendants that inherited the run group after the main command exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        reader.join(timeout=15)
        if reader.is_alive():
            raise ComputeError("A detached writer still owns the job log pipe")
        if log_errors:
            raise ComputeError("Job log capture failed: " + log_errors[0])
        if uploader:
            uploader.join()
        log.rotate()
        outcome = stop_reason or (
            "completed"
            if returncode == 0
            else "timed_out"
            if returncode in (124, 137)
            else "failed"
        )
        atomic_json(
            root / "result.json",
            {"status": outcome, "exit_code": returncode, "finished_epoch": time.time()},
        )
    except Exception as error:
        stop_group(
            read_json(control / "child.json")
            or (process_identity(process.pid) if process else None)
        )
        if process:
            process.wait(timeout=15)
        if reader:
            reader.join(timeout=15)
        if uploader:
            uploader.join()
        log.rotate()
        state["error"] = error_summary(error)
        atomic_json(
            root / "result.json",
            {
                "status": "runner_failed",
                "exit_code": None,
                "error": state["error"],
                "finished_epoch": time.time(),
            },
        )
    state.update(phase="finalizing", heartbeat_epoch=time.time())
    atomic_json(control / "status.json", state)
    try:
        sync_run(root, token, final=True)
        state.update(phase="finished", error=None)
    except Exception as error:
        state.update(phase="backup_failed", error=error_summary(error))
    state["heartbeat_epoch"] = time.time()
    atomic_json(control / "status.json", state)


def recover_finished(root):
    status = run_status(root)
    if status["supervisor_alive"] or status["job_alive"]:
        raise ComputeError("Job or supervisor is still active; stop it before finalizing")
    if status["result"] is None:
        log = root / ".control/live.log"
        if log.is_file() and log.stat().st_size:
            publish_artifact("logs/recovered.log", source=log, run_dir=root)
            log.write_bytes(b"")
        atomic_json(
            root / "result.json",
            {"status": "interrupted", "exit_code": None, "finished_epoch": time.time()},
        )


def dispatch(request):
    root = Path(request["root"])
    if root != Path("/workspace/LeWM_Experiment1"):
        raise ComputeError("Unexpected remote workspace")
    session = read_json(root / ".control/session.json")
    if not session or session["session_id"] != request["session_id"]:
        raise ComputeError("Remote session identity does not match the selected rental")
    action = request["action"]
    if action == "status":
        runs = all_status(root)
        return {"runs": runs, "ready_to_destroy": all(r["ready_to_destroy"] for r in runs)}
    if action == "launch":
        return launch(root, request)
    run_id = request.get("run_id")
    directories = (
        [root / "runs" / valid_id(run_id)]
        if run_id
        else [root / "runs" / item["run_id"] for item in all_status(root)]
    )
    if action == "logs":
        if len(directories) != 1:
            raise ComputeError("Specify a run ID for logs")
        lines = collections.deque(maxlen=min(int(request.get("lines", 100)), 2000))
        paths = sorted((directories[0] / "logs").glob("stdout-*.log"))[-4:]
        paths.append(directories[0] / ".control/live.log")
        for path in paths:
            if path.is_file():
                with path.open(errors="replace") as stream:
                    lines.extend(stream)
        return {"text": "".join(lines)}
    results = []
    for directory in directories:
        if not (directory / "run.json").is_file():
            raise ComputeError("Unknown run ID")
        status = run_status(directory)
        if action == "stop":
            (directory / ".control/stop.request").touch()
            if not status["supervisor_alive"]:
                stop_group(read_json(directory / ".control/child.json"))
            results.append(run_status(directory))
        elif action == "sync":
            if status["supervisor_alive"]:
                (directory / ".control/sync.request").touch()
                results.append({"run_id": directory.name, "sync_requested": True})
            else:
                recover_finished(directory)
                sync_run(directory, request["token"], final=True)
                state = read_json(directory / ".control/status.json", {})
                state.update(phase="finished", error=None, heartbeat_epoch=time.time())
                atomic_json(directory / ".control/status.json", state)
                results.append(run_status(directory))
        else:
            raise ComputeError("Unknown worker action")
    return {"runs": results}


def main():
    try:
        if sys.argv[1:] and sys.argv[1] in ("supervise", "child"):
            root = Path(sys.argv[2]).resolve()
            if sys.argv[1] == "child":
                child_entry(root)
            else:
                supervise(root, json.load(sys.stdin)["token"])
        else:
            print(json.dumps(dispatch(json.load(sys.stdin))))
    except Exception as error:
        print(json.dumps({"error": error_summary(error)}))
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
