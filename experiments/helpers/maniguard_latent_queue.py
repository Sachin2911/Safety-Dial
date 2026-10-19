"""Durable serial physical jobs with explicit failure and interruption accounting."""

import calendar
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from helpers.maniguard_latent_hardware import assignment
from helpers.maniguard_latent_results import read_episode

ROOT = Path(__file__).resolve().parents[2]


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def process_identity(pid):
    """PID plus Linux process start ticks protects against PID reuse on resume."""
    path = Path(f"/proc/{pid}/stat")
    try:
        text = path.read_text()
        fields = text[text.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": fields[19]}
    except FileNotFoundError:
        return None


class PhysicalQueue:
    def __init__(self, run, protocol):
        self.run, self.protocol_path = Path(run), Path(protocol).resolve()
        self.protocol = json.loads(self.protocol_path.read_text())
        self.run.mkdir(parents=True, exist_ok=True)
        self.lock = (self.run / "coordinator.lock").open("a")
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = self.run / "queue.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "schema": "maniguard-latent-queue-v1", "jobs": {}, "attempts": []}
        self.protocol_hash = hashlib.sha256(self.protocol_path.read_bytes()).hexdigest()
        self.cutoff = calendar.timegm(time.strptime(
            self.protocol["physical_cutoff_utc"], "%Y-%m-%dT%H:%M:%SZ"))

    def save(self):
        atomic(self.path, self.state)

    def plan(self, jobs):
        for job in jobs:
            if job["id"] in self.state["jobs"]:
                if self.state["jobs"][job["id"]]["definition"] != job:
                    raise ValueError("An immutable planned job changed: " + job["id"])
            else:
                self.state["jobs"][job["id"]] = {"definition": job, "status": "pending",
                                               "attempt_ids": []}
        self.save()

    def _cost_gate(self, engineering):
        if engineering:
            # Earlier manual starts are retained in the root ledger as well.
            ledger = self.run / "attempts.jsonl"
            previous = [json.loads(line) for line in ledger.read_text().splitlines()] \
                if ledger.exists() else []
            starts = {row["id"] for row in previous if row.get("stage") == "engineering"}
            if len(starts) >= 6:
                raise RuntimeError("Six-start engineering repair allowance exhausted")
        else:
            full = [a for a in self.state["attempts"] if not a["engineering"]]
            if len(full) >= 64:
                raise RuntimeError("64 full physical attempts exhausted")
            invalid = [a for a in full if a.get("status") == "invalid"]
            if len(invalid) > 4:
                raise RuntimeError("Infrastructure replacement allowance exhausted")

    def command(self, job, output):
        devices = assignment(job.get("worker"))
        if "worker" in job and self.protocol.get("workers") != 2:
            raise ValueError("Concurrent worker placement requires the declared two-worker protocol")
        command = [str(ROOT / "data/maniguard/envs/sim/bin/python"), "-u",
                   str(ROOT / "experiments/scripts/maniguard_latent_episode.py"),
                   "--family", job["family"], "--scene", job["scene"],
                   "--seed", str(job["seed"]), "--controller", job["controller"],
                   "--protocol", str(self.protocol_path), "--output", str(output),
                   "--port", str(devices["port"]),
                   "--simulation-device", str(devices["simulation_device"]),
                   "--policy-device", str(devices["policy_device"])]
        if job.get("engineering"):
            command += ["--engineering-max-steps", str(job["max_steps"])]
        return command

    def environment(self, job):
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES=str(assignment(job.get("worker"))["simulation_device"]),
                           OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                           LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
                           __EGL_VENDOR_LIBRARY_FILENAMES=str(
                               ROOT / "configs/maniguard/latent-evolution/nvidia-egl-vendor.json"),
                           OMNI_KIT_ACCEPT_EULA="YES", OMNIGIBSON_HEADLESS="1")
        return environment

    def evaluate(self, job_id, *, repair_note=None, episode_seconds=180):
        row = self.state["jobs"][job_id]
        job = row["definition"]
        if row["status"] == "completed":
            receipt = read_episode(row["output"], require_full=not job.get("engineering"))
            return receipt
        if row["status"] == "invalid" and not repair_note:
            raise RuntimeError("Diagnose and repair before resuming failed job: " + job_id)
        if row["status"] == "running":
            attempt = next(a for a in self.state["attempts"] if a["id"] == row["attempt_ids"][-1])
            identity = attempt.get("process_identity")
            if identity is not None and process_identity(identity["pid"]) == identity:
                # A process can survive a coordinator interruption. Observe the
                # same handle rather than launching an overlapping replacement.
                while process_identity(identity["pid"]) == identity:
                    if time.time() > attempt["deadline_epoch"]:
                        os.killpg(identity["pid"], signal.SIGTERM)
                        time.sleep(5)
                        if process_identity(identity["pid"]) == identity:
                            os.killpg(identity["pid"], signal.SIGKILL)
                        break
                    time.sleep(5)
            return self._reconcile(row, attempt)
        self._cost_gate(job.get("engineering", False))
        if time.time() + episode_seconds > self.cutoff:
            raise RuntimeError("Insufficient prospective time before preservation cutoff")
        attempt_id = job_id + f"-a{len(row['attempt_ids'])}"
        output = self.run / "episodes" / attempt_id
        output.parent.mkdir(exist_ok=True)
        command = self.command(job, output)
        attempt = {"id": attempt_id, "job_id": job_id, "stage": job["stage"],
                   "engineering": job.get("engineering", False), "status": "starting",
                   "command": command, "output": str(output), "started_epoch": time.time(),
                   "deadline_epoch": min(time.time() + (1200 if job.get("engineering") else 3600),
                                         self.cutoff),
                   "protocol_sha256": self.protocol_hash, "repair_note": repair_note}
        row.update(status="running", output=str(output))
        row["attempt_ids"].append(attempt_id)
        self.state["attempts"].append(attempt)
        self.save()
        append(self.run / "attempts.jsonl", attempt)
        environment = self.environment(job)
        with (self.run / "episodes" / (attempt_id + ".log")).open("x") as log:
            request = self.run / "episodes" / (attempt_id + "-request.json")
            terminal = self.run / "episodes" / (attempt_id + "-terminal.json")
            atomic(request, {"command": command, "cwd": str(ROOT)})
            attempt["terminal_receipt"] = str(terminal)
            worker_command = [str(ROOT / "data/maniguard/envs/tools/bin/python"), "-u",
                              str(ROOT / "experiments/scripts/maniguard_latent_worker.py"),
                              "--request", str(request), "--receipt", str(terminal)]
            worker = subprocess.Popen(worker_command, cwd=ROOT, env=environment, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            attempt["process_identity"] = process_identity(worker.pid)
            attempt["status"] = "running"
            self.save()
            try:
                code = worker.wait(timeout=max(1, attempt["deadline_epoch"] - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGTERM)
                try:
                    worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()
                code = worker.returncode
                attempt["infrastructure_timeout"] = True
            attempt["exit_code"] = code
        return self._reconcile(row, attempt)

    def _reconcile(self, row, attempt):
        attempt["finished_epoch"] = time.time()
        attempt["elapsed_seconds"] = attempt["finished_epoch"] - attempt["started_epoch"]
        try:
            receipt = read_episode(attempt["output"], require_full=not attempt["engineering"])
            terminal = json.loads(Path(attempt["terminal_receipt"]).read_text())
            attempt["observed_finished_epoch"] = attempt["finished_epoch"]
            attempt["finished_epoch"] = terminal["finished_epoch"]
            attempt["elapsed_seconds"] = terminal["finished_epoch"] - terminal["started_epoch"]
            if terminal["exit_code"] != 0 or attempt.get("exit_code", 0) != 0:
                raise ValueError("Nonzero process exit despite a physical receipt")
        except Exception as exc:
            attempt.update(status="invalid", error=type(exc).__name__ + ": " + str(exc))
            row["status"] = "invalid"
            self.save()
            append(self.run / "attempts.jsonl", attempt)
            raise RuntimeError("Worker returned to recovery loop: " + attempt["error"]) from exc
        attempt["status"] = row["status"] = "completed"
        atomic(Path(attempt["output"]) / "verified-receipt.json", receipt)
        row["score"] = receipt["score"]
        self.save()
        append(self.run / "attempts.jsonl", attempt)
        return receipt
