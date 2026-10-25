"""Run the frozen DSRL06 engineering or development queue on one prepared host.

Run with the BEHAVIOR interpreter. Process and HTTP intentions are durable: losing
an observation never authorizes a second simulator start or a duplicate mutation.
The independent financial watchdog remains responsible for provider shutdown.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_protocol import CAPS, ProtocolLedger, frozen_jobs, select_probe_snapshots
from helpers.dsrl06_validation import build_episode


class PendingOperation(RuntimeError):
    """An existing operation must be reconciled, never replaced automatically."""


class AdmissionClosed(RuntimeError):
    """A frozen time, resource or repair condition prevents further work."""


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)


def timestamp(value):
    result = dt.datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Queue timestamps require an explicit timezone")
    return result.timestamp()


def process_identity(pid):
    """Linux start ticks and boot identity prevent following a reused PID."""
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return {
            "pid": int(pid),
            "start_ticks": int(stat[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except (OSError, ValueError, IndexError):
        return None


def identity_alive(identity):
    return bool(identity and process_identity(identity["pid"]) == identity)


def group_members(group):
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            values = path.read_text().rsplit(")", 1)[1].split()
            if int(values[2]) == group:
                identity = process_identity(int(path.parent.name))
                if identity:
                    members.append(identity)
        except (OSError, ValueError, IndexError):
            continue
    return members


def worker(spec_path):
    """Detached, journaled child owner, including when its coordinator disconnects."""
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text())
    with spec_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = Path(spec["receipt"])
        if receipt.exists():
            return 0
        atomic_json(spec["identity"], {**process_identity(os.getpid()), "started_utc": utc()})
        environment = dict(os.environ)
        environment.update(spec["environment"])
        if spec["kind"] == "episode":
            environment.pop("EVAL_USE_GPU_DYNAMICS", None)
        begin = time.time()
        try:
            with Path(spec["log"]).open("a") as log:
                child = subprocess.Popen(
                    spec["command"],
                    cwd=spec["cwd"],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                atomic_json(spec["child_identity"], process_identity(child.pid))
                result = child.wait()
            atomic_json(
                receipt,
                {
                    "returncode": result,
                    "started_utc": spec["created_utc"],
                    "finished_utc": utc(),
                    "elapsed_seconds": time.time() - begin,
                },
            )
            return result
        except BaseException as error:
            atomic_json(
                receipt,
                {
                    "returncode": 125,
                    "error_type": type(error).__name__,
                    "finished_utc": utc(),
                    "elapsed_seconds": time.time() - begin,
                },
            )
            return 125


def http_worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    request = urllib.request.Request(
        spec["url"],
        data=json.dumps(spec["body"]).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=21600) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            body, body_is_json = json.loads(raw), True
        except (ValueError, UnicodeDecodeError):
            body, body_is_json = raw.decode("utf-8", errors="replace"), False
        output = Path(spec["output"])
        error_path = (
            output.with_name(output.name.removesuffix(".response.json") + ".error.json")
            if output.name.endswith(".response.json")
            else output.with_suffix(".error.json")
        )
        receipt = {
            "schema": "dsrl06-http-error-v1",
            "http_status": int(error.code),
            "body": body,
            "body_is_json": body_is_json,
            "body_sha256": hashlib.sha256(raw).hexdigest(),
            "request_sha256": sha256(spec_path),
        }
        error_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=error_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
                try:
                    os.link(temporary, error_path)
                except FileExistsError:
                    if json.loads(error_path.read_text()) != receipt:
                        raise ValueError(
                            "Refusing to replace a different immutable HTTP error receipt"
                        )
            finally:
                temporary.unlink(missing_ok=True)
        raise
    atomic_json(spec["output"], result)
    return 0


def actor_parity_worker(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    with np.load(spec["inputs"], allow_pickle=False) as arrays:
        pixels, states = arrays["pixels"], arrays["states"]
    latencies, differences = [], []
    for index in range(min(len(pixels), spec["max_observations"])):
        body = {
            "job_id": "E3",
            "query_index": index,
            "rng_seed": spec["rng_seed"],
            "actor_version": spec["actor_version"],
            "state": states[index].tolist(),
            "pixels_b64": base64.b64encode(pixels[index].tobytes()).decode("ascii"),
        }
        samples = []
        for port in (9009, 9010):
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/act",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            begin = time.monotonic()
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.load(response)
            elapsed = time.monotonic() - begin
            if port == 9010:
                latencies.append(elapsed)
            sample = np.asarray(result["noise"], np.float32)
            if result["actor_version"] != spec["actor_version"] or sample.shape != (32,):
                raise ValueError("CPU parity actor version or latent shape differs")
            if not np.isfinite(sample).all() or np.max(np.abs(sample)) > 1.000001:
                raise ValueError("CPU parity requires finite bounded actor samples")
            samples.append(sample)
        differences.append(float(np.max(np.abs(samples[0] - samples[1]))))
        if not np.allclose(samples[0], samples[1], atol=spec["atol"], rtol=spec["rtol"]):
            raise ValueError("Published CPU actor differs from the GPU actor")
    atomic_json(
        spec["output"],
        {
            "valid": True,
            "observations": len(differences),
            "maximum_difference": max(differences),
            "cpu_seconds": latencies,
            "median_cpu_seconds": float(np.median(latencies[1:] or latencies)),
            "atol": spec["atol"],
            "rtol": spec["rtol"],
        },
    )
    return 0


class Processes:
    """One immutable command per key; resume adopts the original live worker."""

    def __init__(self, directory, python, script):
        self.directory, self.python, self.script = Path(directory), str(python), str(script)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, key):
        if not key or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
            for c in key
        ):
            raise ValueError("Invalid managed process identity")
        return self.directory / f"{key}.json"

    def spec(self, key):
        return json.loads(self.path(key).read_text())

    def status(self, key):
        spec = self.spec(key)
        if spec.get("stopped"):
            return {"state": "stopped"}
        receipt = Path(spec["receipt"])
        if receipt.exists():
            return {"state": "exited", **json.loads(receipt.read_text())}
        identity_path = Path(spec["identity"])
        if identity_path.exists():
            saved = json.loads(identity_path.read_text())
            identity = {k: saved[k] for k in ("pid", "start_ticks", "boot_id")}
            return {"state": "running" if identity_alive(identity) else "unresolved", **saved}
        # Reconcile the launch gap by the exact wrapper argument, without inspecting env secrets.
        matches = []
        expected = str(self.path(key).resolve()).encode()
        for path in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                args = path.read_bytes().split(b"\0")
                if b"--managed-worker" in args and expected in args:
                    identity = process_identity(int(path.parent.name))
                    if identity:
                        matches.append(identity)
            except OSError:
                continue
        if len(matches) == 1:
            atomic_json(identity_path, {**matches[0], "started_utc": spec["created_utc"]})
            return {"state": "running", **matches[0]}
        return {"state": "unresolved"}

    def launch(self, key, command, *, cwd, environment=None, kind="process"):
        path = self.path(key)
        stable = {
            "command": list(map(str, command)),
            "cwd": str(cwd),
            "environment": dict(environment or {}),
            "kind": kind,
        }
        if path.exists():
            previous = self.spec(key)
            if any(previous.get(k) != v for k, v in stable.items()):
                raise ValueError("A resumed managed process cannot change its command")
            return self.status(key)
        spec = {
            **stable,
            "created_utc": utc(),
            "receipt": str(path.with_suffix(".exit.json")),
            "identity": str(path.with_suffix(".pid.json")),
            "child_identity": str(path.with_suffix(".child.json")),
            "log": str(path.with_suffix(".log")),
        }
        atomic_json(path, spec)  # Intention precedes process creation.
        with Path(spec["log"]).open("a") as log:
            process = subprocess.Popen(
                [self.python, self.script, "--managed-worker", str(path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        identity = process_identity(process.pid)
        if identity:
            atomic_json(spec["identity"], {**identity, "started_utc": spec["created_utc"]})
        return self.status(key)

    def stop(self, key):
        spec = self.spec(key)
        status = self.status(key)
        if status["state"] == "running":
            identity = {k: status[k] for k in ("pid", "start_ticks", "boot_id")}
            members = group_members(identity["pid"])
            os.killpg(identity["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 5
            while any(identity_alive(member) for member in members) and time.monotonic() < deadline:
                time.sleep(0.1)
            if any(identity_alive(member) for member in members):
                os.killpg(identity["pid"], signal.SIGKILL)
            if any(identity_alive(member) for member in members):
                deadline = time.monotonic() + 5
                while group_members(identity["pid"]) and time.monotonic() < deadline:
                    time.sleep(0.1)
            if group_members(identity["pid"]):
                raise PendingOperation(f"Managed process group {key} did not quiesce")
        elif status["state"] == "unresolved":
            raise PendingOperation(f"Cannot verify stopped managed operation {key}")
        elif Path(spec["identity"]).exists():
            saved = json.loads(Path(spec["identity"]).read_text())
            if group_members(saved["pid"]):
                raise PendingOperation(f"Exited wrapper {key} still has a live process group")
        spec["stopped"] = True
        spec["stopped_utc"] = utc()
        atomic_json(self.path(key), spec)

    def actual_pids(self):
        result = set()
        for suffix in ("*.pid.json", "*.child.json"):
            for path in self.directory.glob(suffix):
                value = json.loads(path.read_text())
                if value:
                    result.add(value["pid"])
        return sorted(result)


def verify_plan(root, path, stage):
    plan = json.loads(Path(path).read_text())
    if not plan.get("authorization", {}).get("user_launch"):
        raise ValueError("Source freeze lacks launch authorization")
    for name, digest in plan["source_sha256"].items():
        if sha256(Path(root) / name) != digest:
            raise ValueError(f"Frozen source changed: {name}")
    if plan["jobs"] != frozen_jobs() or plan["caps"] != dict(CAPS):
        raise ValueError("Frozen jobs or counters differ from the reviewed protocol")
    if plan["money"]["total_usd"] != 15:
        raise ValueError("The total authorization is $15")
    if stage == "pilot" and (
        plan.get("stage") not in ("B", "pilot") or not plan.get("stage_a_summary_sha256")
    ):
        raise ValueError("Pilot requires a separate reviewed stage-B freeze")
    return plan


def validate_amendment(old, new, receipt, old_hash, new_hash):
    if (
        new.get("supersedes_sha256") != old_hash
        or new.get("amendment", {}).get("type") != "engineering_repair"
        or not receipt
        or not receipt.get("repair")
        or receipt.get("old_freeze_sha256") != old_hash
        or receipt.get("new_freeze_sha256") != new_hash
    ):
        raise ValueError("A source amendment needs an explicit old/new freeze repair receipt")
    permitted = {"source_sha256", "created_utc", "supersedes_sha256", "amendment"}
    if {k: v for k, v in old.items() if k not in permitted} != {
        k: v for k, v in new.items() if k not in permitted
    }:
        raise ValueError("A technical repair cannot change the scientific or financial freeze")


class Coordinator:
    def __init__(self, root, plan_path, rental_path, stage, *, repair_receipt=None):
        self.root, self.stage = Path(root).resolve(), stage
        self.plan_path, self.rental_path = Path(plan_path), Path(rental_path)
        self.plan = verify_plan(self.root, self.plan_path, stage)
        self.rental = json.loads(self.rental_path.read_text())
        if str(self.rental["instance_id"]) != os.environ.get("CONTAINER_ID"):
            raise ValueError("Queue is not running on the frozen rental")
        self.output = self.root / "outputs/dsrl06"
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / "queue.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state_path = self.output / "queue-state.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.exists()
            else {
                "schema": "dsrl06-queue-v1",
                "instance_id": self.rental["instance_id"],
                "episodes": {},
                "stages": {},
                "http": {},
                "server_generations": {},
            }
        )
        if self.state["instance_id"] != self.rental["instance_id"]:
            raise ValueError("Contemporary comparator jobs cannot span physical hosts")
        self.repair = json.loads(Path(repair_receipt).read_text()) if repair_receipt else None
        if self.repair is not None:
            self.repair["receipt_sha256"] = sha256(repair_receipt)
        digest = sha256(self.plan_path)
        if stage in self.state["stages"] and self.state["stages"][stage] != digest:
            old_hash = self.state["stages"][stage]
            old = json.loads((self.output / "source-freezes" / f"{old_hash}.json").read_text())
            validate_amendment(old, self.plan, self.repair, old_hash, digest)
        frozen_path = self.output / "source-freezes" / f"{digest}.json"
        if not frozen_path.exists():
            frozen_path.parent.mkdir(parents=True, exist_ok=True)
            with frozen_path.open("xb") as stream:
                stream.write(self.plan_path.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
        self.state["stages"][stage] = digest
        self.ledger = ProtocolLedger(self.output / "resources.jsonl")
        self.behavior = "/opt/miniforge3/envs/behavior/bin/python"
        self.openpi = str(self.root / "third_party/openpi/.venv/bin/python")
        self.learner_python = str(self.root / "third_party/dsrl-venv/bin/python")
        self.processes = Processes(
            self.output / "processes", self.behavior, Path(__file__).resolve()
        )
        self.query_ledger = self.output / "queries.sqlite"
        self.query_root = self.output / "queries"
        self.learner_dir = self.output / "learner"
        self.actor_url = "http://127.0.0.1:9009"
        self.repair = json.loads(Path(repair_receipt).read_text()) if repair_receipt else None
        if self.repair is not None:
            self.repair["receipt_sha256"] = sha256(repair_receipt)
        self.save()
        self.sync_counters()

    def save(self):
        atomic_json(self.state_path, self.state)

    def event(self, event, **fields):
        with (self.output / "queue-events.jsonl").open("a") as stream:
            stream.write(
                json.dumps({"utc": utc(), "event": event, **fields}, allow_nan=False) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())

    def deadline(self):
        result = timestamp(self.rental["allocation_deadline_utc"]) - 1200
        if self.stage == "engineering":
            result = min(result, timestamp(self.rental["created_utc"]) + 90 * 60)
        return result

    def guard(self):
        if (self.output / "STOP").exists() or time.time() >= self.deadline():
            raise AdmissionClosed("Working-stage or preservation deadline reached")

    def sync_counters(self):
        counts = {"policy_controls": 0, "warmup_controls": 0, "model_queries": 0}
        cache = self.state.setdefault("source_counters", {})

        def events(path):
            state = cache.setdefault(str(path), {"offset": 0, "counts": {}})
            if path.stat().st_size < state["offset"]:
                raise ValueError("An append-only source journal was truncated")
            with path.open("rb") as stream:
                stream.seek(state["offset"])
                while True:
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    row = json.loads(line)
                    label = row["event"]
                    if label == "update_reserved":
                        label += ":" + row["phase"]
                        if row.get("origin") == "workstation_cpu_preflight":
                            if row.get("receipt") != self.plan["prior_engineering_receipt"]:
                                raise ValueError("Prior engineering receipt changed")
                            label += ":imported"
                    state["counts"][label] = state["counts"].get(label, 0) + 1
                    state["offset"] = stream.tell()
            return state["counts"]

        for path in (self.output / "episodes").rglob("*.journal.jsonl"):
            observed = events(path)
            counts["policy_controls"] += observed.get("before_action", 0)
            counts["warmup_controls"] += observed.get("warmup_attempt", 0)
        if self.query_ledger.exists():
            with sqlite3.connect(f"file:{self.query_ledger}?mode=ro", uri=True) as db:
                counts["model_queries"] = db.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        updates = {"engineering": 0, "learning": 0}
        learner_ledger = self.learner_dir / "learner-ledger.jsonl"
        if learner_ledger.exists():
            observed = events(learner_ledger)
            updates["engineering"] = observed.get("update_reserved:engineering", 0)
            updates["learning"] = observed.get("update_reserved:training", 0)
            imported = observed.get("update_reserved:engineering:imported", 0)
            if imported > self.plan["prior_engineering_updates"]:
                raise ValueError("Learner imported more prior updates than the source freeze")
        # Imported workstation attempts count even before the GPU learner is started.
        updates["engineering"] += self.plan["prior_engineering_updates"]
        usage = self.ledger.summary()["usage"]
        for kind, actual in counts.items():
            if actual < usage[kind]:
                raise ValueError(f"Source accounting decreased for {kind}")
            if actual > usage[kind]:
                self.ledger.charge(kind, actual - usage[kind], event_id=f"observed:{kind}:{actual}")
        for phase, actual in updates.items():
            recorded = usage[f"{phase}_updates"]
            if actual < recorded:
                raise ValueError("Source learner accounting decreased")
            if actual > recorded:
                self.ledger.charge(
                    "learner_updates",
                    actual - recorded,
                    phase=phase,
                    event_id=f"observed:{phase}_updates:{actual}",
                )
        actor_calls = sum(value["counts"].get("actor_request", 0) for value in cache.values())
        for path in self.output.rglob("actor-ledger.jsonl"):
            previous = cache.get(str(path), {}).get("counts", {}).get("actor_request", 0)
            observed = events(path)
            actor_calls += observed.get("actor_request", 0) - previous
        self.save()
        return {
            **self.ledger.summary(),
            "actor_calls": actor_calls,
            "scene_internal_physics_controls": None,
        }

    def tick(self):
        self.sync_counters()
        self.guard()
        self.poll_episodes()
        if time.time() - self.state.get("last_memory_sample", 0) >= 10:
            self.state["last_memory_sample"] = time.time()
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            memory = {
                line.split(":")[0]: line.split(":")[1].strip()
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith(("MemTotal:", "MemAvailable:"))
            }
            self.event(
                "resource_sample",
                gpu_rows=result.stdout.splitlines() if result.returncode == 0 else [],
                host_memory=memory,
            )
        time.sleep(1)

    def wait(self, key, *, allow_failure=False):
        while True:
            status = self.processes.status(key)
            if status["state"] == "exited":
                self.sync_counters()
                if status["returncode"] and not allow_failure:
                    raise RuntimeError(f"Managed operation {key} exited {status['returncode']}")
                return status
            if status["state"] != "running":
                raise PendingOperation(f"Reconcile unresolved managed operation {key}")
            self.tick()

    def gpu_environment(self, lane):
        result = {
            "CUDA_VISIBLE_DEVICES": str(lane),
            "PYTHONUNBUFFERED": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "WANDB_MODE": "disabled",
            "MPLBACKEND": "Agg",
        }
        return result

    def server(self, lane, generation=None):
        if generation is None:
            current = self.state["server_generations"].get(str(lane))
            generation = current.split(".", 1)[1] if current else "collection"
        key = f"server{lane}.{generation}"
        environment = self.gpu_environment(lane)
        environment["PYTHONPATH"] = f"{self.root}/third_party/ManiGuard:{self.root}/experiments"
        command = [
            self.openpi,
            "-u",
            str(self.root / "experiments/scripts/serve_dsrl06.py"),
            "--checkpoint",
            str(self.root / "third_party/checkpoints/jar/7400"),
            "--port",
            str(8000 + lane),
            "--query-ledger",
            str(self.query_ledger),
            "--record-root",
            str(self.query_root),
            "--actor-url",
            self.actor_url,
        ]
        self.processes.launch(key, command, cwd=self.root, environment=environment, kind="server")
        self.state["server_generations"][str(lane)] = key
        self.save()
        while True:
            if self.processes.status(key)["state"] != "running":
                raise PendingOperation(f"Policy server {key} is not running")
            try:
                with socket.create_connection(("127.0.0.1", 8000 + lane), timeout=1):
                    break
            except OSError:
                self.tick()
        startup = f"startup{lane}.{generation}"
        receipt = self.output / f"{startup}.json"
        command = [
            self.openpi,
            str(self.root / "experiments/scripts/probe_dsrl06.py"),
            "startup",
            "--port",
            str(8000 + lane),
            "--query-ledger",
            str(self.query_ledger),
            "--record-root",
            str(self.query_root),
            "--output",
            str(receipt),
            "--queries",
            "3",
            "--startup-id",
            f"{self.rental['instance_id']}:{key}",
        ]
        self.processes.launch(startup, command, cwd=self.root, kind="probe")
        self.wait(startup)
        if not json.loads(receipt.read_text()).get("valid"):
            raise ValueError("Server startup probe failed")
        return key

    def get_status(self):
        with urllib.request.urlopen(self.actor_url + "/status", timeout=5) as response:
            return json.load(response)

    def learner(self, phase):
        key = self.state.get("learner_process", "learner")
        resume = None
        if self.processes.path(key).exists() and self.processes.status(key)["state"] != "running":
            if self.processes.status(key)["state"] == "unresolved":
                raise PendingOperation("Reconcile the original learner process before any restore")
            resume = (self.repair or {}).get("resume_checkpoint")
            if not resume:
                raise PendingOperation(
                    "A stopped learner needs a repair receipt naming its checkpoint"
                )
            resume = str(Path(resume).resolve())
            if not Path(resume).is_relative_to((self.learner_dir / "checkpoints").resolve()):
                raise ValueError("Learner resume must use this run's preserved checkpoint")
            checkpoint = json.loads((Path(resume) / "state.json").read_text())
            ledger = self.learner_dir / "learner-ledger.jsonl"
            appended = {
                json.loads(line)["round_id"]
                for line in ledger.read_text().splitlines()
                if json.loads(line)["event"] == "round_appended"
            }
            if not appended <= set(checkpoint["append_rounds"]):
                raise PendingOperation(
                    "Uncheckpointed replay admissions require diagnosis before restore"
                )
            generation = self.state.get("learner_generation", 0) + 1
            key = f"learner.resume{generation}"
            self.state["learner_generation"] = generation
            self.state["learner_process"] = key
            self.state.setdefault("learner_resume_checkpoints", {})[key] = resume
            self.save()
        if not self.processes.path(key).exists():
            resume = resume or self.state.get("learner_resume_checkpoints", {}).get(key)
            command = [
                self.learner_python,
                "-u",
                str(self.root / "experiments/scripts/dsrl06_learner_service.py"),
                "--work-dir",
                str(self.learner_dir),
                "--data-root",
                str(self.output),
                "--phase",
                "engineering",
                "--prior-engineering-updates",
                str(self.plan["prior_engineering_updates"]),
                "--prior-engineering-receipt",
                self.plan["prior_engineering_receipt"],
            ]
            if resume:
                command.extend(["--resume", resume])
            self.processes.launch(
                key, command, cwd=self.root, environment=self.gpu_environment(3), kind="learner"
            )
            self.state["learner_process"] = key
            self.save()
        while True:
            if self.processes.status(key)["state"] != "running":
                raise PendingOperation("Learner is not running; restore its last full checkpoint")
            try:
                status = self.get_status()
                break
            except (OSError, ValueError):
                self.tick()
        if phase == "training" and status["phase"] != "training":
            self.http(
                f"initialize-training-{self.state.get('learner_generation', 0)}",
                "/initialize",
                {"phase": "training"},
            )
        return self.get_status()

    def begin_http(self, key, route, body):
        directory = self.output / "http"
        request_path, output = directory / f"{key}.request.json", directory / f"{key}.response.json"
        spec = {"url": self.actor_url + route, "body": body, "output": str(output)}
        if request_path.exists() and json.loads(request_path.read_text()) != spec:
            raise ValueError("A resumed learner operation cannot change its request")
        if not request_path.exists():
            atomic_json(request_path, spec)
        self.processes.launch(
            f"http.{key}",
            [self.behavior, str(Path(__file__).resolve()), "--http-worker", str(request_path)],
            cwd=self.root,
            kind="http",
        )
        return output

    def finish_http(self, key):
        output = self.output / "http" / f"{key}.response.json"
        if (output.parent / f"{key}.error.json").exists():
            raise PendingOperation(f"HTTP operation {key} has a preserved failure receipt")
        if output.exists():
            return json.loads(output.read_text())
        status = self.wait(f"http.{key}", allow_failure=True)
        if output.exists():
            return json.loads(output.read_text())
        recovered = self.recover_http(key)
        if recovered is not None:
            atomic_json(output, recovered)
            return recovered
        # A lost HTTP response cannot restart a live source update. Preserve the
        # original operation and its update ledger for explicit diagnosis/resume.
        raise PendingOperation(f"HTTP operation {key} has no response ({status['returncode']})")

    def recover_http(self, key):
        request_path = self.output / "http" / f"{key}.request.json"
        if not request_path.exists():
            return None
        request = json.loads(request_path.read_text())
        route, body = request["url"].rsplit("/", 1)[1], request["body"]
        if (request_path.parent / f"{key}.error.json").exists():
            return None
        # Match LearnerBridge.json_hash exactly, including its default separators.
        request_fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        ledger = self.learner_dir / "learner-ledger.jsonl"
        rows = (
            [json.loads(line) for line in ledger.read_text().splitlines()]
            if ledger.exists()
            else []
        )
        event = {
            "train": "train_completed",
            "publish": "actor_published",
            "append": "round_appended",
            "profile": "profile_completed",
        }.get(route)
        candidates = [
            row
            for row in rows
            if row["event"] == event
            and ("round_id" not in body or row.get("round_id") == body["round_id"])
            and (route != "profile" or row.get("request_sha256") == request_fingerprint)
        ]
        if candidates:
            row = candidates[-1]
            if route == "profile" and (
                row.get("timed_updates") != int(body.get("updates", 0))
                or ("source_parity" in row) != bool(body.get("parity", True))
            ):
                return None
            if route == "append":
                if (
                    row["jobs"] != [d["job_id"] for d in body["episodes"]]
                    or row["warmup"] != body["warmup"]
                ):
                    raise ValueError("Recovered append differs from the durable request")
                return {
                    "round_id": row["round_id"],
                    "new_transitions": row["transitions"],
                    "warmup": row["warmup"],
                    "status": self.get_status(),
                }
            return {**row, "status": self.get_status()} if route == "profile" else row
        if route in ("save", "export-actor"):
            path = (
                self.learner_dir
                / ("checkpoints" if route == "save" else "actor_exports")
                / body["name"]
            )
            manifest_path = path / "manifest.json"
            if manifest_path.exists():
                for name, digest in json.loads(manifest_path.read_text()).items():
                    if Path(name).name != name or sha256(path / name) != digest:
                        raise ValueError("Recovered learner snapshot hash differs")
                result = {"path": str(path), "manifest_sha256": sha256(manifest_path)}
                if route == "export-actor":
                    result.update(
                        actor_version=json.loads((path / "state.json").read_text())[
                            "actor_version"
                        ],
                        actor_evaluation_mode="cpu_published_actor",
                    )
                return result
        if route == "initialize":
            status = self.get_status()
            if status["initialized"] and status["phase"] == body["phase"]:
                return status
        return None

    def http(self, key, route, body):
        self.begin_http(key, route, body)
        return self.finish_http(key)

    def start_episode(self, job, *, mode=None, actor_version=-1):
        job_id = job["job_id"]
        previous = self.state["episodes"].get(job_id)
        if previous and previous.get("valid"):
            return previous
        if previous and previous.get("valid") is None:
            return previous  # Adopt its original live or completed process, never relaunch.
        attempt = 1 if previous is None else previous["attempt"] + 1
        if attempt > 1 and (
            not self.repair
            or self.repair.get("job_id") != job_id
            or self.repair.get("attempt") != attempt
            or not self.repair.get("repair")
        ):
            raise AdmissionClosed("Invalid attempts require a recorded repair before replacement")
        if any(
            e.get("valid") is False for key, e in self.state["episodes"].items() if key != job_id
        ):
            raise AdmissionClosed("Resolve the recorded invalid attempt before new admissions")
        first_lane_start = not any(
            e["lane"] == job["lane"] for e in self.state["episodes"].values()
        )
        self.guard()
        required_seconds = 3600 if first_lane_start else 2700
        if timestamp(self.rental["allocation_deadline_utc"]) - time.time() < required_seconds:
            raise AdmissionClosed("Insufficient 40/25-minute episode plus preservation reservation")
        if attempt > 1:
            self.processes.stop(self.state["server_generations"][str(job["lane"])])
            self.server(job["lane"], f"repair-{job_id}-{attempt}")
        self.guard()
        usage = self.sync_counters()["usage"]
        pending_controls = pending_warmups = 0
        for entry in self.state["episodes"].values():
            if entry.get("valid") is not None:
                continue
            attempted = warmups = 0
            for path, cache in self.state.get("source_counters", {}).items():
                if Path(path).is_relative_to(Path(entry["directory"])):
                    attempted += cache["counts"].get("before_action", 0)
                    warmups += cache["counts"].get("warmup_attempt", 0)
            pending_controls += max(0, 2000 - attempted)
            pending_warmups += max(0, 10 - warmups)
        if usage["policy_controls"] + pending_controls + 2000 > CAPS["policy_controls"]:
            raise AdmissionClosed(
                "Insufficient policy-control headroom for the full admitted panel"
            )
        if usage["warmup_controls"] + pending_warmups + 10 > CAPS["warmup_controls"]:
            raise AdmissionClosed("Insufficient warm-up-control headroom")
        details = {"repair_receipt": self.repair["receipt_sha256"]} if attempt > 1 else {}
        self.ledger.reserve_start(job_id, attempt, event_id=f"{job_id}:{attempt}:start", **details)
        lane = job["lane"]
        directory = self.output / "episodes" / f"{job_id}.attempt{attempt}"
        directory.mkdir(parents=True, exist_ok=False)
        settings = {
            "mode": mode or job["mode"],
            "job_id": job_id,
            "rng_seed": job["actor_rng_seed"],
            "actor_version": actor_version,
            "actor_url": self.actor_url,
            "purpose": "policy",
            "attempt_id": str(attempt),
        }
        atomic_json(directory / "job.json", {**job, "attempt": attempt})
        atomic_json(directory / "routing.json", settings)
        environment = self.gpu_environment(lane)
        environment.update(
            {
                "OMNIGIBSON_HEADLESS": "1",
                "NVIDIA_DRIVER_CAPABILITIES": "all",
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
                "PYTHONPATH": f"{self.root}/experiments:{self.root}/experiments/helpers:{self.root}/scripts/todo2",
                "SAFETY_DIAL_DSRL_CONFIG": str(directory / "routing.json"),
            }
        )
        environment["__EGL_VENDOR_LIBRARY_FILENAMES"] = (
            str(self.root / "scripts/todo1b/nvidia-egl-vendor.json")
            if Path("/opt/nvidia-drivers/lib64/libEGL_nvidia.so.0").exists()
            else "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
        )
        command = [
            self.behavior,
            "-m",
            "maniguard.eval.benchmark",
            "--config",
            "configs/eval/jar_transport_joint_legacy256.yaml",
            "--host",
            "127.0.0.1",
            "--port",
            str(8000 + lane),
            "--benchmark-root",
            "outputs/lerobot_datasets/maniguard-bench/jar_transport",
            "--scenes",
            f"task_{job['task']:04d}/base",
            "--max-scenes",
            "1",
            "--output-dir",
            str(directory),
            "--run-name",
            "eval",
            "--metrics",
            "success",
            "safety",
            "--headless",
            "--seed",
            str(job["seed"]),
            "--recording-factory",
            "dsrl06_recording:create_observer",
            "--recording-output-dir",
            str(directory / "recordings"),
        ]
        entry = {
            "job_id": job_id,
            "attempt": attempt,
            "lane": lane,
            "directory": str(directory),
            "process_key": f"episode.{job_id}.{attempt}",
            "started_utc": utc(),
            "timeout_seconds": 2400 if first_lane_start else 1500,
            "valid": None,
            "mode": settings["mode"],
            "actor_version": actor_version,
        }
        self.state["episodes"][job_id] = entry
        self.save()
        self.processes.launch(
            entry["process_key"],
            command,
            cwd=self.root / "third_party/ManiGuard",
            environment=environment,
            kind="episode",
        )
        self.event("episode_started", **entry)
        return entry

    def poll_episodes(self):
        for entry in list(self.state["episodes"].values()):
            if entry.get("valid") is not None:
                continue
            status = self.processes.status(entry["process_key"])
            if status["state"] == "running":
                if time.time() - timestamp(entry["started_utc"]) <= entry["timeout_seconds"]:
                    continue
                self.processes.stop(entry["process_key"])
                status = {"state": "stopped", "returncode": 124}
            if status["state"] == "unresolved":
                raise PendingOperation(f"Episode {entry['job_id']} lost its process receipt")
            try:
                if status.get("returncode") != 0:
                    raise ValueError("Evaluator did not exit cleanly")
                built = build_episode(entry["directory"], Path(entry["directory"]) / "learner.npz")
                atomic_json(Path(entry["directory"]) / "validation.json", built["validation"])
                atomic_json(Path(entry["directory"]) / "descriptor.json", built["descriptor"])
                entry.update(
                    valid=True,
                    finished_utc=status.get("finished_utc", utc()),
                    validated_utc=utc(),
                    validation=str(Path(entry["directory"]) / "validation.json"),
                )
            except (OSError, ValueError, KeyError) as error:
                entry.update(
                    valid=False,
                    finished_utc=status.get("finished_utc", utc()),
                    error_type=type(error).__name__,
                    error=str(error),
                )
            self.ledger.finish_start(
                entry["job_id"],
                entry["attempt"],
                valid=entry["valid"],
                event_id=f"{entry['job_id']}:{entry['attempt']}:finish",
            )
            self.save()
            self.event("episode_finished", **entry)

    def wait_episodes(self, job_ids):
        while any(self.state["episodes"][job_id].get("valid") is None for job_id in job_ids):
            self.tick()
        invalid = [job_id for job_id in job_ids if not self.state["episodes"][job_id]["valid"]]
        if invalid:
            raise AdmissionClosed(f"Invalid attempts need diagnosis and repair: {invalid}")

    def first_response(self, job_id):
        if not self.query_ledger.exists():
            return False
        with sqlite3.connect(f"file:{self.query_ledger}?mode=ro", uri=True) as db:
            return (
                db.execute(
                    "SELECT COUNT(*) FROM queries WHERE job_id=? AND status='returned'", (job_id,)
                ).fetchone()[0]
                > 0
            )

    def episode(self, job_id):
        return build_episode(self.state["episodes"][job_id]["directory"])

    def profile(self):
        snapshots = self.episode("E3")["snapshots"]
        path = self.output / "engineering-inputs.npz"
        if not path.exists():
            with path.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    pixels=np.stack([q["pixels"] for q in snapshots]),
                    states=np.stack([q["state"] for q in snapshots]),
                )
        self.begin_http(
            "engineering-profile",
            "/profile",
            {"path": str(path), "sha256": sha256(path), "updates": 128, "parity": True},
        )

    def actor_server(self, key, checkpoint, port):
        self.processes.launch(
            key,
            [
                self.learner_python,
                "-u",
                str(self.root / "experiments/scripts/dsrl06_actor_service.py"),
                "--checkpoint",
                checkpoint,
                "--work-dir",
                str(self.output / key),
                "--source",
                str(self.root / "third_party/dsrl_pi0"),
                "--port",
                str(port),
            ],
            cwd=self.root,
            environment={
                "CUDA_VISIBLE_DEVICES": "",
                "JAX_PLATFORMS": "cpu",
                "PYTHONUNBUFFERED": "1",
            },
            kind="actor",
        )
        while True:
            if self.processes.status(key)["state"] != "running":
                raise PendingOperation("Published CPU actor did not become available")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/status", timeout=2
                ) as response:
                    return json.load(response)
            except (OSError, ValueError):
                self.tick()

    def cpu_actor_parity(self):
        completed = self.output / "engineering-cpu-parity.json"
        if completed.exists():
            return json.loads(completed.read_text())
        if self.plan.get("actor_evaluation_mode") != "cpu_published_actor":
            raise AdmissionClosed("Stage A must freeze CPU published-actor evaluation")
        parity = self.plan["actor_cpu_parity"]
        if parity != {"atol": 1e-5, "rtol": 1e-5, "max_observations": 8}:
            raise ValueError("CPU actor parity differs from the reviewed engineering check")
        export = self.http("export-engineering", "/export-actor", {"name": "engineering"})
        self.actor_server("engineering-cpu-actor", export["path"], 9010)
        output = self.output / "engineering-cpu-parity.json"
        spec = {
            **parity,
            "inputs": str(self.output / "engineering-inputs.npz"),
            "rng_seed": self.plan["jobs"]["engineering_jobs"][3]["actor_rng_seed"],
            "actor_version": export["actor_version"],
            "output": str(output),
        }
        path = self.output / "cpu-parity-request.json"
        if path.exists() and json.loads(path.read_text()) != spec:
            raise ValueError("Resumed actor parity inputs changed")
        atomic_json(path, spec)
        self.processes.launch(
            "engineering-cpu-parity",
            [self.behavior, str(Path(__file__).resolve()), "--actor-parity-worker", str(path)],
            cwd=self.root,
            kind="actor_probe",
        )
        self.wait("engineering-cpu-parity")
        self.processes.stop("engineering-cpu-actor")
        return json.loads(output.read_text())

    def authority(self):
        episodes = {
            job_id: {"task": job["task"], "queries": self.episode(job_id)["snapshots"]}
            for job_id, job in ((j["job_id"], j) for j in self.plan["jobs"]["engineering_jobs"][:4])
        }
        selected = select_probe_snapshots(episodes)
        panel = []
        for snapshot in selected:
            descriptor = {k: v for k, v in snapshot.items() if not isinstance(v, np.ndarray)}
            descriptor["snapshot_id"] = f"{snapshot['episode_id']}:q{snapshot['query_id']}"
            with np.load(snapshot["record_path"], allow_pickle=False) as arrays:
                descriptor["input_sha256"] = {
                    key: hashlib.sha256(arrays[key].tobytes()).hexdigest()
                    for key in (
                        "observation_image_left",
                        "observation_wrist_image",
                        "observation_state",
                        "prompt",
                        "noise",
                        "actions",
                    )
                }
            panel.append(descriptor)
        path = self.output / "authority-panel.json"
        frozen = {"schema": "dsrl06-panel-v1", "snapshots": panel}
        if path.exists() and json.loads(path.read_text()) != frozen:
            raise ValueError("Resumed authority panel changed")
        atomic_json(path, frozen)
        result = self.output / "authority-result.json"
        command = [
            self.openpi,
            str(self.root / "experiments/scripts/probe_dsrl06.py"),
            "authority",
            "--port",
            "8000",
            "--query-ledger",
            str(self.query_ledger),
            "--record-root",
            str(self.query_root),
            "--output",
            str(result),
            "--panel",
            str(path),
            "--spec",
            str(self.root / "progress/receipts/06-dsrl-decoder-probe.json"),
            "--candidates",
            str(
                self.root
                / json.loads(
                    (self.root / "progress/receipts/06-dsrl-decoder-probe.json").read_text()
                )["candidate_npy_path"]
            ),
        ]
        self.processes.launch("authority", command, cwd=self.root, kind="probe")
        self.wait("authority", allow_failure=True)
        measured = json.loads(result.read_text())
        if measured["status"] == "pending":
            raise PendingOperation("Authority probe is pending; reconcile its saved packet state")
        return measured

    def engineering(self):
        jobs = {j["job_id"]: j for j in self.plan["jobs"]["engineering_jobs"]}
        summary_path = self.output / "stage-a-summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text())
        if not self.state["episodes"].get("E3", {}).get("valid"):
            self.server(3)
            self.start_episode(jobs["E3"])
            while not self.first_response("E3"):
                if self.state["episodes"]["E3"].get("valid") is not None:
                    raise AdmissionClosed("E3 ended before its first returned policy response")
                self.tick()
        for lane in range(3):
            self.server(lane)
            self.start_episode(jobs[f"E{lane}"])
        self.wait_episodes(["E3"])
        self.processes.stop(self.state["server_generations"]["3"])
        while not all(self.first_response(job_id) for job_id in ("E0", "E1", "E2")):
            if any(
                self.state["episodes"][job_id].get("valid") is False
                for job_id in ("E0", "E1", "E2")
            ):
                raise AdmissionClosed("A cold rollout failed before learner compilation")
            self.tick()
        self.learner("engineering")
        if "engineering_overlap" not in self.state:
            self.poll_episodes()
            self.state["engineering_overlap"] = [
                j for j in ("E0", "E1", "E2") if self.state["episodes"][j]["valid"] is None
            ]
            self.state["profile_start_utc"] = utc()
            self.save()
        overlap = self.state["engineering_overlap"]
        self.profile()
        self.wait_episodes(["E0", "E1", "E2"])
        profile = self.finish_http("engineering-profile")
        atomic_json(self.output / "engineering-profile.json", profile)
        engineering_checkpoint = self.checkpoint("engineering-profile")
        cpu_parity = self.cpu_actor_parity()
        try:
            authority = self.authority()
        except ValueError as error:
            authority = {"gate": {"valid": False, "admit": False, "reason": str(error)}}
        admitted = bool(authority.get("gate", {}).get("valid") and authority["gate"].get("admit"))
        if admitted:
            self.start_episode(jobs["E4"], actor_version=0)
            self.start_episode(jobs["E5"], actor_version=0)
            self.wait_episodes(["E4", "E5"])
        actual = self.sync_counters()
        durations = [
            timestamp(e["finished_utc"]) - timestamp(e["started_utc"])
            for key, e in self.state["episodes"].items()
            if key in ("E4", "E5") and e["valid"]
        ]
        cached_seconds = max(durations) if durations else None
        update_seconds = profile.get("median_update_seconds")
        projected = (
            20 * cached_seconds
            + 90000 * update_seconds
            + 3000 * cpu_parity["median_cpu_seconds"]
            + 1200
            + 300
            if cached_seconds is not None and update_seconds is not None
            else None
        )
        remaining = timestamp(self.rental["allocation_deadline_utc"]) - time.time()
        profile_spec = self.processes.spec("http.engineering-profile")
        profile_exit = self.processes.status("http.engineering-profile")
        profile_begin = timestamp(profile_spec["created_utc"])
        profile_end = timestamp(profile_exit["finished_utc"])
        rollout_end = min(
            timestamp(self.state["episodes"][job_id]["finished_utc"])
            for job_id in ("E0", "E1", "E2")
        )
        overlap_seconds = max(0, min(profile_end, rollout_end) - profile_begin)
        topology = len(overlap) == 3 and profile_end <= rollout_end
        source_difference = profile.get("source_parity", {}).get("maximum_state_difference")
        source_parity_valid = (
            isinstance(source_difference, (int, float))
            and np.isfinite(source_difference)
            and 0 <= source_difference <= 1e-6
        )
        admit = (
            admitted
            and topology
            and source_parity_valid
            and cpu_parity.get("valid") is True
            and projected is not None
            and projected <= remaining
        )
        summary = {
            "schema": "dsrl06-stage-a-summary-v1",
            "created_utc": utc(),
            "plan_sha256": sha256(self.plan_path),
            "instance_id": self.rental["instance_id"],
            "admit": admit,
            "authority_admit": admitted,
            "authority": authority.get("gate"),
            "topology_three_rollouts": topology,
            "three_rollout_profile_overlap_seconds": overlap_seconds,
            "entire_profile_overlapped_three_rollouts": profile_end <= rollout_end,
            "overlapping_jobs_at_profile": overlap,
            "profile": profile,
            "engineering_checkpoint": engineering_checkpoint,
            "source_parity_valid": bool(source_parity_valid),
            "cpu_actor_parity": cpu_parity,
            "resources": actual,
            "cached_episode_seconds": cached_seconds,
            "projected_stage_b_seconds_including_preservation": projected,
            "remaining_allocation_seconds": remaining,
            "reason": "ready_for_review"
            if admit
            else "authority_absent_or_invalid"
            if not admitted
            else "topology_or_budget_not_established",
        }
        atomic_json(summary_path, summary)
        return summary

    def checkpoint(self, name):
        result = self.http(f"save-{name}", "/save", {"name": name})
        self.state["last_checkpoint"] = result["path"]
        self.save()
        return result

    def training_phase(self):
        for lane in range(3):
            self.server(lane)
        status = self.learner("training")
        jobs = self.plan["jobs"]["training_jobs"]
        for round_index in range(8):
            round_id = f"round{round_index:02d}"
            if self.state.get("completed_rounds", {}).get(round_id):
                continue
            status = self.get_status()
            rounds = self.state.setdefault("rounds", {})
            if round_id not in rounds:
                rounds[round_id] = {
                    "warmup": not status["warmup_complete"],
                    "actor_version": status["actor_version"],
                }
                self.save()
            round_state = rounds[round_id]
            warmup = round_state["warmup"]
            if warmup and round_index >= 4:
                raise AdmissionClosed("Insufficient warm-up transitions after twelve episodes")
            batch = jobs[round_index * 3 : round_index * 3 + 3]
            for job in batch:
                self.start_episode(
                    job,
                    mode="gaussian" if warmup else "actor",
                    actor_version=round_state["actor_version"],
                )
            self.wait_episodes([job["job_id"] for job in batch])
            descriptors = []
            for job in batch:
                path = Path(self.state["episodes"][job["job_id"]]["directory"]) / "descriptor.json"
                descriptor = json.loads(path.read_text())
                if descriptor["actor_version"] != round_state["actor_version"]:
                    raise ValueError("Validated replay differs from the frozen round actor version")
                descriptors.append(descriptor)
            appended = self.http(
                f"append-{round_id}",
                "/append",
                {"round_id": round_id, "warmup": warmup, "episodes": descriptors},
            )
            if not warmup:
                self.http(
                    f"train-{round_id}",
                    "/train",
                    {"round_id": round_id, "expected_new_transitions": appended["new_transitions"]},
                )
                self.http(f"publish-{round_id}", "/publish", {"round_id": round_id})
            episodes = (round_index + 1) * 3
            self.checkpoint(f"episode{episodes}" if episodes in (12, 24) else round_id)
            self.state.setdefault("completed_rounds", {})[round_id] = True
            self.save()
            self.sync_counters()
        final_status = self.get_status()
        final_checkpoint = self.state["last_checkpoint"]
        exported = self.http("export-final24", "/export-actor", {"name": "final24"})
        self.state["training_complete"] = {
            "status": final_status,
            "checkpoint": final_checkpoint,
            "export": exported,
        }
        self.save()

    def pilot(self):
        summary_path = self.output / "stage-a-summary.json"
        if sha256(summary_path) != self.plan["stage_a_summary_sha256"]:
            raise ValueError("Stage-B freeze identifies a different engineering receipt")
        if not json.loads(summary_path.read_text())["admit"]:
            raise AdmissionClosed("Engineering did not admit a development pilot")
        evaluation_mode = self.plan.get("actor_evaluation_mode")
        if evaluation_mode != "cpu_published_actor":
            raise AdmissionClosed("Freeze an actor-only evaluation topology before pilot admission")
        if "training_complete" not in self.state:
            self.training_phase()
        completed = self.state["training_complete"]
        final_status, final_checkpoint, exported = (
            completed["status"],
            completed["checkpoint"],
            completed["export"],
        )
        for lane in range(3):
            self.server(lane)
        # The actor-only CPU service has a separate frozen implementation and owns no updates.
        self.http("shutdown-training", "/shutdown", {})
        self.wait(self.state.get("learner_process", "learner"))
        self.actor_server("evaluation-actor", exported["path"], 9009)
        while True:
            try:
                if self.get_status()["actor_version"] == final_status["actor_version"]:
                    break
            except (OSError, ValueError):
                pass
            self.tick()
        self.server(3, "evaluation")
        evaluations = self.plan["jobs"]["evaluation_jobs"]
        for block in range(12):
            batch = evaluations[block * 4 : block * 4 + 4]
            for job in batch:
                self.start_episode(job, actor_version=final_status["actor_version"])
            self.wait_episodes([job["job_id"] for job in batch])
        result = {
            "schema": "dsrl06-pilot-summary-v1",
            "finished_utc": utc(),
            "plan_sha256": sha256(self.plan_path),
            "checkpoint": final_checkpoint,
            "actor_version": final_status["actor_version"],
            "resources": self.sync_counters(),
            "episodes": self.state["episodes"],
        }
        atomic_json(self.output / "pilot-summary.json", result)
        return result

    def close(self, *, stop=False):
        if stop:
            for path in self.processes.directory.glob("*.json"):
                if any(
                    path.name.endswith(suffix)
                    for suffix in (".pid.json", ".child.json", ".exit.json")
                ):
                    continue
                self.processes.stop(path.stem)
            pids = self.processes.actual_pids()
            if any(process_identity(pid) for pid in pids):
                raise PendingOperation("Managed worker identities have not all stopped")
            atomic_json(
                self.output / "quiescence.json",
                {
                    "instance_id": self.rental["instance_id"],
                    "workers_stopped": True,
                    "worker_pids": pids,
                    "created_utc": utc(),
                },
            )
        self.sync_counters()
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--managed-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--http-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--actor-parity-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("engineering", "pilot"))
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "progress/receipts/06-dsrl-stage-a.json"
    )
    parser.add_argument(
        "--rental", type=Path, default=ROOT / "progress/rental-dsrl06-20261025.json"
    )
    parser.add_argument("--repair-receipt", type=Path)
    args = parser.parse_args()
    if args.managed_worker:
        return worker(args.managed_worker)
    if args.http_worker:
        return http_worker(args.http_worker)
    if args.actor_parity_worker:
        return actor_parity_worker(args.actor_parity_worker)
    if not args.stage:
        parser.error("--stage is required")
    coordinator = Coordinator(
        ROOT, args.plan, args.rental, args.stage, repair_receipt=args.repair_receipt
    )
    stop = False
    try:
        result = coordinator.engineering() if args.stage == "engineering" else coordinator.pilot()
        print(
            json.dumps(
                {key: result[key] for key in ("schema", "admit", "reason") if key in result}
            ),
            flush=True,
        )
        stop = args.stage == "pilot" or not result.get("admit", False)
        return 0
    except PendingOperation as error:
        coordinator.event("pending", reason=str(error))
        print(json.dumps({"status": "pending", "reason": str(error)}), flush=True)
        return 2
    except BaseException as error:
        coordinator.event("queue_halted", error_type=type(error).__name__, reason=str(error))
        stop = isinstance(error, AdmissionClosed)
        raise
    finally:
        coordinator.close(stop=stop)


if __name__ == "__main__":
    raise SystemExit(main())
