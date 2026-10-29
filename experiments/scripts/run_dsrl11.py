"""Run the frozen task-11 queue on one prepared four-GPU host.

Stages: ``numerical`` (A1 critic grid on the saved task-7 replay, A2 decoder checks),
``engineering`` (E0-E7, live learner profile and parity, CPU/GPU actor agreement),
``bank`` (B00-B11 and the strict-safe gate), ``admission`` (measured stage-C freeze),
``pilot`` (four-condition collection with learner updates, then the forty frozen
evaluations). ``all`` runs them in order and stops at the first unmet gate.

Run with the learner interpreter, which carries NumPy; simulator, VLA and learner
processes use their own pinned interpreters. Process and HTTP intentions are durable:
losing an observation never authorizes a second simulator start or a duplicate
learner mutation. The independent financial watchdogs own provider shutdown.
"""

from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sqlite3
import sys
import time
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))
import run_dsrl06
from helpers.dsrl07_protocol import decoder_comparison
from helpers.dsrl11_protocol import (
    BANK_SUCCESS_MINIMUM,
    CAPS,
    COLLECTION_LANES,
    CONDITIONS,
    DIAGNOSTIC_SEEDS,
    LEARNER_GPU,
    PHASES,
    Dsrl11Ledger,
    condition_order,
    frozen_jobs,
    select_learner_setting,
)
from helpers.dsrl11_infra import LIMITS
from helpers.dsrl11_validation import build_residual_episode
from run_dsrl06 import (
    AdmissionClosed,
    PendingOperation,
    Processes,
    atomic_json,
    process_identity,
    sha256,
    timestamp,
    utc,
)

OUTPUT = "outputs/dsrl11"
LEARNER_PORT = 9011
ENGINEERING_ACTOR_PORT = 9013
EVALUATION_ACTOR_PORTS = {condition: 9021 + i for i, condition in enumerate(CONDITIONS)}
A1_CELLS = [(utd, reduction) for utd in (20, 5) for reduction in ("min", "mean")]
A1_GPU = {("min", 11061): 0, ("min", 11062): 1, ("mean", 11061): 2, ("mean", 11062): 3}
MONEY = dict(LIMITS)


def verify_plan(root, path, stage):
    plan = json.loads(Path(path).read_text())
    if plan.get("schema") != "dsrl11-freeze-v1":
        raise ValueError("A task-11 source freeze is required")
    if plan.get("authorization", {}).get("user_launch") is not True:
        raise ValueError("Task-11 launch authorization missing")
    for name, digest in {**plan["source_sha256"], **plan["input_sha256"]}.items():
        if sha256(Path(root) / name) != digest:
            raise ValueError(f"Frozen task-11 file changed: {name}")
    if plan["jobs"] != frozen_jobs() or plan["caps"] != dict(CAPS):
        raise ValueError("Task-11 frozen jobs or caps differ")
    if plan["money"] != MONEY:
        raise ValueError("The task-11 financial envelope changed")
    if plan["condition_order"] != condition_order():
        raise ValueError("The frozen condition permutation changed")
    return plan


class Coordinator(run_dsrl06.Coordinator):
    output_relative = OUTPUT
    verify_plan = staticmethod(verify_plan)

    def __init__(self, root, plan_path, rental_path, stage, *, repair_receipt=None):
        self.root, self.stage = Path(root).resolve(), stage
        self.plan_path, self.rental_path = Path(plan_path), Path(rental_path)
        self.plan = self.verify_plan(self.root, self.plan_path, stage)
        self.rental = json.loads(self.rental_path.read_text())
        if str(self.rental["instance_id"]) != os.environ.get("CONTAINER_ID"):
            raise ValueError("Queue is not running on the frozen rental")
        self.output = self.root / self.output_relative
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / "queue.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state_path = self.output / "queue-state.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.exists()
            else {
                "schema": "dsrl11-queue-v1",
                "instance_id": self.rental["instance_id"],
                "plan_sha256": sha256(self.plan_path),
                "episodes": {},
                "http": {},
                "server_generations": {},
                "published": {},
                "exports": {},
                "learner_ops": {},
            }
        )
        if self.state["instance_id"] != self.rental["instance_id"]:
            raise ValueError("Contemporary comparator jobs cannot span physical hosts")
        if self.state["plan_sha256"] != sha256(self.plan_path):
            raise ValueError("A resumed queue cannot change its source freeze")
        self.repair = json.loads(Path(repair_receipt).read_text()) if repair_receipt else None
        if self.repair is not None:
            self.repair["receipt_sha256"] = sha256(repair_receipt)
        self.ledger = Dsrl11Ledger(self.output / "resources.jsonl", self.plan["jobs"])
        self.python = sys.executable
        self.behavior = "/opt/miniforge3/envs/behavior/bin/python"
        self.openpi = str(self.root / "third_party/openpi/.venv/bin/python")
        self.learner_python = str(self.root / "third_party/dsrl-venv/bin/python")
        self.processes = Processes(
            self.output / "processes", self.python, Path(run_dsrl06.__file__).resolve()
        )
        self.query_ledger = self.output / "queries.sqlite"
        self.query_root = self.output / "queries"
        self.actor_url = f"http://127.0.0.1:{LEARNER_PORT}"
        self.jobs = {job["job_id"]: job for group in self.plan["jobs"].values() for job in group}
        self.save()
        self.sync_counters()

    # -- deadlines and accounting -------------------------------------------
    def deadline(self):
        return timestamp(self.rental["allocation_deadline_utc"]) - 1200

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
                    state["counts"][label] = state["counts"].get(label, 0) + 1
                    state["offset"] = stream.tell()
            return state["counts"]

        for path in (self.output / "episodes").rglob("*.journal.jsonl"):
            observed = events(path)
            counts["policy_controls"] += observed.get("before_action", 0)
            counts["warmup_controls"] += observed.get("warmup_attempt", 0)
        model_only = 0
        if self.query_ledger.exists():
            with sqlite3.connect(f"file:{self.query_ledger}?mode=ro", uri=True) as db:
                counts["model_queries"] = db.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
                model_only = db.execute(
                    "SELECT COUNT(*) FROM queries WHERE purpose IN ('startup','probe')"
                ).fetchone()[0]
        # Real-backend workstation preflight updates are engineering charges too.
        updates = {
            "diagnostic": 0,
            "engineering": int(self.plan.get("prior_engineering_updates", 0)),
            "learning": 0,
        }
        for path in sorted((self.output / "a1").glob("*/diagnostic-ledger.jsonl")):
            updates["diagnostic"] += events(path).get("update_reserved:diagnostic", 0)
        for name in ("learner-engineering", "learner-learning"):
            path = self.output / name / "learner-ledger.jsonl"
            if path.exists():
                observed = events(path)
                updates["engineering"] += observed.get("update_reserved:engineering", 0)
                updates["learning"] += observed.get("update_reserved:learning", 0)
        usage = self.ledger.summary()["usage"]
        for kind, actual in counts.items():
            if actual < usage[kind]:
                raise ValueError(f"Source accounting decreased for {kind}")
        recorded_model_only = usage["model_only_queries"]
        if model_only < recorded_model_only:
            raise ValueError("Model-only query accounting decreased")
        if model_only > recorded_model_only:
            self.ledger.charge(
                "model_queries",
                model_only - recorded_model_only,
                model_only=True,
                event_id=f"observed:model_only:{model_only}",
            )
        usage = self.ledger.summary()["usage"]
        for kind, actual in counts.items():
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
        actor_calls = sum(
            events(path).get("actor_request", 0) for path in self.output.rglob("*-ledger.jsonl")
        )
        self.save()
        return {**self.ledger.summary(), "actor_calls": actor_calls}

    def require_headroom(self, kind, amount, *, phase=None):
        usage = self.sync_counters()["usage"]
        key = f"{phase}_updates" if phase else kind
        if usage[key] + amount > CAPS[key] or usage[kind] + amount > CAPS[kind]:
            raise AdmissionClosed(f"Frozen {key} ceiling cannot cover {amount} more")

    # -- processes ------------------------------------------------------------
    def server(self, lane, generation=None):
        return self.servers([lane], generation)[lane]

    def servers(self, lanes, generation=None):
        """Launch every requested policy server first, then wait and probe each."""
        keys = {lane: self.launch_server(lane, generation) for lane in lanes}
        for lane, key in keys.items():
            self.ready_server(lane, key)
        return keys

    def launch_server(self, lane, generation=None):
        if generation is None:
            current = self.state["server_generations"].get(str(lane))
            generation = current.split(".", 1)[1] if current else "collection"
        key = f"server{lane}.{generation}"
        environment = self.gpu_environment(lane)
        environment["PYTHONPATH"] = f"{self.root}/third_party/ManiGuard:{self.root}/experiments"
        command = [
            self.openpi,
            "-u",
            str(self.root / "experiments/scripts/serve_dsrl11.py"),
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
        return key

    def ready_server(self, lane, key):
        generation = key.split(".", 1)[1]
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
        self.processes.launch(
            startup,
            [
                self.openpi,
                str(self.root / "experiments/scripts/probe_dsrl11.py"),
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
            ],
            cwd=self.root,
            kind="probe",
        )
        self.wait(startup)
        if not json.loads(receipt.read_text()).get("valid"):
            raise ValueError("Server startup probe failed")
        return key

    def server_running(self, lane):
        key = self.state["server_generations"].get(str(lane))
        return bool(key) and self.processes.status(key)["state"] == "running"

    def stop_server(self, lane):
        key = self.state["server_generations"].get(str(lane))
        if key and self.processes.status(key)["state"] == "running":
            self.processes.stop(key)

    def service_status(self, port):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=5) as response:
            return json.load(response)

    def wait_service(self, key, port):
        while True:
            if self.processes.status(key)["state"] != "running":
                raise PendingOperation(f"Service {key} is not running")
            try:
                return self.service_status(port)
            except (OSError, ValueError):
                self.tick()

    def learner_service(self, phase, *, resume=None):
        key = self.state.get(f"learner_process_{phase}", f"learner-{phase}")
        if not self.processes.path(key).exists():
            command = [
                self.learner_python,
                "-u",
                str(self.root / "experiments/scripts/dsrl11_learner_service.py"),
                "--work-dir",
                str(self.output / f"learner-{phase}"),
                "--data-root",
                str(self.output),
                "--selection",
                str(self.output / "stage-a/learner-selection.json"),
                "--port",
                str(LEARNER_PORT),
            ]
            command += ["--resume", resume] if resume else ["--phase", phase]
            self.processes.launch(
                key,
                command,
                cwd=self.root,
                environment=self.gpu_environment(LEARNER_GPU),
                kind="learner",
            )
            self.state[f"learner_process_{phase}"] = key
            self.save()
        status = self.wait_service(key, LEARNER_PORT)
        if status["phase"] != phase:
            raise PendingOperation("The learner endpoint serves a different phase")
        return status

    def actor_service(self, key, exports, port):
        command = [
            self.learner_python,
            "-u",
            str(self.root / "experiments/scripts/dsrl11_learner_service.py"),
            "--work-dir",
            str(self.output / key),
            "--data-root",
            str(self.output),
            "--port",
            str(port),
        ]
        for condition, path in sorted(exports.items()):
            command += ["--actor-export", f"{condition}={path}"]
        self.processes.launch(
            key,
            command,
            cwd=self.root,
            environment={
                "CUDA_VISIBLE_DEVICES": "",
                "JAX_PLATFORMS": "cpu",
                "PYTHONUNBUFFERED": "1",
            },
            kind="actor",
        )
        return self.wait_service(key, port)

    # -- HTTP to the learner ----------------------------------------------------
    def begin_http(self, key, route, body, port=LEARNER_PORT):
        directory = self.output / "http"
        request_path = directory / f"{key}.request.json"
        output = directory / f"{key}.response.json"
        spec = {"url": f"http://127.0.0.1:{port}{route}", "body": body, "output": str(output)}
        if request_path.exists() and json.loads(request_path.read_text()) != spec:
            raise ValueError("A resumed learner operation cannot change its request")
        if not request_path.exists():
            atomic_json(request_path, spec)
        self.processes.launch(
            f"http.{key}",
            [
                self.python,
                str(Path(run_dsrl06.__file__).resolve()),
                "--http-worker",
                str(request_path),
            ],
            cwd=self.root,
            kind="http",
        )
        return output

    def poll_http(self, key):
        """Non-blocking: the response, None while running, or a reconciliation stop."""
        output = self.output / "http" / f"{key}.response.json"
        if (output.parent / f"{key}.error.json").exists():
            raise PendingOperation(f"HTTP operation {key} has a preserved failure receipt")
        if output.exists():
            return json.loads(output.read_text())
        status = self.processes.status(f"http.{key}")
        if status["state"] == "running":
            return None
        if output.exists():
            return json.loads(output.read_text())
        recovered = self.recover_http(key)
        if recovered is not None:
            atomic_json(output, recovered)
            return recovered
        raise PendingOperation(f"HTTP operation {key} has no response ({status.get('returncode')})")

    def finish_http(self, key):
        while True:
            result = self.poll_http(key)
            if result is not None:
                return result
            self.tick()

    def recover_http(self, key):
        request_path = self.output / "http" / f"{key}.request.json"
        if not request_path.exists() or (request_path.parent / f"{key}.error.json").exists():
            return None
        request = json.loads(request_path.read_text())
        route, body = request["url"].rsplit("/", 1)[1], request["body"]
        fingerprint = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        rows = []
        for name in ("learner-engineering", "learner-learning"):
            ledger = self.output / name / "learner-ledger.jsonl"
            if ledger.exists():
                rows += [json.loads(line) for line in ledger.read_text().splitlines()]

        def matching(event):
            return [
                row
                for row in rows
                if row["event"] == event
                and row.get("condition") == body.get("condition")
                and row.get("round_id") == body.get("round_id")
            ]

        if route == "append" and matching("round_appended"):
            row = matching("round_appended")[-1]
            if row["jobs"] != [d["job_id"] for d in body["episodes"]]:
                raise ValueError("Recovered append differs from the durable request")
            return {
                "condition": row["condition"],
                "round_id": row["round_id"],
                "phase": row["phase"],
                "new_transitions": row["transitions"],
                "update_credit": 0 if row["phase"] == "warmup" else row["transitions"],
            }
        if route == "train" and matching("train_completed"):
            return matching("train_completed")[-1]
        if route == "publish" and matching("actor_published"):
            row = matching("actor_published")[-1]
            return {k: row[k] for k in ("condition", "round_id", "actor_version")}
        if route == "load-bank":
            loaded = [
                r for r in rows if r["event"] == "bank_loaded" and r["fingerprint"] == fingerprint
            ]
            if loaded:
                return {
                    "bank_episodes": [d["job_id"] for d in body["episodes"]],
                    "bank_transitions": loaded[-1]["transitions"],
                }
        if route == "profile":
            done = [
                r
                for r in rows
                if r["event"] == "profile_completed" and r.get("request_sha256") == fingerprint
            ]
            if done:
                return done[-1]
        if route in ("save", "export-actor"):
            for name in ("learner-engineering", "learner-learning"):
                folder = "checkpoints" if route == "save" else "actor_exports"
                path = self.output / name / folder / body["name"]
                manifest_path = path / "manifest.json"
                if manifest_path.exists():
                    for file, digest in json.loads(manifest_path.read_text()).items():
                        if Path(file).name != file or sha256(path / file) != digest:
                            raise ValueError("Recovered learner snapshot hash differs")
                    result = {"path": str(path), "manifest_sha256": sha256(manifest_path)}
                    if route == "export-actor":
                        state = json.loads((path / "state.json").read_text())
                        result.update(
                            condition=state["condition"],
                            actor_version=state["actor_version"],
                            actor_evaluation_mode="cpu_published_actor",
                        )
                    return result
        return None

    def http(self, key, route, body, port=LEARNER_PORT):
        self.begin_http(key, route, body, port)
        return self.finish_http(key)

    # -- episodes ---------------------------------------------------------------
    def routing(self, job, attempt):
        settings = {
            "mode": "residual",
            "job_id": job["job_id"],
            "attempt_id": str(attempt),
            "purpose": "policy",
            "residual_source": job["residual_source"],
            "epsilon_seed": job["epsilon_seed"],
        }
        if job["residual_source"] == "uniform":
            settings["uniform_seed"] = job["uniform_seed"]
        if job["residual_source"] == "actor":
            port = (
                EVALUATION_ACTOR_PORTS[job["condition"]]
                if job["stage"] == "evaluation"
                else LEARNER_PORT
            )
            settings.update(
                actor_url=f"http://127.0.0.1:{port}",
                actor_condition=job["condition"],
                actor_version=job["actor_version"],
                rng_seed=job["actor_rng_seed"],
            )
        return settings

    def lane_busy(self, lane):
        return any(
            entry["lane"] == lane and entry.get("valid") is None
            for entry in self.state["episodes"].values()
        )

    def start_episode(self, job, *, mode=None, actor_version=-1):
        job_id = job["job_id"]
        previous = self.state["episodes"].get(job_id)
        if previous and previous.get("valid") in (True, None):
            return previous
        attempt = 1 if previous is None else previous["attempt"] + 1
        if attempt > 1 and (
            not self.repair
            or self.repair.get("job_id") != job_id
            or self.repair.get("attempt") != attempt
            or not self.repair.get("repair")
        ):
            raise AdmissionClosed("Invalid attempts require a recorded repair before replacement")
        if any(e.get("valid") is False for k, e in self.state["episodes"].items() if k != job_id):
            raise AdmissionClosed("Resolve the recorded invalid attempt before new admissions")
        lane = job["lane"]
        if self.lane_busy(lane):
            raise ValueError("A lane runs one simulator episode at a time")
        first_lane_start = not any(e["lane"] == lane for e in self.state["episodes"].values())
        self.guard()
        required = 3600 if first_lane_start else 2700
        if timestamp(self.rental["allocation_deadline_utc"]) - time.time() < required:
            raise AdmissionClosed("Insufficient 60/45-minute episode plus preservation reservation")
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
            raise AdmissionClosed("Insufficient policy-control headroom")
        if usage["warmup_controls"] + pending_warmups + 10 > CAPS["warmup_controls"]:
            raise AdmissionClosed("Insufficient warm-up-control headroom")
        if usage["starts"] + 1 > CAPS["starts"]:
            raise AdmissionClosed("Simulator start ceiling reached")
        details = {"repair_receipt": self.repair["receipt_sha256"]} if attempt > 1 else {}
        self.ledger.reserve_start(job_id, attempt, event_id=f"{job_id}:{attempt}:start", **details)
        directory = self.output / "episodes" / f"{job_id}.attempt{attempt}"
        directory.mkdir(parents=True, exist_ok=False)
        settings = self.routing(job, attempt)
        atomic_json(directory / "job.json", {**job, "attempt": attempt})
        atomic_json(directory / "routing.json", settings)
        environment = self.gpu_environment(lane)
        environment.update(
            {
                "OMNIGIBSON_HEADLESS": "1",
                "NVIDIA_DRIVER_CAPABILITIES": "all",
                "OMNI_KIT_ACCEPT_EULA": "YES",
                "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
                "PYTHONPATH": (
                    f"{self.root}/experiments:{self.root}/experiments/helpers:"
                    f"{self.root}/scripts/todo2"
                ),
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
            "dsrl11_recording:create_observer",
            "--recording-output-dir",
            str(directory / "recordings"),
        ]
        entry = {
            "job_id": job_id,
            "attempt": attempt,
            "lane": lane,
            "stage": job["stage"],
            "directory": str(directory),
            "process_key": f"episode.{job_id}.{attempt}",
            "started_utc": utc(),
            "timeout_seconds": 2400 if first_lane_start else 1500,
            "valid": None,
            "residual_source": job["residual_source"],
            "actor_version": job.get("actor_version", -1),
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
            directory = Path(entry["directory"])
            try:
                if status.get("returncode") != 0:
                    raise ValueError("Evaluator did not exit cleanly")
                built = build_residual_episode(
                    directory, directory / "learner.npz", job_schedule=self.plan["jobs"]
                )
                atomic_json(directory / "validation.json", built["validation"])
                atomic_json(directory / "descriptor.json", built["descriptor"])
                entry.update(
                    valid=True,
                    finished_utc=status.get("finished_utc", utc()),
                    validated_utc=utc(),
                    labels=built["validation"]["labels"],
                    returns=built["validation"]["returns"],
                )
            except (OSError, ValueError, KeyError) as error:
                entry.update(
                    valid=False,
                    finished_utc=status.get("finished_utc", utc()),
                    error_type=type(error).__name__,
                    error=str(error)[:2000],
                )
            self.ledger.finish_start(
                entry["job_id"],
                entry["attempt"],
                valid=entry["valid"],
                event_id=f"{entry['job_id']}:{entry['attempt']}:finish",
            )
            self.save()
            self.event("episode_finished", **entry)

    def valid(self, job_id):
        return self.state["episodes"].get(job_id, {}).get("valid") is True

    def descriptor(self, job_id):
        entry = self.state["episodes"][job_id]
        return json.loads((Path(entry["directory"]) / "descriptor.json").read_text())

    def duration(self, job_id):
        entry = self.state["episodes"][job_id]
        return timestamp(entry["finished_utc"]) - timestamp(entry["started_utc"])

    def schedule(self, sequences, ready, step=None):
        """Per-lane frozen order: a free lane starts its next job once it is ready."""
        while True:
            self.tick()
            if step is not None:
                step()
            pending = False
            for lane, jobs in sequences.items():
                remaining = [job for job in jobs if not self.valid(job["job_id"])]
                if not remaining:
                    continue
                pending = True
                job = remaining[0]
                entry = self.state["episodes"].get(job["job_id"])
                if entry and entry.get("valid") is False:
                    raise AdmissionClosed(f"Invalid attempt needs diagnosis: {job['job_id']}")
                if self.lane_busy(lane) or not ready(job):
                    continue
                self.start_episode(job)
            if not pending:
                return

    # -- stage A on the rental ----------------------------------------------------
    def a1(self):
        """Launch the approved grid: one (reduction, seed) pair per GPU, UTD 20 then 5."""
        results = {}
        for (reduction, seed), gpu in A1_GPU.items():
            for utd in (20, 5):
                key = f"a1.{reduction}.utd{utd}.seed{seed}"
                path = self.output / "a1" / f"{reduction}-utd{utd}-seed{seed}" / "result.json"
                results[(utd, reduction, seed)] = (key, path, gpu)
        return results

    def a1_step(self, cells):
        """Advance the per-GPU A1 queues without blocking; True once every cell exited."""
        done = True
        for (reduction, seed), gpu in A1_GPU.items():
            for utd in (20, 5):
                key, path, _ = cells[(utd, reduction, seed)]
                if not self.processes.path(key).exists():
                    launched = sum(
                        750 * 2 * u
                        for (u, _, _), (k, _, _) in cells.items()
                        if self.processes.path(k).exists()
                    )
                    if launched + 750 * 2 * utd > CAPS["diagnostic_updates"]:
                        raise AdmissionClosed("The diagnostic update ceiling cannot cover the grid")
                    self.processes.launch(
                        key,
                        [
                            self.learner_python,
                            "-u",
                            str(self.root / "experiments/scripts/diagnose_dsrl11.py"),
                            "critic-cell",
                            "--seed",
                            str(seed),
                            "--reduction",
                            reduction,
                            "--utd",
                            str(utd),
                            "--work-dir",
                            str(self.output / "a1"),
                            "--task7-root",
                            str(self.output / "inputs/task7"),
                        ],
                        cwd=self.root,
                        environment=self.gpu_environment(gpu),
                        kind="a1",
                    )
                    done = False
                    break
                state = self.processes.status(key)["state"]
                if state == "running":
                    done = False
                    break
                if state != "exited":
                    raise PendingOperation(f"Reconcile A1 cell {key}")
        return done

    def a2(self):
        manifest = self.output / "inputs/pool/manifest.json"
        for lane in range(4):
            key = f"decoder.lane{lane}"
            self.processes.launch(
                key,
                [
                    self.openpi,
                    str(self.root / "experiments/scripts/probe_dsrl11.py"),
                    "decoder",
                    "--port",
                    str(8000 + lane),
                    "--lane",
                    str(lane),
                    "--operation",
                    "initial",
                    "--inputs",
                    str(manifest),
                    "--query-ledger",
                    str(self.query_ledger),
                    "--record-root",
                    str(self.query_root),
                    "--output",
                    str(self.output / "stage-a" / f"decoder-lane{lane}.json"),
                ],
                cwd=self.root,
                kind="probe",
            )

    def numerical(self):
        destination = self.output / "stage-a/numerical-summary.json"
        if destination.exists():
            result = json.loads(destination.read_text())
            if result["admit"]:
                return result
            raise PendingOperation("Recorded numerical failure needs a diagnosis and repair")
        cells = self.a1()
        ready = self.output / "ready"
        while not (ready / "learner.json").exists():
            self.tick()
        while not (ready / "runtime.json").exists():
            self.a1_step(cells)
            self.tick()
        self.servers(range(4))
        self.a2()
        while not self.a1_step(cells) or any(
            self.processes.status(f"decoder.lane{lane}")["state"] == "running" for lane in range(4)
        ):
            self.tick()
        decoders = []
        for lane in range(4):
            path = self.output / "stage-a" / f"decoder-lane{lane}.json"
            decoders.append(json.loads(path.read_text()) if path.exists() else {"valid": False})
        cross = []
        for lane in range(1, 4):
            if not decoders[0].get("valid") or not decoders[lane].get("valid"):
                continue
            for base, other in zip(decoders[0]["cases"], decoders[lane]["cases"], strict=True):
                cross.append(
                    {
                        "lane": lane,
                        "snapshot_id": base["snapshot_id"],
                        **decoder_comparison(base["native_actions"], other["native_actions"]),
                    }
                )
        grid = {}
        for (utd, reduction, seed), (key, path, gpu) in cells.items():
            grid.setdefault((utd, reduction), {})[seed] = (
                json.loads(path.read_text())
                if path.exists()
                else {"admitted": False, "missing": True, "exit": self.processes.status(key)}
            )
        summary_cells = {
            key: {
                "admitted": all(
                    grid[key][seed].get("admitted") is True for seed in DIAGNOSTIC_SEEDS
                ),
                "seeds": {
                    str(seed): {
                        k: grid[key][seed].get(k)
                        for k in (
                            "admitted",
                            "gradient_updates",
                            "pause",
                            "nonfinite",
                            "final_policy",
                        )
                    }
                    for seed in DIAGNOSTIC_SEEDS
                },
            }
            for key in A1_CELLS
        }
        selection = select_learner_setting(summary_cells)
        a2_pass = all(
            d.get("valid")
            and d.get("same_process_native_repeat_pass")
            and d.get("zero_residual_pass")
            for d in decoders
        )
        a2_pass = (
            a2_pass and decoders[0].get("influence_present") and decoders[0].get("influence_finite")
        )
        result = {
            "schema": "dsrl11-numerical-summary-v1",
            "created_utc": utc(),
            "plan_sha256": sha256(self.plan_path),
            "a1_cells": {f"{u}-{r}": v for (u, r), v in summary_cells.items()},
            "a1_selection": selection,
            "a2": {
                "lanes": [
                    {
                        k: d.get(k)
                        for k in (
                            "valid",
                            "same_process_native_repeat_pass",
                            "zero_residual_pass",
                            "historical_draw_reproduced_all",
                            "influence_present",
                            "influence_finite",
                            "query_counts",
                        )
                    }
                    for d in decoders
                ],
                "pass": bool(a2_pass),
                "cross_process_descriptive": cross,
            },
            "resources": self.sync_counters(),
        }
        result["admit"] = bool(a2_pass and selection["selected"])
        if result["admit"]:
            atomic_json(
                self.output / "stage-a/learner-selection.json",
                {
                    "schema": "dsrl11-learner-selection-v1",
                    "selected": selection["selected"],
                    "numerical_summary_created_utc": result["created_utc"],
                    "rule": selection["rule"],
                },
            )
        atomic_json(destination, result)
        if not result["admit"]:
            raise PendingOperation("Numerical gate failed; diagnose before any learning")
        return result

    # -- stage B engineering ----------------------------------------------------------
    def engineering(self):
        summary_path = self.output / "engineering-summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text())
        jobs = self.plan["jobs"]["engineering_jobs"]
        self.servers([lane for lane in range(4) if not self.server_running(lane)])
        sequences = {lane: [j for j in jobs if j["lane"] == lane] for lane in range(4)}
        state = self.state.setdefault("engineering", {})

        def ready(job):
            if job["lane"] != LEARNER_GPU or job["round"] == 0:
                return True
            return state.get("learner_done", False) and self.server_running(LEARNER_GPU)

        def step():
            # After E3, GPU 3 measures the learner with three live rollouts elsewhere.
            if state.get("learner_done") or not self.valid("E3"):
                return
            if not state.get("learner_started"):
                self.stop_server(LEARNER_GPU)
                self.learner_service("engineering")
                state["learner_started"] = True
                self.save()
            if not state.get("profile_begun"):
                episodes = [self.descriptor(j) for j in ("E3",)]
                self.begin_http(
                    "engineering-profile", "/profile", {"updates": 128, "episodes": episodes}
                )
                state["profile_begun"] = utc()
                state["overlapping_rollouts"] = [
                    j for j in ("E0", "E1", "E2") if self.state["episodes"][j].get("valid") is None
                ]
                self.save()
            profile = self.poll_http("engineering-profile")
            if profile is None:
                return
            atomic_json(self.output / "engineering-profile.json", profile)
            export = self.http(
                "export-engineering",
                "/export-actor",
                {"condition": "engineering", "name": "engineering", "actor_version": 1},
            )
            self.actor_service(
                "engineering-cpu-actor", {"engineering": export["path"]}, ENGINEERING_ACTOR_PORT
            )
            parity = self.actor_parity(export)
            self.processes.stop("engineering-cpu-actor")
            self.http("shutdown-engineering", "/shutdown", {})
            self.wait(self.state["learner_process_engineering"])
            state.update(learner_done=True, profile_finished=utc(), parity=parity)
            self.save()
            self.server(LEARNER_GPU, "collection-after-engineering")

        self.schedule(sequences, ready, step)
        while not state.get("learner_done"):
            self.tick()
            step()
        profile = json.loads((self.output / "engineering-profile.json").read_text())
        parity = state["parity"]
        difference = profile.get("source_parity", {}).get("maximum_state_difference")
        source_pass = (
            isinstance(difference, (int, float))
            and math.isfinite(difference)
            and difference <= 1e-6
        )
        validations = {
            j["job_id"]: json.loads(
                (
                    Path(self.state["episodes"][j["job_id"]]["directory"]) / "validation.json"
                ).read_text()
            )
            for j in jobs
        }
        summary = {
            "schema": "dsrl11-engineering-summary-v1",
            "created_utc": utc(),
            "plan_sha256": sha256(self.plan_path),
            "episodes_valid": all(self.valid(j["job_id"]) for j in jobs),
            "episodes": {
                job_id: {
                    "steps": v["steps"],
                    "labels": {
                        k: v["labels"][k]
                        for k in (
                            "success",
                            "strict_safe_requested",
                            "raw_violation",
                            "counted_violation",
                            "ever_grasped",
                            "closed_once",
                        )
                    },
                    "returns": v["returns"],
                    "reward_counts": v["reward_counts"],
                    "geometry": v["geometry"],
                    "residual": v["residual"],
                    "duration_seconds": self.duration(job_id),
                }
                for job_id, v in validations.items()
            },
            "profile": {
                k: profile.get(k)
                for k in (
                    "source_parity",
                    "timed_updates",
                    "median_update_seconds",
                    "seconds",
                    "live_episodes",
                    "live_transitions",
                )
            },
            "profile_overlapping_rollouts": state.get("overlapping_rollouts"),
            "source_parity_pass": bool(source_pass),
            "actor_parity": parity,
            "resources": self.sync_counters(),
        }
        summary["admit"] = bool(
            summary["episodes_valid"] and source_pass and parity.get("valid") is True
        )
        atomic_json(summary_path, summary)
        if not summary["admit"]:
            raise AdmissionClosed("Engineering did not admit the prior bank")
        return summary

    def actor_parity(self, export):
        """GPU learner versus CPU export on eight live observations, then decoder identity."""
        output = self.output / "engineering-actor-parity.json"
        if output.exists():
            return json.loads(output.read_text())
        snapshots = build_residual_episode(
            self.state["episodes"]["E3"]["directory"], job_schedule=self.plan["jobs"]
        )["snapshots"][:8]
        rows, panel = [], []
        for index, snapshot in enumerate(snapshots):
            body = {
                "condition": "engineering",
                "actor_version": export["actor_version"],
                "job_id": "engineering-parity",
                "query_index": index,
                "rng_seed": 110,
                "pixels_b64": base64.b64encode(snapshot["pixels"].tobytes()).decode("ascii"),
                "state": snapshot["state"].tolist(),
            }
            samples = []
            for port in (LEARNER_PORT, ENGINEERING_ACTOR_PORT):
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/act",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                begin = time.monotonic()
                with urllib.request.urlopen(request, timeout=600) as response:
                    result = json.load(response)
                samples.append(
                    (np.asarray(result["residual"], np.float32), time.monotonic() - begin)
                )
            difference = float(np.max(np.abs(samples[0][0] - samples[1][0])))
            rows.append(
                {"index": index, "maximum_difference": difference, "cpu_seconds": samples[1][1]}
            )
            panel.append(
                {
                    "record_path": snapshot["record_path"],
                    "rng_before": snapshot["rng_before"],
                    "gpu_residual": samples[0][0].tolist(),
                    "cpu_residual": samples[1][0].tolist(),
                }
            )
        agree = all(
            np.allclose(
                np.asarray(p["gpu_residual"]), np.asarray(p["cpu_residual"]), atol=1e-5, rtol=1e-5
            )
            for p in panel
        )
        panel_path = self.output / "engineering-parity-panel.json"
        atomic_json(panel_path, {"schema": "dsrl11-parity-panel-v1", "cases": panel})
        decode = self.output / "stage-a" / "parity-decode.json"
        if not self.server_running(0):
            self.server(0)
        self.processes.launch(
            "parity-decode",
            [
                self.openpi,
                str(self.root / "experiments/scripts/probe_dsrl11.py"),
                "parity-decode",
                "--port",
                "8000",
                "--panel",
                str(panel_path),
                "--query-ledger",
                str(self.query_ledger),
                "--record-root",
                str(self.query_root),
                "--output",
                str(decode),
            ],
            cwd=self.root,
            kind="probe",
        )
        self.wait("parity-decode", allow_failure=True)
        decoded = json.loads(decode.read_text())
        result = {
            "observations": len(rows),
            "rows": rows,
            "maximum_difference": max(r["maximum_difference"] for r in rows),
            "median_cpu_seconds": float(
                np.median([r["cpu_seconds"] for r in rows[1:]] or [rows[0]["cpu_seconds"]])
            ),
            "agreement_1e5": bool(agree),
            "decoder_gripper_identity": decoded.get("valid") is True,
            "decoder": {k: decoded.get(k) for k in ("valid", "cases_passed", "cases")},
        }
        result["valid"] = bool(agree and result["decoder_gripper_identity"])
        atomic_json(output, result)
        return result

    # -- stage B prior bank -------------------------------------------------------------
    def bank(self):
        summary_path = self.output / "bank-summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text())
        jobs = self.plan["jobs"]["bank_jobs"]
        self.servers([lane for lane in range(4) if not self.server_running(lane)])
        self.schedule(
            {lane: [j for j in jobs if j["lane"] == lane] for lane in range(4)}, lambda job: True
        )
        labels = {j["job_id"]: self.state["episodes"][j["job_id"]]["labels"] for j in jobs}
        strict = sum(bool(v["strict_safe_requested"]) for v in labels.values())
        summary = {
            "schema": "dsrl11-bank-summary-v1",
            "created_utc": utc(),
            "plan_sha256": sha256(self.plan_path),
            "descriptors": [self.descriptor(j["job_id"]) for j in jobs],
            "labels": labels,
            "strict_safe_completions": strict,
            "required": BANK_SUCCESS_MINIMUM,
            "durations_seconds": {j["job_id"]: self.duration(j["job_id"]) for j in jobs},
            "resources": self.sync_counters(),
        }
        summary["admit"] = strict >= BANK_SUCCESS_MINIMUM
        atomic_json(summary_path, summary)
        if not summary["admit"]:
            raise AdmissionClosed("The prior bank missed its strict-safe completion gate")
        return summary

    # -- measured stage-C admission ------------------------------------------------------
    def admission(self):
        path = self.output / "stage-c-freeze.json"
        if path.exists():
            frozen = json.loads(path.read_text())
            if not frozen["admit"]:
                raise AdmissionClosed("Measured stage-C admission failed")
            return frozen
        engineering = json.loads((self.output / "engineering-summary.json").read_text())
        bank = json.loads((self.output / "bank-summary.json").read_text())
        selection = json.loads((self.output / "stage-a/learner-selection.json").read_text())
        utd = selection["selected"][0]
        episode = max(bank["durations_seconds"].values())
        update = engineering["profile"]["median_update_seconds"]
        # Twelve three-lane rounds, the final condition's last round of updates, ten
        # four-lane evaluation rounds, and the last admission plus the local copy.
        collection = 12 * episode
        final_training = 750 * utd * update * 1.10
        evaluation = 10 * episode
        reserve = max(2700.0, episode + 1200.0) - episode
        projected = collection + final_training + evaluation + reserve
        remaining = timestamp(self.rental["allocation_deadline_utc"]) - time.time()
        frozen = {
            "schema": "dsrl11-stage-c-freeze-v1",
            "created_utc": utc(),
            "plan_sha256": sha256(self.plan_path),
            "learner_selection": selection,
            "learner_selection_sha256": sha256(self.output / "stage-a/learner-selection.json"),
            "numerical_summary_sha256": sha256(self.output / "stage-a/numerical-summary.json"),
            "engineering_summary_sha256": sha256(self.output / "engineering-summary.json"),
            "bank_summary_sha256": sha256(self.output / "bank-summary.json"),
            "bank_descriptors": bank["descriptors"],
            "online_jobs": self.plan["jobs"]["online_jobs"],
            "evaluation_jobs": self.plan["jobs"]["evaluation_jobs"],
            "condition_order": self.plan["condition_order"],
            "primary_condition": self.plan["primary_condition"],
            "final_checkpoint": "episode-nine actor (version 2) of each condition",
            "projection_seconds": {
                "max_bank_episode": episode,
                "median_update": update,
                "collection": collection,
                "final_training": final_training,
                "evaluation": evaluation,
                "last_start_and_copy_reserve": reserve,
                "total": projected,
            },
            "remaining_allocation_seconds": remaining,
            "resources": self.sync_counters(),
        }
        frozen["admit"] = bool(projected <= remaining)
        with path.open("x") as stream:
            json.dump(frozen, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        self.event("stage_c_freeze", sha256=sha256(path), admit=frozen["admit"])
        if not frozen["admit"]:
            raise AdmissionClosed("The measured schedule cannot fit the authorized allowance")
        return frozen

    # -- stage C and D ----------------------------------------------------------------------
    def learner_step(self):
        """One serialized learner operation at a time, advanced without blocking."""
        ops = self.state["learner_ops"]
        active = self.state.get("learner_active")
        if active:
            try:
                result = self.poll_http(active["key"])
            except PendingOperation:
                status = self.service_status(LEARNER_PORT)
                if status.get("paused"):
                    self.state["paused"] = status["paused"]
                    self.save()
                    raise AdmissionClosed("Calibration pause rule stopped the comparison")
                raise
            if result is None:
                return
            ops[active["key"]] = {
                "done": utc(),
                **{
                    k: result.get(k)
                    for k in (
                        "actor_version",
                        "new_transitions",
                        "updates",
                        "path",
                        "manifest_sha256",
                    )
                },
            }
            if active["route"] == "/publish":
                self.state["published"][active["condition"]] = result["actor_version"]
            if active["route"] == "/export-actor":
                self.state["exports"][active["condition"]] = result
            self.state["learner_active"] = None
            self.save()
        for condition in self.plan["condition_order"]:
            for operation in self.condition_operations(condition):
                key = operation["key"]
                if key in ops:
                    continue
                if not operation["ready"]():
                    break
                if operation["route"] == "/train":
                    self.require_headroom("learner_updates", operation["updates"], phase="learning")
                self.begin_http(key, operation["route"], operation["body"]())
                self.state["learner_active"] = {
                    "key": key,
                    **{k: operation[k] for k in ("route", "condition")},
                }
                self.save()
                return

    def condition_operations(self, condition):
        jobs = [j for j in self.plan["jobs"]["online_jobs"] if j["condition"] == condition]
        by_phase = {phase: [j for j in jobs if j["phase"] == phase] for phase in PHASES}
        ops = self.state["learner_ops"]
        result = []
        for phase in PHASES:
            round_id = f"{condition}:{phase}"
            phase_jobs = by_phase[phase]
            result.append(
                {
                    "key": f"append-{condition}-{phase}",
                    "route": "/append",
                    "condition": condition,
                    "ready": lambda pj=phase_jobs: all(self.valid(j["job_id"]) for j in pj),
                    "body": lambda pj=phase_jobs, ph=phase, r=round_id: {
                        "condition": condition,
                        "phase": ph,
                        "round_id": r,
                        "episodes": [self.descriptor(j["job_id"]) for j in pj],
                    },
                }
            )
            if phase == "warmup":
                continue
            append_key = f"append-{condition}-{phase}"
            result.append(
                {
                    "key": f"train-{condition}-{phase}",
                    "route": "/train",
                    "condition": condition,
                    "updates": 750 * self.selected_utd(),
                    "ready": lambda: True,
                    "body": lambda r=round_id, a=append_key: {
                        "condition": condition,
                        "round_id": r,
                        "expected_new_transitions": ops[a]["new_transitions"],
                    },
                }
            )
            result.append(
                {
                    "key": f"publish-{condition}-{phase}",
                    "route": "/publish",
                    "condition": condition,
                    "ready": lambda: True,
                    "body": lambda r=round_id: {"condition": condition, "round_id": r},
                }
            )
            result.append(
                {
                    "key": f"save-{condition}-{phase}",
                    "route": "/save",
                    "condition": condition,
                    "ready": lambda: True,
                    "body": lambda ph=phase: {"name": f"{condition}-{ph}"},
                }
            )
        result.append(
            {
                "key": f"export-{condition}-final",
                "route": "/export-actor",
                "condition": condition,
                "ready": lambda: True,
                "body": lambda: {
                    "condition": condition,
                    "name": f"{condition}-final",
                    "actor_version": 2,
                },
            }
        )
        return result

    def selected_utd(self):
        return json.loads((self.output / "stage-a/learner-selection.json").read_text())["selected"][
            0
        ]

    def pilot(self):
        summary_path = self.output / "pilot-summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text())
        frozen = self.admission()
        self.stop_server(LEARNER_GPU)
        status = self.learner_service("learning")
        if not status.get("bank_loaded"):
            bank = json.loads((self.output / "bank-summary.json").read_text())
            self.http(
                "load-bank",
                "/load-bank",
                {
                    "episodes": frozen["bank_descriptors"],
                    "strict_safe_completions": bank["strict_safe_completions"],
                },
            )
        self.servers([lane for lane in COLLECTION_LANES if not self.server_running(lane)])
        online, evaluation = self.plan["jobs"]["online_jobs"], self.plan["jobs"]["evaluation_jobs"]
        sequences = {
            lane: sorted([j for j in online if j["lane"] == lane], key=lambda j: j["round"])
            + [j for j in evaluation if j["lane"] == lane]
            for lane in range(4)
        }
        evaluation_actors = self.state.setdefault("evaluation_actors", {})

        def ready(job):
            if job["stage"] == "online":
                return (
                    job["phase"] != "batch_b"
                    or self.state["published"].get(job["condition"], 0) >= 1
                )
            if job["lane"] == LEARNER_GPU and not self.state.get("learner_retired"):
                return False
            if job["arm"] == "native":
                return True
            return job["condition"] in evaluation_actors

        def step():
            self.learner_step()
            for condition, export in self.state["exports"].items():
                if condition not in evaluation_actors:
                    if export["actor_version"] != 2:
                        raise ValueError("Only the frozen episode-nine actor is evaluated")
                    self.actor_service(
                        f"evaluation-actor-{condition}",
                        {condition: export["path"]},
                        EVALUATION_ACTOR_PORTS[condition],
                    )
                    evaluation_actors[condition] = export
                    self.save()
            if len(self.state["exports"]) == len(CONDITIONS) and not self.state.get(
                "learner_retired"
            ):
                if self.state.get("learner_active"):
                    return
                if "final-learner-status" not in self.state:
                    self.state["final-learner-status"] = self.service_status(LEARNER_PORT)
                    self.save()
                self.http("shutdown-learning", "/shutdown", {})
                self.wait(self.state["learner_process_learning"])
                self.state["learner_retired"] = utc()
                self.save()
                self.server(LEARNER_GPU, "evaluation")

        self.schedule(sequences, ready, step)
        while not self.state.get("learner_retired"):
            self.tick()
            step()
        result = {
            "schema": "dsrl11-pilot-summary-v1",
            "finished_utc": utc(),
            "plan_sha256": sha256(self.plan_path),
            "stage_c_freeze_sha256": sha256(self.output / "stage-c-freeze.json"),
            "exports": self.state["exports"],
            "final_learner_status": self.state.get("final-learner-status"),
            "resources": self.sync_counters(),
            "episodes": self.state["episodes"],
        }
        atomic_json(summary_path, result)
        return result

    def all(self):
        self.numerical()
        self.engineering()
        self.bank()
        self.admission()
        return self.pilot()

    def close(self, *, stop=False):
        if stop:
            for path in self.processes.directory.glob("*.json"):
                if any(path.name.endswith(s) for s in (".pid.json", ".child.json", ".exit.json")):
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
    parser.add_argument(
        "--stage",
        choices=("numerical", "engineering", "bank", "admission", "pilot", "all"),
        required=True,
    )
    parser.add_argument(
        "--plan", type=Path, default=ROOT / "progress/receipts/11-dsrl-stage-a.json"
    )
    parser.add_argument(
        "--rental", type=Path, default=ROOT / "progress/rental-dsrl11-20261025.json"
    )
    parser.add_argument("--repair-receipt", type=Path)
    parser.add_argument("--stop-workers", action="store_true")
    args = parser.parse_args()
    coordinator = Coordinator(
        ROOT, args.plan, args.rental, args.stage, repair_receipt=args.repair_receipt
    )
    stop = args.stop_workers
    try:
        result = getattr(coordinator, args.stage)()
        stop = stop or args.stage in ("pilot", "all")
        print(json.dumps({k: result[k] for k in ("schema", "admit") if k in result}), flush=True)
        return 0
    except PendingOperation as error:
        coordinator.event("pending", reason=str(error))
        print(json.dumps({"pending": str(error)}), flush=True)
        return 2
    except AdmissionClosed as error:
        coordinator.event("admission_closed", reason=str(error))
        print(json.dumps({"admission_closed": str(error)}), flush=True)
        stop = True
        return 3
    except BaseException as error:
        coordinator.event("queue_halted", error_type=type(error).__name__, reason=str(error)[:2000])
        raise
    finally:
        coordinator.close(stop=stop)


if __name__ == "__main__":
    raise SystemExit(main())
