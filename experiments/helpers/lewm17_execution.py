"""Attempt accounting and registered-worker infrastructure for task 17."""

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from helpers.lewm17_budget import ceilings
from helpers.lewm17_protocol import OUT, ROOT, digest, write_json


def ready():
    if not (OUT / "environment/ready").exists():
        raise RuntimeError("Task-17 setup is not ready")
    status = json.loads((OUT / "control/guard-status.json").read_text())
    if status["closing"] or (OUT / "STOP").exists():
        raise InterruptedError("Task-17 STOP or closeout")
    if not Path(f"/proc/{status['pid']}/cmdline").exists():
        raise RuntimeError("Independent guard is missing")
    if b"guard_lewm17.py" not in Path(f"/proc/{status['pid']}/cmdline").read_bytes():
        raise RuntimeError("Independent guard identity mismatch")
    if time.time() > (OUT / "control/guard-status.json").stat().st_mtime + 45:
        raise RuntimeError("Independent guard status is stale")
    if time.time() >= status["budget"]["deadline_epoch"] - 1800:
        raise InterruptedError("Closeout reserve has started")


def register():
    subprocess.run(
        [
            sys.executable,
            "experiments/scripts/control_lewm17.py",
            "register",
            "--pid",
            str(os.getpid()),
            "--script",
            sys.argv[0],
        ],
        check=True,
    )


def configure(gpu):
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        MUJOCO_GL="egl",
        MUJOCO_EGL_DEVICE_ID=str(gpu),
        PYOPENGL_PLATFORM="egl",
        SDL_VIDEODRIVER="dummy",
        SDL_AUDIODRIVER="dummy",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        HF_HOME=str(OUT / "hf-cache"),
        MPLCONFIGDIR=str(OUT / "cache/matplotlib"),
        STABLEWM_HOME=str(ROOT / "data/lewm17/stablewm"),
    )


def verify_freeze(path):
    manifest = json.loads(Path(path).read_text())
    for relative, checksum in manifest["source_sha256"].items():
        if digest(ROOT / relative) != checksum:
            raise ValueError(f"Source changed since freeze: {relative}")
    return manifest


class Executor:
    def __init__(self, gpu, manifest, group, allowance):
        self.gpu, self.manifest, self.group = gpu, manifest, group
        self.allowance = allowance
        self.runtimes = {}
        self.used = 0
        self.actual = {}
        self.started = time.time()

    def run(self, job):
        ready()
        if self.used >= self.allowance:
            raise RuntimeError("Frozen per-run episode/resource reservation exhausted")
        attempt = OUT / "attempts" / job["id"]
        # Existing attempts are never overwritten or silently rerun.
        attempt.mkdir(parents=True, exist_ok=False)
        reservation = ceilings(job["env"])
        write_json(
            attempt / "ticket.json",
            {
                "job": job,
                "group": self.group,
                "reservation": reservation,
                "ordinal": self.used,
                "group_episode_ceiling": self.allowance,
                "group_resource_ceilings": {k: v * self.allowance for k, v in reservation.items()},
            },
        )
        write_json(
            attempt / "worker.json",
            {
                "gpu": self.gpu,
                "pid": os.getpid(),
                "started_epoch": time.time(),
                "manifest": str(self.manifest),
                "manifest_sha256": digest(self.manifest),
                "host": "33061",
                "instance_id": 57809511,
            },
        )
        self.used += 1
        runtime = None
        try:
            if job["env"] not in self.runtimes:
                from helpers.lewm17_runtime import Runtime

                self.runtimes[job["env"]] = Runtime(job["env"], threads=2)
            runtime = self.runtimes[job["env"]]
            # Clear stale counters before any new attempt can fail during reset.
            runtime.counters = {}
            result = runtime.run(job, attempt)
        except (
            RuntimeError,
            ValueError,
            TypeError,
            KeyError,
            OSError,
            IndexError,
            AssertionError,
            ImportError,
        ) as exc:
            result = {
                "status": "error",
                "job": job,
                "error": f"{type(exc).__name__}: {exc}",
                "counters": dict(getattr(runtime, "counters", {})),
            }
            write_json(attempt / "result.json", result)
            (attempt / "traceback.txt").write_text(traceback.format_exc())
        for key, value in result.get("counters", {}).items():
            self.actual[key] = self.actual.get(key, 0) + value
        write_json(
            OUT / "groups" / f"{self.group}.json",
            {
                "episodes": self.used,
                "allowance": self.allowance,
                "actual": self.actual,
                "wall_seconds": time.time() - self.started,
                "status": result["status"],
            },
        )
        print(
            json.dumps(
                {
                    "id": job["id"],
                    "status": result["status"],
                    "safe": result.get("requested_safe_completion"),
                    "seconds": result.get("seconds"),
                }
            ),
            flush=True,
        )
        if result["status"] != "complete":
            raise RuntimeError(f"Attempt failed, retained for repair: {job['id']}")
        decisions = json.loads((attempt / "decisions.json").read_text())
        return result, decisions
