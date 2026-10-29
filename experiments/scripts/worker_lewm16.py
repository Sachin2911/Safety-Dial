"""Resumable task-16 worker. Atomic case claims and conservative global action reservation."""

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm16_protocol import OUT, digest, write_json


def reserve(attempt, job):
    with (OUT / "control/budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        used = 0
        for ticket in (OUT / "attempts").glob("*/ticket.json"):
            p = ticket.parent
            result = p / "result.json"
            if result.exists():
                data = json.loads(result.read_text())
                used += data.get("counters", {}).get(
                    "environment_actions", data.get("env_steps", 0)
                )
            else:
                used += json.loads(ticket.read_text()).get("max_steps", 50)
        requested = int(job.get("max_steps", 50))
        if used + requested > 160000:
            raise RuntimeError("Campaign action ceiling would be exceeded")
        write_json(attempt / "ticket.json", job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--mapping", default="0,1:2,3")
    args = parser.parse_args()
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        MUJOCO_GL="egl",
        MUJOCO_EGL_DEVICE_ID=str(args.gpu),
        PYOPENGL_PLATFORM="egl",
        SDL_VIDEODRIVER="dummy",
        SDL_AUDIODRIVER="dummy",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    import torch
    from helpers.lewm16_runtime import Runtime

    torch.set_num_threads(1)
    manifest = json.loads(Path(args.manifest).read_text())
    for path, sha in manifest["source_sha256"].items():
        if digest(ROOT / path) != sha:
            raise ValueError(f"Source changed after freeze: {path}")
    mapping = {
        env: [int(x) for x in part.split(",")]
        for env, part in zip(("tworoom", "reacher"), args.mapping.split(":"))
    }
    runtimes = {}
    for job in manifest["jobs"]:
        if (OUT / "STOP").exists():
            break
        assigned = mapping[job["env"]]
        block_hash = int(hashlib.sha256(job["block"].encode()).hexdigest()[:8], 16)
        if assigned[block_hash % len(assigned)] != args.gpu:
            continue
        if block_hash // len(assigned) % args.lanes != args.lane:
            continue
        attempt = OUT / "attempts" / job["id"]
        try:
            attempt.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        reserve(attempt, job)
        write_json(
            attempt / "worker.json",
            {
                "gpu": args.gpu,
                "lane": args.lane,
                "pid": os.getpid(),
                "started_epoch": time.time(),
                "manifest": args.manifest,
                "manifest_sha256": digest(args.manifest),
            },
        )
        runtime = None
        try:
            if job["env"] not in runtimes:
                runtimes[job["env"]] = Runtime(job["env"], threads=2)
            runtime = runtimes[job["env"]]
            result = runtime.run(job, attempt)
            print(
                json.dumps(
                    {
                        "id": job["id"],
                        "status": result["status"],
                        "safe": result["requested_safe_completion"],
                        "seconds": result["seconds"],
                    }
                ),
                flush=True,
            )
            if result["status"] == "error":
                return 1
        except Exception as exc:
            # Local numeric/runtime errors contain no provider credential responses.
            counters = dict(getattr(runtime, "counters", {})) if runtime else {}
            write_json(
                attempt / "result.json",
                {
                    "status": "error",
                    "job": job,
                    "error": f"{type(exc).__name__}: {exc}",
                    "counters": counters,
                },
            )
            (attempt / "traceback.txt").write_text(traceback.format_exc())
            print(
                json.dumps({"id": job["id"], "status": "error", "type": type(exc).__name__}),
                flush=True,
            )
            # One bad engineering batch should not waste its entire queue.
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
