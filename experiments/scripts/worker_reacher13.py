"""Persistent task-13 worker: load the frozen model once, run queued episodes durably.

Jobs arrive as files in outputs/reacher13/queue/pending. A worker claims one by atomic
rename into queue/claimed, runs it, writes the attempt directory (result.json, states.npz,
history.json) and moves the ticket to queue/done or queue/failed. Several workers share
one GPU; each is a separate process with its own CUDA context.
"""

import argparse
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("STABLEWM_HOME", str(ROOT / "data/reacher13/stablewm"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.reacher13_protocol import digest, verify_freeze

OUT = ROOT / "outputs/reacher13"
QUEUE = OUT / "queue"


def claim(worker_id):
    for ticket in sorted((QUEUE / "pending").glob("*.json")):
        target = QUEUE / "claimed" / ticket.name
        try:
            os.rename(ticket, target)
        except FileNotFoundError:
            continue
        data = json.loads(target.read_text())
        data["claimed_utc"] = utc()
        data["worker"] = worker_id
        atomic_json(target, data)
        return target, data
    return None, None


def finish(ticket, data, folder, status):
    data["finished_utc"] = utc()
    data["status"] = status
    atomic_json(folder / "ticket.json", data)
    destination = QUEUE / ("done" if status == "complete" else "failed") / ticket.name
    atomic_json(destination, data)
    ticket.unlink(missing_ok=True)


def main():
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--idle-exit", type=float, default=0.0, help="seconds idle before exit")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    freeze = verify_freeze(args.freeze)
    freeze_sha = digest(args.freeze)
    for name in ("pending", "claimed", "done", "failed"):
        (QUEUE / name).mkdir(parents=True, exist_ok=True)
    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    from helpers.reacher13_runtime import Runtime

    torch.set_num_threads(2)
    started = time.perf_counter()
    runtime = Runtime(device=args.device, spec=freeze["design"])
    heartbeat = OUT / "workers" / f"{worker_id}.json"
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        heartbeat,
        {
            "utc": utc(),
            "pid": os.getpid(),
            "state": "ready",
            "load_seconds": time.perf_counter() - started,
            "probe_stats": runtime.probe_stats,
            "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu",
        },
    )
    print(json.dumps({"event": "ready", "worker": worker_id}), flush=True)
    idle_since = time.monotonic()
    while True:
        if (OUT / "STOP").exists():
            print(json.dumps({"event": "stop_file", "worker": worker_id}), flush=True)
            return
        ticket, data = claim(worker_id)
        if ticket is None:
            if args.idle_exit and time.monotonic() - idle_since > args.idle_exit:
                return
            atomic_json(heartbeat, {"utc": utc(), "pid": os.getpid(), "state": "idle"})
            time.sleep(0.5)
            continue
        idle_since = time.monotonic()
        folder = Path(data["attempt_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(
            heartbeat,
            {"utc": utc(), "pid": os.getpid(), "state": "running", "job": data["job"]["id"]},
        )
        try:
            if data["freeze_sha256"] != freeze_sha:
                raise ValueError("Ticket belongs to a different source freeze")
            job = data["job"]
            atomic_json(folder / "worker-start.json", {"utc": utc(), "worker": worker_id})
            tick = time.perf_counter()
            outcome = runtime.run(job)
            elapsed = time.perf_counter() - tick
            np.savez_compressed(
                folder / "states.npz",
                states=outcome["states"],
                rewards=outcome["rewards"],
                actions=outcome["actions"],
            )
            (folder / "history.json").write_text(json.dumps(outcome["history"]) + "\n")
            result = {
                "status": "complete",
                "job": job,
                "worker": worker_id,
                "utc": utc(),
                "episode_seconds": elapsed,
                "terminated": outcome["terminated"],
                **outcome["metrics"],
                "states_sha256": digest(folder / "states.npz"),
            }
            atomic_json(folder / "result.json", result)
            finish(ticket, data, folder, "complete")
            print(
                json.dumps(
                    {
                        "event": "complete",
                        "job": job["id"],
                        "success": result["success"],
                        "exposure": result["exposure"],
                        "J": result["J"],
                        "seconds": round(elapsed, 1),
                    }
                ),
                flush=True,
            )
        except Exception:
            (folder / "error.txt").write_text(traceback.format_exc())
            finish(ticket, data, folder, "failed")
            print(json.dumps({"event": "failed", "job": data["job"]["id"]}), flush=True)
            if "CUDA" in (folder / "error.txt").read_text():
                raise


if __name__ == "__main__":
    main()
