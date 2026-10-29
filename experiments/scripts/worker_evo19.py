"""Task 19 host worker: run queued Jar episodes one at a time against the local server.

Jobs are JSON files in outputs/evo19/queue/ (written by the workstation orchestrator).
The worker claims the lexically first job, re-seeds the native sampler, runs one
evaluator episode with the job's adapter settings through the task 15 command line,
validates the recording and replays every adapter decision, then writes
outputs/evo19/results/<job>.json. A crash or timeout earns one retry; a STOP file halts
the worker between episodes. Run with the behavior environment's Python.
"""

import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/helpers"))
from evo19_validation import validate_recording  # noqa: E402

OUT = ROOT / "outputs/evo19"
QUEUE, CLAIMED, RESULTS = OUT / "queue", OUT / "claimed", OUT / "results"
TIMEOUT_S = 2400


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def log(event, **fields):
    with (OUT / "worker-events.jsonl").open("a") as handle:
        handle.write(json.dumps({"utc": utc(), "event": event, **fields}) + "\n")


def run_episode(job, attempt):
    leaf = f"{job['id']}_a{attempt}"
    directory = OUT / "episodes" / leaf
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "job.json").write_text(json.dumps({**job, "attempt": attempt}, indent=2))
    with (directory / "reseed.log").open("w") as handle:
        probe = subprocess.run(
            [str(ROOT / "third_party/openpi/.venv/bin/python"), "-I",
             str(ROOT / "experiments/scripts/probe_regrasp15.py"), "--reseed-only"],
            stdout=handle, stderr=subprocess.STDOUT, timeout=600)
    if probe.returncode:
        raise RuntimeError("Re-seeding probe failed")
    config_path = directory / "adapter-config.json"
    config_path.write_text(json.dumps(job["settings"], indent=2) + "\n")
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "0", "OMNIGIBSON_HEADLESS": "1",
        "NVIDIA_DRIVER_CAPABILITIES": "all", "OMNI_KIT_ACCEPT_EULA": "YES",
        "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json", "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": f"{ROOT}/experiments/helpers:{ROOT}/scripts/todo2",
        "SAFETY_DIAL_REGRASP_CONFIG": str(config_path),
        "SAFETY_DIAL_REGRASP_JOURNAL": str(directory / "adapter.jsonl"),
    })
    env["__EGL_VENDOR_LIBRARY_FILENAMES"] = (
        str(ROOT / "scripts/todo1b/nvidia-egl-vendor.json")
        if Path("/opt/nvidia-drivers/lib64/libEGL_nvidia.so.0").exists()
        else "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    env.pop("EVAL_USE_GPU_DYNAMICS", None)
    cmd = [sys.executable, "-m", "maniguard.eval.benchmark",
           "--config", str(ROOT / "configs/regrasp15/jar_eval.yaml"),
           "--host", "127.0.0.1", "--port", "8000",
           "--benchmark-root", "outputs/lerobot_datasets/maniguard-bench/jar_transport",
           "--scenes", f"task_{job['task']:04d}/base", "--max-scenes", "1",
           "--output-dir", str(directory), "--run-name", "eval",
           "--metrics", "success", "safety", "--headless", "--seed", str(job["seed"]),
           "--recording-factory", "accounted_recorder:create_observer",
           "--recording-output-dir", str(directory / "recordings")]
    started = time.monotonic()
    with (directory / "evaluator.log").open("w") as handle:
        process = subprocess.Popen(cmd, cwd=ROOT / "third_party/ManiGuard", env=env,
                                   stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            rc = process.wait(timeout=TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            rc = 124
    elapsed = round(time.monotonic() - started, 3)
    rows = list(directory.rglob("results.jsonl"))
    if len(rows) != 1:
        return None, {"rc": rc, "elapsed_s": elapsed, "leaf": leaf}
    found = [json.loads(x) for x in rows[0].read_text().splitlines() if x]
    if len(found) != 1 or found[0].get("status") != "completed":
        return None, {"rc": rc, "elapsed_s": elapsed, "leaf": leaf}
    recordings = list(directory.glob("recordings/*_base.npz"))
    sidecars = list(directory.rglob("*_ltl.json"))
    if len(recordings) != 1 or len(sidecars) != 1:
        raise ValueError("Completed episode is missing a unique recording or sidecar")
    result = validate_recording(recordings[0], rows[0], sidecars[0],
                                directory / "adapter.jsonl", job["settings"])
    if (result["row"]["scene_name"] != f"task_{job['task']:04d}/base"
            or result["row"]["seed"] != job["seed"]):
        raise ValueError("Completed episode does not match its job")
    (directory / "validation.json").write_text(json.dumps(result, indent=2))
    return result, {"rc": rc, "elapsed_s": elapsed, "leaf": leaf}


def main():
    for folder in (QUEUE, CLAIMED, RESULTS, OUT / "episodes"):
        folder.mkdir(parents=True, exist_ok=True)
    host = (OUT / "host-name.txt").read_text().strip()
    log("worker_start", host=host, pid=os.getpid())
    while not (OUT / "STOP").exists():
        jobs = sorted(QUEUE.glob("*.json"))
        if not jobs:
            time.sleep(3)
            continue
        claimed = CLAIMED / jobs[0].name
        try:
            os.rename(jobs[0], claimed)
        except FileNotFoundError:
            continue
        job = json.loads(claimed.read_text())
        job["settings"] = {k: float(v) if v in ("inf", "Infinity") else v
                           for k, v in job["settings"].items()}
        record = {"job": job, "host": host, "started_utc": utc(), "attempts": []}
        for attempt in (1, 2):
            if (OUT / "STOP").exists():
                break
            log("episode_start", job=job["id"], attempt=attempt)
            try:
                result, info = run_episode(job, attempt)
            except Exception as error:  # validation or launch failure: keep the evidence
                info = {"error": type(error).__name__, "detail": str(error)[-500:],
                        "trace": traceback.format_exc()[-1500:]}
                result = None
            record["attempts"].append(info)
            log("episode_end", job=job["id"], attempt=attempt, ok=result is not None, **{
                k: v for k, v in info.items() if k != "trace"})
            if result is not None:
                record.update({"outcome": result["outcome"], "adapter": result["adapter"],
                               "settings": result["settings"], "steps": result["steps"],
                               "elapsed_s": info["elapsed_s"], "leaf": info["leaf"]})
                break
        record["finished_utc"] = utc()
        record["valid"] = "outcome" in record
        tmp = RESULTS / f".{claimed.name}.tmp"
        tmp.write_text(json.dumps(record, indent=2))
        os.rename(tmp, RESULTS / claimed.name)
        claimed.unlink()
    log("worker_stop", host=host)


if __name__ == "__main__":
    main()
