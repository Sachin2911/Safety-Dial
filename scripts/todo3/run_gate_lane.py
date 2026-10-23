"""Run one lane of the frozen todo 3 gate plan (progress/receipts/03-plan.json), item 6.

Each job of the lane, in plan order, is one episode through scripts/todo1/eval_family_lane.sh
(symlinked into the ManiGuard checkout as scripts/eval_family_lane.sh) with TASKS, SEED and
PORT from the plan and the plan's output leaf. A job is skipped when its leaf already holds a
completed row (resume after an interruption). A job whose first attempt leaves no completed
row is rerun once into <leaf>_rerun1; a second failure is recorded and the lane moves on.
Progress goes to third_party/logs/03-gate-lane<L>.jsonl, one line per attempt.

    python3 -I scripts/todo3/run_gate_lane.py --lane 0 [--plan progress/receipts/03-plan.json] [--only JOB,...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/Safety-Dial")
TP = ROOT / "third_party"
MG = TP / "ManiGuard"
PY_BEHAVIOR = "/opt/miniforge3/envs/behavior/bin/python"
CONFIG = "configs/eval/jar_transport_joint_legacy256.yaml"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def completed_row(leaf: str):
    p = MG / "outputs/eval_logs" / leaf / "ID" / "results.jsonl"
    if not p.is_file():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    done = [r for r in rows if r.get("status") == "completed"]
    return done[-1] if done else None


def attempted(leaf: str) -> bool:
    return (MG / "outputs/eval_logs" / leaf).exists()


def run_job(job: dict, leaf: str, log_dir: Path) -> int:
    env = dict(os.environ)
    env.update({
        "PYTHON_CMD": PY_BEHAVIOR, "LEVELS": "base", "TASKS": job["task"], "SEED": str(job["seed"]),
        "PORT": str(job["port"]), "CUDA_VISIBLE_DEVICES": str(job["lane"]),
        "OMNIGIBSON_HEADLESS": "1", "NVIDIA_DRIVER_CAPABILITIES": "all", "OMNI_KIT_ACCEPT_EULA": "YES",
        "VK_ICD_FILENAMES": env.get("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json"),
    })
    if "__EGL_VENDOR_LIBRARY_FILENAMES" not in env:
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = (
            str(ROOT / "scripts/todo1b/nvidia-egl-vendor.json")
            if Path("/opt/nvidia-drivers/lib64/libEGL_nvidia.so.0").is_file()
            else "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
    log = log_dir / f"job{job['job']:03d}_{leaf.replace('/', '_')}.log"
    with open(log, "w") as f:
        return subprocess.call(["bash", "scripts/eval_family_lane.sh", "jar_transport", leaf, CONFIG],
                               cwd=MG, env=env, stdout=f, stderr=subprocess.STDOUT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lane", type=int, required=True)
    ap.add_argument("--plan", default=str(ROOT / "progress/receipts/03-plan.json"))
    ap.add_argument("--only", default="", help="comma-separated job numbers to run (default: every job of the lane)")
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    only = {int(x) for x in args.only.split(",") if x}
    jobs = [j for j in plan["jobs"] if j["lane"] == args.lane and (not only or j["job"] in only)]
    log_dir = TP / "logs" / "03-gate"
    log_dir.mkdir(parents=True, exist_ok=True)
    progress = TP / "logs" / f"03-gate-lane{args.lane}.jsonl"

    def record(rec: dict) -> None:
        rec = {"t": now(), "lane": args.lane, **rec}
        with open(progress, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    record({"event": "lane_start", "n_jobs": len(jobs), "plan_frozen_utc": plan["frozen_utc"]})
    for job in jobs:
        leaf = job["leaf"]
        if completed_row(leaf) or completed_row(leaf + "_rerun1"):
            record({"event": "skip_done", "job": job["job"], "leaf": leaf})
            continue
        for attempt, use_leaf in ((1, leaf), (2, leaf + "_rerun1")):
            if attempted(use_leaf):
                record({"event": "skip_attempted", "job": job["job"], "attempt": attempt, "leaf": use_leaf})
                continue
            t0 = time.time()
            record({"event": "start", "job": job["job"], "attempt": attempt, "leaf": use_leaf, "seed": job["seed"],
                    "task": job["task"], "arm": job["arm"], "port": job["port"]})
            rc = run_job(job, use_leaf, log_dir)
            row = completed_row(use_leaf)
            record({"event": "end", "job": job["job"], "attempt": attempt, "leaf": use_leaf, "rc": rc,
                    "elapsed_s": round(time.time() - t0, 1), "status": row.get("status") if row else "no_completed_row",
                    "success": row.get("success") if row else None,
                    "counted_violation": row.get("counted_violation") if row else None,
                    "steps": row.get("steps") if row else None})
            if row:
                break
    record({"event": "lane_end"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
