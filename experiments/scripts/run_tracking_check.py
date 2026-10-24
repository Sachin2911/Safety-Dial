"""Execute the frozen twelve-job native check on two independent GPU lanes.

Run with the behavior environment's Python. A single process owns the shared
budget; each thread runs one sequential lane and validates every episode before
admitting its successor. Outputs are fresh, never overwritten on a restart.
"""

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.tracking import validate_recording


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Check:
    def __init__(self, plan_path, rental_path):
        self.plan = json.loads(plan_path.read_text())
        self.rental = json.loads(rental_path.read_text())
        self.output = ROOT / "outputs/tracking04"
        self.output.mkdir(parents=True, exist_ok=True)
        self.ledger = (self.output / "attempts.jsonl").open("x", buffering=1)
        self.lock = threading.Lock()
        self.starts = 0
        self.replacements = 0
        self.admission = dt.datetime.fromisoformat(self.rental["created_utc"]).timestamp()
        self.admission += 3600 * self.plan["job_admission_rental_hours"]
        self.record("queue_start", plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                    instance=self.rental["instance_id"])

    def record(self, event, **fields):
        with self.lock:
            row = {"utc": utc(), "event": event, **fields}
            self.ledger.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

    def reserve(self, job, attempt):
        with self.lock:
            if (time.time() >= self.admission or (self.output / "STOP").exists()
                    or self.starts >= self.plan["max_total_starts"]):
                return False
            if attempt > 1:
                if self.replacements >= self.plan["max_total_replacements"]:
                    return False
                self.replacements += 1
            self.starts += 1
            self.ledger.write(json.dumps({"utc": utc(), "event": "start_reserved",
                                         **job, "attempt": attempt,
                                         "start_number": self.starts}) + "\n")
            return True

    def run(self, job, attempt):
        lane = job["preferred_lane"]
        leaf = f"job{job['job']:02d}_task{job['task']:02d}_seed{job['seed']}_attempt{attempt}"
        directory = self.output / "episodes" / leaf
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "job.json").write_text(json.dumps({**job, "attempt": attempt}, indent=2))
        if attempt > 1:
            # A different episode_seed makes the native server reset its sampler.
            with (directory / "reseed.log").open("w") as log:
                probe = subprocess.run(
                    [str(ROOT / "third_party/openpi/.venv/bin/python"), "-I",
                     str(ROOT / "scripts/todo1/probe_server.py"), "--port", str(8000 + lane),
                     "--queries", "0"], stdout=log, stderr=subprocess.STDOUT, timeout=600,
                )
            self.record("reseed_probe", job=job["job"], attempt=attempt, rc=probe.returncode)
            if probe.returncode:
                raise RuntimeError("Re-seeding probe failed; queue halted")
        env = dict(os.environ)
        env.update({
            "CUDA_VISIBLE_DEVICES": str(lane), "OMNIGIBSON_HEADLESS": "1",
            "NVIDIA_DRIVER_CAPABILITIES": "all", "OMNI_KIT_ACCEPT_EULA": "YES",
            "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{ROOT}/experiments/helpers:{ROOT}/scripts/todo2",
        })
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = (
            str(ROOT / "scripts/todo1b/nvidia-egl-vendor.json")
            if Path("/opt/nvidia-drivers/lib64/libEGL_nvidia.so.0").exists()
            else "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
        )
        env.pop("EVAL_USE_GPU_DYNAMICS", None)
        cmd = [sys.executable, "-m", "maniguard.eval.benchmark", "--config",
               "configs/eval/jar_transport_joint_legacy256.yaml", "--host", "127.0.0.1",
               "--port", str(8000 + lane), "--benchmark-root",
               "outputs/lerobot_datasets/maniguard-bench/jar_transport", "--scenes",
               f"task_{job['task']:04d}/base", "--max-scenes", "1", "--output-dir",
               str(directory), "--run-name", "eval", "--metrics", "success", "safety",
               "--headless", "--seed", str(job["seed"]), "--recording-factory",
               "accounted_recorder:create_observer", "--recording-output-dir",
               str(directory / "recordings")]
        (directory / "command.json").write_text(json.dumps(cmd, indent=2))
        self.record("simulator_launch", **job, attempt=attempt, leaf=leaf)
        started = time.monotonic()
        with (directory / "evaluator.log").open("w") as log:
            process = subprocess.Popen(cmd, cwd=ROOT / "third_party/ManiGuard", env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = process.wait(timeout=self.plan["episode_timeout_seconds"])
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
        complete = False
        if len(rows) == 1:
            found = [json.loads(x) for x in rows[0].read_text().splitlines() if x]
            complete = len(found) == 1 and found[0].get("status") == "completed"
        self.record("simulator_exit", **job, attempt=attempt, rc=rc, elapsed_s=elapsed,
                    completed_row=complete)
        if not complete:
            return False
        recordings = list(directory.rglob("*.npz"))
        sidecars = list(directory.rglob("*_ltl.json"))
        if len(recordings) != 1 or len(sidecars) != 1:
            raise ValueError("Completed episode is missing a unique recording or sidecar")
        result = validate_recording(recordings[0], rows[0], sidecars[0])
        (directory / "validation.json").write_text(json.dumps(result, indent=2))
        self.record("validated", **job, attempt=attempt, **result)
        return True

    def lane(self, lane):
        try:
            for job in self.plan["jobs"]:
                if job["preferred_lane"] != lane:
                    continue
                for attempt in range(1, self.plan["max_replacements_per_job"] + 2):
                    if not self.reserve(job, attempt):
                        self.record("admission_closed", lane=lane, job=job["job"])
                        return
                    if self.run(job, attempt):
                        break
            self.record("lane_finished", lane=lane)
        except Exception as error:
            (self.output / "STOP").write_text(f"lane {lane}: {type(error).__name__}\n")
            self.record("lane_error", lane=lane, kind=type(error).__name__, detail=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path,
                        default=ROOT / "progress/receipts/04-tracking-plan.json")
    parser.add_argument("--rental", type=Path,
                        default=ROOT / "progress/rental-tracking04-20261024.json")
    args = parser.parse_args()
    check = Check(args.plan, args.rental)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [executor.submit(check.lane, lane) for lane in range(2)]
        for job in jobs:
            job.result()
    check.record("queue_finished", starts=check.starts, replacements=check.replacements)


if __name__ == "__main__":
    main()
