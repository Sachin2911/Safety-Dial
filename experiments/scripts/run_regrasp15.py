"""Run task 15 stage A, the stage B dial screen or the stage C confirmation.

Run with the behavior environment's Python on the recorded rental. One sequential
lane: one simulator at a time against one policy server. Every episode is validated
and its adapter decisions replayed before its successor is admitted. A simulator crash
or timeout earns one replacement attempt with the same seed; a validation failure halts
the queue. Outputs are fresh, never overwritten on a restart.
"""

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/helpers"))
from regrasp15_validation import validate_recording  # noqa: E402

OUT = ROOT / "outputs/regrasp15"
SERVER_LEDGER = OUT / "server/candidates.jsonl"


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Queue:
    def __init__(self, plan_path, design_path, rental_path, stage):
        self.plan = json.loads(plan_path.read_text())
        self.design = json.loads(design_path.read_text())
        self.rental = json.loads(rental_path.read_text())
        if os.environ.get("CONTAINER_ID") != str(self.rental["instance_id"]):
            raise ValueError("This queue must run on its recorded rental")
        for name, expected in self.plan["source_sha256"].items():
            if sha256(ROOT / name) != expected:
                raise ValueError(f"Source changed after the protocol freeze: {name}")
        if self.plan["source_sha256"].get(str(design_path.relative_to(ROOT))) != sha256(
            design_path
        ):
            raise ValueError("The design is not part of this freeze")
        OUT.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.queue_lock = (OUT / "queue.lock").open("a")
        fcntl.flock(self.queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.history = []
        for path in sorted(OUT.glob("attempts_*.jsonl")):
            self.history.extend(json.loads(x) for x in path.read_text().splitlines() if x)
        self.ledger = (OUT / f"attempts_{stage}.jsonl").open("a", buffering=1)
        self.starts = sum(e["event"] == "start_reserved" for e in self.history)
        self.replacements = sum(
            e["event"] == "start_reserved" and e["attempt"] > 1 for e in self.history
        )
        self.jobs = self.stage_jobs()
        self.admission = (
            dt.datetime.fromisoformat(self.rental["allocation_deadline_utc"]).timestamp() - 2700
        )
        self.record("queue_start", stage=stage, plan_sha256=sha256(plan_path),
                    design_sha256=sha256(design_path), instance=self.rental["instance_id"])

    def stage_jobs(self):
        design = self.design
        if self.stage == "stage_a":
            return design["stage_a_jobs"]
        validated = {e.get("job") for e in self.history if e["event"] == "validated"}
        if not all(j["job"] in validated for j in design["stage_a_jobs"]):
            raise ValueError("Every stage A episode must validate first")
        passed = json.loads((OUT / "stage-a-pass.json").read_text())
        if not passed.get("pass"):
            raise ValueError("Stage A did not pass its engineering rules")
        if self.stage == "stage_b":
            return design["stage_b_jobs"]
        decision = json.loads((OUT / "stage-c-decision.json").read_text())
        if decision.get("design_sha256") != sha256(ROOT / "configs/regrasp15/design.json"):
            raise ValueError("Stage C decision is not bound to this design")
        for name, expected in decision["evidence_sha256"].items():
            if sha256(OUT / name) != expected:
                raise ValueError("Stage B evidence changed after the stage C decision")
        arms = ["native", decision["arm"]]
        jobs, number = [], 300
        for seed in design["stage_c_seeds"]:
            for task in design["stage_c_tasks"]:
                for arm in (arms if (seed + task) % 2 == 0 else arms[::-1]):
                    jobs.append({"job": number, "stage": "stage_c", "arm": arm,
                                 "task": task, "seed": seed})
                    number += 1
        return jobs

    def record(self, event, **fields):
        row = {"utc": utc(), "event": event, **fields}
        self.ledger.write(json.dumps(row) + "\n")
        self.ledger.flush()
        os.fsync(self.ledger.fileno())
        self.history.append(row)
        print(json.dumps(row), flush=True)

    def decodes(self):
        if not SERVER_LEDGER.exists():
            return 0
        return sum(json.loads(x)["k"] for x in SERVER_LEDGER.read_text().splitlines() if x)

    def admit(self, job, attempt):
        ceilings = self.design["ceilings"]
        worst = self.design["k"] * 250
        if (time.time() >= self.admission or (OUT / "STOP").exists()
                or self.starts >= ceilings["simulator_starts"]
                or self.decodes() + worst > ceilings["candidate_decodes"]):
            return False
        if attempt > 1:
            if self.replacements >= ceilings["replacements"]:
                return False
            self.replacements += 1
        self.starts += 1
        self.record("start_reserved", **job, attempt=attempt, start_number=self.starts)
        return True

    def run(self, job, attempt):
        leaf = f"job{job['job']}_{job['arm']}_task{job['task']:02d}_seed{job['seed']}_a{attempt}"
        directory = OUT / "episodes" / leaf
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "job.json").write_text(json.dumps({**job, "attempt": attempt}, indent=2))
        # A different episode_seed makes the native server reset its sampler.
        with (directory / "reseed.log").open("w") as log:
            probe = subprocess.run(
                [str(ROOT / "third_party/openpi/.venv/bin/python"), "-I",
                 str(ROOT / "experiments/scripts/probe_regrasp15.py"), "--reseed-only"],
                stdout=log, stderr=subprocess.STDOUT, timeout=600,
            )
        self.record("reseed_probe", job=job["job"], attempt=attempt, rc=probe.returncode)
        if probe.returncode:
            raise RuntimeError("Re-seeding probe failed; queue halted")
        settings = self.design["arms"][job["arm"]]
        config_path = directory / "adapter-config.json"
        config_path.write_text(json.dumps(settings, indent=2) + "\n")
        env = dict(os.environ)
        env.update({
            "CUDA_VISIBLE_DEVICES": "0",
            "OMNIGIBSON_HEADLESS": "1",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": f"{ROOT}/experiments/helpers:{ROOT}/scripts/todo2",
            "SAFETY_DIAL_REGRASP_CONFIG": str(config_path),
            "SAFETY_DIAL_REGRASP_JOURNAL": str(directory / "adapter.jsonl"),
        })
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = (
            str(ROOT / "scripts/todo1b/nvidia-egl-vendor.json")
            if Path("/opt/nvidia-drivers/lib64/libEGL_nvidia.so.0").exists()
            else "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
        )
        env.pop("EVAL_USE_GPU_DYNAMICS", None)
        cmd = [
            sys.executable, "-m", "maniguard.eval.benchmark",
            "--config", str(ROOT / "configs/regrasp15/jar_eval.yaml"),
            "--host", "127.0.0.1", "--port", "8000",
            "--benchmark-root", "outputs/lerobot_datasets/maniguard-bench/jar_transport",
            "--scenes", f"task_{job['task']:04d}/base", "--max-scenes", "1",
            "--output-dir", str(directory), "--run-name", "eval",
            "--metrics", "success", "safety", "--headless", "--seed", str(job["seed"]),
            "--recording-factory", "accounted_recorder:create_observer",
            "--recording-output-dir", str(directory / "recordings"),
        ]
        (directory / "command.json").write_text(json.dumps(cmd, indent=2))
        self.record("simulator_launch", **job, attempt=attempt, leaf=leaf)
        started = time.monotonic()
        with (directory / "evaluator.log").open("w") as log:
            process = subprocess.Popen(cmd, cwd=ROOT / "third_party/ManiGuard", env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                rc = process.wait(timeout=self.design["episode_timeout_seconds"])
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
                    completed_row=complete, leaf=leaf)
        if not complete:
            return False
        recordings = list(directory.glob("recordings/*_base.npz"))
        sidecars = list(directory.rglob("*_ltl.json"))
        if len(recordings) != 1 or len(sidecars) != 1:
            raise ValueError("Completed episode is missing a unique recording or sidecar")
        result = validate_recording(recordings[0], rows[0], sidecars[0],
                                    directory / "adapter.jsonl", settings)
        if (result["row"]["scene_name"] != f"task_{job['task']:04d}/base"
                or result["row"]["seed"] != job["seed"]):
            raise ValueError("Completed episode does not match its frozen job")
        (directory / "validation.json").write_text(json.dumps(result, indent=2))
        self.record("validated", **job, attempt=attempt, leaf=leaf, elapsed_s=elapsed,
                    outcome=result["outcome"], adapter=result["adapter"])
        return True

    def drain(self):
        validated = {e.get("job") for e in self.history if e["event"] == "validated"}
        failures = 0
        for job in self.jobs:
            if job["job"] in validated:
                continue
            previous = [e["attempt"] for e in self.history
                        if e["event"] == "start_reserved" and e.get("job") == job["job"]]
            first = max(previous, default=0) + 1
            for attempt in range(first, self.design["ceilings"]["attempts_per_job"] + 1):
                if not self.admit(job, attempt):
                    self.record("admission_closed", job=job["job"])
                    return False
                if self.run(job, attempt):
                    failures = 0
                    break
                failures += 1
                if failures >= 2:
                    raise RuntimeError("Two consecutive incomplete attempts; queue halted")
            else:
                self.record("job_abandoned", job=job["job"])
        return True


def stage_a_rules(queue):
    """Engineering pass rules, beyond per-episode validation."""
    validated = [e for e in queue.history if e["event"] == "validated"
                 and e.get("stage") == "stage_a"]
    requests = sum(e["adapter"]["closest_passing"] + e["adapter"]["least_excess"]
                   for e in validated)
    projections = sum(e["adapter"]["projected"] for e in validated)
    rules = {
        "all_validated": {e["job"] for e in validated}
        == {j["job"] for j in queue.design["stage_a_jobs"]},
        "candidate_requests_returned": requests > 0,
        "projection_exercised": projections > 0,
    }
    return {"utc": utc(), "rules": rules, "candidate_requests": requests,
            "projections": projections, "pass": all(rules.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=ROOT / "configs/regrasp15/design.json")
    parser.add_argument("--rental", type=Path,
                        default=ROOT / "progress/rental-regrasp15-20261025.json")
    parser.add_argument("--stage", choices=["stage_a", "stage_b", "stage_c"], required=True)
    args = parser.parse_args()
    queue = Queue(args.plan.resolve(), args.design.resolve(), args.rental.resolve(), args.stage)
    try:
        finished = queue.drain()
    except Exception as error:
        (OUT / "STOP").write_text(f"{args.stage}: {type(error).__name__}\n")
        queue.record("queue_error", kind=type(error).__name__, detail=str(error))
        raise
    queue.record("queue_finished", stage=args.stage, starts=queue.starts,
                 replacements=queue.replacements, finished=finished)
    if args.stage == "stage_a":
        rules = stage_a_rules(queue)
        (OUT / "stage-a-pass.json").write_text(json.dumps(rules, indent=2) + "\n")
        if not rules["pass"]:
            raise SystemExit("Stage A failed its engineering rules")
    validated = {e.get("job") for e in queue.history if e["event"] == "validated"}
    missing = [j["job"] for j in queue.jobs if j["job"] not in validated]
    (OUT / f"{args.stage}-complete.json").write_text(json.dumps(
        {"utc": utc(), "missing_jobs": missing, "finished": finished}, indent=2) + "\n")
    if missing and args.stage == "stage_a":
        raise SystemExit(f"Stage A incomplete: {len(missing)} jobs have no valid recording")


if __name__ == "__main__":
    main()
