"""Bounded new probe runner. Frozen inputs, eight starts maximum, no replacements."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zlib

import yaml


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preseed(job):
    """One counted synthetic query ensures the next episode seed is a new seed."""
    import numpy as np
    from openpi_client import msgpack_numpy
    import websockets.sync.client

    record = {"attempted": 0, "completed": 0, "synthetic": True, "episode_seed": 0}
    path = job / "preseed-query.json"
    atomic(path, record)
    with websockets.sync.client.connect(
        "ws://127.0.0.1:8000", compression=None, max_size=None,
        ping_interval=None, open_timeout=10,
    ) as client:
        metadata = msgpack_numpy.unpackb(client.recv(timeout=15))
        if metadata["serve_config"] != "pi05-base_datagen_v1_jar_joint_2cam_lora":
            raise ValueError("Unexpected policy server")
        observation = {
            "observation/image_left": np.zeros((256, 256, 3), np.uint8),
            "observation/wrist_image": np.zeros((256, 256, 3), np.uint8),
            "observation/state": np.zeros(8, np.float32),
            "prompt": "Close the lid, then carry the closed jar to the goal.", "episode_seed": 0,
        }
        record["attempted"] = 1
        atomic(path, record)
        started = time.monotonic()
        client.send(msgpack_numpy.Packer().pack(observation))
        result = msgpack_numpy.unpackb(client.recv(timeout=120))
        actions = np.asarray(result["actions"])
        if actions.shape != (16, 8) or not np.isfinite(actions).all():
            raise ValueError("Invalid sampler-reset inference")
        record.update({"completed": 1, "seconds": time.monotonic() - started})
        atomic(path, record)


def validate_freeze(package, freeze):
    for relative, digest in freeze["files"].items():
        if sha(package / relative) != digest:
            raise ValueError("Changed frozen input: " + relative)
    benchmark = Path(freeze["benchmark_path"])
    if sha(benchmark) != freeze["patched_benchmark_sha256"]:
        raise ValueError("Changed patched evaluator")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--remaining-engineering", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    freeze = json.loads((package / "engineering-freeze.json").read_text())
    validate_freeze(package, freeze)
    plan_path = package / "plan.json"
    plan = json.loads(plan_path.read_text())
    root = Path(plan["remote_output_path"])
    previous = None
    if args.remaining_engineering:
        previous = json.loads((root / "status.json").read_text())
        if not (
            previous["status"] == "closed_at_gate_or_limit"
            and previous["reserved"] == 1 and not previous["jobs"]
            and previous["error"] == "Simulator process exited with code 1"
        ):
            raise ValueError("Remaining-slot admission is not valid for this terminal state")
        original_job = root / previous["active"]["name"]
        rows = list((original_job / "eval").rglob("results.jsonl"))
        outcome = json.loads(rows[0].read_text().strip())
        if outcome["status"] != "load_failed" or outcome["steps"] != 0:
            raise ValueError("Correction is restricted to the zero-control configuration failure")
        if "must be equal to rendering dt" not in outcome["error"]:
            raise ValueError("Unexpected first engineering failure")
        if (root / "scored-freeze.json").exists():
            raise ValueError("Cannot repair engineering after scored admission")
        atomic(root / "initial-engineering-terminal.json", previous)
    else:
        root.mkdir(parents=True, exist_ok=False)
    admitted = previous["admitted_epoch"] if previous else time.time()
    end = min(
        admitted + plan["limits"]["max_wall_minutes_after_admission"] * 60,
        dt.datetime.fromisoformat(plan["limits"]["rental_deadline_utc"]).timestamp() - 60,
    )
    engineering = [
        {"stage": "engineering", "task": 1, "policy_seed": seed, "arm": "native"}
        for seed in plan["engineering"]["policy_seeds"]
    ]
    scored = [{"stage": "scored", **job} for job in plan["schedule"]]
    ledger = root / "ledger.jsonl"
    status = {"status": "running", "pid": os.getpid(), "admitted_utc":
              (previous["admitted_utc"] if previous else dt.datetime.now(dt.timezone.utc).isoformat()), "admitted_epoch": admitted,
              "hard_end_epoch": end, "reserved": 0, "jobs": [],
              "engineering_freeze_sha256": sha(package / "engineering-freeze.json")}
    if previous:
        status["reserved"] = 1
        status["configuration_repair"] = json.loads((package / "engineering-correction.json").read_text())
        status["failed_engineering_starts"] = 1
    atomic(root / "status.json", status)
    with ledger.open("a" if previous else "x", buffering=1) as journal:
        def log(event, **fields):
            journal.write(json.dumps({"event": event, "time": time.time(), **fields}) + "\n")

        try:
            for index, spec in enumerate(engineering + scored):
                if previous and index == 0:
                    continue
                validate_freeze(package, freeze)
                if time.time() >= end:
                    raise TimeoutError("Fixed experiment admission window ended")
                if index == len(engineering):
                    scored_freeze = {"frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                     "engineering_validation_sha256": {
                                         row["name"]: sha(root / row["name"] / "validation.json")
                                         for row in status["jobs"]},
                                     "source_freeze_sha256": sha(package / "engineering-freeze.json"),
                                     "schedule": scored}
                    atomic(root / "scored-freeze.json", scored_freeze)
                    log("scored_stage_frozen", sha256=sha(root / "scored-freeze.json"))
                name = f"job{index:02d}_{spec['stage']}_{spec['arm']}_s{spec['policy_seed']}"
                job = root / name
                job.mkdir()
                atomic(job / "job.json", spec)
                if not (zlib.crc32(
                    f"{spec['policy_seed']}:task_{spec['task']:04d}/base".encode()
                ) & 0x7FFFFFFF):
                    raise ValueError("Episode seed collides with the preseed query")
                status["reserved"] += 1
                atomic(root / "status.json", status)
                log("start_reserved", name=name, specification=spec)
                preseed(job)
                base = dict(plan["evaluator_config"])
                base.update({"seed": spec["policy_seed"], "scenes": [f"task_{spec['task']:04d}/base"],
                             "max_steps": 32 if spec["stage"] == "engineering" else 2000,
                             "output_dir": str(job / "eval"), "run_name": "run",
                             "recording_factory": "jar_grasp_probe:create_observer",
                             "recording_output_dir": str(job / "recordings")})
                config = job / "eval.yaml"
                config.write_text(yaml.safe_dump(base, sort_keys=False))
                environment = dict(os.environ)
                environment.update({"JAR_GRASP_PROBE_PLAN": str(plan_path),
                                    "JAR_GRASP_PROBE_JOB": str(job),
                                    "JAR_GRASP_PROBE_ARM": spec["arm"]})
                command = [sys.executable, "-u", "-m", "maniguard.eval.benchmark", "--config", str(config)]
                atomic(job / "command.json", {"argv": command})
                with (job / "stdout.log").open("wb") as output:
                    child = subprocess.Popen(command, env=environment, stdout=output,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    log("simulator_process_started", name=name, pid=child.pid)
                    status["active"] = {"name": name, "pid": child.pid, **spec}
                    atomic(root / "status.json", status)
                    timeout = min(end - time.time(), 900)
                    try:
                        code = child.wait(timeout=max(1, timeout))
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        log("time_cut", name=name)
                        raise TimeoutError("Bounded episode/process time limit reached") from None
                log("simulator_process_exited", name=name, code=code)
                if code:
                    raise RuntimeError("Simulator process exited with code " + str(code))
                from analyze_jar_grasp_probe import analyze

                validation = analyze(job, plan)
                if spec["stage"] == "engineering":
                    if validation["controls"] != 32 or validation["overridden_controls"]:
                        raise ValueError("Identity engineering gate failed")
                status["jobs"].append({"name": name, **spec, "valid": True})
                status.pop("active", None)
                atomic(root / "status.json", status)
                log("validated", name=name, primary=validation["primary"],
                    violation=validation["raw_violation"], overrides=validation["overridden_controls"])
                # Every budget includes startup/reset work and the synthetic preseed queries.
                reports = [json.loads((root / item["name"] / "validation.json").read_text())
                           for item in status["jobs"]]
                controls = sum(item["controls"] for item in reports)
                queries = 0
                for reserved_job in root.glob("job*"):
                    counter = reserved_job / "counters.json"
                    seed_query = reserved_job / "preseed-query.json"
                    if counter.exists():
                        queries += json.loads(counter.read_text())["query_attempts"]
                    if seed_query.exists():
                        queries += json.loads(seed_query.read_text())["attempted"]
                if controls > 12064 or queries > 2000:
                    raise RuntimeError("Fixed interaction allowance exceeded")
            status["status"] = "completed"
        except Exception as error:
            status["status"] = "closed_at_gate_or_limit"
            status["error_type"] = type(error).__name__
            status["error"] = str(error)[:400]
            log("closed", error_type=type(error).__name__, message=str(error)[:400])
        finally:
            status["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            status["wall_seconds"] = time.time() - admitted
            atomic(root / "status.json", status)
            log("pipeline_finished", status=status["status"])
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
