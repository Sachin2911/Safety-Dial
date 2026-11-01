"""Serial ManiGuard episode backend with durable reservations and measured work."""

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

from maniguard_evolution_scoring import analyze_episode


class BudgetExhausted(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def totals(root):
    result = {"reserved_starts": 0, "valid_episodes": 0, "invalid_starts": 0,
              "incomplete_reserved_starts": 0,
              "model_query_attempts": 0, "model_query_returns": 0,
              "environment_step_attempts": 0, "environment_step_returns": 0,
              "simulator_step_attempts": 0, "simulator_step_returns": 0,
              "reset_attempts": 0, "reset_returns": 0, "step_physics_attempts": 0,
              "step_physics_returns": 0, "physics_callbacks": 0,
              "warmup_attempts": 0, "warmup_returns": 0, "unknown_bootstrap_starts": 0,
              "model_query_seconds": 0.0}
    for path in Path(root).glob("episodes/*/reservation.json"):
        result["reserved_starts"] += 1
        job = path.parent
        if (job / "outcome.json").exists():
            valid = json.loads((job / "outcome.json").read_text()).get("valid", False)
            result["valid_episodes" if valid else "invalid_starts"] += 1
        else:
            result["incomplete_reserved_starts"] += 1
        seed = job / "preseed-query.json"
        counts = job / "counters.json"
        if seed.exists():
            value = json.loads(seed.read_text())
            result["model_query_attempts"] += value["attempted"]
            result["model_query_returns"] += value["completed"]
            result["model_query_seconds"] += value.get("seconds", 0.0)
        if counts.exists():
            value = json.loads(counts.read_text())
            for key in result:
                if key in value:
                    result[key] += value[key]
            result["model_query_attempts"] += value["query_attempts"]
            result["model_query_returns"] += value["query_returns"]
            result["model_query_seconds"] += value.get("query_seconds", 0.0)
        # Initial simulator bootstrap precedes the installed counters, even on failure.
        result["unknown_bootstrap_starts"] += int((job / "process.json").exists())
    result["bootstrap_physics"] = "Unknown before accounting installation; never assumed zero"
    config_path = Path(root) / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        preparation_path = config.get("server_preparation_accounting")
        if preparation_path and Path(preparation_path).exists():
            preparation = json.loads(Path(preparation_path).read_text())
            if preparation.get("attempt_directory"):
                attempts = [json.loads(p.read_text()) for p in Path(
                    preparation["attempt_directory"]).glob("*/preseed-query.json")]
                preparation.update(
                    model_query_attempts=sum(row["attempted"] for row in attempts),
                    model_query_returns=sum(row["completed"] for row in attempts),
                    model_query_seconds=sum(row.get("seconds", 0) for row in attempts))
            result["server_preparation_accounting"] = preparation
            for field in ("model_query_attempts", "model_query_returns", "model_query_seconds"):
                result[field] += preparation.get(field, 0)
        predecessor = config.get("engineering_predecessor")
        if predecessor:
            previous_path = Path(predecessor["run"]) / "accounting.json"
            if previous_path.exists():
                previous = json.loads(previous_path.read_text())
                result["predecessor_accounting"] = previous
                charged = previous.get("aggregate_with_previous_versions", previous)
                result["aggregate_with_previous_versions"] = {
                    key: value + charged.get(key, 0) for key, value in result.items()
                    if isinstance(value, (int, float))}
    return result


def reset_sampler(job, host, port):
    """One charged synthetic query forces a distinct episode reseed on this server."""
    import numpy as np
    from openpi_client import msgpack_numpy
    import websockets.sync.client

    report = {"attempted": 0, "completed": 0, "synthetic": True, "episode_seed": 0}
    path = job / "preseed-query.json"
    atomic_json(path, report)
    with websockets.sync.client.connect(
        f"ws://{host}:{port}", compression=None, max_size=None,
        ping_interval=None, open_timeout=10,
    ) as client:
        metadata = msgpack_numpy.unpackb(client.recv(timeout=15))
        if metadata["serve_config"] != "pi05-base_datagen_v1_jar_joint_2cam_lora":
            raise ValueError("Unexpected frozen policy server")
        observation = {
            "observation/image_left": np.zeros((256, 256, 3), np.uint8),
            "observation/wrist_image": np.zeros((256, 256, 3), np.uint8),
            "observation/state": np.zeros(8, np.float32),
            "prompt": "Close the lid, then carry the closed jar to the goal.", "episode_seed": 0,
        }
        report["attempted"] = 1
        atomic_json(path, report)
        start = time.monotonic()
        try:
            client.send(msgpack_numpy.Packer().pack(observation))
            result = msgpack_numpy.unpackb(client.recv(timeout=120))
            actions = np.asarray(result["actions"])
            if actions.shape != (16, 8) or not np.isfinite(actions).all():
                raise ValueError("Invalid seed-reset inference")
            report["completed"] = 1
        finally:
            report["seconds"] = time.monotonic() - start
            atomic_json(path, report)


class ManiGuardEvaluator:
    def __init__(self, root, config, generated, package_root):
        self.root, self.config = Path(root), config
        self.generated, self.package_root = Path(generated), Path(package_root)
        self.end = dt.datetime.fromisoformat(config["limits"]["deadline_utc"]).timestamp()
        self.end -= config["limits"]["shutdown_reserve_seconds"]
        (self.root / "episodes").mkdir(exist_ok=True)

    def admission(self):
        limits = self.config["limits"]
        counts = totals(self.root)
        preparation = counts.get("server_preparation_accounting", {})
        if preparation.get("model_query_attempts", 0) > limits.get(
                "max_server_preparation_model_query_attempts", float("inf")):
            raise BudgetExhausted("Server-preparation query allowance exceeded")
        counts = counts.get("aggregate_with_previous_versions", counts)
        if counts.get("invalid_starts", 0) >= limits.get("max_failed_starts", float("inf")):
            raise BudgetExhausted("Failed engineering-start allowance exhausted")
        if counts["reserved_starts"] >= limits["max_starts"]:
            raise BudgetExhausted("Episode allowance exhausted")
        if counts["environment_step_attempts"] + 2010 > limits["max_environment_step_attempts"]:
            raise BudgetExhausted("Insufficient remaining environment-step allowance")
        if counts["model_query_attempts"] + 251 > limits["max_model_query_attempts"]:
            raise BudgetExhausted("Insufficient remaining model-query allowance")
        if time.time() + limits["minimum_start_window_seconds"] >= self.end:
            raise BudgetExhausted("Insufficient time before the existing shutdown deadline")
        minimum = limits.get("minimum_remaining_seconds_before_first_episode", 0)
        rental_end = self.end + limits["shutdown_reserve_seconds"]
        if not counts.get("valid_episodes") and time.time() + minimum > rental_end:
            raise BudgetExhausted("Seven-hour first-scored-episode launch window has closed")

    def verify_sources(self):
        manifest = json.loads((self.root / "source-manifest.json").read_text())
        for name, expected in manifest["sources"].items():
            path = (Path(self.config["search"]["initial_controller"])
                    if name == "initial_controller" else self.package_root / name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Run source changed during execution: " + name)
        generated = json.loads((self.generated / "evaluator-source.json").read_text())
        if hashlib.sha256((self.generated / "benchmark_adapter.py").read_bytes()).hexdigest() != (
                generated["generated_sha256"]):
            raise ValueError("Generated evaluator changed during execution")

    def __call__(self, label, task, policy_seed, artifact):
        import zlib

        if not zlib.crc32(f"{policy_seed}:task_{task:04d}/base".encode()) & 0x7FFFFFFF:
            raise ValueError("Episode seed collides with sampler-reset query")
        job = self.root / "episodes" / f"{label}_t{task:02d}_s{policy_seed}"
        expected = {"label": label, "task": task, "policy_seed": policy_seed,
                    "controller": artifact}
        outcome = job / "outcome.json"
        if job.exists():
            reserved = json.loads((job / "reservation.json").read_text())
            if reserved["specification"] != expected:
                raise ValueError("Resume would change a reserved episode")
            if outcome.exists():
                return json.loads(outcome.read_text())
            # Recover saved output without ever silently rerunning a reserved episode.
            process_path = job / "process.json"
            if process_path.exists():
                recorded = json.loads(process_path.read_text())
                proc = Path('/proc') / str(recorded["pid"])
                while proc.exists() and time.time() < self.end:
                    try:
                        stat = (proc / "stat").read_text().split()
                    except FileNotFoundError:
                        break
                    if stat[21] != recorded["start_ticks"] or stat[2] == "Z":
                        break
                    time.sleep(2)
            try:
                report = analyze_episode(job)
                report.update(job=job.name, recovered=True)
            except Exception as error:
                report = {"valid": False, "job": job.name, "error_type": type(error).__name__,
                          "error": str(error), "interrupted_reservation": True}
            atomic_json(outcome, report)
            return report
        self.admission()
        self.verify_sources()
        job.mkdir()
        atomic_json(job / "reservation.json", {"reserved_utc": utc(), "specification": expected})
        atomic_json(job / "controller.json", artifact)
        base = dict(self.config["evaluator"])
        base.update(seed=policy_seed, scenes=[f"task_{task:04d}/base"],
                    output_dir=str(job / "eval"), run_name="run",
                    recording_factory="maniguard_adapter_runtime:create_observer",
                    recording_output_dir=str(job / "recordings"))
        (job / "eval.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
        environment = dict(os.environ)
        environment.update(self.config.get("environment", {}))
        environment.update(MANIGUARD_ADAPTER_JOB=str(job),
                           MANIGUARD_ADAPTER_END_EPOCH=str(self.end))
        paths = [self.package_root / "experiments/helpers", self.package_root / "scripts/todo2"]
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(path) for path in paths] + [environment.get("PYTHONPATH", "")])
        command = [sys.executable, "-u", str(self.generated / "benchmark_adapter.py"),
                   "--config", str(job / "eval.yaml")]
        atomic_json(job / "command.json", {"argv": command})
        start = time.monotonic()
        child = None
        try:
            reset_sampler(job, base["host"], base["port"])
            with (job / "stdout.log").open("wb") as stream:
                child = subprocess.Popen(command, cwd=self.config["maniguard_checkout"],
                                         env=environment, stdout=stream, stderr=stream,
                                         start_new_session=True)
                ticks = (Path('/proc') / str(child.pid) / "stat").read_text().split()[21]
                atomic_json(job / "process.json", {"pid": child.pid, "start_ticks": ticks})
                code = child.wait(timeout=max(1, min(
                    self.end - time.time(), self.config["limits"]["episode_timeout_seconds"])))
            if code:
                raise RuntimeError(f"Simulator process exited with code {code}")
            report = analyze_episode(job)
        except BaseException as error:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            report = {"valid": False, "job": job.name, "error_type": type(error).__name__,
                      "error": str(error)[:800]}
            atomic_json(outcome, report)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
        report.update(job=job.name, wall_seconds=time.monotonic() - start)
        atomic_json(outcome, report)
        atomic_json(self.root / "accounting.json", totals(self.root))
        return report
