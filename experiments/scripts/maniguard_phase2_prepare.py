"""Execute explicitly assigned preparation cases with durable attempt and cost records.

Run this entry point under supervisor. A claimed case is never automatically retried,
including after a coordinator crash. Policy-family switching is external to this runner;
the episode handshake checks the actual checkpoint before any physics action.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_phase2_model import genome_hash, identity_genome, load_model


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_preparation(path):
    protocol = json.loads(path.read_text())
    raw = {key: value for key, value in protocol.items() if key != "sha256"}
    digest = hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode())
    if digest.hexdigest() != protocol["sha256"]:
        raise ValueError("Preparation manifest checksum mismatch")
    if protocol["status"] != "assignments_frozen_before_new_behavior":
        raise ValueError("Freeze preparation assignments before execution")
    if len(protocol["short_jobs"]) != 12 or len(protocol["full_jobs"]) != 24:
        raise ValueError("Preparation allocation differs from the declared ceiling")
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "configs/maniguard/phase2/preparation.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/maniguard/phase2/preparation-v1")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "data/maniguard/phase2/models/identity.npz")
    args = parser.parse_args()
    protocol = load_preparation(args.protocol)
    mapping = {row["job_id"]: row for row in protocol["short_jobs"] + protocol["full_jobs"]}
    if len(args.jobs) != len(set(args.jobs)) or not set(args.jobs) <= set(mapping):
        raise ValueError("Select distinct, predeclared preparation jobs")
    jobs = [mapping[key] for key in args.jobs]
    if len({row["family"] for row in jobs}) != 1:
        raise ValueError("One preparation invocation uses one verified policy family")
    if genome_hash(load_model(args.model)[0]) != genome_hash(identity_genome()):
        raise ValueError("Preparation uses the explicitly declared identity weights")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipts = output / "attempts"
    receipts.mkdir(exist_ok=True)
    snapshot = output / "preparation.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != protocol:
        raise ValueError("Existing preparation records use a different protocol")
    if not snapshot.exists():
        atomic_json(snapshot, protocol)
    for job in jobs:
        receipt_path = receipts / (job["job_id"] + ".json")
        if receipt_path.exists():
            raise ValueError("Already claimed; inspect authoritative process state before recovery: "
                             + job["job_id"])
        short = job["max_steps"] is not None
        if not short:
            required = [receipts / (row["job_id"] + ".json")
                        for row in protocol["short_jobs"]]
            if any(not path.exists() or json.loads(path.read_text()).get("status") != "valid"
                   for path in required):
                raise ValueError("Complete and audit assigned short integration checks first")
        timeout = protocol["budget"]["short_process_timeout_seconds" if short else
                                     "full_process_timeout_seconds"]
        hour_limit = protocol["budget"]["integration_gpu_hour_ceiling" if short else
                                        "full_preparation_gpu_hour_ceiling"]
        prior = [json.loads(path.read_text()) for path in receipts.glob("*.json")]
        relevant = [row for row in prior if bool(row["short_check"]) == short]
        if any(row.get("status") == "started" for row in prior):
            raise ValueError("An earlier attempt is unresolved; do not duplicate live work")
        spent = sum(row.get("elapsed_seconds", row["timeout_seconds"]) for row in relevant)
        if spent + timeout > hour_limit * 3600:
            raise ValueError("Insufficient remaining preparation time ceiling for another attempt")
        episode = output / job["job_id"]
        command = [sys.executable, "-u", str(ROOT / "experiments/scripts/maniguard_phase2_episode.py"),
                   "--family", job["family"], "--scene", job["scene"],
                   "--purpose", job["purpose"], "--seed", str(job["seed"]),
                   "--sampling-seed", str(job["sampling_seed"]), "--mode", job["mode"],
                   "--model", str(args.model.resolve()), "--output", str(episode)]
        if short:
            command += ["--max-steps", str(job["max_steps"])]
        if job["fixed_choices"] is not None:
            command += ["--fixed-choices", *map(str, job["fixed_choices"])]
        if job["stochastic"]:
            command.append("--stochastic")
        if job["baseline_executor"]:
            command.append("--baseline-executor")
        record = {"job": job, "short_check": short, "status": "started", "command": command,
                  "protocol_sha256": protocol["sha256"], "timeout_seconds": timeout,
                  "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "coordinator_pid": os.getpid(), "output": str(episode),
                  "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        # Exclusive claim is durable before spawning physics work.
        with receipt_path.open("x") as stream:
            stream.write(json.dumps(record, indent=2) + "\n")
        started = time.monotonic()
        print("Starting " + job["job_id"], flush=True)
        environment = dict(os.environ, OMNI_KIT_ALLOW_ROOT="1", OMP_NUM_THREADS="4",
                           OPENBLAS_NUM_THREADS="4")
        with (output / (job["job_id"] + ".log")).open("x") as log:
            try:
                completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, timeout=timeout, check=False)
                record["returncode"] = completed.returncode
            except (OSError, subprocess.TimeoutExpired) as error:
                record["returncode"] = None
                record["process_error"] = str(error)
        record["elapsed_seconds"] = time.monotonic() - started
        record["status"] = "invalid"
        try:
            if job["baseline_executor"]:
                from helpers.maniguard_results import read_episode
                row = read_episode(episode)
                row["valid"] = True
            else:
                from helpers.maniguard_phase2_results import read_episode
                row = read_episode(episode)
            record["evidence"] = row
            if record["returncode"] == 0 and row["valid"]:
                record["status"] = "valid"
        except (ValueError, OSError, KeyError, TypeError) as error:
            record["audit_error"] = {"type": type(error).__name__, "message": str(error)}
        finally:
            if (episode / "ledger.json").exists():
                record["ledger"] = json.loads((episode / "ledger.json").read_text())
            atomic_json(receipt_path, record)
        print(json.dumps({"job": job["job_id"], "status": record["status"],
                          "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
        if record["status"] != "valid":
            raise RuntimeError("Preparation failed; attempt/costs preserved and queue stopped")


if __name__ == "__main__":
    main()
