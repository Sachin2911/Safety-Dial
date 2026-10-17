"""Execute explicitly assigned preparation cases with durable attempt and cost records.

Run this entry point under supervisor. A claimed case is never automatically retried,
including after a coordinator crash. Policy-family switching is external to this runner;
the episode handshake checks the actual checkpoint before any physics action.
"""
import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_phase2_model import genome_hash, identity_genome, load_model
from helpers.maniguard_phase2_services import (
    active_episode_processes,
    atomic_json,
    execution_lease,
    run_fresh_episode,
)


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



def process_succeeded(record):
    process = record.get("process", {})
    return (record.get("returncode") == 0 and not record.get("process_error") and
            not any(process.get(key) for key in (
                "error", "cleanup_error", "timed_out", "interrupted", "orphaned_descendants",
                "process_group_remaining")))


def audit_attempt(record):
    if record["job"]["baseline_executor"]:
        from helpers.maniguard_results import read_episode
        row = read_episode(Path(record["output"]))
        row["valid"] = True
    else:
        from helpers.maniguard_phase2_results import read_episode
        row = read_episode(Path(record["output"]))
    return row


def reaudit_attempt(path, job, protocol_sha256):
    record = json.loads(path.read_text())
    if (record.get("status") not in ("valid", "invalid") or "returncode" not in record
            or record.get("job") != job or record.get("protocol_sha256") != protocol_sha256):
        raise ValueError("Reaudit requires the matching terminal attempt; no unresolved processes")
    before = {key: record.get(key) for key in ("status", "evidence", "audit_error")}
    before["receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record.setdefault("audit_history", []).append(before)
    started = time.perf_counter()
    record["status"] = "invalid"
    record.pop("audit_error", None)
    try:
        row = audit_attempt(record)
        record["evidence"] = row
        if process_succeeded(record) and row["valid"]:
            record["status"] = "valid"
    except (ValueError, OSError, KeyError, TypeError) as error:
        record["audit_error"] = {"type": type(error).__name__, "message": str(error)}
    record["reaudit"] = {
        "elapsed_seconds": time.perf_counter() - started,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "physics_rerun": False,
        "source_hashes": {str(source.relative_to(ROOT)): hashlib.sha256(source.read_bytes()).hexdigest()
                          for source in (ROOT / "experiments/helpers").glob("maniguard_phase2_*.py")},
    }
    atomic_json(path, record)
    return record


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--reaudit", action="store_true",
                        help="Reanalyse terminal saved attempts; preserve prior verdicts and do no physics")
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "configs/maniguard/phase2/preparation.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "runs/maniguard/phase2/preparation-v1")
    parser.add_argument("--reservation", type=Path,
                        default=ROOT / "runs/maniguard/phase2/setup/reservation.json")
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
        if args.reaudit:
            record = reaudit_attempt(receipt_path, job, protocol["sha256"])
            print(json.dumps({"job": job["job_id"], "status": record["status"],
                              "physics_rerun": False}), flush=True)
            if record["status"] != "valid":
                raise RuntimeError("Saved attempt remains invalid after reaudit")
            continue
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
        reservation = json.loads(args.reservation.read_text())
        if reservation.get("allocated_gpus") != 1:
            raise ValueError("This preparation plan budgets one allocated GPU")
        allocated_seconds = time.time() - reservation["start_unix_seconds"]
        global_hours = protocol["budget"]["integration_gpu_hour_ceiling"]
        if not short:
            global_hours += protocol["budget"]["full_preparation_gpu_hour_ceiling"]
            if relevant:
                import calendar
                first_full = min(calendar.timegm(time.strptime(row["started_utc"],
                                 "%Y-%m-%dT%H:%M:%SZ")) for row in relevant)
                if time.time() - first_full + timeout > hour_limit * 3600:
                    raise ValueError("Full preparation reservation ceiling would be exceeded")
        if allocated_seconds < 0 or allocated_seconds + timeout > global_hours * 3600:
            raise ValueError("Preparation GPU reservation ceiling would be exceeded")
        if active_episode_processes():
            raise RuntimeError("A manipulation episode is already active; preparation remains unclaimed")
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
                  "allocated_gpu_seconds_before_attempt": allocated_seconds,
                  "reservation_sha256": hashlib.sha256(args.reservation.read_bytes()).hexdigest(),
                  "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        # Exclusive claim is durable before spawning physics work.
        with receipt_path.open("x") as stream:
            stream.write(json.dumps(record, indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(receipts, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        print("Starting " + job["job_id"], flush=True)
        environment = dict(os.environ, OMNI_KIT_ALLOW_ROOT="1", OMP_NUM_THREADS="4",
                           OPENBLAS_NUM_THREADS="4")
        def on_started(process, record=record, receipt_path=receipt_path):
            record["process"] = process
            atomic_json(receipt_path, record)

        process = run_fresh_episode(command, output / (job["job_id"] + ".log"), timeout,
                                    env=environment, on_started=on_started)
        record["process"] = process
        record["returncode"] = process["returncode"]
        record["elapsed_seconds"] = process["process_elapsed_seconds"]
        record["status"] = "invalid"
        try:
            row = audit_attempt(record)
            record["evidence"] = row
            if process_succeeded(record) and row["valid"]:
                record["status"] = "valid"
        except (ValueError, OSError, KeyError, TypeError) as error:
            record["audit_error"] = {"type": type(error).__name__, "message": str(error)}
        finally:
            try:
                if (episode / "ledger.json").exists():
                    record["ledger"] = json.loads((episode / "ledger.json").read_text())
            except (OSError, ValueError) as error:
                record["status"] = "invalid"
                record["ledger_read_error"] = {"type": type(error).__name__, "message": str(error)}
            # Process completion and its cost survive malformed outcome/accounting files.
            atomic_json(receipt_path, record)
        print(json.dumps({"job": job["job_id"], "status": record["status"],
                          "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
        if record["status"] != "valid":
            raise RuntimeError("Preparation failed; attempt/costs preserved and queue stopped")


def main():
    def interrupted(signum, frame):
        raise SystemExit(f"Preparation interrupted by signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        with execution_lease():
            _main()
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
