"""Run only the separately frozen four-start screen after original-study clearance."""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_continuation_admission import (
    ScreenLedger,
    acquire_service_lease,
    policy_lease_path,
    require_original_clearance,
)
from helpers.maniguard_continuation_guardian import atomic_record
from helpers.maniguard_continuation_jobs import physical_environment, prepare_job
from helpers.maniguard_continuation_limits import resource_contract
from helpers.maniguard_continuation_progress import attribute_progress
from helpers.maniguard_continuation_results import read_episode
from helpers.maniguard_continuation_service import capture_service_records
from helpers.maniguard_latent_queue import process_identity


def record_verification(output, receipt):
    """Recheck on resume without overwriting an earlier recorded result."""
    for name, value in (
        ("verified-receipt.json", receipt),
        ("progress-attribution.json", attribute_progress(Path(output) / "trace.jsonl", receipt)),
    ):
        path = Path(output) / name
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError("An existing screen result differs from fresh validation: " + name)
        else:
            atomic_record(path, value)


def read_resources(protocol, run):
    if str(os.environ.get("CONTAINER_ID")) != str(protocol["instance_id"]):
        raise ValueError("The screen is not authorized on this instance")
    token = os.environ.get("CONTAINER_API_KEY")
    if not token:
        raise ValueError("Existing own-instance read credential is unavailable")
    request = urllib.request.Request(
        f"https://console.vast.ai/api/v0/instances/{protocol['instance_id']}/",
        headers={"Authorization": "Bearer " + token},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = json.load(response)
    instance = payload.get("instances", payload.get("instance"))
    if not isinstance(instance, dict):
        raise TypeError("Own-instance API returned no unique instance object")
    safe = {key: instance.get(key) for key in (
        "id", "machine_id", "actual_status", "num_gpus", "gpu_name", "gpu_ram",
        "disk_space", "dph_total", "start_date",
    )}
    clock = json.loads(Path(protocol["watchdog_receipt"]).read_text())
    now = time.time()
    stop = resource_contract(safe, clock, now=now,
                             watchdog_live=process_identity(clock["pid"]) is not None)
    atomic_record(Path(run) / f"resource-readback-{time.time_ns()}.json", {
        "checked_epoch": now, "http_status": 200, "instance": safe, "clock": clock,
        "scope": "Own-instance GET only. No credential value or provider change.",
    })
    return stop


def reconcile(ledger, attempts):
    """Observe exact existing handles; never start a replacement after a lost coordinator."""
    for attempt in attempts:
        output = Path(attempt["output"])
        identity = attempt.get("guardian_identity")
        while True:
            path = output / "guardian.json"
            if path.exists():
                identity = json.loads(path.read_text())["guardian"]
            if identity is not None and process_identity(identity["pid"]) == identity:
                time.sleep(1)
                continue
            # A guardian may have inherited the lease before publishing its PID.
            lease_path = policy_lease_path(ledger.protocol, attempt["worker"])
            try:
                available = acquire_service_lease(lease_path)
            except BlockingIOError:
                if time.time() > attempt["deadline_epoch"] + 10:
                    raise RuntimeError("Policy lease is still occupied after its recorded guardian deadline")
                time.sleep(1)
                continue
            try:
                capture_service_records(output)
            except (OSError, ValueError, KeyError) as exc:
                attempt["service_snapshot_error"] = type(exc).__name__ + ": " + str(exc)
            finally:
                available.close()
            break
        try:
            if attempt.get("service_snapshot_error"):
                raise ValueError(attempt["service_snapshot_error"])
            terminal = json.loads((output / "guardian-terminal.json").read_text())
            if terminal["exit_code"] != 0 or terminal["technical_stop_reason"] is not None:
                raise ValueError("The charged start failed its independent guardian")
            receipt = read_episode(output)
            record_verification(output, receipt)
            attempt.update(status="completed", score=receipt["score"],
                           finished_epoch=terminal["finished_epoch"])
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            attempt.update(status="invalid", finished_epoch=time.time(),
                           invalid_reason=type(exc).__name__ + ": " + str(exc))
        ledger.save()


def run_screen(run, protocol_path):
    ledger = ScreenLedger(run, protocol_path)
    try:
        for attempt in ledger.state["attempts"]:
            if attempt["status"] == "completed":
                record_verification(attempt["output"], read_episode(attempt["output"]))
        unfinished = [a for a in ledger.state["attempts"] if a["status"] in ("reserved", "running")]
        if unfinished:
            reconcile(ledger, unfinished)
        for wave in (0, 1):
            rows = [r for r in ledger.protocol["episodes"] if r["wave"] == wave]
            done = {a["protocol_episode_id"] for a in ledger.state["attempts"] if a["status"] == "completed"}
            if all(row["id"] in done for row in rows):
                continue
            clearance = require_original_clearance(ledger.protocol)
            stop = read_resources(ledger.protocol, ledger.run)
            leases = {}
            children = []
            try:
                for row in rows:
                    path = policy_lease_path(ledger.protocol, row["worker"])
                    leases[row["worker"]] = acquire_service_lease(path)
                attempts = ledger.reserve_wave(wave, now=time.time(), actual_stop_epoch=stop)
                for attempt in attempts:
                    try:
                        job = prepare_job(ledger.protocol_path, attempt,
                                          ledger.protocol["closure_controller_file"], clearance)
                        descriptor = leases[attempt["worker"]].fileno()
                        command = [str(ROOT / "data/maniguard/envs/tools/bin/python"), "-u",
                                   str(ROOT / "experiments/scripts/maniguard_continuation_episode.py"),
                                   "--job", str(job), "--lease-fd", str(descriptor)]
                        with (Path(attempt["output"]) / "guardian.log").open("x") as log:
                            child = subprocess.Popen(command, cwd=ROOT,
                                env=physical_environment(ledger.protocol, attempt["worker"]),
                                pass_fds=(descriptor,), stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
                        children.append(child)
                        attempt.update(status="running", guardian_identity=process_identity(child.pid),
                                       guardian_command=command)
                    except (OSError, ValueError, KeyError, RuntimeError) as exc:
                        attempt.update(status="invalid", invalid_reason=type(exc).__name__ + ": " + str(exc))
                    ledger.save()
            finally:
                # Each launched guardian/physical child retains its own lease FD.
                for handle in leases.values():
                    handle.close()
            running = [a for a in attempts if a["status"] == "running"]
            reconcile(ledger, running)
            for child in children:
                child.wait()
            if any(a["status"] == "invalid" for a in attempts):
                raise RuntimeError("Screen stopped after a charged invalid start; diagnose before any new stage")
        ledger.state["screen_complete"] = len(ledger.state["attempts"]) == 4 and all(
            a["status"] == "completed" for a in ledger.state["attempts"])
        ledger.save()
        return ledger.state
    finally:
        ledger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="Read readiness only; no reservation or service call")
    args = parser.parse_args()
    if args.check:
        protocol = json.loads(args.protocol.read_text())
        try:
            if protocol["status"] != "prospectively_frozen":
                raise ValueError("The prospective source/protocol freeze is unfinished")
            require_original_clearance(protocol)
            print(json.dumps({"original_clearance_ready": True, "resource_preflight_required": True,
                              "reserved_starts": 0, "physical_dispatch": False}))
        except (OSError, ValueError, KeyError) as exc:
            print(json.dumps({"original_clearance_ready": False, "reason": str(exc), "reserved_starts": 0,
                              "physical_dispatch": False}))
        return
    state = run_screen(args.run, args.protocol)
    print(json.dumps({"screen_complete": state["screen_complete"],
                      "charged_starts": len(state["attempts"]), "new_stage_auto_admitted": False}))


if __name__ == "__main__":
    main()
