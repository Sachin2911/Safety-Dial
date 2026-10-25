"""Stop task-8 compute only after complete results have a verified local snapshot.

This does not change the experiment or destroy its disk. Private upload and its
pinned-revision verification are separate required steps before destruction.
"""

import argparse
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.screen08_infra import (
    FinancialBook, atomic_json, instances, provider_stopped, ssh_endpoint, ssh_options, utc,
)
from helpers.screen08_preservation import verify_snapshot

OUT = ROOT / "outputs/screen08"


def check(*, stop_compute):
    rental = json.loads((ROOT / "progress/rental-screen08-20261025.json").read_text())
    instance_id = rental["instance_id"]
    owned = next((r for r in instances() if r["id"] == instance_id), None)
    if provider_stopped(owned):
        return {"status": "already_stopped", "instance_id": instance_id}
    host, port = ssh_endpoint(owned)
    source = (ROOT / "experiments/helpers/screen08_preservation.py").read_text()
    source += ("\nprint(json.dumps(snapshot(Path('/workspace/Safety-Dial'), "
               + str(int(instance_id)) + ")))\n")
    remote = subprocess.run(
        ["ssh", *ssh_options(port), f"root@{host}", "python3 -"],
        input=source, capture_output=True, text=True, timeout=180,
    )
    if remote.returncode:
        raise RuntimeError("Remote completion or snapshot check failed")
    state = json.loads(remote.stdout)
    if state["status"] != "snapshot_ready":
        return state
    if state["instance_id"] != instance_id:
        raise ValueError("Remote snapshot ownership differs from the recorded allocation")
    destination = OUT / "completed-snapshot"
    destination.mkdir(parents=True, exist_ok=True)
    # The live mirror contains raw files directly. Match that relative layout
    # before copying the enclosing receipt, so rsync can reuse verified bytes.
    for suffix in ("raw/", "receipt.json"):
        target = destination / suffix
        if suffix == "raw/":
            target.mkdir(parents=True, exist_ok=True)
        reuse = ["--copy-dest=" + str(OUT / "remote")] if suffix == "raw/" else []
        transfer = subprocess.run(
            ["rsync", "-az", "--checksum", *reuse, "--partial", "--timeout=90", "-e",
             shlex.join(["ssh", *ssh_options(port)]),
             f"root@{host}:{state['path']}/{suffix}",
             str(target) + ("/" if suffix == "raw/" else "")],
            capture_output=True, text=True, timeout=1200,
        )
        if transfer.returncode:
            raise RuntimeError("Terminal snapshot transfer did not complete")
    receipt = verify_snapshot(destination)
    if receipt["instance_id"] != instance_id:
        raise ValueError("Local snapshot ownership differs from the recorded allocation")
    verified = {"utc": utc(), "instance_id": instance_id, "local_verified": True,
                "snapshot": str(destination), "files": len(receipt["files"]),
                "bytes": sum(p["bytes"] for p in receipt["files"].values()),
                "private_hf_verified": False}
    atomic_json(OUT / "control/terminal-local-verification.json", verified)
    if not stop_compute:
        return {"status": "local_snapshot_verified", **verified}
    stop = subprocess.run(
        [sys.executable, str(ROOT / "experiments/scripts/manage_screen08.py"), "stop"],
        capture_output=True, text=True, timeout=180,
    )
    if stop.returncode:
        raise RuntimeError("Stop observation failed; recheck the same allocation")
    row = next((r for r in instances() if r["id"] == instance_id), None)
    if not provider_stopped(row):
        raise RuntimeError("Compute stop is not confirmed by provider state")
    verified.update(status="stopped_with_verified_local_snapshot", stop_verified_utc=utc(),
                    finance=FinancialBook(OUT / "control/finance.json").snapshot())
    atomic_json(OUT / "control/completion-stop.json", verified)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--stop-compute", action="store_true")
    args = parser.parse_args()
    control = OUT / "control"
    control.mkdir(parents=True, exist_ok=True)
    with (control / "completion-monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                state = check(stop_compute=args.stop_compute)
            except Exception as error:
                # Provider/network exception text can contain authentication material.
                state = {"status": "observation_failed", "error_type": type(error).__name__}
            state["checked_utc"] = utc()
            state["pid"] = os.getpid()
            atomic_json(control / "completion-monitor-status.json", state)
            print(json.dumps(state), flush=True)
            if not args.wait or state["status"] not in ("pending", "observation_failed"):
                return
            time.sleep(45)


if __name__ == "__main__":
    main()
