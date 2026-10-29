"""Resume a bounded evidence download from the existing task-9 rental."""

import argparse
import fcntl
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.unisafe09_infra import atomic_json, instances, ssh_endpoint, ssh_options, utc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    if not 120 <= args.timeout <= 1800:
        parser.error("Download timeout must be between 120 and 1800 seconds")
    output = ROOT / "outputs/unisafe09"
    control = output / "control"
    control.mkdir(parents=True, exist_ok=True)
    with (control / "sync.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("An evidence download is already active") from None
        rental = json.loads((ROOT / "progress/rental-unisafe09-20261025.json").read_text())
        provider = next((r for r in instances() if r["id"] == rental["instance_id"]), None)
        if provider is None or provider["actual_status"] != "running":
            raise RuntimeError("The existing rental must be running for SSH; no resume requested")
        host, port = ssh_endpoint(provider)
        log_path = control / "sync-progress.log"
        started = time.monotonic()
        with log_path.open("w") as log:
            process = subprocess.Popen(
                [
                    "rsync", "-az", "--partial", "--timeout=120", "--info=progress2",
                    "--outbuf=L", "-e", shlex.join(["ssh", *ssh_options(port)]),
                    f"root@{host}:/workspace/Safety-Dial/outputs/unisafe09/",
                    str(output / "remote") + "/",
                ], stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            atomic_json(control / "sync-process.json", {
                "pid": process.pid, "utc": utc(), "instance_id": provider["id"],
                "log": str(log_path), "wrapper_timeout_seconds": args.timeout,
            })
            print(json.dumps({"sync_pid": process.pid, "log": str(log_path)}), flush=True)
            timed_out = False
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                # Close any SSH/rsync children that outlived their process-group leader.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = 124
        result = {
            "exit_code": code, "utc": utc(), "seconds": time.monotonic() - started,
            "instance_id": provider["id"], "download_wrapper_timed_out": timed_out,
            "gpu_job_state_inferred_from_download": False,
        }
        atomic_json(control / "sync-result.json", result)
        print(json.dumps(result), flush=True)
        raise SystemExit(code)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
