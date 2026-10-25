"""Continue evidence copying with a longer transfer window and no experiment edits."""

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

OUT = ROOT / "outputs/screen08"


def existing_transfers():
    pids = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = path.read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if (args and Path(args[0]).name == "rsync"
                and str(OUT / "remote") + "/" in args):
            pids.append(int(path.parent.name))
    return pids


def main():
    control = OUT / "control"
    with (control / "extended-mirror.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            record = {"started_utc": utc(), "pid": os.getpid(), "wall_timeout_s": 900,
                      "implementation": "mirror_screen08.py"}
            if not FinancialBook(control / "finance.json").snapshot()["active_instances"]:
                return
            prior = existing_transfers()
            if prior:
                record.update(status="waiting_for_previous_transfer", previous_pids=prior)
                atomic_json(control / "sync-status.json", record)
                print(json.dumps(record), flush=True)
                time.sleep(15)
                continue
            try:
                rental = json.loads((ROOT / "progress/rental-screen08-20261025.json").read_text())
                provider = next((r for r in instances() if r["id"] == rental["instance_id"]), None)
                if provider_stopped(provider):
                    return
                host, port = ssh_endpoint(provider)
                result = subprocess.run(
                    ["rsync", "-az", "--partial", "--stats", "--timeout=90", "-e",
                     shlex.join(["ssh", *ssh_options(port)]),
                     f"root@{host}:/workspace/Safety-Dial/outputs/screen08/",
                     str(OUT / "remote") + "/"],
                    capture_output=True, text=True, timeout=900,
                )
                record["exit_code"] = result.returncode
                # Summaries contain counts and byte totals, never network exception text.
                record["transfer_summary"] = [line for line in result.stdout.splitlines()
                                              if line.startswith(("Number of files:",
                                                                  "Number of regular files",
                                                                  "Total file size:",
                                                                  "Total transferred file size:",
                                                                  "Literal data:",
                                                                  "Matched data:"))]
            except Exception as error:
                record["error_type"] = type(error).__name__
            record["ended_utc"] = utc()
            atomic_json(control / "sync-status.json", record)
            with (control / "extended-mirror-events.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            time.sleep(45)


if __name__ == "__main__":
    main()
