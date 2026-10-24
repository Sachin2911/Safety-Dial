"""Copy this rental's evidence to the workstation without exporting credentials."""

import argparse
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    rental = json.loads((ROOT / "progress/rental-adapter05-20261024.json").read_text())
    provider = rental["provider"]
    port = provider["ports"]["22/tcp"][0]["HostPort"]
    host = provider["public_ipaddr"]
    dest = ROOT / "outputs/adapter05/remote"
    dest.mkdir(parents=True, exist_ok=True)
    command = [
        "rsync",
        "-az",
        "--partial",
        "--timeout=90",
        "-e",
        f"ssh -p {port} -o ConnectTimeout=15 -o BatchMode=yes",
        f"root@{host}:/workspace/Safety-Dial/outputs/adapter05/",
        str(dest) + "/",
    ]
    until = dt.datetime.fromisoformat(rental["created_utc"]).timestamp() + 6 * 3600
    while True:
        process = subprocess.run(command, capture_output=True)
        row = {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "sync_exit": process.returncode}
        ledgers = list(dest.glob("attempts_*.jsonl"))
        if ledgers:
            events = []
            for line in [line for ledger in ledgers for line in ledger.read_text().splitlines()]:
                try:
                    events.append(json.loads(line))
                except ValueError:
                    continue  # A later copy will finish an actively written line.
            row["validated"] = sum(e["event"] == "validated" for e in events)
            row["queue_finished"] = (dest / "pilot-complete.txt").exists()
        print(json.dumps(row), flush=True)
        if (
            not args.watch
            or row.get("queue_finished")
            or time.time() >= until
            or (dest / "STOP").exists()
        ):
            break
        time.sleep(45)


if __name__ == "__main__":
    main()
