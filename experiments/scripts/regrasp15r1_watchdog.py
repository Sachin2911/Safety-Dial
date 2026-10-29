"""Instance-scoped financial stop guard for the task 15 recovery rental (remote side).

The local guard is regrasp15_watchdog.py run with the recovery book and rental. This
remote guard is regrasp15_watchdog.py --remote with one change: it writes to
outputs/regrasp15r1, where the recovery queue looks for STOP. The rental record carries
the imported historical active and transfer estimates, so the $10 stop includes every
earlier task 15 charge.
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments/scripts"))
from helpers.regrasp15_infra import atomic_json, utc  # noqa: E402
from regrasp15_watchdog import remote_stop  # noqa: E402

OUT = ROOT / "outputs/regrasp15r1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rental", type=Path, required=True)
    parser.add_argument("--poll", type=float, default=10)
    args = parser.parse_args()
    rental = json.loads(args.rental.read_text())
    deadline = dt.datetime.fromisoformat(rental["allocation_deadline_utc"]).timestamp()
    instance = rental["instance_id"]
    if os.environ.get("CONTAINER_ID") != str(instance) or not os.environ.get(
        "CONTAINER_API_KEY"
    ):
        raise ValueError("Instance-scoped shutdown credentials are missing")
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(
        OUT / "watchdog-ready.json",
        {"instance_id": instance, "pid": os.getpid(), "utc": utc(),
         "deadline_utc": rental["allocation_deadline_utc"]},
    )
    print(json.dumps({"event": "remote_watchdog_started", "instance": instance,
                      "deadline_utc": rental["allocation_deadline_utc"]}), flush=True)
    while time.time() < deadline:
        rx = tx = 0
        for interface in Path("/sys/class/net").iterdir():
            if interface.name == "lo":
                continue
            try:
                rx += int((interface / "statistics/rx_bytes").read_text())
                tx += int((interface / "statistics/tx_bytes").read_text())
            except OSError:
                continue
        offer = rental["offer"]
        transfer = rx / 1e9 * offer["inet_down_cost"] + tx / 1e9 * offer["inet_up_cost"]
        cumulative_transfer = rental.get("prior_transfer_estimate_usd", 0) + max(
            rental.get("transfer_estimate_usd", 0), transfer
        )
        active = (
            rental.get("prior_active_estimate_usd", 0)
            + (time.time() - dt.datetime.fromisoformat(rental["created_utc"]).timestamp())
            / 3600
            * offer["dph_total"]
        )
        atomic_json(
            OUT / "network-cost.json",
            {"utc": utc(), "rx_bytes": rx, "tx_bytes": tx,
             "transfer_estimate_usd": transfer, "active_estimate_usd": active},
        )
        if active + max(2, cumulative_transfer) >= 9.9:
            break
        time.sleep(min(args.poll, max(0, deadline - time.time())))
    (OUT / "STOP").write_text("Financial deadline reached\n")
    for attempt in range(12):
        try:
            if remote_stop(instance):
                print(json.dumps({"event": "stop_requested", "utc": utc()}), flush=True)
                return
        except Exception as error:
            print(json.dumps({"event": "stop_error", "kind": type(error).__name__,
                              "attempt": attempt}), flush=True)
        time.sleep(5)
    raise SystemExit("Remote stop not acknowledged; local watchdog must intervene")


if __name__ == "__main__":
    main()
