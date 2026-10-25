"""Independent task-6 financial stop guard, usable locally or on its own rental."""

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import FinancialBook, atomic_json, instances, provider_stopped, utc, vast


def remote_stop(instance_id):
    if os.environ.get("CONTAINER_ID") != str(instance_id):
        raise ValueError("Instance-scoped watchdog cannot stop another rental")
    request = urllib.request.Request(
        f"https://console.vast.ai/api/v0/instances/{instance_id}/",
        data=b'{"state":"stopped"}',
        method="PUT",
        headers={
            "Authorization": "Bearer " + os.environ["CONTAINER_API_KEY"],
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.load(response).get("success") is True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", type=Path)
    parser.add_argument("--rental", type=Path)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--poll", type=float, default=10)
    args = parser.parse_args()
    if args.remote:
        rental = json.loads(args.rental.read_text())
        deadline = dt.datetime.fromisoformat(rental["allocation_deadline_utc"]).timestamp()
        instance = rental["instance_id"]
        if os.environ.get("CONTAINER_ID") != str(instance) or not os.environ.get(
            "CONTAINER_API_KEY"
        ):
            raise ValueError("Instance-scoped shutdown credentials are missing")
        atomic_json(
            ROOT / "outputs/dsrl06/watchdog-ready.json",
            {
                "instance_id": instance,
                "pid": os.getpid(),
                "utc": utc(),
                "deadline_utc": rental["allocation_deadline_utc"],
            },
        )
        print(
            json.dumps(
                {
                    "event": "remote_watchdog_started",
                    "instance": instance,
                    "deadline_utc": rental["allocation_deadline_utc"],
                }
            ),
            flush=True,
        )
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
            output = ROOT / "outputs/dsrl06"
            output.mkdir(parents=True, exist_ok=True)
            atomic_json(
                output / "network-cost.json",
                {
                    "utc": utc(),
                    "rx_bytes": rx,
                    "tx_bytes": tx,
                    "transfer_estimate_usd": transfer,
                    "active_estimate_usd": active,
                },
            )
            if active + max(2, cumulative_transfer) >= 14.9:
                break
            time.sleep(min(args.poll, max(0, deadline - time.time())))
        output = ROOT / "outputs/dsrl06"
        output.mkdir(parents=True, exist_ok=True)
        (output / "STOP").write_text("Financial deadline reached\n")
        for attempt in range(12):
            try:
                if remote_stop(instance):
                    print(json.dumps({"event": "stop_requested", "utc": utc()}), flush=True)
                    return
            except Exception as error:
                print(
                    json.dumps(
                        {"event": "stop_error", "kind": type(error).__name__, "attempt": attempt}
                    ),
                    flush=True,
                )
            time.sleep(5)
        raise SystemExit("Remote stop not acknowledged; local watchdog must intervene")
    if args.book is None:
        parser.error("--book is required for local monitoring")
    while True:
        book = FinancialBook(args.book)
        snapshot = book.snapshot()
        if snapshot["pending_labels"]:
            try:
                current = instances()
                for label in snapshot["pending_labels"]:
                    matches = [r for r in current if r.get("label") == label]
                    if len(matches) == 1:
                        book.attach(label, matches[0]["id"])
                    elif len(matches) > 1:
                        book.reconcile_duplicate_label(label, [r["id"] for r in matches])
                book = FinancialBook(args.book)
                snapshot = book.snapshot()
            except RuntimeError:
                pass
        atomic_json(
            args.book.parent / "watchdog-status.json",
            {
                "utc": utc(),
                "pid": os.getpid(),
                **snapshot,
            },
        )
        if snapshot["stop"] or (
            snapshot["remaining_seconds"] is not None and snapshot["remaining_seconds"] <= 300
        ):
            for instance in snapshot["active_instances"]:
                try:
                    vast(["stop", "instance", str(instance)])
                except RuntimeError:
                    pass  # An acknowledgement is not evidence of terminal state.
            try:
                current = {r["id"]: r for r in instances()}
                for instance in snapshot["active_instances"]:
                    row = current.get(instance)
                    if provider_stopped(row):
                        book.stopped(
                            instance, utc(), status="destroyed" if row is None else "exited"
                        )
            except RuntimeError:
                pass
            print(json.dumps({"event": "budget_stop", "utc": utc(), **snapshot}), flush=True)
            final = FinancialBook(args.book).snapshot()
            if not final["active_instances"] and not final["pending_labels"]:
                return
        if args.rental and not snapshot["active_instances"] and not snapshot["pending_labels"]:
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
