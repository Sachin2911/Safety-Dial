"""Guard the requested additional host and the combined task-9 $15 ceiling."""

import argparse
import datetime as dt
import fcntl
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments/scripts")]
from helpers.unisafe09p_infra import (
    NEW_BOOK, OLD_BOOK, RENTAL, FinancialBook, OriginalFinancialBook, atomic_json,
    current_finances, instances, provider_stopped, utc, vast,
)
from unisafe09_watchdog import remote_stop


def remote(rental_path):
    rental = json.loads(rental_path.read_text())
    instance = rental["instance_id"]
    if str(instance) != os.environ.get("CONTAINER_ID") or not os.environ.get("CONTAINER_API_KEY"):
        raise ValueError("Additional-host shutdown identity is missing")
    out = ROOT / "outputs/unisafe09p"
    out.mkdir(parents=True, exist_ok=True)
    deadline = dt.datetime.fromisoformat(rental["allocation_deadline_utc"]).timestamp()
    atomic_json(out / "watchdog-ready.json", {
        "utc": utc(), "pid": os.getpid(), "instance_id": instance,
        "deadline_utc": rental["allocation_deadline_utc"],
    })
    while time.time() < deadline:
        rx = tx = 0
        for interface in Path("/sys/class/net").iterdir():
            if interface.name != "lo":
                rx += int((interface / "statistics/rx_bytes").read_text())
                tx += int((interface / "statistics/tx_bytes").read_text())
        offer = rental["offer"]
        transfer = rx / 1e9 * offer["inet_down_cost"] + tx / 1e9 * offer["inet_up_cost"]
        active = (time.time() - dt.datetime.fromisoformat(rental["created_utc"]).timestamp())
        active = active / 3600 * offer["dph_total"]
        worst_total = active + rental["original_active_upper_bound_usd"] + max(
            2, rental["original_transfer_reservation_usd"] + transfer
        )
        atomic_json(out / "network-cost.json", {
            "utc": utc(), "rx_bytes": rx, "tx_bytes": tx,
            "transfer_estimate_usd": transfer, "active_estimate_usd": active,
            "combined_worst_case_estimate_usd": worst_total,
        })
        if active >= 6.5 or worst_total >= 14.85:
            break
        time.sleep(min(10, max(0, deadline - time.time())))
    (out / "STOP").write_text("Additional-host time or combined money ceiling reached\n")
    for attempt in range(12):
        try:
            if remote_stop(instance):
                print(json.dumps({"event": "stop_requested", "utc": utc()}), flush=True)
                return
        except Exception as error:
            print(json.dumps({"error_type": type(error).__name__, "attempt": attempt}), flush=True)
        time.sleep(5)
    raise RuntimeError("Additional-host stop acknowledgement unavailable")


def local():
    control = NEW_BOOK.parent
    control.mkdir(parents=True, exist_ok=True)
    lock = (control / "watchdog.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (control / "watchdog.pid").write_text(str(os.getpid()) + "\n")
    while True:
        book = FinancialBook(NEW_BOOK)
        new = book.snapshot()
        if new["pending_labels"]:
            try:
                rows = instances()
                for label in new["pending_labels"]:
                    matches = [row for row in rows if row["label"] == label]
                    if len(matches) == 1:
                        book.attach(label, matches[0]["id"])
                    elif len(matches) > 1:
                        book.reconcile_duplicate_label(label, [row["id"] for row in matches])
            except RuntimeError:
                pass
        old, new, combined = current_finances()
        atomic_json(control / "watchdog-status.json", {
            "utc": utc(), "pid": os.getpid(), "old": old, "new": new, "combined": combined,
        })
        targets = []
        if combined["combined_stop"]:
            targets = old["active_instances"] + new["active_instances"]
        elif new["stop"] or (
            new["remaining_seconds"] is not None and new["remaining_seconds"] <= 300
        ):
            targets = new["active_instances"]
        if targets:
            for instance in targets:
                try:
                    vast(["stop", "instance", str(instance)])
                except RuntimeError:
                    pass
            try:
                provider = {row["id"]: row for row in instances()}
                for instance in targets:
                    row = provider.get(instance)
                    if provider_stopped(row):
                        target_book = (
                            OriginalFinancialBook(OLD_BOOK) if instance in old["active_instances"]
                            else FinancialBook(NEW_BOOK)
                        )
                        target_book.stopped(instance, utc(), status="destroyed" if row is None else "exited")
            except RuntimeError:
                pass
        if not new["active_instances"] and not new["pending_labels"]:
            return
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--rental", type=Path, default=RENTAL)
    arguments = parser.parse_args()
    if arguments.remote:
        remote(arguments.rental)
    else:
        local()
