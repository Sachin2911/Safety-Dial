"""Create only the additional four-GPU host authorized for UNISafe task 9."""

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.unisafe09_protocol import digest
from helpers.unisafe09p_infra import (
    IMAGE, NEW_BOOK, RENTAL, FinancialBook, atomic_json, current_finances, instances,
    utc, vast,
)


def offers():
    occupied = {row["machine_id"] for row in instances()}
    query = (
        "gpu_name=RTX_4090 num_gpus=4 cpu_ram>=250 cpu_cores_effective>=48 "
        "disk_space>=200 reliability>0.98 direct_port_count>=2 rented=False"
    )
    result = vast(["search", "offers", query, "--storage", "200", "--order", "dph", "--limit", "80"])
    rows = result if isinstance(result, list) else result.get("offers", [])
    fields = (
        "id", "machine_id", "gpu_name", "num_gpus", "dph_total", "cpu_ram",
        "cpu_cores_effective", "cpu_name", "driver_version", "inet_down_cost",
        "inet_up_cost", "storage_cost", "inet_down", "inet_up", "geolocation",
        "reliability", "disk_space", "gpu_ram",
    )
    accepted = []
    for row in rows:
        if row["machine_id"] in occupied:
            continue
        try:
            FinancialBook.validate_offer(row)
        except (ValueError, KeyError, TypeError):
            continue
        accepted.append({key: row.get(key) for key in fields})
    return sorted(accepted, key=lambda row: (row["dph_total"], -row["cpu_cores_effective"]))


def create(freeze_path):
    if RENTAL.exists():
        raise ValueError("Reconcile the existing extra-host record; do not create again")
    freeze = json.loads(freeze_path.read_text())
    approval = ROOT / "progress/receipts/09-unisafe-parallel-authorization.json"
    if freeze["authorization_sha256"] != digest(approval):
        raise ValueError("Preparation is not bound to the explicit user authorization")
    if freeze["combined_cap_usd"] != 15 or freeze["schema"] != "unisafe09p-preparation-v1":
        raise ValueError("Unexpected additional-host preparation")
    for name, expected in freeze["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Preparation source changed: {name}")
    candidates = offers()
    if not candidates:
        raise ValueError("No fresh offer fits the approved resources and bounded allocation")
    offer = candidates[0]
    old, new, combined = current_finances()
    if new["active_instances"] or new["pending_labels"] or combined["combined_stop"]:
        raise ValueError("Additional allocation already exists or combined budget is exhausted")
    created = dt.datetime.now(dt.timezone.utc)
    label = "block-unisafe09p-" + created.strftime("%m%d%H%M%S")
    book = FinancialBook(NEW_BOOK)
    old_rates = {row["rate_usd_hour"] for row in json.loads(
        (ROOT / "outputs/unisafe09/control/finance.json").read_text()
    )["allocations"]}
    if len(old_rates) != 1:
        raise ValueError("Recalculate original-host upper bound before creating")
    old_upper = 15 * max(old_rates)
    if old_upper + 6.5 + max(2, old["aggregate_transfer_estimate_usd"] +
                            150 * offer["inet_down_cost"] + 40 * offer["inet_up_cost"]) >= 15:
        raise ValueError("Independent host ceilings do not fit the combined $15 cap")
    book.reserve(label, created.isoformat(), offer)
    rental = {
        "schema": "unisafe09p-rental-v1", "created_utc": created.isoformat(),
        "label": label, "image": IMAGE, "disk_gb": 200, "offer": offer,
        "instance_id": None, "status": "reserved_before_create",
        "combined_cap_usd": 15, "active_shutdown_usd": 6.5,
        "allocation_deadline_utc": (created + dt.timedelta(hours=3.5)).isoformat(),
        "original_instance_id": 57647787, "original_active_upper_bound_usd": old_upper,
        "original_transfer_reservation_usd": old["aggregate_transfer_estimate_usd"],
        "preparation_sha256": digest(freeze_path),
    }
    atomic_json(RENTAL, rental)
    with (NEW_BOOK.parent / "watchdog.log").open("a") as log:
        guard = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "experiments/scripts/unisafe09p_watchdog.py")],
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, start_new_session=True,
        )
    for _ in range(100):
        ready = NEW_BOOK.parent / "watchdog-status.json"
        if guard.poll() is not None:
            book.cancel_rejected(label, absence_verified_utc=utc())
            raise RuntimeError("Additional-host watchdog exited before creation")
        if ready.exists() and json.loads(ready.read_text()).get("pid") == guard.pid:
            break
        time.sleep(0.1)
    else:
        book.cancel_rejected(label, absence_verified_utc=utc())
        raise RuntimeError("Additional-host watchdog readiness not established")
    rejected = False
    try:
        result = vast([
            "create", "instance", str(offer["id"]), "--image", IMAGE, "--disk", "200",
            "--label", label, "--cancel-unavail", "--ssh", "--direct", "--env",
            "-e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=yes "
            "-e OMNI_KIT_ALLOW_ROOT=1 -e PYTHONUNBUFFERED=1 -e UV_HTTP_TIMEOUT=300 "
            "-e UV_LINK_MODE=copy",
            "--onstart-cmd", "exec /opt/instance-tools/bin/entrypoint.sh",
        ], timeout=60)
        if result.get("success") is True and isinstance(result.get("new_contract"), int):
            rental.update(instance_id=result["new_contract"], status="create_acknowledged")
            book.attach(label, result["new_contract"])
        elif result.get("success") is False:
            rejected = True
            rental["status"] = "provider_rejected_create"
        else:
            rental["status"] = "create_acknowledgement_uncertain"
    except RuntimeError:
        rental["status"] = "create_acknowledgement_uncertain"
    atomic_json(RENTAL, rental)
    for _ in range(36):
        try:
            matches = [row for row in instances() if row["label"] == label]
        except RuntimeError:
            time.sleep(5)
            continue
        if len(matches) > 1:
            book.reconcile_duplicate_label(label, [row["id"] for row in matches])
            raise RuntimeError("Duplicate label; watchdog will stop and reconcile the copies")
        if matches:
            rental.update(instance_id=matches[0]["id"], status="provider_observed")
            book.attach(label, matches[0]["id"])
            atomic_json(RENTAL, rental)
            print(json.dumps({
                "instance_id": rental["instance_id"], "status": rental["status"],
                "hourly_compute_disk_usd": offer["dph_total"],
                "deadline_utc": rental["allocation_deadline_utc"], "watchdog_pid": guard.pid,
                "combined_cap_usd": 15,
            }))
            return
        if rejected:
            book.cancel_rejected(label, absence_verified_utc=utc())
            rental.update(status="create_rejected_absent", absence_verified_utc=utc())
            atomic_json(RENTAL, rental)
            return
        time.sleep(5)
    raise RuntimeError("Creation remains uncertain; reconcile this label without repeating create")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["offers", "create", "status"])
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    if args.command == "offers":
        print(json.dumps(offers()))
    elif args.command == "create":
        if args.freeze is None:
            parser.error("--freeze is required before creation")
        create(args.freeze)
    else:
        rental = json.loads(RENTAL.read_text())
        provider = next((row for row in instances() if row["id"] == rental["instance_id"]), None)
        print(json.dumps({"provider": provider, "combined_finance": current_finances()[2]}))
