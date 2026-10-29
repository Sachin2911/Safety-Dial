"""Task-16-only rental control and conservative cumulative accounting."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import time

from helpers.dsrl06_infra import instances, ssh_endpoint, ssh_options as old_ssh_options
from helpers.lewm16_protocol import ROOT

RENTAL = ROOT / "progress/rental-lewm16-20261026.json"
IMAGE = "vastai/base-image@sha256:c1d2b5326fae806b04d2c2d97a2b3948d0ccb0026dd3085e9a2ad55e304193f8"
DISK_GB = 180
VAST = "/home/sachin/.local/bin/vastai"


def ssh_options(port):
    return [
        x.replace("/tmp/dsrl06-known-hosts", "/tmp/lewm16-known-hosts")
        for x in old_ssh_options(port)
    ]


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def deadline():
    auth = json.loads((ROOT / "progress/receipts/16-lewm-authorization.json").read_text())
    return dt.datetime.fromisoformat(auth["limits"]["target_end_utc"]).timestamp()


def validate_offer(offer):
    if offer["gpu_name"] not in ("RTX 3090", "RTX 4090") or offer["num_gpus"] not in (2, 4):
        raise ValueError("Offer outside approved hardware family")
    if float(offer["dph_total"]) > 1.75:
        raise ValueError("Combined rate exceeds the approved ceiling")
    if float(offer["cpu_ram"]) < 60000 or float(offer["cpu_cores_effective"]) < 24:
        raise ValueError("Insufficient resources for independent workers")
    if float(offer["disk_space"]) < DISK_GB:
        raise ValueError("Insufficient storage")
    hours = max(0, deadline() - time.time()) / 3600
    transfer = 30 * float(offer["inet_down_cost"]) + 15 * float(offer["inet_up_cost"])
    estimate = hours * float(offer["dph_total"]) + transfer + 0.75
    if estimate > 10:
        raise ValueError("Full remaining allocation plus transfers/closeout exceeds $10")
    return {
        "remaining_hours": hours,
        "transfer_reserve_usd": transfer,
        "forecast_through_deadline_usd": estimate,
        "cleanup_reserve_usd": 0.75,
    }


def safe_mutation(args):
    """No raw CLI output is exposed. Always reconcile authoritative state afterwards."""
    try:
        p = subprocess.run([VAST, *args, "--raw"], capture_output=True, text=True, timeout=90)
        try:
            response = json.loads(p.stdout)
        except ValueError:
            response = {}
        return {
            "returncode": p.returncode,
            "success": response.get("success"),
            "new_contract": response.get("new_contract"),
            "acknowledgement": "received",
        }
    except subprocess.TimeoutExpired:
        return {"acknowledgement": "timeout_reconcile_required"}


def current():
    rental = json.loads(RENTAL.read_text())
    matches = [r for r in instances() if r["label"] == rental["label"]]
    if len(matches) > 1:
        raise RuntimeError("Duplicate allocation label; reconcile all charges before proceeding")
    if matches and rental.get("instance_id") not in (None, matches[0]["id"]):
        raise RuntimeError("Instance identity changed unexpectedly")
    return rental, matches[0] if matches else None


def snapshot(rental):
    start = dt.datetime.fromisoformat(rental["created_utc"]).timestamp()
    end = (
        dt.datetime.fromisoformat(rental["destroyed_utc"]).timestamp()
        if rental.get("destroyed_utc")
        else time.time()
    )
    # Includes full compute+storage even after a stop until destruction, conservatively.
    cost = max(0, end - start) / 3600 * float(rental["offer"]["dph_total"])
    total = cost + rental["admission"]["transfer_reserve_usd"]
    return {
        "utc": utc(),
        "compute_storage_upper_estimate_usd": cost,
        "transfer_reserved_usd": rental["admission"]["transfer_reserve_usd"],
        "cumulative_estimate_usd": total,
        "stop": total >= 8.5 or time.time() >= deadline() - 20 * 60,
    }


def remote(command, check=True, timeout=60):
    rental, row = current()
    if row is None:
        raise RuntimeError("Task-16 rental absent")
    host, port = ssh_endpoint(row)
    return subprocess.run(
        ["ssh", *ssh_options(port), f"root@{host}", command],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )
