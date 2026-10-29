"""Portable task-17 lease identity, accounting and instance-scoped provider access."""

import datetime as dt
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/lewm17"
RENTAL = ROOT / "progress/rental-lewm17-20261026.json"


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.part")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def rental():
    if not os.environ.get("CONTAINER_ID") or not os.environ.get("CONTAINER_API_KEY"):
        # Vast injects these into PID 1; plain SSH sessions need not inherit them.
        for entry in Path("/proc/1/environ").read_bytes().split(b"\0"):
            key, separator, value = entry.partition(b"=")
            if separator and key in {b"CONTAINER_ID", b"CONTAINER_API_KEY"}:
                os.environ.setdefault(key.decode(), value.decode())
    result = json.loads(RENTAL.read_text())
    if not result["label"].startswith("lewm17-") or result["financial_cap_usd"] != 10:
        raise ValueError("Unexpected task-17 allocation")
    if str(result["instance_id"]) != os.environ.get("CONTAINER_ID"):
        raise ValueError("This process is not on the authorized task-17 rental")
    return result


def finance(record, now=None, rx=None, tx=None):
    now = time.time() if now is None else now
    if rx is None or tx is None:
        rx = tx = 0
        for net in Path("/sys/class/net").iterdir():
            if net.name != "lo":
                rx += int((net / "statistics/rx_bytes").read_text())
                tx += int((net / "statistics/tx_bytes").read_text())
    rate = record["offer"]
    start = dt.datetime.fromisoformat(record["created_utc"]).timestamp()
    deadline = dt.datetime.fromisoformat(record["deadline_utc"]).timestamp()
    transfer = rx / 1e9 * rate["inet_down_cost"] + tx / 1e9 * rate["inet_up_cost"]
    compute_disk = max(0, now - start) / 3600 * rate["dph_total"]
    estimate = compute_disk + max(transfer, 0.75)
    return dict(
        utc=utc(),
        compute_disk_usd=compute_disk,
        measured_transfer_usd=transfer,
        conservative_total_usd=estimate,
        deadline_epoch=deadline,
        closeout_due=now >= deadline - 1800 or estimate >= 8.5,
        hard_stop_due=now >= deadline or estimate >= 9.5,
        rx_bytes=rx,
        tx_bytes=tx,
    )


def request_self(method, payload=None):
    record = rental()
    token = os.environ.get("CONTAINER_API_KEY")
    if not token:
        raise RuntimeError("Instance-scoped provider credential unavailable")
    body = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"https://console.vast.ai/api/v0/instances/{record['instance_id']}/",
        data=body,
        method=method,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.load(response)


def hf_secrets():
    # The provisioner writes only JSON-quoted HF_TOKEN/HF_NAMESPACE into this file.
    values = {}
    for line in (ROOT / ".env").read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"HF_TOKEN", "HF_NAMESPACE"}:
            values[key] = json.loads(value)
    if not values.get("HF_TOKEN"):
        raise RuntimeError("Private preservation credential unavailable")
    return values
