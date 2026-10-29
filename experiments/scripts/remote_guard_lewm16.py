"""Independent instance-scoped cost/deadline stop, using only the rental's own token."""

import datetime as dt
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm16_protocol import OUT, write_json

rental = json.loads((ROOT / "progress/rental-lewm16-20261026.json").read_text())
authorization = json.loads((ROOT / "progress/receipts/16-lewm-authorization.json").read_text())
instance = rental["instance_id"]
assert os.environ.get("CONTAINER_ID") == str(instance)
assert os.environ.get("CONTAINER_API_KEY")
end = dt.datetime.fromisoformat(authorization["limits"]["target_end_utc"]).timestamp() - 1200
start = dt.datetime.fromisoformat(rental["created_utc"]).timestamp()
write_json(
    OUT / "remote-guard-ready.json",
    {"instance_id": instance, "pid": os.getpid(), "stop_epoch": end},
)
while True:
    rx = tx = 0
    for interface in Path("/sys/class/net").iterdir():
        if interface.name == "lo":
            continue
        rx += int((interface / "statistics/rx_bytes").read_text())
        tx += int((interface / "statistics/tx_bytes").read_text())
    offer = rental["offer"]
    transfer = rx / 1e9 * offer["inet_down_cost"] + tx / 1e9 * offer["inet_up_cost"]
    active = (time.time() - start) / 3600 * offer["dph_total"]
    write_json(
        OUT / "network-cost.json",
        {
            "epoch": time.time(),
            "rx_bytes": rx,
            "tx_bytes": tx,
            "transfer_usd": transfer,
            "active_usd": active,
        },
    )
    if time.time() >= end or active + max(transfer, 0.75) >= 8.5:
        (OUT / "STOP").touch()
        for _ in range(12):
            try:
                request = urllib.request.Request(
                    f"https://console.vast.ai/api/v0/instances/{instance}/",
                    data=b'{"state":"stopped"}',
                    method="PUT",
                    headers={
                        "Authorization": "Bearer " + os.environ["CONTAINER_API_KEY"],
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(request, timeout=20) as response:
                    if json.load(response).get("success") is True:
                        raise SystemExit(0)
            except Exception as exc:
                print(json.dumps({"stop_error_type": type(exc).__name__}), flush=True)
            time.sleep(5)
        raise SystemExit(1)
    time.sleep(20)
