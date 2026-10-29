"""Independent local deadline/cost guard; never operates an unrelated rental."""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm16_infra import RENTAL, current, safe_mutation, snapshot, utc
from helpers.lewm16_protocol import OUT, write_json

OUT.mkdir(parents=True, exist_ok=True)
write_json(OUT / "watchdog-ready.json", {"pid": os.getpid(), "utc": utc()})
while True:
    if (OUT / "watchdog-finished").exists():
        break
    try:
        if not RENTAL.exists():
            time.sleep(10)
            continue
        rental, row = current()
        status = snapshot(rental)
        status["provider"] = (
            {k: row.get(k) for k in ("id", "actual_status", "label")} if row else None
        )
        write_json(OUT / "watchdog-status.json", status)
        if row and status["stop"]:
            (OUT / "STOP").touch()
            if row["actual_status"] not in ("stopped", "exited"):
                safe_mutation(["stop", "instance", str(row["id"])])
        if not row and rental.get("instance_id") and rental.get("provider_seen_utc"):
            break
    except Exception as exc:
        # Provider helpers suppress potentially credential-bearing network output.
        write_json(OUT / "watchdog-error.json", {"utc": utc(), "type": type(exc).__name__})
    time.sleep(30)
