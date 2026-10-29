"""Instance-scoped deadline stop for one task 19 host (runs on the host).

Writes outputs/evo19/STOP and asks the provider to stop this instance at the task 19
deadline (03:20 UTC, 27 October 2026). It can only stop its own container.
"""

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments/scripts"))
from regrasp15_watchdog import remote_stop  # noqa: E402

OUT = ROOT / "outputs/evo19"
DEADLINE = dt.datetime(2026, 10, 27, 3, 20, tzinfo=dt.timezone.utc).timestamp()


def main():
    instance = os.environ.get("CONTAINER_ID")
    if not instance or not os.environ.get("CONTAINER_API_KEY"):
        raise SystemExit("Instance-scoped shutdown credentials are missing")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "guard-ready.json").write_text(json.dumps(
        {"instance_id": instance, "pid": os.getpid(), "deadline_epoch": DEADLINE}))
    while time.time() < DEADLINE:
        time.sleep(min(30, max(1, DEADLINE - time.time())))
    (OUT / "STOP").write_text("Task 19 deadline reached\n")
    for _ in range(12):
        try:
            if remote_stop(instance):
                print(json.dumps({"event": "stop_requested", "utc": time.time()}), flush=True)
                return
        except Exception as error:  # keep retrying; the local guard is the backstop
            print(json.dumps({"event": "stop_error", "kind": type(error).__name__}), flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
