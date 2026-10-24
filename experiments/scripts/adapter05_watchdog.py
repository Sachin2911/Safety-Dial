"""Stop this rental at its frozen deadline, preserving the container disk.

Uses only the provider's existing, instance-scoped credential. It never destroys
the instance: evidence must be copied and verified from the workstation first.
"""

import argparse
import datetime as dt
import json
import os
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rental", type=Path)
    args = parser.parse_args()
    rental = json.loads(args.rental.read_text())
    instance = rental["instance_id"]
    if os.environ.get("CONTAINER_ID") != str(instance):
        raise SystemExit("Refusing to stop an instance other than this rental")
    deadline = dt.datetime.fromisoformat(rental["created_utc"]).timestamp()
    deadline += 3600 * rental["max_rental_hours"]
    print(json.dumps({"event": "watchdog_started", "instance": instance,
                      "deadline_utc": dt.datetime.fromtimestamp(
                          deadline, dt.timezone.utc).isoformat()}), flush=True)
    while time.time() < deadline:
        time.sleep(min(30, max(0, deadline - time.time())))
    root = Path("/workspace/Safety-Dial/outputs/adapter05")
    (root / "STOP").write_text("Rental deadline reached; do not admit more jobs.\n")
    for attempt in range(12):
        request = urllib.request.Request(
            f"https://console.vast.ai/api/v0/instances/{instance}/",
            data=b'{"state":"stopped"}', method="PUT",
            headers={"Authorization": "Bearer " + os.environ["CONTAINER_API_KEY"],
                     "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                result = json.load(response)
            print(json.dumps({"event": "stop_requested", "attempt": attempt,
                              "success": result.get("success") is True}), flush=True)
            if result.get("success"):
                return
        except Exception as error:
            print(json.dumps({"event": "stop_error", "kind": type(error).__name__}),
                  flush=True)
        time.sleep(10)
    raise SystemExit("Provider did not acknowledge self-stop; workstation must stop it")


if __name__ == "__main__":
    main()
