"""Independent absolute-deadline stop, retaining this authorized rental's disk."""

import argparse
import calendar
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def deadline_configuration(rental, current_instance):
    instance = rental["instance_id"]
    if instance != 56777435 or str(instance) != str(current_instance):
        raise ValueError("Deadline watchdog is restricted to the authorized instance")
    deadline = rental["stop"]["deadline_utc"]
    if deadline == "2026-10-20T05:00:00Z":
        extension = rental.get("deadline_extension", {})
        if (extension.get("previous_deadline_utc") != "2026-10-20T03:00:00Z"
                or extension.get("new_deadline_utc") != deadline
                or extension.get("user_message") != "Ok great, Also you can extend the time to 5 UTC"):
            raise ValueError("The extended deadline requires the user's recorded authorization")
    elif deadline == "2026-10-20T20:00:00Z":
        extension = rental.get("deadline_extension", {})
        messages = extension.get("user_messages", [])
        if (extension.get("previous_deadline_utc") != "2026-10-20T05:00:00Z"
                or extension.get("new_deadline_utc") != deadline
                or "Go ahead I give you authorisation" not in messages
                or ("You can run the full comparison, you need to get the contribution and result, "
                    "that is most important") not in messages
                or extension.get("planned_valid_episodes") != 60
                or extension.get("maximum_full_attempts") != 64):
            raise ValueError("The full-comparison extension requires the recorded authorization")
    elif deadline != "2026-10-20T03:00:00Z":
        raise ValueError("The user's absolute deadline cannot be postponed without authorization")
    return instance, calendar.timegm(time.strptime(deadline, "%Y-%m-%dT%H:%M:%SZ"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rental", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rental = json.loads(args.rental.read_text())
    instance, deadline = deadline_configuration(rental, os.environ.get("CONTAINER_ID"))
    if not os.environ.get("CONTAINER_API_KEY"):
        raise ValueError("Vast's own-instance API credential is unavailable")

    def record(status, **extra):
        value = {"instance_id": instance, "deadline_utc": rental["stop"]["deadline_utc"],
                 "action": "stop_retaining_disk", "status": status,
                 "checked_epoch": time.time(), "pid": os.getpid(), **extra}
        temporary = args.receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(args.receipt)
        return value

    if args.check:
        print(json.dumps({"instance_id": instance, "deadline_epoch": deadline,
                          "seconds_remaining": deadline - time.time(), "credential_present": True}))
        return
    print("Armed independent stop for", instance, "at", rental["stop"]["deadline_utc"], flush=True)
    while time.time() < deadline:
        record("armed", seconds_remaining=deadline - time.time())
        time.sleep(min(30., max(0., deadline - time.time())))
    # This transition depends only on the absolute clock. Worker failures and
    # scientific outcomes cannot invoke, cancel or postpone provider shutdown.
    while True:
        record("stop_request_started")
        request = urllib.request.Request(
            f"https://console.vast.ai/api/v0/instances/{instance}/",
            data=json.dumps({"state": "stopped"}).encode(), method="PUT",
            headers={"Authorization": "Bearer " + os.environ["CONTAINER_API_KEY"],
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
            if result.get("success") is not True:
                raise RuntimeError("Provider did not acknowledge the stop request")
            record("provider_stop_acknowledged")
            print("Provider acknowledged stop retaining disk", flush=True)
            return
        except (urllib.error.URLError, OSError, RuntimeError) as exc:
            record("stop_request_failed", error_type=type(exc).__name__,
                   http_status=getattr(exc, "code", None))
            print("Deadline stop request failed:", type(exc).__name__, flush=True)
            time.sleep(10)


if __name__ == "__main__":
    main()
