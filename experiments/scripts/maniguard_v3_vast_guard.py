"""Bound one explicitly requested Vast rental by an absolute GPU stop deadline.

Local orchestration only. Credentials are read from the existing account key file;
only minimal instance identity, timing and status fields enter the guard record.
"""

import argparse
import datetime as dt
import json
from pathlib import Path
import time
import urllib.request


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def api(key, suffix, body=None):
    request = urllib.request.Request(
        "https://console.vast.ai/api/v0/" + suffix,
        data=None if body is None else json.dumps(body).encode(),
        method="GET" if body is None else "PUT",
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        return json.load(response)


def run(settings_path, state_path):
    key = (Path.home() / ".config/vastai/vast_api_key").read_text().strip()
    bound = None
    while True:
        settings = json.loads(settings_path.read_text())
        deadline = dt.datetime.fromisoformat(settings["deadline_utc"]).timestamp()
        report = {
            "checked_utc": now(), "label": settings["label"],
            "deadline_utc": settings["deadline_utc"],
            "requested_duration_seconds": 36000,
            "action_at_deadline": "stop_only_preserve_disk",
            "initial_scope": "rent_and_prepare_only",
        }
        try:
            rows = api(key, "instances/?owner=me")["instances"]
            expected = settings.get("instance_id") or bound
            matches = [row for row in rows if (row["id"] == expected if expected is not None
                       else row.get("label") == settings["label"])]
            if len(matches) > 1:
                raise ValueError("Multiple instances match the unique new-rental label")
            if not matches:
                report["status"] = "instance_absent" if expected else "waiting_for_creation"
                atomic(state_path, report)
                print(json.dumps(report), flush=True)
                if expected or time.time() >= deadline:
                    return
                time.sleep(10)
                continue
            row = matches[0]
            if (row.get("label") != settings["label"]
                    or row["machine_id"] != settings["machine_id"]):
                raise ValueError("Bound instance identity differs from the requested rental")
            bound = row["id"]
            report.update(instance_id=bound, machine_id=row["machine_id"],
                          provider_status=row.get("actual_status"),
                          intended_status=row.get("intended_status"))
            if time.time() >= deadline:
                if row.get("actual_status") in ("stopped", "exited"):
                    report["status"] = "stopped_at_deadline"
                    atomic(state_path, report)
                    print(json.dumps(report), flush=True)
                    return
                response = api(key, f"instances/{bound}/", {"state": "stopped"})
                report.update(status="deadline_stop_requested",
                              stop_request_accepted=bool(response.get("success")))
            else:
                report["status"] = "monitoring"
        except Exception as error:
            # Exception messages can contain URLs or request details; retain type only.
            report.update(status="observation_or_stop_error", error_type=type(error).__name__)
        atomic(state_path, report)
        print(json.dumps(report), flush=True)
        time.sleep(min(20, max(0.2, deadline - time.time())) if time.time() < deadline else 10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    run(args.settings, args.state)
