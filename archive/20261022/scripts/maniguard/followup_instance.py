"""Read or stop the authorized instance, retaining disk and bounded state verification."""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

from backup_followup import ROOT, RUN, digest

sys.path.insert(0, str(ROOT / "experiments"))
from helpers.maniguard_followup_budget import atomic, clock

INSTANCE = 56582094
URL = f"https://console.vast.ai/api/v0/instances/{INSTANCE}/"
FIELDS = (
    "id",
    "actual_status",
    "intended_status",
    "start_date",
    "gpu_name",
    "num_gpus",
    "driver_version",
)


def request(method="GET"):
    headers = {
        "Authorization": "Bearer " + os.environ["CONTAINER_API_KEY"],
        "Content-Type": "application/json",
    }
    payload = json.dumps({"state": "stopped"}).encode() if method == "PUT" else None
    req = urllib.request.Request(
        URL + ("?owner=me" if method == "GET" else ""), headers=headers, data=payload, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.load(response)
            if method == "PUT":
                return {
                    "http_status": response.status,
                    "success": data.get("success") is True,
                    "response_keys": sorted(data),
                }
            row = data.get("instances")
            if isinstance(row, list):
                row = next((r for r in row if r.get("id") == INSTANCE), None)
            if not row or row.get("id") != INSTANCE:
                raise ValueError("Expected instance missing from provider response")
            return {"http_status": response.status, **{k: row.get(k) for k in FIELDS}}
    except urllib.error.HTTPError as error:
        return {"http_status": error.code, "error": "HTTPError"}
    except (OSError, ValueError, KeyError) as error:
        return {"error": type(error).__name__}


def stop():
    if os.environ.get("CONTAINER_ID") != str(INSTANCE):
        raise ValueError("Current instance identity differs from the authorized replacement")
    preservation = json.loads((RUN / "final-preservation.json").read_text())
    if not preservation["ready_for_provider_stop"]:
        raise ValueError(
            "Final tested reports, remote main and private backups must be verified first"
        )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != preservation["git_commit"] or head != preservation["remote_main"]:
        raise ValueError("Current code differs from the final preservation receipt")
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if gpu:
        raise ValueError("GPU compute workers must exit before rental stop")
    terminal = json.loads((RUN / "part3/terminal.json").read_text())
    if terminal["status"] not in ("scientifically_gated", "complete"):
        raise ValueError("The experiment has no terminal closeout outcome")
    intent = {
        "schema": "maniguard-followup-provider-stop-v1",
        "instance": INSTANCE,
        "action": "stop_retaining_disk",
        "requested_at": time.time(),
        "clock_at_request": clock(),
        "git_commit": head,
        "evidence_preservation_sha256": digest(RUN / "final-preservation.json"),
        "source": "https://github.com/vast-ai/vast-cli/blob/master/vast.py",
        "actual_stopped_state_verified": False,
    }
    atomic(RUN / "provider-stop-intent.json", intent)
    print(json.dumps(intent), flush=True)
    attempts, observations = [], []
    for index in range(12):
        if index in (0, 4, 8) and (
            not observations or observations[-1].get("intended_status") != "stopped"
        ):
            attempted = {**request("PUT"), "at": time.time(), "clock": clock()}
            attempts.append(attempted)
            atomic(
                RUN / "provider-stop-response.json",
                {**intent, "attempts": attempts, "response_at": time.time()},
            )
        observed = {**request(), "observed_at": time.time(), "clock": clock()}
        observations.append(observed)
        verified = (
            observed.get("actual_status") == "stopped"
            and observed.get("intended_status") == "stopped"
        )
        result = {
            **intent,
            "attempts": attempts,
            "observations": observations,
            "actual_stopped_state_verified": verified,
            "clock_at_last_observation": clock(),
            "scope": "Only provider actual=stopped and intended=stopped is marked verified. Other states remain explicit, without equating an acknowledgement to shutdown.",
        }
        atomic(RUN / "provider-stop-verification.json", result)
        print(
            json.dumps({"poll": index, **observed, "actual_stopped_state_verified": verified}),
            flush=True,
        )
        if verified:
            return result
        time.sleep(5)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "stop"))
    args = parser.parse_args()
    if args.action == "status":
        row = {**request(), "observed_at": time.time(), "clock": clock()}
        atomic(RUN / "provider-status-before-closeout.json", row)
        print(json.dumps(row))
    else:
        stop()
