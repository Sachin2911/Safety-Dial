"""Observe or stop only the current authorized Part 4 rental, retaining its disk."""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
from helpers.maniguard_part4_protocol import CONFIG, ROOT, RUN, atomic, clock, read


def request(method="GET"):
    instance = read(CONFIG / "allocation.json")["instance"]
    if os.environ.get("CONTAINER_ID") != str(instance):
        raise ValueError("Allocation is not for the current instance")
    url = f"https://console.vast.ai/api/v0/instances/{instance}/"
    payload = json.dumps({"state": "stopped"}).encode() if method == "PUT" else None
    req = urllib.request.Request(
        url + ("?owner=me" if method == "GET" else ""),
        headers={"Authorization": "Bearer " + os.environ["CONTAINER_API_KEY"],
                 "Content-Type": "application/json"},
        data=payload, method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.load(response)
        if method == "PUT":
            return {"success": data.get("success") is True, "at": time.time()}
        rows = data.get("instances")
        rows = rows if isinstance(rows, list) else [rows]
        row = next((r for r in rows if r and r.get("id") == instance), None)
        if row is None:
            raise ValueError("Expected provider instance missing")
        return {k: row.get(k) for k in (
            "id", "actual_status", "intended_status", "start_date", "gpu_name", "num_gpus"
        )}
    except urllib.error.HTTPError as error:
        return {"error": "HTTPError", "http_status": error.code}
    except (OSError, ValueError) as error:
        return {"error": type(error).__name__}


def stop(hard_deadline=False, publish_status=False):
    if not hard_deadline:
        receipt = read(RUN / "final-preservation.json")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        if (not receipt.get("ready_for_provider_stop") or head != receipt.get("git_commit")
                or head != receipt.get("remote_main")):
            raise ValueError("Verify final GitHub and private HF preservation first")
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
        ).strip()
        if gpu:
            raise ValueError("Active GPU compute must exit first")
    elif clock()["hard_remaining_seconds"] > 0:
        raise ValueError("Hard-deadline override is not yet authorized")
    observations = []
    result = {"action": "stop_retaining_disk", "hard_deadline": hard_deadline,
              "clock": clock(), "actual_stopped_state_verified": False}
    atomic(RUN / "provider-stop-intent.json", result)
    for index in range(12):
        if index in (0, 4, 8):
            result["response"] = request("PUT")
            atomic(RUN / "provider-stop-response.json", result)
        observed = {**request(), "clock": clock()}
        observations.append(observed)
        result.update(observations=observations, actual_stopped_state_verified=(
            observed.get("actual_status") == "stopped"
            and observed.get("intended_status") == "stopped"
        ))
        atomic(RUN / "provider-stop-verification.json", result)
        print(json.dumps(observed), flush=True)
        if publish_status:
            from huggingface_hub import HfApi

            repo = "Sachioster/safety-dial-phase3-part4-20261019"
            api = HfApi()
            if api.dataset_info(repo).private is not True:
                raise ValueError("Provider state destination must remain private")
            api.upload_file(
                path_or_fileobj=str(RUN / "provider-stop-verification.json"),
                path_in_repo="closeout/provider-stop-verification.json",
                repo_id=repo, repo_type="dataset",
                commit_message="Record Part4 provider stop observation",
            )
        if result["actual_stopped_state_verified"]:
            return result
        time.sleep(5)
    return result


def watch():
    while clock()["hard_remaining_seconds"] > 0:
        atomic(RUN / "independent-guard.json", {"pid": os.getpid(), "clock": clock()})
        time.sleep(min(30, max(0.1, clock()["hard_remaining_seconds"])))
    stop(hard_deadline=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "stop", "watch"))
    parser.add_argument("--publish-status", action="store_true")
    args = parser.parse_args()
    if args.action == "watch":
        watch()
    elif args.action == "stop":
        stop(publish_status=args.publish_status)
    else:
        result = {**request(), "clock": clock()}
        atomic(RUN / "provider-status.json", result)
        print(json.dumps(result))
