"""Stop this Vast instance from inside it, after the todo 1c evidence is preserved.

Stopping ends GPU billing (the disk is billed until the instance is destroyed from the
account, which the reviewer does after verifying the upload). The provider call is the
same one the October rentals used: PUT /api/v0/instances/<id>/ with {"state": "stopped"}
and the instance's own CONTAINER_API_KEY, which is never printed.

Guards, all of which must pass:
  1. CONTAINER_ID is the expected instance (argument --instance).
  2. The repository has no uncommitted changes and HEAD equals origin/main after a fetch.
  3. the preservation receipt (--receipt, default progress/receipts/01c-preserve.md)
     exists and names a 40-hex dataset revision.
  4. No compute process is on either GPU (policy servers and simulators are gone).

    python -I scripts/todo1c/stop_instance.py --instance <id> [--receipt <path>] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
API = "https://console.vast.ai/api/v0/instances/{id}/"


def sh(*args: str) -> str:
    return subprocess.check_output(list(args), cwd=ROOT, text=True).strip()


def request(instance: int, method: str) -> dict:
    headers = {"Authorization": "Bearer " + os.environ["CONTAINER_API_KEY"],
               "Content-Type": "application/json"}
    url = API.format(id=instance) + ("?owner=me" if method == "GET" else "")
    data = json.dumps({"state": "stopped"}).encode() if method == "PUT" else None
    req = urllib.request.Request(url, headers=headers, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as err:
        return {"error": "HTTPError", "http_status": err.code}
    except (OSError, ValueError) as err:
        return {"error": type(err).__name__}
    if method == "PUT":
        return {"success": body.get("success") is True}
    rows = body.get("instances")
    rows = rows if isinstance(rows, list) else [rows]
    row = next((r for r in rows if r and r.get("id") == instance), None)
    if row is None:
        return {"error": "instance missing from provider response"}
    return {k: row.get(k) for k in ("id", "actual_status", "intended_status", "cur_state")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=int, required=True)
    ap.add_argument("--receipt", default="progress/receipts/01c-preserve.md",
                    help="preservation receipt that must name the dataset revision (repo-relative)")
    ap.add_argument("--dry-run", action="store_true", help="run the guards, do not stop")
    args = ap.parse_args()

    if os.environ.get("CONTAINER_ID") != str(args.instance):
        sys.exit(f"guard 1: CONTAINER_ID is not {args.instance}")
    if "CONTAINER_API_KEY" not in os.environ:
        sys.exit("guard 1: CONTAINER_API_KEY is not set")

    if sh("git", "status", "--porcelain"):
        sys.exit("guard 2: uncommitted changes in the repository")
    sh("git", "fetch", "-q", "origin", "main")
    head, remote = sh("git", "rev-parse", "HEAD"), sh("git", "rev-parse", "origin/main")
    if head != remote:
        sys.exit(f"guard 2: HEAD {head[:8]} differs from origin/main {remote[:8]}; push first")

    receipt = os.path.join(ROOT, args.receipt)
    if not os.path.exists(receipt):
        sys.exit(f"guard 3: {args.receipt} is missing")
    if not re.search(r"\b[0-9a-f]{40}\b", open(receipt).read()):
        sys.exit(f"guard 3: {args.receipt} names no 40-hex dataset revision")

    gpu = sh("nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader")
    if gpu:
        sys.exit("guard 4: compute processes still on the GPUs:\n" + gpu)

    before = request(args.instance, "GET")
    print("provider state before:", before)
    if args.dry_run:
        print("dry run: all guards passed, not stopping")
        return 0
    result = request(args.instance, "PUT")
    print("stop request:", result)
    if not result.get("success"):
        return 1
    for _ in range(24):
        time.sleep(10)
        state = request(args.instance, "GET")
        print("provider state:", state)
        if state.get("actual_status") in ("stopped", "exited") or state.get("cur_state") == "stopped":
            return 0
    print("stop requested; provider has not reported stopped within 4 minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
