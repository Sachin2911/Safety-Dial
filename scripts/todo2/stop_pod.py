"""Stop this RunPod pod from inside it, after the checklist's evidence is preserved.

Stopping ends GPU billing (the volume is billed until the pod is terminated from the
account, which the reviewer does after verifying the upload). The call is RunPod's
GraphQL `podStop` with the pod's own id (`RUNPOD_POD_ID`) and the account key
(`RUNPOD_API_KEY`), which is never printed.

Guards, all of which must pass:
  1. RUNPOD_POD_ID and RUNPOD_API_KEY are set.
  2. The repository has no uncommitted changes and HEAD equals origin/main after a fetch.
  3. The preservation receipt (--receipt) exists and names a 40-hex dataset revision.
  4. No compute process is on the GPU.

    python -I scripts/todo2/stop_pod.py --receipt progress/receipts/02-preserve.md [--dry-run]
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
API = "https://api.runpod.io/graphql"


def sh(*args: str) -> str:
    return subprocess.check_output(list(args), cwd=ROOT, text=True).strip()


def gql(query: str, variables: dict) -> dict:
    req = urllib.request.Request(API, data=json.dumps({"query": query, "variables": variables}).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + os.environ["RUNPOD_API_KEY"]})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as err:
        return {"error": "HTTPError", "http_status": err.code}
    except (OSError, ValueError) as err:
        return {"error": type(err).__name__}


def state(pod_id: str) -> dict:
    r = gql("query($id: String!) { pod(input: {podId: $id}) { id desiredStatus runtime { uptimeInSeconds } } }",
            {"id": pod_id})
    return (r.get("data") or {}).get("pod") or r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", required=True, help="preservation receipt that must name the dataset revision")
    ap.add_argument("--dry-run", action="store_true", help="run the guards, do not stop")
    args = ap.parse_args()

    pod_id = os.environ.get("RUNPOD_POD_ID")
    if not pod_id or "RUNPOD_API_KEY" not in os.environ:
        sys.exit("guard 1: RUNPOD_POD_ID or RUNPOD_API_KEY is not set")

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
        sys.exit("guard 4: compute processes still on the GPU:\n" + gpu)

    print("pod state before:", state(pod_id))
    if args.dry_run:
        print("dry run: all guards passed, not stopping")
        return 0
    r = gql("mutation($id: String!) { podStop(input: {podId: $id}) { id desiredStatus } }", {"id": pod_id})
    print("stop request:", r)
    if "errors" in r or "error" in r:
        return 1
    for _ in range(24):
        time.sleep(10)
        s = state(pod_id)
        print("pod state:", s)
        if s.get("desiredStatus") == "EXITED":
            return 0
    print("stop requested; pod has not reported EXITED within 4 minutes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
