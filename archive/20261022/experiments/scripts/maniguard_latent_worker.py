"""Persist an episode child's exit status independently of its coordinator."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    started = time.time()
    code = subprocess.call(request["command"], cwd=request["cwd"])
    receipt = {"command": request["command"], "started_epoch": started,
               "finished_epoch": time.time(), "exit_code": code}
    temporary = args.receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(args.receipt)
    return 0 if code == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
