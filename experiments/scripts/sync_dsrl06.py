"""Continuously copy task-6 evidence locally while its compute remains active."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import FinancialBook, atomic_json, utc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=45)
    args = parser.parse_args()
    control = ROOT / "outputs/dsrl06/control"
    control.mkdir(parents=True, exist_ok=True)
    while True:
        snapshot = FinancialBook(control / "finance.json").snapshot()
        if not snapshot["active_instances"] and not snapshot["pending_labels"]:
            return
        started = utc()
        try:
            result = subprocess.run(
                [sys.executable, str(ROOT / "experiments/scripts/manage_dsrl06.py"), "sync"],
                capture_output=True,
                text=True,
                timeout=200,
            )
            status = {"exit_code": result.returncode}
        except subprocess.TimeoutExpired:
            status = {"error_type": "SyncObservationTimeout"}
        # Raw network exceptions are deliberately not printed.
        record = {"started_utc": started, "ended_utc": utc(), **status}
        atomic_json(control / "sync-status.json", record)
        print(json.dumps(record), flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
