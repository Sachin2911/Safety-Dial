"""Bound storage charges on the stopped, backed-up setup while approval is pending."""

import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.unisafe09_infra import atomic_json, instances, provider_stopped, utc


def main():
    rental_path = ROOT / "progress/rental-unisafe09-20261025.json"
    marker = ROOT / "progress/receipts/09-unisafe-setup-preservation.json"
    rental = json.loads(rental_path.read_text())
    instance = rental["instance_id"]
    deadline = dt.datetime.fromisoformat(rental["allocation_deadline_utc"]).timestamp()
    backup = json.loads((ROOT / "outputs/unisafe09/control/approval-setup-backup.json").read_text())
    if not backup["no_simulator_attempts"] or backup["instance_id"] != instance:
        raise ValueError("This guard is only for the unused, backed-up setup")
    while True:
        current = json.loads(rental_path.read_text())
        if current["instance_id"] != instance or not current.get("approval_pending"):
            return
        atomic_json(
            ROOT / "outputs/unisafe09/control/approval-guard-status.json",
            {
                "utc": utc(),
                "instance_id": instance,
                "cleanup_deadline_utc": rental["allocation_deadline_utc"],
                "status": "waiting_for_transfer_approval",
            },
        )
        if time.time() >= deadline:
            try:
                row = next((r for r in instances() if r["id"] == instance), None)
                if not provider_stopped(row):
                    # An externally resumed host may contain new work. Do not erase it.
                    atomic_json(
                        ROOT / "outputs/unisafe09/control/approval-guard-status.json",
                        {
                            "utc": utc(),
                            "instance_id": instance,
                            "status": "external_resume_detected_no_destruction",
                        },
                    )
                    return
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "experiments/scripts/manage_unisafe09.py"),
                        "destroy",
                        "--marker",
                        str(marker),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode:
                    raise RuntimeError("Setup destruction was not verified")
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                atomic_json(
                    ROOT / "outputs/unisafe09/control/approval-guard-status.json",
                    {
                        "utc": utc(),
                        "instance_id": instance,
                        "status": "cleanup_retry_pending",
                        "error_type": type(error).__name__,
                    },
                )
                time.sleep(30)
                continue
            atomic_json(
                ROOT / "outputs/unisafe09/control/approval-guard-status.json",
                {"utc": utc(), "instance_id": instance, "status": "unused_setup_destroyed"},
            )
            return
        time.sleep(min(30, max(0, deadline - time.time())))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__}), flush=True)
        raise SystemExit(1) from None
