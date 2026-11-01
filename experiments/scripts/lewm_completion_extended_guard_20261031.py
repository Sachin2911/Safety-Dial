"""Prospective operational time revision under the unchanged user-approved $13 cap."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import lewm_campaign_pods as prior
import lewm_completion_pods_20261031 as completion
import lewm_completion_replacement_20261031 as replacement

OUT = replacement.OUT
EXTENSION = completion.OUT / "completion-window-extension-v1.json"


def arm():
    if EXTENSION.exists():
        raise RuntimeError("Existing extension; inspect without replay")
    original_path = completion.OUT / "authorization.json"
    original = prior.read(original_path)
    begin = dt.datetime.fromisoformat(original["authorized_utc"])
    deadline = begin + dt.timedelta(minutes=120)
    pod = prior.read(OUT / "session.json")
    at = dt.datetime.now(dt.timezone.utc)
    cost = replacement.amount(at)
    reserved = cost + max(0, (deadline - at).total_seconds()) / 3600 * 0.65
    if reserved >= 12.70:
        raise RuntimeError("Revised time does not fit the unchanged money cap")
    value = {
        "schema": "lewm-completion-operational-window-extension-v1",
        "recorded_utc": at.isoformat(),
        "original_authority": {
            "path": str(original_path),
            "sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        },
        "original_deadline_utc": original["deadline_utc"],
        "new_deadline_utc": deadline.isoformat(),
        "maximum_duration_seconds_from_original_extension_start": 7200,
        "total_campaign_cap_usd": 13,
        "user_authority": "Go ahead with the full comparison, $3 more; destroy the pods afterward.",
        "operational_time_extension_set_by_agent": True,
        "reason": "Original hosts lost GPU capacity; zero-GPU recovery and full-file verification took longer than the initial 60-90 minute estimate. No new outcomes have been collected.",
        "no_new_training_cases_parameters_seeds_or_experiment_budgets": True,
        "replacement_pod_id": pod["id"],
        "conservative_cost_at_revision_usd": cost,
        "conservative_reserved_bound_usd": reserved,
    }
    prior.write(EXTENSION, value)
    old = prior.read(OUT / "guard-process.json")
    prior.write(OUT / "guard-process-before-window-extension.json", old)
    with (OUT / "extended-guard-v1.log").open("ab", buffering=0) as log:
        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "guard"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    current = {
        "pid": worker.pid,
        "start_ticks": Path(f"/proc/{worker.pid}/stat").read_text().split()[21],
    }
    prior.write(OUT / "extended-guard-process-v1.json", current)
    heartbeat = OUT / "extended-guard-state-v1.json"
    for _ in range(30):
        if heartbeat.exists():
            break
        time.sleep(0.1)
    if not heartbeat.exists():
        raise RuntimeError("New guard failed; old guard remains armed")
    process = Path(f"/proc/{old['pid']}/stat")
    if process.exists():
        fields = process.read_text().split()
        if fields[2] != "Z" and fields[21] == str(old["start_ticks"]):
            os.kill(old["pid"], signal.SIGTERM)
    print(
        json.dumps(
            {
                **value,
                "new_provider_guard": current,
                "previous_provider_guard_retired_after_new_heartbeat": True,
            }
        ),
        flush=True,
    )


def guard():
    value = prior.read(EXTENSION)
    deadline = dt.datetime.fromisoformat(value["new_deadline_utc"])
    if value["total_campaign_cap_usd"] != 13:
        raise RuntimeError("Money cap changed")
    while not (OUT / "extended-guard-cancelled").exists():
        at = dt.datetime.now(dt.timezone.utc)
        amount = replacement.amount(at)
        prior.write(
            OUT / "extended-guard-state-v1.json",
            {
                "heartbeat_utc": at.isoformat(),
                "deadline_utc": deadline.isoformat(),
                "conservative_total_usd": amount,
                "total_cap_usd": 13,
            },
        )
        if at >= deadline or amount >= 12.70:
            try:
                replacement.stop()
            except Exception as exc:
                prior.write(
                    OUT / "extended-guard-error-v1.json",
                    {"utc": at.isoformat(), "type": type(exc).__name__},
                )
        time.sleep(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("arm", "guard"))
    args = parser.parse_args()
    globals()[args.operation]()
