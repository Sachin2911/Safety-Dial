"""Read-only cost snapshots from every attempt, including current partial work."""

import json
import time
from pathlib import Path

from helpers.maniguard_latent_queue import ROOT, atomic


def snapshot(run):
    run = Path(run).resolve()
    attempts = {}
    for line in (run / "attempts.jsonl").read_text().splitlines():
        row = json.loads(line)
        attempts[row["id"]] = row
    queue_path = run / "queue.json"
    queue = json.loads(queue_path.read_text()) if queue_path.exists() else {"attempts": []}
    attempts.update({row["id"]: row for row in queue["attempts"]})
    counters = ["physics_steps", "physics_seconds", "env_step_attempts", "env_steps_returned",
                "policy_control_steps", "policy_queries", "policy_query_seconds",
                "encoder_images", "encoder_image_calls", "encoder_text_calls", "encoder_seconds",
                "encoder_load_seconds"]
    totals = dict.fromkeys(counters, 0)
    engineering_controls, rows = 0, []
    for row in attempts.values():
        output = Path(row.get("output", run / row["id"]))
        if not output.is_absolute():
            output = ROOT / output
        path = output / "ledger.json"
        ledger = json.loads(path.read_text()) if path.exists() else {}
        for key in counters:
            totals[key] += ledger.get(key, 0)
        if row["stage"] == "engineering":
            engineering_controls += ledger.get("policy_control_steps", 0)
        rows.append({"id": row["id"], "status": row["status"], "stage": row["stage"],
                     "output": str(output), "worker": row.get("worker"),
                     "phase": ledger.get("phase"), "ledger_elapsed_seconds": ledger.get("elapsed_seconds"),
                     "policy_control_steps": ledger.get("policy_control_steps"),
                     "physics_accounting_attached": ledger.get("physics_accounting_attached")})
    full = [row for row in queue["attempts"] if not row["engineering"]]
    engineering = sum(row["stage"] == "engineering" for row in attempts.values())
    decoder_calls = {}
    for family in ["jar", "clutter"]:
        path = run / ("authority-" + family) / "requests.jsonl"
        decoder_calls[family] = len(path.read_text().splitlines()) if path.exists() else 0
    servers = []
    for path in sorted(run.glob("server-*.jsonl")):
        # A live writer can be between writes. Keep complete records and expose
        # any incomplete tail rather than treating it as a returned call.
        contents = path.read_text()
        lines = contents.splitlines()
        pending = bool(contents and not contents.endswith("\n"))
        calls = [json.loads(line) for line in (lines[:-1] if pending else lines)]
        servers.append({"ledger": path.name, "calls": len(calls),
                        "returned": sum(row.get("status") == "returned" for row in calls),
                        "failed": sum(row.get("status") == "failed" for row in calls),
                        "elapsed_seconds": sum(row.get("elapsed_seconds", 0) for row in calls),
                        "incomplete_tail": pending})
    rental = json.loads((ROOT / "configs/vast/maniguard-rental-20261019.json").read_text())
    now = time.time()
    wall_hours = (now - rental["provider_started_epoch"]) / 3600
    return {"schema": "maniguard-latent-cost-snapshot-v1", "observed_epoch": now,
            "attempts": rows, "unique_attempts": len(attempts), "recorded_totals": totals,
            "decoder_calls": decoder_calls, "decoder_calls_remaining": 320 - sum(decoder_calls.values()),
            "server_model_calls": servers,
            "engineering_starts": engineering, "engineering_starts_remaining": 6 - engineering,
            "engineering_controls": engineering_controls, "engineering_controls_remaining": 96 - engineering_controls,
            "full_attempts_started": len(full), "full_attempts_remaining": 64 - len(full),
            "full_valid_completed": sum(row["status"] == "completed" for row in full),
            "full_invalid": sum(row["status"] == "invalid" for row in full),
            "allocation_wall_hours": wall_hours, "allocated_gpu_hours": 2 * wall_hours,
            "rental_usd_accrued_at_observed_rate": wall_hours * rental["pricing"]["observed_hourly_usd_with_disk"],
            "limitations": "Partial ledgers are lower bounds. Physics before callback attachment is unknown. No missing counter is evidence of zero work. Both allocated GPUs, setup, idle time and preservation count from the original provider start; transfer and stopped-storage billing are separate."}


def refresh(run):
    run = Path(run)
    result = snapshot(run)
    atomic(run / "cost-snapshot.json", result)
    return result
