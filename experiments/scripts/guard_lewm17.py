"""Detached task-17 deadline, private backup and self-destruction supervisor."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm17_remote import OUT, finance, rental, request_self, utc, write_json


def stop_registered_workers():
    for path in (OUT / "control/workers").glob("*.json"):
        try:
            worker = json.loads(path.read_text())
            pid = int(worker["pid"])
            proc = Path(f"/proc/{pid}")
            if not proc.exists() or pid == os.getpid():
                continue
            start_ticks = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
            if str(worker["start_ticks"]) != start_ticks:
                continue
            cmdline = (proc / "cmdline").read_bytes().split(b"\0")
            if not any(worker["script"].encode() == item for item in cmdline):
                continue
            if "lewm17" not in worker["script"] or "guard_lewm17" in worker["script"]:
                continue
            os.kill(pid, signal.SIGTERM)
        except (OSError, ValueError, KeyError):
            continue


def main():
    record = rental()
    (OUT / "control/workers").mkdir(parents=True, exist_ok=True)
    write_json(OUT / "control/guard-ready.json", dict(pid=os.getpid(), utc=utc()))
    next_backup, backup, closing_at = time.time() + 120, None, None
    backup_started = 0
    final_after_closeout = False
    while True:
        budget = finance(record)
        requested = (OUT / "control/finish-requested.json").exists()
        if (budget["closeout_due"] or requested) and closing_at is None:
            closing_at = time.time()
            (OUT / "STOP").touch()
            write_json(OUT / "control/closeout-started.json", dict(utc=utc(), budget=budget))
        if closing_at and time.time() - closing_at >= 30:
            stop_registered_workers()
        if backup is not None and backup.poll() is not None:
            if closing_at and backup_started >= closing_at + 45 and backup.returncode == 0:
                final_after_closeout = True
            backup = None
            next_backup = time.time() + (30 if closing_at else 600)
        if backup is None and (time.time() >= next_backup or (closing_at and not final_after_closeout)):
            # Allow workers to flush and terminate before the terminal snapshot.
            if not closing_at or time.time() >= closing_at + 45:
                backup_started = time.time()
                log = (OUT / "control/preservation.log").open("a")
                backup = subprocess.Popen(
                    [sys.executable, str(ROOT / "experiments/scripts/preserve_lewm17.py")],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
                log.close()
        write_json(OUT / "control/guard-status.json", dict(
            utc=utc(), pid=os.getpid(), instance_id=record["instance_id"], budget=budget,
            closing=closing_at is not None, final_backup_verified=final_after_closeout,
            backup_pid=backup.pid if backup else None,
        ))
        if final_after_closeout or budget["hard_stop_due"]:
            write_json(OUT / "control/destroy-request.json", dict(
                utc=utc(), instance_id=record["instance_id"],
                final_backup_verified=final_after_closeout, budget=budget,
                note="Hard lease expires even if a final upload is unavailable; earlier verified snapshots remain private.",
            ))
            try:
                result = request_self("DELETE", {})
                write_json(OUT / "control/destroy-acknowledgement.json", dict(
                    utc=utc(), success=result.get("success"),
                    external_absence_verification_pending=True,
                ))
                if result.get("success"):
                    return
            except Exception as exc:
                write_json(OUT / "control/guard-error.json", dict(utc=utc(), error_type=type(exc).__name__))
        time.sleep(10)


if __name__ == "__main__":
    main()
