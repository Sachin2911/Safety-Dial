"""Inspect the authorized lease or hand terminal preservation to its independent guard."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm17_remote import OUT, finance, rental, utc, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "finish", "register"])
    parser.add_argument("--pid", type=int)
    parser.add_argument("--script")
    args = parser.parse_args()
    record = rental()
    if args.action == "finish":
        marker = OUT / "control/preservation-latest.json"
        saved = json.loads(marker.read_text())
        if not saved["verified"] or not saved["private"]:
            raise ValueError("Make a verified private backup before requesting finish")
        (OUT / "STOP").touch()
        write_json(OUT / "control/finish-requested.json", dict(utc=utc(), last_backup=saved))
        print("Independent guard now owns final preservation and self-destruction. Stop writing results.")
    elif args.action == "register":
        pid = args.pid or os.getppid()
        if not args.script or "lewm17" not in args.script or "guard_lewm17" in args.script:
            raise ValueError("Register only task-17 experiment worker scripts")
        proc = Path(f"/proc/{pid}")
        if args.script.encode() not in (proc / "cmdline").read_bytes().split(b"\0"):
            raise ValueError("Worker command identity does not match")
        ticks = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
        write_json(OUT / f"control/workers/{pid}.json", dict(pid=pid, start_ticks=ticks, script=args.script))
    else:
        print(json.dumps(dict(instance_id=record["instance_id"], finance=finance(record))))
        for name in ("guard-status.json", "preservation-latest.json"):
            path = OUT / "control" / name
            if path.exists():
                print(path.read_text())


if __name__ == "__main__":
    main()
