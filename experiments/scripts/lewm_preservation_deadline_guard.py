"""Credential-free heartbeat and process cutoff for a newly authorized completion window."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import signal
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--pod-id", required=True)
    p.add_argument("--deadline", required=True)
    args = p.parse_args()
    if os.environ.get("RUNPOD_POD_ID") != args.pod_id:
        raise RuntimeError("Guard belongs on the explicitly authorized pod")
    root = args.project_root.resolve()
    deadline = dt.datetime.fromisoformat(args.deadline)
    heartbeat = root / "environment/deadline-guard.json"
    while True:
        stamp = dt.datetime.now(dt.timezone.utc)
        expired = stamp >= deadline
        value = {
            "pod_id": args.pod_id,
            "heartbeat_utc": stamp.isoformat(),
            "deadline_utc": deadline.isoformat(),
            "status": "expired" if expired else "armed",
            "authority": "explicit comparison-completion extension; original campaign preserved",
        }
        temporary = heartbeat.with_suffix(".part")
        temporary.write_text(json.dumps(value) + "\n")
        temporary.replace(heartbeat)
        if expired:
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit() or int(entry.name) == os.getpid():
                    continue
                try:
                    argv = (entry / "cmdline").read_bytes().split(b"\0")
                    cwd = Path(os.readlink(entry / "cwd"))
                    related = cwd.is_relative_to(root) or any(str(root).encode() in a for a in argv)
                    python = bool(argv) and Path(os.fsdecode(argv[0])).name.startswith(
                        ("python", "pytest", "torchrun")
                    )
                    if related and python:
                        os.kill(int(entry.name), signal.SIGTERM)
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    pass
            return
        time.sleep(5)


if __name__ == "__main__":
    main()
