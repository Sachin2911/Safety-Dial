"""Part 4 paths, atomic records, immutable protocol hashes and rental clock."""

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/maniguard/part4"
RUN = ROOT / "runs/maniguard/phase3-part4/20261019-56676451"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def clock(now=None):
    allocation = read(CONFIG / "allocation.json")
    now = time.time() if now is None else now
    return {
        "at": now,
        "elapsed_seconds": now - allocation["provider_start_epoch"],
        "allocated_gpu_hours": (now - allocation["provider_start_epoch"]) / 3600,
        "physical_remaining_seconds": (
            datetime.fromisoformat(allocation["physical_cutoff_utc"]).timestamp() - now
        ),
        "hard_remaining_seconds": (
            datetime.fromisoformat(allocation["hard_deadline_utc"]).timestamp() - now
        ),
    }


def assert_partitioning(partitions):
    seen = set()
    for group, tasks in partitions.items():
        for task in tasks:
            base = task.split("/")[0]
            if base in seen:
                raise ValueError("Base task leaks across groups: " + base)
            seen.add(base)
    return True
