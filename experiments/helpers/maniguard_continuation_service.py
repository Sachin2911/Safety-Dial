"""Snapshot new requests from an unchanged service's append-only evidence ledger."""

import hashlib
import json
from pathlib import Path

from helpers.maniguard_continuation_guardian import atomic_record


def ledger_path(protocol, worker):
    expected = Path(protocol["original_run"]) / f"server-jar-worker{worker}-physical.jsonl"
    declared = Path(protocol["workers"][str(worker)]["service_ledger"])
    if declared != expected or declared.is_symlink():
        raise ValueError("Reuse only the declared original policy service ledger")
    return expected


def service_baseline(protocol, worker):
    path = ledger_path(protocol, worker)
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("The idle service ledger has an incomplete record")
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def capture_service_records(output):
    """Run only after the guardian exits and before another start uses its lease."""
    output = Path(output)
    job = json.loads((output / "job.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    baseline = job["service_ledger_baseline"]
    path = ledger_path(protocol, job["worker"])
    if baseline["path"] != str(path):
        raise ValueError("The service baseline belongs to a different worker")
    raw = path.read_bytes()
    offset = baseline["bytes"]
    if (type(offset) is not int or not 0 <= offset <= len(raw)
            or hashlib.sha256(raw[:offset]).hexdigest() != baseline["sha256"]):
        raise ValueError("The original append-only service history changed or was truncated")
    suffix = raw[offset:]
    if suffix and not suffix.endswith(b"\n"):
        raise ValueError("The service has an incomplete request record")
    records = [json.loads(line) for line in suffix.splitlines()]
    if any(row["episode"] != job["episode_id"] for row in records):
        raise ValueError("Another episode used this service within the charged start")
    snapshot = output / "server-receipts.jsonl"
    if snapshot.exists() and snapshot.read_bytes() != suffix:
        raise ValueError("An earlier service snapshot differs from the new readback")
    if not snapshot.exists():
        with snapshot.open("xb") as stream:
            stream.write(suffix)
    atomic_record(output / "service-ledger-readback.json", {
        "baseline": baseline, "bytes": len(suffix), "sha256": hashlib.sha256(suffix).hexdigest(),
        "records": len(records), "episode": job["episode_id"],
        "scope": "Copied service tail only; original ledger was not modified. A failed or "
                 "interrupted start can leave unknown in-flight work and remains charged.",
    })
    return records


def verify_service_records(output, job, rows):
    """Restorable verification uses copied bytes, without the original live service."""
    output = Path(output)
    raw = (output / "server-receipts.jsonl").read_bytes()
    saved = json.loads((output / "service-ledger-readback.json").read_text())
    records = [json.loads(line) for line in raw.splitlines()]
    actual = [row["receipt"] for row in rows if row["type"] == "noise_receipt"]
    if (saved["baseline"] != job["service_ledger_baseline"]
            or saved["bytes"] != len(raw) or saved["sha256"] != hashlib.sha256(raw).hexdigest()
            or saved["records"] != len(records) or saved["episode"] != job["episode_id"]
            or records != actual):
        raise ValueError("The saved service requests differ from actual returned policy receipts")
    return len(records)
