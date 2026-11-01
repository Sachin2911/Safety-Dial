"""Seal a completed smoke project after retiring its exact remote process guard."""

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import tarfile
import time


def read_metadata(path):
    raw = path.read_bytes()
    try:
        return json.loads(raw), None
    except json.JSONDecodeError:
        if raw.endswith(b"\\n"):
            return json.loads(raw[:-2]), "Literal terminal backslash-n; original bytes retained"
        raise


def main():
    root = Path(__file__).resolve().parents[2]
    design = json.loads((root / "design.json").read_text())
    if os.environ.get("RUNPOD_POD_ID") != design["pod_id"]:
        raise RuntimeError("Only the declared fresh rented pod can package this project")
    audit = json.loads((root / "results/attempt-002/audit.json").read_text())
    if not audit["integration_verified"] or audit["optimizer_updates"] != 0:
        raise ValueError("Completed untrained integration audit required")
    if not (root / "results/attempt-002/media/episode.mp4").is_file():
        raise ValueError("Preserve the saved-frame replay before sealing")
    lifecycle = root / "lifecycle"
    lifecycle.mkdir(exist_ok=True)
    records = {}
    for path in (root / "environment").glob("*.json"):
        record, qualification = read_metadata(path)
        if "pid" not in record:
            continue
        records[path.name] = {"record": record, "encoding_qualification": qualification}
        proc = Path("/proc") / str(record["pid"])
        if not proc.exists():
            continue
        stat = (proc / "stat").read_text().split()
        if stat[21] != str(record["start_ticks"]) or stat[2] == "Z":
            continue
        if path.name == "deadline-process.json":
            command = (proc / "cmdline").read_bytes()
            if b"/root/.config/lewm-proposal-smoke/deadline.py" not in command:
                raise RuntimeError("Guard process identity differs")
            os.kill(record["pid"], signal.SIGTERM)
            time.sleep(0.2)
            if proc.exists() and (proc / "stat").read_text().split()[2] != "Z":
                raise RuntimeError("Remote guard has not retired")
        else:
            raise RuntimeError("A recorded project process is still running: " + path.name)
    account = Path("/root/.config/lewm-proposal-smoke")
    if (account / "account.env").exists():
        raise RuntimeError("Unexpected remote account credential file")
    for name in ("deadline.py", "lifecycle.json"):
        shutil.copy2(account / name, lifecycle / name)
    (lifecycle / "process-closeout.json").write_text(
        json.dumps(
            {
                "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "records": records,
                "experiment_processes_retired": True,
                "remote_guard_retired": True,
                "provider_guard": "Independent local controller remains active until deletion",
                "remote_provider_credential_present": False,
            },
            indent=2,
        )
        + "\n"
    )
    entries = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(p in ("__pycache__", ".pytest_cache", ".ruff_cache") for p in relative.parts):
            continue
        if path.is_symlink() or path.name.startswith(".env"):
            raise RuntimeError("Unexpected link or environment file in project")
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        entries.append({"path": str(relative), "bytes": path.stat().st_size, "sha256": sha})
    archive = root.parent / "lewm-proposal-smoke-20261030-v1.tar.gz"
    manifest = root.parent / "lewm-proposal-smoke-manifest.json"
    if archive.exists() or manifest.exists():
        raise FileExistsError("Do not overwrite an earlier terminal artifact")
    with tarfile.open(archive, "w:gz") as tar:
        for item in entries:
            tar.add(root / item["path"], arcname=item["path"], recursive=False)
    with tarfile.open(archive, "r:gz") as tar:
        for item in entries:
            with tar.extractfile(item["path"]) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != item["sha256"]:
                raise RuntimeError("Archive readback failed")
    with archive.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {
        "pod_id": design["pod_id"],
        "archive": archive.name,
        "archive_sha256": sha,
        "archive_bytes": archive.stat().st_size,
        "files": entries,
        "file_count": len(entries),
        "total_file_bytes": sum(x["bytes"] for x in entries),
        "remote_readback_verified": True,
        "audit": audit,
    }
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "pod_id",
                    "archive",
                    "archive_sha256",
                    "archive_bytes",
                    "file_count",
                    "total_file_bytes",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
