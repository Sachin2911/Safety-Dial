"""Seal verified pilot evidence after retiring its exact credential-free guard."""

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import tarfile
import time


def main():
    root = Path(__file__).resolve().parents[2]
    design = json.loads((root / "configs/lewm_screen_20261031/protocol.json").read_text())
    if os.environ.get("RUNPOD_POD_ID") != design["execution_pod_id"]:
        raise RuntimeError("Only the declared rented pod may package this project")
    result_root = root / "results/screen-v1"
    audit = json.loads((result_root / "audit.json").read_text())
    if not audit["verified"] or audit["optimizer_updates"] != 0:
        raise RuntimeError("Complete independent saved-evidence audit required")
    media = json.loads((result_root / "media/provenance.json").read_text())
    state = json.loads((result_root / "state.json").read_text())
    expected = sum(
        r["category"] == "scored" and r["role"] == "native" for r in state["jobs"].values()
    )
    if len(media["replays"]) != expected or any(
        not (result_root / "media" / r["video"]).is_file() for r in media["replays"]
    ):
        raise RuntimeError("All declared saved-frame replays must be preserved")
    lifecycle = root / "lifecycle"
    lifecycle.mkdir(exist_ok=False)
    records = {}
    for path in (root / "environment").glob("*.json"):
        record = json.loads(path.read_text())
        if "pid" not in record:
            continue
        records[path.name] = record
        proc = Path("/proc") / str(record["pid"])
        if not proc.exists():
            continue
        stat = (proc / "stat").read_text().split()
        if stat[2] == "Z":
            continue
        if "start_ticks" not in record:
            raise RuntimeError("An unverified recorded process remains live: " + path.name)
        if stat[21] != str(record["start_ticks"]):
            continue
        if path.name != "deadline-process.json":
            raise RuntimeError("A recorded experiment process remains live: " + path.name)
        command = (proc / "cmdline").read_bytes()
        if b"/root/.config/lewm-proposal-smoke/deadline.py" not in command:
            raise RuntimeError("Remote guard process identity differs")
        os.kill(record["pid"], signal.SIGTERM)
        time.sleep(0.2)
        if proc.exists() and (proc / "stat").read_text().split()[2] != "Z":
            raise RuntimeError("Remote process guard did not retire")
    account = Path("/root/.config/lewm-proposal-smoke")
    if (account / "account.env").exists():
        raise RuntimeError("Unexpected remote provider credential file")
    for name in ("deadline.py", "lifecycle.json"):
        shutil.copy2(account / name, lifecycle / name)
    (lifecycle / "process-closeout.json").write_text(
        json.dumps(
            {
                "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "records": records,
                "experiment_processes_retired": True,
                "remote_guard_retired": True,
                "provider_guard": "Local provider-stop guard remains active until verified deletion",
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
    archive = root.parent / "lewm-screen-20261031-v1.tar.gz"
    manifest = root.parent / "lewm-screen-manifest.json"
    if archive.exists() or manifest.exists():
        raise FileExistsError("Never overwrite a sealed terminal artifact")
    with tarfile.open(archive, "w:gz") as tar:
        for entry in entries:
            tar.add(root / entry["path"], arcname=entry["path"], recursive=False)
    with tarfile.open(archive, "r:gz") as tar:
        for entry in entries:
            with tar.extractfile(entry["path"]) as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            if sha != entry["sha256"]:
                raise RuntimeError("Archive readback failed")
    with archive.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    value = {
        "pod_id": design["execution_pod_id"],
        "archive": archive.name,
        "archive_sha256": sha,
        "archive_bytes": archive.stat().st_size,
        "files": entries,
        "file_count": len(entries),
        "total_file_bytes": sum(e["bytes"] for e in entries),
        "remote_readback_verified": True,
        "audit": audit,
    }
    manifest.write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: value[k]
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
