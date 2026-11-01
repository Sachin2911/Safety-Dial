"""Privately mirror completed v3 artifacts with pinned, downloaded byte verification."""

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import time
import urllib.request

from preserve_maniguard_evolution import atomic, digest, eligible, upload_and_verify


INSTANCE = 58363639
MACHINE = 141415
LABEL = "maniguard-v3-prepared-20261030T064632Z"
DEADLINE = dt.datetime.fromisoformat("2026-10-30T16:46:32.937332+00:00")
PREFIX = "22_conservative_adapter_development/20261030T075407Z"
REPO = "Sachioster/maniguard-jar-replication-20261022"


def verify_provider():
    request = urllib.request.Request(
        "https://console.vast.ai/api/v0/instances/?owner=me",
        headers={"Authorization": "Bearer " + os.environ["VAST_API_KEY"]},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        rows = json.load(response)["instances"]
    row = next((row for row in rows if row["id"] == INSTANCE), None)
    if row is None:
        return False
    if row["machine_id"] != MACHINE or row["label"] != LABEL:
        raise RuntimeError("Provider identity mismatch")
    return row["actual_status"] == "running"


def included(project, path):
    relative = path.relative_to(project)
    if not eligible(relative):
        return False
    if relative.name in {"auth.json", "credentials.json", "omnigibson.key"}:
        raise ValueError("Credential-like file inside evidence tree")
    if relative.parts[0].startswith("run-"):
        if len(relative.parts) > 2 and relative.parts[1] == "episodes":
            return (project.joinpath(*relative.parts[:3]) / "outcome.json").is_file()
        return relative.name != "run.lock"
    # Policy/trainer logs are live. Final terminal preservation includes their closed bytes.
    if relative.name == "policy-server.log" or relative.name.startswith("trainer-"):
        return False
    # This watcher must not recursively snapshot its own changing progress.
    return relative.name != "hf-mirror-latest.json"


def snapshot(project, base, state):
    files = []
    for path in sorted(project.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink inside evidence project")
        if not path.is_file() or not included(project, path):
            continue
        try:
            stat = path.stat()
            signature = [stat.st_size, stat.st_mtime_ns]
            relative = path.relative_to(project).as_posix()
            if state["inventory"].get(relative, {}).get("signature") != signature:
                files.append((path, relative, signature))
        except FileNotFoundError:
            continue
    if not files:
        return None
    sequence = state.get("sequence", 0) + 1
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = f"live-{sequence:04d}-{stamp}"
    release = base / "snapshots" / name
    release.mkdir(parents=True)
    entries, inventory = {}, {}
    known_secrets = [os.environ[key].encode() for key in
                     ("HF_TOKEN", "GITHUB_TOKEN", "VAST_API_KEY") if os.environ.get(key)]
    text_suffixes = {".json", ".jsonl", ".py", ".yaml", ".yml", ".sh", ".md", ".txt", ".log"}
    for source, relative, signature in files:
        destination = release / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            before = digest(source)
            shutil.copy2(source, destination)
            stat = source.stat()
            if ([stat.st_size, stat.st_mtime_ns] != signature
                    or digest(destination) != before or digest(source) != before):
                destination.unlink()
                continue
        except FileNotFoundError:
            destination.unlink(missing_ok=True)
            continue
        if source.suffix in text_suffixes:
            content = destination.read_bytes()
            if any(secret in content for secret in known_secrets):
                raise ValueError("Known credential found in candidate evidence")
        entries[relative] = {"bytes": signature[0], "sha256": before}
        inventory[relative] = {**entries[relative], "signature": signature}
    index = {
        "schema": "maniguard-v3-incremental-private-mirror-v1", "repo": REPO,
        "private": True, "base_prefix": PREFIX, "final": False,
        "complete_schedule": False, "point_in_time_observation": True,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "previous_verified_revision": state.get("last_verified_revision"),
        "files": entries, "file_count_excluding_index": len(entries),
        "bytes_excluding_index": sum(row["bytes"] for row in entries.values()),
    }
    atomic(release / "index.json", index)
    return {"name": name, "sequence": sequence, "release": str(release),
            "receipt": str(base / "receipts" / (name + ".json")), "inventory": inventory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    project, base = args.project.resolve(), args.out.resolve()
    if project == base or project in base.parents:
        raise ValueError("Preservation directories must be outside the live project")
    base.mkdir(parents=True, exist_ok=True)
    (base / "receipts").mkdir(exist_ok=True)
    state_path = base / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"inventory": {}}
    while dt.datetime.now(dt.timezone.utc) < DEADLINE - dt.timedelta(seconds=60):
        try:
            if not verify_provider():
                break
            pending = state.get("pending")
            if pending is None:
                pending = snapshot(project, base, state)
                state["pending"] = pending
                atomic(state_path, state)
            if pending:
                upload_and_verify(Path(pending["release"]), base / "verification",
                                  pending["name"], Path(pending["receipt"]))
                receipt = json.loads(Path(pending["receipt"]).read_text())
                assert receipt["private"] and receipt["verification_complete"]
                state["inventory"].update(pending["inventory"])
                state.update(sequence=pending["sequence"], pending=None,
                             last_verified_revision=receipt["revision"],
                             last_verified_prefix=receipt["prefix"],
                             last_verified_utc=receipt["verified_utc"])
                atomic(state_path, state)
                atomic(project / "supervision/hf-mirror-latest.json", {
                    "instance_id": INSTANCE, "status": "verified", **receipt,
                    "mirror_state": str(state_path),
                    "retire_exact_mirror_process_before_final_seal": True,
                })
            if args.once:
                return
        except Exception as error:
            # Do not dump request objects, account environments or credential values.
            print(json.dumps({"mirror_error_type": type(error).__name__,
                              "observed_utc": dt.datetime.now(dt.timezone.utc).isoformat()}),
                  flush=True)
            if args.once:
                raise SystemExit(1) from None
        time.sleep(60)


if __name__ == "__main__":
    main()
