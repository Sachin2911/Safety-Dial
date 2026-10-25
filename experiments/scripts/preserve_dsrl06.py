"""Freeze, stage and privately verify all task-6 evidence before rental destruction.

Remote preparation follows worker shutdown. SQLite is preserved with its backup API,
never by copying a live database or its WAL. Uploads use fresh task-6 prefixes only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
DATASET = "Sachioster/maniguard-jar-replication-20261022"
MODEL = "Sachioster/safety-dial-dsrl06-20261025"
MUTABLE_NAMES = {
    "network-cost.json",
    "watchdog-status.json",
    "watchdog.log",
    "remote-watchdog.log",
    "STOP",
}
IGNORED_NAMES = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
MODEL_SUFFIXES = {".msgpack", ".safetensors", ".pt", ".pth", ".ckpt"}
PUBLISHED_MANIFEST = "receipts/dsrl06/preservation-manifest.json"


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def files(root, *, ignore_top=(), exclude_subtrees=()):
    root = Path(root)
    if root.is_file():
        return [root]
    output = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] in ignore_top or any(
            Path(folder).absolute() in path.absolute().parents for folder in exclude_subtrees
        ):
            continue
        if any(part in IGNORED_NAMES for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError("Evidence must contain regular files, not symbolic links")
        if path.is_file():
            if path.name == ".env" or path.name.startswith(".env."):
                raise ValueError("Credential environment file found in evidence")
            if path.suffix != ".pyc":
                output.append(path)
    return output


def inventory(root, exclude=()):
    root = Path(root)
    excluded = set(exclude)
    return {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in files(root)
        if str(path.relative_to(root)) not in excluded
    }


def sqlite_backup(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise ValueError("Query ledger is unavailable for a consistent SQLite backup")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("SQLite evidence backup must be a new file")
    source_db = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    target_db = sqlite3.connect(destination)
    try:
        source_db.backup(target_db)
        if target_db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite backup integrity check failed")
        queries = target_db.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        target_db.commit()
    finally:
        target_db.close()
        source_db.close()
    return {"path": str(destination), "sha256": digest(destination), "queries": queries}


def verify_stopped(path, instance_id):
    evidence = json.loads(Path(path).read_text())
    if evidence.get("instance_id") != instance_id or evidence.get("workers_stopped") is not True:
        raise ValueError("A matching worker-shutdown receipt is required")
    if not isinstance(evidence.get("worker_pids"), list):
        raise ValueError("Worker-shutdown receipt must identify all launched worker PIDs")
    for pid in evidence["worker_pids"]:
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("Invalid worker PID in shutdown receipt")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            continue
        if state == "Z":
            continue  # A terminated wrapper awaiting reaping cannot write science data.
        raise ValueError("A worker is still present; final SQLite backup is not yet allowed")
    return evidence


def prove_no_query_service(root, query_ledger):
    """Missing provisioning-time ledgers require explicit absence of query evidence."""
    if Path(str(query_ledger) + "-wal").exists() or Path(str(query_ledger) + "-shm").exists():
        raise ValueError("A missing query database still has SQLite sidecars")
    for path in files(root):
        relative = path.relative_to(root)
        if "preservation" in relative.parts:
            continue
        if path.suffix in {".sqlite", ".sqlite3", ".db"}:
            with path.open("rb") as stream:
                sqlite_header = stream.read(16) == b"SQLite format 3\x00"
            if sqlite_header:
                with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
                    if db.execute("SELECT 1 FROM sqlite_master WHERE name='queries'").fetchone():
                        raise ValueError("Missing query ledger has another query-service database")
        if path.name.startswith("query_") or (
            relative.parts[0] in {"queries", "captures"} and path.suffix in {".npz", ".part"}
        ):
            raise ValueError("Missing query ledger has surviving query artifacts")
        if path.name.endswith(".journal.jsonl"):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get("event") in {"query_attempt", "query_returned", "proposal"}:
                    raise ValueError("Missing query ledger has recorded query attempts")
        if path.name.endswith(".state.json"):
            value = json.loads(path.read_text())
            if value.get("schema") == "dsrl06-probe-state-v1" and value.get("requests"):
                raise ValueError("Missing query ledger has unresolved probe send intentions")
        if path.name == "resources.jsonl":
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get("event") == "start_reserved":
                    raise ValueError("Missing query ledger has a reserved simulator start")
                if row.get("event") == "charge" and row.get("data", {}).get("kind") in {
                    "model_queries",
                    "policy_controls",
                    "warmup_controls",
                }:
                    if row["data"]["count"] > 0:
                        raise ValueError("Missing query ledger has positive interaction charges")
        if path.name == "queue-state.json":
            value = json.loads(path.read_text())
            for source in value.get("source_counters", {}).values():
                if any(
                    source.get("counts", {}).get(name, 0) > 0
                    for name in (
                        "query_attempt",
                        "query_returned",
                        "before_action",
                        "warmup_attempt",
                    )
                ):
                    raise ValueError("Missing query ledger has positive cached interaction counts")
        if path.name.startswith(("startup", "authority-result")) and path.suffix == ".json":
            value = json.loads(path.read_text())
            if value.get("query_counts", {}).get("reserved", 0) > 0:
                raise ValueError("Missing query ledger has positive probe-query counts")
    return {"source_missing": True, "queries": 0, "reason": "no_query_service_started"}


def prepare_remote(root, instance_id, query_ledger, stopped_marker):
    root, query_ledger = Path(root).resolve(), Path(query_ledger).resolve()
    verify_stopped(stopped_marker, instance_id)
    relative_db = query_ledger.relative_to(root)
    destination = root / "preservation"
    destination.mkdir(exist_ok=False)
    excluded = {str(relative_db), str(relative_db) + "-wal", str(relative_db) + "-shm"}
    for path in files(root):
        if path.name in MUTABLE_NAMES and "preservation" not in path.relative_to(root).parts:
            relative = path.relative_to(root)
            target = destination / "mutable-snapshots" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            excluded.add(str(relative))
    backup = (
        sqlite_backup(query_ledger, destination / "query-ledger.sqlite")
        if query_ledger.is_file()
        else prove_no_query_service(root, query_ledger)
    )
    shutil.copy2(stopped_marker, destination / "workers-stopped.json")
    raw = {
        name: metadata
        for name, metadata in inventory(root, excluded).items()
        if not name.startswith("preservation/") and Path(name).name not in MUTABLE_NAMES
    }
    frozen = inventory(destination)
    marker = {
        "schema": "dsrl06-remote-preservation-v1",
        "instance_id": instance_id,
        "prepared_utc": utc(),
        "workers_stopped": True,
        "live_sqlite_excluded": sorted(excluded),
        "query_ledger_relative": str(relative_db),
        "query_backup": backup,
        "raw_inventory": raw,
        "preservation_inventory": frozen,
    }
    atomic_json(destination / "prepared.json", marker)
    return marker


def verify_remote(root):
    root = Path(root)
    prepared = json.loads((root / "preservation/prepared.json").read_text())
    excluded = prepared["live_sqlite_excluded"]
    current = {
        name: metadata
        for name, metadata in inventory(root, excluded).items()
        if not name.startswith("preservation/") and Path(name).name not in MUTABLE_NAMES
    }
    if current != prepared["raw_inventory"]:
        raise ValueError("Remote evidence changed after the final worker-shutdown freeze")
    frozen = inventory(root / "preservation", {"prepared.json"})
    if frozen != prepared["preservation_inventory"]:
        raise ValueError("Remote SQLite backup or shutdown evidence changed")
    if prepared["query_backup"].get("source_missing"):
        source = root / prepared["query_ledger_relative"]
        if source.exists():
            raise ValueError("A query service appeared after the provisioning-failure freeze")
        prove_no_query_service(root, source)
    return prepared


def is_model(path):
    path = Path(path)
    return bool({"checkpoints", "actor_exports"} & set(path.parts)) or path.suffix in MODEL_SUFFIXES


def local_sources(local_root, stage, repository):
    local_root, stage, repository = Path(local_root), Path(stage), Path(repository)
    selected = []
    excluded_top = {"remote", "hf-stage", "hf-verify-cache", "preservation-verification.json"}
    excluded_names = {
        "preservation-staging.json",
        "preservation-upload-progress.json",
        "preservation.lock",
    }
    if local_root.exists():
        ignored = excluded_top | {
            path.name
            for path in local_root.iterdir()
            if path.name.startswith(("remote", "hf-stage", "hf-verify-cache"))
        }
        for path in files(local_root, ignore_top=ignored, exclude_subtrees=(stage,)):
            relative = path.relative_to(local_root)
            if stage.resolve() in path.resolve().parents:
                continue
            if (
                relative.parts[0] in excluded_top
                or relative.parts[0].startswith("remote")
                or path.name in excluded_names
            ):
                continue
            if path.name in MUTABLE_NAMES:
                continue  # A point-in-time copy is made separately below.
            selected.append((path, Path("receipts/dsrl06/local") / relative))
    for name in ("AGENTS.md", "firstPrinciples.md", "progress/06-dsrl-plan.md"):
        path = repository / name
        if path.is_file():
            selected.append((path, Path("receipts/dsrl06/source") / name))
    patterns = (
        "experiments/helpers/dsrl06*.py",
        "experiments/scripts/*dsrl06*",
        "experiments/tests/test_dsrl06*.py",
        "configs/dsrl06/**/*",
        "progress/receipts/06-*",
        "progress/rental-dsrl06*",
    )
    for pattern in patterns:
        for path in sorted(repository.glob(pattern)):
            if path.is_file():
                selected.append(
                    (path, Path("receipts/dsrl06/source") / path.relative_to(repository))
                )
    # Stage freezes can reference ignored candidate arrays or exact upstream metadata.
    # Preserve every declared source, including files outside the standard source globs.
    for receipt in sorted((repository / "progress/receipts").glob("06-*.json")):
        value = json.loads(receipt.read_text())
        declared = value.get("source_sha256", {}) if isinstance(value, dict) else {}
        if not isinstance(declared, dict):
            continue
        for name in declared:
            path = repository / name
            if not path.resolve().is_relative_to(repository.resolve()):
                raise ValueError("Frozen source reference escapes the repository")
            if not path.is_file():
                raise ValueError("A declared frozen source is absent from preservation")
            selected.append((path, Path("receipts/dsrl06/source") / Path(name)))
    return list(dict.fromkeys(selected))


def stage_evidence(remotes, local_root, stage, instance_id, repository=ROOT):
    stage, local_root = Path(stage), Path(local_root)
    stage.mkdir(parents=True, exist_ok=False)
    snapshots, entries = [], []
    dataset, model = stage / "dataset", stage / "model"
    dataset.mkdir()

    def copy(source, destination, kind="dataset"):
        target = stage / kind / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError("Two evidence sources have the same preserved path")
        shutil.copy2(source, target)
        entries.append(
            {
                "source": str(Path(source).resolve()),
                "kind": kind,
                "path": str(destination),
                "bytes": target.stat().st_size,
                "sha256": digest(target),
            }
        )

    for index, remote in enumerate(remotes):
        remote = Path(remote)
        prepared = verify_remote(remote)
        snapshots.append(
            {
                "source": str(remote.resolve()),
                "instance_id": prepared["instance_id"],
                "prepared_sha256": digest(remote / "preservation/prepared.json"),
                "query_backup": prepared["query_backup"],
            }
        )
        allowed = (
            set(prepared["raw_inventory"])
            | {"preservation/" + name for name in prepared["preservation_inventory"]}
            | {"preservation/prepared.json"}
        )
        for name in sorted(allowed):
            kind = "model" if is_model(name) else "dataset"
            prefix = Path(f"remote{index}")
            destination = (
                Path("checkpoints") / prefix / name
                if kind == "model"
                else Path("recordings/dsrl06/raw") / prefix / name
            )
            copy(remote / name, destination, kind)
    if instance_id not in {item["instance_id"] for item in snapshots}:
        raise ValueError("Final rental is absent from the preserved remote snapshots")
    for source, destination in local_sources(local_root, stage, repository):
        if (
            is_model(source.relative_to(local_root))
            if source.is_relative_to(local_root)
            else is_model(source)
        ):
            copy(source, Path("checkpoints/local") / destination, "model")
        else:
            copy(source, destination)
    ignored = {
        path.name
        for path in local_root.iterdir()
        if path.name.startswith(("remote", "hf-stage", "hf-verify-cache"))
    }
    for path in files(local_root, ignore_top=ignored, exclude_subtrees=(stage,)):
        relative = path.relative_to(local_root)
        if (
            path.name in MUTABLE_NAMES
            and not relative.parts[0].startswith("remote")
            and relative.parts[0] != "hf-verify-cache"
            and stage.resolve() not in path.resolve().parents
        ):
            copy(path, Path("receipts/dsrl06/financial-snapshots") / path.relative_to(local_root))
            # Financial guard continues to run; this is explicitly a point-in-time snapshot.
            entries[-1]["mutable_snapshot"] = True
    # Canonical task entries point to raw attempts and retain failures as distinct jobs.
    task_entries = []
    for item in list(entries):
        if item["kind"] != "dataset" or Path(item["source"]).name != "job.json":
            continue
        job = json.loads(Path(item["source"]).read_text())
        if "job_id" not in job or "task" not in job:
            continue
        label = re.sub(r"[^A-Za-z0-9_.-]", "_", str(job["job_id"]))
        prefix = f"dsrl06_{job.get('stage', 'attempts')}"
        path = dataset / "jar" / prefix / label / "index.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {"job": job, "raw_job_path": item["path"], "all_attempts_preserved": True}
        atomic_json(path, value)
        task_entries.append(str(path.relative_to(dataset)))
    atomic_json(
        dataset / "jar/dsrl06_attempts/index.json",
        {
            "schema": "dsrl06-attempt-index-v1",
            "entries": task_entries,
            "remote_snapshots": snapshots,
            "all_failed_and_partial_attempts_retained": True,
        },
    )
    atomic_json(
        dataset / "receipts/dsrl06/preservation-layout.json",
        {
            "dataset": DATASET,
            "model_repo_if_present": MODEL,
            "raw": "recordings/dsrl06/raw",
            "tasks": "jar/dsrl06_*",
            "receipts": "receipts/dsrl06",
            "engineering_only_supported": True,
            "final_sqlite_backup_after_worker_shutdown": True,
        },
    )
    manifest = {
        "schema": "dsrl06-preservation-manifest-v1",
        "instance_id": instance_id,
        "created_utc": utc(),
        "remote_snapshots": snapshots,
        "local_root": str(local_root.resolve()),
        "repository": str(Path(repository).resolve()),
        "sources": entries,
        "dataset": inventory(dataset),
        "model": inventory(model) if model.exists() else {},
    }
    atomic_json(stage / "manifest.json", manifest)
    published = dataset / PUBLISHED_MANIFEST
    published.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stage / "manifest.json", published)
    return manifest


def verify_local(stage):
    stage = Path(stage)
    manifest = json.loads((stage / "manifest.json").read_text())
    if not manifest["dataset"]:
        raise ValueError("An empty dataset cannot authorize destruction")
    for kind in ("dataset", "model"):
        expected = manifest[kind]
        actual = inventory(stage / kind) if (stage / kind).exists() else {}
        if kind == "dataset":
            published = actual.pop(PUBLISHED_MANIFEST, None)
            if published is None or published["sha256"] != digest(stage / "manifest.json"):
                raise ValueError("Published manifest differs from the local frozen manifest")
        if actual != expected:
            raise ValueError("Staged evidence no longer matches the complete manifest")
    for source in manifest["sources"]:
        if not source.get("mutable_snapshot"):
            path = Path(source["source"])
            if not path.is_file() or digest(path) != source["sha256"]:
                raise ValueError("Source evidence changed or disappeared after staging")
    for remote in manifest["remote_snapshots"]:
        verify_remote(remote["source"])
    expected_local = {
        (str(source.resolve()), str(destination))
        for source, destination in local_sources(
            manifest["local_root"], stage, manifest["repository"]
        )
    }
    actual_local = set()
    for source in manifest["sources"]:
        if source.get("mutable_snapshot"):
            continue
        if source["path"].startswith("receipts/dsrl06/"):
            actual_local.add((source["source"], source["path"]))
        elif source["kind"] == "model" and source["path"].startswith("checkpoints/local/"):
            actual_local.add((source["source"], source["path"][len("checkpoints/local/") :]))
    if expected_local != actual_local:
        raise ValueError("Late source or receipt files require a new complete preservation stage")
    return manifest


def read_secrets(repository=ROOT):
    from dotenv import dotenv_values

    values = {**dotenv_values(Path(repository) / ".env"), **os.environ}
    token = values.get("HF_TOKEN")
    if not token:
        raise ValueError("Hugging Face token is unavailable in the process environment")
    secrets = [
        str(value).encode()
        for key, value in values.items()
        if value
        and ("TOKEN" in key or "API_KEY" in key or key in {"GITHUB_TOKEN", "WANDB_API_KEY"})
    ]
    return token, secrets


def scan_secrets(stage, secrets):
    for path in files(stage):
        # Check all file types in blocks, retaining overlap for cross-block matches.
        overlap = max((len(value) for value in secrets), default=1) - 1
        tail = b""
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                data = tail + block
                if any(value in data for value in secrets):
                    raise ValueError("A known secret was found in staged evidence; upload refused")
                tail = data[-overlap:] if overlap else b""


def upload_verified(stage, marker, instance_id, *, api, download, token, secrets=(), workers=6):
    stage, marker = Path(stage), Path(marker)
    # A stale marker must not survive a new failed verification attempt.
    if marker.exists():
        marker.unlink()
    manifest = verify_local(stage)
    if manifest["instance_id"] != instance_id:
        raise ValueError("Preservation manifest belongs to another rental")
    scan_secrets(stage, secrets)
    pinned, verified, verified_bytes = {}, {}, {}
    for kind, repo, repo_type in (("dataset", DATASET, "dataset"), ("model", MODEL, "model")):
        if not manifest[kind]:
            continue
        if kind == "model":
            api.create_repo(repo_id=repo, repo_type=repo_type, private=True, exist_ok=True)
        info = api.repo_info(repo, repo_type=repo_type)
        if info.private is not True:
            raise ValueError("Preservation destination must already be private")
        commit = api.upload_folder(
            folder_path=stage / kind,
            repo_id=repo,
            repo_type=repo_type,
            commit_message="task6: preserve frozen DSRL experiment evidence",
            parent_commit=info.sha,
        )
        revision = commit.oid
        if not revision:
            raise ValueError("Upload returned no pinned commit revision")
        pinned[kind] = {"repo": repo, "repo_type": repo_type, "revision": revision}

        def check(item):
            name, expected = item
            downloaded = Path(
                download(
                    repo_id=repo,
                    filename=name,
                    repo_type=repo_type,
                    revision=revision,
                    token=token,
                    cache_dir=stage.parent / "hf-verify-cache",
                )
            )
            if (
                downloaded.stat().st_size != expected["bytes"]
                or digest(downloaded) != expected["sha256"]
            ):
                raise ValueError("A pinned uploaded file differs from the local frozen evidence")
            return name

        uploaded_files = dict(manifest[kind])
        if kind == "dataset":
            manifest_file = stage / "dataset" / PUBLISHED_MANIFEST
            uploaded_files[PUBLISHED_MANIFEST] = {
                "bytes": manifest_file.stat().st_size,
                "sha256": digest(manifest_file),
            }
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            verified[kind] = list(executor.map(check, uploaded_files.items()))
        verified_bytes[kind] = sum(item["bytes"] for item in uploaded_files.values())
        if api.repo_info(repo, repo_type=repo_type).private is not True:
            raise ValueError("Preservation destination privacy changed during verification")
    # Catch late receipts, source edits or inventory growth during the network work.
    verify_local(stage)
    receipt = {
        "schema": "dsrl06-preservation-verification-v1",
        "instance_id": instance_id,
        "verified_utc": utc(),
        "local_verified": True,
        "private_hf_verified": True,
        "pinned_revisions": pinned,
        "manifest_sha256": digest(stage / "manifest.json"),
        "manifest_path": str((stage / "manifest.json").resolve()),
        "verified_files": {kind: len(names) for kind, names in verified.items()},
        "verified_bytes": verified_bytes,
        "verification": "Every staged file downloaded at pinned revisions and SHA256 matched",
        "complete_inventory_frozen": True,
        "query_ledger_backup_verified": all(
            not snapshot["query_backup"].get("source_missing")
            for snapshot in manifest["remote_snapshots"]
        ),
        "query_accounting_verified": True,
        "missing_query_ledgers_no_service": [
            snapshot["instance_id"]
            for snapshot in manifest["remote_snapshots"]
            if snapshot["query_backup"].get("source_missing")
        ],
        "model_checkpoint_present": bool(manifest["model"]),
    }
    atomic_json(marker, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    remote = commands.add_parser("prepare-remote")
    remote.add_argument("--root", type=Path, required=True)
    remote.add_argument("--instance-id", type=int, required=True)
    remote.add_argument("--query-ledger", type=Path, required=True)
    remote.add_argument("--workers-stopped-marker", type=Path, required=True)
    stage = commands.add_parser("stage")
    stage.add_argument("--remote", type=Path, action="append", required=True)
    stage.add_argument("--local-root", type=Path, default=ROOT / "outputs/dsrl06")
    stage.add_argument("--stage", type=Path, required=True)
    stage.add_argument("--instance-id", type=int, required=True)
    stage.add_argument("--repository", type=Path, default=ROOT)
    upload = commands.add_parser("upload")
    upload.add_argument("--stage", type=Path, required=True)
    upload.add_argument("--marker", type=Path, required=True)
    upload.add_argument("--instance-id", type=int, required=True)
    verify = commands.add_parser("verify-local")
    verify.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-remote":
        result = prepare_remote(
            args.root, args.instance_id, args.query_ledger, args.workers_stopped_marker
        )
        print(
            json.dumps(
                {
                    "prepared": True,
                    "instance_id": args.instance_id,
                    "queries": result["query_backup"]["queries"],
                }
            )
        )
    elif args.command == "stage":
        result = stage_evidence(
            args.remote, args.local_root, args.stage, args.instance_id, args.repository
        )
        print(
            json.dumps(
                {
                    "staged": True,
                    "files": {kind: len(result[kind]) for kind in ("dataset", "model")},
                }
            )
        )
    elif args.command == "verify-local":
        result = verify_local(args.stage)
        print(json.dumps({"local_verified": True, "instance_id": result["instance_id"]}))
    else:
        from huggingface_hub import HfApi, hf_hub_download

        token, secrets = read_secrets()
        result = upload_verified(
            args.stage,
            args.marker,
            args.instance_id,
            api=HfApi(token=token),
            download=hf_hub_download,
            token=token,
            secrets=secrets,
        )
        print(json.dumps(result))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Third-party exceptions can contain credential-bearing URLs.
        print("Task 6 preservation failed:", type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
