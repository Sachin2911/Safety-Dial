"""Preserve Part 1 run evidence privately, excluding assets, keys and environments.

Each file is snapshotted once and hashed from the bytes actually archived. Active
attempt files can represent different instants; final backups follow GPU shutdown.
Credentials are read from the account environment and never copied into receipts.
"""

import argparse
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
SECRETS = ("HF_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY", "CONTAINER_API_KEY",
           "OPEN_BUTTON_TOKEN", "WEB_PASSWORD")


def backup(run, label, *, upload=True, repo_id=None):
    run = Path(run).resolve()
    run.relative_to(ROOT / "runs/maniguard/phase3-part1")
    destination = ROOT / "data/maniguard/evidence-backups"
    destination.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive = destination / f"part1-{stamp}-{label}.tar.gz"
    secrets = [os.environ[k].encode() for k in SECRETS if len(os.environ.get(k, "")) >= 8]
    inventory = []
    with tarfile.open(archive, "x:gz") as bundle:
        for path in sorted(run.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(run)
            if any(p in ("appdata", ".cache", "__pycache__") for p in relative.parts):
                continue
            if path.name.startswith(".env") or path.suffix in (".key", ".pem"):
                raise ValueError("Unexpected secret-bearing file in evidence tree")
            raw = path.read_bytes()
            if any(value in raw for value in secrets):
                raise ValueError("Credential value detected; refusing evidence backup")
            item = tarfile.TarInfo(str(relative))
            item.size = len(raw)
            item.mtime = int(path.stat().st_mtime)
            bundle.addfile(item, io.BytesIO(raw))
            inventory.append({"path": str(relative), "bytes": len(raw),
                              "sha256": hashlib.sha256(raw).hexdigest()})
    receipt = {"schema": "maniguard-part1-private-evidence-v1", "label": label,
               "created_at": time.time(), "run": str(run.relative_to(ROOT)),
               "archive": archive.name, "bytes": archive.stat().st_size,
               "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "files": inventory,
               "scope": "Evidence and source snapshots only; no datasets, model weights or keys"}
    receipt_path = archive.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if upload:
        if repo_id is None:
            raise ValueError("Upload requires an explicitly approved --repo-id")
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        if api.dataset_info(repo_id).private is not True:
            raise ValueError("Evidence repository must be private")
        for path in (archive, receipt_path):
            api.upload_file(path_or_fileobj=str(path), path_in_repo=path.name,
                            repo_id=repo_id, repo_type="dataset",
                            commit_message=f"Preserve Part 1 {label} evidence {stamp}")
        receipt.update(repo_id=repo_id, revision=api.dataset_info(repo_id).sha, private=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--repo-id")
    args = parser.parse_args()
    backup(args.run, args.label, upload=not args.no_upload, repo_id=args.repo_id)
