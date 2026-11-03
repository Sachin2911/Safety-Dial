"""Upload and independently verify the explicitly approved November archive batch.

This tool never removes local evidence or modifies existing research repositories.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "SafeLeWM/archive"
OUTPUT = ROOT / "SafeLeWM/outputs/cloud-migration-20261103"
INDEX = ARCHIVE / "cloud-index-20261103.json"
RECEIPT = ARCHIVE / "cloud-receipt-20261103.json"
REPO = "Sachioster/safety-dial-evidence"


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(receipt):
    temporary = RECEIPT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(RECEIPT)


def check(path, row):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")
    if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
        raise ValueError(f"File differs from frozen index: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["upload", "verify"])
    args = parser.parse_args()
    index = json.loads(INDEX.read_text())
    if index["repo_id"] != REPO:
        raise ValueError("Unexpected repository")
    receipt = json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {
        "repo_id": REPO,
        "repo_type": "dataset",
        "private": True,
        "started_utc": utc(),
        "index_sha256": digest(INDEX),
        "uploads": {},
        "verified_files": {},
        "local_evidence_removed": False,
    }
    if receipt["index_sha256"] != digest(INDEX):
        raise ValueError("Index changed during migration")
    api = HfApi()
    if api.whoami()["name"] != "Sachioster":
        raise ValueError("Unexpected authenticated account")
    if args.action == "upload":
        for row in index["files"]:
            check(ROOT / row["local_path"], row)
        api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
        info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)
        if not info.private:
            raise ValueError("Refusing to upload to a public repository")
        existing = {f.rfilename: f for f in info.siblings}
        parent = info.sha
        rows = index["files"] + [{
            "local_path": str(INDEX.relative_to(ROOT)),
            "remote_path": "SafeLeWM/index-20261103.json",
            "sha256": digest(INDEX),
            "bytes": INDEX.stat().st_size,
        }]
        for row in rows:
            path = ROOT / row["local_path"]
            remote = row["remote_path"]
            check(path, row)
            if remote in existing:
                item = existing[remote]
                lfs = item.lfs
                actual = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
                if actual is None:
                    actual = hashlib.sha1(
                        b"blob " + str(row["bytes"]).encode() + b"\0" + path.read_bytes()
                    ).hexdigest()
                    matches = actual == item.blob_id
                else:
                    matches = actual == row["sha256"]
                if not matches or item.size != row["bytes"]:
                    raise ValueError(f"Refusing to overwrite different remote bytes: {remote}")
                print(f"Already uploaded: {remote}", flush=True)
            else:
                print(f"Uploading {remote} ({row['bytes']} bytes)", flush=True)
                commit = api.upload_file(
                    repo_id=REPO,
                    repo_type="dataset",
                    path_or_fileobj=path,
                    path_in_repo=remote,
                    parent_commit=parent,
                    commit_message=f"Preserve {remote}",
                )
                parent = commit.oid
            receipt["uploads"][remote] = {"sha256": row["sha256"], "bytes": row["bytes"]}
            receipt["revision"] = parent
            save(receipt)
        receipt["status"] = "uploaded-awaiting-independent-download"
        receipt["uploaded_utc"] = utc()
        save(receipt)
        print(f"Pinned revision: {parent}", flush=True)
    else:
        if os.environ.get("HF_HUB_DISABLE_XET") == "1":
            method = "Fresh full HTTP download with Xet disabled, then SHA-256 and size check"
        else:
            isolated = OUTPUT / "verification-xet"
            if Path(os.environ.get("HF_XET_CACHE", "")).resolve() != isolated.resolve():
                raise ValueError("Use the dedicated verification-xet directory for Xet state")
            # hf-xet 1.8 has no local download chunk cache. Keep upload state separate too.
            method = (
                "Fresh force_download at pinned revision with isolated Xet state, "
                "then SHA-256 and size check"
            )
        revision = receipt["revision"]
        if not api.repo_info(REPO, repo_type="dataset", revision=revision).private:
            raise ValueError("Expected private repository")
        rows = index["files"] + [{
            "remote_path": "SafeLeWM/index-20261103.json",
            "sha256": receipt["index_sha256"],
            "bytes": INDEX.stat().st_size,
        }]
        pending = []
        for row in rows:
            remote = row["remote_path"]
            prior = receipt["verified_files"].get(remote)
            target = OUTPUT / "verified-downloads" / remote
            if prior and prior["revision"] == revision and target.exists():
                check(target, row)
                continue
            pending.append(row)
        receipt["verification_tool"] = {
            "local_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": digest(Path(__file__)),
            "note": "Four concurrent downloads; the earlier tool snapshot remains in the catalogue",
        }
        save(receipt)

        def verify_one(row):
            remote = row["remote_path"]
            print(f"Fresh download: {remote} ({row['bytes']} bytes)", flush=True)
            path = Path(hf_hub_download(
                repo_id=REPO,
                repo_type="dataset",
                revision=revision,
                filename=remote,
                force_download=True,
                local_dir=OUTPUT / "verified-downloads",
            ))
            check(path, row)
            return remote, {
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "revision": revision,
                "verified_utc": utc(),
                "method": method,
            }
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(verify_one, row)
                for row in sorted(pending, key=lambda r: r["bytes"], reverse=True)
            ]
            for future in as_completed(futures):
                remote, verified = future.result()
                receipt["verified_files"][remote] = verified
                save(receipt)
                print(f"Verified: {remote}", flush=True)
        receipt["status"] = "all-cloud-files-redownload-verified"
        receipt["verified_utc"] = utc()
        save(receipt)
        print(f"Verified {len(rows)} cloud files at {revision}", flush=True)


if __name__ == "__main__":
    main()
