"""Append an explicit artifact allowlist to a private Hub repo and read it back.

    --files JSON contains a mapping of remote relative path to local file path.
Only these files and the generated manifest are uploaded. Credentials, licensed
assets and historical runtime archives must never be supplied in this allowlist.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path, PurePosixPath

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[2]


def validate_source_commit(source_commit):
    """Reject abbreviated, unknown or noncommit source references before upload."""
    if (not isinstance(source_commit, str) or len(source_commit) != 40
            or any(c not in "0123456789abcdef" for c in source_commit)):
        raise ValueError("Preservation requires an existing full Git commit SHA")
    result = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-t", source_commit],
                            capture_output=True, text=True, check=False)
    if result.returncode != 0 or result.stdout.strip() != "commit":
        raise ValueError("Preservation requires an existing full Git commit SHA")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def preserve(repo, kind, prefix, files, source_commit, output):
    validate_source_commit(source_commit)
    api = HfApi()
    info = api.repo_info(repo, repo_type=kind)
    if info.private is not True:
        raise ValueError("Study artifacts require an existing verified private repository")
    prefix_path = PurePosixPath(prefix)
    if prefix_path.is_absolute() or ".." in prefix_path.parts or not prefix_path.parts:
        raise ValueError("Invalid immutable stage prefix")
    prefix = str(prefix_path).rstrip("/") + "/"
    if any(entry.rfilename.startswith(prefix) for entry in info.siblings):
        raise FileExistsError("An existing immutable stage must never be overwritten")
    rows, operations = [], []
    for name, source in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
            raise ValueError("Invalid artifact relative path")
        path = Path(source)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Artifacts must be ordinary explicitly selected local files")
        rows.append({"path": prefix + name, "bytes": path.stat().st_size,
                     "sha256": digest(path)})
        operations.append(CommitOperationAdd(path_in_repo=prefix + name, path_or_fileobj=path))
    manifest = {"schema": "maniguard-latent-preservation-manifest-v1",
                "repo_id": repo, "repo_type": kind, "private": True,
                "prefix": prefix, "source_commit": source_commit, "files": rows}
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
    operations.append(CommitOperationAdd(path_in_repo=prefix + "manifest.json",
                                         path_or_fileobj=manifest_bytes))
    result = api.create_commit(repo, repo_type=kind, operations=operations,
                               commit_message="Preserve latent-evolution stage " + prefix)
    revision = result.oid
    remote = api.repo_info(repo, repo_type=kind, revision=revision, files_metadata=True)
    if remote.private is not True or remote.sha != revision:
        raise ValueError("Remote privacy/revision verification failed")
    entries = {entry.rfilename: entry for entry in remote.siblings}
    for row in rows:
        entry = entries[row["path"]]
        if entry.size != row["bytes"]:
            raise ValueError("Remote artifact size mismatch")
        if row["bytes"] > 8 * 1024 * 1024 and entry.lfs is not None:
            if entry.lfs.sha256 != row["sha256"]:
                raise ValueError("Remote large-file SHA256 mismatch")
            row["verified_by"] = "remote_lfs_sha256"
        else:
            downloaded = hf_hub_download(repo, row["path"], repo_type=kind, revision=revision)
            if digest(downloaded) != row["sha256"]:
                raise ValueError("Redownloaded artifact SHA256 mismatch")
            row["verified_by"] = "redownload_sha256"
    downloaded_manifest = hf_hub_download(repo, prefix + "manifest.json",
                                          repo_type=kind, revision=revision)
    if Path(downloaded_manifest).read_bytes() != manifest_bytes:
        raise ValueError("Remote preservation manifest differs")
    receipt = {**manifest, "revision": revision, "verified_utc_epoch": time.time(),
               "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "remote_files_verified": True, "remote_manifest_redownloaded": True}
    output = Path(output)
    with output.open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--kind", choices=["model", "dataset"], required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--files", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = preserve(args.repo, args.kind, args.prefix, json.loads(args.files.read_text()),
                       args.source_commit, args.output)
    print(json.dumps({"repo_id": receipt["repo_id"], "revision": receipt["revision"],
                      "private": True, "verified_files": len(receipt["files"])}))


if __name__ == "__main__":
    main()
