"""Upload this completed study to the designated private dataset and verify every byte."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import disable_progress_bars


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def token():
    value = os.environ.get("HF_TOKEN")
    if value:
        return value
    for line in Path(".env").read_text().splitlines():
        name, separator, value = line.partition("=")
        if name.strip().removeprefix("export ").strip() == "HF_TOKEN" and separator:
            parsed = shlex.split(value, comments=True)
            if parsed:
                return parsed[0]
    raise RuntimeError("HF credential unavailable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--verification", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = json.loads((root / "analysis/results.json").read_text())
    status = json.loads((root / "run/status.json").read_text())
    if status["status"] == "running":
        raise ValueError("Final preservation requires a terminal study")
    files = {}
    for folder in ("run", "source-v1", "source-v2", "analysis", "source-upstream"):
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            if path.name.endswith((".tmp", ".pyc", ".partial")):
                continue
            if path.name in (".env", "id_rsa", "id_ed25519", "vast_api_key"):
                raise ValueError("Unexpected credential-like file in study")
            files[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": digest(path),
            }
    prefix = "20_grasp_initiation_probe/20261029T175341Z"
    repo = "Sachioster/maniguard-jar-replication-20261022"
    index = {
        "schema": "jar-grasp-probe-index-v1", "study": "20_grasp_initiation_probe",
        "run": "20261029T175341Z", "status": status["status"],
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo": repo, "private": True, "prefix": prefix,
        "files": files, "file_count_excluding_index": len(files),
        "bytes_excluding_index": sum(value["bytes"] for value in files.values()),
        "completed_fixed_comparison": summary["completed_fixed_comparison"],
        "arms": summary["arms"], "accounting": summary["accounting"],
        "source_freeze_sha256": summary["source_freeze_sha256"],
        "scored_freeze_sha256": summary["scored_freeze_sha256"],
    }
    (root / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    credential = token()
    api = HfApi(token=credential)
    if not api.repo_info(repo, repo_type="dataset").private:
        raise ValueError("Evidence repository must be private")
    disable_progress_bars()
    commit = api.upload_folder(
        repo_id=repo, repo_type="dataset", folder_path=str(root), path_in_repo=prefix,
        allow_patterns=[*files, "index.json"],
        commit_message="Preserve completed one-case Jar grasp-initiation probe and all failed work",
    )
    receipt_path = Path("progress/receipts/20261029-grasp-probe-preservation.json")
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    receipt.update({"repo": repo, "private": True, "prefix": prefix,
                    "final_revision": commit.oid, "run_preservation_complete": True,
                    "verification_complete": False,
                    "index_sha256": digest(root / "index.json")})
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"uploaded_revision": commit.oid, "files": len(files) + 1}), flush=True)
    snapshot_download(repo, repo_type="dataset", revision=commit.oid, token=credential,
                      allow_patterns=[prefix + "/*"], local_dir=args.verification, max_workers=4)
    expected = {**files, "index.json": {"bytes": (root / "index.json").stat().st_size,
                                         "sha256": digest(root / "index.json")}}
    for relative, metadata in expected.items():
        path = args.verification / prefix / relative
        if path.stat().st_size != metadata["bytes"] or digest(path) != metadata["sha256"]:
            raise ValueError("Pinned private byte verification failed: " + relative)
    receipt.update({"verification_complete": True, "verified_files": len(expected),
                    "verified_bytes": sum(item["bytes"] for item in expected.values()),
                    "verified_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "verification": "Every source, result, recording and video downloaded at final revision and SHA-256 identical"})
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"verified_revision": commit.oid, "files": len(expected),
                      "bytes": receipt["verified_bytes"]}), flush=True)


if __name__ == "__main__":
    main()
