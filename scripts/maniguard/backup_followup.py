"""Credential-scanned private followup evidence, excluding assets and checkpoints."""

import argparse
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/maniguard/phase3-parts2-3/20261018-56582094"
REPO = "Sachioster/safety-dial-phase3-part1-20261018"
SECRETS = ("HF_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY", "CONTAINER_API_KEY",
           "OPEN_BUTTON_TOKEN", "WEB_PASSWORD")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def backup(label, complete=False, upload=False):
    if not label.replace("-", "").isalnum():
        raise ValueError("Path-free label required")
    destination = ROOT / "data/maniguard/evidence-backups"
    destination.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = destination / f"followup-{stamp}-{label}.tar.gz"
    secrets = [os.environ[key].encode() for key in SECRETS if len(os.environ.get(key, "")) >= 8]
    inventory = []
    with tarfile.open(path, "x:gz", compresslevel=1) as archive:
        for source in sorted(RUN.rglob("*")):
            if not source.is_file() or source.is_symlink():
                continue
            relative = source.relative_to(RUN)
            if any(p in ("appdata", ".cache", "__pycache__") for p in relative.parts):
                continue
            if not complete and any(p in ("observations", "features") for p in relative.parts):
                continue
            if source.name.startswith(".env") or source.suffix in (".key", ".pem", ".pt", ".safetensors"):
                raise ValueError("Unexpected credential/checkpoint file in evidence scope")
            raw = source.read_bytes()
            if any(value in raw for value in secrets):
                raise ValueError("Credential value detected in " + str(relative))
            entry = tarfile.TarInfo(str(relative))
            entry.size, entry.mtime = len(raw), int(source.stat().st_mtime)
            archive.addfile(entry, io.BytesIO(raw))
            inventory.append({"path": str(relative), "bytes": len(raw),
                              "sha256": hashlib.sha256(raw).hexdigest()})
    receipt = {"schema": "maniguard-followup-backup-v1", "at": time.time(), "label": label,
               "run": str(RUN.relative_to(ROOT)), "complete_raw_capture_included": complete,
               "archive": path.name, "bytes": path.stat().st_size, "sha256": digest(path),
               "scope": "Pilot logs, traces, recorded images/videos, derived features and source/protocol/report snapshots. No asset bundles, credentials or checkpoints.",
               "active_snapshot": not complete, "files": inventory, "credential_scan": "passed"}
    receipt_path = path.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if upload:
        authorization = json.loads((RUN / "backup-authorization.json").read_text())
        if authorization.get("repo_id") != REPO or authorization.get("parts") != [2, 3]:
            raise ValueError("Explicit followup backup authorization is required")
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        if api.dataset_info(REPO).private is not True:
            raise ValueError("Evidence destination is not private")
        for artifact in (path, receipt_path):
            api.upload_file(path_or_fileobj=str(artifact), path_in_repo="followup/" + artifact.name,
                            repo_id=REPO, repo_type="dataset",
                            commit_message=f"Preserve followup {label} evidence {stamp}")
        revision = api.dataset_info(REPO).sha
        downloaded = hf_hub_download(REPO, "followup/" + path.name, repo_type="dataset",
                                     revision=revision)
        if digest(downloaded) != receipt["sha256"]:
            raise ValueError("Off-instance backup download failed SHA256 verification")
        receipt.update(repo_id=REPO, revision=revision, private=True,
                       download_verified=True, off_instance=True)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    (RUN / (label + "-backup-receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    backup(args.label, args.complete, args.upload)
