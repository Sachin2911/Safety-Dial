"""Private, revision-pinned preservation with downloaded-byte hash verification."""

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
from helpers.maniguard_part4_protocol import CONFIG, ROOT, RUN, atomic, digest, read

DATASET = "Sachioster/safety-dial-phase3-part4-20261019"
MODEL = "Sachioster/maniguard-phase3-part4-20261019"


def scan(raw):
    values = [v.encode() for k, v in os.environ.items() if len(v) >= 8 and any(
        term in k.upper() for term in ("TOKEN", "API_KEY", "PASSWORD", "SECRET"))]
    if any(value in raw for value in values):
        raise ValueError("Credential detected in preservation payload")


def upload_paths(repo, kind, paths, label):
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    disable_progress_bars()
    api = HfApi()
    api.create_repo(repo, repo_type=kind, private=True, exist_ok=True)
    info = api.repo_info(repo, repo_type=kind)
    if info.private is not True:
        raise ValueError("Preservation destination must be private")
    inventory = []
    for name, path in paths:
        scan(path.read_bytes())
        inventory.append({"path": name, "sha256": digest(path), "bytes": path.stat().st_size})
    commit = api.create_commit(
        repo_id=repo, repo_type=kind,
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
                    for name, path in paths], commit_message="Preserve Part4 " + label)
    for item in inventory:
        downloaded = hf_hub_download(repo, item["path"], repo_type=kind, revision=commit.oid)
        if digest(downloaded) != item["sha256"]:
            raise ValueError("Downloaded preservation file hash mismatch")
    return {"repo_id": repo, "repo_type": kind, "revision": commit.oid,
            "private": True, "download_verified": True, "files": inventory,
            "at": time.time(), "credential_scan": "passed"}


def preserve_model():
    directory = RUN / "models/baseline"
    frozen = read(directory / "baseline-freeze.json")
    for name, expected in frozen["files"].items():
        if digest(directory / name) != expected:
            raise ValueError("Frozen baseline changed before preservation")
    card = directory / "README.md"
    card.write_text(
        "---\ntags:\n- robotics\n- research\n---\n"
        "# ManiGuard Part 4 bounded direct-event predictor\n\n"
        "Private preservation of five bootstrap MLPs, logistic references, fit-only "
        "PCA/scalers, initial weights and calibration. The frozen development gate "
        "failed. These weights are not a validated warning system or safety filter.\n\n"
        "No comparative acquisition, repair or fresh final audit was launched. "
        "See the complete offline audit and the frozen protocol. External DINOv2 "
        "weights, VLA checkpoints, assets and credentials are excluded. NPZ files "
        "contain arrays only and load with allow_pickle=False.\n")
    paths = [(p.name, p) for p in sorted(directory.iterdir()) if p.is_file()]
    paths += [("offline-audit.json", RUN / "offline-audit.json"),
              ("offline-protocol.json", CONFIG / "offline-protocol.json")]
    receipt = upload_paths(MODEL, "model", paths, "frozen baseline and failed premise")
    atomic(RUN / "model-backup.json", receipt)
    return receipt


def preserve_evidence(label):
    if not label.replace("-", "").isalnum():
        raise ValueError("Path-free preservation label required")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    export = ROOT / "data/maniguard/part4-backups"
    export.mkdir(parents=True, exist_ok=True)
    path = export / (stamp + "-" + label + ".tar.gz")
    inventory = []
    with tarfile.open(path, "x:gz", compresslevel=1) as tf:
        for source in sorted(RUN.rglob("*")):
            if not source.is_file() or source.is_symlink():
                continue
            relative = source.relative_to(RUN)
            if relative.parts[0] in ("donors", "models"):
                continue
            if source.name.startswith(".env") or source.suffix in (".key", ".pem"):
                raise ValueError("Unexpected credential file in evidence")
            raw = source.read_bytes()
            scan(raw)
            item = tarfile.TarInfo(str(relative))
            item.size, item.mtime = len(raw), int(source.stat().st_mtime)
            tf.addfile(item, io.BytesIO(raw))
            inventory.append({"path": str(relative), "bytes": len(raw),
                              "sha256": hashlib.sha256(raw).hexdigest()})
    manifest = path.with_suffix(".json")
    atomic(manifest, {"files": inventory, "sha256": digest(path),
                      "donor_policy": "Original exact inputs remain private and revision-pinned in donor-restore.json; no duplicate acquisition charge.",
                      "model_policy": "Own trained baseline separately preserved in private model repository.",
                      "excluded": ["external checkpoints", "asset bundles", "credentials", "duplicate donors"]})
    paths = [(path.name, path), (manifest.name, manifest)]
    paths += [("videos/" + p.name, p) for p in sorted((RUN / "evidence/videos").glob("*.mp4"))]
    if (RUN / "evidence/gallery.json").exists():
        paths.append(("videos/gallery.json", RUN / "evidence/gallery.json"))
    receipt = upload_paths(DATASET, "dataset", paths, label)
    atomic(RUN / (label + "-evidence-backup.json"), receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("model", "evidence"))
    parser.add_argument("--label", default="progress")
    args = parser.parse_args()
    receipt = preserve_model() if args.kind == "model" else preserve_evidence(args.label)
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))
