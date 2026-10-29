"""Preserve compressed raw attempts and verify every private object by pinned revision."""

import argparse
import concurrent.futures
import gzip
import json
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.unisafe09_protocol import digest


def archive_entries(destination, entries):
    """Keep unchanged raw attempts byte-identical across preservation milestones."""
    with destination.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path, name in entries:
                    archive.add(path, arcname=name)


def verify_local(stage):
    stage = Path(stage)
    if stage.name == "unisafe09" and stage.parent.name == "receipts":
        stage = stage.parent.parent
    manifest_path = stage / "receipts/unisafe09/sha256-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    actual = {str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file()}
    if actual != set(manifest) | {str(manifest_path.relative_to(stage))}:
        raise ValueError("Preservation inventory differs")
    for name, item in manifest.items():
        path = stage / name
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Preservation hash differs: {name}")
    return manifest


def stage_evidence(remote, stage):
    stage.mkdir(parents=True, exist_ok=False)
    recordings = stage / "recordings/unisafe09"
    receipts = stage / "receipts/unisafe09"
    recordings.mkdir(parents=True)
    receipts.mkdir(parents=True)
    # Each attempt retains all frames, state arrays, source copies and partial logs.
    raw_manifest = {}
    for path in sorted(remote.rglob("*")):
        if path.is_file():
            raw_manifest[str(path.relative_to(remote))] = {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
    atomic_json(receipts / "raw-files.json", raw_manifest)
    for folder in sorted((remote / "attempts").glob("*")):
        if not folder.is_dir():
            continue
        archive_entries(recordings / f"{folder.name}.tar.gz", [(folder, folder.name)])
        summary = folder / "summary.json"
        job_path = folder / "job.json"
        if not job_path.exists():
            job_path = folder / "intent.json"
            job = json.loads(job_path.read_text())["job"]
        else:
            job = json.loads(job_path.read_text())
        row = json.loads(summary.read_text()) if summary.exists() else {"outcome": "invalid"}
        invalidation = folder / "invalidation.json"
        if invalidation.exists():
            row = {
                "outcome": "invalid",
                "invalidation": json.loads(invalidation.read_text()),
                "original_summary": row,
            }
        outcome = row["outcome"]
        leaf = (
            stage
            / "unisafe"
            / f"unisafe09_{job['stage']}_{job['condition']}"
            / f"seed{job['seed']}"
            / f"block_plucking_{outcome}"
            / folder.name
        )
        leaf.mkdir(parents=True)
        atomic_json(
            leaf / "index.json",
            {
                "attempt": folder.name,
                "job": job,
                "outcome": outcome,
                "recording": f"recordings/unisafe09/{folder.name}.tar.gz",
                "summary": row,
            },
        )
    archive_entries(
        recordings / "run-metadata.tar.gz",
        [(child, child.name) for child in sorted(remote.iterdir()) if child.name != "attempts"],
    )
    for path in (ROOT / "progress/receipts").glob("09-unisafe-*"):
        if path.is_file():
            shutil.copy2(path, receipts / path.name)
    for name in ("progress/09-unisafe-dial-plan.md", "progress/rental-unisafe09-20261025.json"):
        shutil.copy2(ROOT / name, receipts / Path(name).name)
    manifest = {
        str(p.relative_to(stage)): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(stage.rglob("*"))
        if p.is_file()
    }
    atomic_json(receipts / "sha256-manifest.json", manifest)
    verify_local(stage)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", type=Path, default=ROOT / "outputs/unisafe09/remote")
    parser.add_argument("--stage", type=Path, default=ROOT / "outputs/unisafe09/hf-stage")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    if not args.stage.exists():
        manifest = stage_evidence(args.remote, args.stage)
    else:
        manifest = verify_local(args.stage)
    print(
        json.dumps({"files": len(manifest), "bytes": sum(x["bytes"] for x in manifest.values())}),
        flush=True,
    )
    if not args.upload:
        return
    from dotenv import dotenv_values
    from huggingface_hub import HfApi, hf_hub_download

    secrets = dotenv_values(ROOT / ".env")
    sensitive = [v.encode() for k, v in secrets.items() if v and ("TOKEN" in k or "KEY" in k)]
    # Scan the uncompressed originals, not only compressed archives.
    for parent in (args.remote, args.stage / "receipts"):
        for path in parent.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".log", ".txt", ".md", ".py"}:
                data = path.read_bytes()
                if any(value in data for value in sensitive):
                    raise ValueError(f"Sensitive value detected in {path.name}")
    repo = "Sachioster/maniguard-jar-replication-20261022"
    api = HfApi(token=secrets.get("HF_TOKEN"))
    info = api.repo_info(repo, repo_type="dataset")
    if not info.private:
        raise ValueError("Destination is not private")
    commit = api.upload_folder(
        folder_path=args.stage,
        repo_id=repo,
        repo_type="dataset",
        parent_commit=info.sha,
        commit_message="task9: preserve UNISafe adapter experiment evidence",
    )
    revision = commit.oid
    atomic_json(args.stage.parent / "upload.json", {"revision": revision, "utc": utc()})

    def verify(path):
        rel = str(path.relative_to(args.stage))
        copied = Path(
            hf_hub_download(
                repo,
                rel,
                repo_type="dataset",
                revision=revision,
                token=secrets.get("HF_TOKEN"),
                cache_dir=ROOT / "outputs/unisafe09/hf-verify-cache",
            )
        )
        if copied.stat().st_size != path.stat().st_size or digest(copied) != digest(path):
            raise ValueError(f"Private preservation hash mismatch: {rel}")
        return rel

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(verify, [p for p in args.stage.rglob("*") if p.is_file()]))
    manifest_path = args.stage / "receipts/unisafe09/sha256-manifest.json"
    rental = json.loads((ROOT / "progress/rental-unisafe09-20261025.json").read_text())
    marker = {
        "utc": utc(),
        "instance_id": rental["instance_id"],
        "local_verified": True,
        "private_hf_verified": True,
        "pinned_revisions": {repo: revision},
        "verified_files": len(checked),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": digest(manifest_path),
    }
    atomic_json(ROOT / "progress/receipts/09-unisafe-preservation.json", marker)
    print(json.dumps(marker))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
