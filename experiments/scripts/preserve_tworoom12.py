"""Preserve the task-12 evidence and verify every private object by pinned revision."""

import argparse
import concurrent.futures
import json
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.tworoom12_protocol import digest

REPO = "Sachioster/maniguard-jar-replication-20261022"
RENTAL = ROOT / "progress/rental-tworoom12-20261025.json"
LOCAL_RECORDS = (
    "progress/12-tworoom-dial-plan.md",
    "progress/rental-tworoom12-20261025.json",
    "outputs/tworoom12/control/finance.json",
)


def verify_local(stage):
    stage = Path(stage)
    if stage.name == "tworoom12" and stage.parent.name == "receipts":
        stage = stage.parent.parent
    manifest_path = stage / "receipts/tworoom12/sha256-manifest.json"
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
    recordings = stage / "recordings/tworoom12"
    receipts = stage / "receipts/tworoom12"
    recordings.mkdir(parents=True)
    receipts.mkdir(parents=True)
    raw_manifest = {}
    for path in sorted(remote.rglob("*")):
        if path.is_file():
            raw_manifest[str(path.relative_to(remote))] = {
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
    atomic_json(receipts / "raw-files.json", raw_manifest)
    attempts = remote / "attempts"
    by_stage = {}
    for folder in sorted(attempts.glob("*")):
        if folder.is_dir():
            by_stage.setdefault(folder.name.split("-")[0], []).append(folder)
    with (receipts / "results.jsonl").open("w") as stream:
        for name, folders in sorted(by_stage.items()):
            with tarfile.open(recordings / f"attempts-{name}.tar.gz", "w:gz") as archive:
                for folder in folders:
                    archive.add(folder, arcname=folder.name)
                    result = folder / "result.json"
                    if result.exists():
                        row = json.loads(result.read_text())
                        row["attempt_dir"] = folder.name
                        stream.write(json.dumps(row) + "\n")
    with tarfile.open(recordings / "run-metadata.tar.gz", "w:gz") as archive:
        for child in sorted(remote.iterdir()):
            if child.name not in {"attempts", "queue"}:
                archive.add(child, arcname=child.name)
    for path in (ROOT / "progress/receipts").glob("12-tworoom-*"):
        if path.is_file():
            shutil.copy2(path, receipts / path.name)
    for name in LOCAL_RECORDS:
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, receipts / Path(name).name)
    analysis = ROOT / "outputs/tworoom12/analysis"
    if analysis.exists():
        for path in analysis.rglob("*"):
            if path.is_file():
                target = receipts / "analysis" / path.relative_to(analysis)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
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
    parser.add_argument("--remote", type=Path, default=ROOT / "outputs/tworoom12/remote")
    parser.add_argument("--stage", type=Path, default=ROOT / "outputs/tworoom12/hf-stage")
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
    for parent in (args.remote, args.stage / "receipts"):
        if not parent.exists():
            continue
        for path in parent.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".jsonl", ".log", ".txt", ".md", ".py"}:
                data = path.read_bytes()
                if any(value in data for value in sensitive):
                    raise ValueError(f"Sensitive value detected in {path.name}")
    api = HfApi(token=secrets.get("HF_TOKEN"))
    info = api.repo_info(REPO, repo_type="dataset")
    if not info.private:
        raise ValueError("Destination is not private")
    commit = api.upload_folder(
        folder_path=args.stage,
        repo_id=REPO,
        repo_type="dataset",
        parent_commit=info.sha,
        commit_message="task12: preserve SafeCEM TwoRoom dial experiment evidence",
    )
    revision = commit.oid
    atomic_json(args.stage.parent / "upload.json", {"revision": revision, "utc": utc()})

    def verify(path):
        rel = str(path.relative_to(args.stage))
        copied = Path(
            hf_hub_download(
                REPO,
                rel,
                repo_type="dataset",
                revision=revision,
                token=secrets.get("HF_TOKEN"),
                cache_dir=ROOT / "outputs/tworoom12/hf-verify-cache",
            )
        )
        if copied.stat().st_size != path.stat().st_size or digest(copied) != digest(path):
            raise ValueError(f"Private preservation hash mismatch: {rel}")
        return rel

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(verify, [p for p in args.stage.rglob("*") if p.is_file()]))
    manifest_path = args.stage / "receipts/tworoom12/sha256-manifest.json"
    rental = json.loads(RENTAL.read_text())
    marker = {
        "utc": utc(),
        "instance_id": rental["instance_id"],
        "local_verified": True,
        "private_hf_verified": True,
        "pinned_revisions": {REPO: revision},
        "verified_files": len(checked),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": digest(manifest_path),
    }
    atomic_json(ROOT / "progress/receipts/12-tworoom-preservation.json", marker)
    print(json.dumps(marker))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
