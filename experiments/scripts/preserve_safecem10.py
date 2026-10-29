"""Preserve the task-10 evidence and verify every private object by pinned revision."""

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
from helpers.safecem10_protocol import digest

REPO = "Sachioster/maniguard-jar-replication-20261022"
RENTAL = ROOT / "progress/rental-safecem10-20261025.json"
# Stage C2 ran partly on a second rental fed by relay_safecem10.py. Its episode results are
# installed in the primary's attempts; its own logs, queue and determinism check are mirrored
# to HELPER_REMOTE and archived here, and it gets its own destroy marker.
HELPER_RENTAL = ROOT / "progress/rental-safecem10b-20261025.json"
HELPER_REMOTE = ROOT / "outputs/safecem10/helper-remote"
LOCAL_RECORDS = (
    "progress/10-safecem-dial-plan.md",
    "progress/rental-safecem10-20261025.json",
    "progress/rental-safecem10b-20261025.json",
    "progress/rental-safecem10b-20261025-rejected-offer.json",
    "outputs/safecem10/control/finance.json",
    "outputs/safecem10/control/relay.log",
    "outputs/safecem10/control/relay-state.json",
)


def verify_local(stage):
    stage = Path(stage)
    if stage.name == "safecem10" and stage.parent.name == "receipts":
        stage = stage.parent.parent
    manifest_path = stage / "receipts/safecem10/sha256-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    actual = {str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file()}
    if actual != set(manifest) | {str(manifest_path.relative_to(stage))}:
        raise ValueError("Preservation inventory differs")
    for name, item in manifest.items():
        path = stage / name
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Preservation hash differs: {name}")
    return manifest


def stage_evidence(remote, stage, helper=None):
    stage.mkdir(parents=True, exist_ok=False)
    recordings = stage / "recordings/safecem10"
    receipts = stage / "receipts/safecem10"
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
    for path in (ROOT / "progress/receipts").glob("10-safecem-*"):
        if path.is_file():
            shutil.copy2(path, receipts / path.name)
    for name in LOCAL_RECORDS:
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, receipts / Path(name).name)
    if helper is not None and Path(helper).exists():
        with tarfile.open(recordings / "helper-run-metadata.tar.gz", "w:gz") as archive:
            archive.add(helper, arcname="helper")
    analysis = ROOT / "outputs/safecem10/analysis"
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
    parser.add_argument("--remote", type=Path, default=ROOT / "outputs/safecem10/remote")
    parser.add_argument("--stage", type=Path, default=ROOT / "outputs/safecem10/hf-stage")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--helper-remote", type=Path, default=HELPER_REMOTE)
    args = parser.parse_args()
    if not args.stage.exists():
        manifest = stage_evidence(args.remote, args.stage, args.helper_remote)
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
    for parent in (args.remote, args.stage / "receipts", args.helper_remote):
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
        commit_message="task10: preserve SafeCEM Push-T dial experiment evidence",
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
                cache_dir=ROOT / "outputs/safecem10/hf-verify-cache",
            )
        )
        if copied.stat().st_size != path.stat().st_size or digest(copied) != digest(path):
            raise ValueError(f"Private preservation hash mismatch: {rel}")
        return rel

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        checked = list(pool.map(verify, [p for p in args.stage.rglob("*") if p.is_file()]))
    manifest_path = args.stage / "receipts/safecem10/sha256-manifest.json"
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
    atomic_json(ROOT / "progress/receipts/10-safecem-preservation.json", marker)
    print(json.dumps(marker))
    staged_helper = args.stage / "recordings/safecem10/helper-run-metadata.tar.gz"
    if HELPER_RENTAL.exists() and staged_helper.exists():
        helper_marker = {
            **marker,
            "instance_id": json.loads(HELPER_RENTAL.read_text())["instance_id"],
            "evidence": "recordings/safecem10/helper-run-metadata.tar.gz and the relay records "
            "in receipts/safecem10; the helper's episode results are installed in the "
            "primary's attempts",
        }
        atomic_json(ROOT / "progress/receipts/10-safecem-preservation-helper.json", helper_marker)
        print(json.dumps(helper_marker))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"error_type": type(error).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
