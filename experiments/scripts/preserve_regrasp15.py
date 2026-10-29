"""Stage complete task 15 evidence locally, then upload and verify at a pinned HF revision.

Authentication remains on the workstation. Existing dataset paths are not deleted
or rewritten; only the fresh regrasp15 prefixes are added.
"""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/todo1"))
from upload_hf import index_entry, task_folder, write_index


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stage_evidence(remote, analysis, stage):
    stage.mkdir(parents=True, exist_ok=False)
    raw = stage / "recordings/regrasp15/raw"
    shutil.copytree(remote, raw)
    receipts = stage / "receipts/regrasp15"
    receipts.mkdir(parents=True)
    if analysis.exists():
        shutil.copytree(analysis, receipts / "analysis")
    for path in (ROOT / "progress/receipts").glob("15-regrasp-*"):
        if path.is_file():
            shutil.copy2(path, receipts / path.name)
    for path in [ROOT / "progress/15-regrasp-dial-plan.md",
                 *sorted((ROOT / "progress").glob("rental-regrasp15-*.json"))]:
        shutil.copy2(path, receipts / path.name)
    plan = json.loads((ROOT / "progress/receipts/15-regrasp-live-v1.json").read_text())
    for name in plan["source_sha256"]:
        target = receipts / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    by_run = {}
    for validated in sorted(remote.glob("episodes/*/validation.json")):
        folder = validated.parent
        row = json.loads(validated.read_text())["row"]
        task, level = row["scene_name"].split("/")
        job = json.loads((folder / "job.json").read_text())
        prefix = f"regrasp15_{job['stage']}_{job['arm']}"
        dest = stage / "jar" / prefix / f"seed{row['seed']}"
        leaf = dest / task_folder(task, row)
        leaf.mkdir(parents=True, exist_ok=False)
        (leaf / "row.json").write_text(json.dumps(row, indent=2) + "\n")
        for suffix in ("main.mp4", "wrist.mp4", "ltl.json"):
            path = next(folder.rglob(f"{level}_{suffix}"))
            shutil.copy2(path, leaf / path.name)
        by_run.setdefault((prefix, row["seed"]), []).append(row)
        config = next(folder.rglob("eval_config.json"), None)
        if config is not None:
            shutil.copy2(config, dest / f"eval_config_task{int(task.split('_')[1]):02d}.json")
    for (prefix, seed), rows in by_run.items():
        dest = stage / "jar" / prefix / f"seed{seed}"
        (dest / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        entries = [index_entry(*r["scene_name"].split("/"), r) for r in rows]
        write_index(dest, f"{prefix}/seed{seed}", entries)
    (receipts / "README.md").write_text(
        "# Task 15 regrasp adapter with a safety dial\n\n"
        "Fresh study with its own host and $10 ceiling: stage 0 steerability from task 7 "
        "query records, four stage A engineering episodes, the stage B dial screen and the "
        "stage C confirmation. Consult the frozen protocol for exact denominators. "
        "All attempts and original evidence are under recordings/regrasp15/raw. "
        "Canonical outcome folders are jar/regrasp15_<stage>_<arm>/seed<seed>.\n"
    )
    manifest = {
        str(p.relative_to(stage)): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(stage.rglob("*"))
        if p.is_file()
    }
    (receipts / "sha256-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_local(stage):
    stage = Path(stage)
    if (stage / "sha256-manifest.json").is_file() and stage.name == "regrasp15":
        stage = stage.parent.parent
    manifest_path = stage / "receipts/regrasp15/sha256-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    actual = {str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file()}
    if actual != set(manifest) | {str(manifest_path.relative_to(stage))}:
        raise ValueError("Local preservation inventory differs from the manifest")
    for name, spec in manifest.items():
        path = stage / name
        if path.stat().st_size != spec["bytes"] or digest(path) != spec["sha256"]:
            raise ValueError(f"Local evidence hash mismatch: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", type=Path, default=ROOT / "outputs/regrasp15/remote")
    parser.add_argument("--analysis", type=Path, default=ROOT / "outputs/regrasp15/analysis")
    parser.add_argument("--stage", type=Path, default=ROOT / "outputs/regrasp15/hf-stage")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    if not args.stage.exists():
        manifest = stage_evidence(args.remote, args.analysis, args.stage)
        print(
            json.dumps(
                {"staged_files": len(manifest), "bytes": sum(v["bytes"] for v in manifest.values())}
            ),
            flush=True,
        )
    if not args.upload:
        return
    from dotenv import dotenv_values
    from huggingface_hub import HfApi, hf_hub_download

    secrets = [
        v for k, v in dotenv_values(ROOT / ".env").items() if v and ("TOKEN" in k or "KEY" in k)
    ]
    # Refuse rather than publishing any known workstation secret in text evidence.
    for path in args.stage.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".log", ".txt", ".md", ".py"):
            data = path.read_bytes()
            if any(secret.encode() in data for secret in secrets):
                raise ValueError(f"Sensitive value detected in {path.name}; upload refused")
    verify_local(args.stage)
    token = dotenv_values(ROOT / ".env").get("HF_TOKEN")
    repo = "Sachioster/maniguard-jar-replication-20261022"
    api = HfApi(token=token)
    info = api.repo_info(repo, repo_type="dataset")
    if not info.private:
        raise ValueError("Expected the established private dataset")
    commit = api.upload_folder(
        folder_path=args.stage,
        repo_id=repo,
        repo_type="dataset",
        commit_message="task15: preserve regrasp adapter dial study and complete evidence",
        parent_commit=info.sha,
    )
    revision = commit.oid
    (args.stage.parent / "uploaded-revision.txt").write_text(revision + "\n")
    files = sorted(path for path in args.stage.rglob("*") if path.is_file())

    def verify(path):
        rel = str(path.relative_to(args.stage))
        downloaded = Path(
            hf_hub_download(
                repo,
                rel,
                repo_type="dataset",
                revision=revision,
                token=token,
                cache_dir=ROOT / "outputs/regrasp15/hf-verify-cache",
            )
        )
        if digest(downloaded) != digest(path):
            raise ValueError(f"Upload hash mismatch: {rel}")
        return rel

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        verified = list(executor.map(verify, files))
    receipt = {
        "repo": repo,
        "private": api.repo_info(repo, repo_type="dataset").private,
        "revision": revision,
        "verified_files": len(verified),
        "verified_bytes": sum(p.stat().st_size for p in files),
        "verification": "Downloaded every staged file at pinned revision; SHA-256 identical",
    }
    (args.stage.parent / "upload-verification.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Third-party network exceptions may contain credential-bearing URLs.
        print("Preservation failed:", type(error).__name__, file=sys.stderr)
        raise SystemExit(1) from None
