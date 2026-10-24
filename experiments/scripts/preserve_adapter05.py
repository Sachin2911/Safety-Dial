"""Stage complete evidence locally, then upload and verify at a pinned HF revision.

Authentication remains on the workstation. Existing dataset paths are not deleted
or rewritten; only the fresh adapter05 prefixes are added.
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
    raw = stage / "recordings/adapter05/raw"
    shutil.copytree(remote, raw)
    receipts = stage / "receipts/adapter05"
    receipts.mkdir(parents=True)
    if analysis.exists():
        shutil.copytree(analysis, receipts / "analysis")
    gate_evidence = ROOT / "outputs/adapter05/task4-gate-evidence"
    if gate_evidence.exists():
        shutil.copytree(gate_evidence, receipts / "task4-gate-evidence")
    for path in (ROOT / "progress/receipts").glob("05-adapter-*"):
        if path.is_file():
            shutil.copy2(path, receipts / path.name)
    for path in [
        ROOT / "progress/05-execution-adapter.md",
        ROOT / "progress/rental-adapter05-20261024.json",
    ]:
        shutil.copy2(path, receipts / path.name)
    shutil.copytree(
        ROOT / "experiments",
        receipts / "experiments",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    by_run = {}
    for validated in sorted(remote.glob("episodes/*/validation.json")):
        folder = validated.parent
        row = json.loads(validated.read_text())["row"]
        task, level = row["scene_name"].split("/")
        job = json.loads((folder / "job.json").read_text())
        prefix = (
            f"adapter05_{job['arm']}"
            if job["stage"] == "pilot"
            else f"adapter05_engineering_{job['arm']}"
        )
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
        "# Task 5 bounded execution adapter\n\n"
        "APEX-inspired implementation, not verified author code. "
        "Two engineering episodes are separate from the 36 planned scored episodes. "
        "Task 4 evidence gates scored admission; see the launch decision and run receipts.\n\n"
        "Canonical pilot folders: `jar/adapter05_<arm>/seed<seed>`. Engineering folders: "
        "`jar/adapter05_engineering_<arm>/seed<seed>`. Every raw attempt, NPZ, journal, "
        "log and executed source archive is under `recordings/adapter05/raw`. "
        "Analysis and protocol are under `receipts/adapter05`. "
        "All failed or partial attempts remain preserved.\n"
    )
    manifest = {
        str(p.relative_to(stage)): {"bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(stage.rglob("*"))
        if p.is_file()
    }
    (receipts / "sha256-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", type=Path, default=ROOT / "outputs/adapter05/remote")
    parser.add_argument("--analysis", type=Path, default=ROOT / "outputs/adapter05/analysis")
    parser.add_argument("--stage", type=Path, default=ROOT / "outputs/adapter05/hf-stage")
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
        commit_message="todo5: preserve execution adapter implementation and pilot evidence",
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
                cache_dir=ROOT / "outputs/adapter05/hf-verify-cache",
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
