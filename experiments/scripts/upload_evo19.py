"""Preserve task 19 in the private dataset and verify every file at the pinned revision.

Stages the mirrored host outputs (episodes, recordings, journals, results), the search
state and analysis, the rental and freeze records, the protocol and the task 19 code under
`evo19/` in Sachioster/maniguard-jar-replication-20261022, refuses to rewrite existing
paths or to upload anything containing a local secret, then re-downloads every staged file
at the new revision and compares SHA-256. Writes outputs/evo19/upload-verification.json.
"""

import concurrent.futures
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "Sachioster/maniguard-jar-replication-20261022"
OUT = ROOT / "outputs/evo19"
STAGE = OUT / "hf-stage"
CODE = ["configs/evo19/protocol.json", "progress/19-evo-adapter-plan.md",
        "experiments/helpers/evo19_check.py", "experiments/helpers/evo19_runtime.py",
        "experiments/helpers/evo19_validation.py", "experiments/scripts/evolve_evo19.py",
        "experiments/scripts/worker_evo19.py", "experiments/scripts/patch_evo19.py",
        "experiments/scripts/launch_evo19.sh", "experiments/scripts/manage_evo19.py",
        "experiments/scripts/bootstrap_evo19.sh", "experiments/scripts/prepare_evo19.sh",
        "experiments/scripts/evo19_remote_guard.py", "experiments/scripts/analyze_evo19.py",
        "experiments/scripts/preserve_evo19.py", "experiments/scripts/upload_evo19.py",
        "experiments/scripts/swap_worker_evo19.sh", "experiments/scripts/closeout_evo19.py",
        "experiments/scripts/revalidate_evo19.py", "experiments/scripts/report_evo19.py",
        "experiments/tests/test_evo19.py"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    target = STAGE / "evo19"
    for host in sorted((OUT / "remote").iterdir()):
        if not (host / "preserved.json").exists():
            raise SystemExit(f"{host.name} has no verified local mirror")
        shutil.copytree(host, target / "hosts" / host.name)
    for folder in ("search", "analysis"):
        if (OUT / folder).exists():
            shutil.copytree(OUT / folder, target / folder)
    shutil.copytree(OUT / "control", target / "control",
                    ignore=shutil.ignore_patterns("*.lock", "*.pid"))
    for rel in CODE + [p.relative_to(ROOT).as_posix()
                       for p in (ROOT / "progress/receipts").glob("19-*")]:
        source = ROOT / rel
        if source.exists():
            (target / "source" / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / "source" / rel)


def main():
    from dotenv import dotenv_values
    from huggingface_hub import HfApi, hf_hub_download

    stage()
    values = dotenv_values(ROOT / ".env")
    secrets = [v for k, v in values.items() if v and ("TOKEN" in k or "KEY" in k)]
    files = sorted(p for p in STAGE.rglob("*") if p.is_file())
    for path in files:
        if path.suffix in (".json", ".jsonl", ".log", ".txt", ".md", ".py", ".sh", ".yaml",
                           ".diff"):
            data = path.read_bytes()
            if any(secret.encode() in data for secret in secrets):
                raise SystemExit(f"Sensitive value detected in {path}; upload refused")
    token = values.get("HF_TOKEN")
    api = HfApi(token=token)
    info = api.repo_info(REPO, repo_type="dataset")
    if not info.private:
        raise SystemExit("Expected the established private dataset")
    existing = set(api.list_repo_files(REPO, repo_type="dataset", revision=info.sha))
    staged = {str(p.relative_to(STAGE)) for p in files}
    if existing & staged:
        raise SystemExit("Refusing to rewrite existing dataset paths")
    commit = api.upload_folder(folder_path=STAGE, repo_id=REPO, repo_type="dataset",
                               parent_commit=info.sha,
                               commit_message="task19: preserve the evolved Jar adapter campaign")
    revision = commit.oid
    cache = OUT / "hf-verify-cache"

    def verify(path):
        rel = str(path.relative_to(STAGE))
        downloaded = hf_hub_download(REPO, rel, repo_type="dataset", revision=revision,
                                     token=token, cache_dir=cache)
        if digest(downloaded) != digest(path):
            raise ValueError(f"Upload hash mismatch: {rel}")
        return rel

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        verified = list(executor.map(verify, files))
    receipt = {"repo": REPO, "prefix": "evo19/", "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "private": api.repo_info(REPO, repo_type="dataset").private,
               "revision": revision, "parent_revision": info.sha,
               "verified_files": len(verified),
               "verified_bytes": sum(p.stat().st_size for p in files),
               "verification": "Re-downloaded every staged file at the pinned revision; SHA-256 identical"}
    (OUT / "upload-verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    shutil.rmtree(cache, ignore_errors=True)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
