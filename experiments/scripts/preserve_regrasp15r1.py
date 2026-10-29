"""Stage, upload and verify task 15 evidence for both host cohorts, kept separate.

historical: the original host (instance 57727030) under receipts/regrasp15/,
recordings/regrasp15/ and jar/regrasp15_*; recovery: the replacement host under
receipts/regrasp15r1/, recordings/regrasp15r1/ and jar/regrasp15r1_*. Sources are the
executed source bundles each host received, never copies of the current worktree.
Every staged file is re-downloaded at the pinned revision and compared by SHA-256.
Existing dataset paths are never deleted or rewritten. Authentication stays on the
workstation; text files are scanned for workstation secrets before upload.

    .venv/bin/python experiments/scripts/preserve_regrasp15r1.py historical --upload
    .venv/bin/python experiments/scripts/preserve_regrasp15r1.py recovery --upload --marker
"""

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/todo1"))
from upload_hf import index_entry, task_folder, write_index  # noqa: E402

REPO = "Sachioster/maniguard-jar-replication-20261022"
RECEIPTS = ROOT / "progress/receipts"
COHORTS = {
    "historical": {
        "prefix": "regrasp15",
        "raw": ROOT / "outputs/regrasp15/remote",
        "local": {name: ROOT / "outputs/regrasp15" / name
                  for name in ("stage0", "stage0-partial", "design", "setup", "control")},
        "bundles": ROOT / "outputs/regrasp15/source",
        "documents": [ROOT / "progress/15-regrasp-dial-plan.md",
                      ROOT / "progress/goal-prompt-15-regrasp-handoff.md",
                      *sorted((ROOT / "progress").glob("rental-regrasp15-*.json")),
                      ROOT / "outputs/regrasp15r1/historical/inventory.json",
                      RECEIPTS / "15-regrasp-recovery-evidence-20261026.json"],
        "receipts": lambda p: p.name.startswith("15-regrasp-") and "recovery" not in p.name,
        "stage": ROOT / "outputs/regrasp15r1/hf-stage-historical",
        "readme": (
            "# Task 15 regrasp adapter: original host (instance 57727030)\n\n"
            "Stage 0, the four stage A engineering episodes, the 80-episode stage B dial "
            "screen and the start of stage C on one RTX 3090. The host left the provider "
            "during stage C (about 05:40 UTC on 26 October 2026); jobs 300 to 303 are "
            "complete, job 304 survives only as its host validation event in "
            "raw/logs/stage-c-queue.log, and later work was lost. This cohort is reported "
            "descriptively and never pooled with the recovery cohort (receipts/regrasp15r1/). "
            "raw/ is the complete local mirror; source-bundles/ are the executed source "
            "archives. Canonical outcome folders are jar/regrasp15_<stage>_<arm>/seed<seed>.\n"
        ),
    },
    "recovery": {
        "prefix": "regrasp15r1",
        "raw": ROOT / "outputs/regrasp15r1/remote",
        "local": {"control": ROOT / "outputs/regrasp15r1/control"},
        "bundles": ROOT / "outputs/regrasp15r1/source",
        "documents": [ROOT / "progress/15-regrasp-recovery-amendment.md",
                      ROOT / "progress/15-regrasp-dial-plan.md",
                      *sorted((ROOT / "progress").glob("rental-regrasp15r1-*.json")),
                      ROOT / "configs/regrasp15r1/recovery.json",
                      ROOT / "configs/regrasp15r1/historical-stage-c-decision.json"],
        "receipts": lambda p: p.name.startswith("15-regrasp-") and (
            "recovery" in p.name or p.name in ("15-regrasp-results.md",
                                               "15-regrasp-stage-d.json")),
        "analysis": ROOT / "outputs/regrasp15r1/analysis",
        "stage": ROOT / "outputs/regrasp15r1/hf-stage-recovery",
        "readme": (
            "# Task 15 regrasp adapter: recovery cohort on a replacement host\n\n"
            "One complete new-host cohort after the original host disappeared: stage A "
            "engineering, stage C (native against select-0.06) and stage D (project-0.10), "
            "per progress/15-regrasp-recovery-amendment.md. Comparisons are within this "
            "cohort only; the original host's episodes are under receipts/regrasp15/. raw/ is "
            "the complete mirror including failed attempts; source-bundles/ are the executed "
            "source archives. Canonical outcome folders are jar/regrasp15r1_<stage>_<arm>/"
            "seed<seed>.\n"
        ),
    },
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(remote, stage, prefix):
    by_run = {}
    for validated in sorted(remote.glob("episodes/*/validation.json")):
        folder = validated.parent
        row = json.loads(validated.read_text())["row"]
        task, level = row["scene_name"].split("/")
        job = json.loads((folder / "job.json").read_text())
        name = f"{prefix}_{job['stage']}_{job['arm']}"
        dest = stage / "jar" / name / f"seed{row['seed']}"
        leaf = dest / task_folder(task, row)
        leaf.mkdir(parents=True, exist_ok=False)
        (leaf / "row.json").write_text(json.dumps(row, indent=2) + "\n")
        for suffix in ("main.mp4", "wrist.mp4", "ltl.json"):
            path = next(folder.rglob(f"{level}_{suffix}"))
            shutil.copy2(path, leaf / path.name)
        by_run.setdefault((name, row["seed"]), []).append(row)
        config = next(folder.rglob("eval_config.json"), None)
        if config is not None:
            shutil.copy2(config, dest / f"eval_config_task{int(task.split('_')[1]):02d}.json")
    for (name, seed), rows in by_run.items():
        dest = stage / "jar" / name / f"seed{seed}"
        (dest / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        write_index(dest, f"{name}/seed{seed}",
                    [index_entry(*r["scene_name"].split("/"), r) for r in rows])
    return sum(len(v) for v in by_run.values())


def stage_cohort(kind):
    spec = COHORTS[kind]
    stage, prefix = spec["stage"], spec["prefix"]
    stage.mkdir(parents=True, exist_ok=False)
    shutil.copytree(spec["raw"], stage / "recordings" / prefix / "raw")
    for name, path in spec["local"].items():
        if path.exists():
            shutil.copytree(path, stage / "recordings" / prefix / "local" / name)
    receipts = stage / "receipts" / prefix
    (receipts / "source-bundles").mkdir(parents=True)
    for bundle in sorted(spec["bundles"].glob("*.tar.gz")):
        shutil.copy2(bundle, receipts / "source-bundles" / bundle.name)
    for path in sorted(RECEIPTS.iterdir()):
        if path.is_file() and spec["receipts"](path):
            shutil.copy2(path, receipts / path.name)
    for path in spec["documents"]:
        if path.exists():
            shutil.copy2(path, receipts / path.name)
    if spec.get("analysis") and spec["analysis"].exists():
        shutil.copytree(spec["analysis"], receipts / "analysis")
    episodes = canonical(spec["raw"], stage, prefix)
    (receipts / "README.md").write_text(spec["readme"])
    manifest = {str(p.relative_to(stage)): {"bytes": p.stat().st_size, "sha256": digest(p)}
                for p in sorted(stage.rglob("*")) if p.is_file()}
    (receipts / "sha256-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"files": len(manifest), "bytes": sum(v["bytes"] for v in manifest.values()),
            "canonical_episodes": episodes}


def manifest_path(stage):
    found = sorted(Path(stage).glob("receipts/*/sha256-manifest.json"))
    if len(found) != 1:
        raise ValueError("Expected exactly one preservation manifest")
    return found[0]


def verify_local(stage):
    stage = Path(stage)
    path = manifest_path(stage)
    manifest = json.loads(path.read_text())
    actual = {str(p.relative_to(stage)) for p in stage.rglob("*") if p.is_file()}
    if actual != set(manifest) | {str(path.relative_to(stage))}:
        raise ValueError("Local preservation inventory differs from the manifest")
    for name, spec in manifest.items():
        target = stage / name
        if target.stat().st_size != spec["bytes"] or digest(target) != spec["sha256"]:
            raise ValueError(f"Local evidence hash mismatch: {name}")
    return manifest


def upload(kind):
    from dotenv import dotenv_values
    from huggingface_hub import HfApi, hf_hub_download

    spec = COHORTS[kind]
    stage = spec["stage"]
    values = dotenv_values(ROOT / ".env")
    secrets = [v for k, v in values.items() if v and ("TOKEN" in k or "KEY" in k)]
    for path in stage.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".jsonl", ".log", ".txt", ".md", ".py",
                                              ".sh", ".yaml", ".diff"):
            data = path.read_bytes()
            if any(secret.encode() in data for secret in secrets):
                raise ValueError(f"Sensitive value detected in {path.name}; upload refused")
    manifest = verify_local(stage)
    token = values.get("HF_TOKEN")
    api = HfApi(token=token)
    info = api.repo_info(REPO, repo_type="dataset")
    if not info.private:
        raise ValueError("Expected the established private dataset")
    existing = set(api.list_repo_files(REPO, repo_type="dataset", revision=info.sha))
    if existing & set(manifest):
        raise ValueError("Refusing to rewrite existing dataset paths")
    commit = api.upload_folder(
        folder_path=stage, repo_id=REPO, repo_type="dataset", parent_commit=info.sha,
        commit_message=f"task15: preserve the {kind} regrasp adapter cohort",
    )
    revision = commit.oid
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    cache = stage.parent / f"hf-verify-cache-{kind}"

    def verify(path):
        rel = str(path.relative_to(stage))
        downloaded = hf_hub_download(REPO, rel, repo_type="dataset", revision=revision,
                                     token=token, cache_dir=cache)
        if digest(downloaded) != digest(path):
            raise ValueError(f"Upload hash mismatch: {rel}")
        return rel

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        verified = list(executor.map(verify, files))
    receipt = {
        "repo": REPO, "cohort": kind, "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "private": api.repo_info(REPO, repo_type="dataset").private, "revision": revision,
        "parent_revision": info.sha, "verified_files": len(verified),
        "verified_bytes": sum(p.stat().st_size for p in files),
        "manifest_sha256": digest(manifest_path(stage)),
        "verification": "Downloaded every staged file at the pinned revision; SHA-256 identical",
    }
    (stage.parent / f"upload-verification-{kind}.json").write_text(
        json.dumps(receipt, indent=2) + "\n")
    shutil.rmtree(cache, ignore_errors=True)
    return receipt


def marker():
    """Destroy guard input, written only from two successful verifications."""
    rental = json.loads((ROOT / "progress/rental-regrasp15r1-20261026.json").read_text())
    receipts = {kind: json.loads((ROOT / f"outputs/regrasp15r1/upload-verification-{kind}.json")
                                 .read_text()) for kind in COHORTS}
    stage = COHORTS["recovery"]["stage"]
    verify_local(stage)
    path = manifest_path(stage)
    for kind, receipt in receipts.items():
        if not (receipt["private"] and receipt["verified_files"] > 0
                and receipt["manifest_sha256"] == digest(manifest_path(COHORTS[kind]["stage"]))):
            raise ValueError(f"The {kind} cohort is not privately verified")
    data = {
        "instance_id": rental["instance_id"], "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "local_verified": True, "private_hf_verified": True,
        "pinned_revisions": {kind: r["revision"] for kind, r in receipts.items()},
        "manifest_path": str(path), "manifest_sha256": digest(path),
    }
    out = ROOT / "outputs/regrasp15r1/preservation-marker.json"
    out.write_text(json.dumps(data, indent=2) + "\n")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=sorted(COHORTS))
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--marker", action="store_true")
    args = parser.parse_args()
    if not COHORTS[args.cohort]["stage"].exists():
        print(json.dumps({"staged": stage_cohort(args.cohort)}), flush=True)
    if args.upload:
        print(json.dumps(upload(args.cohort), indent=2), flush=True)
    if args.marker:
        print(json.dumps(marker(), indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Third-party network exceptions may contain credential-bearing URLs.
        print("Preservation failed:", type(error).__name__, str(error)[:200]
              if isinstance(error, ValueError) else "", file=sys.stderr)
        raise SystemExit(1) from None
