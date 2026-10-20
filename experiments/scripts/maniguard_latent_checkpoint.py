"""Preserve completed new-study evidence and controller state in private repositories."""

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from maniguard_latent_preserve import preserve

from helpers.maniguard_latent_queue import atomic


def archive(path, entries):
    if path.exists():
        return
    temporary = path.with_suffix(".partial")
    secrets = [os.environ[name].encode() for name in
               ["HF_TOKEN", "GITHUB_TOKEN", "WANDB_API_KEY", "CONTAINER_API_KEY"]
               if len(os.environ.get(name, "")) >= 12]
    with tarfile.open(temporary, "w:gz", compresslevel=3) as output:
        for name, source in entries.items():
            if source.is_symlink() or not source.is_file():
                raise ValueError("Only ordinary explicitly selected study files may be preserved")
            if any(part in [".env", ".vast_api_key", "credentials", "behavior-1k.key"]
                   for part in Path(name).parts):
                raise ValueError("Credential files are excluded from study artifacts")
            contents = source.read_bytes()
            if any(value in contents for value in secrets):
                raise ValueError("A selected study artifact contains a credential: " + name)
            output.add(source, arcname=name, recursive=False)
    temporary.replace(path)


def snapshot_analysis(analysis, directory):
    """Keep flat reports and explicitly indexed nested evidence as frozen bytes."""
    analysis = Path(analysis).resolve()
    selected = {
        source.name: None for source in analysis.glob("*")
        if source.is_file() and not source.is_symlink()
        and source.suffix in [".json", ".csv", ".md", ".png", ".svg"]
    }
    index = analysis / "preservation-artifacts.json"
    if index.exists():
        manifest = json.loads(index.read_text())
        if manifest.get("schema") != "maniguard-explicit-analysis-artifacts-v1":
            raise ValueError("Unknown explicit analysis artifact manifest")
        seen = set()
        for item in manifest["files"]:
            relative = Path(item["path"])
            if (relative.is_absolute() or ".." in relative.parts or not relative.parts
                    or relative.as_posix() in seen
                    or any(part in [".env", ".vast_api_key", "credentials", "behavior-1k.key"]
                           for part in relative.parts)
                    or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
                    or type(item["bytes"]) is not int or item["bytes"] < 0):
                raise ValueError("Invalid explicitly selected analysis artifact")
            seen.add(relative.as_posix())
            selected[relative.as_posix()] = item
    exported = {}
    for name, expected in selected.items():
        source = analysis / name
        if source.resolve() != source or not source.is_file():
            raise ValueError("Analysis evidence must be an ordinary file inside analysis")
        contents = source.read_bytes()
        if expected is not None and (
            len(contents) != expected["bytes"]
            or hashlib.sha256(contents).hexdigest() != expected["sha256"]
        ):
            raise ValueError("Explicit analysis artifact changed before snapshot: " + name)
        target = Path(directory) / "analysis" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
        exported["analysis/" + name] = str(target)
    return exported


def _checkpoint(run, stage):
    if not stage or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in stage):
        raise ValueError("Invalid immutable stage name")
    run = Path(run).resolve()
    directory = run / "backups" / stage
    directory.mkdir(parents=True, exist_ok=True)
    index_path = run / "backups/index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"stages": {}}
    if stage in index["stages"]:
        return index["stages"][stage]
    frozen_index = directory / "selected.json"
    if not frozen_index.exists():
        queue = json.loads((run / "queue.json").read_text())
        backed = {name for item in index["stages"].values() for name in item["episodes"]}
        episodes = [Path(a["output"]).resolve() for a in queue["attempts"]
                    if a.get("status") in ["completed", "invalid"]]
        episodes += [p for p in run.iterdir() if p.is_dir()
                     and (p.name.startswith("engineering-") or p.name == "camera-v1")]
        episodes = sorted({p for p in episodes if p.name not in backed})
        backed_auxiliary = {name for item in index["stages"].values()
                            for name in item.get("auxiliary", [])}
        auxiliary = [run / name for name in ["authority-jar", "authority-clutter",
                                           "gripper-authority-20261020"]
                     if name not in backed_auxiliary and (run / name / "summary.json").exists()]
        source_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                text=True).strip()
        data, models = {}, {}
        for episode in episodes:
            for source in episode.rglob("*"):
                if source.is_file() and not source.is_symlink():
                    relative = str(Path("episodes") / episode.name / source.relative_to(episode))
                    (models if source.name == "controller.npz" else data)[relative] = str(source)
        # Exit receipts and stderr live beside episode directories. They are
        # required to distinguish valid outcomes from infrastructure failures.
        for attempt in queue["attempts"]:
            if attempt.get("status") not in ["completed", "invalid"]:
                continue
            for suffix in [".log", "-request.json", "-terminal.json"]:
                source = Path(attempt["output"]).parent / (attempt["id"] + suffix)
                if source.exists():
                    data["episodes/" + source.name] = str(source)
        for folder in auxiliary:
            for source in folder.rglob("*"):
                if source.is_file() and not source.is_symlink():
                    data[str(Path("authority") / folder.name / source.relative_to(folder))] = str(source)
        # Copy live ledgers now so archive creation and upload use a stable snapshot.
        for name in ["status.json", "queue.json", "attempts.jsonl", "recovery.jsonl",
                     "deadline-watchdog.json", "hf-assets.json", "handoff-runtime.json",
                     "deadline-credential-preflight.json",
                     "instance-resume-readback-20261020.json",
                     "instance-accounting-readback-20261020.json",
                     "provider-resume-schedule-readback.json", "goal-blocked-audit.json",
                     "restore/restored.json", "restore/pending-links.json",
                     "sim-freeze.txt", "policy-freeze.txt", "tools-freeze.txt",
                     "backups/index.json", "development-driver.json", "active-policy.json",
                     "active-worker-policies.json", "cost-snapshot.json",
                     "diagnostic-admission-hold-20261020.json",
                     "paired-recovery-before.json", "paired-recovery-verified.json",
                     "provider-extension-readback.json", "provider-extension-update.json",
                     "provider-extension-confirmation.json"]:
            source = run / name
            if source.exists():
                target = directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                data["run/" + name] = str(target)
        for source in run.glob("*.log"):
            target = directory / source.name
            target.write_bytes(source.read_bytes())
            data["logs/" + source.name] = str(target)
        for source in run.glob("server-*.jsonl"):
            target = directory / source.name
            target.write_bytes(source.read_bytes())
            data["server/" + source.name] = str(target)
        data.update(snapshot_analysis(run / "analysis", directory))
        for folder in [run / "search", run / "initial-controllers"]:
            if folder.exists():
                for source in folder.iterdir():
                    if source.is_file() and source.suffix in [".npz", ".json"]:
                        target = directory / (folder.name + "-" + source.name)
                        target.write_bytes(source.read_bytes())
                        models[str(Path(folder.name) / source.name)] = str(target)
        for source in (ROOT / "configs/maniguard/latent-evolution").glob("*.json"):
            target = directory / ("config-" + source.name)
            target.write_bytes(source.read_bytes())
            models["configs/" + source.name] = str(target)
        rental = ROOT / "configs/vast/maniguard-rental-20261020-comparison.json"
        if rental.exists():
            target = directory / rental.name
            target.write_bytes(rental.read_bytes())
            models["rental/" + rental.name] = str(target)
        atomic(frozen_index, {"episodes": [p.name for p in episodes],
                              "auxiliary": [p.name for p in auxiliary], "source_commit": source_commit,
                              "data": data, "models": models})
    selected = json.loads(frozen_index.read_text())
    for kind, field, repo in [
        ("model", "models", "Sachioster/maniguard-latent-evolution-controllers"),
        ("dataset", "data", "Sachioster/maniguard-latent-evolution-runs"),
    ]:
        package = directory / (kind + ".tar.gz")
        archive(package, {name: Path(source) for name, source in selected[field].items()})
        receipt = directory / (kind + "-receipt.json")
        if not receipt.exists():
            preserve(repo, kind, "20261019/" + stage,
                     {package.name: str(package), "selected.json": str(frozen_index)},
                     selected["source_commit"], receipt)
    index["stages"][stage] = {"episodes": selected["episodes"],
                               "auxiliary": selected.get("auxiliary", []),
                               "receipts": [str(directory / (k + "-receipt.json"))
                                            for k in ["model", "dataset"]]}
    atomic(index_path, index)
    print(json.dumps({"stage": stage, "episodes": selected["episodes"],
                      "remote_readback_verified": True}), flush=True)
    return index["stages"][stage]


def checkpoint(run, stage):
    directory = Path(run) / "backups"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "checkpoint.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _checkpoint(run, stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    checkpoint(args.run, args.stage)


if __name__ == "__main__":
    main()
