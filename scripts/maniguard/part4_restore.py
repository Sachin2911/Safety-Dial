"""Restore only pinned, hash-verified Part 4 historical inputs into a new namespace."""

import argparse
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
from helpers.maniguard_part4_protocol import CONFIG, RUN, atomic, digest, read


def restore(run):
    run = Path(run)
    if (run / "donor-restore.json").exists() or (run / "donors").exists():
        raise FileExistsError("Never overwrite a donor restore or an existing scientific run")
    spec = read(CONFIG / "offline-protocol.json")["donors"]
    if HfApi().dataset_info(spec["repo_id"], revision=spec["revision"]).private is not True:
        raise ValueError("Original donor repository must remain private")
    manifest_name = spec["archive"].removesuffix(".gz") + ".json"
    manifest_path = Path(hf_hub_download(spec["repo_id"], manifest_name,
                                        repo_type="dataset", revision=spec["revision"]))
    manifest = read(manifest_path)
    if manifest["sha256"] != spec["sha256"]:
        raise ValueError("Donor manifest differs from frozen archive hash")
    archive = hf_hub_download(spec["repo_id"], spec["archive"],
                               repo_type="dataset", revision=spec["revision"])
    if digest(archive) != spec["sha256"]:
        raise ValueError("Downloaded donor archive failed SHA256 verification")
    run.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, run / manifest_path.name)
    expected = {r["path"]: r for r in manifest["files"]}
    inventory = []
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf:
            name = member.name
            if not member.isfile() or name not in expected:
                continue
            if not (name.startswith("part3/features/") or name.endswith(
                    ("/trace.jsonl", "/job.json", "/audit.json"))):
                continue
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe archive path")
            destination = run / "donors" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output)
            if digest(destination) != expected[name]["sha256"]:
                raise ValueError("Restored donor member hash mismatch")
            inventory.append(expected[name])
    atomic(run / "donor-restore.json", {
        "schema": "maniguard-part4-donor-restore-v1", "repo_id": spec["repo_id"],
        "revision": spec["revision"], "private": True,
        "archive_sha256": spec["sha256"], "archive_download_hash_verified": True,
        "files_hash_verified": True, "files": inventory, "at": time.time(),
    })
    metadata = run / "benchmark-metadata"
    metadata.mkdir(exist_ok=True)
    entries = []
    protocol = read(CONFIG / "offline-protocol.json")
    for tasks in protocol["partitions"].values():
        for task in tasks:
            name = "clutter_pickup/" + task + "/base/diagnostics.jsonl"
            revision = "af09438cebcd8f532e82cb5302701173ebb59f6e"
            source = hf_hub_download("IDEAS-Lab-Northwestern/ManiGuard-Bench", name,
                                     repo_type="dataset", revision=revision)
            destination = metadata / (task + "-diagnostics.jsonl")
            shutil.copyfile(source, destination)
            entries.append({"task": task, "path": name, "sha256": digest(destination)})
    atomic(metadata / "manifest.json", {"revision": revision, "files": entries})
    print(json.dumps({"restored_files": len(inventory), "new_physical_controls": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    restore(parser.parse_args().run)
