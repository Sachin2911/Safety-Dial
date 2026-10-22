"""Download pinned public research assets; never upload the BEHAVIOR data bundle."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/maniguard"


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def snapshot(spec, destination, kind="dataset", patterns=None):
    result = Path(snapshot_download(
        spec["repo"], repo_type=kind, revision=spec["revision"],
        local_dir=destination, allow_patterns=patterns, max_workers=4,
    ))
    files = [p for p in result.rglob("*") if p.is_file() and ".cache" not in p.parts]
    record = {
        "repo": spec["repo"], "revision": spec["revision"], "repo_type": kind,
        "files": {str(p.relative_to(result)): {"bytes": p.stat().st_size, "sha256": digest(p)}
                  for p in sorted(files) if p.name != "download-provenance.json"},
    }
    (result / "download-provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Verified local hashes: {spec['repo']} ({len(files)} files)", flush=True)


def assets(spec):
    for filename, folder in spec["archives"].items():
        destination = DATA / "datasets" / folder
        marker = destination / ".safety-dial-download.json"
        if marker.exists():
            previous = json.loads(marker.read_text())
            if previous["revision"] != spec["revision"]:
                raise ValueError(f"Existing asset revision differs: {destination}")
            print(f"Already extracted: {folder}", flush=True)
            continue
        archive = Path(hf_hub_download(
            spec["repo"], filename, repo_type="dataset", revision=spec["revision"],
        ))
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                (destination / name).resolve().relative_to(destination.resolve())
            bundle.extractall(destination)
        marker.write_text(json.dumps({
            "repo": spec["repo"], "revision": spec["revision"], "archive": filename,
            "sha256": digest(archive), "bytes": archive.stat().st_size,
        }, indent=2) + "\n")
        print(f"Extracted pinned asset bundle: {folder}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("benchmark", "assets", "robot", "jar", "lid", "stack"))
    args = parser.parse_args()
    sources = json.loads((ROOT / "configs/maniguard/sources.json").read_text())
    if args.kind == "benchmark":
        snapshot(sources["benchmark"], DATA / "benchmark", patterns=[
            "README.md", "LICENSE", "jar_transport/*", "lid_transport/*", "stack_retrieve/*",
        ])
    elif args.kind == "assets":
        assets(sources["assets"])
    elif args.kind == "robot":
        snapshot(sources["robot"], DATA / "datasets/omnigibson-robot-assets/models/franka/"
                 "franka_panda_longfinger")
    else:
        spec = sources["policies"][args.kind]
        snapshot(spec, DATA / "policies" / args.kind, kind="model",
                 patterns=[f"{spec['step']}/*", "README.md", "LICENSE*"])


if __name__ == "__main__":
    main()
