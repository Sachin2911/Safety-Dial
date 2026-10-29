"""Verify that compressed preservation covers every recorded raw byte exactly once."""

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments/scripts")]
from helpers.unisafe09_infra import atomic_json, utc
from helpers.unisafe09_protocol import digest
from preserve_unisafe09 import verify_local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = verify_local(args.stage)
    raw_path = args.stage / "receipts/unisafe09/raw-files.json"
    raw = json.loads(raw_path.read_text())
    seen = {}
    archives = sorted((args.stage / "recordings/unisafe09").glob("*.tar.gz"))
    if not archives:
        raise ValueError("No evidence archives found")
    for path in archives:
        prefix = "" if path.name == "run-metadata.tar.gz" else "attempts/"
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                rel = PurePosixPath(member.name)
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError("Unsafe archive member name")
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ValueError("Unexpected nonregular archive member")
                name = prefix + str(rel)
                if name not in raw or name in seen:
                    raise ValueError("Archive contains unexpected or duplicate raw file")
                contents = archive.extractfile(member)
                if contents is None:
                    raise ValueError("Archive member has no readable contents")
                hasher = hashlib.sha256()
                size = 0
                with contents:
                    while block := contents.read(1024 * 1024):
                        size += len(block)
                        hasher.update(block)
                observed = {"bytes": size, "sha256": hasher.hexdigest()}
                if observed != raw[name]:
                    raise ValueError(f"Archived bytes differ from raw manifest: {name}")
                seen[name] = observed
    if set(seen) != set(raw):
        raise ValueError("The preservation package omits raw evidence")
    result = {
        "utc": utc(), "passed": True,
        "archive_count": len(archives), "raw_files_verified": len(seen),
        "raw_bytes_verified": sum(row["bytes"] for row in seen.values()),
        "packaged_files_verified": len(manifest) + 1,
        "sha256_manifest": str(args.stage / "receipts/unisafe09/sha256-manifest.json"),
        "sha256_manifest_hash": digest(args.stage / "receipts/unisafe09/sha256-manifest.json"),
        "raw_manifest_sha256": digest(raw_path),
        "audit_source_sha256": digest(Path(__file__)),
        "scope": (
            "Every regular raw file in the snapshot manifest is present exactly once "
            "in the compressed evidence, with matching size and SHA-256. This validates "
            "packaging coverage, not whether the underlying experiment is complete."
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
