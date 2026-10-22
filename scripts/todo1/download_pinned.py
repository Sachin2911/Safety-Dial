"""Download one pinned Hugging Face source for todo 1 and verify every file's hash.

Usage (any interpreter with huggingface_hub; HF_TOKEN is read from the environment):

    python scripts/todo1/download_pinned.py benchmark|robot|jar <destination>

Hashes are checked against the Hub manifest: LFS files by sha256, plain files by
git blob sha1. A provenance JSON is written beside the files. Never prints tokens.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

SOURCES = {
    "benchmark": {
        "repo": "IDEAS-Lab-Northwestern/ManiGuard-Bench",
        "repo_type": "dataset",
        "revision": "af09438cebcd8f532e82cb5302701173ebb59f6e",
        "allow_patterns": ["README.md", "LICENSE", "jar_transport/*"],
    },
    "robot": {
        "repo": "IDEAS-Lab-Northwestern/franka-panda-longfinger",
        "repo_type": "dataset",
        "revision": "5f65ec0a715bc88ea865c2c69133b08b395c66bc",
        "allow_patterns": None,
    },
    "jar": {
        "repo": "IDEAS-Lab-Northwestern/pi05-base-datagen-v1-jar-joint-2cam-lora",
        "repo_type": "model",
        "revision": "1d84eda070313a202a595e449fcf41a1a1e8a546",
        "allow_patterns": ["7400/*", "README.md", "LICENSE*", ".gitattributes"],
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    h = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(SOURCES))
    parser.add_argument("destination")
    args = parser.parse_args()
    spec = SOURCES[args.kind]
    dest = Path(args.destination)
    started = datetime.now(timezone.utc)

    local = Path(snapshot_download(
        spec["repo"], repo_type=spec["repo_type"], revision=spec["revision"],
        local_dir=dest, allow_patterns=spec["allow_patterns"], max_workers=8,
    ))

    api = HfApi()
    manifest = {
        entry.path: entry
        for entry in api.list_repo_tree(
            spec["repo"], repo_type=spec["repo_type"], revision=spec["revision"],
            recursive=True, expand=True,
        )
        if hasattr(entry, "size")
    }
    files = sorted(
        p for p in local.rglob("*")
        if p.is_file() and ".cache" not in p.parts and p.name != "download-provenance.json"
    )
    records, mismatches = {}, []
    for p in files:
        rel = str(p.relative_to(local))
        entry = manifest.get(rel)
        if entry is None:
            mismatches.append((rel, "not in manifest"))
            continue
        lfs = getattr(entry, "lfs", None)
        if lfs is not None:
            expected, actual, algo = lfs.sha256, sha256(p), "sha256"
        else:
            expected, actual, algo = entry.blob_id, git_blob_sha1(p), "git-blob-sha1"
        ok = expected == actual
        if not ok:
            mismatches.append((rel, f"{algo} expected {expected} got {actual}"))
        records[rel] = {"bytes": p.stat().st_size, algo: actual, "verified": ok}

    missing = sorted(set(manifest) - set(records))
    if spec["allow_patterns"] is not None:
        import fnmatch
        missing = [m for m in missing if any(fnmatch.fnmatch(m, pat) for pat in spec["allow_patterns"])]

    provenance = {
        "repo": spec["repo"], "repo_type": spec["repo_type"], "revision": spec["revision"],
        "allow_patterns": spec["allow_patterns"], "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(), "n_files": len(records),
        "total_bytes": sum(r["bytes"] for r in records.values()),
        "mismatches": mismatches, "missing_from_pattern": missing, "files": records,
    }
    (local / "download-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"{spec['repo']}@{spec['revision'][:8]}: {len(records)} files, "
          f"{provenance['total_bytes'] / 1e9:.2f} GB, mismatches={len(mismatches)}, missing={len(missing)}")
    for rel, why in mismatches:
        print(f"  MISMATCH {rel}: {why}")
    return 1 if mismatches or missing else 0


if __name__ == "__main__":
    sys.exit(main())
