"""Download and hash-check the pinned private UNISafe checkpoint mirror.

Run with /venv/main/bin/python scripts/download_unisafe.py and HF_TOKEN set.
This does not install dependencies or load checkpoint objects.
"""

import argparse
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/unisafe/pretrained")
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env", override=False)
    source = json.loads((REPO_ROOT / "configs/download/unisafe.json").read_text())
    destination = Path(
        snapshot_download(
            repo_id=source["repo_id"],
            revision=source["revision"],
            local_dir=args.output,
        )
    )
    manifest_path = destination / "manifest.json"
    with manifest_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != source["manifest_sha256"]:
        raise RuntimeError("Downloaded manifest differs from the inspected mirror.")
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["files"]:
        path = destination / item["path"]
        if path.stat().st_size != item["bytes"]:
            raise RuntimeError(f"Size mismatch: {item['path']}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch: {item['path']}")
    print(f"Verified {len(manifest['files'])} original files at {source['revision']}")
    print(f"World model: {destination / 'latest.pt'}")
    print(f"Safety filter root: {destination / 'filter'}")


if __name__ == "__main__":
    main()
