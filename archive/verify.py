"""Verify that archived source and evidence still match their pre-move contents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-local", action="store_true", help="Also check ignored local build artifacts."
    )
    args = parser.parse_args()
    archive = Path(__file__).resolve().parent
    manifest = json.loads((archive / "manifest.json").read_text())
    entries = list(manifest["files"])
    if args.include_local:
        entries.extend(manifest["local_only_files"])
    failures = []

    for item in entries:
        path = archive / item["archive_path"]
        if item["kind"] == "symlink":
            if not path.is_symlink() or os.readlink(path) != item["target"]:
                failures.append(f"Changed symlink: {item['archive_path']}")
            continue
        if path.is_symlink() or not path.is_file():
            failures.append(f"Missing regular file: {item['archive_path']}")
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
            failures.append(f"Changed content: {item['archive_path']}")
        if bool(path.stat().st_mode & 0o111) != bool(item["mode"] & 0o111):
            failures.append(f"Changed executable status: {item['archive_path']}")

    for item in manifest["shared_paths"]:
        path = archive / item["path"]
        wrong_link = not path.is_symlink() or os.readlink(path) != item["target"]
        missing_target = not path.exists() and not item.get("optional_local", False)
        if wrong_link or missing_target:
            failures.append(f"Broken shared path: {item['path']}")

    if failures:
        print("\n".join(failures))
        return 1
    print(f"Verified {len(entries)} archived files with unchanged contents.")
    print(f"Verified {len(manifest['shared_paths'])} shared paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
