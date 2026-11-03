"""Verify or restore the preserved SafeLeWM files, without running experiments."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile


def digest(stream):
    checksum = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        checksum.update(block)
    return checksum.hexdigest()


def safe_destination(destination, relative):
    path = destination / relative
    if not path.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f"Unsafe restoration path: {relative}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="manifest.json", help="Manifest path within archive/")
    parser.add_argument("--verify", action="store_true", help="Verify all preserved file bytes")
    parser.add_argument("--restore", type=Path, help="Destination corresponding to SafeLeWM/")
    parser.add_argument("--prefix", default="", help="Original path prefix to restore")
    args = parser.parse_args()
    if not args.verify and args.restore is None:
        parser.error("Choose --verify or --restore DESTINATION")
    root = Path(__file__).resolve().parent
    manifest = json.loads(safe_destination(root, args.manifest).read_text())
    files = manifest["original_files"]
    selected = {name: row for name, row in files.items() if name.startswith(args.prefix)}
    if args.restore is not None and not selected:
        parser.error("Prefix matched no original files")
    wanted = files if args.verify else selected
    needed = {row.get("archive", row.get("archive_file")) for row in wanted.values()}
    verified = set()
    for archive in sorted(needed):
        source = safe_destination(root, archive)
        with source.open("rb") as stream:
            actual = digest(stream)
        if actual != manifest["archives"][archive]:
            raise ValueError(f"Archive checksum mismatch: {archive}")
        members = {}
        for name, row in wanted.items():
            if row.get("archive") == archive:
                members.setdefault(row["member"], []).append((name, row))
            elif row.get("archive_file") == archive:
                if actual != row["sha256"]:
                    raise ValueError(f"Original archive checksum mismatch: {name}")
                verified.add(name)
                if args.restore is not None and name in selected:
                    target = safe_destination(args.restore, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        raise FileExistsError(target)
                    shutil.copyfile(source, target)
                    target.chmod(row["mode"])
        with tarfile.open(source, "r:gz") as handle:
            for member in handle:
                if member.name not in members:
                    continue
                if not member.isfile():
                    raise ValueError(f"Expected regular member: {member.name}")
                with handle.extractfile(member) as stream:
                    actual = digest(stream)
                for name, row in members.pop(member.name):
                    if actual != row["sha256"]:
                        raise ValueError(f"Member checksum mismatch: {name}")
                    verified.add(name)
                    if args.restore is not None and name in selected:
                        target = safe_destination(args.restore, name)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with handle.extractfile(member) as stream, target.open("xb") as output:
                            shutil.copyfileobj(stream, output)
                        target.chmod(row["mode"])
                        # Ordering audits use the archived filesystem timestamps.
                        # Restoring bytes must not replace them with extraction time.
                        os.utime(target, (member.mtime, member.mtime))
        if members:
            raise ValueError(f"Missing members in {archive}: {list(members)}")
    if verified != set(wanted):
        raise ValueError("Not every requested original file was verified")
    print(f"Verified {len(verified)} original files across {len(needed)} archives.")
    if args.restore is not None:
        print(f"Restored {len(selected)} files to {args.restore}.")


if __name__ == "__main__":
    main()
