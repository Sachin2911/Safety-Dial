"""Private archive of every mechanism evidence file, with per-member hashes."""

import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/mechanism-20261101"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    assert json.loads((SOURCE / "run-v1/audit.json").read_text())["passed"]
    assert (SOURCE / "run-v1/analysis.json").exists()
    destination = ROOT / "outputs/mechanism-20261101-evidence.tar.gz"
    assert not destination.exists()
    supporting = SOURCE / "supporting-sources"
    supporting.mkdir(exist_ok=True)
    for name in ["preserve_mechanism.py", "launch_mechanism.py"]:
        (supporting / name).write_bytes((ROOT / "experiments/scripts" / name).read_bytes())
    # Packaging logs remain open and are collected separately after the archive closes.
    paths = sorted(
        p
        for p in SOURCE.rglob("*")
        if p.is_file() and p.name not in {"orchestration.log", "preservation.log"}
    )
    entries = {}
    with tarfile.open(destination, "w:gz") as archive:
        for path in paths:
            member = str(path.relative_to(SOURCE.parent))
            entries[str(path.relative_to(ROOT))] = dict(
                sha256=digest(path),
                bytes=path.stat().st_size,
                mode=path.stat().st_mode & 0o777,
                archive="evidence/mechanism-20261101.tar.gz",
                member=member,
            )
            archive.add(path, arcname=member)
    manifest = dict(
        schema="safelewm-cleanup-v1",
        created="2026-11-01",
        archives={"evidence/mechanism-20261101.tar.gz": digest(destination)},
        original_files=entries,
    )
    (ROOT / "outputs/mechanism-20261101-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(files=len(entries), bytes=destination.stat().st_size, sha256=digest(destination))
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
