"""Archive completed pilot evidence and hash every member. No provider operations."""

import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/adapter-pilot-20261101"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    assert json.loads((SOURCE / "run-v2/audit.json").read_text())["passed"]
    assert json.loads((SOURCE / "run-v2/inspection.json").read_text())["native_state_audit_passed"]
    assert json.loads((SOURCE / "run-v1/interruption-audit.json").read_text())["accounted"]
    destination = ROOT / "outputs/adapter-pilot-20261101-evidence.tar.gz"
    assert not destination.exists()
    supporting = SOURCE / "supporting-sources"
    supporting.mkdir(exist_ok=True)
    for name in [
        "inspect_adapter_pilot.py",
        "preserve_adapter_pilot.py",
        "run_adapter_audit.py",
        "inspect_interrupted_pilot.py",
        "setup_adapter_host.sh",
        "fetch_adapter_assets.py",
    ]:
        (supporting / name).write_bytes((ROOT / "experiments/scripts" / name).read_bytes())
    paths = sorted(path for path in SOURCE.rglob("*") if path.is_file())
    entries = {}
    with tarfile.open(destination, "w:gz") as archive:
        for path in paths:
            member = str(path.relative_to(SOURCE.parent))
            entries[str(path.relative_to(ROOT))] = dict(
                sha256=digest(path),
                bytes=path.stat().st_size,
                mode=path.stat().st_mode & 0o777,
                archive="evidence/adapter-pilot-20261101.tar.gz",
                member=member,
            )
            archive.add(path, arcname=member)
    manifest = dict(
        schema="safelewm-cleanup-v1",
        created="2026-11-01",
        archives={"evidence/adapter-pilot-20261101.tar.gz": digest(destination)},
        original_files=entries,
    )
    target = ROOT / "outputs/adapter-pilot-20261101-manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            dict(files=len(entries), bytes=destination.stat().st_size, sha256=digest(destination))
        )
    )


if __name__ == "__main__":
    main()
