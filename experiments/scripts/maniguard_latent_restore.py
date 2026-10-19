"""Restore only verified historical interpreters into staging, without reviving jobs."""

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = "cca645c08ea7cb6f67e2f5939cbe8cf37aac8d50d7610e232020edb0200fd135"
ALLOWED = ["third_party/openpi/.venv", "data/maniguard/envs/sim", "data/maniguard/sim-freeze.txt"]
PYTHON_LINK = {"name": "third_party/openpi/.venv/bin/python",
               "target": "/.uv/python_install/cpython-3.11-linux-x86_64-gnu/bin/python3.11"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    args = parser.parse_args()
    if str(ROOT) != "/workspace/Safety-Dial":
        raise ValueError("The historical editable installations require /workspace/Safety-Dial")
    digest = hashlib.sha256()
    with args.archive.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != EXPECTED:
        raise ValueError("Historical runtime archive checksum mismatch")
    for name in ALLOWED:
        if (ROOT / name).exists() or (ROOT / name).is_symlink():
            raise FileExistsError("Refuse to overwrite an existing runtime: " + name)
    args.staging.mkdir(parents=True, exist_ok=False)
    links, count = [], 0
    with tarfile.open(args.archive, "r|gz") as archive:
        for member in archive:
            if not any(member.name == p or member.name.startswith(p + "/") for p in ALLOWED):
                continue
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe archive member")
            try:
                archive.extract(member, path=args.staging, filter="data")
            except (tarfile.LinkOutsideDestinationError, tarfile.AbsoluteLinkError):
                link = {"name": member.name, "target": member.linkname}
                if not member.issym() or link != PYTHON_LINK:
                    raise ValueError("Unexpected external interpreter link") from None
                links.append(link)
            count += 1
    for name in ALLOWED:
        destination = ROOT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        (args.staging / name).rename(destination)
    for link in links:
        (ROOT / link["name"]).symlink_to(link["target"])
    receipt = {"archive_sha256": EXPECTED, "members": count, "external_links": links,
               "restored_epoch": time.time(), "restored_jobs": False}
    (args.staging / "restore-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
