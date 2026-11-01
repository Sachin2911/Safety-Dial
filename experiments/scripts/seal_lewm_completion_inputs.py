"""Seal the verified immutable analysis sibling into a canonical curated archive."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


class Reader:
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()

    def read(self, size=-1):
        raw = self.stream.read(size)
        self.digest.update(raw)
        return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("RUNPOD_POD_ID") == "uddcaj5qijcl3j", "Only the authorized rented host")
    source, out = args.source.resolve(), args.out.resolve()
    require(not out.exists() and not out.is_relative_to(source), "Fresh archive destination")
    assembled = json.loads((source / ".assembly-manifest.json").read_text())
    require(
        assembled["verified"]
        and assembled["namespace_count"] == 35
        and assembled["media_pairs"] == 48,
        "Full verified curated input set required",
    )
    expected = {name: row["sha256"] for name, row in assembled["files"].items()}
    for name in (".assembly-manifest.json", ".analysis-only-input-root.json"):
        expected[name] = sha(source / name)
    names = {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()}
    require(names == set(expected), "Curated input inventory changed")
    before = {n: ((source / n).stat().st_size, (source / n).stat().st_mtime_ns) for n in names}
    out.mkdir(parents=True)
    archive = out / "curated-inputs.tar.gz"
    partial = out / "curated-inputs.tar.gz.part"
    records = {}
    with tarfile.open(partial, "w:gz", compresslevel=1) as tar:
        for name in sorted(names):
            path = source / name
            require(
                not path.is_symlink() and path.resolve().is_relative_to(source),
                "Unsafe curated source",
            )
            with path.open("rb") as incoming:
                reader = Reader(incoming)
                info = tarfile.TarInfo(name)
                info.size, info.mode = before[name][0], 0o400
                tar.addfile(info, reader)
            require(reader.digest.hexdigest() == expected[name], "Curated source SHA changed")
            records[name] = {"sha256": expected[name], "bytes": before[name][0]}
        manifest = {
            "schema": "lewm-curated-analysis-inputs-v1",
            "files": records,
            "namespace_count": 35,
            "fixed_media_pairs": 48,
            "assembly_sha256": expected[".assembly-manifest.json"],
            "new_model_queries": 0,
            "new_simulator_actions": 0,
        }
        raw = (json.dumps(manifest, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo("CURATED-INPUTS-MANIFEST.json")
        info.size, info.mode = len(raw), 0o400
        tar.addfile(info, io.BytesIO(raw))
    checked = {}
    with tarfile.open(partial, "r|gz") as tar:
        for member in tar:
            require(member.isfile() and member.name not in checked, "Unexpected archive member")
            incoming = tar.extractfile(member)
            hashed = hashlib.sha256()
            for block in iter(lambda: incoming.read(1024 * 1024), b""):
                hashed.update(block)
            checked[member.name] = hashed.hexdigest()
    require(set(checked) == names | {"CURATED-INPUTS-MANIFEST.json"}, "Archive member set differs")
    require(all(checked[n] == expected[n] for n in names), "Archive member readback differs")
    require(
        checked["CURATED-INPUTS-MANIFEST.json"] == hashlib.sha256(raw).hexdigest(),
        "Inner manifest differs",
    )
    require(
        before
        == {n: ((source / n).stat().st_size, (source / n).stat().st_mtime_ns) for n in names},
        "Read-only source snapshot changed",
    )
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    partial.replace(archive)
    (out / "manifest.json").write_text(json.dumps(manifest) + "\n")
    receipt = {
        "schema": "lewm-curated-inputs-seal-v1",
        "archive_sha256": sha(archive),
        "archive_bytes": archive.stat().st_size,
        "members": len(checked),
        "every_member_readback_verified": True,
        "source_immutable_during_seal": True,
        "new_model_queries": 0,
        "new_simulator_actions": 0,
    }
    (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
