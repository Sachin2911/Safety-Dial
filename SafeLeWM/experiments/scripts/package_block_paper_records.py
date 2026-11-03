"""Preserve audited Block records for paper integration, excluding camera/model data."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import time


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    repo = Path("/workspace/Safety-Dial")
    assert repo.is_dir(), "Run on the existing rented study host"
    audit = args.audit.resolve()
    result = json.loads((audit / "audit.json").read_text())
    assert result["passed"] and result["repeat"] == args.repeat
    run = repo / f"outputs/search2-20261102-r{args.repeat}"
    assert json.loads((run / "status.json").read_text())["phase"] == "complete"
    archive = audit.with_name(audit.name + "-records.tar.gz")
    assert not archive.exists()
    files = {}
    originals = json.loads((audit / "manifest.json").read_text())
    sources = {}
    for relative, entry in originals.items():
        path = repo / relative
        assert path.stat().st_size == entry["bytes"] and sha(path) == entry["sha256"]
        files["records/" + str(path.relative_to(run))] = path
        if path.name == "summary.json":
            summary = json.loads(path.read_text())
            for source, expected in summary["implementation_hashes"].items():
                snapshot = path.parent / "implementation" / source
                if expected not in sources:
                    assert sha(snapshot) == expected
                    sources[expected] = dict(bytes=snapshot.stat().st_size, original_paths=[])
                    files["sources/" + expected] = snapshot
                if source not in sources[expected]["original_paths"]:
                    sources[expected]["original_paths"].append(source)
    for pattern in ("*.json", "genotypes/*.json", "cma/*.json", "random/*.json",
                    "select/*.json", "smoke/*.json", "*/ledger.jsonl"):
        for path in sorted(run.glob(pattern)):
            files["metadata/" + str(path.relative_to(run))] = path
    config = repo / f"SafeBlock/configs/search2-20261102-r{args.repeat}.json"
    assert sha(config) == result["config_sha256"]
    files["audit/config.json"] = config
    files["audit/audit.json"] = audit / "audit.json"
    files["audit/record-manifest.json"] = audit / "manifest.json"
    files["audit/source-v1.py"] = audit.parent / "source-v1.py"
    assert sha(files["audit/source-v1.py"]) == result["auditor_sha256"]
    files["audit/package-source.py"] = Path(__file__).resolve()
    source_index = audit / "source-snapshot-index.json"
    assert not source_index.exists()
    source_index.write_text(json.dumps(sources, indent=2) + "\n")
    files["audit/source-snapshot-index.json"] = source_index
    manifest = {
        name: dict(bytes=path.stat().st_size, sha256=sha(path))
        for name, path in files.items()
    }
    scope = (
        "All audited step/summary/accounting/start records, selection/training metadata and "
        "deduplicated source copies. Excludes camera bytes, models, datasets, binary physical "
        "snapshots and discarded launches. This is not a full raw-study backup."
    )
    payload = json.dumps(dict(scope=scope, files=manifest), indent=2).encode() + b"\n"
    with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
        for name, path in files.items():
            tar.add(path, arcname=name, recursive=False)
        info = tarfile.TarInfo("manifest.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    receipt = dict(
        completed=time.time(), archive=str(archive), bytes=archive.stat().st_size,
        sha256=sha(archive), files=len(files), scope=scope,
    )
    (audit / "packaging.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
