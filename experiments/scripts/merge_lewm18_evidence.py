"""Assemble completed per-task cohorts without pooling methods across physical hosts."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

from package_lewm18_evidence import ROOT, read, sha, write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", action="append", required=True, help="INSTANCE[-COHORT]=SYNC_DIRECTORY"
    )
    parser.add_argument(
        "--select",
        action="append",
        default=[],
        help="ENV=COHORT for a prospectively admitted repair",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lewm18-combined")
    parser.add_argument("--env", choices=("cube", "pusht"))
    args = parser.parse_args()
    sources = {}
    for item in args.source:
        host, path = item.split("=", 1)
        assert re.fullmatch(r"[0-9]+(?:-[a-z0-9-]+)?", host) and host not in sources
        sources[host] = Path(path)
    requested = dict(item.split("=", 1) for item in args.select)
    assert set(requested) <= {"cube", "pusht"}
    selected = {}
    for kind in (args.env,) if args.env else ("cube", "pusht"):
        candidates = [
            (host, path)
            for host, path in sources.items()
            if (path / f"campaign-complete-{kind}.json").exists()
            and (kind not in requested or requested[kind] == host)
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected exactly one completed {kind} cohort, found {len(candidates)}"
            )
        selected[kind] = candidates[0]
    target = args.output
    if target.exists():
        raise FileExistsError(target)
    (target / "attempts").mkdir(parents=True)
    hashes = {}
    for host, path in sources.items():
        links = [p for p in path.rglob("*") if p.is_symlink()]
        assert all(p == path / "STOP" for p in links), links
        shutil.copytree(path, target / "hosts" / host, symlinks=True)
        snapshot = target / "hosts" / host
        hashes[host] = {
            str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob("*")) if p.is_file()
        }
    stages, selections = {}, {}
    for kind, (host, path) in selected.items():
        complete = read(path / f"campaign-complete-{kind}.json")
        manifests = {"engineering": f"engineering-{kind}-freeze.json", **complete["stages"]}
        for stage, manifest_path in manifests.items():
            name = Path(manifest_path).name
            manifest = read(path / name)
            archive_path = path / "source" / f"{manifest['name']}.tar.gz"
            assert sha(archive_path) == manifest["source_archive_sha256"]
            with tarfile.open(archive_path) as archive:
                for source, expected in manifest["source_sha256"].items():
                    assert (
                        hashlib.sha256(archive.extractfile(source).read()).hexdigest() == expected
                    )
            assert all(job["env"] == kind for job in manifest["jobs"])
            for job in manifest["jobs"]:
                result = read(path / "attempts" / job["id"] / "result.json")
                assert result["job"] == job and not result["error"] and not result["censored"]
            stages[f"{kind}:{stage}"] = name
        for folder in sorted((path / "attempts").iterdir()):
            metadata = next(
                (
                    folder / n
                    for n in ("result.json", "failed-start.json", "started.json")
                    if (folder / n).exists()
                ),
                None,
            )
            if metadata is None:
                raise ValueError(f"Attempt has no identity: {folder.name}")
            if read(metadata)["job"]["env"] == kind:
                shutil.copytree(folder, target / "attempts" / folder.name)
        for file in path.glob("*.json"):
            if kind in file.name:
                shutil.copy2(file, target / file.name)
        selections[kind] = read(path / f"final-selection-{kind}.json")
    write(
        target / "campaign-complete.json",
        {
            "stages": stages,
            "hosts": {k: v[0] for k, v in selected.items()},
            "physical_instances": {k: v[0].split("-")[0] for k, v in selected.items()},
            "cohort_selection": requested,
            "note": "Cohort IDs distinguish the mechanical PushT repair on the same physical host.",
        },
    )
    write(target / "final-selection-by-environment.json", selections)
    write(target / "host-file-hashes.json", hashes)
    print(
        json.dumps(
            {"hosts_by_environment": {k: v[0] for k, v in selected.items()}, "output": str(target)}
        )
    )


if __name__ == "__main__":
    main()
