"""Exact-byte metadata/software distribution, with all live scientific pins immutable.

Reads only source/config/worker metadata. Never imports experiment code or reads
final-case data, arrays, models, environment account configs or secrets.
"""

import argparse
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile


ROOT = Path(__file__).resolve().parents[2]
PODS = {"vukovc3t4o1dui", "wq4ar2qx2ff6r4", "nj2fqkp2prgk8x"}
PLANS = (
    "configs/lewm_campaign_20261031/navigation-joint-final-design-v2.json",
    "configs/lewm_campaign_20261031/navigation-scientific-media-plan-v2.json",
    "configs/lewm_campaign_20261031/navigation-final-analysis-plan-v3.json",
    "configs/lewm_campaign_20261031/navigation-final-cases-template-v1.json",
    "configs/lewm_campaign_20261031/navigation-decoded-target-pilot-v3.json",
)
EXTRA = (
    "experiments/scripts/freeze_lewm_navigation_selection.py",
    "experiments/scripts/freeze_lewm_navigation_selection_opening_v2.py",
    "experiments/scripts/freeze_lewm_navigation_final_comparison.py",
    "experiments/scripts/assemble_lewm_navigation_analysis_inputs.py",
    "experiments/scripts/stage_lewm_navigation_media_inputs.py",
    "experiments/scripts/audit_lewm_campaign_evidence_v6.py",
    "experiments/scripts/collect_lewm_campaign_20261031.py",
    "experiments/scripts/preserve_lewm_campaign_20261031.py",
    "experiments/scripts/launch_lewm_campaign_frozen.py",
    "experiments/scripts/launch_lewm_campaign_frozen_overlap.py",
)
# Existing old version paths and frozen protocol JSON may never be replaced.
DRAFT_REPLACE = {
    "experiments/helpers/lewm_navigation_joint_final_v2.py",
    "experiments/scripts/freeze_lewm_navigation_selection_opening_v2.py",
    "experiments/scripts/freeze_lewm_navigation_joint_final_v2.py",
    "experiments/scripts/prepare_lewm_navigation_joint_final_cases_v2.py",
    "experiments/scripts/analyze_lewm_navigation_campaign_20261031_v3.py",
    "experiments/scripts/render_lewm_navigation_scientific_media_v2.py",
    "experiments/scripts/assemble_lewm_navigation_analysis_inputs.py",
    "experiments/scripts/stage_lewm_navigation_media_inputs.py",
    "experiments/scripts/audit_lewm_campaign_evidence_v6.py",
}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def safe(value):
    path = PurePosixPath(value)
    require(
        value
        and "\\" not in value
        and not path.is_absolute()
        and ".." not in path.parts
        and str(path) == value,
        "Unsafe distribution path",
    )
    require(
        value.startswith(("experiments/", "configs/lewm_campaign_20261031/")),
        "Distribution is restricted to software and campaign metadata",
    )
    return value


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def bundle(args):
    require(not args.out_dir.exists(), "Use a fresh software bundle namespace")
    expected = {}

    def add(name, digest=None):
        safe(name)
        path = ROOT / name
        require(path.is_file() and not path.is_symlink(), "Missing regular local source: " + name)
        actual = sha(path)
        require(digest is None or digest == actual, "Local source differs from its freeze: " + name)
        require(name not in expected or expected[name] == actual, "Conflicting frozen SHA")
        expected[name] = actual

    def maps(value):
        if isinstance(value, dict):
            for field in ("source_sha256",):
                mapping = value.get(field, {})
                if not isinstance(mapping, dict):
                    continue
                for name, digest in mapping.items():
                    if "/" in name:
                        add(name, digest)
            if "analysis_script" in value:
                add(value["analysis_script"], value["analysis_script_sha256"])
            for child in value.values():
                maps(child)
        elif isinstance(value, list):
            for child in value:
                maps(child)

    for name in PLANS:
        add(name)
        maps(load(ROOT / name))
    for name in EXTRA:
        add(name)
    args.out_dir.mkdir(parents=True)
    manifest = {
        "schema": "lewm-final-software-distribution-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "installer_sha256": sha(Path(__file__)),
        "files": {
            name: {
                "sha256": digest,
                "bytes": (ROOT / name).stat().st_size,
                "replace_unpinned_metadata_draft": name in DRAFT_REPLACE,
            }
            for name, digest in sorted(expected.items())
        },
        "no_tests_workers_or_final_data": True,
    }
    archive = args.out_dir / "software.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=3) as stream:
        for name, record in manifest["files"].items():
            raw = (ROOT / name).read_bytes()
            require(hashlib.sha256(raw).hexdigest() == record["sha256"], "Local source changed")
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(raw), 0o600
            stream.addfile(info, io.BytesIO(raw))
        raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        info = tarfile.TarInfo("DISTRIBUTION.json")
        info.size, info.mode = len(raw), 0o600
        stream.addfile(info, io.BytesIO(raw))
    write_new(args.out_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "archive": str(archive),
                "sha256": sha(archive),
                "files": len(expected),
                "installer_sha256": sha(Path(__file__)),
            },
            indent=2,
        )
    )


def owner(pid):
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text()
        fields = raw[raw.rfind(")") + 2 :].split()
        return {"pid": int(pid), "start_ticks": fields[19], "state": fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def active_sources(root):
    active, pins = [], {}
    for path in sorted((root / "environment/workers").glob("*.json")):
        worker = load(path)
        current = owner(worker["pid"])
        if (
            not current
            or current["state"] == "Z"
            or str(worker["start_ticks"]) != current["start_ticks"]
        ):
            continue
        require(isinstance(worker.get("protocol"), str), "A live worker lacks a source protocol")
        protocol_path = root / worker["protocol"]
        protocol = load(protocol_path)
        expected = worker.get("expected_protocol_sha256", worker.get("protocol_sha256"))
        require(expected is None or sha(protocol_path) == expected, "Active protocol bytes changed")
        sources = {}
        for declaration in (protocol, protocol.get("evaluation", {})):
            for name, digest in declaration.get("source_sha256", {}).items():
                safe(name)
                require(name not in sources or sources[name] == digest, "Conflicting active pins")
                sources[name] = digest
                require(sha(root / name) == digest, "An active scientific source already differs")
                require(
                    name not in pins or pins[name] == digest, "Active stages disagree on source"
                )
                pins[name] = digest
        active.append(
            {
                "record": str(path.relative_to(root)),
                "record_sha256": sha(path),
                "identity": current,
                "protocol": worker["protocol"],
                "protocol_sha256": sha(protocol_path),
                "source_sha256": sources,
            }
        )
    return active, pins


def install(args):
    require(
        args.pod_id in PODS and os.environ.get("RUNPOD_POD_ID") == args.pod_id,
        "Distribution belongs on the actual declared rented host",
    )
    root = Path("/workspace/Safety-Dial")
    require(not args.receipt.exists(), "Never overwrite a distribution receipt")
    require(sha(args.archive) == args.archive_sha256, "Distribution archive SHA differs")
    payload, manifest = {}, None
    with tarfile.open(args.archive, "r|*") as stream:
        for member in stream:
            require(
                member.isfile()
                and not member.issym()
                and not member.islnk()
                and member.name not in payload,
                "Unsafe distribution archive",
            )
            raw = stream.extractfile(member).read()
            if member.name == "DISTRIBUTION.json":
                require(manifest is None, "Duplicate distribution manifest")
                manifest = json.loads(raw)
            else:
                payload[safe(member.name)] = raw
    require(
        manifest
        and manifest["installer_sha256"] == sha(Path(__file__))
        and set(payload) == set(manifest["files"]),
        "Wrong exact installer or member set",
    )
    for name, raw in payload.items():
        require(
            hashlib.sha256(raw).hexdigest() == manifest["files"][name]["sha256"],
            "Source payload SHA differs",
        )
    before, pins = active_sources(root)
    records, blockers = {}, []
    for name, record in manifest["files"].items():
        path = root / name
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), "Unsafe target")
        exists = path.exists()
        actual = sha(path) if exists else None
        item = {
            "expected_sha256": record["sha256"],
            "before_sha256": actual,
            "active_scientific_pin": pins.get(name),
        }
        if actual == record["sha256"]:
            item["action"] = "already_exact_no_write"
        elif name in pins:
            item["action"] = "blocked_active_scientific_source_immutable"
            blockers.append(name)
        elif exists and (not record["replace_unpinned_metadata_draft"] or path.suffix != ".py"):
            item["action"] = "blocked_frozen_or_past_version_immutable"
            blockers.append(name)
        else:
            if exists:
                backup = root / "environment/final-source-distribution-v1-preserved" / name
                backup.parent.mkdir(parents=True, exist_ok=True)
                with backup.open("xb") as stream:
                    stream.write(path.read_bytes())
                require(sha(backup) == actual, "Original metadata draft backup differs")
                item.update(
                    action="preserved_unpinned_draft_then_atomic_exact_update",
                    preserved_original=str(backup.relative_to(root)),
                    preserved_original_sha256=actual,
                    absence_from_active_scientific_source_pins_verified=True,
                )
            else:
                item["action"] = "missing_exact_bytes_staged_atomically"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".distribution-v1.part")
            with temporary.open("xb") as stream:
                stream.write(payload[name])
                stream.flush()
                os.fsync(stream.fileno())
            require(sha(temporary) == record["sha256"], "Staged source SHA differs")
            os.replace(temporary, path)
        item["after_sha256"] = sha(path) if path.exists() else None
        records[name] = item
    after, after_pins = active_sources(root)
    for name, digest in pins.items():
        require(sha(root / name) == digest, "Initial active scientific source changed")
    for name, digest in after_pins.items():
        require(sha(root / name) == digest, "Current active scientific source changed")
    receipt = {
        "schema": "lewm-final-software-distribution-receipt-v1",
        "pod_id": args.pod_id,
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "archive_sha256": args.archive_sha256,
        "installer_sha256": sha(Path(__file__)),
        "active_workers_before": before,
        "active_workers_after": after,
        "all_active_scientific_sources_unchanged": True,
        "files": records,
        "blockers": blockers,
        "complete_exact": not blockers,
        "workers_guards_tests_final_data_touched": False,
        "new_model_simulator_calls": 0,
    }
    write_new(args.receipt, receipt)
    print(
        json.dumps(
            {
                "receipt": str(args.receipt),
                "sha256": sha(args.receipt),
                "complete_exact": not blockers,
                "files": len(records),
                "changed_paths": [
                    name
                    for name, row in records.items()
                    if row["action"].startswith(("missing_", "preserved_"))
                ],
                "blockers": blockers,
                "new_model_simulator_calls": 0,
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("bundle")
    make.add_argument("--out-dir", type=Path, required=True)
    apply = commands.add_parser("install")
    apply.add_argument("--pod-id", required=True)
    apply.add_argument("--archive", type=Path, required=True)
    apply.add_argument("--archive-sha256", required=True)
    apply.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    (bundle if args.command == "bundle" else install)(args)


if __name__ == "__main__":
    main()
