"""JSON-only, owner-bound analysis transfer AFTER final three-host accounting.

No experiment, array, checkpoint, model or image is opened. Imported evidence lives
outside /workspace/Safety-Dial and is never a collector input. Freeze/export/import
are disabled until a complete three-host accounting aggregate is supplied.
"""

import argparse
import datetime as dt
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PROJECT = Path("/workspace/Safety-Dial")
DESTINATION = Path("/workspace/lewm-analysis-inputs-20261031")
PLAN_SHA256 = "e06ca071e57cf30cddfbf7a367966aa621c021093f25ecb8a1ed681ff82fa6db"
PODS = {"vukovc3t4o1dui", "wq4ar2qx2ff6r4", "nj2fqkp2prgk8x"}
NAV = "vukovc3t4o1dui"
SCHEMA = "lewm-analysis-json-assembly-20261031-v1"
ARCHIVE_MANIFEST = "ANALYSIS-OWNERSHIP-MANIFEST.json"
MARKER = ".analysis-only-input-root.json"
MAX_FILE_BYTES = 64 * 1024 * 1024


def require(value, message):
    if not value:
        raise RuntimeError(message)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable_identity(path):
    info = path.stat()
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def load(path):
    return json.loads(path.read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def name(value):
    path = PurePosixPath(value)
    require(
        isinstance(value, str)
        and value
        and "\\" not in value
        and not path.is_absolute()
        and ".." not in path.parts
        and str(path) == value,
        "Unsafe noncanonical relative path",
    )
    return value


def normalize_evidence(value):
    """Match the frozen analyzer's absolute Safety-Dial evidence-path relocation."""
    path = PurePosixPath(value)
    if path.is_absolute():
        require("Safety-Dial" in path.parts, "Unrecognized absolute evidence path")
        value = str(PurePosixPath(*path.parts[path.parts.index("Safety-Dial") + 1 :]))
    return name(value)


def file_at(root, value):
    path = root / name(value)
    require(path.resolve().is_relative_to(root.resolve()), "Evidence escaped root")
    require(not path.is_symlink(), "Symlinks are not analysis inputs")
    info = path.stat()
    require(
        stat.S_ISREG(info.st_mode) and info.st_size <= MAX_FILE_BYTES,
        "Analysis input must be a bounded regular JSON file",
    )
    require(path.suffix == ".json", "Only JSON bytes may be transferred")
    return path


def ref(root, path):
    path = path.resolve()
    require(path.is_relative_to(root.resolve()), "Reference must stay inside source project")
    return {"path": name(str(path.relative_to(root))), "sha256": sha(path)}


def same_ref(root, reference):
    require(
        not Path(reference["path"]).is_absolute(),
        "Frozen guard references must be relative to the artifact root",
    )
    path = file_at(root, reference["path"])
    require(sha(path) == reference["sha256"], "Frozen JSON reference changed: " + str(path))
    return path


def rented(pod):
    require(
        pod in PODS and os.environ.get("RUNPOD_POD_ID") == pod,
        "This metadata transfer belongs on its actual declared rented host",
    )


def accounting(aggregate_path, reports):
    aggregate = load(aggregate_path)
    require(
        aggregate.get("complete_final_accounting") is True
        and aggregate.get("unresolved") == []
        and aggregate.get("schema", "").endswith("-aggregate"),
        "Final per-host accounting must be collected and merged BEFORE analysis transfers",
    )
    expected = {item["sha256"] for item in aggregate["source_reports"]}
    require(
        len(expected) == 3 and len(reports) == 3, "Exactly three authoritative reports required"
    )
    actual = {}
    for path in reports:
        report = load(path)
        pod = report["owning_pod_id"]
        require(
            pod in PODS
            and pod not in actual
            and report["complete_final_accounting"]
            and report["unresolved"] == [],
            "Every owning pod needs one complete final report",
        )
        actual[pod] = {"path": str(path), "sha256": sha(path)}
    require(
        set(actual) == PODS and {item["sha256"] for item in actual.values()} == expected,
        "Aggregate does not bind the exact three supplied final reports",
    )
    return {
        "aggregate_sha256": sha(aggregate_path),
        "host_reports": actual,
        "complete_final_accounting": True,
        "analysis_transfer_must_follow_this_aggregate": True,
    }


def namespace_owner(path, namespaces):
    matches = [
        entry
        for entry in namespaces
        if path == entry["output_namespace"] or path.startswith(entry["output_namespace"] + "/")
    ]
    require(len(matches) <= 1, "Overlapping evidence namespaces")
    return matches[0]["owning_pod_id"] if matches else None


def metadata_references(kind, value):
    """The actual v3 analyzer/guard JSON read graph, not every provenance breadcrumb."""
    fields = {
        "inputs": {"plan": "plan", "final_manifest": "manifest"},
        "manifest": {"joint_opening": "joint"},
        "joint": {
            "primary_opening": "opening",
            "secondary_opening": "opening",
            "secondary_activation": "plain",
            "secondary_relocation": "relocation",
        },
        "opening": {"secondary_relocation": "relocation"},
        "relocation": {"unstarted_proof": "plain", "original_secondary_manifest": "plain"},
        "protocol": {"selection_lock": "plain"},
        "plan": {
            "secondary_activation": "plain",
            "secondary_relocation": "relocation",
            "secondary_unstarted_proof": "plain",
        },
    }
    for field, target_kind in fields.get(kind, {}).items():
        if field in value:
            yield value[field], target_kind
    if kind == "inputs" and "joint_readiness" in value:
        yield value["joint_readiness"]["joint_opening"], "joint"
    if kind == "opening":
        for reference in value["selection_locks"]:
            yield reference, "plain"
    if kind == "plan":
        for profile in value.get("profiles", {}).values():
            yield profile["training_manifest"], "plain"


def freeze_request(args):
    rented(args.pod_id)
    require(args.pod_id == NAV, "Freeze the common assembly request on navigation")
    root = args.project_root.resolve()
    require(root == SOURCE_PROJECT, "Use only the canonical owning project as export source")
    require(not (root / MARKER).exists(), "An analysis-only tree is never an accounting source")
    account = accounting(args.accounting_merge, args.host_report)
    inputs, plan = load(args.inputs), load(args.plan)
    require(sha(args.plan) == PLAN_SHA256, "Require the exact frozen analysis-v3 plan bytes")
    require(
        plan["analysis_script_sha256"]
        == sha(ROOT / "experiments/scripts/analyze_lewm_navigation_campaign_20261031_v3.py")
        and set(plan["execution"]["allowed_pod_ids"]) == PODS,
        "Require the exact three-host frozen v3 analysis source/plan",
    )
    for source_name, expected in plan["source_sha256"].items():
        require(sha(ROOT / source_name) == expected, "Frozen analysis code dependency changed")
    require(
        inputs["schema"] == "lewm-navigation-analysis-inputs-20261031-v3"
        and inputs["plan"]["sha256"] == sha(args.plan)
        and inputs["all_final_output_namespaces_absent_at_binding"] is True,
        "Require the prospectively bound v3 inputs",
    )
    require(
        len(inputs["training_runs"]) == 24 and len(inputs["profiles"]) == 11,
        "This assembly preserves all24 training runs and all11 final profiles",
    )
    namespaces = []
    seeds = [(ref(root, args.inputs), "inputs"), (ref(root, args.plan), "plan")]
    for entry in inputs["training_runs"]:
        path = same_ref(root, entry["protocol"])
        protocol = load(path)
        namespaces.append(
            {
                "kind": "training",
                "method": entry["method"],
                "owning_pod_id": protocol["evaluation"]["execution_pod_id"],
                "output_namespace": name(entry["output_namespace"]),
                "protocol": entry["protocol"],
            }
        )
        seeds.append((entry["protocol"], "protocol"))
    for entry in inputs["profiles"]:
        protocol = load(same_ref(root, entry["protocol"]))
        require(
            entry["host"]["pod_id"] == protocol["evaluation"]["execution_pod_id"],
            "Profile physical owner differs",
        )
        namespaces.append(
            {
                "kind": "final_profile",
                "family": entry["family"],
                "owning_pod_id": entry["host"]["pod_id"],
                "output_namespace": name(protocol["output_namespace"]),
                "protocol": entry["protocol"],
            }
        )
        seeds.append((entry["protocol"], "protocol"))
    require(
        len({entry["output_namespace"] for entry in namespaces}) == 35
        and all(entry["owning_pod_id"] in PODS for entry in namespaces),
        "Require35 distinct owner-bound namespaces",
    )
    for entry in namespaces:
        namespace_owner(entry["output_namespace"], namespaces)
    metadata, deferred, pending, expanded = {}, {}, seeds, set()
    while pending:
        reference, kind = pending.pop()
        path_name = reference["path"]
        require(
            isinstance(reference["sha256"], str)
            and re.fullmatch("[0-9a-f]{64}", reference["sha256"]),
            "Invalid JSON SHA",
        )
        require(not Path(path_name).is_absolute(), "Metadata guards require relative JSON refs")
        name(path_name)
        if not path_name.endswith(".json"):
            continue  # No NPZ, images, checkpoint, binary or source imports.
        owner = namespace_owner(path_name, namespaces)
        collection = deferred if owner else metadata
        if path_name in collection:
            require(
                collection[path_name]["sha256"] == reference["sha256"],
                "Two frozen hashes claim the same metadata path",
            )
            if (path_name, kind) in expanded:
                continue
        collection[path_name] = {"sha256": reference["sha256"], "owning_pod_id": owner}
        if owner:
            continue  # Its physical owner exports this result, never navigation by proxy.
        path = same_ref(root, reference)
        expanded.add((path_name, kind))
        pending.extend(metadata_references(kind, load(path)))
    request = {
        "schema": SCHEMA,
        "purpose": "analysis_only_no_new_compute",
        "frozen_utc": utc(),
        "helper_sha256": sha(Path(__file__)),
        "source_project": str(SOURCE_PROJECT),
        "destination": str(DESTINATION),
        "allowed_pod_ids": sorted(PODS),
        "common_metadata_export_owner": NAV,
        "accounting": account,
        "inputs": ref(root, args.inputs),
        "plan": ref(root, args.plan),
        "namespaces": namespaces,
        "metadata_files": metadata,
        "deferred_owned_json_references": deferred,
        "no_npz_models_frames_checkpoints_or_raw_data": True,
        "never_collect_from_import_destination": True,
    }
    write_new(args.out, request)
    print(
        json.dumps(
            {
                "request": str(args.out),
                "sha256": sha(args.out),
                "namespaces": len(namespaces),
                "metadata_json_files": len(metadata),
            }
        )
    )


def load_request(path, expected):
    require(sha(path) == expected, "Prospective assembly request SHA changed")
    request = load(path)
    require(
        request["schema"] == SCHEMA and request["helper_sha256"] == sha(Path(__file__)),
        "Assembly helper/request source differs",
    )
    require(
        request["source_project"] == str(SOURCE_PROJECT)
        and request["destination"] == str(DESTINATION),
        "Assembly roots changed",
    )
    require(
        set(request["allowed_pod_ids"]) == PODS
        and request["accounting"]["complete_final_accounting"] is True,
        "Invalid three-host post-accounting request",
    )
    return request


def add_json(selected, root, path_name, owner, role):
    path = file_at(root, path_name)
    record = {
        "sha256": sha(path),
        "bytes": path.stat().st_size,
        "owning_pod_id": owner,
        "role": role,
    }
    if path_name in selected:
        require(selected[path_name] == record, "A JSON path has conflicting ownership")
    selected[path_name] = record


def select_namespace(root, entry, selected):
    namespace, owner = entry["output_namespace"], entry["owning_pod_id"]
    folder = root / namespace
    require(not folder.is_symlink() and folder.is_dir(), "Missing owned physical namespace")
    state = load(file_at(root, namespace + "/state.json"))
    freeze = load(file_at(root, namespace + "/source-freeze.json"))
    protocol = load(same_ref(root, entry["protocol"]))
    require(
        state["phase"] == "complete" and state["active"] is None,
        "Only complete retired training/final evidence may be exported",
    )
    if entry["kind"] == "training" and entry["method"] != "tuned_fixed":
        require(
            freeze["protocol"] == protocol
            and freeze["identity"] == state["identity"]
            and not state["unfinalized_job_ids"]
            and state["action_accounting_exact"],
            "Training source/accounting is unresolved",
        )
        jobs = state["completed_job_ids"]
    else:
        require(
            freeze["protocol_sha256"] == entry["protocol"]["sha256"] and freeze["pod_id"] == owner,
            "Actual physical execution owner/source differs",
        )
        jobs = list(state["jobs"])
    require(len(jobs) == len(set(jobs)), "Duplicate retained job")
    for path in (folder / "state.json", folder / "source-freeze.json"):
        add_json(selected, root, str(path.relative_to(root)), owner, "owned_physical_evidence")
    for pattern in ("candidate-*.json", "update-*.json", "generation-*.json", "nominees.json"):
        for path in sorted(folder.glob(pattern)):
            add_json(selected, root, str(path.relative_to(root)), owner, "owned_physical_evidence")
    for job in jobs:
        require(isinstance(job, str) and Path(job).name == job, "Unsafe job identity")
        prefix = namespace + "/jobs/" + job
        for suffix in ("started.json", "result.json"):
            add_json(selected, root, prefix + "/" + suffix, owner, "owned_physical_evidence")
        row = load(root / prefix / "result.json")
        inner = normalize_evidence(row["extra"]["episode_result"])
        require(inner == prefix + "/episode/result.json", "Inner result escapes its actual job")
        add_json(selected, root, inner, owner, "owned_physical_evidence")
        require(
            selected[inner]["sha256"] == row["extra"]["episode_result_sha256"],
            "Inner result hash differs from its outer row",
        )


def verify_archive(archive, receipt):
    require(sha(archive) == receipt["archive_sha256"], "Export archive hash changed")
    observed, manifest = {}, None
    with tarfile.open(archive, "r|*") as stream:
        for member in stream:
            name(member.name)
            require(
                member.isfile()
                and not member.issym()
                and not member.islnk()
                and member.name not in observed
                and member.size <= MAX_FILE_BYTES,
                "Unsafe or duplicate archive member",
            )
            raw = stream.extractfile(member).read()
            observed[member.name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
            require(len(raw) == member.size, "Truncated archive member")
            if member.name == ARCHIVE_MANIFEST:
                manifest = json.loads(raw)
            else:
                require(member.name.endswith(".json"), "Non-JSON analysis archive member")
    require(manifest is not None and manifest["schema"] == SCHEMA, "Missing ownership manifest")
    require(
        observed[ARCHIVE_MANIFEST]["sha256"] == receipt["manifest_sha256"],
        "Ownership manifest differs",
    )
    require(
        set(observed) == set(manifest["files"]) | {ARCHIVE_MANIFEST},
        "Archive and ownership manifest members differ",
    )
    for key, record in manifest["files"].items():
        require(
            observed[key] == {k: record[k] for k in ("sha256", "bytes")},
            "JSON member bytes/hash differ",
        )
    return manifest


def export(args):
    rented(args.pod_id)
    root = args.project_root.resolve()
    require(root == SOURCE_PROJECT, "Use only the canonical owning project as export source")
    require(not (root / MARKER).exists(), "Never export/recollect an imported analysis tree")
    request = load_request(args.request, args.request_sha256)
    require(
        sha(args.accounting_merge) == request["accounting"]["aggregate_sha256"]
        and load(args.accounting_merge)["complete_final_accounting"] is True,
        "Export must follow the exact complete accounting merge",
    )
    report = load(args.host_report)
    require(
        report["owning_pod_id"] == args.pod_id
        and report["complete_final_accounting"]
        and sha(args.host_report) == request["accounting"]["host_reports"][args.pod_id]["sha256"],
        "Export host differs from the exact final-accounting owner",
    )
    require(not args.out_dir.exists(), "Never overwrite an export namespace")
    selected = {}
    for entry in request["namespaces"]:
        if entry["owning_pod_id"] == args.pod_id:
            select_namespace(root, entry, selected)
    if args.pod_id == NAV:
        for path_name, record in request["metadata_files"].items():
            add_json(selected, root, path_name, None, "immutable_shared_metadata")
            require(selected[path_name]["sha256"] == record["sha256"], "Metadata source changed")
    for path_name, expected in request["deferred_owned_json_references"].items():
        if expected["owning_pod_id"] == args.pod_id:
            require(
                path_name in selected and selected[path_name]["sha256"] == expected["sha256"],
                "A frozen owner-specific JSON reference was omitted",
            )
    manifest = {
        "schema": SCHEMA,
        "analysis_only": True,
        "owning_pod_id": args.pod_id,
        "request_sha256": args.request_sha256,
        "created_utc": utc(),
        "accounting_aggregate_sha256": request["accounting"]["aggregate_sha256"],
        "files": selected,
        "new_model_simulator_calls": 0,
    }
    args.out_dir.mkdir(parents=True)
    archive = args.out_dir / ("analysis-json-" + args.pod_id + ".tar.gz")
    with tarfile.open(archive, "w:gz", compresslevel=3) as stream:
        for path_name, record in sorted(selected.items()):
            path = file_at(root, path_name)
            before = stable_identity(path)
            with path.open("rb") as source:
                raw = source.read()
            require(
                stable_identity(path) == before
                and hashlib.sha256(raw).hexdigest() == record["sha256"],
                "Owned JSON changed during export",
            )
            info = tarfile.TarInfo(path_name)
            info.size, info.mode = len(raw), 0o600
            stream.addfile(info, io.BytesIO(raw))
        raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        info = tarfile.TarInfo(ARCHIVE_MANIFEST)
        info.size, info.mode = len(raw), 0o600
        stream.addfile(info, io.BytesIO(raw))
    receipt = {
        "schema": SCHEMA,
        "owning_pod_id": args.pod_id,
        "request_sha256": args.request_sha256,
        "archive_sha256": sha(archive),
        "archive": str(archive),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "new_model_simulator_calls": 0,
    }
    verify_archive(archive, receipt)
    for path_name, record in selected.items():
        require(sha(file_at(root, path_name)) == record["sha256"], "Source changed after export")
    write_new(args.out_dir / "ownership-manifest.json", manifest)
    write_new(args.out_dir / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2))


def import_archive(args):
    rented(args.pod_id)
    require(args.pod_id == NAV, "Assemble the analysis-only sibling tree on navigation")
    request = load_request(args.request, args.request_sha256)
    destination = args.destination.resolve()
    require(
        str(destination) == request["destination"]
        and not destination.is_relative_to(Path(request["source_project"])),
        "Analysis imports must stay outside canonical project/results",
    )
    receipt = load(args.receipt)
    require(receipt["request_sha256"] == args.request_sha256, "Export used a different request")
    manifest = verify_archive(args.archive, receipt)
    owner = manifest["owning_pod_id"]
    require(
        owner in PODS
        and owner == receipt["owning_pod_id"]
        and manifest["request_sha256"] == args.request_sha256
        and manifest["accounting_aggregate_sha256"] == request["accounting"]["aggregate_sha256"],
        "Export ownership/accounting differs",
    )
    for path_name, record in manifest["files"].items():
        physical_owner = namespace_owner(path_name, request["namespaces"])
        if physical_owner:
            require(
                physical_owner == owner == record["owning_pod_id"]
                and record["role"] == "owned_physical_evidence",
                "Foreign physical ownership",
            )
        else:
            require(
                owner == NAV
                and record["owning_pod_id"] is None
                and record["role"] == "immutable_shared_metadata"
                and request["metadata_files"][path_name]["sha256"] == record["sha256"],
                "Unrecognized shared metadata ownership",
            )
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".assembly.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = destination / MARKER
        if marker.exists():
            require(
                load(marker)["request_sha256"] == args.request_sha256, "Assembly request changed"
            )
        else:
            require(
                set(destination.iterdir()) == {destination / ".assembly.lock"},
                "First import requires a fresh analysis-only root",
            )
            write_new(
                marker,
                {
                    "schema": SCHEMA,
                    "request_sha256": args.request_sha256,
                    "analysis_only": True,
                    "never_run_collector_here": True,
                    "new_model_simulator_calls": 0,
                },
            )
        ownership_path = destination / ".ownership" / (owner + ".json")
        require(not ownership_path.exists(), "Never import the same owner twice")
        require(
            all(not (destination / path_name).exists() for path_name in manifest["files"]),
            "No existing analysis member may be overwritten",
        )
        with tarfile.open(args.archive, "r|*") as stream:
            for member in stream:
                if member.name == ARCHIVE_MANIFEST:
                    continue
                path = destination / name(member.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                require(path.parent.resolve().is_relative_to(destination), "Import parent escaped")
                with path.open("xb") as output:
                    source = stream.extractfile(member)
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
                require(
                    sha(path) == manifest["files"][member.name]["sha256"], "Imported hash changed"
                )
        write_new(
            ownership_path,
            {
                "receipt": receipt,
                "manifest": manifest,
                "imported_utc": utc(),
                "analysis_only": True,
            },
        )
    print(
        json.dumps(
            {
                "destination": str(destination),
                "imported_owner": owner,
                "files": len(manifest["files"]),
                "new_model_simulator_calls": 0,
            }
        )
    )


def validate(args):
    rented(args.pod_id)
    request = load_request(args.request, args.request_sha256)
    root = args.destination.resolve()
    require(
        str(root) == request["destination"] and load(root / MARKER)["analysis_only"],
        "Wrong analysis-only root",
    )
    require(
        {path.stem for path in (root / ".ownership").glob("*.json")} == PODS,
        "All three owner exports must be imported before analysis",
    )
    verified = 0
    for path in (root / ".ownership").glob("*.json"):
        record = load(path)
        for path_name, expected in record["manifest"]["files"].items():
            require(sha(file_at(root, path_name)) == expected["sha256"], "Retained JSON changed")
            verified += 1
    for path_name, expected in request["metadata_files"].items():
        require(sha(file_at(root, path_name)) == expected["sha256"], "Metadata closure incomplete")
    for entry in request["namespaces"]:
        # Rebuild exactly the file selection against the assembled owner-specific tree.
        selected = {}
        select_namespace(root, entry, selected)
    report = {
        "schema": SCHEMA,
        "complete": True,
        "request_sha256": args.request_sha256,
        "json_files_verified": verified,
        "namespace_count": len(request["namespaces"]),
        "artifact_root": str(root),
        "analysis_uses_canonical_frozen_code_ROOT": True,
        "never_run_collector_here": True,
        "new_model_simulator_calls": 0,
    }
    write_new(args.out, report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze-request")
    freeze.add_argument("--project-root", type=Path, default=ROOT)
    freeze.add_argument("--pod-id", required=True)
    freeze.add_argument("--inputs", type=Path, required=True)
    freeze.add_argument("--plan", type=Path, required=True)
    freeze.add_argument("--accounting-merge", type=Path, required=True)
    freeze.add_argument("--host-report", type=Path, action="append", required=True)
    freeze.add_argument("--out", type=Path, required=True)
    for command in ("export", "import", "validate"):
        child = commands.add_parser(command)
        child.add_argument("--pod-id", required=True)
        child.add_argument("--request", type=Path, required=True)
        child.add_argument("--request-sha256", required=True)
        if command == "export":
            child.add_argument("--project-root", type=Path, default=ROOT)
            child.add_argument("--accounting-merge", type=Path, required=True)
            child.add_argument("--host-report", type=Path, required=True)
            child.add_argument("--out-dir", type=Path, required=True)
        else:
            child.add_argument(
                "--destination", type=Path, default=Path("/workspace/lewm-analysis-inputs-20261031")
            )
        if command == "import":
            child.add_argument("--archive", type=Path, required=True)
            child.add_argument("--receipt", type=Path, required=True)
        if command == "validate":
            child.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    operations = {
        "freeze-request": freeze_request,
        "export": export,
        "import": import_archive,
        "validate": validate,
    }
    operations[args.command](args)


if __name__ == "__main__":
    main()
