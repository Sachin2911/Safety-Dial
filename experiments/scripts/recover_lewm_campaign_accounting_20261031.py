"""Reconstruct old Recovery accounting on the explicitly authorized replacement pod.

Only preserved JSON/journal ledgers are read. The source owner is the old pod;
the collection execution host is the new pod, and both remain explicit. No source
archive, extracted original record, model, image array or simulator is modified.
"""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import traceback

ROOT = Path(__file__).resolve().parents[2]
SOURCE_POD = "nj2fqkp2prgk8x"
COLLECTOR = "experiments/scripts/collect_lewm_campaign_20261031_v4.py"
COLLECTOR_SHA = "d9d5da4893b4eb483bd072766fa6e428865bb43ea988e98d760f2d4845a7a7fb"
PRESERVER_SHA = "d8f393d9adea59e721e7ce2a63a66654a82f99bd4fc7c6db8318af424d482568"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def bound_original(origin, manifest, name, expected):
    relative = Path(name)
    require(
        not relative.is_absolute() and ".." not in relative.parts and "\\" not in name,
        "Unsafe original ledger path",
    )
    path = origin / relative
    require(path.resolve().is_relative_to(origin), "Original ledger escaped its archive tree")
    require(stat.S_ISREG(path.lstat().st_mode), "Original ledger must be a regular file")
    record = manifest["files"].get(name)
    require(
        record is not None
        and record["sha256"] == expected == digest(path)
        and record["size"] == path.stat().st_size,
        "Reconstructed accounting input differs from the verified original: " + name,
    )


def reconstruct(args):
    actual = os.environ.get("RUNPOD_POD_ID")
    authority_path = args.authority.resolve()
    require(authority_path.is_relative_to(ROOT), "Authority must be in the canonical project")
    require(digest(authority_path) == args.authority_sha256, "Relocation authority differs")
    authority = load(authority_path)
    require(
        authority["schema"] == "lewm-frozen-final-relocation-authority-v1"
        and authority["execution_enabled"] is True
        and authority["actual_final_host"]["pod_id"] == actual == args.execution_pod_id
        and actual != SOURCE_POD
        and authority["total_campaign_cap_usd"] == 13
        and authority["maximum_duration_seconds"] == 5400,
        "Accounting must execute on the one explicitly authorized replacement pod",
    )
    require(
        dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(authority["new_deadline_utc"]),
        "The authorized rented analysis window has expired",
    )
    origin = args.origin_root.resolve()
    output = args.out.resolve()
    require(
        origin.is_dir()
        and not origin.is_relative_to(ROOT)
        and not ROOT.is_relative_to(origin)
        and not output.is_relative_to(origin)
        and not output.exists(),
        "Keep the immutable original outside canonical/output trees and use fresh output",
    )
    receipt = load(args.archive_receipt)
    manifest_path = origin / "MANIFEST-20261031.json"
    manifest = load(manifest_path)
    require(
        manifest["schema"] == "lewm-campaign-preservation-20261031-v1"
        and manifest["pod_id"] == SOURCE_POD
        and manifest["source_builder_sha256"] == PRESERVER_SHA
        and receipt["source_builder_sha256"] == PRESERVER_SHA
        and receipt["every_member_streamed_and_hashed"] is True
        and receipt["manifest_sha256"] == digest(manifest_path)
        and receipt["archive_sha256"] == args.archive_sha256 == digest(args.archive)
        and receipt["archive_bytes"] == args.archive.stat().st_size
        and receipt["source_files"] == len(manifest["files"])
        and receipt["final_quiescence"]["all_declared_workers_retired"] is True,
        "Original Recovery archive/all-member/retirement proof differs",
    )
    collector_path = ROOT / COLLECTOR
    require(digest(collector_path) == COLLECTOR_SHA, "Qualified collector-v4 source differs")
    journal_name = "experiments/helpers/lewm_progress_journal.py"
    bound_original(origin, manifest, journal_name, digest(ROOT / journal_name))
    spec = importlib.util.spec_from_file_location(
        "recovered_accounting_collector_v4", collector_path
    )
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    report = collector.collect(
        argparse.Namespace(
            project_root=origin,
            pod_id=SOURCE_POD,
            transfer_manifest=[],
            preparation_ledger=[],
            zero_work_proofs=None,
            final=True,
        )
    )
    for record in [*report["components"], *report["verified_copied_preparation_references"]]:
        source = record["source"]
        bound_original(origin, manifest, source["path"], source["sha256"])
        if source.get("source_freeze_path"):
            bound_original(
                origin, manifest, source["source_freeze_path"], source["source_freeze_sha256"]
            )
    for worker in report["worker_attempts"]:
        bound_original(origin, manifest, worker["record"], worker["sha256"])
        worker["identity_probe_execution_pod_id"] = actual
        worker["original_worker_retirement_verified_by_archive_preserver"] = True
        worker["original_pod_live_probe_performed"] = False
    report.update(
        collection_execution_pod_id=actual,
        recorded_source_owner_pod_id=SOURCE_POD,
        numeric_collection_executed_only_on_declared_rented_host=False,
        numeric_collection_executed_on_explicit_authorized_replacement=True,
        original_source_pod_resumed_by_this_adapter=False,
        original_source_pod_used_for_reconstruction_execution=False,
        reconstruction_scope=(
            "Saved original Recovery ledgers are reconstructed on the explicit replacement pod. "
            "Old worker retirement comes from the verified archive preserver; local /proc "
            "observations belong to the analysis host and are not original-host live probes. "
            "No experiment, model or simulator executes during this reconstruction."
        ),
        reconstruction_source={
            "adapter": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": digest(__file__)},
            "collector": {"path": COLLECTOR, "sha256": COLLECTOR_SHA},
            "authority": {
                "path": str(authority_path.relative_to(ROOT)),
                "sha256": args.authority_sha256,
            },
            "original_archive": {
                "path": str(args.archive.resolve()),
                "sha256": args.archive_sha256,
                "bytes": args.archive.stat().st_size,
            },
            "original_manifest": {
                "path": str(manifest_path),
                "sha256": digest(manifest_path),
            },
            "original_receipt": {
                "path": str(args.archive_receipt.resolve()),
                "sha256": digest(args.archive_receipt),
            },
            "original_all_member_verification_receipt_bound": True,
            "all_consumed_ledger_worker_and_freeze_hashes_match_original_manifest": True,
        },
        original_archive_and_extracted_records_unchanged=True,
        new_model_simulator_calls=0,
    )
    require(digest(manifest_path) == receipt["manifest_sha256"], "Original manifest changed")
    write_new(output, report)
    return {
        "output": str(output),
        "sha256": digest(output),
        "collection_execution_pod_id": actual,
        "recorded_source_owner_pod_id": SOURCE_POD,
        "complete_final_accounting": report["complete_final_accounting"],
        "unresolved_components": len(report["unresolved"]),
        "totals": report["totals"],
        "new_model_simulator_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-pod-id", required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--origin-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--archive-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = reconstruct(args)
    except BaseException:
        failure = args.out.with_suffix(".failure.json")
        if not failure.exists():
            write_new(
                failure,
                {
                    "adapter_source_sha256": digest(__file__),
                    "execution_pod_id": os.environ.get("RUNPOD_POD_ID"),
                    "recorded_source_owner_pod_id": SOURCE_POD,
                    "error": traceback.format_exc(),
                    "new_model_simulator_calls": 0,
                },
            )
        raise
    print(json.dumps(report))


if __name__ == "__main__":
    main()
