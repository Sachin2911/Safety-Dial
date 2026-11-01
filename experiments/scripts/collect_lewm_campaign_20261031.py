"""Rented-host JSON-only accounting for the current20261031 campaign.

Physical episodes are result.json files adjacent to trace.npz. Wrapper/state rows
are evidence references, never additional episodes. No model/array/simulator is
loaded. Missing ledgers remain unresolved, never guessed as zero.
"""

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm_progress_journal import replay_progress_journal

SCHEMA = "lewm-campaign-accounting-20261031-v1"
PREPARATION_PREFIXES = (
    "navigation-memory-",
    "navigation-cases-",
    "navigation-readout-",
    "navigation-stress",
    "navigation-final-cases-",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(path.read_text())


def software(path):
    return any(
        part.startswith(("test_", "pytest", ".pytest")) or "-tests-" in part for part in path.parts
    )


def identity_live(pid, start_ticks):
    path = Path("/proc") / str(pid) / "stat"
    try:
        text = path.read_text()
    except FileNotFoundError:
        return False
    fields = text[text.rfind(")") + 2 :].split()
    return fields[0] != "Z" and fields[19] == str(start_ticks)


def counters(row):
    require(isinstance(row, dict), "A resource ledger must be a dictionary")
    result = {}
    for key, value in row.items():
        require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0,
            "Invalid cumulative resource counter: " + key,
        )
        result[str(key)] = value
    return result


def owner_from(value):
    if not isinstance(value, dict):
        return None
    for key in ("pod_id", "execution_pod_id"):
        if value.get(key):
            return value[key]
    for key in ("evaluation", "protocol"):
        found = owner_from(value.get(key))
        if found:
            return found
    return None


def nearest_freeze(path, root):
    for folder in (path.parent, *path.parents):
        if not folder.is_relative_to(root):
            break
        candidate = folder / "source-freeze.json"
        if candidate.is_file() and not candidate.is_symlink():
            return candidate, load(candidate)
    return None, None


def component_id(pod, kind, key):
    return hashlib.sha256(f"{pod}/{kind}/{key}".encode()).hexdigest()


def normalized(resource):
    """Aliases choose one measure. A positive legacy measure never implies attempts."""

    def first(*names, default=0):
        return next((resource[name] for name in names if name in resource), default)

    frames = first("actual_encoder_frames_attempted", "encoder_attempted_frames", "encoder_frames")
    batches = first("model_batch_calls")
    safety_rows = first("actual_safety_transition_rows_attempted", "learned_transition_rows")
    return {
        "environment_actions": resource.get("environment_actions", 0),
        "environment_resets": resource.get("environment_resets", 0),
        "encoder_frames_recorded": frames,
        "encoder_call_attempts_explicit": first(
            "actual_encoder_call_attempts",
            "encoder_attempted_batches",
            default=None if frames else 0,
        ),
        "cost_call_attempts_explicit": first(
            "actual_cost_call_attempts", default=None if batches else 0
        ),
        "legacy_model_batch_calls_recorded": batches,
        "predictor_block_steps_recorded": first(
            "actual_candidate_block_steps_attempted", "predictor_steps"
        ),
        "safety_call_attempts_explicit": first(
            "actual_safety_call_attempts", default=None if safety_rows else 0
        ),
        "safety_transition_rows_recorded": safety_rows,
        "goal_readout_call_attempts": first("goal_readout_call_attempts", "goal_readout_queries"),
        "public_download_bytes": resource.get("public_download_bytes", 0),
        "extracted_bytes": resource.get("extracted_bytes", 0),
        "probe_compressed_bytes_read": resource.get("probe_compressed_bytes_read", 0),
    }


def summarize(components):
    raw, mapped = collections.Counter(), collections.Counter()
    missing = collections.Counter()
    for record in components:
        raw.update(record["resources"])
        for key, value in normalized(record["resources"]).items():
            if value is None:
                missing[key] += 1
            else:
                mapped[key] += value
    return {
        "raw_counter_totals_by_field": dict(raw),
        "normalized_recorded_totals": dict(mapped),
        "components_without_explicit_measure": dict(missing),
        "do_not_sum_alias_fields_into_one_nn_total": True,
        "interpretation": "Sparse keys mean zero only inside an existing valid counter ledger. Explicit attempted-query fields and legacy successful/planner fields remain distinguishable; missing ledgers/unknown failed work are unresolved.",
    }


def collect(args):
    root = args.project_root.resolve()
    results = root / "results"
    require(results.is_dir(), "Current results tree absent")
    pod = args.pod_id
    default_transfer = root / "environment/navigation-transfer-v1.json"
    transfer_paths = ([default_transfer] if default_transfer.exists() else []) + list(
        args.transfer_manifest
    )
    transfer_records, copied_hashes, transfer_proofs = [], {}, {}
    for transfer_path in dict.fromkeys(path.resolve() for path in transfer_paths):
        require(
            transfer_path.is_relative_to(root),
            "Transfer manifest must be a retained project record",
        )
        transfer = load(transfer_path)
        require(
            transfer.get("verified") is True and isinstance(transfer.get("sha256"), dict),
            "Verified exact-hash transfer manifest required",
        )
        proof = {"path": str(transfer_path.relative_to(root)), "sha256": digest(transfer_path)}
        transfer_records.append((transfer_path, transfer))
        for name, sha in transfer["sha256"].items():
            require(
                name not in copied_hashes or copied_hashes[name] == sha,
                "A transferred path has multiple versions; preserve immutable data namespaces",
            )
            copied_hashes[name] = sha
            transfer_proofs.setdefault(name, []).append(proof)
    components, copied, unresolved, covered_jobs = [], [], [], set()
    used_ledgers = set()

    def add(path, kind, resource, *, extra=None):
        resource = counters(resource)
        relative = str(path.relative_to(root))
        sha = digest(path)
        freeze_path, freeze = nearest_freeze(path, root)
        recorded_owner = owner_from(freeze)
        source = {"path": relative, "sha256": sha}
        if freeze_path:
            source.update(
                source_freeze_path=str(freeze_path.relative_to(root)),
                source_freeze_sha256=digest(freeze_path),
                source_owner_pod_id=recorded_owner,
            )
        transferred = copied_hashes.get(relative) == sha
        if transferred:
            valid_origin = (
                freeze_path is not None
                and copied_hashes.get(str(freeze_path.relative_to(root))) == digest(freeze_path)
                and recorded_owner
                and recorded_owner != pod
            )
            if not valid_origin:
                unresolved.append({"kind": "copied_origin_not_hash_bound", "source": source})
                return
            copied.append(
                {
                    "kind": kind,
                    "source": source,
                    "owning_pod_id": recorded_owner,
                    "current_host_compute_charged": False,
                    "transfer_manifest_proofs": transfer_proofs[relative],
                }
            )
            return
        if recorded_owner and recorded_owner != pod:
            unresolved.append({"kind": "foreign_owner_without_verified_transfer", "source": source})
            return
        key = (
            relative
            if kind == "verified_asset_transfer_operation"
            else str(path.parent.relative_to(root))
        )
        components.append(
            {
                "component_id": component_id(pod, kind, key),
                "kind": kind,
                "owning_pod_id": pod,
                "source": source,
                "resources": resource,
                "sparse_counter_ledger_present": True,
                **(extra or {}),
            }
        )
        used_ledgers.add(path)

    physical = []
    for path in sorted(results.rglob("result.json")):
        if software(path) or path.is_symlink() or not (path.parent / "trace.npz").is_file():
            continue
        row = load(path)
        if "resources" not in row:
            unresolved.append(
                {
                    "kind": "physical_episode_missing_resources",
                    "path": str(path.relative_to(root)),
                    "sha256": digest(path),
                }
            )
            continue
        resource = counters(row["resources"])
        action = row.get("environment_actions", row.get("actions"))
        if action is not None:
            require(
                resource.get("environment_actions", action) == action,
                "Episode action counter differs",
            )
            resource["environment_actions"] = action
        add(
            path,
            "physical_episode",
            resource,
            extra={
                "status": "technical_failure" if row.get("error") else "recorded_episode",
                "trace_path": str((path.parent / "trace.npz").relative_to(root)),
                "trace_bytes_loaded": 0,
                "wrapper_rows_not_counted_again": True,
                "episode_seconds": row.get("episode_seconds"),
                "planning_seconds": row.get("planning_seconds"),
                "method_or_configuration": row.get("method", row.get("configuration")),
                "job_id": row.get("job_id", row.get("id")),
            },
        )
        physical.append(path)
        for parent in path.parents:
            if not parent.is_relative_to(results):
                break
            if (parent / "started.json").exists():
                covered_jobs.add(parent)
                break

    for started in sorted(results.rglob("started.json")):
        if software(started) or started.parent in covered_jobs or started.is_symlink():
            continue
        folder = started.parent
        # Only actual wrapper jobs use started.json. Metadata/source-freeze starts
        # without a jobs parent are classified separately, not invented as episodes.
        if folder.parent.name != "jobs":
            continue
        candidates = []
        journal = folder / "progress-journal.jsonl"
        if journal.exists():
            replay = replay_progress_journal(journal)
            candidates.append(
                (
                    journal,
                    replay["last"]["resources"],
                    "durable_journal_prefix",
                    replay["truncated_tail_bytes"],
                )
            )
        if not candidates:
            for name in ("live.json", "failure.json", "result.json"):
                path = folder / name
                if path.exists():
                    row = load(path)
                    if isinstance(row.get("resources"), dict):
                        resource = dict(row["resources"])
                        action = row.get(
                            "environment_actions", row.get("observed_environment_actions")
                        )
                        if action is not None:
                            resource["environment_actions"] = action
                        candidates.append((path, resource, "legacy_partial_counter_lower_bound", 0))
                        break
        if candidates:
            path, resource, authority, tail = candidates[0]
            add(
                path,
                "incomplete_physical_attempt",
                resource,
                extra={
                    "started_path": str(started.relative_to(root)),
                    "started_sha256": digest(started),
                    "authority": authority,
                    "truncated_journal_tail_bytes": tail,
                    "actual_work_may_exceed_prefix": authority != "durable_journal_prefix",
                    "partial_job_not_replayed_or_called_complete": True,
                },
            )
            unresolved.append(
                {
                    "kind": "incomplete_attempt_requires_closeout",
                    "path": str(folder.relative_to(root)),
                    "known_prefix_recorded": True,
                    "unobserved_tail_not_guessed_zero": True,
                }
            )
        else:
            unresolved.append(
                {
                    "kind": "started_without_work_ledger",
                    "path": str(started.relative_to(root)),
                    "sha256": digest(started),
                    "actual_work": None,
                }
            )

    # One preparation ledger per stage folder. Priority prevents complete/live/report
    # duplicate counting of the same cumulative acquisition/encoding/preparation run.
    prep_folders = set()
    explicit = {path.resolve() for path in args.preparation_ledger}
    for name in ("resources.json", "complete.json", "report.json", "failure.json", "live.json"):
        for path in sorted(results.rglob(name)):
            if (
                software(path)
                or path.is_symlink()
                or path.parent in prep_folders
                or path in used_ledgers
            ):
                continue
            relative_parts = path.relative_to(results).parts
            if "jobs" in relative_parts or "episodes" in relative_parts:
                continue
            known = relative_parts[0].startswith(PREPARATION_PREFIXES)
            if not known and path.resolve() not in explicit:
                continue
            if any(item.is_relative_to(path.parent) for item in physical):
                continue
            row = load(path)
            resource = row if name == "resources.json" else row.get("resources")
            if not isinstance(resource, dict):
                continue
            add(
                path,
                "preparation_acquisition_encoding",
                resource,
                extra={
                    "cumulative_stage_ledger_priority": name,
                    "public_original_collection_not_new_simulator_work": True,
                    "wall_seconds": row.get("total_wall_seconds", row.get("elapsed_seconds"))
                    if name != "resources.json"
                    else None,
                },
            )
            prep_folders.add(path.parent)
    for path in explicit:
        require(path.is_relative_to(root), "Preparation ledger is outside current project")
        if path not in used_ledgers and not any(
            row["source"]["path"] == str(path.relative_to(root)) for row in copied
        ):
            unresolved.append(
                {
                    "kind": "explicit_preparation_ledger_not_classified",
                    "path": str(path.relative_to(root)),
                }
            )

    for transfer_path, transfer in transfer_records:
        verified_bytes = 0
        for name, expected in transfer["sha256"].items():
            path = root / name
            require(
                path.is_file() and digest(path) == expected,
                "Transferred source bytes changed: " + name,
            )
            verified_bytes += path.stat().st_size
        if "new_simulator_actions" not in transfer or "new_model_queries" not in transfer:
            unresolved.append(
                {
                    "kind": "transfer_operation_missing_explicit_work_ledger",
                    "path": str(transfer_path.relative_to(root)),
                    "sha256": digest(transfer_path),
                }
            )
            continue
        add(
            transfer_path,
            "verified_asset_transfer_operation",
            {
                "verified_asset_transfer_files": len(transfer["sha256"]),
                "verified_asset_transfer_bytes": verified_bytes,
                "environment_actions": transfer["new_simulator_actions"],
                "new_model_queries": transfer["new_model_queries"],
            },
            extra={
                "copied_preparation_compute_is_not_recharged": True,
                "verified_asset_footprint_is_not_inferred_transport_byte_count": True,
            },
        )

    proofs = load(args.zero_work_proofs) if args.zero_work_proofs else []
    if isinstance(proofs, dict):
        proofs = proofs["proofs"]
    by_name = {proof["worker_name"]: proof for proof in proofs}
    worker_attempts = []
    classified_paths = [record["source"]["path"] for record in components]
    classified_paths += [record["source"]["path"] for record in copied]
    for path in sorted((root / "environment/workers").glob("*.json")):
        worker = load(path)
        namespace = worker.get("out")
        active = identity_live(worker["pid"], worker["start_ticks"])
        require(
            not args.final or not active or worker["pid"] == os.getpid(),
            "A registered worker remains live; use a provisional report or wait for exact retirement",
        )
        proof = by_name.get(worker.get("name"))
        record = {
            "record": str(path.relative_to(root)),
            "sha256": digest(path),
            "name": worker.get("name"),
            "pid": worker.get("pid"),
            "start_ticks": worker.get("start_ticks"),
            "output_namespace": namespace,
            "identity_live_at_collection": active,
        }
        if proof:
            evidence = root / proof["evidence_path"]
            require(digest(evidence) == proof["evidence_sha256"], "Zero-work proof changed")
            require(
                proof.get("reviewed_by") and proof.get("reason"),
                "Zero-work proof needs explicit review/reason",
            )
            record.update(classification="explicit_reviewed_zero_work", proof=proof)
        elif namespace and any(
            source.startswith(namespace.rstrip("/") + "/") for source in classified_paths
        ):
            record["classification"] = "physical_or_preparation_evidence_present"
        else:
            record["classification"] = "no_classified_ledger_unknown_work"
            unresolved.append(
                {
                    "kind": "worker_without_classified_ledger",
                    **record,
                    "zero_not_inferred_from_missing_output": True,
                }
            )
        worker_attempts.append(record)
    report = {
        "schema": SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "owning_pod_id": pod,
        "collector_sha256": digest(Path(__file__)),
        "journal_replayer_sha256": digest(ROOT / "experiments/helpers/lewm_progress_journal.py"),
        "components": components,
        "verified_copied_preparation_references": copied,
        "worker_attempts": worker_attempts,
        "unresolved": unresolved,
        "totals": summarize(components),
        "physical_episode_files": len(physical),
        "incomplete_wrapper_rows_do_not_duplicate_inner_episodes": True,
        "numeric_collection_executed_only_on_declared_rented_host": True,
        "new_model_simulator_calls": 0,
        "complete_final_accounting": args.final and not unresolved,
        "note": "Current new work only; public dataset's original collection and exact-hash transferred preparation are excluded from current-host compute. Legacy missing attempt fields remain labeled, not silently equated with modern attempt counters.",
    }
    return report


def merge(args):
    reports = [load(path) for path in args.reports]
    require(all(row["schema"] == SCHEMA for row in reports), "Unknown host accounting report")
    require(
        len({row["owning_pod_id"] for row in reports}) == len(reports),
        "Merge exactly one authoritative report per host; never combine provisional and final versions",
    )
    unique, copied, unresolved = {}, [], []
    for row in reports:
        for record in row["components"]:
            key = record["component_id"]
            if key in unique:
                require(
                    unique[key] == record,
                    "Two versions of one physical component; select one final host report",
                )
            unique[key] = record
        copied.extend(row["verified_copied_preparation_references"])
        unresolved.extend(row["unresolved"])
    return {
        "schema": SCHEMA + "-aggregate",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_reports": [{"path": str(path), "sha256": digest(path)} for path in args.reports],
        "components": list(unique.values()),
        "totals": summarize(list(unique.values())),
        "copied_references_not_recharged": copied,
        "unresolved": unresolved,
        "complete_final_accounting": all(row["complete_final_accounting"] for row in reports)
        and not unresolved,
        "collector_rented_pod_id": args.pod_id,
        "new_model_simulator_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    host = commands.add_parser("collect")
    host.add_argument("--project-root", type=Path, default=ROOT)
    host.add_argument("--preparation-ledger", type=Path, action="append", default=[])
    host.add_argument("--zero-work-proofs", type=Path)
    host.add_argument(
        "--transfer-manifest",
        type=Path,
        action="append",
        default=[],
        help="Additional verified exact-hash copies, e.g. final-case transfer; original navigation-transfer-v1 is automatic",
    )
    host.add_argument("--final", action="store_true")
    combined = commands.add_parser("merge")
    combined.add_argument("--reports", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    require(
        os.environ.get("RUNPOD_POD_ID") == args.pod_id,
        "All numerical accounting runs on the declared rented host only",
    )
    require(not args.out.exists(), "Use a new immutable accounting report path")
    report = collect(args) if args.command == "collect" else merge(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "sha256": digest(args.out),
                "complete_final_accounting": report["complete_final_accounting"],
                "unresolved_components": len(report["unresolved"]),
                "new_model_simulator_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
