"""Relocate only the complete unstarted secondary R2 metadata block.

Standard-library JSON/hash work only. Scientific protocol values and source hashes
are unchanged; no worker, environment, model or final data is accessed or started.
"""

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_SHA = "b6900ee408bd25bf8e73f2928750639ec81d75a81c50d813662c69d1e2a37225"
OLD_NAMESPACE = "results/navigation-budget30-comparison-v1/r2"
NEW_NAMESPACE = "results/navigation-budget30-relocated-v1/r2"
NEW_POD = "nj2fqkp2prgk8x"
NAMES = (
    "training-cma-r2.json",
    "training-random-r2.json",
    "training-reinforce-r2.json",
    "fixed-tuning-r2.json",
    "training-controls-r2.json",
    "selection-reservation-r2.json",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reference(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    require(not path.exists(), "Never overwrite frozen relocation metadata")
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return reference(path)


def transform(value, refs):
    if isinstance(value, dict):
        return {key: transform(item, refs) for key, item in value.items()}
    if isinstance(value, list):
        return [transform(item, refs) for item in value]
    if isinstance(value, str):
        if value in refs:
            return refs[value]
        if value == OLD_NAMESPACE or value.startswith(OLD_NAMESPACE + "/"):
            return NEW_NAMESPACE + value[len(OLD_NAMESPACE) :]
    return value


def changed_paths(old, new, refs, path=()):
    if old == new:
        return []
    if isinstance(old, dict) and isinstance(new, dict):
        require(
            set(new) == set(old) | ({"relocation_provenance"} if not path else set()),
            "Unexpected metadata key change",
        )
        return [
            item for key in old for item in changed_paths(old[key], new[key], refs, (*path, key))
        ]
    if isinstance(old, list) and isinstance(new, list):
        require(len(old) == len(new), "Relocation cannot change a panel or schedule length")
        return [
            item
            for index in range(len(old))
            for item in changed_paths(old[index], new[index], refs, (*path, index))
        ]
    host_change = (
        path in (("allowed_pod_ids", 0), ("evaluation", "execution_pod_id"), ("host", "pod_id"))
        and old == "wq4ar2qx2ff6r4"
        and new == NEW_POD
    )
    label_change = path == ("host", "host") and old == "pusht" and new == "recovery"
    reference_change = isinstance(old, str) and transform(old, refs) == new
    require(
        host_change or label_change or reference_change,
        "Scientific setting changed at " + "/".join(map(str, path)),
    )
    return [{"path": list(path), "before": old, "after": new}]


def freeze(args):
    original_path = args.original_manifest.resolve()
    require(
        digest(original_path) == ORIGINAL_SHA, "Original secondary manifest differs from its freeze"
    )
    original, proof = load(original_path), load(args.unstarted_proof)
    require(
        proof["schema"] == "lewm-secondary-r2-unstarted-proof-v1"
        and proof["pod_id"] == "wq4ar2qx2ff6r4",
        "Qualified old-host proof required",
    )
    require(
        proof["original_secondary_manifest"]["sha256"] == ORIGINAL_SHA
        and proof["old_secondary_r2_output_namespace"] == OLD_NAMESPACE
        and proof["old_secondary_r2_output_exists"] is False
        and proof["old_secondary_r2_output_children"] == [],
        "Old repeat2 is not proved unstarted",
    )
    require(proof["new_nn_calls"] == proof["new_simulator_calls"] == 0, "Proof must be read-only")
    out = args.out_dir.resolve()
    require(not out.exists() and not args.record.exists(), "Use fresh relocation paths")
    old_dir = original_path.parent
    old_refs = {Path(ref["path"]).name: ref for ref in original["files"]}
    for name in NAMES:
        require(
            digest(old_dir / name) == old_refs[name]["sha256"],
            "Original R2 protocol changed: " + name,
        )
    mapping = {}
    for name in NAMES:
        target = str((out / name).relative_to(ROOT))
        mapping[str(old_dir / name)] = target
        mapping[str((old_dir / name).relative_to(ROOT))] = target
    args.record.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "lewm-navigation-secondary-relocation-v1",
        "frozen_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "study_family": "secondary30",
        "repeat_id": 2,
        "old_host": {"host": "pusht", "pod_id": "wq4ar2qx2ff6r4"},
        "new_host": {"host": "recovery", "pod_id": NEW_POD},
        "provider": {
            "gpu": "NVIDIA RTX 4000 Ada 20GB",
            "hourly_usd": 0.28,
            "global_cutoff_utc": "2026-10-31T07:23:14+00:00",
            "root_reserved_campaign_estimate_usd": 9.20725,
        },
        "original_secondary_manifest": reference(original_path),
        "unstarted_proof": {"path": args.proof_remote_path, "sha256": digest(args.unstarted_proof)},
        "retained_local_unstarted_proof": reference(args.unstarted_proof.resolve()),
        "qualified_unstarted_observed_utc": proof["observed_utc"],
        "earlier_stdout_observation": proof["earlier_stdout_observation"],
        "whole_unstarted_repeat_block": True,
        "scientific_settings_unchanged": True,
        "all_cases_seeds_budgets_preserved": True,
        "all_local_methods_and_controls_moved": True,
        "old_output_namespace": OLD_NAMESPACE,
        "new_output_namespace": NEW_NAMESPACE,
        "new_secondary_manifest_path": str((out / "manifest.json").relative_to(ROOT)),
        "final_pixels_read": 0,
        "workers_started": 0,
        "builder_sha256": digest(Path(__file__)),
        "reason": "Root moved the whole unstarted repeat2 to the authorized third host to finish within the common deadline. Existing repeats/results and all scientific settings remain as frozen. All methods and local controls stay within the repeat's actual physical host.",
    }
    record_ref = write(args.record, record)
    out.mkdir()
    revised_refs, diffs = {}, {}
    for name in NAMES:
        old = load(old_dir / name)
        revised = transform(copy.deepcopy(old), mapping)
        if "allowed_pod_ids" in revised:
            require(
                revised["allowed_pod_ids"] == ["wq4ar2qx2ff6r4"],
                "Unexpected original R2 allowed pod",
            )
            revised["allowed_pod_ids"] = [NEW_POD]
        revised["evaluation"]["execution_pod_id"] = NEW_POD
        if "host" in revised:
            revised["host"] = {**revised["host"], "host": "recovery", "pod_id": NEW_POD}
        revised["relocation_provenance"] = {
            "record": record_ref,
            "parent_protocol": reference(old_dir / name),
        }
        require(
            revised["source_sha256"] == old["source_sha256"], "Scientific source hashes changed"
        )
        diffs[name] = changed_paths(old, revised, mapping)
        revised_refs[name] = write(out / name, revised)
    manifest = copy.deepcopy(original)
    require(
        [row["repeat_id"] for row in manifest["schedule"]] == [0, 1, 2],
        "Original repeat schedule changed",
    )
    original_r0_r1 = copy.deepcopy(manifest["schedule"][:2])
    schedule = transform(manifest["schedule"][2], mapping)
    schedule.update(host="recovery", pod_id=NEW_POD)
    manifest["schedule"][2] = schedule
    require(
        manifest["schedule"][:2] == original_r0_r1, "Original R0/R1 references must remain exact"
    )
    manifest["files"] = [revised_refs.get(Path(ref["path"]).name, ref) for ref in original["files"]]
    manifest.update(
        created_utc=record["frozen_utc"],
        metadata_relocation=record_ref,
        original_secondary_manifest=reference(original_path),
        no_workers_launched_by_this_relocation=True,
        existing_original_repeats_0_1_retained_unchanged=True,
        relocated_repeat2_was_unstarted=True,
        relocation_builder_sha256=digest(Path(__file__)),
    )
    require(
        manifest["source_sha256"] == original["source_sha256"],
        "Authoritative scientific source hashes changed",
    )
    final_ref = write(out / "manifest.json", manifest)
    audit_ref = write(
        out / "relocation-diff-audit.json",
        {
            "schema": "lewm-navigation-secondary-relocation-audit-v1",
            "record": record_ref,
            "authoritative_secondary_manifest": final_ref,
            "changed_metadata_paths": diffs,
            "copied_R2_protocol_files": 6,
            "original_R0_R1_schedule_and_file_hashes_unchanged": True,
            "scientific_source_sha_unchanged": True,
            "all_numeric_scientific_values_unchanged": True,
            "all_cases_seeds_genotypes_budgets_methods_learning_settings_preserved": True,
            "final_pixels_read": 0,
            "workers_started": 0,
        },
    )
    return {
        "record": record_ref,
        "manifest": final_ref,
        "audit": audit_ref,
        "relocated_files": list(revised_refs.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-manifest",
        type=Path,
        default=ROOT
        / "configs/lewm_campaign_20261031/navigation-budget30-comparison-v1/manifest.json",
    )
    parser.add_argument("--unstarted-proof", type=Path, required=True)
    parser.add_argument(
        "--proof-remote-path",
        default="results/evidence-audits/navigation-secondary-r2-unstarted-v2.json",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=ROOT / "configs/lewm_campaign_20261031/navigation-secondary-r2-relocation-v1.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "configs/lewm_campaign_20261031/navigation-budget30-relocated-v1",
    )
    args = parser.parse_args()
    print(json.dumps(freeze(args), indent=2))


if __name__ == "__main__":
    main()
