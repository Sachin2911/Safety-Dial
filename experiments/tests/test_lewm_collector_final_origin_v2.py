"""Rented-Nav dummy proof-chain regression for copied final preparation ownership.

Logical destination pods are fixtures; no actual final data, arrays, NN or simulator.
"""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
NAV = "vukovc3t4o1dui"
DEST = "wq4ar2qx2ff6r4"
PREFIX = "results/navigation-final-cases-v1"


def test_exact_nav_origin_and_wrong_proof_rejections(tmp_path):
    assert os.environ.get("RUNPOD_POD_ID") == NAV

    def module(filename, label):
        spec = importlib.util.spec_from_file_location(
            label, ROOT / "experiments/scripts" / filename
        )
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result

    old = module("collect_lewm_campaign_20261031.py", "old_collector")
    new = module("collect_lewm_campaign_20261031_v2.py", "new_collector")
    assert old.SCHEMA == new.SCHEMA == "lewm-campaign-accounting-20261031-v1"
    assert old.digest(ROOT / "experiments/scripts/collect_lewm_campaign_20261031.py") == (
        "e646ca31df73ce75ec280544d01bccc013213c165e0c61b6025822a4efa830e2"
    )

    def fixture(folder, corruption=None):
        # The collector intentionally excludes test_* directory components. Keep
        # dummy evidence under a sibling fixture root so its real scan is exercised.
        root = tmp_path.parent / "collector-origin-fixtures" / folder / "Safety-Dial"

        def put(name, payload):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
            return target

        def reference(name):
            return {"path": name, "sha256": new.digest(root / name)}

        primary_name = "configs/lewm_campaign_20261031/navigation-final-cases-v1.json"
        primary = {
            "schema": "lewm-navigation-final-case-preparation-v1",
            "execution_enabled": True,
            "allowed_pod_ids": [NAV],
            "source_sha256": {},
            "final_opening": {"path": "configs/unused-opening.json", "sha256": "a" * 64},
        }
        if corruption == "primary_not_exclusive":
            primary["allowed_pod_ids"].append(DEST)
        put(primary_name, primary)
        primary_ref = reference(primary_name)
        joint_name = "configs/lewm_campaign_20261031/navigation-joint-final-cases-v2.json"
        opening = {"path": "configs/unused-joint-opening.json", "sha256": "b" * 64}
        put(
            joint_name,
            {
                "schema": "lewm-navigation-joint-final-preparation-v2",
                "execution_enabled": True,
                "primary_preparation_protocol": primary_ref,
                "joint_opening": opening,
            },
        )
        joint_ref = reference(joint_name)
        freeze_name = PREFIX + "/source-freeze.json"
        put(
            freeze_name,
            {
                "protocol_sha256": primary_ref["sha256"],
                "sources": {},
                "final_opening": {
                    "opening_sha256": "a" * 64,
                    "final_pixels_read_before_opening": 0,
                },
            },
        )
        # This is the legitimate old preparer schema: no pod_id or embedded protocol.
        assert new.owner_from(new.load(root / freeze_name)) is None
        put(
            PREFIX + "/resources.json",
            {"environment_resets": 16, "environment_actions": 0, "render_frames": 32},
        )
        dummy_array = root / PREFIX / "inputs/case.npz"
        dummy_array.parent.mkdir()
        dummy_array.write_bytes(b"synthetic bytes only; never decoded as an array")
        access_name = PREFIX + "-joint-access.json"
        put(
            access_name,
            {
                "schema": "lewm-navigation-joint-final-access-v2",
                "protocol_sha256": joint_ref["sha256"],
                "primary_preparation_protocol": primary_ref,
                "joint_opening": opening,
                "all_six_repeat_selections_fixed": True,
                "final_pixels_read_before_delegation": 0,
            },
        )
        worker_name = "environment/final-preparation-origin-v1/source-worker.json"
        put(
            worker_name,
            {
                "pod_id": DEST if corruption == "worker_wrong_origin" else NAV,
                "out": PREFIX,
                "protocol": joint_name,
                "entrypoint": "experiments/scripts/prepare_lewm_navigation_joint_final_cases_v2.py",
                "expected_protocol_sha256": joint_ref["sha256"],
                "pid": 99999999,
                "start_ticks": "0",
            },
        )
        transferred_names = [primary_name, joint_name, access_name, worker_name]
        transferred_names += [
            str(p.relative_to(root)) for p in (root / PREFIX).rglob("*") if p.is_file()
        ]
        origin = {
            "schema": "lewm-final-preparation-nav-origin-v1",
            "source_freeze": reference(freeze_name),
            "primary_preparation_protocol": primary_ref,
            "joint_preparation_protocol": joint_ref,
            "joint_access": reference(access_name),
            "source_worker": reference(worker_name),
        }
        if corruption == "origin_ref_wrong_hash":
            origin["source_freeze"] = {**origin["source_freeze"], "sha256": "0" * 64}
        transfer = {
            "verified": True,
            "source_pod_id": DEST if corruption == "wrong_origin" else NAV,
            "destination_pod_id": NAV if corruption == "wrong_destination" else DEST,
            "sha256": {name: new.digest(root / name) for name in transferred_names},
            "final_preparation_origin": origin,
            "new_simulator_actions": 0,
            "new_model_queries": 0,
        }
        receipt = put("environment/final-cases-transfer-v1.json", transfer)
        if corruption == "unlisted_preparation_file":
            (root / PREFIX / "unlisted.txt").write_text("not covered by the full-tree transfer\n")
        if corruption == "changed_transferred_bytes":
            (root / PREFIX / "source-freeze.json").write_text("{}\n")
        # A separately owned physical episode still charges exactly once under both versions.
        put("results/owned/source-freeze.json", {"pod_id": DEST})
        episode = put(
            "results/owned/jobs/job0/episode/result.json",
            {
                "environment_actions": 3,
                "resources": {
                    "environment_actions": 3,
                    "environment_resets": 1,
                    "render_frames": 4,
                },
            },
        )
        (episode.parent / "trace.npz").write_bytes(b"dummy physical evidence, no array loading")
        return root, SimpleNamespace(
            project_root=root,
            pod_id=DEST,
            transfer_manifest=[receipt],
            preparation_ledger=[],
            zero_work_proofs=None,
            final=True,
        )

    root, args = fixture("valid")
    first = old.collect(args)
    assert any(row["kind"] == "copied_origin_not_hash_bound" for row in first["unresolved"])
    result = new.collect(args)
    assert result["complete_final_accounting"] and not result["unresolved"]
    copied = result["verified_copied_preparation_references"]
    assert len(copied) == 1 and copied[0]["owning_pod_id"] == NAV
    assert copied[0]["current_host_compute_charged"] is False
    assert copied[0]["source"]["source_owner_resolution"]["full_preparation_files_verified"] == 3
    assert result["totals"]["raw_counter_totals_by_field"]["environment_actions"] == 3
    assert result["totals"]["raw_counter_totals_by_field"]["environment_resets"] == 1
    assert result["totals"]["raw_counter_totals_by_field"]["render_frames"] == 4
    # Standard merge/schema/API remain compatible with existing assembly and reporting readers.
    assert old.summarize(result["components"]) == new.summarize(result["components"])
    failures = (
        "wrong_origin",
        "wrong_destination",
        "origin_ref_wrong_hash",
        "worker_wrong_origin",
        "primary_not_exclusive",
        "unlisted_preparation_file",
    )
    for label in failures:
        _, invalid = fixture(label, label)
        report = new.collect(invalid)
        assert not report["complete_final_accounting"]
        assert not report["verified_copied_preparation_references"]
        assert any(row["kind"] == "copied_origin_not_hash_bound" for row in report["unresolved"])
    _, changed = fixture("changed_bytes", "changed_transferred_bytes")
    with pytest.raises(RuntimeError, match="Transferred source bytes changed"):
        new.collect(changed)
    proof = {
        "schema": "lewm-collector-final-origin-synthetic-v2",
        "qualified": True,
        "actual_execution_pod": NAV,
        "logical_destination_pod": DEST,
        "old_collector_sha256": new.digest(
            ROOT / "experiments/scripts/collect_lewm_campaign_20261031.py"
        ),
        "new_collector_sha256": new.digest(
            ROOT / "experiments/scripts/collect_lewm_campaign_20261031_v2.py"
        ),
        "test_sha256": new.digest(Path(__file__)),
        "schema_and_counter_semantics_unchanged": True,
        "original_missing_owner_case_fixed": True,
        "wrong_proofs_rejected": list(failures),
        "changed_bytes_rejected": True,
        "actual_final_data_read": 0,
        "new_model_simulator_calls": 0,
    }
    (tmp_path / "qualification.json").write_text(json.dumps(proof, indent=2) + "\n")
