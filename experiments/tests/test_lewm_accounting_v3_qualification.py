"""Rented-only stdlib JSON qualification of two narrow accounting repairs."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
POD = "wq4ar2qx2ff6r4"


def module(version):
    path = ROOT / f"experiments/scripts/collect_lewm_campaign_20261031_{version}.py"
    spec = importlib.util.spec_from_file_location("collector_" + version, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def build(root):
    folder = root / "results/controlled-stage/jobs/one"
    inner = folder / "episode/result.json"
    row = {
        "actions": 3,
        "resources": {"environment_actions": 3, "environment_resets": 1},
        "error": None,
    }
    write(inner, row)
    (inner.parent / "trace.npz").write_bytes(b"synthetic bytes, never decoded")
    write(inner.parent / "started.json", {})
    outer = {
        "environment_actions": 3,
        "resources": dict(row["resources"]),
        "error": None,
        "extra": {
            "episode_result": str(inner),
            "episode_result_sha256": hashlib.sha256(inner.read_bytes()).hexdigest(),
        },
    }
    write(folder / "result.json", outer)
    write(folder / "started.json", {"key": "one"})
    write(
        folder.parent.parent / "state.json",
        {"phase": "complete", "active": None, "jobs": {"one": outer}},
    )
    write(folder.parent.parent / "source-freeze.json", {"pod_id": POD})
    return folder, inner, row, outer


def args(root):
    return argparse.Namespace(
        project_root=root,
        pod_id=POD,
        transfer_manifest=[],
        preparation_ledger=[],
        zero_work_proofs=None,
        final=True,
    )


def gate():
    assert os.environ.get("RUNPOD_POD_ID") == POD


def test_nested_count_once_and_old_duplicate():
    gate()
    with tempfile.TemporaryDirectory(prefix="lewm-accounting-qual-", dir="/workspace") as name:
        root = Path(name)
        build(root)
        old, new = module("v2").collect(args(root)), module("v3").collect(args(root))
        assert old["totals"]["normalized_recorded_totals"]["environment_actions"] == 6
        assert len(old["unresolved"]) == 1 and not old["complete_final_accounting"]
        assert new["totals"]["normalized_recorded_totals"]["environment_actions"] == 3
        assert new["complete_final_accounting"] and new["unresolved"] == []
        assert len(new["components"]) == 1 and new["physical_episode_files"] == 1
        assert new["components"][0]["trace_bytes_loaded"] == 0


def test_bad_hash_action_resources_and_incomplete_state_rejected():
    gate()
    new = module("v3")
    with tempfile.TemporaryDirectory(prefix="lewm-accounting-qual-", dir="/workspace") as name:
        root = Path(name)
        folder, inner, row, outer = build(root)
        assert new.completed_outer_binds_physical(root, folder, inner, row)
        for field in ("hash", "actions", "resources", "state"):
            folder, inner, row, outer = build(root)
            if field == "hash":
                outer["extra"]["episode_result_sha256"] = "0" * 64
            elif field == "actions":
                outer["environment_actions"] = 4
            elif field == "resources":
                outer["resources"]["environment_resets"] = 2
            else:
                write(
                    folder.parent.parent / "state.json",
                    {"phase": "running", "active": None, "jobs": {"one": outer}},
                )
            write(folder / "result.json", outer)
            assert not new.completed_outer_binds_physical(root, folder, inner, row)


def test_generic_completed_identity_and_job_hash():
    gate()
    new = module("v3")
    with tempfile.TemporaryDirectory(prefix="lewm-accounting-qual-", dir="/workspace") as name:
        root = Path(name)
        folder, inner, row, outer = build(root)
        job = {"job_id": "one"}
        fingerprint = hashlib.sha256(
            json.dumps(job, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        outer.update(
            status="complete",
            accounting_finalized=True,
            identity="own-stage",
            input_sha256=fingerprint,
        )
        write(folder / "result.json", outer)
        write(
            folder / "started.json",
            {"identity": "own-stage", "input_sha256": fingerprint, "job": job},
        )
        write(
            folder.parent.parent / "state.json",
            {
                "phase": "complete",
                "active": None,
                "identity": "own-stage",
                "completed_job_ids": ["one"],
            },
        )
        assert new.completed_outer_binds_physical(root, folder, inner, row)
        outer["input_sha256"] = "0" * 64
        write(folder / "result.json", outer)
        assert not new.completed_outer_binds_physical(root, folder, inner, row)


def test_existing_origin_chain_and_wrong_receipt_rejected():
    gate()
    new = module("v3")
    receipt_path = ROOT / "environment/navigation-transfer-v1.json"
    receipt = json.loads(receipt_path.read_text())
    freeze_path = ROOT / "results/navigation-cases-v3/source-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    fields = (
        ROOT,
        POD,
        "results/navigation-cases-v3/resources.json",
        "preparation_acquisition_encoding",
        freeze_path,
        freeze,
    )
    proof = new.known_transferred_nav_cases_owner(*fields, [(receipt_path, receipt)])
    assert proof and proof["pod_id"] == "vukovc3t4o1dui"
    with tempfile.TemporaryDirectory(prefix="lewm-accounting-qual-", dir="/workspace") as name:
        wrong = Path(name) / "wrong-receipt.json"
        write(wrong, {})
        assert new.known_transferred_nav_cases_owner(*fields, [(wrong, receipt)]) is None
