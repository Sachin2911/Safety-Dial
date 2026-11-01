"""Rented-only saved-path accounting qualification, with synthetic JSON and no models."""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OLD = Path("/workspace/Safety-Dial")
OWNER = "nj2fqkp2prgk8x"


def modules():
    assert os.environ.get("RUNPOD_POD_ID") == "uddcaj5qijcl3j"
    loaded = []
    for filename in (
        "collect_lewm_campaign_20261031_v4.py",
        "recover_lewm_campaign_accounting_20261031_v3.py",
    ):
        spec = importlib.util.spec_from_file_location(
            filename[:-3], ROOT / "experiments/scripts" / filename
        )
        value = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(value)
        loaded.append(value)
    return loaded


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def fixture(origin):
    folder = origin / "results/mock-stage/jobs/one"
    inner_path = folder / "episode/result.json"
    inner = {
        "actions": 3,
        "resources": {"environment_actions": 3, "environment_resets": 1},
        "error": None,
    }
    write(inner_path, inner)
    write(inner_path.parent / "started.json", {})
    (inner_path.parent / "trace.npz").write_bytes(b"synthetic bytes, never opened as an array")
    outer = {
        "environment_actions": 3,
        "resources": dict(inner["resources"]),
        "error": None,
        "extra": {
            "episode_result": str(OLD / inner_path.relative_to(origin)),
            "episode_result_sha256": hashlib.sha256(inner_path.read_bytes()).hexdigest(),
        },
    }
    write(folder / "result.json", outer)
    write(folder / "started.json", {"key": "one"})
    state = {"phase": "complete", "active": None, "jobs": {"one": outer}}
    write(folder.parent.parent / "state.json", state)
    write(folder.parent.parent / "source-freeze.json", {"pod_id": OWNER})
    return folder, inner_path, inner, outer, state


def test_exact_absolute_mapping_preserves_original_json_bytes():
    collector, adapter = modules()
    with tempfile.TemporaryDirectory(
        prefix="lewm-recovered-qualification-", dir="/tmp"
    ) as directory:
        origin = Path(directory)
        folder, path, inner, _, _ = fixture(origin)
        original = (folder / "result.json").read_bytes()
        assert not collector.completed_outer_binds_physical(origin, folder, path, inner)
        check = adapter.qualified_outer_binding(collector, OLD, origin)
        assert check(origin, folder, path, inner)
        assert (folder / "result.json").read_bytes() == original


def test_whole_collector_counts_once_instead_of_duplicating_completed_wrapper():
    collector, adapter = modules()
    with tempfile.TemporaryDirectory(
        prefix="lewm-recovered-qualification-", dir="/tmp"
    ) as directory:
        origin = Path(directory)
        fixture(origin)
        args = argparse.Namespace(
            project_root=origin,
            pod_id=OWNER,
            transfer_manifest=[],
            preparation_ledger=[],
            zero_work_proofs=None,
            final=True,
        )
        old = collector.collect(args)
        assert old["totals"]["normalized_recorded_totals"]["environment_actions"] == 6
        assert len(old["unresolved"]) == 1 and not old["complete_final_accounting"]
        collector.completed_outer_binds_physical = adapter.qualified_outer_binding(
            collector, OLD, origin
        )
        new = collector.collect(args)
        assert new["totals"]["normalized_recorded_totals"]["environment_actions"] == 3
        assert new["complete_final_accounting"] and new["unresolved"] == []
        assert len(new["components"]) == 1


def test_bad_hash_counters_state_and_outside_prefix_still_rejected():
    collector, adapter = modules()
    with tempfile.TemporaryDirectory(
        prefix="lewm-recovered-qualification-", dir="/tmp"
    ) as directory:
        origin = Path(directory)
        check = adapter.qualified_outer_binding(collector, OLD, origin)
        for kind in ("hash", "resources", "state", "prefix"):
            folder, path, inner, outer, state = fixture(origin)
            outer = copy.deepcopy(outer)
            if kind == "hash":
                outer["extra"]["episode_result_sha256"] = "0" * 64
            elif kind == "resources":
                outer["resources"]["environment_actions"] = 4
            elif kind == "state":
                state["phase"] = "running"
            else:
                outer["extra"]["episode_result"] = (
                    "/unrelated/Safety-Dial/results/mock-stage/jobs/one/episode/result.json"
                )
            state["jobs"]["one"] = outer
            write(folder / "result.json", outer)
            write(folder.parent.parent / "state.json", state)
            assert not check(origin, folder, path, inner)


def test_generic_input_fingerprint_and_identity_checks_remain_active():
    collector, adapter = modules()
    with tempfile.TemporaryDirectory(
        prefix="lewm-recovered-qualification-", dir="/tmp"
    ) as directory:
        origin = Path(directory)
        folder, path, inner, outer, _ = fixture(origin)
        job = {"job_id": "one"}
        fingerprint = hashlib.sha256(
            json.dumps(job, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        outer.update(
            status="complete", accounting_finalized=True, identity="stage", input_sha256=fingerprint
        )
        write(folder / "result.json", outer)
        write(
            folder / "started.json", {"job": job, "input_sha256": fingerprint, "identity": "stage"}
        )
        write(
            folder.parent.parent / "state.json",
            {
                "phase": "complete",
                "active": None,
                "completed_job_ids": ["one"],
                "identity": "stage",
            },
        )
        check = adapter.qualified_outer_binding(collector, OLD, origin)
        assert check(origin, folder, path, inner)
        outer["input_sha256"] = "0" * 64
        write(folder / "result.json", outer)
        assert not check(origin, folder, path, inner)
