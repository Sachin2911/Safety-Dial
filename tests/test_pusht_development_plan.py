"""Development feasibility must bind future sources/candidates before final generation."""
import copy
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from helpers import pushtDevelopmentPlan as plans
from helpers.pushtBankSampling import SAMPLING_PROTOCOL, ProgressBankWriter
from helpers.pushtFeasibility import (
    PROTOCOL, feasibility_gate, generator_identity, require_feasibility_report,
)
from helpers.pushtGeometry import Disc, T_LOCAL, clearance_trace
from helpers.pushtSourceFamilies import SOURCE_FAMILY_PROTOCOL
from helpers.runManifest import file_sha256
from helpers.splitIntegrity import bank_identity
from scripts.pusht_development_witness import validate_witness_inputs


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    study = tmp_path / "study"
    dev = study / "dev"
    dev.mkdir(parents=True)
    repo = tmp_path / "repo"
    monkeypatch.setattr(plans, "REPO_ROOT", repo)
    hashes = {}
    for relative in plans.REQUIRED_SOURCES:
        original, saved = repo / relative, study / "source_snapshot" / relative
        original.parent.mkdir(parents=True, exist_ok=True)
        saved.parent.mkdir(parents=True, exist_ok=True)
        original.write_text(f"# frozen test source: {relative}\n")
        saved.write_bytes(original.read_bytes())
        hashes[relative] = file_sha256(saved)
    roles = {"dev": {"familiar": [0, 1, 2]}, "test": {"familiar": [10], "heldout": [20]}}
    start, goal = [100., 100., 149., 150., 0., 0., 0.], [100., 100., 400., 150., 0., 0., 0.]
    schedules = {}
    for role, families in roles.items():
        schedules[role] = {}
        for family, episodes in families.items():
            schedules[role][family] = [{"episode": ep, "t0": 0, "goal_offset": 30, "k": 0,
                "start": start, "goal": goal, "root_id": f"{role}-{family}-{ep}", "root_seed": 123 + ep,
                "pair_draw": 0, "candidate_round": 0, "source_position": i}
                for i, ep in enumerate(episodes)]
    plan = {"protocol_version": plans.PLAN_PROTOCOL, "sampling_protocol": SAMPLING_PROTOCOL, "source_roles": roles,
            "source_family_protocol": SOURCE_FAMILY_PROTOCOL, "source_snapshot_sha256": hashes,
            "candidate_schedule": schedules,
            "settings": {"dev_roots": 3, "test_roots": 1, "dev_tapes": 2, "test_tapes": 2}}
    plan_path = study / "sampling-plan.json"
    write_json(plan_path, plan)
    manifest = {"status": "complete", "data": {"source_roles": roles,
        "sampling_plan_sha256": file_sha256(plan_path), "sampling_protocol": SAMPLING_PROTOCOL,
        "source_family_protocol": SOURCE_FAMILY_PROTOCOL}}
    roots, records, layouts = [], [], []
    hz = Disc(250., 200., 20.)
    nominal = np.array([start[2:5], [250., 150., 0.], goal[2:5]])
    route = np.array([start[2:5], [149., 20., 0.], [400., 20., 0.], goal[2:5]])
    for c in schedules["dev"]["familiar"]:
        root = {"root_id": c["root_id"], "seed": c["root_seed"], "start_state": start,
            "goal_state": goal, "prefix": [[0., 0.]] * 10,
            "meta": {**{k: c[k] for k in ("episode", "t0", "goal_offset", "k")},
                "state_at_root": start, "source_family": "familiar",
                "source_family_protocol": SOURCE_FAMILY_PROTOCOL["version"]}}
        layout = {"root_id": c["root_id"], "family": "familiar", "hazard": hz.to_dict(),
            "start_clearance": float(clearance_trace(np.array([start[2:5]]), hz)[0]),
            "goal_clearance": float(clearance_trace(np.array([goal[2:5]]), hz)[0]),
            "nominal_min_clearance": float(clearance_trace(nominal, hz).min())}
        states = np.tile(start, (len(route), 1))
        states[:, 2:5] = route
        records.append({"role": "development", "root": root, "layout": layout,
            "local_polygons": [p.tolist() for p in T_LOCAL], "root_pose": start[2:5],
            "nominal_route_poses": nominal.tolist(), "states": states.tolist(),
            "actions": np.zeros((3, 2)).tolist(), "observed_steps": 3,
            "replay": {"repeats": 2, "bitwise_equal": True}})
        roots.append(root)
        layouts.append(layout)
    blob = {"status": "complete", "manifest": manifest, "roots": roots, "n_branches": 6}
    write_json(dev / "roots.json", blob)
    write_json(dev / "manifest.json", manifest)
    write_json(dev / "layouts.json", {"layouts": layouts})
    (dev / "sampling-plan.json").write_bytes(plan_path.read_bytes())
    with h5py.File(dev / "branches.h5", "w") as f:
        f["root_index"] = np.repeat(np.arange(3), 2)
        f["tape"] = np.zeros((6, 5, 5, 2))
    return {"study": study, "dev": dev, "plan_path": plan_path, "plan": plan,
            "blob": blob, "records": records, "repo": repo}


def sync_plan(f):
    write_json(f["plan_path"], f["plan"])
    manifest = f["blob"]["manifest"]
    manifest["data"]["source_roles"] = f["plan"]["source_roles"]
    manifest["data"]["sampling_plan_sha256"] = file_sha256(f["plan_path"])
    manifest["data"]["source_family_protocol"] = f["plan"]["source_family_protocol"]
    write_json(f["dev"] / "manifest.json", manifest)
    write_json(f["dev"] / "roots.json", f["blob"])
    (f["dev"] / "sampling-plan.json").write_bytes(f["plan_path"].read_bytes())


def witness_report(f, *, bind=True):
    report = {"protocol": PROTOCOL, "generator_identity": generator_identity(),
        "development_bank_identity": bank_identity(f["dev"]), "witnesses": f["records"],
        "gate": feasibility_gate(f["records"], [0, 1, 2])}
    assert report["gate"]["passes"]
    if bind:
        report["frozen_sampling_plan"] = plans.validate_development_plan(f["dev"], f["plan_path"])
    path = f["study"] / "witnesses.json"
    write_json(path, report)
    return path


def test_development_only_proves_reserved_split_without_final_banks(frozen):
    f = frozen
    assert not (f["study"] / "test").exists()
    assert not (f["study"] / "stress").exists()
    audit, binding = validate_witness_inputs(f["study"], development_only=True,
                                           sampling_plan=f["plan_path"])
    assert audit["passes"] and audit["final_banks_evaluated"] is False
    assert audit["source_role_counts"]["test"] == {"familiar": 1, "heldout": 1}
    assert binding["sha256"] == file_sha256(f["plan_path"])
    report = witness_report(f)
    gate = require_feasibility_report(report, f["dev"], sampling_plan=f["plan_path"])
    assert gate["passes"] and gate["frozen_sampling_plan"] == binding


@pytest.mark.parametrize("mutation", ["overlap", "unreserved_candidate", "missing_future",
    "changed_geometry", "incomplete", "wrong_root", "wrong_seed", "wrong_family",
    "wrong_root_count", "wrong_tape_count", "missing_plan_copy", "modified_plan_copy",
    "modified_source_snapshot", "modified_current_source"])
def test_cannot_weaken_the_frozen_development_gate(frozen, mutation):
    f = frozen
    if mutation == "overlap":
        f["plan"]["source_roles"]["test"]["familiar"] = [0]
        sync_plan(f)
    elif mutation == "unreserved_candidate":
        f["plan"]["candidate_schedule"]["test"]["familiar"][0]["episode"] = 0
        sync_plan(f)
    elif mutation == "missing_future":
        del f["plan"]["candidate_schedule"]["test"]["heldout"]
        sync_plan(f)
    elif mutation == "changed_geometry":
        f["plan"]["source_family_protocol"] = {**SOURCE_FAMILY_PROTOCOL, "guard_px": 0.}
        sync_plan(f)
    elif mutation in {"incomplete", "wrong_root", "wrong_seed", "wrong_family", "wrong_root_count"}:
        if mutation == "incomplete":
            f["blob"]["status"] = "running"
        elif mutation == "wrong_root":
            f["blob"]["roots"][0]["meta"]["t0"] = 1
        elif mutation == "wrong_seed":
            f["blob"]["roots"][0]["seed"] += 1
        elif mutation == "wrong_family":
            f["blob"]["roots"][0]["meta"]["state_at_root"] = [100., 100., 100., 150., 0., 0., 0.]
        else:
            f["blob"]["roots"].pop()
        write_json(f["dev"] / "roots.json", f["blob"])
    elif mutation == "wrong_tape_count":
        with h5py.File(f["dev"] / "branches.h5", "r+") as h5:
            h5["root_index"][:] = [0, 0, 0, 1, 1, 2]  # total matches; per-root counts do not
    elif mutation == "missing_plan_copy":
        (f["dev"] / "sampling-plan.json").unlink()
    elif mutation == "modified_plan_copy":
        (f["dev"] / "sampling-plan.json").write_text("{}\n")
    else:
        relative = next(iter(plans.REQUIRED_SOURCES))
        source = (f["study"] / "source_snapshot" if mutation == "modified_source_snapshot" else f["repo"]) / relative
        source.write_text("# changed generator\n")
    with pytest.raises(ValueError):
        plans.validate_development_plan(f["dev"], f["plan_path"])


def test_passing_witness_cannot_authorize_a_different_future_candidate(frozen):
    f = frozen
    report = witness_report(f)
    old_binding = json.loads(report.read_text())["frozen_sampling_plan"]
    f["plan"]["candidate_schedule"]["test"]["heldout"][0]["root_seed"] += 1
    sync_plan(f)  # internally consistent new plan/dev metadata, but witness is already frozen
    assert plans.validate_development_plan(f["dev"], f["plan_path"])["sha256"] != old_binding["sha256"]
    with pytest.raises(ValueError, match="different frozen future"):
        require_feasibility_report(report, f["dev"], sampling_plan=f["plan_path"])


def test_final_phase_requires_binding_even_for_a_valid_legacy_witness(frozen):
    f = frozen
    report = witness_report(f, bind=False)
    assert require_feasibility_report(report, f["dev"])["passes"]
    with pytest.raises(ValueError, match="does not bind"):
        require_feasibility_report(report, f["dev"], sampling_plan=f["plan_path"])


def test_normal_mode_still_requires_and_validates_all_banks(frozen):
    f = frozen
    with pytest.raises(FileNotFoundError):
        validate_witness_inputs(f["study"])
    for name in ("test", "stress"):
        folder = f["study"] / name
        folder.mkdir()
        write_json(folder / "roots.json", {"roots": [{"root_id": "test-familiar-10", "meta": {"episode": 10}}]})
    audit, binding = validate_witness_inputs(f["study"])
    assert audit["passes"] and binding is None
    with pytest.raises(ValueError, match="require.*frozen"):
        validate_witness_inputs(f["study"], development_only=True)
    with pytest.raises(ValueError, match="requires development-only"):
        validate_witness_inputs(f["study"], sampling_plan=f["plan_path"])


def test_unbounded_or_duplicate_future_attempts_are_rejected(frozen):
    f = frozen
    schedule = f["plan"]["candidate_schedule"]["test"]["heldout"]
    original = copy.deepcopy(schedule[0])
    schedule[:] = [{**original, "t0": t} for t in range(9)]
    sync_plan(f)
    with pytest.raises(ValueError, match="bounded schedule"):
        plans.validate_development_plan(f["dev"], f["plan_path"])
    schedule[:] = [original, copy.deepcopy(original)]
    sync_plan(f)
    with pytest.raises(ValueError, match="duplicate attempts"):
        plans.validate_development_plan(f["dev"], f["plan_path"])


def test_real_progress_writer_bundle_passes_pre_final_witness_validation(frozen):
    """Use the actual phase producer's writer/protocol and its complete manifest schema."""
    from helpers.branchBank import Branch
    from helpers.pushtLayouts import Layout
    from helpers.pushtReplay import DenseLog, Root, StepLedger

    f = frozen
    generated = f["study"] / "generated-dev"
    writer = ProgressBankWriter(generated, bank_name="dev", plan_path=f["plan_path"], with_frames=False)
    for record in f["records"]:
        root = Root.from_dict(record["root"])
        writer.add_root(root)
        tape = np.zeros((5, 5, 2))
        log = DenseLog(np.tile(root.start_state, (26, 1)), np.zeros((26, 2)),
            np.zeros(26), np.zeros(26, int), tape.reshape(-1, 2), None)
        writer.add_branches([Branch(root.root_id, tape, "random", {}, log) for _ in range(2)])
        writer.layouts.append(Layout(**{**record["layout"], "route_fraction": .5}))
    manifest = {"status": "complete", "data": {
        "source_roles": f["plan"]["source_roles"], "source_family_protocol": SOURCE_FAMILY_PROTOCOL,
        "sampling_plan_sha256": file_sha256(f["plan_path"]), "sampling_protocol": SAMPLING_PROTOCOL,
        "source_snapshot_sha256": f["plan"]["source_snapshot_sha256"]},
        "metrics": {"n_layouts": len(writer.layouts), "n_roots": len(writer.roots)}}
    writer.finish(StepLedger(), manifest)
    write_json(generated / "manifest.json", manifest)
    binding = plans.validate_development_plan(generated, f["plan_path"])
    assert binding["sha256"] == file_sha256(f["plan_path"])
    assert json.loads((generated / "roots.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("final_bank", ["test", "stress"])
def test_prospective_creation_rejects_existing_final_banks_but_reverification_allows_them(frozen, final_bank):
    f = frozen
    report = witness_report(f)
    (f["study"] / final_bank).mkdir()
    with pytest.raises(ValueError, match="Prospective.*absent final banks"):
        validate_witness_inputs(f["study"], development_only=True, sampling_plan=f["plan_path"])
    # The same correctly created witness must remain verifiable after final generation.
    assert require_feasibility_report(report, f["dev"], sampling_plan=f["plan_path"])["passes"]
