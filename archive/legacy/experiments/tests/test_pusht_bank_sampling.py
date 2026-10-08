from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import pushtBankSampling as sampling
from helpers.branchBank import Bank, Branch
from helpers.pushtGeometry import Box
from helpers.pushtLayouts import Layout
from helpers.pushtReplay import DenseLog, Root, StepLedger
from helpers.runManifest import file_sha256


@pytest.fixture
def source_h5(tmp_path):
    path = tmp_path / "expert.h5"
    lengths = np.array([80, 80, 80, 80, 10])
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    states = np.tile([256, 256, 192, 192, 0, 3, 4], (sum(lengths), 1)).astype(float)
    with h5py.File(path, "w") as handle:
        handle["ep_len"], handle["ep_offset"], handle["state"] = lengths, offsets, states
    return path


def test_finite_balanced_source_schedule_has_distinct_temporal_draws(source_h5):
    candidates = sampling.candidate_schedule(source_h5, [0, 1, 2, 3], seed=13)
    assert candidates == sampling.candidate_schedule(source_h5, [0, 1, 2, 3], seed=13)
    sampling.validate_candidate_schedule(candidates, [0, 1, 2, 3])
    assert len(candidates) == 32
    assert sorted(c["k"] for c in candidates[:4]) == [0, 2, 4, 6]
    for episode in range(4):
        rows = [c for c in candidates if c["episode"] == episode]
        assert len(rows) == 8
        assert len({(c["t0"], c["goal_offset"]) for c in rows}) == 2
        assert all(c["start"][5:] == c["goal"][5:] == [0, 0] for c in rows)
        for draw in (0, 1):
            assert sorted(c["k"] for c in rows if c["pair_draw"] == draw) == [0, 2, 4, 6]


def test_short_episode_has_one_declared_temporal_pair_and_four_depths(source_h5):
    candidates = sampling.candidate_schedule(source_h5, [4], seed=4)
    assert len(candidates) == 4
    assert {(c["t0"], c["goal_offset"]) for c in candidates} == {(0, 9)}
    sampling.validate_candidate_schedule(candidates, [4])


def test_source_roles_and_schedule_refuse_duplicates_or_cross_role_use(source_h5):
    roles = {"dev": {"familiar": [0]}, "test": {"familiar": [1], "heldout": [0, 2]}}
    with pytest.raises(ValueError, match="cross source"):
        sampling.validate_source_roles(roles)
    candidates = sampling.candidate_schedule(source_h5, [0], seed=1)
    candidates[-1] = {**candidates[0], "candidate_index": len(candidates) - 1}
    with pytest.raises(ValueError, match="uniqueness"):
        sampling.validate_candidate_schedule(candidates, [0])


def root_context():
    state = np.array([256, 256, 192, 192, 0, 0, 0], float)
    root = Root(1, state.copy(), state.copy(), np.zeros((10, 2)), "fixture",
                {"nominal_plan": np.zeros((5, 5, 2)).tolist(), "episode": 0})
    return root, SimpleNamespace(state=state)


@pytest.mark.parametrize("count,quotas", [(8, [3, 2, 2]), (16, [5, 5, 5])])
def test_exact_proposals_fill_remainder_slots_and_preserve_guard_and_uniqueness(count, quotas):
    root, ctx = root_context()
    proposals, audit = sampling.exact_proposals(np.random.default_rng(22), root, ctx, count)
    assert audit["complete"] and len(proposals) == count
    assert proposals[0].kind == "nominal"
    assert all(p.tape.dtype == np.float32 for p in proposals)
    assert [sum(p.params["sigma"] == sigma for p in proposals if p.kind == "random") for sigma in (.05, .1, .2)] == quotas
    assert len({p.tape.tobytes() for p in proposals}) == count
    assert all(not sampling.exits_arena(p.tape, ctx.state[:2]) for p in proposals)
    assert sum(t["outcome"] == "accepted" for t in audit["trials"]) == count


def test_stress_slots_use_only_declared_primitives_on_same_root():
    root, ctx = root_context()
    proposals, audit = sampling.exact_proposals(np.random.default_rng(7), root, ctx, 16,
                                               stress=True, hazard_centre=[350, 256])
    assert audit["complete"] and len(proposals) == 16
    assert [sum(p.kind == kind for p in proposals) for kind in ("stress", "toward_hazard")] == [8, 8]
    assert len({p.tape.tobytes() for p in proposals}) == 16
    assert all(not sampling.exits_arena(p.tape, ctx.state[:2]) for p in proposals)


@pytest.mark.parametrize("failure", ["duplicate", "arena"])
def test_proposal_shortfalls_are_bounded_and_never_filled_with_duplicates(monkeypatch, failure):
    root, ctx = root_context()
    if failure == "duplicate":
        monkeypatch.setattr(sampling, "perturbed_tapes", lambda rng, nominal, n, sigma: nominal[None])
    else:
        monkeypatch.setattr(sampling, "exits_arena", lambda *args: True)
    proposals, audit = sampling.exact_proposals(np.random.default_rng(0), root, ctx, 8)
    assert not audit["complete"]
    assert len(proposals) == (1 if failure == "duplicate" else 0)
    assert len(audit["trials"]) == (21 if failure == "duplicate" else 1)
    assert sum(t["outcome"] == failure + "_rejected" for t in audit["trials"]) == (20 if failure == "duplicate" else 1)


@pytest.fixture
def builder(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/pusht_e1_banks.py"
    spec = importlib.util.spec_from_file_location("bank_recovery_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ProgressBankWriter", lambda *args, **kwargs: sampling.ProgressBankWriter(*args, **kwargs, with_frames=False))
    monkeypatch.setattr(module, "nominal_route", lambda env, root, ctx, ledger: route(env))
    monkeypatch.setattr(module, "generate_layout", lambda rng, route, start, goal, family, root_id:
        Layout(Box(330, 360, 220, 250).to_dict(), family, root_id, .5, -1, 30, 30))
    monkeypatch.setattr(module, "execute_proposals", fake_execute)
    return module


def route(env):
    env.category = "layout_route"
    env.step(None)
    env.step(None)
    return np.tile([192, 192, 0], (3, 1))


def fake_execute(env, root, proposals):
    env.step(None)
    env.step(None)
    # Query outcomes remain in the bank even when the observed horizon is censored.
    log = DenseLog(np.tile(root.start_state, (26, 1)), np.zeros((26, 2)), np.zeros(26), np.zeros(26),
                   proposals[0].tape.reshape(-1, 2), None)
    log.observed = np.r_[np.ones(3, bool), np.zeros(23, bool)]
    return [Branch(root.root_id, proposals[0].tape, proposals[0].kind, proposals[0].params, log)]


def candidate_inputs(source_h5):
    candidates = sampling.candidate_schedule(source_h5, [0, 1], seed=4)
    for row in candidates:
        row.update(root_seed=row["candidate_index"] + 100, root_id=f"dev-familiar-c{row['candidate_index']:05d}")
    return candidates


def install_root_builder(builder, monkeypatch, *, fail_episode=None):
    calls = []

    def fake_build(env, planner, pair, *, seed, k, root_id):
        for _ in range(3):
            env.step(None)
        calls.append(pair["episode"])
        if pair["episode"] == fail_episode:
            raise ValueError("Nominal root has a censored or out-of-domain prefix")
        root = Root(seed, pair["start"], pair["goal"], np.zeros((10, 2)), root_id,
            {"episode": pair["episode"], "t0": pair["t0"], "goal_offset": pair["goal_offset"], "k": k,
             "nominal_plan": np.zeros((5, 5, 2)).tolist(), "state_at_root": pair["start"].tolist(), "in_contact_last_block": False})
        return root, SimpleNamespace(state=pair["start"]), None

    monkeypatch.setattr(builder, "build_root", fake_build)
    return calls


class FakeEnv:
    steps = 0

    def step(self, action):
        self.steps += 1


def test_builder_keeps_one_root_per_source_and_retains_censored_outcomes(builder, source_h5, tmp_path, monkeypatch):
    calls = install_root_builder(builder, monkeypatch)
    plan = tmp_path / "sampling-plan.json"
    plan.write_text("{}")
    env, ledger = FakeEnv(), StepLedger()
    writer, directory, _ = builder.build_bank("dev", env, object(), np.random.default_rng(0),
        {"familiar": candidate_inputs(source_h5)}, {"familiar": 2}, 8, ledger=ledger, plan_path=plan, study_dir=tmp_path)
    writer.finish(ledger, {"status": "complete"})
    bank = Bank(directory)
    try:
        assert len(bank.roots) == 2 and len(bank) == 16
        assert len(calls) == len(set(calls)) == 2
        assert ledger.total == env.steps == 42
        assert not bank.branch(0)["observed"][-1]
        assert (directory / "sampling-plan.json").read_bytes() == plan.read_bytes()
        assert not (directory / "source_snapshot").exists()
    finally:
        bank.h5.close()


def test_exhaustion_preserves_partial_roots_metadata_and_actual_costs(builder, source_h5, tmp_path, monkeypatch):
    calls = install_root_builder(builder, monkeypatch, fail_episode=1)
    plan = tmp_path / "sampling-plan.json"
    plan.write_text("{}")
    env, ledger = FakeEnv(), StepLedger()
    with pytest.raises(RuntimeError, match="Exhausted frozen"):
        builder.build_bank("dev", env, object(), np.random.default_rng(0), {"familiar": candidate_inputs(source_h5)},
            {"familiar": 2}, 8, ledger=ledger, plan_path=plan, study_dir=tmp_path)
    directory = tmp_path / "dev"
    roots = json.loads((directory / "roots.json").read_text())
    progress = json.loads((directory / "progress.json").read_text())
    assert roots["status"] == progress["status"] == "incomplete"
    assert len(roots["roots"]) == 1 and roots["n_branches"] == 8
    assert calls.count(0) == 1 and calls.count(1) == 8
    assert progress["skipped_accepted_sources"] == 7
    assert ledger.total == env.steps == roots["ledger"]["total_steps"] == 45
    assert progress["snapshot_sha256"]["roots.json"] == file_sha256(directory / "roots.json")
    journal = [json.loads(line) for line in (directory / "attempts.jsonl").read_text().splitlines()]
    assert sum(row["charged_steps"] for row in journal if row["event"] == "completed") == 45
    with pytest.raises(ValueError, match="incomplete bank"):
        Bank(directory)


def test_final_phase_requires_feasibility_before_loading_model(builder, tmp_path, monkeypatch):
    import helpers.pushtFeasibility as feasibility

    study = tmp_path / "e1"
    study.mkdir()
    plan = study / "sampling-plan.json"
    plan.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["banks", "--phase", "final", "--output-dir", str(study),
        "--results-dir", str(tmp_path / "results"), "--plan", str(plan), "--feasibility-report", str(tmp_path / "witness.json")])
    monkeypatch.setattr(builder, "verify_plan_sources", lambda *args, **kwargs: None)
    monkeypatch.setattr(feasibility, "require_feasibility_report", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("feasibility failed")))
    monkeypatch.setattr(builder, "load_model", lambda *args: pytest.fail("Loaded model before feasibility passed"))
    with pytest.raises(ValueError, match="feasibility failed"):
        builder.main()
    assert not (study / "test").exists() and not (tmp_path / "results").exists()


def test_tape_shortfall_rejects_before_any_root_or_branch_is_added(builder, source_h5, tmp_path, monkeypatch):
    install_root_builder(builder, monkeypatch)
    monkeypatch.setattr(builder, "exact_proposals", lambda *args, **kwargs: ([], {"complete": False, "trials": []}))
    monkeypatch.setattr(builder, "execute_proposals", lambda *args, **kwargs: pytest.fail("Queried an incomplete tape set"))
    plan = tmp_path / "sampling-plan.json"
    plan.write_text("{}")
    env, ledger = FakeEnv(), StepLedger()
    with pytest.raises(RuntimeError, match="Exhausted frozen"):
        builder.build_bank("dev", env, object(), np.random.default_rng(0), {"familiar": candidate_inputs(source_h5)},
            {"familiar": 2}, 8, ledger=ledger, plan_path=plan, study_dir=tmp_path)
    roots = json.loads((tmp_path / "dev/roots.json").read_text())
    assert roots["roots"] == [] and roots["n_branches"] == 0
    assert env.steps == ledger.total == 16 * 5
    assert ledger.branches["discarded_proposal_shortfall"] == 16


def test_frozen_source_and_input_changes_prevent_final_generation(tmp_path, source_h5):
    repo, study = tmp_path / "repo", tmp_path / "study"
    source = repo / "experiments/generator.py"
    snapshot = study / "source_snapshot/experiments/generator.py"
    for path in (source, snapshot):
        path.parent.mkdir(parents=True)
        path.write_text("frozen source")
    plan_path = study / "sampling-plan.json"
    plan_path.write_text("{}")
    roles = {"dev": {"familiar": [0]}, "test": {"familiar": [1], "heldout": [2]}}
    plan = {"protocol_version": sampling.SAMPLING_PROTOCOL["version"], "sampling_protocol": sampling.SAMPLING_PROTOCOL,
        "source_roles": roles, "candidate_schedule": {role: {family: sampling.candidate_schedule(source_h5, episodes, seed=11)
            for family, episodes in families.items()} for role, families in roles.items()},
        "source_snapshot_sha256": {"experiments/generator.py": file_sha256(source)},
        "input_sha256": {str(source_h5): file_sha256(source_h5)}}
    sampling.verify_plan_sources(plan, plan_path, repo)
    source.write_text("changed source")
    with pytest.raises(ValueError, match="generator identity changed"):
        sampling.verify_plan_sources(plan, plan_path, repo)
    source.write_text("frozen source")
    with h5py.File(source_h5, "a") as handle:
        handle["state"][0, 0] += 1
    with pytest.raises(ValueError, match="Frozen input changed"):
        sampling.verify_plan_sources(plan, plan_path, repo)


def test_invalid_nominal_cannot_be_replaced_by_a_random_tape(monkeypatch):
    root, ctx = root_context()
    root.meta["nominal_plan"] = np.full((5, 5, 2), .6).tolist()
    monkeypatch.setattr(sampling, "perturbed_tapes", lambda *args: pytest.fail("Replaced required nominal slot"))
    proposals, audit = sampling.exact_proposals(np.random.default_rng(0), root, ctx, 8)
    assert not audit["complete"] and proposals == []
    assert len(audit["trials"]) == 1 and audit["trials"][0]["kind"] == "nominal"


def test_float64_differences_that_disappear_in_stored_dtype_are_duplicates(monkeypatch):
    root, ctx = root_context()
    # Distinct float64 commands would both become the nominal zeros in bank storage.
    monkeypatch.setattr(sampling, "perturbed_tapes", lambda rng, nominal, n, sigma: (nominal + 1e-50)[None])
    proposals, audit = sampling.exact_proposals(np.random.default_rng(0), root, ctx, 8)
    assert len(proposals) == 1 and not audit["complete"]
    assert sum(t["outcome"] == "duplicate_rejected" for t in audit["trials"]) == 20
