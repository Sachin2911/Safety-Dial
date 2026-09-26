"""Regression checks for scientific failures found during the continuation audit."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))

from helpers import walkerBank  # noqa: E402
from helpers.walkerRules import WalkerRoot, propose_tapes, rule_unsafe  # noqa: E402
from helpers.walkerValidation import decision_diagnostics, decomposition_gate, episode_role, gate_decision, load_source_split  # noqa: E402
from walker_s4_prospective import balanced_pick, charged_cost, execute_selected  # noqa: E402


def root():
    return WalkerRoot("test-e2-t20", 2, 20, np.zeros(9), np.zeros(9),
                      np.zeros((3, 9)), np.zeros((3, 9)), np.zeros((2, 10, 6)),
                      np.zeros((100, 6)), {})


@pytest.mark.parametrize("n_random", [0, 1, 2, 7, 8, 13])
def test_proposal_counts_are_exact(n_random):
    items = propose_tapes(np.random.default_rng(2), root(), n_random=n_random, n_bursts=2)
    assert len(items) == 1 + n_random + 2
    assert sum(kind == "random" for _, kind, _ in items) == n_random
    assert all(tape.shape == (100, 6) and (abs(tape) <= 1).all() for tape, _, _ in items)


def test_stress_roots_include_last_half_second(monkeypatch):
    n = 200
    ep = {"qpos": np.zeros((n, 9)), "qvel": np.zeros((n, 9)), "action": np.zeros((n, 6)),
          "x_velocity": np.zeros(n), "healthy": np.ones(n), "terminated": np.arange(n) == n - 1}
    monkeypatch.setattr(walkerBank, "iter_episodes", lambda _: iter([(1, ep)]))
    selected = walkerBank.sample_roots(Path("unused"), np.random.default_rng(3), 1, kind="stress")
    assert len(selected) == 1
    assert n - 62 <= selected[0].step < n
    assert selected[0].policy_tape.shape == (100, 6)


def good_gate_inputs():
    probes = {"height_r2": 0.96, "pitch_r2": 0.96, "speed_r2": 0.9}
    errors = {"height": [0.01, 0.05], "pitch": [0.01, 0.1], "speed": [0.1, 0.3]}
    diag = decision_diagnostics([-1] * 10 + [1] * 10, [True] * 10 + [False] * 10)
    return probes, errors, {"health": dict(diag), "speed": dict(diag)}


def test_always_safe_classifier_cannot_pass_high_overall_accuracy():
    probes, errors, diagnostics = good_gate_inputs()
    # The previous smoke gate passed with 95.8% agreement and zero unsafe detection.
    diagnostics["health"] = decision_diagnostics(np.ones(120), [True] * 4 + [False] * 116)
    result = gate_decision(probes, errors, diagnostics)
    assert result["real_readout_tracks_truth"]["health"] is False
    assert result["go"] is False
    assert result["active_rules"] == []


def test_gate_honors_imagination_and_health_only_fallback():
    probes, errors, diagnostics = good_gate_inputs()
    diagnostics["speed"] = decision_diagnostics(np.ones(20), [True] * 10 + [False] * 10)
    result = gate_decision(probes, errors, diagnostics)
    assert result["go"] is True
    assert result["active_rules"] == ["health"]
    errors["height"] = [0.05, 0.01]
    assert gate_decision(probes, errors, diagnostics)["go"] is False


def test_source_roles_are_disjoint_and_health_boundary_is_strict():
    roles = {r: {e for e in range(100) if episode_role(e) == r}
             for r in ("development", "test", "acquisition")}
    assert not roles["development"] & roles["test"]
    assert not roles["test"] & roles["acquisition"]
    assert rule_unsafe("health", [0.0])[0]
    assert not rule_unsafe("speed", [0.0])[0]


def test_root_balance_survives_concentrated_scores():
    candidates = [{"id": f"r{r}c{c}", "root_index": r, "kind": "policy" if c == 0 else "random"}
                  for r in range(4) for c in range(4)]
    scores = {c["id"]: i for i, c in enumerate(candidates)}
    selected = balanced_pick(candidates, 4, scores)
    assert len({c["root_index"] for c in selected}) == 4
    assert len(candidates) == 16
    with pytest.raises(ValueError):
        balanced_pick(candidates, 17, scores)


def test_every_paid_step_is_charged():
    cost = charged_cost(12345, 128, 512)
    assert cost["charged_steps"] == 76345
    assert cost["charged_steps"] == sum(cost[k] for k in ("root_generation_steps", "common_seed_steps", "acquired_steps"))


def test_execution_never_queries_unchosen_candidates(tmp_path, monkeypatch):
    import walker_s4_prospective as runner

    writes = []
    class RecordingWriter:
        def __init__(self, path):
            self.path = path
        def add_branches(self, root, items, env, *, on_step=None):
            for _ in range(100):
                on_step()
            writes.extend(items)
        def finish(self, metadata):
            (self.path / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(runner, "WalkerBankWriter", RecordingWriter)
    monkeypatch.setattr(runner, "WalkerBank", lambda path: path)
    candidates = [{"id": str(i), "root_index": 0, "kind": "random", "params": {},
                   "tape": np.full((100, 6), i / 10).tolist()} for i in range(10)]
    chosen = balanced_pick(candidates, 2, {c["id"]: -int(c["id"]) for c in candidates})
    execute_selected(tmp_path, [root()], chosen, object(), {})
    assert len(writes) == 2
    assert {item[2]["candidate_id"] for item in writes} == {"8", "9"}
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["selected_ids"] == [item[2]["candidate_id"] for item in writes]
    assert metadata["executed_steps"] == 200


def test_bank_writer_refuses_to_overwrite(tmp_path):
    writer = walkerBank.WalkerBankWriter(tmp_path)
    writer.finish({"provenance": {"data": "original"}})
    with pytest.raises(FileExistsError):
        walkerBank.WalkerBankWriter(tmp_path)
    with pytest.raises(ValueError, match="provenance"):
        walkerBank.load_verified_bank(tmp_path, {"data": "different"})


def test_source_manifest_rejects_legacy_and_overlap(tmp_path):
    p = tmp_path / "splits.json"
    p.write_text(json.dumps({"protocol_version": 1}))
    with pytest.raises(ValueError, match="fresh"):
        load_source_split(tmp_path)
    p.write_text(json.dumps({"protocol_version": 2, "roots": {
        "development": [0], "test": [1], "acquisition": [2, 3]}}))
    assert load_source_split(tmp_path)["protocol_version"] == 2
    p.write_text(json.dumps({"protocol_version": 2, "roots": {
        "development": [0], "test": [0, 1], "acquisition": [2, 3]}}))
    with pytest.raises(ValueError, match="overlap"):
        load_source_split(tmp_path)


def test_failed_branch_charges_attempted_simulator_steps(tmp_path, monkeypatch):
    import walker_s4_prospective as runner

    class FailedWriter:
        def __init__(self, path):
            pass
        def add_branches(self, root, items, env, *, on_step):
            for _ in range(3):
                on_step()
            raise FloatingPointError("simulator failure")
    monkeypatch.setattr(runner, "WalkerBankWriter", FailedWriter)
    chosen = [{"id": "failed", "root_index": 0, "kind": "random", "params": {},
               "tape": np.zeros((100, 6)).tolist()}]
    with pytest.raises(FloatingPointError):
        execute_selected(tmp_path, [root()], chosen, object(), {})
    ledger = json.loads((tmp_path / "execution_ledger.json").read_text())
    assert ledger["attempted_steps"] == 3
    assert ledger["status"] == "failed"


def test_decomposition_gate_stops_when_readout_explains_all_false_safes():
    rows = [{"cmin_dense_health": -1., "cmin_endpoint_health": -1.,
             "cmin_real_readout_health": 1., "cmin_imagined_health": 1.} for _ in range(12)]
    result = decomposition_gate(rows, ["health"])
    assert result["go"] is False
    assert result["status"] == "diagnostic_stop"
    assert result["evidence"]["health"]["n_imagination"] == 0
    for row in rows[:4]:
        row["cmin_real_readout_health"] = -1.
    result = decomposition_gate(rows, ["health"])
    assert result["go"] is True
    assert result["repair_rules"] == ["health"]


def test_decomposition_gate_requires_counts_not_just_share():
    rows = [{"cmin_dense_health": -1., "cmin_endpoint_health": -1.,
             "cmin_real_readout_health": -1., "cmin_imagined_health": 1.}]
    result = decomposition_gate(rows, ["health"])
    assert result["evidence"]["health"]["imagination_share"] == 1.
    assert result["go"] is False
