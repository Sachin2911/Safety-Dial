"""Strict same-experience S5 regressions without simulator or GPU work."""
import copy
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments/scripts"))

from helpers.predictorAdapt import AdaptConfig, ClipSet, adapt  # noqa: E402
from helpers.walkerExperience import adapt_exact_experience, exact_branch_clips, verified_experience, verify_set_b_rows  # noqa: E402


def experience_fixture(tmp_path):
    bank = tmp_path / "bank"
    bank.mkdir()
    path = bank / "random_s0_b1"
    path.mkdir()
    source = {"sha256": "source"}
    qp = np.arange(101, dtype=np.float64)[:, None] + np.arange(9)[None, :]
    qv = qp + 1
    tape = np.repeat(np.arange(100, dtype=np.float32)[:, None], 6, axis=1)
    root = {"root_id": "r0", "episode": 2, "step": 20, "qpos": qp[0].tolist(), "qvel": qv[0].tolist(),
        "history_qpos": np.full((3, 9), -777.).tolist(), "history_qvel": np.full((3, 9), -888.).tolist(),
        "history_actions": np.full((2, 10, 6), -999.).tolist()}
    candidate = {"id": "selected", "root_index": 0, "kind": "random", "params": {}, "tape": tape.tolist()}
    selection = {"selected_candidates": [candidate], "selection_complete": True, "seed": 0,
        "seed_candidates": [{"id": "paid-seed"}], "horizon_steps": 100, "branch_steps": 100,
        "charged_steps": 300, "source_identity": source}
    report = {"planned_budgets": [1], "source_identity": source,
        "adaptation": {"random": {"0": [{"selected_ids": ["selected"], "budget": 1,
            "acquired_steps": 100, "charged_steps": 300}]}}}
    for filename, value in (("random_s0_selection.json", selection),
            ("acquisition_roots.json", {"roots": [root]}),
            ("candidates_s0.json", {"source_identity": source, "candidates": [candidate]})):
        (bank / filename).write_text(json.dumps(value))
    meta = {"roots": [root], "meta": {"selected_ids": ["selected"], "executed_steps": 100}}
    (path / "roots.json").write_text(json.dumps(meta))
    (path / "execution_ledger.json").write_text(json.dumps({"status": "complete", "selected_ids": ["selected"], "attempted_steps": 100}))
    with h5py.File(path / "branches.h5", "w") as f:
        f["qpos"], f["qvel"], f["tape"] = qp[None], qv[None], tape[None]
        f["x_velocity"], f["root_index"] = np.arange(100, dtype=np.float64)[None], [0]
    data_b = tmp_path / "B"
    data_b.mkdir()
    (data_b / "origins.json").write_text(json.dumps([{"candidate_id": "selected", "episode": 0, "role": "training"}]))
    with h5py.File(data_b / "setB.h5", "w") as f:
        f["qpos"], f["qvel"], f["action"] = qp[:-1], qv[:-1], tape
        f["x_velocity"], f["ep_offset"], f["ep_len"] = np.arange(100, dtype=np.float64), [0], [100]
    return bank, report, data_b


def test_recorded_root_tape_ledger_and_B_transition_identity(tmp_path):
    bank, report, data_b = experience_fixture(tmp_path)
    selection, locations = verified_experience(bank, report, "random", 0)
    identity = verify_set_b_rows(data_b, selection, locations)
    assert identity["n_identical_transitions"] == 100
    with h5py.File(data_b / "setB.h5", "r+") as f:
        f["qpos"][42, 1] += 0.1
    with pytest.raises(AssertionError, match="Set B and A-random"):
        verify_set_b_rows(data_b, selection, locations)


def test_seed_leakage_and_missing_execution_charge_fail_closed(tmp_path):
    bank, report, _ = experience_fixture(tmp_path)
    path = bank / "random_s0_selection.json"
    selection = json.loads(path.read_text())
    selection["seed_candidates"] += selection["selected_candidates"]
    path.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="common seed"):
        verified_experience(bank, report, "random", 0)
    selection["seed_candidates"].pop()
    path.write_text(json.dumps(selection))
    ledger = bank / "random_s0_b1/execution_ledger.json"
    value = json.loads(ledger.read_text())
    value["attempted_steps"] = 99
    ledger.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="mischarged"):
        verified_experience(bank, report, "random", 0)


def test_exact_clips_exclude_root_history_and_unstored_terminal_state(tmp_path):
    bank, report, _ = experience_fixture(tmp_path)
    selection, locations = verified_experience(bank, report, "random", 0)
    class Context:
        def render_many(self, qp, qv):
            assert len(qp) == 100
            return qp
    class Imaginer:
        def encode(self, frames):
            return torch.from_numpy(frames)
    clips = exact_branch_clips(selection, locations, Imaginer(), Context(), (np.zeros(6), np.ones(6)))
    assert clips.latents.shape == (70, 4, 9)
    assert clips.actions.shape == (70, 3, 60)
    np.testing.assert_array_equal(clips.latents[0, :, 0], [0, 10, 20, 30])
    np.testing.assert_array_equal(clips.latents[-1, :, 0], [69, 79, 89, 99])
    assert clips.meta["n_transitions"] == 100
    assert clips.meta["seed_steps"] == clips.meta["history_prefix_steps"] == 0
    assert clips.latents.min() >= 0 and clips.actions.min() >= 0


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 3)
        self.obs_proj = torch.nn.Linear(3, 3)
        self.action_encoder = torch.nn.Linear(2, 3)
        self.predictor = torch.nn.Linear(3, 3)
        self.pred_proj = torch.nn.Linear(3, 3)
    def predict(self, z, a):
        return self.pred_proj(self.predictor(z + a))


def toy_clips(n=5, t=4):
    rng = np.random.default_rng(t)
    return ClipSet(rng.normal(size=(n, t, 3)).astype(np.float32),
                   rng.normal(size=(n, t - 1, 2)).astype(np.float32), {})


def test_exact_adapter_matches_S4_optimizer_when_clip_lengths_match():
    torch.manual_seed(0)
    model = TinyModel()
    base = copy.deepcopy(model.state_dict())
    cfg = AdaptConfig(ctx=3, seed=7, batch_size=4, steps=3)
    acquired, replay = toy_clips(7), toy_clips(6)
    original, _ = adapt(base, model, acquired, replay, cfg, device="cpu", verbose=False)
    exact, log = adapt_exact_experience(base, model, acquired, replay, cfg, device="cpu")
    for name in base:
        torch.testing.assert_close(original.state_dict()[name], exact.state_dict()[name], atol=1e-7, rtol=1e-6)
    assert log["cfg"] == cfg.to_dict()


def test_exact_adapter_preserves_frozen_weights_and_unequal_original_replay():
    torch.manual_seed(0)
    model = TinyModel()
    base = copy.deepcopy(model.state_dict())
    replay = toy_clips(t=13)
    replay_before = replay.latents.copy()
    result, log = adapt_exact_experience(base, model, toy_clips(t=4), replay,
        AdaptConfig(ctx=3, seed=7, batch_size=4, steps=2), device="cpu")
    for name in base:
        if name.startswith(("encoder.", "obs_proj.")):
            torch.testing.assert_close(base[name], result.state_dict()[name], atol=0, rtol=0)
    assert not torch.equal(base["predictor.weight"], result.predictor.weight)
    np.testing.assert_array_equal(replay.latents, replay_before)
    assert log["steps"][-1]["step"] == 2


def test_S4_replay_only_encodes_declared_training_episodes(monkeypatch):
    import helpers.walkerBank as bank_module
    from walker_s4_study import replay_clips
    seen = []
    episodes = []
    for episode in range(3):
        episodes.append((episode, {"qpos": np.full((140, 9), episode, dtype=float),
            "qvel": np.zeros((140, 9)), "action": np.zeros((140, 6))}))
    monkeypatch.setattr(bank_module, "iter_episodes", lambda path: iter(episodes))
    class Context:
        def render_many(self, qp, qv):
            seen.append(int(qp[0, 0]))
            return qp
    class Imaginer:
        def encode(self, frames):
            return torch.from_numpy(frames)
    result = replay_clips(Path("unused"), Imaginer(), Context(), np.random.default_rng(0),
                          10, (np.zeros(6), np.ones(6)), episodes=[1])
    assert seen == [1, 1]
    assert len(result) == 2


def test_S5_ready_rejects_partial_S4_and_stale_model_receipt():
    from walker_s5_ready import assess_completion
    report = {"status": "complete", "acquisition_mode": "prospective", "development_gate": {"go": True},
        "planned_budgets": [1, 2], "acquisition_seeds": [0], "adaptation": {}}
    for arm in ("random", "boundary"):
        report["adaptation"][arm] = {"0": [{"budget": b, "selected_ids": list(range(b)), "charged_steps": 100 * b} for b in (1, 2)]}
    kwargs = dict(expected_steps=214780, expected_budget=2, expected_seeds=[0], seed=0)
    state, config, receipt = {"step": 214780}, {"total_steps": 214780}, {"step": 214780}
    assert assess_completion(report, state, config, receipt, **kwargs)["go"]
    receipt["step"] -= 1
    assert not assess_completion(report, state, config, receipt, **kwargs)["go"]
    receipt["step"] += 1
    report["adaptation"]["boundary"]["0"].pop()
    assert not assess_completion(report, state, config, receipt, **kwargs)["go"]
