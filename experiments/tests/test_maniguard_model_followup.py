"""Causal data, global accounting, real model fitting and prospective control checks."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from helpers.maniguard_model_budget import WorkLedger
from helpers.maniguard_model_control import (
    ModelAssistedController,
    PersistentNoiseController,
    RosarlPenalty,
    SequenceProgress,
    decode_with_budget,
    filter_prefixes,
    mutate_candidates,
    select_candidates,
)
from helpers.maniguard_model_data import Transitions, freeze_manifest, load_manifest
from helpers.maniguard_model_transition import TransitionModel

ROOT = Path(__file__).resolve().parents[2]


def work(tmp_path, **overrides):
    limits = {"simulator_controls": 0, "physical_episode_starts": 0, "decoder_requests": 0,
              "encoder_images": 0, "training_row_uses": 100000,
              "model_head_row_queries": 100000, "candidate_screenings": 100}
    limits.update(overrides)
    return WorkLedger(tmp_path / "work.jsonl", limits, "synthetic-engineering-only")


def synthetic(episodes=8, seed=1, prefix="fit"):
    rng = np.random.default_rng(seed)
    n = episodes * 12
    current = np.zeros((n, 44))
    current[:, 0] = 1
    current[:, 1:8] = rng.uniform(-.2, .2, (n, 7))
    current[:, 20:25] = rng.integers(0, 2, (n, 5))
    actions = rng.uniform(-.2, .2, (n, 8, 8))
    following = current.copy()
    following[:, 1:8] += .1 * actions[:, :, :7].mean(axis=1)
    return Transitions(current=current, previous=current.copy(), history_valid=np.ones(n),
                       actions=actions, following=following, stages=following[:, 20:25].copy(),
                       risk=((current[:, 20] == 0) & (actions[:, 0, 0] > 0)).astype(float),
                       risk_eligible=np.ones(n, dtype=bool),
                       episode=np.asarray([f"{prefix}-{i // 12}" for i in range(n)]),
                       task=np.full(n, "synthetic-" + prefix), step=np.arange(n) * 8)


@pytest.fixture
def spec():
    return {"members": 3, "random_features": 0, "ridge": 1e-5, "seed": 7,
            "scale_floor": .05, "support_rows": 128, "support_radius": 2.}


@pytest.fixture
def gate():
    config = json.loads((ROOT / "configs/maniguard/model-assisted/offline-v1.json").read_text())
    return config["gate"]


def test_global_admission_is_not_reset_by_new_consumers_or_failed_returns(tmp_path):
    first = work(tmp_path, decoder_requests=1)
    second = work(tmp_path, decoder_requests=1)
    called = []

    def fail():
        called.append(True)
        raise RuntimeError("Mock decoder failed after admission")

    with pytest.raises(RuntimeError, match="Mock decoder"):
        decode_with_budget(fail, "one", first)
    with pytest.raises(RuntimeError, match="allowance"):
        decode_with_budget(lambda: called.append(False), "two", second)
    assert called == [True]
    assert second.summary()["used"]["decoder_requests"] == 1
    assert second.summary()["terminal_status"] == {"one": "failed"}


def test_concurrent_admission_preserves_last_global_slot(tmp_path):
    ledger = work(tmp_path, candidate_screenings=1)

    def admit(name):
        try:
            work(tmp_path, candidate_screenings=1).admit(name, "candidate_screenings", 1)
            return True
        except RuntimeError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(admit, ["first", "second"])) == 1
    assert ledger.summary()["used"]["candidate_screenings"] == 1


def test_duplicate_unreturned_work_and_incomplete_tails_are_not_replayed(tmp_path):
    ledger = work(tmp_path)
    ledger.admit("pending", "model_head_row_queries", 1)
    with pytest.raises(ValueError, match="already charged"):
        ledger.admit("pending", "model_head_row_queries", 1)
    assert ledger.summary()["unresolved"] == ["pending"]
    with ledger.path.open("a") as stream:
        stream.write('{"type":')
    with pytest.raises(ValueError, match="Incomplete ledger"):
        ledger.admit("next", "model_head_row_queries", 1)


def test_offline_caps_reject_model_service_access_before_calling_it(tmp_path):
    ledger, calls = work(tmp_path), []
    with pytest.raises(RuntimeError, match="decoder_requests"):
        decode_with_budget(lambda: calls.append(True), "forbidden", ledger)
    assert not calls
    assert ledger.summary()["used"]["decoder_requests"] == 0


def test_world_model_learns_executed_action_effect_and_round_trips(tmp_path, spec):
    # More independent trajectories than input degrees of freedom are needed for
    # this deliberately exact linear counterfactual, including bootstrap members.
    ledger, data = work(tmp_path), synthetic(episodes=24)
    model = TransitionModel.fit(data, spec, ledger, "fit")
    current = np.repeat(data.current[:1], 2, axis=0)
    actions = np.stack([np.full((8, 8), -.2), np.full((8, 8), .2)])
    prediction = model.predict(current, current, np.ones(2), actions, ledger, "counterfactual")
    assert prediction["next_mean"][1, 1] - prediction["next_mean"][0, 1] > .039
    assert np.isfinite(prediction["support_distance"]).all()
    artifact = tmp_path / "model.npz"
    model.save(artifact, corpus_sha256="synthetic")
    restored = TransitionModel.load(artifact)
    again = restored.predict(current, current, np.ones(2), actions, ledger, "restored")
    for key in prediction:
        np.testing.assert_array_equal(prediction[key], again[key])
    summary = ledger.summary()
    assert summary["used"]["training_row_uses"] == 3 * spec["members"] * len(data)
    assert summary["used"]["model_head_row_queries"] == 2 * 2 * spec["members"] * 3
    assert not summary["unresolved"]


def test_calibration_counts_whole_episodes_not_frames(tmp_path, spec, gate):
    ledger = work(tmp_path)
    model = TransitionModel.fit(synthetic(), spec, ledger, "fit")
    one = synthetic(episodes=1, prefix="calibration")
    result = model.calibrate(one, gate, ledger, "calibrate")
    assert result["episodes"] == 1 and not result["ready"]
    prediction = model.predict(one.current, one.previous, one.history_valid, one.actions,
                               ledger, "inspect")
    np.testing.assert_array_equal(model.risk_upper_score(prediction, gate), np.ones(len(one)))


def test_readout_failure_and_dynamics_failure_are_separate(tmp_path, spec, gate):
    ledger = work(tmp_path)
    model = TransitionModel.fit(synthetic(), spec, ledger, "fit")
    model.dynamics.coefficients[:] = 0  # Deliberately replace dynamics with persistence.
    diagnosis = model.audit(synthetic(3, 9, "diagnostic"), gate, ledger, "audit",
                            safe_completions=0)
    assert diagnosis["dynamics_blocks"]["robot_history"]["ratio"] == pytest.approx(1)
    assert max(diagnosis["actual_state_readout_mae"].values()) < .001
    assert not diagnosis["prediction_gate_pass"]
    assert any("robot_history" in message for message in diagnosis["failures"])
    assert any("safe task completion" in message for message in diagnosis["failures"])


def test_persistent_controller_has_gripper_authority_and_episode_reset():
    x, beliefs = np.zeros(44), np.zeros(5)
    x[0] = 1
    zero = PersistentNoiseController(np.zeros((5, 49)))
    assert not zero.offset(x, beliefs).any()
    weights = np.zeros((5, 49))
    weights[4, 1] = 1
    controller = PersistentNoiseController(weights, persistence=.8)
    x[1] = 1
    positive = controller.offset(x, beliefs)
    x[1] = -1
    persistent = controller.offset(x, beliefs)
    assert positive[0, 7] > 1 and persistent[0, 7] > 0
    assert not positive[:, :7].any() and not positive[:, 8:].any()
    controller.reset()
    assert controller.offset(x, beliefs)[0, 7] < -1


def test_lid_closure_is_not_robot_finger_closure_or_out_of_order_lifting():
    progress = SequenceProgress()
    result = progress.observe(closed=False, held=True, supported=True,
                              in_goal=False, violated=False)
    assert result["secondary_fitness"] == 0
    progress.observe(closed=False, held=True, supported=False, in_goal=False, violated=False)
    result = progress.observe(closed=True, held=True, supported=False, in_goal=True, violated=False)
    assert result["ordered_stage"] == 0
    valid = SequenceProgress()
    for supported, in_goal, expected in [(True, False, 2), (False, False, 3), (False, True, 4)]:
        result = valid.observe(closed=True, held=True, supported=supported,
                               in_goal=in_goal, violated=False)
        assert result["ordered_stage"] == expected
    assert valid.observe(closed=True, held=True, supported=False,
                         in_goal=True, violated=True)["secondary_fitness"] == 0


def test_rosarl_uses_external_value_estimates_and_freezes_shared_penalty():
    with pytest.raises(ValueError, match="source"):
        RosarlPenalty(value_source="predicted_risk")
    penalty = RosarlPenalty(value_source="synthetic_engineering_reference")
    penalty.observe(1., 2.)
    penalty.observe(0., -1.)
    with pytest.raises(ValueError, match="Freeze"):
        penalty.reward(1., False)
    assert penalty.freeze()["minmax_estimate"] == -3
    assert penalty.reward(1., True) == -3
    assert penalty.reward(1., False) == 1
    with pytest.raises(ValueError, match="frozen"):
        penalty.observe(0., -9.)


def predictions():
    return {"support_distance": np.zeros(3),
            "stage_mean": np.asarray([[0, 0, 0, 1, 1], [1, 0, 0, 1, 1], [1, 1, 1, 1, 0]])}


def test_filter_replaces_action_and_keeps_execution_unobserved_until_acknowledged():
    prefixes = np.zeros((3, 8, 8))
    prefixes[0, :, 0], prefixes[1, :, 7], prefixes[2, :, 0] = 1., -1., .2
    result = filter_prefixes(prefixes, predictions(), [.9, .01, .02], gate_pass=True)
    assert result.status == "replaced" and result.selected_index == 2
    assert result.executed is None
    np.testing.assert_array_equal(result.proposed, prefixes[0])
    np.testing.assert_array_equal(result.filtered, prefixes[2])
    result.record_executed(prefixes[2, :3])  # Native termination can shorten execution.
    assert len(result.executed) == 3
    with pytest.raises(ValueError, match="already"):
        result.record_executed(prefixes[2])


def test_filter_abstains_when_gate_or_supported_alternatives_are_missing():
    prefixes = np.zeros((3, 8, 8))
    for passed, upper in [(False, [0, 0, 0]), (True, [1, 1, 1])]:
        result = filter_prefixes(prefixes, predictions(), upper, gate_pass=passed)
        assert result.status == "abstained" and result.filtered is None and result.executed is None
        with pytest.raises(ValueError, match="No selected"):
            result.record_executed(prefixes[0])


def test_model_priority_never_becomes_physical_fitness_and_failed_gate_has_fallback(tmp_path):
    ledger = work(tmp_path)
    selected = select_candidates(["a", "b", "c"], predictions(), [.9, .01, .02], 1,
                                 ledger, "rank", gate_pass=True, penalty=-3.)
    assert selected["selected_ids"] == ["c"] and selected["physical_fitness"] is None
    fallback = select_candidates(["a", "b", "c"], predictions(), [.9, .01, .02], 1,
                                 ledger, "fallback", gate_pass=False, penalty=-3.)
    assert fallback["mode"] == "declared_model_free_allocation_fallback"
    assert ledger.summary()["used"]["candidate_screenings"] == 6
    pool = mutate_candidates(np.zeros((2, 5, 49)), 8, seed=9)
    assert np.shape(pool) == (8, 5, 49) and np.max(np.abs(pool)) <= 1


def test_combined_model_filter_reward_and_allocation_reject_unqualified_model(tmp_path, spec, gate):
    ledger, data = work(tmp_path), synthetic(episodes=24)
    model = TransitionModel.fit(data, spec, ledger, "fit")
    model.calibrate(synthetic(1, 2, "calibration"), gate, ledger, "calibrate")
    qualification = model.audit(synthetic(3, 3, "diagnostic"), gate, ledger,
                                 "qualify", safe_completions=0)
    penalty = RosarlPenalty(value_source="synthetic_engineering_reference")
    penalty.observe(1., 1.)
    penalty.observe(0., -1.)
    penalty.freeze()
    pipeline = ModelAssistedController(model, gate, qualification, penalty)
    result = pipeline.screen(data.current[0], data.previous[0], True, data.actions[:3],
                              ["native", "candidate-1", "candidate-2"], 1, ledger, "screen")
    assert result["decision"].status == "abstained"
    assert result["decision"].executed is None and result["physical_fitness"] is None
    assert result["allocation"]["mode"] == "declared_model_free_allocation_fallback"
    assert result["rosarl"]["minmax_estimate"] == -2
    assert ledger.summary()["used"]["simulator_controls"] == 0
    assert ledger.summary()["used"]["decoder_requests"] == 0


def record_episode(root, name, task, *, first_doom=None):
    directory = root / name
    directory.mkdir()
    job = {"family": "jar", "scene": task + "/base"}
    (directory / "job.json").write_text(json.dumps(job))
    receipt = {"valid": True, "full_episode": True, "physical": {"steps": 24},
               "outcome": {"S_req": False}}
    (directory / "verified-receipt.json").write_text(json.dumps(receipt))
    np.savez(directory / "controller.npz", weights=np.zeros((4, 44)))
    rows = []
    for step in range(25):
        rows.append({"type": "observed", "step": step, "monitor": {"valid": True,
                     "observation": {"doomed": first_doom is not None and step >= first_doom,
                                     "ap": {"jar_closed": False, "jar_upright": True,
                                            "jar_on_support": True}}}})
        if step:
            rows.append({"type": "goal", "step": step,
                         "detail": {"held": False, "intersects": False}})
        if step < 24:
            rows.append({"type": "executed", "step": step, "action": [step] * 8,
                         "action_scope": "environment_input_before_controller_output_clipping"})
            if step % 8 == 0:
                rows.append({"type": "latent_features", "step": step, "features": [step / 24] * 44})
    (directory / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {"id": name, "stage": "development", "status": "completed", "output": str(directory)}


def frozen_corpus(tmp_path):
    root = tmp_path / "recordings"
    root.mkdir()
    attempts = [record_episode(root, "fit", "task_fit", first_doom=4),
                record_episode(root, "cal", "task_cal"), record_episode(root, "diag", "task_diag")]
    attempts.append({"id": "protected", "stage": "independent_audit", "status": "completed",
                     "output": str(tmp_path / "must-never-open")})
    (root / "queue.json").write_text(json.dumps({"attempts": attempts}))
    config = json.loads((ROOT / "configs/maniguard/model-assisted/offline-v1.json").read_text())
    config["partitions"] = {"fit": ["task_fit"], "calibration": ["task_cal"],
                            "diagnostic": ["task_diag"]}
    config["allowed_stages_by_split"] = {split: ["development"] for split in config["partitions"]}
    manifest = tmp_path / "manifest.json"
    freeze_manifest(root, config, manifest)
    return manifest


def test_data_uses_executed_prefix_and_excludes_protected_outputs_before_opening(tmp_path):
    path = frozen_corpus(tmp_path)
    manifest, splits = load_manifest(path)
    assert len(manifest["entries"]) == 3
    data = splits["fit"]
    assert len(data) == 2 and data.inputs().shape == (2, 153)
    np.testing.assert_array_equal(data.actions[0, :, 0], np.arange(8))
    assert data.inputs()[0, 0] == 0 and data.following[0, 0] == pytest.approx(1 / 3)
    np.testing.assert_array_equal(data.risk_eligible, [True, False])
    np.testing.assert_array_equal(data.risk, [True, True])


def test_changed_corpus_bytes_or_protected_manifest_stage_are_rejected(tmp_path):
    path = frozen_corpus(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["entries"][0]["stage"] = "independent_audit"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Independent checking"):
        load_manifest(path)
    manifest["entries"][0]["stage"] = "development"
    path.write_text(json.dumps(manifest))
    with (Path(manifest["entries"][0]["output"]) / "trace.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed"):
        load_manifest(path)
