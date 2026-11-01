"""Safety contracts: dense labels, leakage, calibration, privilege separation and CEM wiring."""

import collections
from dataclasses import replace
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_worldmodel.policy import PlanConfig

from helpers.lewm18_planner import CostModel
from helpers.pusht_smoke import physical_clearance
from helpers.safety_checker import LearnedSafetyChecker, SafetyPlanningCEM
from helpers.safety_data import source_partitions, trace_arrays
from helpers.safety_model import (
    ClearanceModel,
    Constraint,
    SafetySpec,
    calibrate,
    chunk_features,
    load_model,
    pusht_constraints,
    save_model,
)


def spec(**changes):
    values = dict(
        environment="pusht",
        latent_dim=2,
        action_dim=2,
        action_block=5,
        max_horizon=3,
        length_scale=512,
        units="px",
        protected_body="pusher_disk",
        body_radius=15,
        coordinate_frame="arena_xy",
        world_model_sha256="0" * 64,
        action_scaler_sha256="1" * 64,
    )
    return SafetySpec(**{**values, **changes})


def constant_model(clearance=20):
    model = ClearanceModel(spec(), hidden=(8,))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[-1].bias.fill_(clearance / model.spec.length_scale)
    return model


@pytest.mark.parametrize("kind", ["island", "bar", "gate"])
@pytest.mark.parametrize("reflection", [1, -1])
def test_primitive_compilation_matches_existing_full_body_geometry(kind, reflection):
    angle = 0.4
    basis = [
        [np.cos(angle), np.sin(angle)],
        [-reflection * np.sin(angle), reflection * np.cos(angle)],
    ]
    case = dict(
        kind=kind,
        origin=[213, 240],
        basis=basis,
        radius_px=12,
        segment_half_length_px=30,
        gap_offset_px=40,
        gap_width_px=100,
        extent_px=180,
        band_half_width_px=7,
    )
    points = torch.tensor(np.random.default_rng(7).uniform(0, 512, (200, 2)))
    primitives = pusht_constraints(case, include_arena=False)
    actual = torch.stack([x.clearance(points, 15) for x in primitives]).amin(0)
    torch.testing.assert_close(actual, physical_clearance(points, case))


def test_dense_chunk_label_detects_crossing_with_safe_endpoints(tmp_path):
    constraint = Constraint("circle", (0, 0), radius=1)
    path = torch.tensor([[-3.0, 0], [0.0, 0], [3.0, 0]])
    assert (constraint.clearance(path[[0, -1]]) > 0).all()
    assert constraint.clearance(path).min() == -1
    trace = tmp_path / "trace.npz"
    np.savez(
        trace,
        frames=np.zeros((3, 4, 4, 3), np.uint8),
        actions=np.zeros((2, 2)),
        dense=np.zeros((3, 2)),
    )
    with pytest.raises(ValueError, match="every physics substep"):
        trace_arrays(trace, {"position_columns": [0, 1], "substeps_per_action": 10}, spec())


def test_hazard_relabelings_cannot_cross_source_splits():
    with pytest.raises(ValueError, match="crosses splits"):
        source_partitions(
            [
                {"source_group": "original-episode-91", "split": "train"},
                {"source_group": "original-episode-91", "split": "test"},
            ]
        )


def test_features_mask_future_actions_and_detach_world_model():
    cfg = spec()
    actions = torch.zeros(1, 2, 3, 10, requires_grad=True)
    observed = torch.zeros(1, 2, requires_grad=True)
    predicted = torch.zeros(1, 2, 4, 2, requires_grad=True)
    args = (cfg, observed, observed, predicted)
    tail = (torch.zeros(1, 2), torch.ones(1), [Constraint("circle", (200, 200), radius=12)])
    original = chunk_features(*args, actions, *tail)
    changed = actions.detach().clone()
    changed[:, :, -1] = 100
    second = chunk_features(*args, changed, *tail)
    torch.testing.assert_close(original[:, :, :2], second[:, :, :2])
    assert not torch.equal(original[:, :, 2], second[:, :, 2])
    assert not original.requires_grad
    with pytest.raises(ValueError, match="horizon"):
        chunk_features(*args, torch.zeros(1, 2, 4, 10), *tail)


def test_calibration_counts_source_groups_not_relabelings():
    model = constant_model(0)
    x = torch.zeros(100, model.spec.input_dim)
    calibration = calibrate(model, x, torch.zeros(100), np.array(["same"] * 100), 0.1)
    assert not calibration["ready"]
    assert calibration["source_groups"] == 1
    assert torch.isinf(model.error_allowance)
    # One group has errors 2 and 10; the other has error 4. Group maxima determine q.
    calibration = calibrate(model, x[:3], torch.tensor([-2.0, -10, -4]), ["a", "a", "b"], 0.5)
    assert calibration["ready"] and calibration["allowance"] == 10


def test_checkpoint_roundtrip_and_identity_rejection(tmp_path):
    model = constant_model()
    path = tmp_path / "model.pt"
    save_model(path, model, {"purpose": "unit-test"})
    loaded, _ = load_model(path, expected_spec=model.spec)
    x = torch.ones(2, model.spec.input_dim)
    torch.testing.assert_close(model(x), loaded(x))
    with pytest.raises(ValueError, match="identity"):
        load_model(path, expected_spec=replace(model.spec, body_radius=0))
    with pytest.raises(FileExistsError):
        save_model(path, model, {})


def test_training_fits_a_head_without_normalizing_on_calibration_or_test():
    from scripts.train_safety_model import fit_model

    cfg = spec()
    features = np.zeros((16, cfg.input_dim), dtype=np.float32)
    features[12:] = 1000
    arrays = {
        "features": features,
        "targets": np.full(16, 20, dtype=np.float32),
        "groups": np.array(["training"] * 12 + ["cal-a", "cal-b", "test-a", "test-b"]),
        "splits": np.array(["train"] * 12 + ["calibration"] * 2 + ["test"] * 2),
    }
    config = dict(
        seed=7,
        hidden=[8],
        epochs=20,
        learning_rate=0.01,
        batch_size=12,
        boundary_weight=4.0,
        boundary_band=12.0,
        overprediction_weight=2.0,
        margin=6.0,
        alpha=0.5,
    )
    model, report = fit_model(arrays, cfg, config)
    assert torch.equal(model.feature_mean, torch.zeros(cfg.input_dim))
    assert report["history"][-1]["loss"] < report["history"][0]["loss"]
    assert report["metrics"]["train"]["examples"] == 12
    assert report["metrics"]["test"]["source_groups"] == 2
    assert model.calibration["source_groups"] == 2


class DummyWorld(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.rollout_calls = 0

    def encode(self, info):
        return {"emb": torch.zeros(len(info["pixels"]), 1, 2)}

    def rollout(self, info, actions):
        self.rollout_calls += 1
        info["predicted_emb"] = torch.zeros(*actions.shape[:3], 2)
        info["predicted_emb"] = torch.cat(
            [info["predicted_emb"], info["predicted_emb"][:, :, :1]], dim=2
        )
        return info

    def get_cost(self, info, actions):
        self.rollout(info, actions)
        return actions.square().sum((-1, -2))


def bind(checker):
    info = {"pixels": torch.zeros(1, 1, 3, 2, 2), "action": torch.zeros(1, 1, 2)}
    checker.bind_context(info, torch.zeros(1, 1, 2), torch.zeros(1, 1, 2), torch.zeros(1))


def test_shadow_uses_reference_decisions_and_rechecks_new_candidates():
    world, model, counters = DummyWorld(), constant_model(), collections.Counter()
    reference = SimpleNamespace(state=None)
    reference.check = lambda a: (
        torch.ones(*a.shape[:2]),
        torch.zeros(*a.shape[:2], 1, 2),
        torch.zeros(*a.shape[:2]),
    )
    checker = LearnedSafetyChecker(
        model,
        world,
        [Constraint("circle", (100, 100), radius=12)],
        6,
        counters,
        reference=reference,
    )
    # Even a nonsensical reference state is never presented to the neural network.
    checker.state = np.array([np.nan] * 7)
    bind(checker)
    result = checker.check(torch.zeros(1, 3, 2, 10))[0]
    assert (result == 1).all()
    checker.check(torch.ones(1, 1, 2, 10))
    assert world.rollout_calls == 2
    assert counters["model_candidate_rollouts"] == 4
    assert counters["shadow_raw_false_safe"] == 4
    assert counters["shadow_conservative_false_safe"] == 0


def test_uncalibrated_model_cannot_control_execution():
    with pytest.raises(ValueError, match="held-out calibration"):
        LearnedSafetyChecker(constant_model(), DummyWorld(), [], 6, {}, mode="learned")


def test_actual_cem_uses_cached_imagination_and_rechecks_mean():
    counters, world, model = collections.Counter(), DummyWorld(), constant_model()
    model.calibration = {"ready": True}
    model.error_allowance.fill_(0)
    checker = LearnedSafetyChecker(
        model,
        world,
        [Constraint("circle", (100, 100), radius=12)],
        6,
        counters,
        mode="learned",
    )
    params = dict(
        horizon=2,
        iterations=2,
        budget_scale=0.03,
        variance=1,
        retain=0,
        checked=True,
        objective="learned",
        adaptive=False,
    )
    cost = CostModel(world, checker, params, counters)
    solver = SafetyPlanningCEM(cost, "pusht", params, 7, "cpu")
    solver.configure(
        action_space=gym.spaces.Box(-1.0, 1.0, (1, 2)),
        n_envs=1,
        config=PlanConfig(5, 1, action_block=5),
    )
    info = {
        "pixels": torch.zeros(1, 1, 3, 2, 2),
        "goal": torch.zeros(1, 1, 3, 2, 2),
        "pixels_previous": torch.zeros(1, 1, 3, 2, 2),
        "safety_history_valid": torch.zeros(1),
        "action": torch.zeros(1, 1, 2),
    }
    actions = solver.solve(info)["actions"]
    assert actions.shape == (1, 5, 10)
    # Two search batches, a fresh check of the mean, then the expected-next-state query.
    assert world.rollout_calls == 4
    assert counters["model_batch_calls"] == 4
    assert counters["encoder_frames"] == 3
    assert solver.replans[-1]["executed_violation"] == 0
