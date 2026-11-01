"""Prospective controller checks, executable only on the rented GPU host."""

import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.lewm16_planner import NoSafeAction
from helpers.lewm_navigation_retrieval_v2 import (
    FrozenGoalReadout,
    RetrievalController,
    RetrievalMemory,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Rented GPU required")


def memory():
    points = np.array([[42, 40], [46, 40], [60, 40], [70, 40], [71, 40]], dtype=np.float32)
    count = len(points)
    return RetrievalMemory(
        {
            "source_embeddings": np.tile([40, 40], (count, 1)).astype(np.float32),
            "successor_embeddings": points.copy(),
            "endpoint_embeddings": np.tile([100, 40], (count, 1)).astype(np.float32),
            "latent_scale": np.full(2, 100, dtype=np.float32),
            "source_xy": np.tile([40, 40], (count, 1)).astype(np.float32),
            "successor_xy": points,
            "endpoint_xy": np.tile([100, 40], (count, 1)).astype(np.float32),
            "source_episode": np.arange(128, 128 + count),
            "source_step": np.zeros(count, dtype=int),
            "successor_step": np.full(count, 5, dtype=int),
            "endpoint_step": np.full(count, 25, dtype=int),
        },
        fingerprint="test-observed-memory",
    )


def readout():
    return FrozenGoalReadout(np.eye(2), np.zeros(2), fingerprint="test-frozen-readout")


def controller(**kwargs):
    return RetrievalController(
        memory(),
        RetrievalController.fixed_ap_parameters(),
        kwargs.pop("shape", {"kind": "none"}),
        goal_readout=kwargs.pop("goal_readout", readout()),
        **kwargs,
    )


def query(goal=(100, 40)):
    return torch.tensor([40.0, 40.0], device="cuda"), torch.tensor(
        goal, dtype=torch.float32, device="cuda"
    )


def test_local_band_includes_boundaries_and_removes_reached_and_remote_targets():
    actor = controller()
    current, final = query()
    features, available = actor.features(current, final, [40, 40])
    assert available.tolist() == [False, True, True, True, False, False]
    assert features.shape == (6, 11)
    torch.testing.assert_close(features[1:4, 7], features.new_tensor([0.2, 2 / 3, 1.0]))
    assert actor.margin == 3.0 and actor.body_radius == 7.0


def test_original_safety_margin_and_allowance_still_mask_local_successors():
    shape = {"kind": "island", "centre": [60, 40], "radius": 3, "axis": [1, 0]}
    actor = controller(shape=shape, allowance=12.405431747436523)
    current, final = query()
    _, available = actor.features(current, final, [40, 40])
    assert not available[2] and not available[3]
    assert actor.artifact()["allowance"] == 12.405431747436523


def test_frozen_goal_progress_favors_moving_towards_estimate_without_true_goal_xy():
    actor = controller()
    current, final = query()
    features, _ = actor.features(current, final, [40, 40])
    assert features[3, 9] < features[1, 9] < 0
    assert actor._last_estimated_goal.tolist() == [100, 40]
    assert "true query-goal XY" in actor.artifact()["observations"]
    # The API receives visual latents and causal body XY; no final-position argument exists.
    with pytest.raises(TypeError, match="unexpected keyword"):
        actor.features(current, final, [40, 40], original_goal_xy=[5, 5])


def test_recent_and_stall_penalty_is_causal_scored_and_exactly_replayable():
    actor = controller()
    current, final = query()
    first = actor.select(current, final, [40, 40])
    first_index = actor.history[-1]["memory_index"]
    assert first_index is not None
    second = actor.select(current, final, [40, 40])
    row = actor.history[-1]
    assert float(actor.last_features[first_index, 8]) == pytest.approx(2.0)
    assert actor.last_available[first_index]  # Penalized, never made an arbitrary hard veto.
    replay = controller()
    rebuilt, admitted = replay.features(current, final, [40, 40], context=row["causal_context"])
    torch.testing.assert_close(rebuilt, actor.last_features, rtol=0, atol=0)
    assert torch.equal(admitted, actor.last_available)
    array = rebuilt.detach().cpu().contiguous().numpy()
    assert hashlib.sha256(array.tobytes()).hexdigest() == row["features_sha256"]
    assert any(torch.equal(first, value) for value in actor.memory.tables["successor_embeddings"])
    assert any(torch.equal(second, value) for value in actor.memory.tables["successor_embeddings"])
    assert len(actor.recent_targets) == 2 and actor.readout_queries == 2


def test_direct_original_goal_only_available_when_estimated_within_thirty_pixels():
    actor = controller()
    current, far = query()
    _, available = actor.features(current, far, [40, 40])
    assert not available[-1]
    _, near = query((60, 40))
    _, available = actor.features(current, near, [40, 40])
    assert available[-1]
    _, boundary = query((70, 40))
    _, available = actor.features(current, boundary, [40, 40])
    assert available[-1]
    with pytest.raises(NoSafeAction, match="No admitted local"):
        actor.select(current, far, [-100, -100])
    assert len(actor.refusals) == 1


def test_all_ten_same_family_genes_receive_ordinary_policy_gradient():
    features = torch.arange(44, dtype=torch.float32, device="cuda").reshape(4, 11) / 100
    features[-1, -1] = 2
    features[:3, -1] = 0
    available = torch.tensor([True, True, False, True], device="cuda")
    theta = torch.zeros(10, device="cuda", requires_grad=True)
    loss = -RetrievalController.log_probability(features, available, theta, 1)
    loss.backward()
    assert torch.isfinite(theta.grad).all() and bool((theta.grad.abs() > 1e-8).all())
    assert torch.isneginf(RetrievalController.logits(features, available, theta)[2])
    baseline = RetrievalController.fixed_ap_parameters()
    assert baseline.shape == (10,) and np.count_nonzero(baseline) == 10


def test_readout_npz_formula_and_float64_prediction_parity(tmp_path):
    coefficient = np.array([[0.1, 0.2], [0.3, -0.4]], dtype=np.float64)
    intercept = np.array([21.2, 39.8], dtype=np.float64)
    path = tmp_path / "readout.npz"
    np.savez_compressed(path, raw_feature_coefficient=coefficient, raw_feature_intercept=intercept)
    frozen = FrozenGoalReadout.from_npz(path)
    saved_embeddings = np.array([[0.123456789, 1.23456789], [40, 40]], dtype=np.float64)
    expected = saved_embeddings @ coefficient + intercept
    prediction = frozen(torch.as_tensor(saved_embeddings, device="cuda"))
    assert prediction.dtype == torch.float64 and frozen.calls == 1
    np.testing.assert_allclose(prediction.cpu().numpy(), expected, atol=1e-12, rtol=0)
    assert len(frozen.fingerprint) == 64


def test_qualified_cached_readout_predictions_match_online_float64_when_assets_present():
    root = Path(__file__).resolve().parents[2]
    readout_path = root / "results/navigation-readout-v1/readout.npz"
    embedding_path = root / "results/navigation-memory-cache-v3/observation-embeddings.npy"
    prediction_path = root / "results/navigation-readout-v1/predictions.npz"
    if not all(path.exists() for path in (readout_path, embedding_path, prediction_path)):
        pytest.skip("Qualified navigation cache is available on the navigation host only")
    frozen = FrozenGoalReadout.from_npz(readout_path)
    embeddings = np.load(embedding_path, allow_pickle=False)
    with np.load(prediction_path, allow_pickle=False) as saved:
        expected = saved["predicted_xy"].copy()
    actual = frozen(torch.as_tensor(embeddings, device="cuda")).cpu().numpy()
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=0)


def test_readout_architecture_and_unknown_settings_require_explicit_contract():
    with pytest.raises(ValueError, match="requires a readout"):
        controller(goal_readout=None)
    with pytest.raises(ValueError, match="Unknown retrieval-v2"):
        controller(controller_config={"oracle_goal": True})
    actor = controller(
        goal_readout=None,
        controller_config={"readout_progress": False, "direct_goal_locality": False},
    )
    current, final = query()
    features, available = actor.features(current, final, [40, 40])
    assert bool((features[:, 9] == 0).all()) and available[-1]
