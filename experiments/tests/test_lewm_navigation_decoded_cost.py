"""Synthetic decoded-score checks, executable only on a rented CUDA host."""

import collections
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.lewm_navigation_decoded_cost import DecodedCurrentTargetCost
from helpers.lewm_navigation_retrieval_v2 import FrozenGoalReadout

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Rented CUDA host required")


class FrozenStub(torch.nn.Module):
    def __init__(self, endpoint, *, fail=False, dtype=torch.float32):
        super().__init__()
        self.endpoint, self.fail, self.dtype = endpoint, fail, dtype
        self.cost_calls, self.encode_calls = 0, 0

    def encode(self, info):
        self.encode_calls += 1
        return info

    def get_cost(self, info, candidates):
        self.cost_calls += 1
        if self.fail:
            raise RuntimeError("synthetic predictor failure")
        initial = torch.zeros_like(self.endpoint)
        info["predicted_emb"] = torch.stack([initial, self.endpoint], dim=-2)
        target = info["goal_emb"][..., -1, :]
        while target.ndim < self.endpoint.ndim:
            target = target.unsqueeze(-2)
        return (self.endpoint - target).square().sum(-1).to(self.dtype)


def readout():
    return FrozenGoalReadout(
        np.array([[1, 0], [0, 1], [0, 0]], dtype=np.float64),
        np.array([0.1, -0.2]),
        fingerprint="synthetic-qualified-readout",
    )


def fixture(*, fail=False):
    endpoint = torch.tensor([[[1, 0, 100], [10, 0, 0]]], device="cuda", dtype=torch.float32)
    original = FrozenStub(endpoint, fail=fail)
    counters = collections.Counter()
    decoder = readout()
    model = DecodedCurrentTargetCost(original, decoder, counters)
    info = {"goal_emb": endpoint.new_zeros(1, 1, 3)}
    candidates = endpoint.new_zeros(1, 2, 2, 10)
    return model, original, decoder, counters, info, candidates


def test_decoded_metric_can_reorder_the_raw_terminal_latent_metric():
    model, original, decoder, c, info, candidates = fixture()
    score = model.get_cost(info, candidates)
    assert score.argmin().item() == 0
    assert np.argmin(model.history[0]["original_task_cost"]) == 1
    torch.testing.assert_close(score, score.new_tensor([[1, 100]]), atol=0, rtol=0)
    assert original.cost_calls == 1 and original.encode_calls == 0 and decoder.calls == 2
    assert c["decoded_task_candidate_rows_attempted"] == 2


def test_cost_uses_current_selected_target_and_ignores_an_original_goal_side_channel():
    model, original, _, _, info, candidates = fixture()
    first = model.get_cost(info, candidates)
    info["goal_emb"][0, 0, 0] = 10
    info["original_goal_emb"] = info["goal_emb"].new_tensor([[[-100, -100, -100]]])
    second = model.get_cost(info, candidates)
    assert first.argmin().item() == 0 and second.argmin().item() == 1
    torch.testing.assert_close(second, second.new_tensor([[81, 0]]), atol=0, rtol=0)
    assert original.cost_calls == 2


def test_exact_one_wrapped_cost_call_encode_passthrough_and_unchanged_candidates():
    model, original, _, c, info, candidates = fixture()
    before = candidates.clone()
    assert model.encode(info) is info and original.encode_calls == 1
    model.get_cost(info, candidates)
    assert original.cost_calls == c["decoded_task_cost_call_attempts"] == 1
    assert c["decoded_task_readout_call_attempts"] == 2
    torch.testing.assert_close(before, candidates, atol=0, rtol=0)


@pytest.mark.parametrize("sample_specific_goal", [False, True])
def test_projection_attempt_and_success_rows_include_broadcast_and_sample_targets(sample_specific_goal):
    endpoint = torch.arange(18, device="cuda", dtype=torch.float32).reshape(2, 3, 3)
    original, c = FrozenStub(endpoint), collections.Counter()
    model = DecodedCurrentTargetCost(original, readout(), c)
    goal_shape = (2, 3, 1, 3) if sample_specific_goal else (2, 1, 3)
    info = {"goal_emb": endpoint.new_zeros(goal_shape)}
    candidates = endpoint.new_zeros(2, 3, 2, 10)
    result = model.get_cost(info, candidates)
    targets = 6 if sample_specific_goal else 2
    assert result.shape == (2, 3)
    assert c["decoded_task_endpoint_rows_attempted"] == 6
    assert c["decoded_task_target_rows_attempted"] == targets
    assert c["decoded_task_readout_rows_attempted"] == 6 + targets
    assert c["decoded_task_readout_rows_successful"] == 6 + targets
    assert c["decoded_task_candidate_rows_successful"] == 6


def test_projection_is_float64_and_score_preserves_wrapped_dtype():
    model, _, _, _, info, candidates = fixture()
    result = model.get_cost(info, candidates)
    assert result.dtype == torch.float32
    assert model.history[0]["projection_dtype"] == "torch.float64"
    assert model.history[0]["score_dtype"] == "torch.float32"


def test_original_predictor_failure_keeps_attempt_rows_without_projecting_or_replaying():
    model, original, decoder, c, info, candidates = fixture(fail=True)
    with pytest.raises(RuntimeError, match="predictor failure"):
        model.get_cost(info, candidates)
    assert original.cost_calls == 1 and decoder.calls == 0
    assert c["decoded_task_cost_call_attempts"] == c["decoded_task_cost_call_failures"] == 1
    assert c["decoded_task_candidate_rows_attempted"] == 2
    assert c["decoded_task_candidate_rows_successful"] == 0
    assert c["decoded_task_readout_call_attempts"] == 0


def test_projection_failure_retains_attempted_and_successful_rows_separately():
    model, original, decoder, c, info, candidates = fixture()
    original_readout = decoder.__call__

    class FailingSecondReadout:
        fingerprint = "synthetic-failing-readout"
        calls = 0

        def __call__(self, embedding):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("synthetic target projection failure")
            return original_readout(embedding)

    model.readout = FailingSecondReadout()
    with pytest.raises(RuntimeError, match="target projection failure"):
        model.get_cost(info, candidates)
    assert original.cost_calls == 1
    assert c["decoded_task_readout_rows_attempted"] == 3
    assert c["decoded_task_readout_rows_successful"] == 2
    assert c["decoded_task_readout_call_failures"] == 1
    assert c["decoded_task_cost_call_successes"] == 0


def test_v3_final_sidecars_preserve_initial_controller_and_retained_refusals(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_lewm_navigation_campaign_v3.py"
    spec = importlib.util.spec_from_file_location("navigation_v3_sidecar_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    initial = b'{"initial_controller":"unchanged"}\n'
    (tmp_path / "controller.json").write_bytes(initial)

    class Actor:
        refusals = [{"causal_context": {"recent_target_xy": [[1, 2]]}, "query": 0}]

        def artifact(self):
            return {"format": "synthetic-v2", "refusals": self.refusals}

    model, _, _, _, info, candidates = fixture()
    model.get_cost(info, candidates)
    module.persist_episode_sidecars(tmp_path, Actor(), model)
    assert (tmp_path / "controller.json").read_bytes() == initial
    assert json.loads((tmp_path / "refusal-history.json").read_text()) == Actor.refusals
    assert json.loads((tmp_path / "controller-final.json").read_text())["refusals"] == Actor.refusals
    assert json.loads((tmp_path / "decoded-task-cost.json").read_text())["history"]
