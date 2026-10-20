"""Numerical and evidence-boundary checks for the independent reviewer tool."""

import json

import numpy as np
import pytest

from helpers.maniguard_review_sensitivity import analyze, candidates, sensitivity, trace_data


def test_bias_and_saturated_conditioning_have_distinct_sensitivity():
    features = np.zeros((2, 44), np.float32)
    features[:, 0] = 1
    features[:, 1:20] = [[1] * 19, [.9] * 19]
    weights = np.zeros((4, 44), np.float32)
    weights[0, 0] = np.arctanh(.5)
    constant = sensitivity(features, weights, weights)
    assert constant["code_mean"][0] == pytest.approx(.5)
    assert constant["blocks"]["robot"]["code_rms_change_on_block_removal"] == 0
    assert constant["parent_comparison"]["same_genotype"]
    weights[:, 1:20] = 1
    saturated = sensitivity(features, weights)
    assert saturated["saturated_099_fraction"] == 1
    assert saturated["mean_tanh_derivative"] == 0
    assert saturated["blocks"]["robot"]["code_rms_change_on_block_removal"] > .5


def test_parent_comparison_uses_the_same_factual_inputs():
    features = np.zeros((3, 44), np.float32)
    features[:, 0] = 1
    parent, child = np.zeros((4, 44), np.float32), np.zeros((4, 44), np.float32)
    child[0, 0] = np.arctanh(.5)
    delta = sensitivity(features, child, parent)["parent_comparison"]
    assert delta["code_rms_change"] == pytest.approx(.25)
    assert delta["arm_noise_offset_rms_change"] == pytest.approx(.35 * .5 / np.sqrt(7))
    assert not delta["same_genotype"]


def test_audit_outputs_are_never_opened(tmp_path):
    (tmp_path / "search").mkdir()
    weights = np.zeros((4, 44), np.float32).tolist()
    (tmp_path / "search/state.json").write_text(json.dumps({"cohorts": {
        "initial": [{"id": "initial-0", "weights": weights, "parent": None}]}}))
    (tmp_path / "queue.json").write_text(json.dumps({"attempts": [{
        "id": "audit", "stage": "independent_audit", "status": "completed",
        "output": str(tmp_path / "protected_nonexistent_directory")}]}))
    with pytest.raises(ValueError, match="No completed full development"):
        analyze(tmp_path)


def test_unavailable_parent_and_nonfinite_features_are_rejected():
    with pytest.raises(ValueError, match="parent is missing"):
        candidates({"cohorts": {"g1": [{"id": "child", "parent": "missing",
                                        "weights": np.zeros((4, 44)).tolist()}]}})
    with pytest.raises(ValueError, match="finite"):
        sensitivity(np.full((1, 44), np.nan), np.zeros((4, 44)))


def test_missing_control_observations_are_not_treated_as_absent_grasps(tmp_path):
    rows = [{"type": "latent_features", "step": 0, "features": [1] + [0] * 43},
            {"type": "observed", "step": 0, "monitor": {"valid": True,
                "observation": {"ap": {"jar_closed": False}, "doomed": False}}},
            {"type": "goal", "step": 2, "detail": {"held": False}}]
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="Incomplete or unordered"):
        trace_data(path)
