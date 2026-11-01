"""Regression checks for the unlaunched v3 improvements; no simulator calls."""

import json
from pathlib import Path

import numpy as np
import pytest

from maniguard_adapter import AdapterSpec, FEATURE_NAMES
from maniguard_evolution import Search
from maniguard_evolution_scoring import candidate_score, ordered_progress
from maniguard_evolution_v3 import mutation_scales, ordered_progress_v3, safe_target_contact


ROOT = Path(__file__).resolve().parents[2]


def row(primary=False, progress=0, contact=False, violation=False):
    return {"valid": True, "strict_safe_completion": primary,
            "benchmark_safe_completion": primary, "requested_completion": primary,
            "raw_violation": violation, "counted_violation": violation,
            "unfinished": not primary, "engaged": contact,
            "safe_stage_progress": progress, "safe_target_contact": contact}


def test_early_departure_cannot_receive_later_carry_credit():
    closed, supported, grasped = [True] * 11, [True] + [False] * 10, [True] * 11
    assert ordered_progress(closed, supported, grasped) == 3  # preserved v2 behavior
    assert ordered_progress_v3(closed, supported, grasped) == 2


def test_departure_after_sustained_grasp_receives_carry_credit():
    assert ordered_progress_v3([True] * 11, [True] * 10 + [False], [True] * 11) == 3


@pytest.mark.parametrize("violation,closure,departure,expected", [
    (False, None, None, True), (True, None, None, False),
    (False, None, 3, False), (False, 5, 3, False), (False, 2, 3, True),
])
def test_safe_contact_respects_whole_episode_safety_and_requested_prefix(
        violation, closure, departure, expected):
    assert safe_target_contact([0, 1], [True, True], [True, True], [False, False],
                               violation=violation, closure=closure,
                               departure=departure) is expected


def test_safe_contact_breaks_inactivity_ties_without_outweighing_completion():
    mode = "completion_progress_contact_v3"
    assert candidate_score([row(contact=True)], mode) > candidate_score([row()], mode)
    complete = [row(primary=True)] + [row(violation=True) for _ in range(5)]
    partial = [row(progress=3, contact=True) for _ in range(6)]
    assert candidate_score(complete, mode) > candidate_score(partial, mode)


def test_mutation_profile_reduces_arm_noise_and_preserves_gripper_exploration():
    spec, sigma = AdapterSpec(), 0.3
    actual = sigma * mutation_scales(spec, sigma, "conservative_arm_v3")
    h = spec.hidden_size
    offset = h * len(FEATURE_NAMES) + h * h + h
    output = actual[offset:offset + 9 * h].reshape(9, h)
    np.testing.assert_allclose(output[:7], 0.025)
    np.testing.assert_allclose(output[7], 0.3)
    assert actual[-2] == 0.3
    np.testing.assert_allclose(actual[:offset], 0.05)


@pytest.mark.parametrize("method", ["cmaes", "random"])
def test_native_anchor_is_a_real_population_member_for_both_optimizers(tmp_path, method):
    cfg = json.loads((ROOT / "configs/maniguard_evolution/development-v2.json").read_text())
    cfg["search"].update(method=method, generations=1, population=4, tasks=[0],
                         policy_seeds=[101], native_anchor=True,
                         mutation_profile="conservative_arm_v3",
                         objective="completion_progress_contact_v3")
    cfg["validation"].update(tasks=[3], policy_seeds=[201, 202, 203])
    artifacts = []

    def evaluate(_label, _task, _seed, artifact):
        artifacts.append(artifact)
        return row()

    search = Search(cfg, tmp_path, evaluate)
    state = search.run()
    assert state["status"] == "completed"
    assert any(np.array_equal(theta, state["initial"])
               for theta in state["generations"][0]["parameters"])
    assert all(a["progress_version"] == 3 for a in artifacts)
    assert len(artifacts) == 11
