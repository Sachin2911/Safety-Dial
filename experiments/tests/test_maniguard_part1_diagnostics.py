"""Synthetic tests for post hoc analysis alignment, never experimental evidence."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.maniguard_part1_diagnostics import (
    before_action_held,
    episode_diagnostics,
    first_violation,
    index_trace,
    initial_pair,
)


def test_pre_action_held_does_not_use_action_outcome():
    trace = {"goal": {1: {"detail": {"held": False}}, 2: {"detail": {"held": True}}}}
    assert before_action_held(trace, 0) is None
    assert before_action_held(trace, 1) is False
    assert before_action_held(trace, 2) is True


@pytest.mark.parametrize("observed,number", [(13, 1), (24, 12)])
def test_violation_query_age_aligns_to_preceding_action(observed, number):
    proposal = {
        "original": [0] * 8,
        "adapted": [0] * 8,
        "frame": {"fingers": [0.02, 0.02]},
        "event": {
            "effective_horizon": 12,
            "grasp_evidence": False,
            "cap_active": False,
            "step_limit": 0.04,
        },
    }
    trace = {
        "query": {0: {}, 12: {}},
        "proposal": {k: proposal for k in range(24)},
        "observed": {
            observed: {
                "monitor": {
                    "observation": {
                        "ap": {
                            "container_upright": False,
                            "obstacles_upright": True,
                            "liquid_spilled": False,
                        }
                    }
                }
            }
        },
        "goal": {k: {"detail": {"held": True}} for k in range(1, 25)},
    }
    result = first_violation(trace, {"ltl_violated": True, "ltl_violation_step": observed})
    assert result["action_number_in_prefix"] == number
    assert result["action_step"] == observed - 1
    assert result["failed_propositions"] == ["container_upright"]


def test_trace_requires_complete_post_action_evidence(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({"type": "proposal", "step": 0}) + "\n")
    with pytest.raises(ValueError, match="executed"):
        index_trace(path)


def test_trace_rejects_duplicate_records(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text((json.dumps({"type": "query", "step": 0}) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate query"):
        index_trace(path)


def test_held_coverage_excludes_initial_and_opening_actions():
    def proposal(gripper, grasp):
        return {
            "original": [0] * 7 + [gripper],
            "adapted": [0] * 7 + [gripper],
            "frame": {"joints": [0] * 7, "fingers": [0.04, 0.04]},
            "event": {"grasp_evidence": grasp, "cap_active": False},
        }

    trace = {
        "proposal": {0: proposal(-1, False), 1: proposal(1, False), 2: proposal(-1, False)},
        "goal": {k: {"detail": {"held": True}} for k in (1, 2)},
    }
    result = episode_diagnostics(trace, {"ltl_violated": False})["counts"]
    assert result["initial_held_unknown_actions"] == 1
    assert result["held_1_heuristic_0"] == 2
    assert result["held_closing_actions"] == 1
    assert result["held_closing_outside_finger_band"] == 1


def test_initial_policy_differences_are_measured_before_adaptation():
    def trace(joint):
        action = [joint] + [0] * 7
        return {
            "query": {0: {"state": [0] * 8, "image_hashes": {"camera": str(joint)}}},
            "chunk": {0: {"actions": [action]}},
            "proposal": {0: {"original": action, "adapted": action}},
        }

    result = initial_pair(trace(0), trace(0.01))
    assert result["state_identical"]
    assert not result["image_hashes_identical"]
    assert result["first_action_differs_over_tolerance"]
    assert not result["either_first_action_adapted_over_tolerance"]
