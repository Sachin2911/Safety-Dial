"""Pure execution semantics; these fixtures are not manipulation evidence."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.maniguard_adaptation import Frame
from helpers.maniguard_phase2_adapter import Phase2Adapter
from helpers.maniguard_phase2_model import identity_genome, random_genome


def frame(q=None, **kwargs):
    return Frame(np.zeros(7) if q is None else np.asarray(q), np.array([0.01, 0.01]),
                 np.zeros(3), **kwargs)


def fixed(choices):
    return Phase2Adapter(identity_genome(), mode="fixed", fixed_choices=choices)


def apply(adapter, proposal, state=None):
    action, event = adapter.act(np.asarray(proposal), frame() if state is None else state)
    adapter.applied(action)
    return action, event


def test_identity_preserves_exact_dtype_values_and_does_not_mutate_inputs():
    rng = np.random.default_rng(234)
    adapter = Phase2Adapter(identity_genome())
    for step in range(20):
        if step % 4 == 0:
            record = adapter.decide(rng.normal(size=64), step)
            assert record["choices"] == [0, 0, 0, 0]
        proposal = rng.normal(size=8).astype(np.float32)
        saved = proposal.copy()
        action, event = apply(adapter, proposal, frame(q=rng.normal(size=7)))
        assert action.dtype == proposal.dtype
        np.testing.assert_array_equal(action, saved)
        np.testing.assert_array_equal(proposal, saved)
        assert not event["changed"]
        assert not event["guard_blocked"]


def test_motion_composition_uses_last_executed_then_current_measured_joint_cap():
    adapter = fixed([0, 1, 2, 0])  # cap .03, smoothing .5
    adapter.applied(np.r_[np.full(7, 0.04), -1.0])
    proposal = np.r_[np.full(7, 0.10), 1.0]
    actual, _ = adapter.act(proposal, frame(q=np.full(7, 0.02)))
    # Smooth .04 -> .10 to .07, then cap relative to .02 at .05.
    np.testing.assert_allclose(actual[:7], 0.05)
    assert actual[-1] == 1
    # The transformed proposal does not replace execution history.
    np.testing.assert_array_equal(adapter.last_executed[:7], np.full(7, 0.04))


def test_first_smoothing_reference_is_measured_state_and_applied_command_is_authoritative():
    adapter = fixed([0, 0, 1, 0])  # smoothing .75 only
    actual, _ = adapter.act(np.r_[np.ones(7), -1.0], frame(q=np.full(7, 0.2)))
    np.testing.assert_allclose(actual[:7], 0.8)
    adapter.applied(np.r_[np.full(7, 0.4), -1.0])  # lower-level input clipping fixture
    actual, _ = adapter.act(np.r_[np.ones(7), -1.0], frame(q=np.full(7, 0.2)))
    np.testing.assert_allclose(actual[:7], 0.85)


def test_release_budget_is_captured_once_and_cannot_renew_when_choices_change():
    adapter = fixed([0, 0, 0, 1])
    apply(adapter, np.r_[np.zeros(7), -1.0])
    outputs = []
    for index in range(7):
        if index:
            adapter.choices[-1] = 2  # four-action head cannot extend captured two-action budget
        action, event = apply(adapter, np.r_[np.zeros(7), 1.0])
        outputs.append(action[-1])
        assert event["release_budget"] == 2
    assert outputs == [-1, -1, 1, 1, 1, 1, 1]
    apply(adapter, np.r_[np.zeros(7), -1.0])
    outputs = [apply(adapter, np.r_[np.zeros(7), 1.0])[0][-1] for _ in range(6)]
    assert outputs == [-1, -1, -1, -1, 1, 1]


def test_identity_release_onset_stays_immediate_despite_later_delay_choices():
    adapter = fixed([0, 0, 0, 0])
    apply(adapter, np.r_[np.zeros(7), -1.0])
    assert apply(adapter, np.r_[np.zeros(7), 0.0])[0][-1] == 0
    adapter.choices[-1] = 2
    assert all(apply(adapter, np.r_[np.zeros(7), 1.0])[0][-1] == 1 for _ in range(8))


def test_release_requires_a_previous_executed_close_not_merely_a_proposal():
    adapter = fixed([0, 0, 0, 2])
    assert apply(adapter, np.r_[np.zeros(7), 1.0])[0][-1] == 1
    adapter.act(np.r_[np.zeros(7), -1.0], frame())  # not applied
    adapter.applied(np.r_[np.zeros(7), 1.0])
    assert apply(adapter, np.r_[np.zeros(7), 1.0])[0][-1] == 1


@pytest.mark.parametrize("choice,interval", [(0, 8), (1, 4), (2, 16)])
def test_query_schedule_counts_applied_actions_and_resets_only_on_query(choice, interval):
    adapter = fixed([choice, 0, 0, 0])
    adapter.on_query()
    for _ in range(interval - 1):
        adapter.applied(np.zeros(8))
        assert adapter.replan_reason() is None
    adapter.applied(np.zeros(8))
    assert adapter.replan_reason() is not None
    assert adapter.since_query == interval
    adapter.on_query()
    assert adapter.since_query == 0
    assert adapter.replan_reason() is None
    assert adapter.control_steps == interval


def test_contextual_reference_uses_current_measured_tracking_feature_and_no_task_labels():
    adapter = Phase2Adapter(identity_genome(), mode="contextual")
    features = np.zeros(64)
    features[12] = 0.51
    record = adapter.decide(features, 0)
    assert record["choices"] == [1, 0, 0, 0]
    assert record["log_prob"] is None and record["logits"] is None
    for _ in range(4):
        adapter.applied(np.zeros(8))
    features[12] = -0.5
    assert adapter.decide(features, 4)["choices"] == [0, 0, 0, 0]


def test_decision_cadence_history_and_reset_are_explicit():
    adapter = Phase2Adapter(random_genome(np.random.default_rng(19)))
    feature = np.ones(64)
    first = adapter.decide(feature, 0)
    with pytest.raises(ValueError, match="Decision step"):
        adapter.decide(feature, 0)
    with pytest.raises(ValueError, match="Decision step"):
        adapter.decide(feature, 1)
    with pytest.raises(ValueError, match="applied"):
        adapter.decide(feature, 4)
    for _ in range(4):
        apply(adapter, np.zeros(8))
    second = adapter.decide(feature, 4)
    assert second["hidden_before"] == first["hidden_after"]
    assert second["hidden_after"] != first["hidden_after"]
    json.dumps(second, allow_nan=False)
    adapter.reset()
    assert adapter.decide(feature, 0) == first
    assert adapter.last_executed is None and adapter.last_proposal is None
    assert adapter.opening_steps == 0 and adapter.since_query == 0


def test_stochastic_decisions_are_episode_local_and_reset_repeatably():
    first = Phase2Adapter(identity_genome(), stochastic=True, seed=24)
    second = Phase2Adapter(identity_genome(), stochastic=True, seed=24)
    feature = np.zeros(64)
    record = first.decide(feature, 0)
    assert second.decide(feature, 0) == record
    first.reset()
    assert first.decide(feature, 0) == record
    assert np.isfinite(record["log_prob"])


def test_privileged_labels_and_invalid_observations_cannot_enter_primary_controller():
    adapter = Phase2Adapter(identity_genome())
    for forbidden in ({"prerequisite": False}, {"target_engaged": False}):
        with pytest.raises(ValueError, match="privileged"):
            adapter.act(np.zeros(8), frame(**forbidden))
    with pytest.raises(ValueError):
        adapter.act(np.full(8, np.nan), frame())
    with pytest.raises(ValueError):
        adapter.applied(np.zeros(7))
    with pytest.raises(ValueError):
        adapter.decide(np.full(64, np.inf), 0)
    with pytest.raises(ValueError):
        fixed([0, 1, 2, 3])
    with pytest.raises(ValueError):
        Phase2Adapter(identity_genome(), mode="unknown")
