"""Causality, first-event denominators, tied thresholds and whole-task isolation."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.maniguard_part4_data import causal_design, native_agreement, warning_labels
from helpers.maniguard_part4_model import calibration_rule, capped_windows, event_metrics
from helpers.maniguard_part4_protocol import assert_partitioning


def test_first_onset_no_post_event_or_padded_future():
    upright = np.ones(31, dtype=bool)
    upright[15:18] = False
    upright[24:] = False
    labels = warning_labels(upright, 30)
    assert labels["onset"] == 15
    assert labels["steps"].tolist() == list(range(4, 15))
    assert labels["steps"][labels["y"] == 1].tolist() == list(range(7, 15))
    safe = warning_labels(np.ones(31), 30)
    assert safe["steps"][-1] == 22  # t+8 == final real observation 30
    assert not safe["y"].any()
    assert not warning_labels(np.zeros(31), 30)["steps"].size
    with pytest.raises(ValueError):
        warning_labels(np.ones(30), 30)


def test_future_and_current_executed_actions_cannot_change_input():
    rng = np.random.default_rng(1)
    e = {"steps": np.array([4]), "visual": rng.normal(size=(10, 2)),
         "robot": rng.normal(size=(10, 9)), "proposed": rng.normal(size=(10, 8)),
         "action": rng.normal(size=(10, 8)), "controller": np.zeros(9),
         "query_age": np.arange(10)}
    pca = SimpleNamespace(n_components_=2, transform=lambda x: x)
    original = causal_design(e, pca)
    e["action"][4:] = 1e6
    for name in ("visual", "robot", "proposed"):
        e[name][5:] = 1e6
    assert np.array_equal(original, causal_design(e, pca))
    e["action"][3] += 1
    assert not np.array_equal(original, causal_design(e, pca))


def test_native_object_scopes_and_threshold_agreement():
    props = {"upright": {"state": "upright", "over": ["cup*"],
                         "params": {"max_tilt_deg": 15}}}
    trace = [{"type": "observed", "step": 0, "monitor": {
        "valid": True, "error": None, "observation": {"ap": {"upright": True}}}},
        {"type": "measurement", "step": 0, "protected_objects": [
            {"name": "cup_1", "threshold_deg": 15, "angle_deg": 15, "margin_deg": 0}]}]
    assert native_agreement(trace, props)[0].tolist() == [True]
    trace[1]["protected_objects"][0]["threshold_deg"] = 45
    with pytest.raises(ValueError, match="threshold"):
        native_agreement(trace, props)


def test_one_event_one_quality_and_late_alarm_is_missed():
    label = warning_labels(np.r_[np.ones(15), np.zeros(16)], 30)
    episode = {**label, "episode_id": "a", "task": "x"}
    p = np.zeros(len(label["steps"])) + .01
    p[label["steps"] == 14] = .9
    rule = {"threshold": .5, "low_risk_cutoff": .1, "disagreement_cutoff": .02}
    result = event_metrics(episode, p, np.zeros(len(p)), rule)
    assert result["eligible_event"] and result["late_warning"]
    assert not result["timely_warning"]
    assert result["confident_miss"] and result["quality"] == .99
    p[label["steps"] == 9] = .9
    assert event_metrics(episode, p, np.zeros(len(p)), rule)["quality"] == 0


def test_short_event_history_and_censored_deadline_are_explicit():
    rule = {"threshold": .5, "low_risk_cutoff": .1, "disagreement_cutoff": .02}
    for tau, n, reason in [(10, 30, "insufficient_pre_event_history"),
                           (25, 26, "incomplete_recorded_future_for_warning_window")]:
        label = warning_labels(np.r_[np.ones(tau), np.zeros(n + 1 - tau)], n)
        e = {**label, "episode_id": "a", "task": "x"}
        p = np.zeros(len(label["steps"]))
        assert event_metrics(e, p, p, rule)["event_exclusion"] == reason


def test_tied_calibration_scores_respect_false_alarm_budget():
    raw = np.ones(100) * .2
    y = np.r_[np.zeros(95), np.ones(5)]
    rule = calibration_rule(raw, y, np.ones(100), np.ones(100) * .01)
    assert rule["calibration_negative_window_false_alarm"] == 0
    assert rule["threshold"] > rule["probability"][0]


def test_window_cap_retains_positives_and_episode_mass():
    y = np.zeros(1000, dtype=int)
    y[500:508] = 1
    index, weight = capped_windows({"y": y})
    assert len(index) == 256
    assert y[index].sum() == 8
    assert np.isclose(weight.sum(), 1)
    assert np.isclose(weight[y[index] == 1].sum(), .008)


def test_whole_task_variants_cannot_cross_partition():
    with pytest.raises(ValueError, match="leaks"):
        assert_partitioning({"fit": ["task_0001/base"], "audit": ["task_0001/perturbed"]})
