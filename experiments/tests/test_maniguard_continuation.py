"""Synthetic CPU checks for prospective continuation controls; no physical results."""

import numpy as np
import pytest

from helpers.maniguard_continuation_allocation import FilteredCandidateAllocator
from helpers.maniguard_continuation_residual import BoundedActionResidual
from helpers.maniguard_model_budget import WorkLedger
from helpers.maniguard_model_control import RosarlPenalty


def work(tmp_path, **overrides):
    limits = {"candidate_screenings": 100, "model_head_row_queries": 1000,
              "simulator_controls": 0, "physical_episode_starts": 0,
              "decoder_requests": 0, "encoder_images": 0}
    limits.update(overrides)
    return WorkLedger(tmp_path / "work.jsonl", limits, "synthetic-continuation-checks-only")


def prefixes(values):
    result = np.zeros((len(values), 2, 8))
    result[:, :, 0] = np.asarray(values)[:, None]
    result[:, :, 7] = 1.
    return result


class PrefixModel:
    """Controlled counterexample: risky high-progress proposal versus useful safe action."""

    horizon = 2
    def __init__(self, *, ready=True, unsupported=False, fail=False):
        self.spec = {"support_radius": 2., "members": 2}
        self.calibration = {"ready": ready}
        self.unsupported, self.fail, self.calls = unsupported, fail, 0

    def predict(self, current, previous, history_valid, actions, ledger, request_id):
        self.calls += 1
        n = len(actions)
        ledger.admit(request_id, "model_head_row_queries", n * 3 * self.spec["members"])
        try:
            if self.fail:
                raise RuntimeError("Synthetic model failure")
            assert current.shape == previous.shape == (n, 44)
            assert history_valid.shape == (n,)
            q = actions[:, 0, 0]
            stages = np.zeros((n, 5))
            stages[:, 3:] = 1
            stages[q > .2, 0] = 1
            stages[q > .6, 1:3] = 1
            result = {"stage_mean": stages, "risk_mean": np.where(q > .6, .9, .01),
                      "support_distance": np.full(n, 3. if self.unsupported else 0.)}
        except Exception:
            ledger.finish(request_id, "failed")
            raise
        ledger.finish(request_id, "returned")
        return result

    @staticmethod
    def risk_upper_score(prediction, gate):
        return prediction["risk_mean"]


def allocator(model, *, qualified=True):
    penalty = RosarlPenalty(value_source="synthetic_engineering_reference")
    penalty.observe(1., -1.)
    penalty.freeze()
    assert penalty.penalty == -2.
    return FilteredCandidateAllocator(
        model, {"risk_score_limit": .05}, {"prediction_gate_pass": qualified}, penalty)


def screen(controller, ledger, proposals=None, alternatives=None, ids=None, slots=1,
           request="screen"):
    return controller.screen(
        np.zeros(44), np.zeros(44), False,
        prefixes([1., .4]) if proposals is None else proposals,
        prefixes([0.]) if alternatives is None else alternatives,
        ["risky", "useful"] if ids is None else ids, slots, ledger, request)


def test_filtering_reverses_raw_priority_before_candidate_selection(tmp_path):
    ledger = work(tmp_path)
    result = screen(allocator(PrefixModel()), ledger)
    raw = result["raw_proposal_priority_diagnostic"]
    priority = result["surrogate_priority"]
    assert raw["risky"] > raw["useful"]
    assert priority["risky"] < priority["useful"]
    assert result["selected_ids"] == ["useful"]
    assert result["decisions"]["risky"].status == "replaced"
    assert result["decisions"]["useful"].status == "accepted"
    assert result["mapping"]["risky"]["filtered_pool_index"] == 2
    assert result["physical_fitness"] is None and result["dispatch_authorized"] is False
    assert all(d.executed is None for d in result["decisions"].values())
    assert ledger.summary()["used"]["model_head_row_queries"] == 18
    assert ledger.summary()["used"]["simulator_controls"] == 0


def test_identical_prefixes_share_queries_without_equating_complete_controllers(tmp_path):
    ledger, model = work(tmp_path), PrefixModel()
    proposal = prefixes([.4, .4, .4])
    proposal[1, :, 1] = -0.
    result = screen(allocator(model), ledger, proposal, prefixes([.4, 0., 0.]),
                    ["c", "a", "b"], slots=2)
    assert result["unique_prediction_prefixes"] == 2 and model.calls == 1
    assert ledger.summary()["used"]["model_head_row_queries"] == 12
    assert ledger.summary()["used"]["candidate_screenings"] == 3
    assert result["selected_ids"] == ["a", "b"]
    assert result["prefix_equivalence_groups"] == [["c", "a", "b"]]
    with pytest.raises(ValueError, match="already charged"):
        screen(allocator(model), ledger, proposal, prefixes([]), ["c", "a", "b"])
    assert model.calls == 1


@pytest.mark.parametrize("qualified,ready", [(False, True), (True, False)])
def test_failed_qualification_skips_predictions_and_preserves_abstention(
        tmp_path, qualified, ready):
    ledger, model = work(tmp_path, model_head_row_queries=0), PrefixModel(ready=ready)
    result = screen(allocator(model, qualified=qualified), ledger)
    assert result["mode"] == "declared_model_free_allocation_fallback"
    assert result["prediction"] is None and result["unique_prediction_prefixes"] == 0
    assert model.calls == 0
    assert all(d.status == "abstained" for d in result["decisions"].values())
    assert all(v is None for v in result["surrogate_priority"].values())
    assert ledger.summary()["used"]["model_head_row_queries"] == 0
    assert ledger.summary()["used"]["candidate_screenings"] == 2


def test_unsupported_candidates_cannot_be_given_filtered_priority(tmp_path):
    ledger = work(tmp_path)
    result = screen(allocator(PrefixModel(unsupported=True)), ledger)
    assert result["mode"] == "declared_model_free_allocation_fallback"
    assert result["prefix_equivalence_groups"] == []
    assert all(v is None for v in result["surrogate_priority"].values())
    assert all(d.filtered is None for d in result["decisions"].values())
    assert result["dispatch_authorized"] is False


def test_allocation_cap_is_admitted_before_any_prediction(tmp_path):
    ledger, model = work(tmp_path, candidate_screenings=1), PrefixModel()
    with pytest.raises(RuntimeError, match="candidate_screenings"):
        screen(allocator(model), ledger)
    assert model.calls == 0
    assert ledger.summary()["used"]["model_head_row_queries"] == 0


def test_failed_prediction_remains_charged_with_explicit_failed_receipts(tmp_path):
    ledger = work(tmp_path)
    with pytest.raises(RuntimeError, match="Synthetic model failure"):
        screen(allocator(PrefixModel(fail=True)), ledger)
    summary = ledger.summary()
    assert summary["used"]["candidate_screenings"] == 2
    assert summary["used"]["model_head_row_queries"] == 18
    assert summary["unresolved"] == []
    assert summary["terminal_status"] == {
        "screen/prediction": "failed", "screen/allocation": "failed"}


def residual(weights=None, **kwargs):
    return BoundedActionResidual(np.zeros((8, 49)) if weights is None else weights,
                                 np.full(7, -2.), np.full(7, 2.), **kwargs)


def native():
    actions = np.tile(np.r_[np.linspace(-1.8, 1.8, 7), 1.], (8, 1))
    actions[::2, 7] = -1.
    return actions


def test_zero_residual_is_exact_native_identity_for_absolute_joints_and_gripper():
    actions = native()
    controller = residual()
    result = controller.propose(actions, np.ones(44), np.ones(5))
    np.testing.assert_array_equal(result["proposed_prefix"], actions)
    np.testing.assert_array_equal(actions, native())
    assert not result["gripper_overrides"].any()
    assert result["filtered_prefix"] is result["executed_prefix"] is None
    assert result["physical_validation"] is False and result["new_executed_controls"] == 0


def test_residual_respects_joint_delta_and_runtime_limits_and_discrete_gripper():
    weights = np.zeros((8, 49))
    weights[:, 0] = 1.
    weights[7, 0] = -1.
    actions = native()
    actions[:, 6] = 1.999
    result = residual(weights).propose(actions, np.eye(44)[0], np.zeros(5))
    proposal = result["proposed_prefix"]
    assert np.max(np.abs(result["arm_delta"])) <= .05 + 1e-12
    assert np.all(proposal[:, :7] <= 2.) and np.all(proposal[:, :7] >= -2.)
    np.testing.assert_array_equal(proposal[:, 6], np.full(8, 2.))
    np.testing.assert_array_equal(proposal[:, 7], np.full(8, -1.))
    np.testing.assert_array_equal(result["gripper_overrides"], actions[:, 7] == 1.)


def test_neutral_gripper_preserves_open_lid_phase_while_arm_residual_can_move():
    weights = np.zeros((8, 49))
    weights[:7, 0] = 1.
    actions = native()
    actions[:, 7] = 1.
    result = residual(weights).propose(actions, np.eye(44)[0], np.zeros(5))
    assert np.any(result["arm_delta"])
    np.testing.assert_array_equal(result["proposed_prefix"][:, 7], np.ones(8))


def test_residual_memory_replays_exactly_and_reset_removes_previous_phase():
    weights = np.zeros((8, 49))
    weights[:, 0] = 1.
    controller = residual(weights)
    actions, features = native(), np.eye(44)[0]
    controller.propose(actions, features, np.zeros(5))
    restored = residual(weights)
    restored.load_state_dict(controller.state_dict())
    expected = controller.propose(actions, -features, np.zeros(5))
    actual = restored.propose(actions, -features, np.zeros(5))
    np.testing.assert_array_equal(actual["proposed_prefix"], expected["proposed_prefix"])
    np.testing.assert_array_equal(actual["coefficients"], expected["coefficients"])
    np.testing.assert_array_equal(actual["proposed_prefix"][:, 7], actions[:, 7])
    restored.reset()
    fresh = restored.propose(actions, -features, np.zeros(5))
    np.testing.assert_array_equal(fresh["proposed_prefix"][:, 7], np.full(8, -1.))
    assert not np.array_equal(fresh["coefficients"], expected["coefficients"])


def test_invalid_targets_and_incompatible_memory_fail_without_changing_state():
    controller = residual()
    state = controller.state_dict()
    bad_joint, bad_gripper = native(), native()
    bad_joint[0, 0], bad_gripper[0, 7] = 2.001, 0.
    for actions in (bad_joint, bad_gripper):
        with pytest.raises(ValueError, match="native absolute targets"):
            controller.propose(actions, np.zeros(44), np.zeros(5))
        assert controller.state_dict() == state
    incompatible = residual(persistence=.1)
    with pytest.raises(ValueError, match="different controller contract"):
        incompatible.load_state_dict(state)
    state["previous"] = [2.] * 8
    with pytest.raises(ValueError, match="memory exceeds bounds"):
        controller.load_state_dict(state)
    with pytest.raises(ValueError, match="residual contract"):
        BoundedActionResidual(np.zeros((8, 49)), np.ones(7), np.zeros(7))
