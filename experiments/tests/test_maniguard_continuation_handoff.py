"""CPU handoff semantics; these checks do not establish physical continuation."""

import numpy as np
import pytest

from helpers.maniguard_continuation_handoff import HandoffReference, replay_factual_prefix
from helpers.maniguard_latent import LatentController


def features():
    return np.r_[np.float32(1), np.zeros(43, dtype=np.float32)]


def weights():
    result = np.zeros((4, 44), dtype=np.float32)
    result[0, 0] = 0.25
    return result


def reference(mode="privileged_lid_handoff", **kwargs):
    return HandoffReference(weights(), mode=mode, **kwargs)


def observe(controller, step, *, closed=True, violated=False):
    controller.observe(step, closed=closed, supported=True, violated=violated)


def test_closure_path_matches_original_code_and_offset_without_touching_global_rng():
    controller = reference()
    observe(controller, 0, closed=False)
    before = np.random.get_state()
    result = controller.select(0, features())
    after = np.random.get_state()
    expected = LatentController(weights())
    np.testing.assert_array_equal(result["code"], expected.code(features()))
    np.testing.assert_array_equal(result["noise_offset"], expected.offset(features(), 16, 32))
    np.testing.assert_array_equal(before[1], after[1])
    assert before[0] == after[0] and before[2:] == after[2:]
    assert result["requests_policy_rng_reset"] is False
    assert result["policy_rng_continuity_verified"] is False


def test_debounce_does_not_switch_midchunk_and_native_path_has_no_explicit_noise():
    controller = reference()
    observe(controller, 0, closed=False)
    controller.select(0, features())
    for step in range(1, 4):
        observe(controller, step)
    assert controller.qualified_step == 3 and not controller.native_latched
    with pytest.raises(ValueError, match="exact eight-control query"):
        controller.select(3, features())
    for step in range(4, 9):
        observe(controller, step)
    result = controller.select(8, features())
    assert result["branch"] == "native" and result["noise_offset"] is None
    assert result["call_native_infer_without_explicit_noise"] is True
    assert result["state"]["switch_query_step"] == 8
    assert result["state"]["switch_qualified_step"] == 3
    assert result["dispatch_authorized"] is False and result["physical_fitness"] is None


def test_reopening_before_query_cancels_earlier_qualification():
    controller = reference()
    observe(controller, 0, closed=False)
    controller.select(0, features())
    for step in range(1, 9):
        observe(controller, step, closed=step != 8)
    assert controller.first_debounce_step == 3 and controller.qualified_step is None
    assert controller.select(8, features())["branch"] == "closure"


def test_violation_latches_without_stopping_or_reversing_the_selected_controller():
    failed = reference()
    observe(failed, 0, closed=False)
    failed.select(0, features())
    for step in range(1, 9):
        observe(failed, step, violated=step == 1)
    assert failed.ever_violated and failed.select(8, features())["branch"] == "closure"
    switched = reference()
    observe(switched, 0, closed=False)
    switched.select(0, features())
    for step in range(1, 9):
        observe(switched, step)
    switched.select(8, features())
    for step in range(9, 17):
        observe(switched, step, closed=False, violated=True)
    assert switched.select(16, features())["branch"] == "native"
    assert switched.switch_query_step == 8


def test_clock_reference_cannot_consume_monitor_labels():
    controller = reference("clock_handoff", clock_switch_step=16)
    with pytest.raises(ValueError, match="does not consume monitor"):
        observe(controller, 0)
    assert controller.select(0, features())["branch"] == "closure"
    assert controller.select(8, features())["branch"] == "closure"
    assert controller.select(16, features())["branch"] == "native"
    assert controller.state()["last_observed"] == -1
    assert controller.switch_qualified_step is None


@pytest.mark.parametrize("mode,native", [("native_only", True), ("closure_only", False)])
def test_fixed_references_keep_the_same_branch_and_need_no_monitor(mode, native):
    controller = reference(mode)
    for step in (0, 8, 16):
        result = controller.select(step, features())
        assert result["call_native_infer_without_explicit_noise"] is native
    assert controller.switch_query_step is None


def test_episode_reset_removes_phase_violation_and_query_history():
    controller = reference()
    observe(controller, 0)
    controller.select(0, features())
    for step in range(1, 9):
        observe(controller, step)
    controller.select(8, features())
    observe(controller, 9, violated=True)
    controller.reset()
    assert controller.last_observed == -1 and controller.last_query is None
    assert not controller.ever_violated and not controller.native_latched
    observe(controller, 0, closed=False)
    assert controller.select(0, features())["branch"] == "closure"


def test_missing_observations_queries_and_invalid_inputs_fail_before_state_advance():
    controller = reference()
    with pytest.raises(ValueError, match="Observe this query"):
        controller.select(0, features())
    observe(controller, 0, closed=False)
    before = controller.state()
    bad = features()
    bad[1] = np.nan
    with pytest.raises(ValueError):
        controller.select(0, bad)
    assert controller.state() == before
    with pytest.raises(ValueError, match="query boundary was skipped"):
        observe(controller, 1)
    with pytest.raises(ValueError, match="consecutive factual"):
        observe(controller, 2)
    with pytest.raises(ValueError, match="Invalid prospective"):
        reference(clock_switch_step=7)


def factual_rows(controls=16):
    controller = LatentController(weights())
    for step in range(controls + 1):
        yield {
            "type": "observed",
            "step": step,
            "monitor": {
                "valid": True,
                "observation": {
                    "ap": {"jar_closed": step > 0, "jar_on_support": True},
                    "doomed": False,
                },
            },
        }
        if step < controls and step % 8 == 0:
            yield {
                "type": "latent_features",
                "step": step,
                "features": features(),
                "code": controller.code(features()),
            }


def test_replay_never_consumes_factual_rows_after_the_changed_query():
    def poisoned_tail():
        for row in factual_rows():
            yield row
            if row["type"] == "latent_features" and row["step"] == 8:
                raise AssertionError("The changed controller's future must not be read")

    result = replay_factual_prefix(poisoned_tail(), reference(), controls=16)
    assert result["first_changed_query_step"] == 8
    assert result["factual_closure_codes_compared"] == 1
    assert result["hybrid_physical_outcome"] is None


def test_replay_closure_only_checks_the_full_prefix_and_exact_controller():
    result = replay_factual_prefix(factual_rows(), reference("closure_only"), controls=16)
    assert result["first_changed_query_step"] is None
    assert result["factual_closure_codes_compared"] == 2
    assert result["last_factual_observation_consumed"] == 16
    assert result["monitor_observations_consumed_for_gating"] == 0
    rows = list(factual_rows())
    rows[1]["code"] = np.ones(4)
    with pytest.raises(ValueError, match="differs from the factual code"):
        replay_factual_prefix(rows, reference("closure_only"), controls=16)


def test_replay_clock_does_not_require_privileged_monitor_predicates():
    rows = list(factual_rows())
    for row in rows:
        row.pop("monitor", None)
    result = replay_factual_prefix(
        rows, reference("clock_handoff", clock_switch_step=8), controls=16
    )
    assert result["first_changed_query_step"] == 8
    assert result["monitor_observations_consumed_for_gating"] == 0


def test_replay_rejects_missing_queries_observations_and_incomplete_episodes():
    rows = list(factual_rows())
    for omitted in (1, 2, len(rows) - 1):
        with pytest.raises(ValueError):
            replay_factual_prefix(
                rows[:omitted] + rows[omitted + 1 :], reference("closure_only"), controls=16
            )
