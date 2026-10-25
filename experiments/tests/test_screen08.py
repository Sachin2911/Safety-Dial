"""Check physical intervention semantics, causal timing and financial isolation."""

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.screen08_controller import Controller, ScreenConfig
from helpers.screen08_infra import FinancialBook
from helpers.screen08_protocol import ARMS, TASKS, protocol, select_survivors


def chunk(grip=1.0):
    actions = np.arange(128, dtype=np.float32).reshape(16, 8) / 100
    actions[:, -1] = grip
    return actions


@pytest.mark.parametrize("mode,horizon", [("native", 8), ("h4", 4)])
def test_passthrough_and_required_fresh_queries(mode, horizon):
    controller = Controller(ScreenConfig(mode=mode))
    proposal = chunk()
    controller.proposal(0, proposal)
    for step in range(horizon):
        result, _ = controller.apply(step, proposal[step], np.zeros(7))
        assert result.tobytes() == proposal[step].tobytes()
    with pytest.raises(ValueError, match="Missing fresh proposal"):
        controller.apply(horizon, proposal[horizon], np.zeros(7))
    controller.proposal(horizon, proposal)
    controller.apply(horizon, proposal[0], np.zeros(7))


def test_scaling_activates_at_execution_and_preserves_gripper():
    controller = Controller(ScreenConfig(mode="slow"))
    proposal = chunk()
    proposal[2, -1] = -1
    controller.proposal(0, proposal)
    for step in range(8):
        q = np.full(7, step / 100)
        result, trace = controller.apply(step, proposal[step], q)
        expected = proposal[step].copy()
        if step >= 2:
            expected[:7] = q + .5 * (expected[:7] - q)
        np.testing.assert_array_equal(result, expected)
        assert trace["scaling_active"] == (step >= 2)
        assert result[-1] == proposal[step, -1]


def test_smoothing_uses_unexecuted_prior_prediction_and_never_blends_gripper():
    controller = Controller(ScreenConfig(mode="smooth"))
    old = chunk(-1)
    new = chunk(1) + np.array([2] * 7 + [0], dtype=np.float32)
    controller.proposal(0, old)
    for step in range(8):
        result, _ = controller.apply(step, old[step], np.zeros(7))
        np.testing.assert_array_equal(result, old[step])
    controller.proposal(8, new)
    for offset in range(8):
        result, _ = controller.apply(8 + offset, new[offset], np.zeros(7))
        weight = .5 + .5 * offset / 7
        np.testing.assert_allclose(result[:7], weight * new[offset, :7]
                                   + (1 - weight) * old[8 + offset, :7], rtol=1e-6)
        assert result[-1] == 1
    np.testing.assert_array_equal(result, new[7])


def test_gripper_requires_full_quiet_history_and_any_joint_motion_resets_permission():
    controller = Controller(ScreenConfig(mode="gripwait"))
    proposal = chunk(-1)
    for step in range(40):
        if step % 8 == 0:
            controller.proposal(step, proposal)
        q = np.zeros(7)
        if step >= 18:
            q[6] = .03
        result, _ = controller.apply(step, proposal[step % 8], q)
        allowed = step in (16, 17) or step >= 34
        assert result[-1] == (-1 if allowed else 1)
        np.testing.assert_array_equal(result[:7], proposal[step % 8, :7])


def test_state_does_not_leak_between_episodes():
    old = Controller(ScreenConfig(mode="slow"))
    old.proposal(0, chunk(-1))
    old.apply(0, chunk(-1)[0], np.zeros(7))
    fresh = Controller(ScreenConfig(mode="slow"))
    fresh.proposal(0, chunk(1))
    result, trace = fresh.apply(0, chunk(1)[0], np.zeros(7))
    assert not trace["scaling_active"]
    np.testing.assert_array_equal(result, chunk(1)[0])


def test_financial_ceiling_exclusivity_and_separate_scope(tmp_path):
    book = FinancialBook(tmp_path / "finance.json")
    offer = {"dph_total": .6, "driver_version": "580.95.05", "gpu_name": "RTX 4090",
             "num_gpus": 1, "cpu_ram": 64000, "cpu_cores_effective": 32,
             "inet_down_cost": .0013, "inet_up_cost": .0013}
    now = dt.datetime.now(dt.timezone.utc)
    book.reserve("new", now.isoformat(), offer)
    book.attach("new", 123)
    assert book.snapshot(now.timestamp())["total_estimate_usd"] == 2
    assert book.snapshot(now.timestamp() + 14 * 3600)["stop"]
    with pytest.raises(ValueError, match="existing allocation"):
        book.reserve("other", now.isoformat(), offer)
    book.stopped(123, now.isoformat(), status="stopped")
    assert not book.snapshot()["active_instances"]
    assert len(book.data["allocations"]) == 1
    assert book.data["limits"]["total_usd"] == 10


def test_schedule_uses_fresh_disjoint_panels_with_rotating_arm_order():
    plan = protocol()
    assert len(plan["engineering_jobs"]) == 5
    assert len(plan["jobs"]) == len(plan["stage_b_jobs"]) == 60
    all_jobs = plan["engineering_jobs"] + plan["jobs"] + plan["stage_b_jobs"]
    assert len({j["job"] for j in all_jobs}) == 125
    heldout = {2, 5, 8, 14, 16, 23, 7, 18, 12}
    assert not heldout.intersection(j["task"] for j in all_jobs)
    assert not set(j["task"] for j in plan["engineering_jobs"]).intersection(TASKS)
    assert len({(j["arm"], j["seed"], j["task"]) for j in plan["jobs"]}) == 60
    for arm in ARMS:
        assert sum(j["arm"] == arm for j in plan["jobs"]) == 12
    assert len({plan["jobs"][block * 5]["arm"] for block in range(12)}) == 5


def test_selection_requires_completion_safety_and_actual_mechanism():
    episodes = [{**j, "primary": False, "raw_violation": False,
                 "counted_violation": False, "mechanism_active": False}
                for j in protocol()["jobs"]]
    for arm in ("h4", "slow", "smooth"):
        for e in [e for e in episodes if e["arm"] == arm][:2]:
            e.update(primary=True, mechanism_active=arm != "smooth")
    next(e for e in episodes if e["arm"] == "slow")["raw_violation"] = True
    survivors, _ = select_survivors(episodes)
    assert survivors == ["h4"]
    with pytest.raises(ValueError, match="complete unique"):
        select_survivors(episodes[:-1])
    with pytest.raises(ValueError, match="complete unique"):
        select_survivors(episodes + [episodes[0]])
