"""Task-11 schedule, residual injection, R1 geometry, rewards and accounting."""

import json
import math

import numpy as np
import pytest

from helpers.dsrl06_protocol import benchmark_episode_seed, check_return_identity
from helpers.dsrl11_protocol import (
    BANK_ROOTS,
    CAPS,
    CONDITIONS,
    EVALUATION_ROOTS,
    ONLINE_ROOTS,
    R1_WEIGHTS,
    Dsrl11Ledger,
    all_jobs,
    condition_order,
    derive_seed,
    episode_rewards,
    frozen_jobs,
    inject_residual,
    lane_sequences,
    pause_rule_fired,
    r1_components,
    r1_potential,
    residual_for,
    select_learner_setting,
    uniform_residual,
)


def test_zero_residual_passes_the_native_draw_bitwise():
    epsilon = np.random.default_rng(1).standard_normal((16, 32)).astype(np.float32)
    assert np.array_equal(inject_residual(epsilon, np.zeros(32, np.float32)), epsilon)
    # Negative zero is still a zero action and must not perturb any bit pattern.
    assert np.array_equal(inject_residual(epsilon, -np.zeros(32, np.float32)), epsilon)


def test_nonzero_residual_is_tiled_quarter_scale_and_never_clipped():
    epsilon = np.full((16, 32), 3.9, np.float32)
    residual = np.linspace(-1, 1, 32).astype(np.float32)
    noise = inject_residual(epsilon, residual)
    assert noise.dtype == np.float32
    assert np.array_equal(noise, epsilon + np.float32(0.25) * np.tile(residual, (16, 1)))
    assert noise.max() > 4.0  # full noise is unbounded; only u is bounded


@pytest.mark.parametrize(
    "epsilon, residual",
    [
        (np.zeros((16, 32), np.float64), np.zeros(32, np.float32)),
        (np.zeros((16, 32), np.float32), np.zeros(32, np.float64)),
        (np.zeros((1, 16, 32), np.float32), np.zeros(32, np.float32)),
        (np.zeros((16, 32), np.float32), np.full(32, 1.0001, np.float32)),
        (np.zeros((16, 32), np.float32), np.full(32, np.nan, np.float32)),
    ],
)
def test_injection_rejects_wrong_shapes_types_and_domains(epsilon, residual):
    with pytest.raises(ValueError):
        inject_residual(epsilon, residual)


def test_seed_namespace_is_stable_and_distinct():
    assert derive_seed("epsilon", "bank", 110000) == derive_seed("epsilon", "bank", 110000)
    assert derive_seed("epsilon", "bank", 110000) != derive_seed("epsilon", "bank", 110001)
    assert 0 <= derive_seed("x") < 2**31
    with pytest.raises(ValueError):
        derive_seed("")


def test_uniform_residual_is_reconstructible_per_query():
    first = uniform_residual(123, 7)
    assert np.array_equal(first, uniform_residual(123, 7))
    assert not np.array_equal(first, uniform_residual(123, 8))
    assert first.dtype == np.float32 and np.max(np.abs(first)) <= 1


def test_frozen_schedule_counts_roots_and_unique_streams():
    jobs = frozen_jobs()
    assert [len(jobs[k]) for k in jobs] == [8, 12, 36, 40]
    flat = all_jobs(jobs)
    assert len(flat) == 96 == CAPS["starts"] - CAPS["replacements"]
    assert all(job["task"] == 3 for job in jobs["bank_jobs"] + jobs["online_jobs"])
    assert {j["seed"] for j in jobs["bank_jobs"]} == set(BANK_ROOTS)
    assert {j["seed"] for j in jobs["evaluation_jobs"]} == set(EVALUATION_ROOTS)
    for job in flat.values():
        assert job["episode_seed"] == benchmark_episode_seed(job["task"], job["seed"])
        assert job["simulator_reset_seed"] is None
    online = jobs["online_jobs"]
    for condition in CONDITIONS:
        rows = [j for j in online if j["condition"] == condition]
        assert [j["seed"] for j in rows] == list(ONLINE_ROOTS)
        assert [j["phase"] for j in rows] == ["warmup"] * 3 + ["batch_a"] * 3 + ["batch_b"] * 3
        assert [j["actor_version"] for j in rows] == [-1] * 3 + [0] * 3 + [1] * 3
        assert [j["residual_source"] for j in rows] == ["uniform"] * 3 + ["actor"] * 6
    epsilon = [j["epsilon_seed"] for j in online + jobs["bank_jobs"] + jobs["engineering_jobs"]]
    assert len(set(epsilon)) == len(epsilon)
    # Evaluation shares one native draw stream per root across arms (common random numbers).
    for root in EVALUATION_ROOTS:
        rows = [j for j in jobs["evaluation_jobs"] if j["seed"] == root]
        assert len({j["epsilon_seed"] for j in rows}) == 1
        assert [j["arm"] for j in rows] == ["native", *condition_order()]
        assert rows[0]["residual_source"] == "zero"
        assert all(j["actor_version"] == 2 for j in rows[1:])


def test_one_condition_permutation_balances_lanes_and_rounds():
    order = condition_order()
    assert sorted(order) == sorted(CONDITIONS)
    online = frozen_jobs()["online_jobs"]
    for round_index in range(12):
        rows = [j for j in online if j["round"] == round_index]
        assert len(rows) == 3 and len({j["condition"] for j in rows}) == 1
        assert sorted(j["lane"] for j in rows) == [0, 1, 2]
        assert rows[0]["condition"] == order[round_index % 4]
    lanes = lane_sequences(online)
    assert sorted(lanes) == [0, 1, 2] and all(len(v) == 12 for v in lanes.values())
    evaluation = lane_sequences(frozen_jobs()["evaluation_jobs"])
    jobs = all_jobs()
    for sequence in evaluation.values():
        arms = [jobs[j]["arm"] for j in sequence]
        assert len(sequence) == 10 and all(arms.count(a) == 2 for a in set(arms))


def test_residual_sources_reconstruct_or_defer_to_actor():
    flat = all_jobs()
    assert not np.any(residual_for(flat["B00"], 5))
    uniform = flat["E4"]
    assert np.array_equal(residual_for(uniform, 2), uniform_residual(uniform["uniform_seed"], 2))
    assert residual_for(flat["T_r1_prior_4"], 0) is None


def components(**overrides):
    values = dict(
        eef_pos=[0.0, 0.0, 0.0],
        jar_pos=[0.05, 0.0, 0.0],
        initial_jar_pos=[0.05, 0.0, 0.0],
        jar_aabb_min=[0.0, -0.05, -0.05],
        jar_aabb_max=[0.1, 0.05, 0.05],
        goal_center=[0.5, 0.0, 0.0],
    )
    values.update(overrides)
    return r1_components(**values)


def test_r1_components_follow_the_benchmark_geometry():
    parts = components()
    assert parts["reach"] == pytest.approx(1 - math.tanh(0.05 / 0.10))
    # Closest AABB point to the goal centre is x=0.1, so d = 0.4.
    assert parts["goal_distance_m"] == pytest.approx(0.4)
    assert parts["goal"] == pytest.approx(1 - math.tanh(0.4 / 0.20))
    assert parts["lift"] == 0.0
    lifted = components(jar_pos=[0.05, 0.0, 0.25])
    assert lifted["lift"] == 1.0
    inside = components(goal_center=[0.05, 0.0, 0.0])
    assert inside["goal_distance_m"] == 0.0 and inside["goal"] == 1.0
    with pytest.raises(ValueError):
        components(jar_aabb_min=[0.2, 0, 0])


def test_r1_potential_bounds_ordering_and_zeroing():
    assert sum(R1_WEIGHTS.values()) == pytest.approx(1.0)
    perfect = {"reach": 1.0, "lift": 1.0, "goal": 1.0}
    assert (
        r1_potential(perfect, closed=True, grasp=True, raw_violation=False, order_failed=False)
        == 1.0
    )
    assert r1_potential(
        perfect, closed=False, grasp=True, raw_violation=False, order_failed=False
    ) == pytest.approx(0.10)
    assert r1_potential(
        perfect, closed=True, grasp=False, raw_violation=False, order_failed=False
    ) == pytest.approx(0.30)
    for flag in ("raw_violation", "order_failed", "terminal"):
        kwargs = dict(closed=True, grasp=True, raw_violation=False, order_failed=False)
        kwargs[flag] = True
        assert r1_potential(perfect, **kwargs) == 0.0
    with pytest.raises(ValueError):
        r1_potential(
            {"reach": 1.2, "lift": 0, "goal": 0},
            closed=False,
            grasp=False,
            raw_violation=False,
            order_failed=False,
        )


@pytest.mark.parametrize("steps", [1, 8, 9, 15, 2000])
@pytest.mark.parametrize("strict", [False, True])
def test_episode_rewards_telescope_with_partial_chunks_and_timeouts(steps, strict):
    rng = np.random.default_rng(steps)
    potentials = list(rng.uniform(0, 1, steps + 1))
    rows = episode_rewards(potentials, steps, strict)
    assert len(rows) == math.ceil(steps / 8)
    assert rows[-1]["mask"] == 0.0 and rows[-1]["phi_after"] == 0.0
    assert all(r["mask"] == 1.0 for r in rows[:-1])
    assert rows[-1]["duration"] == steps - 8 * (len(rows) - 1)
    identity = check_return_identity(rows)
    assert abs(identity["return_identity_error"]) < 1e-12
    shaped, weight = 0.0, 1.0
    for row in rows:
        shaped += weight * row["reward"]
        weight *= row["discount"]
    discounted_base = float(strict) * np.prod([r["discount"] for r in rows[:-1]])
    assert shaped == pytest.approx(discounted_base - 0.25 * potentials[0], abs=1e-9)


def test_regrasp_and_violation_histories_keep_identity():
    # Chunk boundaries see grasp, drop, regrasp, then a violation that zeroes the rest.
    boundaries = {0: 0.1, 8: 0.55, 16: 0.3, 24: 0.6, 32: 0.0, 40: 0.0}
    potentials = [boundaries.get(i, 0.45) for i in range(41)]
    rows = episode_rewards(potentials, 40, False)
    check_return_identity(rows)
    signs = [np.sign(r["reward"]) for r in rows]
    assert signs[:4] == [1, -1, 1, -1] and rows[-1]["reward"] == 0.0
    # Interior potentials between boundaries do not enter the latent transitions.
    assert [r["phi_before"] for r in rows] == [0.1, 0.55, 0.3, 0.6, 0.0]


def test_ledger_caps_replacements_and_update_phases(tmp_path):
    ledger = Dsrl11Ledger(tmp_path / "resources.jsonl")
    ledger.reserve_start("B00", 1, event_id="B00:1")
    with pytest.raises(ValueError, match="invalid attempt"):
        ledger.reserve_start("B00", 2, event_id="B00:2", repair_receipt="x")
    ledger.finish_start("B00", 1, valid=False, event_id="B00:1:finish")
    with pytest.raises(ValueError, match="repair receipt"):
        ledger.reserve_start("B00", 2, event_id="B00:2")
    ledger.reserve_start("B00", 2, event_id="B00:2", repair_receipt="sha")
    with pytest.raises(ValueError, match="frozen job"):
        ledger.reserve_start("T00", 1)
    ledger.charge("learner_updates", 150_000, phase="diagnostic", event_id="a1")
    with pytest.raises(ValueError, match="diagnostic_updates"):
        ledger.charge("learner_updates", 30_001, phase="diagnostic")
    ledger.charge("learner_updates", 2_000, phase="engineering", event_id="profile")
    with pytest.raises(ValueError, match="engineering_updates"):
        ledger.charge("learner_updates", 1, phase="engineering")
    with pytest.raises(ValueError, match="diagnostic/engineering/learning"):
        ledger.charge("learner_updates", 1)
    ledger.charge("learner_updates", 120_000, phase="learning", event_id="online")
    with pytest.raises(ValueError, match="learning_updates"):
        ledger.charge("learner_updates", 1, phase="learning")
    # A retried reservation with the same identity is idempotent, not double-charged.
    ledger.charge("learner_updates", 120_000, phase="learning", event_id="online")
    with pytest.raises(ValueError, match="different data"):
        ledger.charge("learner_updates", 5, phase="learning", event_id="online")
    ledger.charge("model_queries", 1_200, model_only=True, event_id="probes")
    with pytest.raises(ValueError, match="model_only"):
        ledger.charge("model_queries", 1, model_only=True)
    usage = Dsrl11Ledger(tmp_path / "resources.jsonl").summary()["usage"]
    assert usage["model_queries"] == 1_200 and usage["starts"] == 2 and usage["replacements"] == 1
    assert usage["learner_updates"] == 272_000 and usage["diagnostic_updates"] == 150_000
    lines = (tmp_path / "resources.jsonl").read_text().splitlines()
    assert all(json.loads(line)["sha256"] for line in lines)


def test_pause_rule_needs_three_successive_failures():
    assert not pause_rule_fired([0.5, 0.5, 0.0, 0.5, 0.5])
    assert pause_rule_fired([0.0, 0.02, 0.5, 0.011])
    assert pause_rule_fired([math.nan, math.inf, 0.2])
    assert not pause_rule_fired([0.01, 0.01, 0.01])


def test_selection_rule_prefers_frozen_order_and_requires_all_cells():
    cells = {
        key: {"admitted": False} for key in [(20, "min"), (20, "mean"), (5, "min"), (5, "mean")]
    }
    assert select_learner_setting(cells)["selected"] is None
    cells[(5, "mean")]["admitted"] = True
    cells[(20, "mean")]["admitted"] = True
    assert select_learner_setting(cells)["selected"] == [20, "mean"]
    cells[(20, "min")]["admitted"] = True
    assert select_learner_setting(cells)["selected"] == [20, "min"]
    with pytest.raises(ValueError):
        select_learner_setting({(20, "min"): {"admitted": True}})
