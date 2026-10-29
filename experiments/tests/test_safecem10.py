"""Task-10 unit tests: geometry, ranking, clearance schedules, search operators, queue."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/scripts"))

from helpers import safecem10_evo as evo  # noqa: E402
from helpers import safecem10_geometry as geo  # noqa: E402
from helpers import safecem10_planner as planner  # noqa: E402
from helpers import safecem10_protocol as protocol  # noqa: E402
from helpers.safecem10_infra import FinancialBook  # noqa: E402

SPEC = geo.design()
G0 = geo.layout("G0", SPEC)


def path_through(points, n=40):
    xs = np.linspace(points[0][0], points[-1][0], n)
    ys = np.linspace(points[0][1], points[-1][1], n)
    states = np.zeros((n, 7))
    states[:, 0], states[:, 1] = xs, ys
    states[:, 2:4] = SPEC["environment"]["start_block"]
    return states


def test_gate_boxes_and_closure():
    left, right = geo.gate_boxes(G0, SPEC)
    assert left == (0.0, 50.0, 190.0, 230.0)
    assert right == (110.0, 330.0, 190.0, 230.0)
    assert not geo.gap_closed(G0, 29.9)
    assert geo.gap_closed(G0, 30.0)
    narrow = geo.layout("N40", SPEC)
    assert geo.gap_closed(narrow, 20.0) and not geo.gap_closed(narrow, 19.0)
    with pytest.raises(ValueError):
        geo.gate_boxes({"gap_center_x": 20.0, "gap_width": 60.0}, SPEC)


def test_realized_metrics_distinguish_gap_pillar_and_detour():
    boxes = geo.gate_boxes(G0, SPEC)
    through_gap = path_through([(80.0, 40.0), (80.0, 400.0)])
    m = geo.realized_metrics(through_gap, G0, SPEC, terminated=True)
    assert m["exposure"] == 0.0 and m["crossing"] == "gap" and m["success"]
    assert m["J"] == len(through_gap) and m["steps_to_success"] == len(through_gap)
    through_pillar = path_through([(200.0, 40.0), (200.0, 400.0)])
    m = geo.realized_metrics(through_pillar, G0, SPEC, terminated=False)
    assert m["any_violation"] and m["crossing"] == "pillar" and m["J"] == 50.0
    assert m["hazard_steps"] == int(geo.inside_any(through_pillar[:, :2], boxes).sum())
    detour = path_through([(400.0, 40.0), (400.0, 400.0)])
    m = geo.realized_metrics(detour, G0, SPEC)
    assert m["exposure"] == 0.0 and m["crossing"] == "detour"
    stays = path_through([(80.0, 40.0), (80.0, 150.0)])
    assert geo.realized_metrics(stays, G0, SPEC)["crossing"] == "none"


def test_out_of_arena_and_detour_bound():
    xy = np.array([[10.0, 10.0], [-1.0, 10.0], [10.0, 600.0]])
    assert geo.out_of_arena(xy).tolist() == [False, True, True]
    bound = geo.detour_length((40, 40), (250, 310), geo.gate_boxes(G0, SPEC), 30.0)
    assert bound["best_corner_route"] >= bound["direct"]


def test_rank_costs_puts_feasible_first_and_grades_infeasible():
    goal = torch.tensor([[3.0, 1.0, 2.0, 0.5]])
    viol = torch.tensor([[0.0, 5.0, 0.0, 20.0]])
    total, feasible = planner.rank_costs(goal, viol, "safe")
    assert feasible.tolist() == [[True, False, True, False]]
    order = torch.argsort(total[0]).tolist()
    assert order == [2, 0, 1, 3]
    assert total[0, 1] < total[0, 3]
    pen, _ = planner.rank_costs(goal, viol, "penalty", lam=0.1)
    assert torch.allclose(pen, goal + 0.1 * viol)
    _, tolerant = planner.rank_costs(goal, viol, "safe", tol=5.0)
    assert tolerant.tolist() == [[True, True, True, False]]


def test_hazard_violation_follows_clearance_schedule():
    boxes = geo.gate_boxes(G0, SPEC)
    # Candidate 0 threads the gap centre, candidate 1 crosses the right pillar.
    xy = torch.tensor(
        [
            [
                [[80.0, 150.0], [80.0, 185.0], [80.0, 210.0], [80.0, 235.0], [80.0, 270.0]],
                [[200.0, 150.0], [200.0, 185.0], [200.0, 210.0], [200.0, 235.0], [200.0, 270.0]],
            ]
        ]
    )
    none = planner.hazard_violation(xy, boxes, [0.0] * 4)
    assert none[0, 0] == 0.0 and none[0, 1] > 0.0
    touching = planner.hazard_violation(xy, boxes, [30.0] * 4)
    assert touching[0, 0] == 0.0, "at exactly half the gap the pillars only touch the centre"
    wide = planner.hazard_violation(xy, boxes, [31.0] * 4)
    assert wide[0, 0] > 0.0, "a 31 px clearance closes the 60 px gap"
    tighter_later = planner.hazard_violation(xy, boxes, [0.0, 0.0, 31.0, 31.0])
    looser_later = planner.hazard_violation(xy, boxes, [31.0, 31.0, 0.0, 0.0])
    assert tighter_later[0, 0] > 0.0 and looser_later[0, 0] > 0.0
    assert planner.hazard_violation(xy, boxes, [5.0] * 4)[0, 1] > none[0, 1]
    with pytest.raises(ValueError):
        planner.hazard_violation(xy, boxes, [0.0, 0.0])


def test_commanded_positions_and_arena_exit():
    actions = torch.full((1, 1, 2, 10), 0.1)
    pos = planner.commanded_positions(actions, (40.0, 40.0), np.zeros(2), np.ones(2))
    assert pos.shape == (1, 1, 10, 2)
    assert torch.allclose(pos[0, 0, 0], torch.tensor([50.0, 50.0]))
    assert torch.allclose(pos[0, 0, -1], torch.tensor([140.0, 140.0]))
    outside = torch.tensor([[-10.0, 20.0], [20.0, 520.0], [100.0, 100.0]])
    assert planner.arena_exit_depth(outside).tolist() == [10.0, 8.0, 0.0]


class FakeBase(torch.nn.Module):
    """Pretends the latent is the pusher position after each action block."""

    def get_cost(self, info, candidates):
        B, S, T, D = candidates.shape
        start = torch.as_tensor(info["start"], dtype=candidates.dtype).view(1, 1, 1, 2)
        steps = candidates.reshape(B, S, T, D // 2, 2).sum(dim=3) * 100.0
        positions = start + torch.cumsum(steps, dim=2)
        current = start.expand(B, S, 1, 2)
        info["predicted_emb"] = torch.cat([current, positions], dim=2)
        goal = torch.as_tensor(info["goal_xy"], dtype=candidates.dtype)
        return torch.linalg.norm(positions[:, :, -1] - goal, dim=-1)


def test_dial_cost_model_ranks_the_gap_route_safe_only_while_open():
    boxes = geo.gate_boxes(G0, SPEC)
    probe = torch.nn.Identity()
    info = {
        "pixels": torch.zeros(1, 1, 1, 3, 2, 2),
        "start": [80.0, 150.0],
        "goal_xy": [80.0, 350.0],
    }
    # Two candidates of two blocks each: straight down the gap, or a right swing that
    # ends 30 px off the gap line and so sits farther from the goal.
    gap = torch.tensor([[0.0, 0.2] * 5, [0.0, 0.2] * 5]).reshape(2, 10) / 5
    swing = torch.tensor([[0.6, 0.2] * 5, [-0.3, 0.2] * 5]).reshape(2, 10) / 5
    candidates = torch.stack([gap, swing]).unsqueeze(0)
    for d, gap_open in ((0.0, True), (31.0, False)):
        model = planner.DialCostModel(
            FakeBase(), probe, boxes, [d, d], action_mean=np.zeros(2), action_scale=np.ones(2)
        )
        model.pusher_xy = np.array([80.0, 150.0])
        total = model.get_cost(dict(info), candidates)
        assert total.shape == (1, 2)
        record = model.history[-1]
        if gap_open:
            assert total[0, 0] < total[0, 1]
            assert record["frac_feasible"] == 1.0 and record["elite_feasible"]
        else:
            assert record["frac_feasible"] == 0.0 and not record["elite_feasible"]
            assert total[0, 0] > record["goal_min"], "infeasible costs sit above goal costs"
    assert model.queries == 2


def test_decode_encode_round_trip_and_grid():
    genes = SPEC["stages"]["C"]["genes"]
    params = evo.decode([0.5, 0.5, 0.5, 0.5, 0.5], genes)
    assert params == {"d0": 30.0, "d1": 0.0, "horizon": 5, "topk_frac": 0.175, "tol": 7.5}
    back = evo.encode(params, genes)
    assert np.allclose(back, 0.5)
    assert evo.decode([0, 0, 0, 0, 0], genes)["horizon"] == 3
    assert evo.decode([1, 1, 1, 1, 1], genes)["horizon"] == 7
    points = evo.grid_points(SPEC["stages"]["C"]["grid"], genes)
    assert len(points) == 96 and all(0 <= p.min() and p.max() <= 1 for p in points)
    schedule = evo.clearance_schedule({"d0": 10.0, "d1": -8.0, "horizon": 4})
    assert schedule == [10.0, 2.0, 0.0, 0.0]
    assert evo.clearance_schedule({"d0": 70.0, "d1": 8.0, "horizon": 3})[-1] == 80.0


def test_non_dominated_sort_crowding_and_hypervolume():
    objectives = [[0.2, 10.0], [0.5, 5.0], [0.6, 20.0], [0.1, 40.0], [0.9, 49.0]]
    fronts = evo.non_dominated_sort(objectives)
    assert fronts[0] == [0, 1, 3]
    assert fronts[1] == [2] and fronts[2] == [4]
    crowd = evo.crowding_distance(objectives, fronts[0])
    assert crowd[1] == float("inf") and crowd[3] == float("inf")
    assert np.isfinite(crowd[0])
    hv = evo.hypervolume_2d([[0.2, 10.0], [0.5, 5.0], [0.6, 20.0]], [1.0, 50.0])
    assert hv == pytest.approx(0.5 * 45 + 0.3 * 40)
    assert evo.hypervolume_2d([[0.0, 50.0]], [1.0, 50.0]) == 0.0
    assert evo.crowded_subset(objectives, fronts[0], 2) == [1, 3] or set(
        evo.crowded_subset(objectives, fronts[0], 2)
    ) == {1, 3}


def test_nsga2_operators_are_bounded_and_deterministic():
    rng = np.random.default_rng(3)
    population = evo.random_points(12, 5, rng)
    objectives = [[float(x[0]), float(50 - 40 * x[0] + 5 * x[1])] for x in population]
    children_a = evo.nsga2_offspring(population, objectives, 12, np.random.default_rng(7))
    children_b = evo.nsga2_offspring(population, objectives, 12, np.random.default_rng(7))
    assert len(children_a) == 12
    assert all(np.array_equal(a, b) for a, b in zip(children_a, children_b))
    assert all(c.min() >= 0.0 and c.max() <= 1.0 for c in children_a)
    merged = population + children_a
    merged_obj = objectives + [[float(x[0]), float(50 - 40 * x[0] + 5 * x[1])] for x in children_a]
    survivors, surv_obj, idx = evo.nsga2_survivors(merged, merged_obj, 12)
    assert len(survivors) == 12 == len(surv_obj) and len(set(idx)) == 12
    first_front = evo.pareto_front(merged_obj)
    assert set(first_front) <= set(idx) or len(first_front) > 12


def test_schedules_are_complete_unique_and_deterministic():
    jobs = protocol.schedule_A(SPEC)
    assert len(jobs) == 14 and len({j["id"] for j in jobs}) == 14
    assert all(len(j["clearances"]) == SPEC["planner"]["horizon"] for j in jobs)
    imagination = {"radial_quantiles_px": {"50": 12.4, "90": 31.0, "95": 47.0}}
    dials = protocol.quantile_dials(imagination, SPEC)
    assert [d["dial"] for d in dials] == [12.0, 47.0], "31 is within 2 px of the grid dial 30"
    b1 = protocol.schedule_B(SPEC, imagination)
    b2 = protocol.schedule_B(SPEC, imagination)
    assert [j["id"] for j in b1] == [j["id"] for j in b2]
    assert len(b1) == 4 * 10 * (7 + 2 + 2) and len({j["id"] for j in b1}) == len(b1)
    assert {j["quantile"] for j in b1 if j["mode"] == "safe"} == {None, 50, 95}
    assert sum(j["mode"] == "penalty" for j in b1) == 4 * 10 * 2


def test_candidate_jobs_and_evaluation_seeds():
    x = [0.0, 1.0, 1.0, 0.0, 1.0]
    job = protocol.candidate_job("C", "G0", 301, x, SPEC, "nsga2-r0-g00-c00", {"method": "nsga2"})
    assert job["planner"]["horizon"] == 7 and len(job["clearances"]) == 7
    assert job["planner"]["topk"] == 15 and job["planner"]["feasible_tol"] == 15.0
    assert job["clearances"] == [0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    low = protocol.candidate_job("C", "G0", 301, [0] * 5, SPEC, "t", {})
    assert low["planner"]["topk"] >= 2
    seeds = {
        tuple(protocol.evaluation_seeds(SPEC, "nsga2", r, g)) for r in range(3) for g in range(8)
    }
    assert len(seeds) == 24
    flat = [s for group in seeds for s in group]
    assert len(flat) == len(set(flat))
    assert protocol.evaluation_seeds(SPEC, "grid", 1, 2) == protocol.evaluation_seeds(
        SPEC, "nsga2", 1, 2
    )
    assert len(protocol.grid_for_budget(SPEC, 48)) == 48
    assert len(protocol.grid_for_budget(SPEC, 200)) == 96
    assert protocol.aggregate([{"exposure": 0.2, "J": 30}, {"exposure": 0.0, "J": 50}]) == [
        0.1,
        40.0,
    ]


def offer(**overrides):
    base = {
        "dph_total": 0.25,
        "driver_version": "580.178.04",
        "gpu_name": "RTX 3090",
        "num_gpus": 1,
        "gpu_ram": 24576,
        "cpu_ram": 64000,
        "cpu_cores_effective": 12,
        "disk_space": 200,
        "inet_down": 800,
        "inet_down_cost": 0.001,
        "inet_up_cost": 0.001,
    }
    base.update(overrides)
    return base


def test_financial_book_limits_and_offer_validation(tmp_path):
    book = FinancialBook(tmp_path / "finance.json")
    assert book.limits()["total_usd"] == 10.0
    FinancialBook.validate_offer(offer())
    for bad in (
        {"dph_total": 0.7},
        {"gpu_name": "RTX 4070"},
        {"num_gpus": 2},
        {"driver_version": "535.1"},
        {"cpu_ram": 16000},
        {"disk_space": 40},
        {"inet_down": 50},
        {"inet_down_cost": 0.1},
    ):
        with pytest.raises(ValueError):
            FinancialBook.validate_offer(offer(**bad))
    row = book.reserve("pusht-safecem10-test", "2026-10-25T12:00:00+00:00", offer())
    assert row["label"] == "pusht-safecem10-test"
    with pytest.raises(ValueError):
        book.reserve("another", "2026-10-25T12:00:00+00:00", offer())
    snapshot = book.snapshot(now=4e9)
    assert snapshot["stop"]
    book.attach("pusht-safecem10-test", 1234)
    book.stopped(1234, "2026-10-25T13:00:00+00:00", status="exited")
    final = book.snapshot(now=4e9)
    assert final["active_estimate_usd"] == pytest.approx(0.25)
    assert not final["active_instances"]


def test_queue_claim_is_exclusive(tmp_path, monkeypatch):
    import worker_safecem10 as worker

    queue = tmp_path / "queue"
    for name in ("pending", "claimed", "done", "failed"):
        (queue / name).mkdir(parents=True)
    monkeypatch.setattr(worker, "QUEUE", queue)
    ticket = {
        "job": {"id": "A-G0-100-lam0"},
        "attempt_dir": str(tmp_path / "a"),
        "freeze_sha256": "x",
    }
    (queue / "pending/A-G0-100-lam0.json").write_text(json.dumps(ticket))
    first, data = worker.claim("w1")
    second, _ = worker.claim("w2")
    assert first is not None and second is None
    assert json.loads(first.read_text())["worker"] == "w1"
    worker.finish(first, data, tmp_path / "a", "complete")
    assert (queue / "done/A-G0-100-lam0.json").exists() and not first.exists()


def test_orchestrator_inventory_skips_valid_results(tmp_path, monkeypatch):
    import run_safecem10 as runner

    monkeypatch.setattr(runner, "OUT", tmp_path)
    attempt = tmp_path / "attempts/A-G0-100-lam0"
    attempt.mkdir(parents=True)
    (tmp_path / "queue/failed").mkdir(parents=True)
    (attempt / "result.json").write_text(
        json.dumps({"status": "complete", "job": {"id": "A-G0-100-lam0"}, "env_steps": 50})
    )
    rows, failed = runner.inventory(tmp_path)
    assert runner.valid_result(rows, "A-G0-100-lam0", "sha") is not None
    assert runner.valid_result(rows, "missing", "sha") is None and failed == {}


def simulate_push_t_step(x, v, action, k_p=100.0, k_v=20.0, dt=0.01, substeps=10, scale=100.0):
    """Independent re-implementation of one Push-T env step for one axis."""
    target = x + scale * action
    for _ in range(substeps):
        v = v + dt * (k_p * (target - x) - k_v * v)
        x = x + dt * v
    return x, v


def test_pd_response_is_causal_and_lags_the_commanded_target():
    c, G = planner.pd_response(6)
    assert G.shape == (60, 6) and c.shape == (60,)
    assert np.allclose(G[:10, 1:], 0.0) and (G[:10, 0] > 0).all()
    x, v = 0.0, 0.0
    for a in (0.3, -0.1, 0.5):
        x, v = simulate_push_t_step(x, v, a)
    c3, G3 = planner.pd_response(3)
    assert np.isclose(G3[-1] @ np.array([0.3, -0.1, 0.5]), x)
    # A constant action settles near 40 percent of its 100 px target offset per step.
    _, G40 = planner.pd_response(40)
    travel = G40 @ np.ones(40)
    assert 38.0 < travel[399] - travel[389] < 42.0


def test_exact_check_follows_the_controller_path():
    boxes = [(100.0, 140.0, 100.0, 140.0)]
    model = planner.DialCostModel(
        FakeBase(),
        torch.nn.Identity(),
        boxes,
        [0.0, 0.0],
        action_mean=np.zeros(2),
        action_scale=np.ones(2),
        check="exact",
    )
    model.pusher_xy = np.array([60.0, 120.0])
    model.pusher_vel = np.zeros(2)
    toward = torch.zeros(2, 10)
    toward[:, 0::2] = 0.5
    away = torch.zeros(2, 10)
    away[:, 1::2] = 0.2
    candidates = torch.stack([toward, away]).unsqueeze(0)
    info = {
        "pixels": torch.zeros(1, 1, 1, 3, 2, 2),
        "start": [60.0, 120.0],
        "goal_xy": [300.0, 120.0],
    }
    model.get_cost(dict(info), candidates)
    path = model.controller_path(candidates)
    assert path.shape == (1, 2, 100, 2)
    _, G = planner.pd_response(10)
    expected = 60.0 + G[-1] @ toward[:, 0::2].reshape(-1).numpy()
    assert np.isclose(float(path[0, 0, -1, 0]), expected, atol=1e-3)
    record = model.history[-1]
    assert record["frac_feasible"] == 0.5 and record["elite_feasible"]
    assert len(record["elite_xy"]) == 3


def test_anchored_check_uses_only_imagined_displacements():
    class Offset(torch.nn.Module):
        def forward(self, z):
            return z + torch.tensor([200.0, -50.0])

    info = {
        "pixels": torch.zeros(1, 1, 1, 3, 2, 2),
        "start": [60.0, 60.0],
        "goal_xy": [100.0, 60.0],
    }
    cand = torch.zeros(1, 1, 2, 10)
    cand[..., 0::2] = 0.02
    for check, first in (("probe", 260.0), ("anchored", 60.0)):
        model = planner.DialCostModel(
            FakeBase(),
            Offset(),
            [(400.0, 450.0, 400.0, 450.0)],
            [0.0, 0.0],
            action_mean=np.zeros(2),
            action_scale=np.ones(2),
            check=check,
        )
        model.pusher_xy = np.array([60.0, 60.0])
        model.get_cost(dict(info), cand)
        xy = model.history[-1]["elite_xy"]
        assert np.isclose(xy[0][0], first) and np.isclose(xy[2][0] - xy[0][0], 20.0)
    with pytest.raises(ValueError):
        planner.DialCostModel(FakeBase(), Offset(), [], [0.0], check="oracle")


def test_best_candidate_tracking_prefers_feasible_then_goal_then_violation():
    model = planner.DialCostModel(
        FakeBase(), torch.nn.Identity(), [], [0.0], track_best=True, arena_constraint=False
    )
    cands = torch.arange(6, dtype=torch.float32).reshape(1, 3, 1, 2)
    xy = torch.zeros(1, 3, 2, 2)
    T = torch.tensor
    model.begin_solve()
    model._track(T([[5.0, 3.0, 1.0]]), T([[0.0, 0.0, 2.0]]), T([[True, True, False]]), cands, xy)
    assert torch.equal(model.best_actions()[0], cands[0, 1])
    infeasible = T([[False, False, False]])
    model._track(T([[0.5, 9.0, 9.0]]), T([[4.0, 1.0, 3.0]]), infeasible, cands + 100, xy)
    assert torch.equal(model.best_actions()[0], cands[0, 1]), "infeasible never replaces feasible"
    model.begin_solve()
    model._track(T([[0.5, 9.0, 9.0]]), T([[4.0, 1.0, 3.0]]), infeasible, cands + 100, xy)
    assert torch.equal(model.best_actions()[0], cands[0, 1] + 100)
    feasible = T([[True, True, True]])
    model._track(T([[7.0, 8.0, 6.5]]), T([[0.0, 0.0, 0.0]]), feasible, cands + 200, xy)
    assert torch.equal(model.best_actions()[0], cands[0, 2] + 200)
    model.history.append({})
    model.record_execution()
    assert model.history[-1]["executed_feasible"] and model.history[-1]["executed_goal"] == 6.5


def test_best_candidate_solver_executes_the_tracked_candidate():
    pytest.importorskip("stable_worldmodel")
    import gymnasium as gym
    import stable_worldmodel as swm

    from helpers.safecem10_runtime import best_candidate_solver

    class Base(torch.nn.Module):
        def get_cost(self, info, candidates):
            B, S, T, _ = candidates.shape
            info["predicted_emb"] = torch.zeros(B, S, T + 1, 2)
            return ((candidates - 0.3) ** 2).sum(dim=(2, 3))

    model = planner.DialCostModel(
        Base(),
        torch.nn.Identity(),
        [],
        [0.0],
        mode="penalty",
        lam=0.0,
        arena_constraint=False,
        track_best=True,
    )
    solver = best_candidate_solver()(
        model=model, num_samples=16, n_steps=3, topk=4, device="cpu", seed=0
    )
    solver.configure(
        action_space=gym.spaces.Box(-1, 1, shape=(1, 2)),
        n_envs=1,
        config=swm.PlanConfig(horizon=2, receding_horizon=1, action_block=5),
    )
    out = solver.solve({"pixels": torch.zeros(1, 1, 3, 2, 2)})
    assert torch.equal(out["actions"], model.best_actions())
    assert "elite_mean_actions" in out and not torch.equal(
        out["actions"], out["elite_mean_actions"]
    )
    assert model.history[-1]["executed"] == "best"
    best_cost = float(((out["actions"] - 0.3) ** 2).sum())
    assert np.isclose(best_cost, model._best[0]["goal"], atol=1e-4)


def test_step_tracking_error_is_zero_for_controller_generated_states():
    from helpers.safecem10_runtime import step_tracking_error

    rng = np.random.default_rng(0)
    actions = rng.uniform(-0.5, 0.5, size=(8, 2))
    start = np.array([100.0, 100.0, 300.0, 310.0, 0.7, 0.0, 0.0])
    pos, vel, states = start[:2].copy(), start[-2:].copy(), []
    for a in actions:
        for k in range(2):
            pos[k], vel[k] = simulate_push_t_step(pos[k], vel[k], a[k])
        states.append([pos[0], pos[1], 300.0, 310.0, 0.7, vel[0], vel[1]])
    err = step_tracking_error(np.array(states), start, actions)
    assert err["max"] < 1e-6


def test_schedule_B2_pairs_with_stage_B_and_sets_the_pipeline():
    imagination = {"radial_quantiles_px": {"50": 6.2, "90": 14.6, "95": 19.0}}
    jobs = protocol.schedule_B2(SPEC, imagination)
    ids = [j["id"] for j in jobs]
    assert len(ids) == len(set(ids))
    per_seed = 2 * (7 + 2) + 2 + 2 * 7 + 7 + 2
    assert len(jobs) == per_seed * 10
    assert {j["seed"] for j in jobs} == set(SPEC["stages"]["B"]["seeds"])
    assert all(j["stage"] == "B2" and j["id"].startswith("B2-") for j in jobs)
    fixed = [j for j in jobs if j["condition"] == "fixed"]
    assert all(
        (j["planner"]["check"], j["planner"]["probe_version"], j["planner"]["execute"])
        == ("anchored", "v2", "best")
        for j in fixed
    )
    assert sorted({j["quantile"] for j in fixed if j["quantile"]}) == [50, 90]
    exact = [j for j in jobs if j["condition"] == "exact"]
    assert {j["layout"] for j in exact} == {"G0", "G80"}
    assert {j["layout"] for j in jobs if j["condition"] == "bestonly"} == {"G0"}
    lam0 = [j for j in jobs if j["condition"] == "bestlam0"]
    assert all(j["mode"] == "penalty" and j["lam"] == 0.0 for j in lam0)
    assert protocol.schedule_B2(SPEC, imagination) == jobs


def test_candidate_job_carries_the_stage_pipeline():
    x = [0.5] * 5
    c = protocol.candidate_job("C", "G0", 300, x, SPEC, "t", {})
    c2 = protocol.candidate_job("C2", "G0", 300, x, SPEC, "t", {})
    assert (c["planner"]["check"], c["planner"]["execute"]) == ("probe", "mean")
    assert (c2["planner"]["check"], c2["planner"]["probe_version"]) == ("anchored", "v2")
    assert c2["planner"]["execute"] == "best" and c2["id"].startswith("C2-")
    assert c["clearances"] == c2["clearances"] and c["planner"]["topk"] == c2["planner"]["topk"]
