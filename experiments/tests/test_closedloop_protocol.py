from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.plannerAudit import PlannerAuditMixin
from helpers.pushtClosedLoop import block_coverage, fixed_cases
from helpers.pushtGeometry import Box
from helpers.pushtLayouts import Layout


class FakePlanner:
    def _info(self, frames, goal_frame):
        return {}

    def _cost_model(self, hist, pusher_xy):
        class Cost:
            def __init__(self):
                self.hist = hist
                self.last_frac_feasible = 0

            def get_cost(self, info, candidates):
                feasible = candidates[:, :, -1, 0] == 1
                self.last_frac_feasible = float(feasible.float().mean())
                return torch.where(feasible, 0., 10.)

        return Cost()

    def _audit(self, cost_model, frames, goal, flat):
        cost_model.get_cost({}, flat[None, None])
        return {"executed_feasible": cost_model.last_frac_feasible == 1}


class AuditedFake(PlannerAuditMixin, FakePlanner):
    pass


def test_unsafe_elite_mean_is_replaced_by_preserved_feasible_candidate():
    planner = AuditedFake()
    cost = planner._cost_model(torch.empty((0, 1)), None)
    cost.get_cost({}, torch.tensor([[[[1.]], [[0.]]]]))
    mean = torch.tensor([[.5]])
    result = planner._audit(cost, None, None, mean)
    assert mean.item() == 1
    assert result["executed_feasible"] and result["audit_replacement_used"]
    assert result["audit_before_replacement"]["executed_feasible"] is False


def test_feasible_mean_is_preserved_and_singleton_audit_cannot_destroy_backup():
    planner = AuditedFake()
    cost = planner._cost_model(torch.empty((0, 1)), None)
    cost.get_cost({}, torch.tensor([[[[1.]], [[0.]]]]))
    plan = torch.tensor([[1.]])
    result = planner._audit(cost, None, None, plan)
    assert not result["audit_replacement_used"]
    assert cost._audit_saved_candidate.item() == 1


def test_fixed_cases_do_not_inspect_rollout_or_nominal_violation_outcomes():
    roots = [SimpleNamespace(root_id=f"r{i:02}", meta={"state_at_root": [100, 100, 250, 250, 0, 0, 0]},
                             goal_state=np.asarray([100, 100, 260, 250, 0, 0, 0])) for i in range(24)]
    layouts = [Layout(Box(10, 20, 10, 20).to_dict(), family, roots[0].root_id, .5, 10, 100, 100)
               for family in ("familiar", "heldout")]
    bank = SimpleNamespace(roots=roots)
    cases = fixed_cases(bank, layouts, seed=4)
    assert len(cases) == len({case["root_id"] for case in cases}) == 20
    assert cases == fixed_cases(bank, layouts, seed=4)
    assert len({case["hazard_family"] for case in cases}) == 2


def test_fixed_case_shortfall_aborts_instead_of_silently_shrinking_design():
    with pytest.raises(ValueError, match="No predeclared"):
        fixed_cases(SimpleNamespace(roots=[]), [])


def test_whole_t_goal_coverage():
    assert block_coverage([250, 250, .2], [250, 250, .2]) == pytest.approx(1.)
    assert block_coverage([100, 100, 0], [400, 400, 0]) == 0


def test_closedloop_stops_before_using_padded_terminal_observations(monkeypatch):
    import helpers.pushtClosedLoop as closedloop
    from helpers.pushtReplay import DenseLog

    start = np.asarray([100, 100, 250, 250, 0, 0, 0], float)
    root = SimpleNamespace(prefix=np.zeros((10, 2)), goal_state=start)
    ctx = SimpleNamespace(frames=[None] * 3, history_actions=np.zeros((2, 5, 2)), state=start, goal_frame=None)
    monkeypatch.setattr(closedloop, "reset_root", lambda *args: ctx)
    states = np.repeat(start[None], 6, axis=0)
    states[3:, 2:4] = 5  # Deliberately dangerous padding, which must never be counted.
    log = DenseLog(states, np.zeros((6, 2)), np.zeros(6), np.zeros(6), np.zeros((5, 2)),
                   [None, None], terminated=np.asarray([0, 0, 1, 0, 0, 0], bool),
                   truncated=np.zeros(6, bool), observed=np.asarray([1, 1, 1, 0, 0, 0], bool),
                   observation_valid=np.asarray([1, 1, 1, 0, 0, 0], bool))
    monkeypatch.setattr(closedloop, "run_actions", lambda *args, **kwargs: log)

    class Planner:
        def __init__(self):
            self.calls = 0

        def plan(self, *args, **kwargs):
            self.calls += 1
            assert self.calls == 1, "Conditioned on padded terminal frames"
            return SimpleNamespace(diag={"executed_feasible": True}, solve_time=.1,
                                   blocks=np.zeros((5, 5, 2)))

    result = closedloop.run_observed_episode(None, Planner(), root,
        {"hazard": Box(0, 10, 0, 10).to_dict(), "planning_seed": 0}, n_blocks=10)
    assert result["observed_env_steps"] == 2
    assert result["censored"] and result["hazard_only_unsafe"] is None
    assert result["unsafe_composite"] is False
    assert len(result["dense_states"]) == 3


def test_fixed_cases_exclude_exact_contact_at_zero_margin():
    roots = [SimpleNamespace(root_id=f"r{i:02}", meta={"state_at_root": [100, 100, 250, 250, 0, 0, 0]},
                             goal_state=np.asarray([100, 100, 250, 250, 0, 0, 0])) for i in range(24)]
    layouts = [Layout(Box(310, 320, 250, 280).to_dict(), family, roots[0].root_id, .5, 10, 100, 100)
               for family in ("familiar", "heldout")]
    with pytest.raises(ValueError, match="Only 0 valid roots"):
        fixed_cases(SimpleNamespace(roots=roots), layouts, margin=0)


def test_original_repaired_controls_are_paired_and_penalty_matching_is_explicit():
    from helpers.pushtClosedLoop import closedloop_arm_specs, penalty_reference_protocol

    base, repaired = object(), object()
    arms = closedloop_arm_specs(base, repaired, dial=3, fixed_margin=8)
    assert len(arms) == 7
    assert arms["nominal"] == (base, None, 3)
    for original, adapted in (("safe_original", "safe_repaired"),
                              ("safe_original_fixed_margin", "safe_repaired_fixed_margin"),
                              ("penalty_original", "penalty_repaired")):
        assert arms[original][0] is base and arms[adapted][0] is repaired
        assert arms[original][1:] == arms[adapted][1:]
    protocol = penalty_reference_protocol(.05)
    assert protocol["lambda_selection"] == "predeclared_before_test"
    assert protocol["matching"] == "equal_candidate_compute_only"
    assert protocol["behavioral_matching"] is False
    with pytest.raises(ValueError, match="nonnegative"):
        penalty_reference_protocol(float("nan"))
