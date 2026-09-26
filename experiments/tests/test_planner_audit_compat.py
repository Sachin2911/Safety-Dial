"""The committed continuation must not require another agent's uncommitted planner edits."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.imagination import NominalPlanner
from helpers.plannerAudit import PlannerAuditMixin
from helpers.safeCemT import SafePlannerT

REPO = Path(__file__).resolve().parents[2]


def archived_module(path, name, monkeypatch):
    source = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO, capture_output=True,
                            text=True, check=True).stdout
    module = ModuleType(name)
    module.__file__ = str(REPO / path)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(source, f"HEAD:{path}", "exec"), module.__dict__)
    return module


class FakeCost(torch.nn.Module):
    def __init__(self, hist):
        super().__init__()
        self.register_buffer("hist", hist)
        self.last_frac_feasible = 1.
        self.last_diag = {}
        self.singleton_calls = 0

    def get_cost(self, info, candidates):
        feasible = candidates[:, :, -1, 0].abs() == 1
        if candidates.shape[1] == 1:
            self.singleton_calls += 1
        self.last_frac_feasible = float(feasible.float().mean())
        self.last_diag = {"frac_feasible": self.last_frac_feasible,
                          "elite_feasible": bool(feasible[0, 0])}
        return torch.where(feasible, 0., 10.)


class FakeSolver:
    def __init__(self, *, model, **kwargs):
        self.model = model

    def configure(self, **kwargs):
        pass

    def solve(self, info, init_action):
        candidates = torch.stack([torch.ones_like(init_action), -torch.ones_like(init_action)], dim=1)
        self.model.get_cost(info, candidates)
        return {"actions": candidates.mean(1), "costs": torch.tensor([123.])}


@pytest.mark.parametrize("revision,kind", [("head", "nominal"), ("head", "safe"),
                                          ("working", "nominal"), ("working", "safe")])
def test_exact_plan_audit_with_committed_and_working_tree_bases(monkeypatch, revision, kind):
    import stable_worldmodel.solver

    monkeypatch.setattr(stable_worldmodel.solver, "CEMSolver", FakeSolver)
    if revision == "head":
        old = archived_module("experiments/helpers/imagination.py", "archived_audit_imagination", monkeypatch)
        assert not hasattr(old.NominalPlanner, "_audit")
        with monkeypatch.context() as imports:
            imports.setitem(sys.modules, "helpers.imagination", old)
            old_safe = archived_module("experiments/helpers/safeCemT.py", "archived_audit_safe", imports)
        base = old.NominalPlanner if kind == "nominal" else old_safe.SafePlannerT
    else:
        base = NominalPlanner if kind == "nominal" else SafePlannerT

    class InjectedCost(base):
        def _cost_model(self, hist, pusher_xy):
            return FakeCost(hist)

        def _info(self, frames, goal_frame):
            return {"pixels": torch.zeros(1, len(frames), 0)}

    class Audited(PlannerAuditMixin, InjectedCost):
        pass

    scaler = SimpleNamespace(mean_=np.zeros(2), scale_=np.ones(2),
                             transform=lambda value: value, inverse_transform=lambda value: value)
    kw = {} if kind == "nominal" else {"probe": None, "hazard": None}
    planner = Audited(torch.nn.Identity(), {"action": scaler}, device="cpu", num_samples=2,
                      n_steps=1, topk=2, **kw)
    history = np.full((2, 5, 2), .2)
    result = planner.plan([None] * 3, history, None, seed=1)
    assert result.diag["executed_feasible"]
    assert result.diag["audit_replacement_used"]
    assert result.diag["audit_before_replacement"]["executed_feasible"] is False
    assert result.frac_feasible == 1.
    assert np.all(result.blocks == 1.)  # The actual env actions match the replacement.
    assert np.allclose(result.flat[:2].numpy(), .2)  # Recorded history is pinned.
    assert result.cost == 0.
    assert result.diag["solver_reported_cost"] == 123.
    assert planner._audit_active_cost_model.singleton_calls == 2  # Mean + replacement; no duplicate outer audit.
    assert result.diag["audit_at_base_hook"] is (revision == "working" and hasattr(base, "_audit"))
