"""Audit the exact returned CEM sequence independently of uncommitted planner changes."""
from __future__ import annotations

import time
from types import MethodType

import torch

from helpers.imagination import NominalPlanner, model_to_blocks
from helpers.safeCemT import SafePlannerT


class PlannerAuditMixin:
    """Preserve a sampled candidate and audit/rebuild the returned plan on every base.

    Committed NominalPlanner does not call an audit hook. A newer working-tree version
    does. This mixin owns the singleton audit and handles both without depending on or
    modifying either implementation. A hook-completed, unchanged plan is not rescored.
    """

    def _cost_model(self, hist, pusher_xy):
        model = super()._cost_model(hist, pusher_xy)
        self._audit_active_cost_model = model
        original = model.get_cost

        def remember(this, info, candidates):
            costs = original(info, candidates)
            if candidates.shape[1] > 1:
                best = int(costs[0].argmin())
                this._audit_saved_candidate = candidates[0, best].detach().cpu().clone()
                n = len(this.hist)
                if n:
                    this._audit_saved_candidate[:n] = this.hist.detach().cpu()
            return costs

        model.get_cost = MethodType(remember, model)
        return model

    @torch.inference_mode()
    def _audit_sequence(self, cost_model, frames, goal_frame, flat):
        """Own the exact-sequence audit; no base-class _audit method is required."""
        info = {key: value.unsqueeze(1) for key, value in self._info(frames, goal_frame).items()}
        candidates = flat.to(getattr(self, "device", flat.device))[None, None]
        cost = cost_model.get_cost(info, candidates)
        if not torch.isfinite(cost).all():
            raise ValueError("The returned CEM sequence has a non-finite audited cost")
        diagnostic = getattr(cost_model, "last_diag", {}) or {}
        out = {"executed_feasible": getattr(cost_model, "last_frac_feasible", 1.0) == 1.0,
               "executed_cost": float(cost[0, 0])}
        out.update({"executed_" + key[len("elite_"):]: value
                    for key, value in diagnostic.items() if key.startswith("elite_")})
        return out

    @torch.inference_mode()
    def _audit(self, cost_model, frames, goal_frame, flat):
        saved = getattr(cost_model, "_audit_saved_candidate", None)
        saved = saved.clone() if saved is not None else None
        before = self._audit_sequence(cost_model, frames, goal_frame, flat)
        replaced, after = False, before
        if not before["executed_feasible"] and saved is not None:
            replacement = saved.to(device=flat.device, dtype=flat.dtype)
            if not torch.equal(flat, replacement):
                flat.copy_(replacement)
                after = self._audit_sequence(cost_model, frames, goal_frame, flat)
                replaced = True
        result = {**after, "audit_replacement_used": replaced,
                  "audit_replacement_reason": "returned_sequence_infeasible" if replaced else None,
                  "audit_before_replacement": before}
        self._audit_final_flat = flat.detach().cpu().clone()
        self._audit_final_diagnostic = dict(result)
        return result

    def plan(self, frames, hist_blocks, goal_frame, *, seed, pusher_xy=None, init_future=None):
        self._audit_active_cost_model = None
        self._audit_final_flat = None
        self._audit_final_diagnostic = None
        result = super().plan(frames, hist_blocks, goal_frame, seed=seed,
                              pusher_xy=pusher_xy, init_future=init_future)
        extra_started = time.perf_counter()
        cost_model = self._audit_active_cost_model
        if cost_model is None:
            raise RuntimeError("The base planner did not create the auditable cost-model hook")
        diagnostic = dict(result.diag or {})
        diagnostic.setdefault("frac_feasible", result.frac_feasible)
        previous_cost = float(result.cost)
        already_audited = (self._audit_final_flat is not None
                           and torch.equal(result.flat.detach().cpu(), self._audit_final_flat))
        audit = (self._audit_final_diagnostic if already_audited else
                 self._audit(cost_model, frames, goal_frame, result.flat))
        diagnostic.update(audit)
        diagnostic.setdefault("fallback_used", bool(audit["audit_replacement_used"]))
        diagnostic["solver_reported_cost"] = previous_cost
        diagnostic["audit_at_base_hook"] = already_audited
        result.diag = diagnostic
        # HEAD constructs blocks before this wrapper can audit. Rebuild from the exact
        # final flat tensor on both bases so the next env actions and audit agree.
        result.blocks = model_to_blocks(self.process, result.flat[len(frames) - 1:])
        result.cost = float(audit["executed_cost"])
        result.solve_time += time.perf_counter() - extra_started
        return result


class AuditedNominalPlanner(PlannerAuditMixin, NominalPlanner):
    """Nominal planner with an audited commanded-action arena guard."""


class AuditedSafePlannerT(PlannerAuditMixin, SafePlannerT):
    """Whole-T planner that cannot silently execute a newly infeasible elite mean."""
