"""Shared task-18 planning, adapted from the tested task-16 CEM implementation.

Fresh task paths and task-specific checkers; thirty CEM iterations and the same
predictor-work allowance across fixed and feedback configurations.
"""

import math
import torch
from stable_worldmodel.solver import CEMSolver
from helpers.lewm16_planner import CostModel as BaseCostModel, NoSafeAction

CANONICAL = {"pusht": {"samples": 300, "elite": 0.1}, "cube": {"samples": 300, "elite": 0.1}}


class CostModel(BaseCostModel):
    def get_cost(self, info, candidates):
        self.counters["model_batch_calls"] += 1
        return super().get_cost(info, candidates)


def adapter_mode(features, params):
    """Only declared task-side features enter. No true position or geometric goal distance."""
    allowed = {"current_cost", "improvement", "prediction_error", "previous_feasible", "replan"}
    if set(features) != allowed:
        raise ValueError("Unexpected adapter feature keys")
    base = int(params.get("horizon", 3))
    h, variance, retain = base, params.get("variance", 1.0), params.get("retain", 0.0)
    if params.get("adaptive") and features["replan"] > 0:
        uncertain = features["prediction_error"] > params.get("error_threshold", 0.25)
        stalled = features["improvement"] < params.get("stall_threshold", 0.02)
        if uncertain:
            h = int(params.get("repair_horizon", 2))
        if stalled and (not uncertain or params.get("joint_response", False)):
            if not uncertain:
                h = int(params.get("stall_horizon", 3))
            variance *= params.get("exploration_multiplier", 1.5)
            retain = params.get("stall_retain", 0.25)
    return h, min(2.0, max(0.5, float(variance))), min(0.5, max(0.0, float(retain)))


class PlanningCEM(CEMSolver):
    """Upstream CEM-compatible native path; all optional mutations are explicitly selected."""

    def __init__(self, model, env, params, seed, device):
        canonical = CANONICAL[env]
        super().__init__(
            model=model,
            batch_size=1,
            num_samples=canonical["samples"],
            n_steps=int(params.get("iterations", 30)),
            topk=max(2, round(canonical["samples"] * canonical["elite"])),
            device=device,
            seed=seed,
        )
        self.kind, self.params = env, params
        self.canonical_samples, self.elite_fraction = canonical["samples"], canonical["elite"]
        self.previous_candidates = None
        self.previous_cost, self.expected_next = None, None
        self.smoothed_improvement = None
        self.previous_feasible = 1.0
        self.replans = []
        self._active_horizon = 5

    @property
    def horizon(self):
        return self._active_horizon

    def embed(self, info):
        # Only task-side tensors are forwarded to the frozen model.
        task_info = {k: info[k].to(self.device, self.dtype) for k in ("pixels", "goal", "action")}
        current = self.model.base.encode({"pixels": task_info["pixels"]})["emb"]
        target = self.model.base.encode({"pixels": task_info["goal"]})["emb"]
        self.model.counters["encoder_frames"] += 2
        return task_info, current, target

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        task_info, current, target = self.embed(info_dict)
        current_cost = float((current[:, -1] - target[:, -1]).square().sum())
        improvement = (
            0.0
            if self.previous_cost is None
            else ((self.previous_cost - current_cost) / max(self.previous_cost, 1e-6))
        )
        if self.previous_cost is not None:
            alpha = float(self.params.get("progress_smoothing", 1.0))
            if not 0 < alpha <= 1:
                raise ValueError("Progress smoothing must lie in (0, 1]")
            self.smoothed_improvement = (
                improvement
                if self.smoothed_improvement is None
                else alpha * improvement + (1 - alpha) * self.smoothed_improvement
            )
            improvement = self.smoothed_improvement
        error = (
            0.0
            if self.expected_next is None
            else float((current[0, -1] - self.expected_next.to(self.device)).square().sum())
            / max(current_cost, 1.0)
        )
        features = dict(
            current_cost=current_cost,
            improvement=improvement,
            prediction_error=error,
            previous_feasible=self.previous_feasible,
            replan=len(self.replans),
        )
        h, variance, retain = adapter_mode(features, self.params)
        self._active_horizon = h
        # Fixed standard predictor-work ceiling; shorter horizons may use more samples.
        work = (
            self.canonical_samples * self.n_steps * 5 * float(self.params.get("budget_scale", 1.0))
        )
        n = max(2, int(work // (self.n_steps * h)))
        k = max(2, min(n, round(n * self.params.get("elite", self.elite_fraction))))
        mean = torch.zeros(1, h, self.action_dim, device=self.device, dtype=self.dtype)
        if init_action is not None:
            kept = min(h, init_action.shape[1])
            mean[:, :kept] = init_action[:, :kept].to(self.device, self.dtype)
        var = torch.full_like(mean, variance)
        expanded = {
            key: value.unsqueeze(1).expand(1, n, *value.shape[1:])
            for key, value in task_info.items()
        }
        expanded["emb"] = current.unsqueeze(1).expand(1, n, *current.shape[1:])
        expanded["goal_emb"] = target
        memory = None
        if retain and self.previous_candidates is not None:
            old = self.previous_candidates.to(self.device)
            memory = torch.zeros(1, old.shape[1], h, self.action_dim, device=self.device)
            take = min(h, old.shape[2] - 1)
            memory[:, :, :take] = old[:, :, 1 : 1 + take]
        self.model.best = None
        retained_counts = []
        for iteration in range(self.n_steps):
            noise = torch.randn(
                1,
                n,
                h,
                self.action_dim,
                generator=self.torch_gen,
                device=self.device,
                dtype=self.dtype,
            )
            rho = float(self.params.get("correlate", 0.0))
            if rho:
                for t in range(1, h):
                    noise[:, :, t] = (
                        rho * noise[:, :, t - 1] + math.sqrt(1 - rho * rho) * noise[:, :, t]
                    )
            candidates = noise * var.unsqueeze(1) + mean.unsqueeze(1)
            candidates[:, 0] = mean
            count = 0
            if memory is not None and retain:
                count = min(memory.shape[1], int(n * retain), n - 1)
                candidates[:, 1 : count + 1] = memory[:, :count]
            retained_counts.append(count)
            costs = self.model.get_cost(expanded, candidates)
            idx = costs.topk(k, dim=1, largest=False).indices
            elite = candidates[torch.arange(1, device=self.device)[:, None], idx]
            mean, var = elite.mean(1), elite.std(1)
            feasible_idx = torch.where(self.model.last["feasible"][0])[0]
            archive_count = min(n // 2, len(feasible_idx))
            archive_idx = feasible_idx[costs[0, feasible_idx].argsort()[:archive_count]]
            memory = candidates[:, archive_idx].detach()
        self.previous_candidates = memory.cpu()
        self.previous_feasible = float(self.model.last["feasible"].float().mean())
        proposed = mean.detach().clone()
        chosen, provenance = proposed, "mean"
        violation, _, _ = self.model.checker.check(chosen[:, None])
        if self.params["checked"] and bool((violation > 0).any()):
            if self.model.best is None:
                self.replans.append({"features": features, "horizon": h, "executed": "none"})
                raise NoSafeAction("No checked plan is available; no unsafe fallback executed")
            chosen = self.model.best[1].to(self.device, self.dtype)
            provenance = "best_checked"
            violation, _, _ = self.model.checker.check(chosen[:, None])
            if bool((violation > 0).any()):
                raise NoSafeAction("Candidate failed its execution-time recheck")
        if self.params["checked"] and self.model.checker.kind == "cube":
            shortlist = [chosen]
            if self.model.best is not None:
                shortlist.append(self.model.best[1].to(self.device, self.dtype))
            shortlist.extend(memory[:, i] for i in range(min(6, memory.shape[1])))
            accepted = None
            for candidate in shortlist[:8]:
                if self.model.checker.execution_feasible(candidate):
                    accepted = candidate
                    break
            if accepted is None:
                raise NoSafeAction("No candidate prefix passes the simulator check")
            if not torch.equal(accepted, chosen):
                provenance = "simulator_checked_alternative"
            chosen = accepted
        # Measure the actually chosen block's predicted latent for the next observed frame.
        single = {key: value.unsqueeze(1) for key, value in task_info.items()}
        single.update(emb=current.unsqueeze(1), goal_emb=target)
        self.model.base.get_cost(single, chosen[:, None])
        self.model.counters["model_batch_calls"] += 1
        self.model.counters["model_candidate_rollouts"] += 1
        self.model.counters["predictor_steps"] += h
        self.expected_next = single["predicted_emb"][0, 0, 1].detach().cpu()
        self.previous_cost = current_cost
        self.replans.append(
            {
                "features": features,
                "horizon": h,
                "variance": variance,
                "retain": retain,
                "samples": n,
                "elite_count": k,
                "retained_counts": retained_counts,
                "executed": provenance,
                "executed_violation": float(violation.max()),
                "proposed_block": proposed[0, 0].cpu().tolist(),
                "chosen_block": chosen[0, 0].cpu().tolist(),
            }
        )
        # Keep the upstream policy's warm-start buffer shape stable across horizon switches.
        padded = torch.zeros(1, 5, self.action_dim, dtype=self.dtype)
        padded[:, :h] = chosen.detach().cpu()
        return {"actions": padded}
