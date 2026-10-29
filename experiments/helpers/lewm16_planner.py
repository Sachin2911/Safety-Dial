"""Frozen LeWM costs, independently counted exact checks, and compact CEM adaptation."""

from __future__ import annotations

import math

import numpy as np
import torch
from stable_worldmodel.solver import CEMSolver

from helpers.lewm16_protocol import CANONICAL
from helpers.reacher13_geometry import fingertip
from helpers.safecem10_planner import box_penetration, expand_box, rank_costs
from helpers.tworoom12_planner import tworoom_path


class NoSafeAction(RuntimeError):
    """An unfinished, explicitly censored episode, not a successful safety outcome."""


class ExactCheck:
    def __init__(self, env, spec, scaler, boxes, margin, counters, arm=None):
        self.env, self.spec, self.scaler = env, spec, scaler
        self.boxes, self.margin, self.counters, self.arm = boxes, margin, counters, arm
        self.guard = 0.001 if env == "tworoom" else 0.000001
        self.state = None
        self.goal = None

    def raw(self, candidates):
        a = candidates.detach().cpu().double().reshape(*candidates.shape[:2], -1, 2)
        return a * torch.as_tensor(self.scaler.scale_) + torch.as_tensor(self.scaler.mean_)

    def paths(self, candidates):
        raw = self.raw(candidates)
        b, s, m, _ = raw.shape
        self.counters["checker_candidate_rollouts"] += b * s
        if self.env == "tworoom":
            path = tworoom_path(self.state[:2], raw, self.spec["environment"]["geometry"])
            self.counters["analytic_path_steps"] += b * s * m
            ends = path[:, :, -1, :].numpy()
            geometric = np.linalg.norm(ends - np.asarray(self.goal), axis=-1)
        else:
            from mujoco import rollout as mj_rollout

            controls = np.ascontiguousarray(raw.numpy().reshape(b * s, m, 2))
            init = np.zeros((b * s, self.arm.nstate))
            init[:, 1:5] = np.asarray(self.state)[:4]
            held = np.repeat(controls, self.arm.repeat, axis=1)
            # This is counterfactual simulator work, charged separately from executed steps.
            self.counters["checker_physics_steps"] += b * s * m * self.arm.repeat
            out, _ = mj_rollout.rollout(self.arm.model, self.arm.datas, init, held)
            q = out[:, :, 1:3]
            xy = fingertip(q.reshape(-1, 2), self.arm.links).reshape(b, s, -1, 2)
            path = torch.from_numpy(xy)
            geometric = np.abs(q[:, -1] - np.asarray(self.goal)).max(axis=-1).reshape(b, s)
        return path, torch.from_numpy(np.asarray(geometric))

    def check(self, candidates):
        path, geometric = self.paths(candidates)
        violation = torch.zeros(path.shape[:2], dtype=torch.float64)
        for box in self.boxes:
            violation += box_penetration(path, expand_box(box, self.margin + self.guard)).sum(-1)
        return violation.to(candidates.device), path, geometric.to(candidates.device)


class CostModel(torch.nn.Module):
    def __init__(self, base, checker, params, counters, diagnostic=False):
        super().__init__()
        self.base, self.checker, self.params, self.counters = base, checker, params, counters
        self.history, self.snapshots = [], []
        self.diagnostic = diagnostic
        self.best = None
        self.last = None

    def get_cost(self, info, candidates):
        b, s, h, _ = candidates.shape
        self.counters["model_candidate_rollouts"] += b * s
        self.counters["predictor_steps"] += b * s * h
        learned = self.base.get_cost(info, candidates)
        violation, path, geometric = self.checker.check(candidates)
        goal = geometric.to(learned.dtype) if self.params["objective"] == "geometric" else learned
        cost, feasible = rank_costs(
            goal, violation.to(goal.dtype), "safe" if self.params["checked"] else "penalty"
        )
        i = int(cost[0].argmin())
        if bool(feasible[0, i]):
            score = float(goal[0, i])
            if self.best is None or score < self.best[0]:
                self.best = (score, candidates[:, i].detach().cpu().clone())
        self.last = dict(
            learned=learned.detach(),
            feasible=feasible,
            geometric=geometric,
            current=info["emb"][0, 0, -1].detach(),
            goal_emb=info["goal_emb"][0, -1].detach(),
        )
        row = {
            "goal_min": float(learned.min()),
            "goal_median": float(learned.median()),
            "frac_feasible": float(feasible.float().mean()),
            "elite_feasible": bool(feasible[0, i]),
            "elite_goal": float(learned[0, i]),
            "elite_geometric": float(geometric[0, i]),
            "elite_xy": path[0, i, :: max(1, len(path[0, i]) // 5)].tolist(),
            "objective": self.params["objective"],
            "horizon": h,
            "samples": s,
        }
        self.history.append(row)
        if self.diagnostic and (len(self.history) - 1) % 10 in (0, 9):
            # Deterministic indices, not outcome-selected logging. Top 8 supplement them.
            uniform = np.linspace(0, s - 1, min(32, s)).round().astype(int)
            top = torch.topk(cost[0], min(8, s), largest=False).indices.cpu().numpy()
            idx = np.unique(np.r_[uniform, top])
            self.snapshots.append(
                {
                    "iteration": len(self.history) - 1,
                    "indices": idx.tolist(),
                    "learned": learned[0, idx].cpu().tolist(),
                    "geometric": geometric[0, idx].cpu().tolist(),
                    "feasible": feasible[0, idx].cpu().tolist(),
                    "paths": path[0, idx].tolist(),
                }
            )
        return cost


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


class CampaignCEM(CEMSolver):
    """Upstream CEM-compatible native path; all optional mutations are explicitly selected."""

    def __init__(self, model, env, params, seed, device):
        canonical = CANONICAL[env]
        super().__init__(
            model=model,
            batch_size=1,
            num_samples=canonical["samples"],
            n_steps=10,
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
        work = self.canonical_samples * 10 * 5 * float(self.params.get("budget_scale", 1.0))
        n = max(2, int(work // (10 * h)))
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
        for iteration in range(10):
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
        # Measure the actually chosen block's predicted latent for the next observed frame.
        single = {key: value.unsqueeze(1) for key, value in task_info.items()}
        single.update(emb=current.unsqueeze(1), goal_emb=target)
        self.model.base.get_cost(single, chosen[:, None])
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
