"""Learned clearance as a SafeCEM checker; shadow mode retains the reference decisions."""

from __future__ import annotations

import torch

from helpers.lewm18_planner import PlanningCEM
from helpers.safety_model import chunk_features


class LearnedSafetyChecker:
    def __init__(
        self, model, world_model, constraints, margin, counters, *, mode="shadow", reference=None
    ):
        if mode not in ("shadow", "learned"):
            raise ValueError("Checker mode must be shadow or learned")
        if mode == "shadow" and reference is None:
            raise ValueError("Shadow mode requires an explicit reference checker")
        if mode == "learned" and reference is not None:
            raise ValueError("Learned mode must not silently use a privileged checker")
        if not torch.isfinite(torch.tensor(margin)) or margin < 0:
            raise ValueError("Margin must be finite and nonnegative")
        if mode == "learned" and (
            not model.calibration
            or not model.calibration["ready"]
            or not torch.isfinite(model.error_allowance)
        ):
            raise ValueError("Learned execution requires a finite held-out calibration allowance")
        self.model, self.world_model = model.eval().requires_grad_(False), world_model
        self.constraints, self.margin, self.counters = constraints, margin, counters
        self.mode, self.reference = mode, reference
        self.kind = model.spec.environment
        self.context, self.last = None, None

    @property
    def state(self):
        return None if self.reference is None else self.reference.state

    @state.setter
    def state(self, value):
        # Ground-truth state goes only to the reference, never into the learned features.
        if self.reference is not None:
            self.reference.state = value

    def bind_context(self, info, current, previous, history_valid):
        self.context = {
            "info": info,
            "current": current[:, -1].detach(),
            "previous": previous[:, -1].detach(),
            "previous_action": info["action"][:, -1].detach(),
            "history_valid": history_valid.detach(),
        }

    @torch.inference_mode()
    def check_with_rollout(self, info, candidates):
        if self.context is None:
            raise RuntimeError("Observation context must be bound before safety checking")
        ctx, spec = self.context, self.model.spec
        features = chunk_features(
            spec,
            ctx["current"],
            ctx["previous"],
            info["predicted_emb"],
            candidates,
            ctx["previous_action"],
            ctx["history_valid"],
            self.constraints,
        )
        parameter = next(self.model.parameters())
        predicted = self.model.physical_clearance(features.to(parameter.device, parameter.dtype))
        if not torch.isfinite(predicted).all():
            raise ValueError("Safety predictor returned a nonfinite clearance")
        predicted = predicted.to(candidates.device)
        conservative = predicted - self.model.error_allowance.to(candidates.device)
        # A tiny numerical guard keeps equality with the requested boundary conservative.
        guard = 1e-3 if spec.units == "px" else 1e-6
        violation = (self.margin + guard - conservative).clamp_min(0).sum((-1, -2))
        b, s = candidates.shape[:2]
        self.counters["learned_candidate_checks"] += b * s
        self.counters["safety_head_rows"] += features.numel() // spec.input_dim
        self.last = {
            "predicted_clearance": predicted.detach(),
            "conservative_clearance": conservative,
            "violation": violation,
        }
        if self.mode == "shadow":
            actual, path, geometric = self.reference.check(candidates)
            raw_safe = (predicted >= self.margin + guard).all((-1, -2))
            accepted = violation <= 0
            safe = actual <= 0
            self.counters["shadow_reference_candidates"] += b * s
            self.counters["shadow_raw_false_safe"] += int((raw_safe & ~safe).sum())
            self.counters["shadow_conservative_false_safe"] += int((accepted & ~safe).sum())
            self.counters["shadow_conservative_false_rejected"] += int((~accepted & safe).sum())
            self.last["reference_violation"] = actual.detach()
            return actual, path, geometric
        return violation, candidates.new_empty(b, s, 0, 2), candidates.new_zeros(b, s)

    @torch.inference_mode()
    def check(self, candidates):
        """A final mean/rechecked candidate needs its own imagination, never a cached score."""
        if self.context is None:
            raise RuntimeError("Missing observation context")
        b, s, h = candidates.shape[:3]
        info = {
            key: value.unsqueeze(1).expand(b, s, *value.shape[1:])
            for key, value in self.context["info"].items()
        }
        info["emb"] = self.context["current"][:, None, None].expand(b, s, 1, -1)
        self.world_model.rollout(info, candidates)
        self.counters["model_batch_calls"] += 1
        self.counters["model_candidate_rollouts"] += b * s
        self.counters["predictor_steps"] += b * s * h
        return self.check_with_rollout(info, candidates)


class SafetyPlanningCEM(PlanningCEM):
    """Existing search algorithm with an additional, frozen previous-image encoding."""

    def embed(self, info):
        task, current, target = super().embed(info)
        if "pixels_previous" not in info or "safety_history_valid" not in info:
            raise ValueError(
                "Supply the previous observed frame and an explicit history-valid flag"
            )
        previous_pixels = info["pixels_previous"].to(self.device, self.dtype)
        previous = self.model.base.encode({"pixels": previous_pixels})["emb"]
        self.model.counters["encoder_frames"] += len(previous_pixels)
        history = info["safety_history_valid"].to(self.device, self.dtype).reshape(-1)
        self.model.checker.bind_context(task, current, previous, history)
        return task, current, target
