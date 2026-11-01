"""Compact evolutionary selection of observed targets for a frozen LeWM planner.

The memory contains real source, short-successor and endpoint observations. This is
an AP-inspired retrieval family, not a claim to reproduce the complete AP algorithm.
No simulator previews, invented images, interpolated latents or final-goal coordinates
enter the controller. The physical task's original goal remains its success rule.
"""

import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from helpers.lewm18_planner import PlanningCEM
from helpers.portable_motion import clearance, shape_constraints


class RetrievalMemory:
    required = (
        "source_embeddings",
        "successor_embeddings",
        "endpoint_embeddings",
        "latent_scale",
        "source_xy",
        "successor_xy",
        "endpoint_xy",
        "source_episode",
        "source_step",
        "successor_step",
        "endpoint_step",
    )

    def __init__(self, arrays, *, device="cuda", fingerprint=None):
        missing = set(self.required) - set(arrays)
        if missing:
            raise ValueError("Observed-memory fields missing: " + ", ".join(sorted(missing)))
        source = np.asarray(arrays["source_embeddings"])
        if source.ndim != 2 or not len(source):
            raise ValueError("Memory source embeddings must have shape (records,latent_features)")
        count, dimension = source.shape
        for key in ("source_embeddings", "successor_embeddings", "endpoint_embeddings"):
            value = np.asarray(arrays[key])
            if value.shape != source.shape or not np.isfinite(value).all():
                raise ValueError("All observed embeddings must have the same finite shape")
        for key in ("source_xy", "successor_xy", "endpoint_xy"):
            if np.asarray(arrays[key]).shape != (count, 2) or not np.isfinite(arrays[key]).all():
                raise ValueError("Memory body positions must be finite XY observations")
        for key in ("source_episode", "source_step", "successor_step", "endpoint_step"):
            if np.asarray(arrays[key]).shape != (count,):
                raise ValueError("Memory provenance must identify each observed record")
        if not bool((np.asarray(arrays["source_step"]) < arrays["successor_step"]).all()):
            raise ValueError("A short successor must occur after its source")
        if not bool((np.asarray(arrays["successor_step"]) <= arrays["endpoint_step"]).all()):
            raise ValueError("A source endpoint cannot precede its short successor")
        scale = np.asarray(arrays["latent_scale"])
        if scale.shape != (dimension,) or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Positive finite latent standardization is required")
        self.count, self.dimension, self.fingerprint = count, dimension, fingerprint
        self.tables = {
            key: torch.as_tensor(np.asarray(arrays[key]), dtype=torch.float32, device=device)
            for key in (
                "source_embeddings",
                "successor_embeddings",
                "endpoint_embeddings",
                "latent_scale",
                "source_xy",
                "successor_xy",
                "endpoint_xy",
            )
        }
        self.metadata = {
            key: np.asarray(arrays[key]).copy()
            for key in ("source_episode", "source_step", "successor_step", "endpoint_step")
        }
        self.tables["successor_offset"] = torch.as_tensor(
            self.metadata["successor_step"] - self.metadata["source_step"],
            dtype=torch.float32,
            device=device,
        )

    @classmethod
    def from_npz(cls, path, *, device="cuda"):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            arrays = dict(data)
        with path.open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        return cls(arrays, device=device, fingerprint=fingerprint)


class RetrievalController:
    """Eight persistent parameters; fixed source-distance coefficient removes a scale tie.

    Seven squared parameters weight endpoint match, transition match, local latent
    length, successor-goal match, successor offset, straight-line hazard deficit and
    current-to-successor XY distance. The eighth sets the direct original-goal cost.
    A target is either a cached observed successor or the original goal observation.
    """

    parameter_count = 8
    feature_names = (
        "source_current_latent",
        "endpoint_goal_latent",
        "transition_goal_latent",
        "successor_current_latent",
        "successor_goal_latent",
        "successor_offset_div_10",
        "straight_line_margin_deficit",
        "successor_current_xy_div_224",
        "direct_goal_latent",
    )

    def __init__(
        self,
        memory,
        parameters,
        shape,
        *,
        body_radius=7.0,
        margin=3.0,
        allowance=0.0,
        stochastic=False,
        temperature=1.0,
        generator=None,
    ):
        theta = np.asarray(parameters, dtype=np.float64)
        if theta.shape != (8,) or not np.isfinite(theta).all():
            raise ValueError("Eight finite persistent retrieval parameters are required")
        if body_radius <= 0 or margin < 0 or allowance < 0 or temperature <= 0:
            raise ValueError("Invalid declared geometry or policy temperature")
        self.memory, self.parameters, self.shape = memory, theta.copy(), shape
        self.body_radius, self.margin, self.allowance = body_radius, margin, allowance
        self.constraints = shape_constraints(shape)
        self.stochastic, self.temperature, self.generator = stochastic, temperature, generator
        self.history, self.training_decisions = [], []
        self.last_features = self.last_available = None

    @staticmethod
    def initial_parameters():
        return np.array([1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.1, np.log(4.0)])

    @staticmethod
    def fixed_ap_parameters():
        # Anchored retrieval metric plus an explicit final-goal competitor.
        return np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.log(4.0)])

    @staticmethod
    def weights(parameters, reference):
        theta = torch.as_tensor(parameters, dtype=reference.dtype, device=reference.device)
        if theta.shape != (8,) or not torch.isfinite(theta).all():
            raise ValueError("Malformed retrieval genotype")
        theta = theta.clamp(-6, 6)
        return torch.cat([theta.new_ones(1), theta[:7].square(), theta[7:].exp()])

    def features(self, current, final, current_xy):
        m = self.memory
        cur, goal = current.reshape(-1), final.reshape(-1)
        if cur.numel() != m.dimension or goal.numel() != m.dimension:
            raise ValueError("Live query and observed memory latent dimensions differ")
        tables = {k: v.to(device=cur.device, dtype=cur.dtype) for k, v in m.tables.items()}
        s, nxt, end = [
            tables[k] for k in ("source_embeddings", "successor_embeddings", "endpoint_embeddings")
        ]
        scale = tables["latent_scale"]

        def mean_square(value):
            return value.square().mean(-1)

        xy = torch.as_tensor(current_xy, device=cur.device, dtype=cur.dtype).reshape(2)
        successors = tables["successor_xy"]
        threshold = float(self.margin + self.allowance)
        successor_clear = clearance(successors, self.constraints, self.body_radius)
        available = successor_clear >= threshold
        fractions = torch.linspace(0, 1, 9, device=cur.device, dtype=cur.dtype)
        line = xy[None, None] + fractions[None, :, None] * (successors[:, None] - xy)
        line_clear = clearance(line, self.constraints, self.body_radius).amin(-1)
        deficit = ((threshold - line_clear) / max(1.0, threshold)).clamp(0, 10)
        feature = torch.stack(
            [
                mean_square((s - cur) / scale),
                mean_square((end - goal) / scale),
                mean_square(((end - s) - (goal - cur)) / scale),
                mean_square((nxt - cur) / scale),
                mean_square((nxt - goal) / scale),
                tables["successor_offset"] / 10.0,
                deficit,
                torch.linalg.vector_norm(successors - xy, dim=-1) / 224.0,
                cur.new_zeros(m.count),
            ],
            -1,
        )
        direct = feature.new_zeros(1, 9)
        direct[0, -1] = mean_square((goal - cur) / scale)
        feature = torch.cat([feature, direct], 0)
        available = torch.cat([available, available.new_ones(1)])
        if not torch.isfinite(feature).all():
            raise ValueError("Nonfinite observed-target retrieval features")
        return feature, available

    @classmethod
    def logits(cls, features, available, parameters, temperature=1.0):
        scores = features @ cls.weights(parameters, features)
        return (-scores / temperature).masked_fill(~available, -torch.inf)

    @classmethod
    def log_probability(cls, features, available, parameters, selected, temperature=1.0):
        """Ordinary REINFORCE can differentiate all eight parameters from the same input."""
        return cls.logits(features, available, parameters, temperature).log_softmax(-1)[selected]

    def select(self, current, final, current_xy):
        features, available = self.features(current, final, current_xy)
        logits = self.logits(features, available, self.parameters, self.temperature)
        if self.stochastic:
            selected = int(torch.multinomial(logits.softmax(-1), 1, generator=self.generator))
        else:
            selected = int(logits.argmax())
        self.last_features, self.last_available = features, available
        feature_array = features.detach().cpu().contiguous().numpy()
        available_array = available.detach().cpu().contiguous().numpy()
        if self.stochastic:
            self.training_decisions.append((feature_array, available_array, selected))
        original = selected == self.memory.count
        target = (
            final
            if original
            else self.memory.tables["successor_embeddings"][selected].to(final).reshape_as(final)
        )
        finite = torch.where(available)[0]
        order = finite[logits[finite].argsort(descending=True)[:8]]
        row = {
            "query": len(self.history),
            "target_kind": "original_goal" if original else "observed_successor",
            "memory_index": None if original else selected,
            "memory_sha256": self.memory.fingerprint,
            "available_observed_targets": int(available[:-1].sum()),
            "selected_features": features[selected].detach().cpu().tolist(),
            "selected_score": float(-logits[selected] * self.temperature),
            "top_target_indices": order.detach().cpu().tolist(),
            "current_latent": current.detach().cpu().reshape(-1).tolist(),
            "final_latent": final.detach().cpu().reshape(-1).tolist(),
            "current_xy": torch.as_tensor(current_xy).detach().cpu().reshape(-1).tolist(),
            "stochastic": self.stochastic,
            "temperature": self.temperature,
            "features_sha256": hashlib.sha256(feature_array.tobytes()).hexdigest(),
            "available_sha256": hashlib.sha256(available_array.tobytes()).hexdigest(),
            "features_shape": list(feature_array.shape),
            "features_dtype": feature_array.dtype.str,
            "available_dtype": available_array.dtype.str,
        }
        if not original:
            row["source_provenance"] = {
                k: int(v[selected]) for k, v in self.memory.metadata.items()
            }
        self.history.append(row)
        return target

    def artifact(self):
        return {
            "format": "lewm-observed-target-retrieval-v1",
            "parameters": self.parameters.tolist(),
            "parameter_count": 8,
            "feature_names": list(self.feature_names),
            "memory_sha256": self.memory.fingerprint,
            "observations": "Frozen live/final visual embeddings, observed memory triples and causal current XY. No final-goal coordinates.",
            "body_radius": self.body_radius,
            "margin": self.margin,
            "allowance": self.allowance,
            "stochastic_training": self.stochastic,
            "temperature": self.temperature,
        }


class RetrievalPlanningCEM(PlanningCEM):
    """Keep live/original-goal encoder work, substitute only a cached observed target."""

    def __init__(self, model, params, seed, device, controller=None, *, proposal_adapter=None):
        # Match the portable TwoRoom family's already-used300-candidate planning budget.
        super().__init__(model, "pusht", params, seed, device, proposal_adapter=proposal_adapter)
        self.target_controller = controller

    def embed(self, info):
        task_info, current, final = super().embed(info)
        if self.target_controller is None:
            return task_info, current, final
        before = time.perf_counter()
        selected = self.target_controller.select(
            current[0, -1], final[0, -1], self.model.checker.observed[-2:]
        )
        target = final.clone()
        target[0, -1] = selected
        self.model.counters["retrieval_queries"] += 1
        self.model.counters["retrieval_candidate_targets"] += (
            self.target_controller.memory.count + 1
        )
        self.model.counters["retrieval_microseconds"] += int((time.perf_counter() - before) * 1e6)
        return task_info, current, target

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        previous = len(self.replans)
        try:
            return super().solve(info_dict, init_action)
        finally:
            if self.target_controller is not None and len(self.replans) > previous:
                self.replans[-1]["target_retrieval"] = self.target_controller.history[-1]
