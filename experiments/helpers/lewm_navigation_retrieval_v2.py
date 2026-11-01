"""Local observed-target selection with causal persistence and frozen visual progress.

This prospective family addresses the reached-target attractors seen in v1. The
planner still acts towards an actual observed image embedding, and the original
physical goal and learned safety checker retain their authority. A memory-only
Ridge readout supplies an estimate, never the query episode's true goal position.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch

from helpers.lewm16_planner import NoSafeAction
from helpers.lewm_navigation_retrieval import (
    RetrievalController as LegacyController,
    RetrievalMemory,
    RetrievalPlanningCEM,
)

__all__ = [
    "FrozenGoalReadout",
    "RetrievalController",
    "RetrievalMemory",
    "RetrievalPlanningCEM",
]


class FrozenGoalReadout:
    """The qualified raw-embedding Ridge formula, evaluated in float64."""

    def __init__(self, coefficient, intercept, *, device="cuda", fingerprint=None):
        coefficient = np.asarray(coefficient, dtype=np.float64)
        intercept = np.asarray(intercept, dtype=np.float64)
        if coefficient.ndim != 2 or coefficient.shape[1] != 2 or intercept.shape != (2,):
            raise ValueError("Readout requires a latent-by-XY coefficient and XY intercept")
        if not np.isfinite(coefficient).all() or not np.isfinite(intercept).all():
            raise ValueError("Frozen visual readout coefficients must be finite")
        self.coefficient = torch.as_tensor(coefficient, dtype=torch.float64, device=device)
        self.intercept = torch.as_tensor(intercept, dtype=torch.float64, device=device)
        self.fingerprint, self.calls = fingerprint, 0

    @classmethod
    def from_npz(cls, path, *, device="cuda"):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            coefficient = data["raw_feature_coefficient"].copy()
            intercept = data["raw_feature_intercept"].copy()
        with path.open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        return cls(coefficient, intercept, device=device, fingerprint=fingerprint)

    def __call__(self, embedding):
        if embedding.shape[-1] != self.coefficient.shape[0]:
            raise ValueError("Readout and frozen embedding dimensions differ")
        self.calls += 1
        value = embedding.to(device=self.coefficient.device, dtype=torch.float64)
        prediction = value @ self.coefficient + self.intercept
        if not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite frozen visual-goal prediction")
        return prediction


class RetrievalController(LegacyController):
    """Ten evolved genes, shared unchanged by fixed, random-search and REINFORCE arms.

    Source-latent match has coefficient one. Nine signed bounded genes weight the
    remaining observed-target features; the tenth exponentially weights the direct
    original-goal latent cost. Signed linear genes avoid zero-gradient square
    parameterization. No geometric trajectory or action is prescribed by the genes.
    """

    parameter_count = 10
    feature_names = (
        *LegacyController.feature_names[:7],
        "successor_current_xy_div_max_distance",
        "recent_target_similarity_and_stall",
        "estimated_goal_distance_change_div_max_distance",
        "direct_goal_latent",
    )
    default_config = {
        "min_target_distance": 6.0,
        "max_target_distance": 30.0,
        "recent_window": 6,
        "recent_scale": 6.0,
        "stall_distance": 3.0,
        "readout_progress": True,
        "direct_goal_locality": True,
    }

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
        goal_readout=None,
        controller_config=None,
    ):
        # Reuse the immutable v1 geometry and bookkeeping initialization only.
        super().__init__(
            memory,
            LegacyController.initial_parameters(),
            shape,
            body_radius=body_radius,
            margin=margin,
            allowance=allowance,
            stochastic=stochastic,
            temperature=temperature,
            generator=generator,
        )
        theta = np.asarray(parameters, dtype=np.float64)
        if theta.shape != (self.parameter_count,) or not np.isfinite(theta).all():
            raise ValueError("Ten finite persistent retrieval-v2 parameters are required")
        self.parameters = theta.copy()
        supplied = controller_config or {}
        unknown = set(supplied) - set(self.default_config)
        if unknown:
            raise ValueError("Unknown retrieval-v2 settings: " + ", ".join(sorted(unknown)))
        self.config = {**self.default_config, **supplied}
        c = self.config
        if not 0 < c["min_target_distance"] < c["max_target_distance"]:
            raise ValueError("Observed targets require a positive ordered distance band")
        if (
            not isinstance(c["recent_window"], int)
            or c["recent_window"] < 1
            or c["recent_scale"] <= 0
            or c["stall_distance"] <= 0
        ):
            raise ValueError("Invalid causal persistence settings")
        for key in ("readout_progress", "direct_goal_locality"):
            if not isinstance(c[key], bool):
                raise ValueError("Readout architecture switches must be explicit booleans")
        if (c["readout_progress"] or c["direct_goal_locality"]) and goal_readout is None:
            raise ValueError("The declared progress/local-final architecture requires a readout")
        self.goal_readout = goal_readout
        self.readout_queries = 0
        self.recent_targets, self.previous_current_xy, self.refusals = [], None, []
        self._last_context, self._last_estimated_goal = None, None

    @staticmethod
    def initial_parameters():
        return np.array([0.25, 0.25, 0.1, 0.25, 0.1, 2.0, 0.25, 2.0, 4.0, np.log(4.0)])

    @staticmethod
    def fixed_ap_parameters():
        """Compatibility name: this is the strong geometry/progress fixed baseline."""
        return RetrievalController.initial_parameters()

    @staticmethod
    def weights(parameters, reference):
        theta = torch.as_tensor(parameters, dtype=reference.dtype, device=reference.device)
        if theta.shape != (10,) or not torch.isfinite(theta).all():
            raise ValueError("Malformed ten-gene retrieval-v2 genotype")
        theta = theta.clamp(-6, 6)
        return torch.cat([theta.new_ones(1), theta[:9], theta[9:].exp()])

    def context(self):
        return {
            "recent_target_xy": [list(point) for point in self.recent_targets],
            "previous_current_xy": self.previous_current_xy,
        }

    def _recent_feature(self, targets, current_xy, context):
        prior = context["recent_target_xy"]
        if not prior:
            return targets.new_zeros(len(targets))
        points = targets.new_tensor(prior)
        distance = torch.linalg.vector_norm(targets[:, None] - points[None], dim=-1)
        similarity = torch.exp(-0.5 * (distance / self.config["recent_scale"]).square())
        penalty = similarity.amax(-1)
        previous = context["previous_current_xy"]
        if previous is not None:
            displacement = torch.linalg.vector_norm(current_xy - current_xy.new_tensor(previous))
            stall = (1.0 - displacement / self.config["stall_distance"]).clamp(0, 1)
            penalty = penalty + stall * similarity[:, -1]
        return penalty

    def features(self, current, final, current_xy, *, context=None):
        """Replayable features use only the saved causal prefix and frozen assets."""
        feature, available = super().features(current, final, current_xy)
        c = self.config
        xy = torch.as_tensor(current_xy, device=feature.device, dtype=feature.dtype).reshape(2)
        successors = self.memory.tables["successor_xy"].to(feature)
        distance = torch.linalg.vector_norm(successors - xy, dim=-1)
        available[:-1] &= (distance >= c["min_target_distance"]) & (
            distance <= c["max_target_distance"]
        )
        feature[:-1, 7] = distance / c["max_target_distance"]
        prefix = self.context() if context is None else context
        recent = self._recent_feature(successors, xy, prefix)
        progress = distance.new_zeros(self.memory.count)
        direct_recent, direct_progress = distance.new_zeros(()), distance.new_zeros(())
        estimated_goal = None
        if self.goal_readout is not None:
            estimated_goal = self.goal_readout(final.reshape(-1)).to(feature).reshape(2)
            self.readout_queries += 1
            goal_distance = torch.linalg.vector_norm(estimated_goal - xy)
            if c["readout_progress"]:
                progress = (
                    (torch.linalg.vector_norm(successors - estimated_goal, dim=-1) - goal_distance)
                    / c["max_target_distance"]
                ).clamp(-1, 1)
                direct_progress = (-goal_distance / c["max_target_distance"]).clamp(-1, 1)
            direct_recent = self._recent_feature(estimated_goal[None], xy, prefix)[0]
            feature[-1, 7] = goal_distance / c["max_target_distance"]
            if c["direct_goal_locality"]:
                available[-1] &= goal_distance <= c["max_target_distance"]
        feature = torch.cat(
            [
                feature[:, :8],
                torch.cat([recent, direct_recent[None]])[:, None],
                torch.cat([progress, direct_progress[None]])[:, None],
                feature[:, 8:],
            ],
            -1,
        )
        if not torch.isfinite(feature).all():
            raise ValueError("Nonfinite retrieval-v2 features")
        self._last_context = prefix
        self._last_estimated_goal = estimated_goal
        return feature, available

    def select(self, current, final, current_xy):
        previous_queries = self.readout_queries
        features, available = self.features(current, final, current_xy)
        self.last_features, self.last_available = features, available
        feature_array = features.detach().cpu().contiguous().numpy()
        available_array = available.detach().cpu().contiguous().numpy()
        provenance = {
            "features_sha256": hashlib.sha256(feature_array.tobytes()).hexdigest(),
            "available_sha256": hashlib.sha256(available_array.tobytes()).hexdigest(),
            "features_shape": list(feature_array.shape),
            "features_dtype": feature_array.dtype.str,
            "available_dtype": available_array.dtype.str,
            "current_latent": current.detach().cpu().reshape(-1).tolist(),
            "final_latent": final.detach().cpu().reshape(-1).tolist(),
            "current_xy": torch.as_tensor(current_xy).detach().cpu().reshape(2).tolist(),
        }
        if not bool(available.any()):
            self.refusals.append({**provenance, "causal_context": self._last_context})
            raise NoSafeAction("No admitted local observed target or local original goal")
        logits = self.logits(features, available, self.parameters, self.temperature)
        selected = (
            int(torch.multinomial(logits.softmax(-1), 1, generator=self.generator))
            if self.stochastic
            else int(logits.argmax())
        )
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
            **provenance,
            "query": len(self.history),
            "target_kind": "original_goal" if original else "observed_successor",
            "memory_index": None if original else selected,
            "memory_sha256": self.memory.fingerprint,
            "available_observed_targets": int(available[:-1].sum()),
            "selected_features": features[selected].detach().cpu().tolist(),
            "selected_score": float(-logits[selected] * self.temperature),
            "top_target_indices": order.detach().cpu().tolist(),
            "stochastic": self.stochastic,
            "temperature": self.temperature,
        }
        if not original:
            row["source_provenance"] = {
                k: int(v[selected]) for k, v in self.memory.metadata.items()
            }
        estimated = self._last_estimated_goal
        row.update(
            controller_version="observed-target-v2",
            causal_context=self._last_context,
            estimated_goal_xy=None if estimated is None else estimated.detach().cpu().tolist(),
            goal_readout_sha256=None
            if self.goal_readout is None
            else self.goal_readout.fingerprint,
            goal_readout_queries=self.readout_queries - previous_queries,
            direct_goal_available=bool(available[-1]),
        )
        selected = row["memory_index"]
        point = (
            estimated
            if selected is None
            else self.memory.tables["successor_xy"][selected].to(features)
        )
        row["selected_target_xy_estimate"] = (
            None if point is None else point.detach().cpu().tolist()
        )
        self.history.append(row)
        if point is not None:
            self.recent_targets.append(point.detach().cpu().tolist())
            self.recent_targets = self.recent_targets[-self.config["recent_window"] :]
        self.previous_current_xy = torch.as_tensor(current_xy).detach().cpu().reshape(2).tolist()
        return target

    def artifact(self):
        return {
            **super().artifact(),
            "format": "lewm-observed-target-retrieval-v2",
            "parameter_count": self.parameter_count,
            "controller_config": self.config,
            "weight_parameterization": "one fixed source coefficient, nine bounded signed linear genes, exp(tenth) direct-goal coefficient",
            "fixed_baseline": "fixed_geometry_progress, compatibility API fixed_ap_parameters",
            "goal_readout_sha256": None
            if self.goal_readout is None
            else self.goal_readout.fingerprint,
            "goal_readout_arithmetic": "torch.float64, raw frozen embedding @ raw coefficient + raw intercept",
            "refusals": self.refusals,
            "observations": "Frozen live/final visual embeddings, real observed memory, causal current XY/history and frozen memory-trained visual-goal estimate. No true query-goal XY.",
            "locality": "Observed successors 6..30px by causal current XY; direct original goal only within readout-estimated 30px when the declared guard is enabled.",
        }
