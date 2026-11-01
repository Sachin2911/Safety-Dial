"""Bounded behavioural corrections and provenance for a CEM population mixture."""

from dataclasses import dataclass
import json

import torch

from helpers.lewm_proposal_adapter import FEATURE_NAMES, ProposalAdapter, ProposalSpec


@dataclass(frozen=True)
class ResidualSpec(ProposalSpec):
    def initial_parameters(self, seed):
        weights = super().initial_parameters(seed)
        # Random hidden features, zero readout: exactly the native search initially.
        weights[self.hidden_size * (len(FEATURE_NAMES) + 1) :] = 0
        return weights


class ResidualAdapter(ProposalAdapter):
    def artifact(self):
        return {**super().artifact(), "format": "lewm-warmstart-residual-v1"}

    @classmethod
    def load(cls, path, case):
        value = json.loads(path.read_text())
        if value["format"] != "lewm-warmstart-residual-v1" or value["feature_names"] != list(
            FEATURE_NAMES
        ):
            raise ValueError("Residual controller contract differs")
        return cls(ResidualSpec(**value["spec"]), value["parameters"], case)


def mix_candidates(candidates, residual, reserved, fraction):
    """Bias a fixed fraction of new samples; retain the native mean and archive slots.

    The residual shifts each current CEM centre, beginning at the shifted previous
    plan. It never changes candidate count or bypasses feasibility checking.
    """
    if not 0 < fraction <= 1 or not 1 <= reserved <= candidates.shape[1]:
        raise ValueError("Invalid mixture fraction or reserved native slots")
    if residual.shape != candidates[:, 0].shape or not torch.isfinite(residual).all():
        raise ValueError("Invalid bounded residual")
    guided = torch.zeros(candidates.shape[:2], dtype=torch.bool, device=candidates.device)
    count = int((candidates.shape[1] - reserved) * fraction)
    if count:
        candidates[:, -count:] += residual[:, None]
        guided[:, -count:] = True
    return guided


def influence_summary(plans):
    iterations = [item for plan in plans for item in plan.get("residual_checks", [])]
    return {
        "iterations": len(iterations),
        "new_guided_samples": sum(item["new_guided_samples"] for item in iterations),
        "guided_feasible_samples": sum(item["guided_feasible"] for item in iterations),
        "guided_elite_slots": sum(item["guided_elites"] for item in iterations),
        "fitted_elite_slots": sum(item["fitted_elites"] for item in iterations),
        "iterations_with_guided_elite": sum(item["guided_elites"] > 0 for item in iterations),
        "best_sample_guided_iterations": sum(item["best_sample_guided"] for item in iterations),
        "residual_l2_sum": sum(item["residual_l2"] for item in iterations),
        "final_best_from_guided_sample": sum(
            plan.get("residual_best_sample_selected", False) for plan in plans
        ),
    }
