"""Prospective factual progress objective, isolated from all frozen searches.

The caller supplies complete, validated physical outcomes. Ordered milestones are
training labels and do not become actor inputs. This module performs no evaluation
or candidate selection and has no simulator, model or policy-service dependency.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from helpers.maniguard_latent import outcome_score

MAX_ORDERED_STAGE = 4


@dataclass(frozen=True)
class FactualProgress:
    """Whole-episode labels plus the existing ordered-milestone stage.

    Any violation removes all secondary progress credit, including earlier safe
    progress. Physical validity and milestone attribution remain the caller's job.
    """

    requested_completion: bool
    violated: bool
    ordered_stage: int

    def __post_init__(self):
        outcome_score(self.requested_completion, self.violated)
        if type(self.ordered_stage) is not int or not 0 <= self.ordered_stage <= MAX_ORDERED_STAGE:
            raise ValueError("Require an integer ordered stage in [0,4]")

    @property
    def primary(self):
        return outcome_score(self.requested_completion, self.violated)

    @property
    def secondary(self):
        return 0 if self.violated else self.ordered_stage


@dataclass(frozen=True)
class PanelScore:
    """Exact integer score for one complete, fixed-size factual case panel."""

    primary: int
    secondary: int
    panel_size: int

    @property
    def rank_key(self):
        return self.primary, self.secondary

    @property
    def scale(self):
        return MAX_ORDERED_STAGE * self.panel_size + 1

    @property
    def scalar(self):
        return self.scale * self.primary + self.secondary


@dataclass(frozen=True)
class ProgressObjective:
    """Share the same supervision and scalar return across future optimizers.

    F = (4N+1) sum(U) + sum(S), with U the unchanged terminal utility and S the
    whole-episode nonviolating ordered stage. The bounded secondary sum is at most
    4N, so scalar ordering equals (sum(U), sum(S)) ordering on complete factual
    panels of N cases. This is not a safety guarantee or a lexicographic statement
    about unknown expected returns. Compare scores only under the same objective.

    A new protocol must freeze the case panel, objective and common information
    before comparative execution. This component does not amend an old ranking.
    """

    case_ids: tuple[str, ...]

    def __post_init__(self):
        if (
            not isinstance(self.case_ids, (tuple, list))
            or not self.case_ids
            or any(not isinstance(case, str) or not case for case in self.case_ids)
            or len(set(self.case_ids)) != len(self.case_ids)
        ):
            raise ValueError("Freeze a nonempty panel of distinct case identifiers")
        object.__setattr__(self, "case_ids", tuple(self.case_ids))

    @property
    def scale(self):
        return MAX_ORDERED_STAGE * len(self.case_ids) + 1

    def _complete(self, outcomes):
        if (
            not isinstance(outcomes, Mapping)
            or set(outcomes) != set(self.case_ids)
            or any(not isinstance(value, FactualProgress) for value in outcomes.values())
        ):
            raise ValueError("Require exactly one validated factual outcome per frozen case")

    def score(self, outcomes: Mapping[str, FactualProgress]):
        self._complete(outcomes)
        return PanelScore(
            primary=sum(outcomes[case].primary for case in self.case_ids),
            secondary=sum(outcomes[case].secondary for case in self.case_ids),
            panel_size=len(self.case_ids),
        )

    def episode_rewards(self, outcomes: Mapping[str, FactualProgress]):
        """Returns for ordinary learning; summing them gives the same panel scalar."""
        self._complete(outcomes)
        return {
            case: self.scale * outcomes[case].primary + outcomes[case].secondary
            for case in self.case_ids
        }

    def specification(self):
        return {
            "schema": "maniguard-factual-progress-objective-v1",
            "case_ids": list(self.case_ids),
            "panel_size": len(self.case_ids),
            "terminal_primary_scale": self.scale,
            "maximum_ordered_stage": MAX_ORDERED_STAGE,
            "definition": "F=(4N+1)*sum(U)+sum(S); S=0 for any whole-episode violation",
            "actor_information_added": False,
            "dispatch_authorized": False,
        }
