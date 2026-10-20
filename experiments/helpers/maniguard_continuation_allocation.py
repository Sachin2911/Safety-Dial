"""Prioritize the action a filter selects, with no physical dispatch or fitness.

This is a separate prospective interface. The original cohort and the R12
controller remain unchanged. All decoded prefixes share one factual observation.
"""

import numpy as np

from helpers.maniguard_model_control import (
    FilterDecision,
    filter_prefixes,
    finite,
    predicted_progress,
)


def _prefixes(value, horizon, name):
    value = np.asarray(value, dtype=float)
    if (value.ndim != 3 or value.shape[1:] != (horizon, 8)
            or not np.isfinite(value).all()):
        raise ValueError("Invalid " + name)
    return value


def _unique_pool(proposals, alternatives):
    """Exact decoded-prefix reuse is valid only at this common factual anchor."""
    pool, indexes, seen = [], [], {}
    for prefix in np.concatenate([proposals, alternatives]):
        prefix = prefix.copy()
        prefix[prefix == 0] = 0.  # Give +0 and -0 the same cache identity.
        key = prefix.tobytes()
        if key not in seen:
            seen[key] = len(pool)
            pool.append(prefix)
        indexes.append(seen[key])
    return np.asarray(pool), indexes[:len(proposals)], indexes[len(proposals):]


class FilteredCandidateAllocator:
    """Screen each proposal through the same declared alternative set.

    Prefix priorities describe the selected filtered action, not the raw proposal
    or a full filtered-policy episode. Identical prefixes are predicted once;
    they do not establish that two complete controllers are equivalent. A failed
    qualification skips prediction. Model-free fallback allocates identifiers
    only and never overrides a filter's abstention or authorizes execution.
    """

    def __init__(self, model, gate, qualification, penalty):
        if not penalty.frozen:
            raise ValueError("A common frozen ROSARL estimate is required")
        self.model, self.gate = model, dict(gate)
        self.qualification, self.penalty = dict(qualification), penalty

    def screen(self, current, previous, history_valid, proposed_prefixes, alternative_prefixes,
               candidate_ids, slots, ledger, request_id, *, seed=6102001):
        ids = list(candidate_ids)
        if (not ids or any(not isinstance(x, str) or not x for x in ids)
                or len(set(ids)) != len(ids) or type(slots) is not int or not 1 <= slots <= len(ids)
                or type(history_valid) is not bool or type(seed) is not int or seed < 0):
            raise ValueError("Invalid candidate identities, slots, history or seed")
        current = finite(current, (44,), "current factual features")
        previous = finite(previous, (44,), "causal previous features")
        proposals = _prefixes(proposed_prefixes, self.model.horizon, "decoded proposals")
        alternatives = _prefixes(alternative_prefixes, self.model.horizon, "decoded alternatives")
        if len(proposals) != len(ids):
            raise ValueError("Candidate identities and proposals differ")
        penalty = self.penalty.penalty
        if not np.isfinite(penalty) or penalty > 0:
            raise ValueError("Expected a finite nonpositive penalty")
        passed = (self.qualification.get("prediction_gate_pass") is True
                  and self.model.calibration.get("ready") is True)
        admission = request_id + "/allocation"
        ledger.admit(admission, "candidate_screenings", len(ids))
        try:
            prediction, pool = None, None
            decisions, mapping = {}, {}
            raw_priorities, priorities, groups = {}, {}, {}
            if passed:
                pool, proposal_indexes, alternative_indexes = _unique_pool(proposals, alternatives)
                n = len(pool)
                prediction = self.model.predict(
                    np.repeat(current[None], n, axis=0), np.repeat(previous[None], n, axis=0),
                    np.full(n, history_valid), pool, ledger, request_id + "/prediction")
                upper = self.model.risk_upper_score(prediction, self.gate)
                utility = predicted_progress(prediction["stage_mean"]) + penalty * upper
                for identity, proposal_index in zip(ids, proposal_indexes, strict=True):
                    # Other proposals are not silently added as filter alternatives.
                    indexes = [proposal_index] + list(dict.fromkeys(
                        i for i in alternative_indexes if i != proposal_index))
                    local = {key: np.asarray(prediction[key])[indexes]
                             for key in ("stage_mean", "support_distance")}
                    decision = filter_prefixes(
                        pool[indexes], local, upper[indexes], gate_pass=True,
                        risk_limit=self.gate["risk_score_limit"],
                        support_radius=self.model.spec["support_radius"])
                    chosen = (None if decision.selected_index is None
                              else indexes[decision.selected_index])
                    decisions[identity] = decision
                    mapping[identity] = {"proposed_pool_index": proposal_index,
                                         "filtered_pool_index": chosen}
                    raw_priorities[identity] = float(utility[proposal_index])
                    priorities[identity] = None if chosen is None else float(utility[chosen])
                    if chosen is not None:
                        groups.setdefault(chosen, []).append(identity)
            else:
                for identity, proposal in zip(ids, proposals, strict=True):
                    decisions[identity] = FilterDecision(
                        "abstained", None, proposal.copy(), None, [], [])
                    mapping[identity] = {"proposed_pool_index": None, "filtered_pool_index": None}
                    raw_priorities[identity] = priorities[identity] = None
            qualified = [identity for identity in ids if priorities[identity] is not None]
            if passed and len(qualified) >= slots:
                selected = sorted(qualified, key=lambda x: (-priorities[x], x))[:slots]
                mode = "model_assisted_filtered_prefix_priority"
            else:
                selected = np.random.default_rng(seed).choice(ids, slots, replace=False).tolist()
                mode = "declared_model_free_allocation_fallback"
            result = {
                "selected_ids": selected, "mode": mode, "decisions": decisions,
                "mapping": mapping, "surrogate_priority": priorities,
                "raw_proposal_priority_diagnostic": raw_priorities,
                "prefix_equivalence_groups": list(groups.values()),
                "prediction": prediction,
                "unique_prediction_prefixes": 0 if pool is None else len(pool),
                "rosarl": self.penalty.snapshot(), "physical_fitness": None,
                "dispatch_authorized": False, "new_executed_controls": 0,
                "scope": "One-anchor allocation only; factual full episodes remain required.",
            }
        except Exception:
            ledger.finish(admission, "failed")
            raise
        ledger.finish(admission, "returned")
        return result
