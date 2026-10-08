"""Pure, pre-query stress recovery on fixed previously collected Push-T roots.

Only the finite rejection-sampling allowance changes. Action distributions, draw
order, float32 conversion, arena bounds, uniqueness and the eight/eight slots are
identical to the frozen E1 protocol. No environment or model is constructed here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from numbers import Integral

import numpy as np

from helpers.branchBank import ACTION_CLIP, Proposal, aimed_tape, exits_arena, t_targets, two_phase_tape

STRESS_RECOVERY_PROTOCOL = {
    "version": "fixed-test-roots-stress-recovery-v1",
    "seed": 20280927,
    "maximum_trials_per_slot": 4096,
    "tapes_per_root": 16,
    "contact_target_slots": 8,
    "toward_hazard_slots": 8,
    "action_dtype": "float32 before arena checks, deduplication, storage and execution",
    "proposal_distribution": "Unchanged frozen E1 stress and toward-hazard primitives and RNG draw order",
    "selection": "First arena-valid unique tape per slot; no branch outcomes or model scores",
    "root_order": "Saved completed test bank order, with exact saved initial branch states",
    "preflight": "Freeze every root's complete proposal pool before any new simulator query",
    "shortfall": "Raise before execution; retain audit and never substitute a root or duplicate a tape",
}
POOL_DIGEST_SCHEMA = "Repeated UTF-8 root_id, NUL, kind, NUL, C-order float32 tape bytes in saved root order and slot order"


class StressProposalShortfall(RuntimeError):
    """A bounded proposal search failed; no partial pool is executable."""

    def __init__(self, audit, root_id=None):
        self.audit = audit
        self.root_id = root_id
        name = f" for root {root_id}" if root_id is not None else ""
        super().__init__(f"Stress proposal shortfall{name}: {audit['filled']}/{audit['requested']} slots")


def _finite_vector(value, size, name):
    result = np.ascontiguousarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite ({size},) vector")
    return result


def _draw_tape(rng, state, hazard_centre, kind):
    # Preserve every primitive and random draw from pushtBankSampling.exact_proposals.
    if kind == "stress":
        targets = t_targets(state)
        name = list(targets)[int(rng.integers(len(targets)))]
        speed = float(rng.choice((0.1, 0.2, 0.3)))
        tape = aimed_tape(state[:2], targets[name] + rng.normal(0.0, 8.0, 2), speed,
            overshoot=float(rng.uniform(0.3, 1.5)))
        tape = np.clip(tape + rng.normal(0.0, 0.03, tape.shape), -ACTION_CLIP, ACTION_CLIP)
        params = {"target": name, "speed": speed}
    else:
        centre = state[2:4] + np.array([0.0, 45.0]) @ np.array([
            [np.cos(state[4]), np.sin(state[4])], [-np.sin(state[4]), np.cos(state[4])]])
        direction = hazard_centre - centre
        direction /= np.linalg.norm(direction) + 1e-9
        side, lateral = float(rng.uniform(70.0, 110.0)), float(rng.normal(0.0, 25.0))
        via = centre - direction * side + np.array([-direction[1], direction[0]]) * lateral
        speed = float(rng.choice((0.1, 0.2, 0.3)))
        tape = two_phase_tape(state[:2], via, centre + direction * 30.0, speed,
            overshoot=float(rng.uniform(0.5, 2.0)))
        tape = np.clip(tape + rng.normal(0.0, 0.02, tape.shape), -ACTION_CLIP, ACTION_CLIP)
        params = {"speed": speed, "side": side}
    return tape, params


def exact_stress_proposals(rng, state, hazard_centre, *, max_trials=4096):
    """Return all 16 guarded unique proposals and a complete per-trial audit.

    ``max_trials=20`` exists for pure regression against the failed protocol.
    Production all-root preflight always uses the fixed recovery allowance.
    A shortfall raises with ``exception.audit`` and returns no partial proposals.
    """
    if isinstance(max_trials, bool) or not isinstance(max_trials, Integral) or max_trials <= 0:
        raise ValueError("max_trials must be a positive integer")
    state = _finite_vector(state, 7, "state")
    hazard_centre = _finite_vector(hazard_centre, 2, "hazard_centre")
    proposals, hashes, trials = [], set(), []
    for slot, kind in enumerate(["stress"] * 8 + ["toward_hazard"] * 8):
        for trial in range(max_trials):
            tape, params = _draw_tape(rng, state, hazard_centre, kind)
            tape = np.ascontiguousarray(tape, dtype=np.float32)
            if tape.shape != (5, 5, 2) or not np.isfinite(tape).all():
                raise ValueError("Stress primitive returned an invalid action tape")
            digest = hashlib.sha256(tape.tobytes()).hexdigest()
            outcome = "arena_rejected" if exits_arena(tape, state[:2]) else (
                "duplicate_rejected" if digest in hashes else "accepted")
            trials.append({"slot": slot, "trial": trial, "kind": kind, "params": params,
                "tape_sha256": digest, "outcome": outcome})
            if outcome == "accepted":
                proposals.append(Proposal(tape, kind, params))
                hashes.add(digest)
                break
        else:
            raise StressProposalShortfall({"complete": False, "requested": 16,
                "filled": len(proposals), "trials": trials})
    return proposals, {"complete": True, "requested": 16, "filled": len(proposals), "trials": trials}


def preflight_stress_roots(roots, states, hazard_centres, *, expected_roots=128):
    """Prepare every fixed root in memory, returning (proposal_pools, JSON plan).

    ``roots`` accepts saved root dictionaries or Root objects. The caller must pin
    the original root/layout artifacts and supply first-branch initial states.
    This helper performs no I/O. The runner writes and verifies the complete plan
    and tape archive before creating an environment or querying any branch.
    """
    roots = list(roots)
    if (isinstance(expected_roots, bool) or not isinstance(expected_roots, Integral)
            or expected_roots <= 0 or len(roots) != expected_roots):
        raise ValueError("Root count differs from the fixed preflight requirement")
    states = np.ascontiguousarray(states, dtype=np.float64)
    hazard_centres = np.ascontiguousarray(hazard_centres, dtype=np.float64)
    if states.shape != (expected_roots, 7) or not np.isfinite(states).all():
        raise ValueError("Initial states must have finite shape (root_count, 7)")
    if hazard_centres.shape != (expected_roots, 2) or not np.isfinite(hazard_centres).all():
        raise ValueError("Hazard centres must have finite shape (root_count, 2)")
    root_ids = [r.get("root_id") if isinstance(r, dict) else r.root_id for r in roots]
    if any(not isinstance(r, str) or not r or "\0" in r for r in root_ids) or len(set(root_ids)) != len(root_ids):
        raise ValueError("Root identities must be nonempty and unique")
    rng = np.random.default_rng(STRESS_RECOVERY_PROTOCOL["seed"])
    pools, rows, digest = [], [], hashlib.sha256()
    total_trials = maximum_slot_trials = 0
    for index, (root_id, state, centre) in enumerate(zip(root_ids, states, hazard_centres)):
        try:
            proposals, audit = exact_stress_proposals(rng, state, centre,
                max_trials=STRESS_RECOVERY_PROTOCOL["maximum_trials_per_slot"])
        except StressProposalShortfall as exc:
            raise StressProposalShortfall(exc.audit, root_id=root_id) from exc
        hashes = []
        for proposal in proposals:
            raw = proposal.tape.tobytes()
            hashes.append(hashlib.sha256(raw).hexdigest())
            digest.update(root_id.encode("utf-8") + b"\0" + proposal.kind.encode("utf-8") + b"\0" + raw)
        slot_counts = np.bincount([trial["slot"] for trial in audit["trials"]], minlength=16)
        maximum_slot_trials = max(maximum_slot_trials, int(slot_counts.max()))
        total_trials += len(audit["trials"])
        rows.append({"root_index": index, "root_id": root_id, "state": state.tolist(),
            "state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
            "hazard_centre": centre.tolist(),
            "hazard_centre_sha256": hashlib.sha256(centre.tobytes()).hexdigest(),
            "accepted_tape_sha256": hashes, "audit": audit})
        pools.append(proposals)
    plan = {"schema_version": 1, "protocol": deepcopy(STRESS_RECOVERY_PROTOCOL),
        "complete": True, "root_count": len(roots), "branch_count": 16 * len(roots),
        "roots": rows, "pool_sha256": digest.hexdigest(), "pool_digest_schema": POOL_DIGEST_SCHEMA,
        "total_trials": total_trials, "maximum_slot_trials": maximum_slot_trials}
    return pools, plan
