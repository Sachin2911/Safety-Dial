"""Finite, source-disjoint E1 candidate schedules and durable partial bank provenance."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np

from helpers.branchBank import (ACTION_CLIP, BankWriter, Proposal, aimed_tape, exits_arena,
    perturbed_tapes, t_targets, two_phase_tape)
from helpers.runManifest import file_sha256

SAMPLING_PROTOCOL = {
    "version": "disjoint-multicandidate-v1",
    "pair_draws_per_source": 2,
    "depths": [0, 2, 4, 6],
    "maximum_candidates_per_source": 8,
    "maximum_accepted_roots_per_source": 1,
    "order": "round robin over fixed source order; initial depth rotates by source position",
    "selection": "first candidate passing unchanged construction filters; no evaluated-method outcome selection",
    "t0_range": [0, 20],
    "goal_offset_range": [25, 50],
    "temporal_distribution": "Two distinct (t0, goal_offset) pairs sampled uniformly without replacement from all valid pairs; differs from historical offset-first sampling",
    "short_episode_fallback": "If no 25-step offset fits, use only (t0=0, offset=episode_length-1); episodes shorter than 2 frames have no candidates",
    "tape_protocol": {"random_sigmas": [0.05, 0.1, 0.2], "maximum_trials_per_slot": 20,
        "ordinary_slots": "Exactly one arena-valid nominal plus remaining sigma slots allocated by cyclic round robin; an invalid nominal rejects the candidate",
        "action_dtype": "float32 before arena checks, execution, deduplication and storage",
        "stress_slots": "First floor(n/2) existing contact-target stress primitives, remainder existing toward-hazard primitives",
        "shortfall": "Discard construction candidate before querying any branch; stress failure preserves exact test roots and stops",
        "uniqueness": "No duplicate action tapes within a root",
        "outcomes": "All queried outcomes retained, including censored or unsafe branches"},
}


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w") as handle:
        json.dump(value, handle, separators=(",", ":"), allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)


def validate_source_roles(roles: dict) -> None:
    if set(roles) != {"dev", "test"} or set(roles["dev"]) != {"familiar"} or set(roles["test"]) != {"familiar", "heldout"}:
        raise ValueError("Source roles must reserve development familiar and both final-test families")
    seen = set()
    for families in roles.values():
        for episodes in families.values():
            if not episodes or any(type(e) is not int or e < 0 for e in episodes):
                raise ValueError("Source assignments require nonempty integer episode lists")
            if len(set(episodes)) != len(episodes) or seen.intersection(episodes):
                raise ValueError("An episode cannot occur twice or cross source roles/families")
            seen.update(episodes)


def candidate_schedule(h5_path: Path, episodes: list[int], *, seed: int) -> list[dict]:
    """Two distinct temporal draws crossed with all four balanced nominal depths.

    Short episodes may offer fewer temporal draws, never duplicated candidates. All
    state reads occur before any simulator/model use and are frozen into the plan.
    """
    import hdf5plugin  # noqa: F401

    if len(set(episodes)) != len(episodes):
        raise ValueError("Candidate sources must be unique")
    source_order = np.random.default_rng(seed).permutation(episodes).tolist()
    temporal = {}
    with h5py.File(h5_path, "r") as handle:
        lengths, offsets = handle["ep_len"][:], handle["ep_offset"][:]
        for position, episode in enumerate(source_order):
            length = int(lengths[episode])
            if length < 2:
                temporal[episode] = []
                continue
            choices = [(t0, goff) for t0 in range(21) for goff in range(25, 51) if t0 + goff < length]
            if not choices:
                choices = [(0, length - 1)]
            rng = np.random.default_rng(np.random.SeedSequence([seed, episode]))
            selected = rng.choice(len(choices), size=min(2, len(choices)), replace=False)
            temporal[episode] = []
            for draw, index in enumerate(selected):
                t0, goff = choices[int(index)]
                # Contiguous read preserves the established expert-pair interface.
                states = handle["state"][int(offsets[episode]) + t0:int(offsets[episode]) + t0 + goff + 1]
                start, goal = np.asarray(states[0], float).copy(), np.asarray(states[-1], float).copy()
                start[5:], goal[5:] = 0, 0
                temporal[episode].append({"episode": episode, "t0": t0, "goal_offset": goff,
                    "start": start.tolist(), "goal": goal.tolist(), "pair_draw": draw, "source_position": position})
    candidates = []
    for draw in range(2):
        for rotation in range(4):
            for position, episode in enumerate(source_order):
                if draw >= len(temporal[episode]):
                    continue
                candidates.append({**temporal[episode][draw], "k": SAMPLING_PROTOCOL["depths"][(position + rotation) % 4],
                    "candidate_round": 4 * draw + rotation, "candidate_index": len(candidates)})
    return candidates


def validate_candidate_schedule(candidates: list[dict], episodes: list[int]) -> None:
    allowed, seen = set(episodes), set()
    by_episode = {}
    for index, candidate in enumerate(candidates):
        episode, draw, depth = candidate["episode"], candidate["pair_draw"], candidate["k"]
        key = (episode, candidate["t0"], candidate["goal_offset"], depth)
        if (episode not in allowed or candidate["candidate_index"] != index or draw not in (0, 1)
                or depth not in SAMPLING_PROTOCOL["depths"] or key in seen):
            raise ValueError("Candidate schedule violates its frozen source/depth/uniqueness contract")
        expected = SAMPLING_PROTOCOL["depths"][(candidate["source_position"] + candidate["candidate_round"] % 4) % 4]
        if depth != expected or candidate["candidate_round"] // 4 != draw:
            raise ValueError("Candidate depth schedule is not the declared balanced rotation")
        seen.add(key)
        by_episode[episode] = by_episode.get(episode, 0) + 1
        if by_episode[episode] > SAMPLING_PROTOCOL["maximum_candidates_per_source"]:
            raise ValueError("Candidate schedule exceeds the predeclared per-source cap")
        for field in ("start", "goal"):
            value = np.asarray(candidate[field], float)
            if value.shape != (7,) or not np.isfinite(value).all() or value[5:].any():
                raise ValueError("Malformed frozen expert state")


def verify_plan_sources(plan: dict, plan_path: Path, repo_root: Path, *, verify_inputs=True) -> None:
    """Fail closed if local source snapshot or current generator changed between phases."""
    if plan.get("protocol_version") != SAMPLING_PROTOCOL["version"] or plan.get("sampling_protocol") != SAMPLING_PROTOCOL:
        raise ValueError("Unsupported or changed sampling protocol")
    validate_source_roles(plan["source_roles"])
    for role, families in plan["source_roles"].items():
        for family, episodes in families.items():
            validate_candidate_schedule(plan["candidate_schedule"][role][family], episodes)
    for relative, digest in plan["source_snapshot_sha256"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Source snapshot path must be repository-relative")
        if file_sha256(repo_root / path) != digest or file_sha256(plan_path.parent / "source_snapshot" / path) != digest:
            raise ValueError(f"Frozen generator identity changed: {relative}")
    for path, digest in (plan["input_sha256"].items() if verify_inputs else ()):
        if file_sha256(Path(path)) != digest:
            raise ValueError(f"Frozen input changed: {path}")


class ProgressBankWriter(BankWriter):
    """An interrupted bank keeps a readable, explicitly incomplete roots/ledger snapshot."""

    def __init__(self, bank_dir: Path, *, bank_name: str, plan_path: Path, **kwargs):
        super().__init__(bank_dir, **kwargs)
        self.bank_name = bank_name
        self.plan_hash = file_sha256(plan_path)
        # The plan is experiment data. Code snapshots stay outside uploaded bank dirs.
        (self.dir / "sampling-plan.json").write_bytes(plan_path.read_bytes())
        self.progress = {"status": "running", "bank": bank_name, "sampling_plan_sha256": self.plan_hash,
            "attempts_completed": 0, "rejections": {}, "accepted_by_family": {}, "skipped_accepted_sources": 0}
        self.layouts = []
        self.journal = (self.dir / "attempts.jsonl").open("x")
        self.started = time.time()

    def event(self, record: dict):
        self.journal.write(json.dumps({"unix": time.time(), **record}, separators=(",", ":"), allow_nan=False) + "\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())

    def checkpoint(self, ledger, *, status="running", manifest=None):
        self.h5.flush()
        self.progress.update({"status": status, "ledger": ledger.to_dict(), "roots_stored": len(self.roots),
            "n_branches": int(self.h5["tape"].shape[0]), "elapsed_seconds": time.time() - self.started})
        atomic_json(self.dir / "roots.json", {"status": status, "roots": self.roots, "ledger": ledger.to_dict(),
            "manifest": manifest or {"status": status, "sampling_plan_sha256": self.plan_hash},
            "n_branches": self.progress["n_branches"], "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        atomic_json(self.dir / "layouts.json", {"meta": {"bank": self.bank_name, "status": status,
            "sampling_plan_sha256": self.plan_hash}, "layouts": [layout.to_dict() for layout in self.layouts]})
        self.progress["snapshot_sha256"] = {name: file_sha256(self.dir / name) for name in ("roots.json", "layouts.json")}
        atomic_json(self.dir / "progress.json", self.progress)

    def finish(self, ledger, manifest=None):
        self.checkpoint(ledger, status="complete", manifest=manifest)
        self.journal.close()
        self.h5.close()

    def fail(self, ledger, exc):
        self.progress["error"] = {"type": type(exc).__name__, "message": str(exc)}
        self.event({"event": "failed", "error": self.progress["error"], "ledger": ledger.to_dict()})
        self.checkpoint(ledger, status="incomplete")
        self.journal.close()
        self.h5.close()


def exact_proposals(rng, root, ctx, count: int, *, stress=False, hazard_centre=None):
    """Fill finite unique slots before any simulator query, with the existing primitives."""
    if count <= 0 or (stress and hazard_centre is None):
        raise ValueError("Proposal count and stress hazard must be specified")
    nominal = np.asarray(root.meta["nominal_plan"], float)
    state = np.asarray(ctx.state, float)
    sigmas = SAMPLING_PROTOCOL["tape_protocol"]["random_sigmas"]
    slots = (["stress"] * (count // 2) + ["toward_hazard"] * (count - count // 2) if stress
             else ["nominal"] + ["random"] * (count - 1))
    out, hashes, trials = [], set(), []
    for slot, kind in enumerate(slots):
        limit = 1 if kind == "nominal" else SAMPLING_PROTOCOL["tape_protocol"]["maximum_trials_per_slot"]
        for trial in range(limit):
            actual_kind = kind
            if kind == "nominal":
                tape, params = nominal.copy(), {"sigma": 0.0}
            elif kind == "random":
                actual_kind = "random"
                sigma = sigmas[(max(slot, 1) - 1) % len(sigmas)]
                tape, params = perturbed_tapes(rng, nominal, 1, sigma)[0], {"sigma": sigma}
            elif kind == "stress":
                targets = t_targets(state)
                name = list(targets)[int(rng.integers(len(targets)))]
                speed = float(rng.choice((0.1, 0.2, 0.3)))
                tape = aimed_tape(state[:2], targets[name] + rng.normal(0.0, 8.0, 2), speed,
                    overshoot=float(rng.uniform(0.3, 1.5)))
                tape = np.clip(tape + rng.normal(0.0, 0.03, tape.shape), -ACTION_CLIP, ACTION_CLIP)
                params = {"target": name, "speed": speed}
            else:
                centre = state[2:4] + np.array([0.0, 45.0]) @ np.array([[np.cos(state[4]), np.sin(state[4])], [-np.sin(state[4]), np.cos(state[4])]])
                direction = np.asarray(hazard_centre, float) - centre
                direction /= np.linalg.norm(direction) + 1e-9
                side, lateral = float(rng.uniform(70.0, 110.0)), float(rng.normal(0.0, 25.0))
                via = centre - direction * side + np.array([-direction[1], direction[0]]) * lateral
                speed = float(rng.choice((0.1, 0.2, 0.3)))
                tape = two_phase_tape(state[:2], via, centre + direction * 30.0, speed, overshoot=float(rng.uniform(0.5, 2.0)))
                tape = np.clip(tape + rng.normal(0.0, 0.02, tape.shape), -ACTION_CLIP, ACTION_CLIP)
                params = {"speed": speed, "side": side}
            tape = np.ascontiguousarray(tape, dtype=np.float32)
            digest = hashlib.sha256(tape.tobytes()).hexdigest()
            outcome = "arena_rejected" if exits_arena(tape, state[:2]) else ("duplicate_rejected" if digest in hashes else "accepted")
            trials.append({"slot": slot, "trial": trial, "kind": actual_kind, "params": params,
                "tape_sha256": digest, "outcome": outcome})
            if outcome == "accepted":
                out.append(Proposal(tape, actual_kind, params))
                hashes.add(digest)
                break
        else:
            return out, {"complete": False, "requested": count, "filled": len(out), "trials": trials}
    return out, {"complete": True, "requested": count, "filled": len(out), "trials": trials}
