"""Validation and simulator accounting for prospective Push-T acquisition.

These helpers require no model or GPU. Historical results remain valid records of the
runs that produced them; they are never silently promoted to protocol-compliant evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from helpers.pushtReplay import StepLedger

COLLECTION_VERSION = "charged-context-cache-v2"


class MeteredEnv:
    """Count every attempted environment transition, including a failing call.

    Reset's internal physics substep is not an environment transition. Prefix replay,
    discarded roots and all branch executions pass through this wrapper's step method.
    """

    def __init__(self, env, ledger: StepLedger, category: str):
        self.env, self.ledger, self.category = env, ledger, category

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        self.ledger.add(self.category, 1)
        return self.env.step(action)


def copy_ledger(source: dict, target: StepLedger, prefix: str = "") -> None:
    for category, steps in source["steps"].items():
        target.add(prefix + category, steps, source.get("branches", {}).get(category, 0))
    for category, branches in source.get("branches", {}).items():
        if category not in source["steps"]:
            target.add(prefix + category, 0, branches)


def enforce_gate(e2: dict, *, diagnostic: bool, arms: list[str], seeds: list[int]) -> dict:
    """A diagnostic is an explicitly limited optimisation check, never an E3 claim."""
    passed = e2.get("gate", {}).get("passes") is True
    if len(set(arms)) != len(arms) or not arms or set(arms) - {"random", "boundary", "learned", "oracle"}:
        raise ValueError("Specify unique supported acquisition arms")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Specify unique acquisition seeds")
    if "learned" in arms and not passed:
        raise ValueError("Learned acquisition requires a recorded passing E2 gate")
    if not passed and not diagnostic:
        raise ValueError("E2 gate did not pass. Only an explicit --diagnostic bounded check is permitted")
    if diagnostic and (len(seeds) != 1 or set(arms) - {"random", "boundary"}):
        raise ValueError("A bounded diagnostic uses one acquisition seed and random/boundary arms only")
    return {"e2_passed": passed, "diagnostic": diagnostic,
            "interpretation": "development_only" if diagnostic else "prospective_acquisition"}


def require_new_paths(paths) -> None:
    for path in map(Path, paths):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite an existing run or bank: {path}")


def validate_acquisition_cache(path: Path, *, n_roots: int, seed: int) -> dict:
    """Reject partial, stale, wrong-size or unaccounted caches before their use."""
    required = ["roots.json", "layouts.json", "branches.h5", "contexts.npz"]
    if any(not (path / name).is_file() for name in required):
        raise ValueError(f"Incomplete acquisition root cache: {path}")
    blob = json.loads((path / "roots.json").read_text())
    expected = {"bank": "acq_roots", "seed": seed, "n_roots": n_roots,
                "collection_version": COLLECTION_VERSION}
    if any(blob.get("manifest", {}).get(k) != v for k, v in expected.items()):
        raise ValueError("Acquisition cache manifest does not match the requested collection")
    roots = blob.get("roots", [])
    if len(roots) != n_roots or len({r["root_id"] for r in roots}) != n_roots:
        raise ValueError("Acquisition cache has missing or duplicate roots")
    episodes = [r.get("meta", {}).get("episode") for r in roots]
    if None in episodes or len(set(episodes)) != n_roots:
        raise ValueError("Acquisition roots need unique source-trajectory identities")
    for r in roots:
        prefix = np.asarray(r["prefix"])
        if prefix.ndim != 2 or prefix.shape[1] != 2 or len(prefix) < 10 or len(prefix) % 5:
            raise ValueError("Invalid acquisition prefix")
        if not np.isfinite(prefix).all():
            raise ValueError("Non-finite acquisition prefix")
    ledger = blob.get("ledger", {})
    steps = ledger.get("steps", {})
    if (not steps or any(not isinstance(v, int) or v < 0 for v in steps.values())
            or ledger.get("total_steps") != sum(steps.values())
            or sum(steps.values()) < sum(len(r["prefix"]) for r in roots)):
        raise ValueError("Acquisition cache lacks valid charged simulator costs")
    layouts = json.loads((path / "layouts.json").read_text())["layouts"]
    if (len(layouts) != n_roots
            or {x["root_id"] for x in layouts} != {r["root_id"] for r in roots}
            or any(x["family"] != "familiar" for x in layouts)):
        raise ValueError("Acquisition layouts must cover every root exactly once")
    with h5py.File(path / "branches.h5", "r") as f:
        required_datasets = {"tape", "states", "block_vel", "block_ang_vel", "n_contacts", "root_index", "kind", "params"}
        if not required_datasets.issubset(f) or any(len(f[k]) != 0 for k in f):
            raise ValueError("An acquisition root cache must contain no prequeried branches")
    with np.load(path / "contexts.npz", allow_pickle=False) as z:
        if (z["frames"].shape != (n_roots, 3, 224, 224, 3)
                or z["history_actions"].shape != (n_roots, 2, 5, 2)
                or list(z["root_ids"]) != [r["root_id"] for r in roots]):
            raise ValueError("Cached root observations do not match the root bank")
        if not np.isfinite(z["history_actions"]).all():
            raise ValueError("Invalid cached history actions")
        for i, r in enumerate(roots):
            if not np.array_equal(z["history_actions"][i], np.asarray(r["prefix"])[-10:].reshape(2, 5, 2)):
                raise ValueError("Cached history actions differ from the recorded prefix")
    return blob
