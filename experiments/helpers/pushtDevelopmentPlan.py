"""Bind development feasibility to a frozen, disjoint future bank sampling plan.

Validation reads only the completed development bank and the precommitted plan. Final
test/stress banks need not exist and are never opened by this helper.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from helpers.pushtBankSampling import SAMPLING_PROTOCOL
from helpers.pushtSourceFamilies import SOURCE_FAMILY_PROTOCOL, geometric_source_family
from helpers.runManifest import file_sha256

PLAN_PROTOCOL = SAMPLING_PROTOCOL["version"]
REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_SOURCES = {
    "experiments/scripts/pusht_e1_banks.py",
    *{f"experiments/helpers/{name}.py" for name in
      ("branchBank", "pushtLayouts", "pushtGeometry", "pushtSourceFamilies")},
}
ROLE_FAMILIES = {"dev": {"familiar"}, "test": {"familiar", "heldout"}}


def source_roles_sha256(roles: dict) -> str:
    return hashlib.sha256(json.dumps(roles, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_roles(roles: dict) -> None:
    if set(roles) != set(ROLE_FAMILIES):
        raise ValueError("Frozen plan must reserve development and final-test source roles")
    assigned = set()
    for role, families in ROLE_FAMILIES.items():
        if set(roles[role]) != families:
            raise ValueError("Frozen plan source families differ from the declared transfer grid")
        for episodes in roles[role].values():
            if (not isinstance(episodes, list) or not episodes
                    or any(type(ep) is not int or ep < 0 for ep in episodes)
                    or len(episodes) != len(set(episodes))):
                raise ValueError("Frozen source roles require nonempty unique episode IDs")
            if assigned.intersection(episodes):
                raise ValueError("Development/final source-role reservations overlap")
            assigned.update(episodes)


def _validate_candidates(plan: dict) -> None:
    schedules = plan.get("candidate_schedule", {})
    if set(schedules) != set(ROLE_FAMILIES):
        raise ValueError("Frozen plan must include the complete future candidate schedule")
    for role, families in ROLE_FAMILIES.items():
        if set(schedules[role]) != families:
            raise ValueError("Frozen candidate schedule omits a source family")
        for family in families:
            episodes = plan["source_roles"][role][family]
            counts = dict.fromkeys(episodes, 0)
            seen = set()
            for candidate in schedules[role][family]:
                ep = candidate.get("episode")
                if ep not in counts:
                    raise ValueError("Candidate source is outside its frozen role")
                if (type(candidate.get("t0")) is not int or candidate["t0"] < 0
                        or type(candidate.get("goal_offset")) is not int
                        or candidate["goal_offset"] < 1 or candidate.get("k") not in (0, 2, 4, 6)
                        or type(candidate.get("root_seed")) is not int
                        or not isinstance(candidate.get("root_id"), str)):
                    raise ValueError("Candidate start/goal/depth is outside the declared protocol")
                key = (ep, candidate["t0"], candidate["goal_offset"], candidate["k"])
                if key in seen:
                    raise ValueError("Frozen candidate schedule contains duplicate attempts")
                seen.add(key)
                counts[ep] += 1
                for name in ("start", "goal"):
                    state = np.asarray(candidate.get(name), dtype=float)
                    if state.shape != (7,) or not np.isfinite(state).all():
                        raise ValueError("Frozen candidate must contain finite full start and goal states")
            if any(n < 1 or n > 8 for n in counts.values()):
                raise ValueError("Each reserved source requires a bounded schedule of 1 to 8 candidates")


def validate_development_plan(dev_bank: Path, plan_path: Path) -> dict:
    """Check dev completion, exact planned roots, reserved roles and frozen source bytes."""
    dev_bank, plan_path = Path(dev_bank).resolve(), Path(plan_path).resolve()
    if plan_path.parent != dev_bank.parent:
        raise ValueError("Sampling plan must reside beside its development bank")
    plan = json.loads(plan_path.read_text())
    digest = file_sha256(plan_path)
    if plan.get("protocol_version") != PLAN_PROTOCOL or plan.get("sampling_protocol") != SAMPLING_PROTOCOL:
        raise ValueError("Unsupported frozen sampling plan protocol")
    if plan.get("source_family_protocol") != SOURCE_FAMILY_PROTOCOL:
        raise ValueError("Frozen plan geometric family definition changed")
    roles = plan.get("source_roles", {})
    _validate_roles(roles)
    _validate_candidates(plan)
    hashes = plan.get("source_snapshot_sha256", {})
    if not REQUIRED_SOURCES <= hashes.keys():
        raise ValueError("Frozen sampling plan lacks required generator source snapshots")
    for relative, expected in hashes.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Frozen source snapshot paths must be repository-relative")
        saved, current = plan_path.parent / "source_snapshot" / path, REPO_ROOT / path
        if (not saved.is_file() or not current.is_file()
                or file_sha256(saved) != expected or file_sha256(current) != expected):
            raise ValueError(f"Frozen generator source snapshot changed: {relative}")
    blob = json.loads((dev_bank / "roots.json").read_text())
    manifest = json.loads((dev_bank / "manifest.json").read_text())
    if blob.get("status") != "complete" or blob.get("manifest") != manifest:
        raise ValueError("Development bank must be complete with matching persisted manifests")
    data = manifest.get("data", {})
    if (data.get("source_roles") != roles or data.get("sampling_plan_sha256") != digest
            or data.get("sampling_protocol") != plan["sampling_protocol"]
            or data.get("source_family_protocol") != SOURCE_FAMILY_PROTOCOL):
        raise ValueError("Development bank does not identify this exact frozen future plan")
    copied_plan = dev_bank / "sampling-plan.json"
    if not copied_plan.is_file() or file_sha256(copied_plan) != digest:
        raise ValueError("Development bank copy differs from the exact frozen sampling plan")
    roots = blob.get("roots", [])
    settings = plan.get("settings", {})
    if (any(type(settings.get(k)) is not int or settings[k] < 1
            for k in ("dev_roots", "test_roots", "dev_tapes", "test_tapes"))
            or len(roots) != settings["dev_roots"]):
        raise ValueError("Completed development bank does not meet the frozen root counts")
    source_ids = [r["meta"]["episode"] for r in roots]
    root_ids = [r["root_id"] for r in roots]
    if (not roots or len(source_ids) != len(set(source_ids))
            or len(root_ids) != len(set(root_ids))
            or not set(source_ids) <= set(roles["dev"]["familiar"])):
        raise ValueError("Development bank requires unique roots from its reserved sources")
    candidates = {(c["episode"], c["t0"], c["goal_offset"], c["k"]): c
                  for c in plan["candidate_schedule"]["dev"]["familiar"]}
    for root in roots:
        meta = root["meta"]
        key = tuple(meta.get(k) for k in ("episode", "t0", "goal_offset", "k"))
        candidate = candidates.get(key)
        if (candidate is None or root["start_state"] != candidate["start"]
                or root["goal_state"] != candidate["goal"]
                or root["seed"] != candidate["root_seed"] or root["root_id"] != candidate["root_id"]):
            raise ValueError("Development root was not in the frozen candidate schedule")
        if (meta.get("source_family") != "familiar"
                or meta.get("source_family_protocol") != SOURCE_FAMILY_PROTOCOL["version"]
                or geometric_source_family(meta["state_at_root"], root["goal_state"]) != "familiar"):
            raise ValueError("Development root violates the frozen geometric family")
    layouts = json.loads((dev_bank / "layouts.json").read_text())["layouts"]
    if (len(layouts) != len(roots) or {lay["root_id"] for lay in layouts} != set(root_ids)
            or any(lay["family"] != "familiar" for lay in layouts)):
        raise ValueError("Development layouts must cover exactly the familiar development roots")
    with h5py.File(dev_bank / "branches.h5", "r") as h5:
        indices = np.asarray(h5["root_index"])
        count = len(h5["tape"])
        if (count != blob.get("n_branches") or len(indices) != count
                or set(map(int, indices)) != set(range(len(roots)))
                or count != settings["dev_roots"] * settings["dev_tapes"]
                or not np.all(np.bincount(indices) == settings["dev_tapes"])):
            raise ValueError("Completed development branches do not cover the persisted roots")
    if file_sha256(plan_path) != digest:
        raise ValueError("Frozen sampling plan changed during validation")
    return {"path": str(plan_path), "sha256": digest, "source_roles_sha256": source_roles_sha256(roles),
            "protocol_version": PLAN_PROTOCOL, "source_snapshot_sha256": hashes}
