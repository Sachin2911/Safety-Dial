"""Development-only whole-T route witnesses for the frozen hazard generator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from helpers.pushtGeometry import Disc, T_LOCAL, clearance_trace, hazard_from_dict
from helpers.pushtLayouts import FAMILIES, MAX_MARGIN, ROUTE_CROSS_TOL, SLACK, cell_of
from helpers.pushtReplay import observation_domain_mask
from helpers.pushtRetention import block_coverage
from helpers.runManifest import file_sha256
from helpers.splitIntegrity import bank_identity

PROTOCOL = "development-feasible-route-v1"
MIN_WITNESSES = 3
MIN_GOAL_COVERAGE = .90


def generator_identity() -> dict:
    root = Path(__file__).resolve().parent
    return {name: file_sha256(root / name) for name in
            ("pushtLayouts.py", "pushtGeometry.py", "pushtSourceFamilies.py")}


def assess_witness(record: dict) -> dict:
    """Recompute route safety/progress; incomplete padded suffixes are never evidence."""
    reasons = []
    local = tuple(np.asarray(p, dtype=float) for p in record["local_polygons"])
    def canonical(polygons):
        return sorted(sorted(map(tuple, p)) for p in polygons)
    if canonical(local) != canonical(T_LOCAL):
        return {"passes": False, "reasons": ["Observed simulator whole-T geometry differs from frozen generator"]}
    states = np.asarray(record["states"], dtype=float)
    actions = np.asarray(record["actions"], dtype=float)
    nominal = np.asarray(record["nominal_route_poses"], dtype=float)
    root_pose = np.asarray(record["root_pose"], dtype=float)
    goal = np.asarray(record["root"]["goal_state"], dtype=float)[2:5]
    layout = record["layout"]
    hz = hazard_from_dict(layout["hazard"])
    if (states.ndim != 2 or states.shape[1] != 7 or len(states) < 2
            or actions.shape != (len(states) - 1, 2)
            or nominal.ndim != 2 or nominal.shape[1] != 3 or not len(nominal)
            or not np.isfinite(states).all() or not np.isfinite(actions).all()
            or not np.isfinite(nominal).all()):
        return {"passes": False, "reasons": ["Invalid or incomplete dense trajectory"]}
    if not np.allclose(states[0, 2:5], root_pose, atol=1e-6, rtol=0):
        reasons.append("Witness does not start at the declared nominal controller root")
    if record.get("role") != "development" or layout["family"] != "familiar":
        reasons.append("Generator witnesses must use development familiar layouts")
    if record.get("observed_steps") != len(actions) or not observation_domain_mask(states).all():
        reasons.append("All witness steps must be observed inside the image domain")
    sc = float(clearance_trace(root_pose[None], hz, local)[0])
    gc = float(clearance_trace(goal[None], hz, local)[0])
    nc = float(clearance_trace(nominal, hz, local).min())
    family = FAMILIES["familiar"]
    size = hz.r if isinstance(hz, Disc) else np.sqrt((hz.x1 - hz.x0) * (hz.y1 - hz.y0)) / 2
    if not family["size"][0] <= size <= family["size"][1]:
        reasons.append("Hazard size differs from the frozen development layout family")
    if cell_of(hz.centre) not in set(map(tuple, family["cells"])):
        reasons.append("Hazard centre is outside the frozen development layout family")
    if sc < MAX_MARGIN + SLACK or gc < MAX_MARGIN + SLACK or nc > ROUTE_CROSS_TOL:
        reasons.append("Layout does not satisfy the frozen nominal-route placement rule")
    for key, value in (("start_clearance", sc), ("goal_clearance", gc),
                       ("nominal_min_clearance", nc)):
        if not np.isclose(layout.get(key, np.nan), value, atol=1e-5, rtol=0):
            reasons.append(f"Frozen layout {key} differs from replay")
    minimum = float(clearance_trace(states[:, 2:5], hz, local).min())
    coverage = block_coverage(states[-1, 2:5], goal, local)
    initial = block_coverage(states[0, 2:5], goal, local)
    if minimum <= 0:
        reasons.append("Whole-T trajectory intersects the hazard, including exact contact")
    if initial >= MIN_GOAL_COVERAGE or coverage < MIN_GOAL_COVERAGE:
        reasons.append("Witness must reach at least 0.90 actual goal coverage from below threshold")
    replay = record.get("replay", {})
    if replay.get("repeats") != 2 or replay.get("bitwise_equal") is not True:
        reasons.append("Two exactly equal reset-and-prefix replays are required")
    return {"passes": not reasons, "reasons": reasons, "min_clearance": minimum,
            "final_coverage": coverage, "initial_coverage": initial,
            "start_clearance": sc, "goal_clearance": gc, "nominal_min_clearance": nc}


def feasibility_gate(records: list[dict], development_episodes: list[int]) -> dict:
    allowed = set(development_episodes)
    accepted, checks = [], []
    for record in records:
        check = assess_witness(record)
        episode = record["root"]["meta"]["episode"]
        if episode not in allowed:
            check["passes"] = False
            check["reasons"].append("Witness source is outside the declared development role")
        checks.append(check)
        if check["passes"]:
            accepted.append(episode)
    unique = set(accepted)
    return {"passes": len(unique) >= MIN_WITNESSES and all(c["passes"] for c in checks),
            "role": "development", "n_independent_witnesses": len(unique),
            "min_witnesses": MIN_WITNESSES, "min_goal_coverage": MIN_GOAL_COVERAGE,
            "checks": checks,
            "scope": "Development generator feasibility, not per-test-case solvability"}


def require_feasibility_report(path: Path, dev_bank: Path, *, sampling_plan: Path | None = None) -> dict:
    """Validate current bank/generator identities and recompute the reported witness gate."""
    path, dev_bank = Path(path), Path(dev_bank)
    report = json.loads(path.read_text())
    frozen_plan = report.get("frozen_sampling_plan")
    if sampling_plan is not None or frozen_plan is not None:
        from helpers.pushtDevelopmentPlan import validate_development_plan

        if not isinstance(frozen_plan, dict) or not frozen_plan.get("path"):
            raise ValueError("Development witness does not bind a frozen future sampling plan")
        actual = validate_development_plan(dev_bank, sampling_plan or Path(frozen_plan["path"]))
        if actual != frozen_plan:
            raise ValueError("Development witness identifies a different frozen future sampling plan")
    if report.get("protocol") != PROTOCOL or report.get("generator_identity") != generator_identity():
        raise ValueError("Feasible-route witness uses a different frozen generator")
    if report.get("development_bank_identity") != bank_identity(dev_bank):
        raise ValueError("Feasible-route witness does not identify this development bank")
    blob = json.loads((dev_bank / "roots.json").read_text())
    roots = {r["root_id"]: r for r in blob["roots"]}
    roles = blob["manifest"]["data"]["source_roles"]
    dev = {int(e) for eps in roles["dev"].values() for e in eps}
    other = {int(e) for role, families in roles.items() if role != "dev"
             for eps in families.values() for e in eps}
    if dev & other:
        raise ValueError("Feasible-route development sources overlap another bank role")
    layouts = json.loads((dev_bank / "layouts.json").read_text())["layouts"]
    for record in report.get("witnesses", []):
        if roots.get(record["root"]["root_id"]) != record["root"] or record["layout"] not in layouts:
            raise ValueError("Witness root/layout does not belong to this development bank")
    gate = feasibility_gate(report.get("witnesses", []), sorted(dev))
    if report.get("gate", {}).get("passes") is not True or not gate["passes"]:
        raise ValueError("Development feasible-route witness gate did not pass")
    return {**gate, "report": str(path.resolve()), "sha256": file_sha256(path),
            "generator_identity": report["generator_identity"],
            **({"frozen_sampling_plan": frozen_plan} if frozen_plan is not None else {})}
