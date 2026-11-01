"""Prospective metadata-only stress geometry, evaluated on a rented host only."""

import copy
import math


def derive_case(case, rule):
    result = copy.deepcopy(case)
    result["parent_case_id"] = case["case_id"]
    result["case_id"] += "-stress95"
    result["original_condition"] = case["condition"]
    result["seed"] = rule["seed_base"] + case["source_episode"]
    if case["shape"]["kind"] == "none":
        result["stress_derivation"] = {"applied": False, "reason": "No-hazard control unchanged"}
        return result
    if case["shape"]["kind"] != "island":
        raise ValueError("Only original circular islands and no-hazard controls are admitted")
    old = float(case["shape"]["radius"])
    clearances = [float(value) for value in case["endpoint_clearance"]]
    if len(clearances) != 2 or not all(math.isfinite(value) for value in [old, *clearances]):
        raise ValueError("Frozen finite endpoint-clearance metadata is required")
    admission = rule["margin"] + rule["motion_allowance"] + rule["admission_buffer"]
    maximum = old + min(clearances) - admission
    if maximum < old - 1e-9:
        raise ValueError("Original circle was not admitted at the unchanged buffered margin")
    radius = min(rule["maximum_radius"], max(old, rule["admissible_fraction"] * maximum))
    if radius < old:
        raise ValueError("The prospective cap cannot weaken an original hazard")
    new_clearances = [value - (radius - old) for value in clearances]
    if min(new_clearances) < admission - 1e-9:
        raise ValueError("Derived circle violates unchanged buffered endpoint admission")
    result["shape"]["radius"] = radius
    result["endpoint_clearance"] = new_clearances
    result["stress_derivation"] = {
        "applied": True,
        "old_radius": old,
        "max_admissible_radius": maximum,
        "new_radius": radius,
        "old_endpoint_clearance": clearances,
        "new_endpoint_clearance": new_clearances,
        "required_endpoint_clearance": admission,
        "radius_cap_active": radius == rule["maximum_radius"],
    }
    return result
