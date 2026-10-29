"""Combine every host's actual, conservatively charged and unmeasured interactions."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from scripts.account_unisafe09 import audit


def caps(attempts, controls, offline, replacements):
    return {"maximum_attempts": attempts, "maximum_controls": controls,
            "maximum_decimated_physics_steps": controls*5,
            "maximum_offline_controller_queries": offline,
            "maximum_controller_query_bundles": controls+offline,
            "infrastructure_replacements": replacements}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--parallel-root", required=True, type=Path)
    parser.add_argument("--parallel-warmup-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    old_source = ROOT / "progress/receipts/09-unisafe-physx-initialization-source-v2.json"
    old = audit(args.original_root, caps(398, 127640, 640, 4), old_source)
    new = audit(args.parallel_root, caps(208, 70720, 4360, 4), args.parallel_warmup_source)
    combined = {"schema": "unisafe09-combined-accounting-v1", "utc": utc(),
                "by_physical_host": {"150513": old, "36475": new},
                "scope": "All engineering, screen and confirmation attempts, including invalid starts",
                "complete_internal_physics_measurement": False,
                "physics_counter_scope": old["physics_counter_scope"],
                "optimizer_updates": 0, "caps": caps(606, 198360, 5000, 8)}
    summed = (
        "known_completed_controls", "explicit_controls_conservatively_charged",
        "known_decimated_physics_steps", "decimated_physics_conservatively_charged",
        "recorded_simulation_context_reset_steps",
        "source_derived_backend_warmup_steps_lower_bound",
        "known_nonrollout_physics_steps_lower_bound", "known_total_physics_steps_lower_bound",
        "recorded_controller_query_bundles_started", "recorded_shadow_query_bundles_started",
        "offline_query_bundles_conservatively_charged", "total_query_bundles_conservatively_charged",
    )
    for key in summed:
        combined[key] = old[key]+new[key]
    for key in ("recorded_component_queries", "recorded_shadow_component_queries"):
        count = Counter(old[key])
        count.update(new[key])
        combined[key] = dict(count)
    combined["attempts"] = [dict(row, physical_host=host)
                            for host, report in combined["by_physical_host"].items()
                            for row in report["attempts"]]
    combined["stages"] = {}
    for report in (old, new):
        for stage, counts in report["stages"].items():
            total = combined["stages"].setdefault(stage, {"attempted": 0, "valid": 0, "invalid": 0})
            for key, value in counts.items():
                total[key] += value
    combined["all_attempts_terminal"] = old["all_attempts_terminal"] and new["all_attempts_terminal"]
    combined["cap_checks"] = {
        "each_host_within_assigned_caps": old["all_caps_respected"] and new["all_caps_respected"],
        "attempts": len(combined["attempts"]) <= 606,
        "explicit_controls": combined["explicit_controls_conservatively_charged"] <= 198360,
        "offline_query_bundles": combined["offline_query_bundles_conservatively_charged"] <= 5000,
        "total_query_bundles": combined["total_query_bundles_conservatively_charged"] <= 203360,
    }
    combined["all_caps_respected"] = all(combined["cap_checks"].values())
    atomic_json(args.output, combined)
    print(json.dumps({k: combined[k] for k in (
        "stages", "all_attempts_terminal", "all_caps_respected", "known_completed_controls",
        "explicit_controls_conservatively_charged", "total_query_bundles_conservatively_charged"
    )}))
    if not combined["all_attempts_terminal"] or not combined["all_caps_respected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
