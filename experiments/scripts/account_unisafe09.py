"""Audit every retained attempt, including invalid starts and query reservations."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments/scripts")]
from helpers.unisafe09_infra import atomic_json, utc
from run_unisafe09 import inventory


def audit(root, caps, warmup_source):
    source = json.loads(warmup_source.read_text())
    if source["sha256"] != "dc1a3fbabbb54a4a3f7c45bf10e25643a346b06a6d9577e8f974584df9869742":
        raise ValueError("Backend warmup inference requires the inspected PhysX source")
    warmup = source["methods"]["_warmup_and_create_views"]["source"]
    if warmup.count("physx.update_simulation(cls.get_physics_dt(), 0.0)") != 2:
        raise ValueError("The inspected first-construction warmup path differs")
    stages = {}
    controls = 0
    reset_physics = 0
    missing_reset_counters = []
    completed_constructions = []
    incomplete = []
    queries, shadows = Counter(), Counter()
    attempted_bundles = shadow_bundles = 0
    rows = inventory(root)
    details = []
    for row in rows:
        folder = Path(row["directory"])
        stage = stages.setdefault(row["job"]["stage"], dict(attempted=0, valid=0, invalid=0))
        stage["attempted"] += 1
        stage["valid" if row["valid"] else "invalid"] += 1
        account_file = folder / "accounting.json"
        account = json.loads(account_file.read_text()) if account_file.exists() else {}
        intent = json.loads((folder / "intent.json").read_text())
        if intent["instance_id"] != source["instance_id"]:
            raise ValueError("Warmup source inspection belongs to another rental")
        if account.get("physics_after_construction") is not None:
            completed_constructions.append(folder.name)
        summary_file = folder / "summary.json"
        summary = json.loads(summary_file.read_text()) if summary_file.exists() else {}
        process_file = folder / "process-result.json"
        if not process_file.exists():
            incomplete.append(folder.name)
        known_controls = int(summary.get("controls", account.get("completed_controls", 0)))
        controls += known_controls
        queries.update(summary.get("query_counts", account.get("controller_queries", {})))
        shadows.update(summary.get("shadow_query_counts", account.get("shadow_queries", {})))
        attempted_bundles += int(account.get("query_bundles_started", 0))
        shadow_bundles += int(account.get("shadow_bundles_started", 0))
        if "reset_internal_physics_steps" in summary:
            reset_physics += summary["reset_internal_physics_steps"]
        else:
            missing_reset_counters.append(folder.name)
        details.append({
            "attempt": folder.name, "job": row["job"]["id"],
            "stage": row["job"]["stage"], "freeze_sha256": row["freeze_sha256"],
            "valid": row["valid"], "replacement": row["replacement"],
            "known_completed_controls": known_controls,
            "charged_controls": row["charged_controls"],
            "charged_offline_query_bundles": row["offline_queries"],
            "original_summary_retained": bool(summary),
            "recorded_simulation_context_reset_steps": summary.get(
                "reset_internal_physics_steps"
            ),
            "source_derived_backend_warmup_steps_lower_bound": (
                2 if folder.name in completed_constructions else 0
            ),
            "invalidation": row.get("invalidation"),
        })
    charged_controls = sum(r["charged_controls"] for r in rows)
    charged_offline = sum(r["offline_queries"] for r in rows)
    checks = {
        "attempts": len(rows) <= caps["maximum_attempts"],
        "replacement_slots": sum(r["replacement"] for r in rows)
        <= caps["infrastructure_replacements"],
        "explicit_controls": charged_controls <= caps["maximum_controls"],
        "decimated_physics": charged_controls * 5 <= caps["maximum_decimated_physics_steps"],
        "offline_query_bundles": charged_offline <= caps["maximum_offline_controller_queries"],
        "total_query_bundles": charged_controls + charged_offline
        <= caps["maximum_controller_query_bundles"],
    }
    return {
        "schema": "unisafe09-accounting-v2",
        "utc": utc(), "stages": stages, "attempts": details,
        "all_attempts_terminal": not incomplete, "incomplete_attempts": incomplete,
        "caps": caps, "cap_checks": checks, "all_caps_respected": all(checks.values()),
        "known_completed_controls": controls,
        "explicit_controls_conservatively_charged": charged_controls,
        "known_decimated_physics_steps": controls * 5,
        "decimated_physics_conservatively_charged": charged_controls * 5,
        "recorded_simulation_context_reset_steps": reset_physics,
        "attempts_without_reset_counter": missing_reset_counters,
        "source_derived_backend_warmup_steps_lower_bound": 2 * len(completed_constructions),
        "warmup_covered_attempts": completed_constructions,
        "known_nonrollout_physics_steps_lower_bound": reset_physics
        + 2 * len(completed_constructions),
        "known_total_physics_steps_lower_bound": controls * 5 + reset_physics
        + 2 * len(completed_constructions),
        "complete_internal_physics_measurement": False,
        "attempts_with_unknown_reset_internal_physics": [
            Path(row["directory"]).name for row in rows
        ],
        "physics_counter_scope": (
            "The recorded counter counts SimulationContext.step calls, not backend warmup. "
            "The pinned PhysX implementation performs two update_simulation calls during "
            "the first completed environment construction in each fresh worker. These are "
            "counted separately as a source-derived lower bound. Additional backend-internal "
            "updates were not instrumented; their complete total remains unknown."
        ),
        "warmup_source_receipt": str(warmup_source),
        "warmup_source_receipt_sha256": hashlib.sha256(warmup_source.read_bytes()).hexdigest(),
        "backend_source_sha256": source["sha256"],
        "recorded_controller_query_bundles_started": attempted_bundles,
        "recorded_shadow_query_bundles_started": shadow_bundles,
        "recorded_component_queries": dict(queries),
        "recorded_shadow_component_queries": dict(shadows),
        "offline_query_bundles_conservatively_charged": charged_offline,
        "total_query_bundles_conservatively_charged": charged_controls + charged_offline,
        "recorded_query_counts_are_lower_bounds_for_interrupted_attempts": True,
        "reset_internal_physics_unknown_is_not_zero": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/unisafe09/remote")
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--warmup-source", type=Path,
        default=ROOT / "progress/receipts/09-unisafe-physx-initialization-source-v2.json",
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text())
    result = audit(args.root, freeze["design"]["caps"], args.warmup_source)
    result["scope"] = "Retained attempts only; stage completeness requires the separate validator"
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in (
        "stages", "all_attempts_terminal", "all_caps_respected", "known_completed_controls",
        "explicit_controls_conservatively_charged",
        "source_derived_backend_warmup_steps_lower_bound", "complete_internal_physics_measurement"
    )}))
    if not result["all_caps_respected"] or (
        not args.allow_incomplete and not result["all_attempts_terminal"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
