"""Account for the fallback: all C on the original host, added host stopped before simulation."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.unisafe09_protocol import digest
from scripts.account_unisafe09 import audit

GLOBAL_CAPS = {"maximum_attempts": 606, "maximum_controls": 198360,
               "maximum_decimated_physics_steps": 991800,
               "maximum_offline_controller_queries": 5000,
               "maximum_controller_query_bundles": 203360}
ADDED_ENGINEERING_ALLOWANCE = 16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--parallel-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    freeze_path = ROOT / "progress/receipts/09-unisafe-C.json"
    freeze = json.loads(freeze_path.read_text())
    warmup = ROOT / "progress/receipts/09-unisafe-physx-initialization-source-v2.json"
    # The original host keeps its own frozen caps (590 attempts) under the fallback.
    old = audit(args.original_root, freeze["design"]["caps"], warmup)
    added = sorted((args.parallel_root / "attempts").glob("*/intent.json"))
    if added:
        raise ValueError("Added-host attempts exist; this fallback accounting does not apply")
    controls = old["explicit_controls_conservatively_charged"]
    offline = old["offline_query_bundles_conservatively_charged"]
    checks = {
        "original_within_frozen_caps": old["all_caps_respected"],
        "added_host_attempts_within_engineering_allowance": len(added)
        <= ADDED_ENGINEERING_ALLOWANCE,
        "attempts": len(old["attempts"]) + len(added) <= GLOBAL_CAPS["maximum_attempts"],
        "explicit_controls": controls <= GLOBAL_CAPS["maximum_controls"],
        "decimated_physics": controls*5 <= GLOBAL_CAPS["maximum_decimated_physics_steps"],
        "offline_query_bundles": offline <= GLOBAL_CAPS["maximum_offline_controller_queries"],
        "total_query_bundles": controls+offline <= GLOBAL_CAPS["maximum_controller_query_bundles"],
    }
    result = dict(old)
    result.update({
        "schema": "unisafe09-fallback-accounting-v1", "utc": utc(),
        "scope": "All engineering, screen and confirmation attempts on both rentals, "
                 "including invalid starts",
        "execution": "Recorded fallback: added-host C admission failed before any added-host "
                     "simulation; all 384 C jobs ran on physical host 150513",
        "by_physical_host": {
            "150513": {"instance_id": 57647787, "attempts": len(old["attempts"]),
                       "caps": freeze["design"]["caps"]},
            "36475": {"instance_id": 57690181, "attempts": 0, "simulator_interactions": 0,
                      "controller_queries": 0,
                      "allowance": f"at most {ADDED_ENGINEERING_ALLOWANCE} engineering attempts"},
        },
        "global_caps": GLOBAL_CAPS, "cap_checks": checks,
        "all_caps_respected": all(checks.values()),
        "admission_failure_sha256": digest(
            ROOT / "progress/receipts/09-unisafe-parallel-C-admission-failure.json"),
        "remaining_admission_sha256": digest(
            ROOT / "progress/receipts/09-unisafe-C-remaining-admission.json"),
    })
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in (
        "stages", "all_attempts_terminal", "all_caps_respected", "known_completed_controls",
        "explicit_controls_conservatively_charged", "total_query_bundles_conservatively_charged",
    )}))
    if not result["all_attempts_terminal"] or not result["all_caps_respected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
