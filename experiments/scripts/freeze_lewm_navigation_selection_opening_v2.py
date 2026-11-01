"""Seal three secondary selection locks on their three declared physical hosts."""

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm_navigation_joint_final_v2 import (
    digest,
    load_reference,
    require,
    validate_family_value,
    validate_relocation,
)


def reference(path):
    path = path.resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}


def seal(args):
    relocation = reference(args.relocation_record)
    validate_relocation(relocation, ROOT)
    refs, locks = [], []
    for path in args.locks:
        ref = reference(path)
        lock = load_reference(ref, ROOT)
        refs.append({**ref, "repeat_id": lock["repeat_id"], "host": lock["host"]})
        locks.append(lock)
    require(
        sorted(lock["repeat_id"] for lock in locks) == [0, 1, 2],
        "All three secondary repeats must be retained",
    )
    require(
        all(lock["source_sha256"] == locks[0]["source_sha256"] for lock in locks),
        "Scientific source families differ",
    )
    fields = (
        "world_model",
        "motion_model",
        "scalers",
        "memory_path",
        "memory_sha256",
        "controller_backend",
        "controller_config",
        "goal_readout",
        "planner",
        "precision",
        "body_radius",
        "margin",
        "motion_allowance",
        "maximum_episode_actions",
    )
    require(
        all(
            all(lock["evaluation"].get(key) == locks[0]["evaluation"].get(key) for key in fields)
            for lock in locks
        ),
        "Shared model/controller/information/planner contract differs",
    )
    shared_assets = {
        locks[0]["evaluation"][key]
        for key in ("world_model", "motion_model", "scalers", "memory_path")
    }
    shared_assets.add(locks[0]["evaluation"]["goal_readout"]["path"])
    require(
        all(
            all(
                lock["evaluation"]["asset_sha256"][name]
                == locks[0]["evaluation"]["asset_sha256"][name]
                for name in shared_assets
            )
            for lock in locks
        ),
        "Shared scientific asset hashes differ",
    )
    value = {
        "schema": "lewm-navigation-final-opening-lock-v2",
        "locked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_locks": refs,
        "secondary_relocation": relocation,
        "all_three_repeat_selections_fixed": True,
        "best_repeat_selection_forbidden": True,
        "physical_host_blocks": 3,
        "episode_actions": 30,
        "final_pixels_read": 0,
        "new_model_simulator_calls": 0,
        "builder_sha256": digest(Path(__file__)),
        "next_stage": "Combine this secondary three-host opening with the unchanged primary two-host opening using the new joint-v2 guard. All six selections must precede any final pixels or final outcome.",
    }
    validate_family_value(value, ROOT, 30)
    require(not args.out.exists(), "Never overwrite a frozen secondary opening")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return reference(args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locks", nargs=3, type=Path, required=True)
    parser.add_argument("--relocation-record", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(seal(args), indent=2))


if __name__ == "__main__":
    main()
