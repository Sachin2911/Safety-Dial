"""Fail-closed JSON lock checks before any final dataset, renderer or model access."""

import hashlib
import json
import math
from pathlib import Path

ARMS = {"native", "strong_anchor", "tuned_fixed", "cma", "random", "reinforce"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_final_opening(protocol, root):
    """Read locks and protocol JSON only; never inspect final metadata or pixels."""
    root = Path(root)
    require(
        protocol.get("schema") == "lewm-navigation-final-case-preparation-v1",
        "Wrong final protocol",
    )
    require(protocol.get("execution_enabled") is True, "Design template cannot open final data")
    reference = protocol.get("final_opening")
    require(isinstance(reference, dict), "All three selection locks must precede final access")
    path = root / reference["path"]
    require(digest(path) == reference["sha256"], "Final-opening lock bytes changed")
    opening = json.loads(path.read_text())
    require(
        opening.get("schema") == "lewm-navigation-final-opening-lock-v1", "Invalid opening schema"
    )
    require(opening.get("all_three_repeat_selections_fixed") is True, "All three repeats required")
    require(
        opening.get("best_repeat_selection_forbidden") is True, "Best-repeat selection forbidden"
    )
    require(opening.get("final_pixels_read") == 0, "Opening must precede final pixels")
    refs = opening.get("selection_locks", [])
    require(
        len(refs) == 3 and refs == protocol.get("selection_locks"),
        "Exact three lock references required",
    )
    repeats, hosts, caps = [], set(), set()
    for ref in refs:
        lock_path = root / ref["path"]
        require(digest(lock_path) == ref["sha256"], "Selected weights/provenance lock changed")
        lock = json.loads(lock_path.read_text())
        require(lock.get("schema") == "lewm-navigation-selection-lock-v1", "Invalid selection lock")
        require(
            lock.get("final_pixels_read") == 0 and lock.get("no_best_repeat_selection") is True,
            "Selection must be sealed before final access",
        )
        require(
            lock["repeat_id"] == ref["repeat_id"] and lock["host"] == ref["host"],
            "Lock repeat/host changed",
        )
        require(set(lock.get("selected", {})) == ARMS, "All six arm selections required")
        for nominee in lock["selected"].values():
            genes = nominee.get("parameters", [])
            require(
                len(genes) == 10 and all(math.isfinite(value) for value in genes),
                "Finite ten-gene nominees required",
            )
            require(
                "selected_is_trained" in nominee and "provenance" in nominee,
                "Nominee training provenance required",
            )
        require(lock.get("denominator_each") == 8, "Complete selection panel required")
        repeats.append(lock["repeat_id"])
        hosts.add(lock["host"]["pod_id"])
        caps.add(lock["evaluation"]["maximum_episode_actions"])
    require(
        sorted(repeats) == [0, 1, 2] and len(hosts) == 2,
        "All repeats and physical-host blocks required",
    )
    require(
        len(caps) == 1 and caps == {protocol["final_episode_actions"]},
        "Final action endpoint differs from selection",
    )
    pool = protocol.get("opened_final_episode_ids")
    require(pool == list(range(5000, 5256)), "Only the prospectively sealed final pool may open")
    require(
        protocol.get("prepare_pools") == {"heldout_final": pool}, "Only final cases may be prepared"
    )
    require(
        protocol.get("eligible_quota_per_condition")
        == {"none": 4, "circle3": 4, "circle5": 4, "circle7": 4},
        "Unchanged balanced sixteen-case quota required",
    )
    require(
        protocol.get("geometry")
        == {
            "source_stride": 5,
            "endpoint_offsets": [100, 75, 50, 25],
            "minimum_displacement": 60.0,
            "maximum_displacement": 160.0,
            "body_radius": 7.0,
            "margin": 3.0,
            "allowance": 12.405431747436523,
            "admission_buffer": 2.0,
            "require_room_contained": True,
            "left_room_max_x": 99.5,
            "right_room_min_x": 124.5,
        },
        "Final admission must preserve the complete v3 geometry/temporal contract",
    )
    require(
        protocol.get("condition_cycle") == ["none", "circle3", "circle5", "circle7"],
        "Original 3/5/7px radius assignment must remain unchanged",
    )
    require(
        protocol.get("planning_seed_base") == 713000
        and protocol.get("maximum_preparation_resets") == 16
        and protocol.get("maximum_preparation_actions") == 0,
        "Final seeds or preparation interaction cap changed",
    )
    return {
        "opening_sha256": reference["sha256"],
        "selection_locks": refs,
        "episode_actions": next(iter(caps)),
        "final_pixels_read_before_opening": 0,
    }
