"""JSON-only readiness checks for both study families before shared final access."""

import hashlib
import json
import math
from pathlib import Path

ARMS = {"native", "strong_anchor", "tuned_fixed", "cma", "random", "reinforce"}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_reference(reference, root):
    path = Path(root) / reference["path"]
    require(
        digest(path) == reference["sha256"], "Joint-final reference bytes changed: " + str(path)
    )
    return json.loads(path.read_text())


def family_opening(reference, root, episode_actions):
    opening = load_reference(reference, root)
    require(
        opening.get("schema") == "lewm-navigation-final-opening-lock-v1",
        "Family opening seal required",
    )
    require(
        opening.get("all_three_repeat_selections_fixed") is True
        and opening.get("best_repeat_selection_forbidden") is True,
        "All three family repeats must be sealed",
    )
    require(opening.get("final_pixels_read") == 0, "Family opening must precede final pixels")
    refs = opening.get("selection_locks", [])
    require(len(refs) == 3, "Three family selection references required")
    locks, hosts = {}, set()
    for ref in refs:
        lock = load_reference(ref, root)
        repeat = lock.get("repeat_id")
        require(
            repeat in (0, 1, 2) and repeat not in locks and repeat == ref["repeat_id"],
            "A family repeat is absent or duplicated",
        )
        require(
            lock.get("schema") == "lewm-navigation-selection-lock-v1"
            and lock.get("final_pixels_read") == 0,
            "Invalid pre-final selection lock",
        )
        require(
            lock.get("no_best_repeat_selection") is True and lock.get("denominator_each") == 8,
            "Complete selection panels and all repeats required",
        )
        require(lock["host"] == ref["host"], "Selected host changed")
        require(
            (lock["host"]["host"], lock["host"]["pod_id"])
            in (("navigation", "vukovc3t4o1dui"), ("pusht", "wq4ar2qx2ff6r4")),
            "Family host label and declared physical pod differ",
        )
        require(
            lock["evaluation"]["maximum_episode_actions"] == episode_actions,
            "Study endpoint changed",
        )
        require(set(lock.get("selected", {})) == ARMS, "Six selected arms required")
        for chosen in lock["selected"].values():
            genes = chosen.get("parameters", [])
            require(
                len(genes) == 10 and all(math.isfinite(v) and -6 <= v <= 6 for v in genes),
                "Finite bounded ten-gene nominees required",
            )
            require(
                all(
                    key in chosen
                    for key in ("selected_is_trained", "selected_source_arm", "provenance")
                ),
                "Selected producer provenance missing",
            )
        hosts.add(lock["host"]["pod_id"])
        locks[repeat] = {"lock": lock, "reference": ref}
    require(
        set(locks) == {0, 1, 2} and len(hosts) == 2,
        "Every family needs all three repeats and two physical hosts",
    )
    return opening, locks


def validate_joint(reference, root):
    joint = load_reference(reference, root)
    require(
        joint.get("schema") == "lewm-navigation-joint-final-opening-v1",
        "Joint opening seal required",
    )
    require(
        joint.get("all_six_repeat_selections_fixed") is True
        and joint.get("final_pixels_read") == 0,
        "All six selections must precede ANY final access",
    )
    activation = load_reference(joint["secondary_activation"], root)
    require(
        activation.get("schema") == "lewm-navigation-secondary-study-choice-v1"
        and activation.get("activate_secondary") is True
        and activation.get("all_six_selection_locks_required_before_final") is True
        and activation.get("final_pixels_read") == 0,
        "Explicit secondary activation with six-lock requirement needed",
    )
    _, primary = family_opening(joint["primary_opening"], root, 50)
    _, secondary = family_opening(joint["secondary_opening"], root, 30)
    all_refs = [value["reference"] for family in (primary, secondary) for value in family.values()]
    require(
        len({ref["sha256"] for ref in all_refs}) == 6, "Six distinct family/repeat locks required"
    )
    primary_hosts = {0: "vukovc3t4o1dui", 1: "wq4ar2qx2ff6r4", 2: "vukovc3t4o1dui"}
    require(
        all(primary[r]["lock"]["host"]["pod_id"] == pod for r, pod in primary_hosts.items()),
        "Primary host assignment is immutable",
    )
    require(
        {row["lock"]["host"]["pod_id"] for row in secondary.values()}
        == set(primary_hosts.values()),
        "Secondary must retain both declared physical hosts",
    )
    return joint, primary, secondary


def validate_joint_preparation(protocol, root):
    require(
        protocol.get("schema") == "lewm-navigation-joint-final-preparation-v1"
        and protocol.get("execution_enabled") is True,
        "Inactive joint preparation design",
    )
    joint, primary, secondary = validate_joint(protocol["joint_opening"], root)
    preparation = load_reference(protocol["primary_preparation_protocol"], root)
    require(
        preparation.get("schema") == "lewm-navigation-final-case-preparation-v1"
        and preparation.get("execution_enabled") is True,
        "Original primary final preparation freeze required",
    )
    require(
        preparation["final_opening"]["sha256"] == joint["primary_opening"]["sha256"],
        "Delegated preparation opening differs",
    )
    require(
        preparation["final_episode_actions"] == 50,
        "Shared images use unchanged primary50 preparation",
    )
    for name, expected in protocol["source_sha256"].items():
        require(digest(Path(root) / name) == expected, "Joint preparation source changed: " + name)
    return joint, primary, secondary, preparation
