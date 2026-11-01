"""Synthetic six-lock guard checks, runnable only on declared rented hosts."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.lewm_navigation_joint_final import validate_joint

pytestmark = pytest.mark.skipif(
    os.environ.get("RUNPOD_POD_ID") not in ("vukovc3t4o1dui", "wq4ar2qx2ff6r4"),
    reason="Rented campaign host required",
)


def write(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def fixture(tmp_path):
    groups = {}
    for family, cap in (("primary", 50), ("secondary", 30)):
        refs = []
        for repeat in range(3):
            nav = repeat == 0 or (family == "primary" and repeat == 2)
            host = {
                "host": "navigation" if nav else "pusht",
                "pod_id": "vukovc3t4o1dui" if nav else "wq4ar2qx2ff6r4",
                "repeat_id": repeat,
            }
            lock = {
                "schema": "lewm-navigation-selection-lock-v1",
                "repeat_id": repeat,
                "host": host,
                "final_pixels_read": 0,
                "no_best_repeat_selection": True,
                "denominator_each": 8,
                "evaluation": {"maximum_episode_actions": cap},
                "selected": {
                    arm: {
                        "parameters": [0.25] * 10,
                        "selected_is_trained": False,
                        "selected_source_arm": "strong_anchor",
                        "provenance": {"synthetic": True},
                    }
                    for arm in (
                        "native",
                        "strong_anchor",
                        "tuned_fixed",
                        "cma",
                        "random",
                        "reinforce",
                    )
                },
            }
            ref = write(tmp_path / f"{family}-lock-{repeat}.json", lock)
            refs.append({**ref, "repeat_id": repeat, "host": host})
        groups[family] = write(
            tmp_path / f"{family}-opening.json",
            {
                "schema": "lewm-navigation-final-opening-lock-v1",
                "all_three_repeat_selections_fixed": True,
                "best_repeat_selection_forbidden": True,
                "final_pixels_read": 0,
                "selection_locks": refs,
            },
        )
    activation = write(
        tmp_path / "activation.json",
        {
            "schema": "lewm-navigation-secondary-study-choice-v1",
            "activate_secondary": True,
            "all_six_selection_locks_required_before_final": True,
            "final_pixels_read": 0,
        },
    )
    joint = {
        "schema": "lewm-navigation-joint-final-opening-v1",
        "primary_opening": groups["primary"],
        "secondary_opening": groups["secondary"],
        "secondary_activation": activation,
        "all_six_repeat_selections_fixed": True,
        "final_pixels_read": 0,
    }
    return write(tmp_path / "joint.json", joint)


def test_all_six_lock_groups_and_secondary_pusht_repeat2_are_retained(tmp_path):
    reference = fixture(tmp_path)
    _, primary, secondary = validate_joint(reference, tmp_path)
    assert len(primary) == len(secondary) == 3
    assert primary[2]["lock"]["host"]["host"] == "navigation"
    assert secondary[2]["lock"]["host"]["host"] == "pusht"


def test_primary_only_readiness_cannot_open_joint_final(tmp_path):
    reference = fixture(tmp_path)
    joint_path = Path(reference["path"])
    joint = json.loads(joint_path.read_text())
    opening_path = Path(joint["secondary_opening"]["path"])
    opening = json.loads(opening_path.read_text())
    opening["selection_locks"] = opening["selection_locks"][:2]
    joint["secondary_opening"] = write(opening_path, opening)
    reference = write(joint_path, joint)
    with pytest.raises(ValueError, match="Three family"):
        validate_joint(reference, tmp_path)


def test_changed_secondary_weight_bytes_are_rejected(tmp_path):
    reference = fixture(tmp_path)
    path = tmp_path / "secondary-lock-2.json"
    lock = json.loads(path.read_text())
    lock["selected"]["cma"]["parameters"][0] = 1.0
    write(path, lock)
    with pytest.raises(ValueError, match="reference bytes changed"):
        validate_joint(reference, tmp_path)


def test_families_cannot_be_sealed_as_one_common_action_endpoint(tmp_path):
    reference = fixture(tmp_path)
    joint_path = Path(reference["path"])
    joint = json.loads(joint_path.read_text())
    opening_path = Path(joint["secondary_opening"]["path"])
    opening = json.loads(opening_path.read_text())
    ref = opening["selection_locks"][0]
    lock_path = Path(ref["path"])
    lock = json.loads(lock_path.read_text())
    lock["evaluation"]["maximum_episode_actions"] = 50
    opening["selection_locks"][0] = {**ref, **write(lock_path, lock)}
    joint["secondary_opening"] = write(opening_path, opening)
    reference = write(joint_path, joint)
    with pytest.raises(ValueError, match="Study endpoint changed"):
        validate_joint(reference, tmp_path)


def test_disabled_wrapper_rejects_before_heavy_import_or_output(tmp_path):
    path = (
        Path(__file__).resolve().parents[1] / "scripts/prepare_lewm_navigation_joint_final_cases.py"
    )
    spec = importlib.util.spec_from_file_location("joint_wrapper_synthetic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = tmp_path / "inactive.json"
    protocol.write_text(
        json.dumps(
            {"schema": "lewm-navigation-joint-final-preparation-v1", "execution_enabled": False}
        )
    )
    before = "prepare_lewm_navigation_final_cases" in sys.modules
    with pytest.raises(ValueError, match="Inactive joint"):
        module.prepare(protocol, tmp_path / "final-data")
    assert not (tmp_path / "final-data").exists()
    assert ("prepare_lewm_navigation_final_cases" in sys.modules) == before
