"""Synthetic JSON-only final-opening checks, executable on rented hosts only."""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.lewm_navigation_final_admission import ARMS, validate_final_opening

pytestmark = pytest.mark.skipif(
    os.environ.get("RUNPOD_POD_ID") not in ("vukovc3t4o1dui", "wq4ar2qx2ff6r4"),
    reason="Rented campaign host required",
)


def write(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path):
    root = Path(__file__).resolve().parents[2]
    protocol = json.loads(
        (
            root / "configs/lewm_campaign_20261031/navigation-final-cases-template-v1.json"
        ).read_text()
    )
    refs = []
    for repeat in range(3):
        host = {"pod_id": "navigation" if repeat != 1 else "pusht", "repeat_id": repeat}
        lock = {
            "schema": "lewm-navigation-selection-lock-v1",
            "repeat_id": repeat,
            "host": host,
            "final_pixels_read": 0,
            "no_best_repeat_selection": True,
            "denominator_each": 8,
            "evaluation": {"maximum_episode_actions": 50},
            "selected": {
                arm: {
                    "parameters": [0.25] * 10,
                    "selected_is_trained": False,
                    "provenance": {"synthetic": True},
                }
                for arm in ARMS
            },
        }
        path = tmp_path / f"lock-{repeat}.json"
        refs.append(
            {"path": str(path), "sha256": write(path, lock), "repeat_id": repeat, "host": host}
        )
    opening = {
        "schema": "lewm-navigation-final-opening-lock-v1",
        "all_three_repeat_selections_fixed": True,
        "best_repeat_selection_forbidden": True,
        "final_pixels_read": 0,
        "selection_locks": refs,
    }
    path = tmp_path / "opening.json"
    protocol.update(
        execution_enabled=True,
        final_opening={"path": str(path), "sha256": write(path, opening)},
        selection_locks=refs,
    )
    return protocol


def test_disabled_template_rejects_before_any_opening_or_dataset_access(tmp_path):
    with pytest.raises(ValueError, match="Design template"):
        validate_final_opening(
            {"schema": "lewm-navigation-final-case-preparation-v1", "execution_enabled": False},
            tmp_path,
        )
    assert not list(tmp_path.iterdir())


def test_complete_three_repeat_json_lock_chain_qualifies_without_dataset(tmp_path):
    protocol = fixture(tmp_path)
    result = validate_final_opening(protocol, tmp_path)
    assert result["episode_actions"] == 50 and len(result["selection_locks"]) == 3
    assert not (tmp_path / "tworoom.h5").exists()


def test_changed_selected_weight_bytes_rejected_before_dataset_access(tmp_path):
    protocol = fixture(tmp_path)
    path = Path(protocol["selection_locks"][0]["path"])
    lock = json.loads(path.read_text())
    lock["selected"]["cma"]["parameters"][0] = 5.0
    write(path, lock)
    with pytest.raises(ValueError, match="weights/provenance lock changed"):
        validate_final_opening(protocol, tmp_path)


def test_final_pool_and_geometry_cannot_change_after_selection(tmp_path):
    protocol = fixture(tmp_path)
    changed = copy.deepcopy(protocol)
    changed["geometry"]["margin"] = 2
    with pytest.raises(ValueError, match="complete v3"):
        validate_final_opening(changed, tmp_path)
    changed = copy.deepcopy(protocol)
    changed["opened_final_episode_ids"] = [5000]
    with pytest.raises(ValueError, match="sealed final pool"):
        validate_final_opening(changed, tmp_path)


def test_single_repeat_or_different_final_action_endpoint_rejected(tmp_path):
    protocol = fixture(tmp_path)
    protocol["selection_locks"] = protocol["selection_locks"][:1]
    with pytest.raises(ValueError, match="Exact three lock"):
        validate_final_opening(protocol, tmp_path)
    protocol = fixture(tmp_path)
    protocol["final_episode_actions"] = 150
    with pytest.raises(ValueError, match="endpoint differs"):
        validate_final_opening(protocol, tmp_path)
