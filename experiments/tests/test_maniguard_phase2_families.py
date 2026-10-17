"""Validate reserved family exposure, base-scene grouping and pinned execution."""

import hashlib
import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_phase2_families import (
    CONFIG,
    deterministic_partitions,
    family_spec,
    load_registry,
    load_sources,
    load_splits,
    scene_base_id,
    task_split,
    validate_registry,
    validate_scene_access,
)


def test_complete_registry_preserves_upstream_execution_and_phase_one():
    audit = validate_registry()
    assert audit == {"families": 6, "base_tasks": 200,
                     "pilot_families": ["jar", "stack", "clutter"],
                     "behavioral_outcomes_loaded": False}
    expected = {"jar": (2000, "assisted"), "stack": (4200, "sticky"),
                "lid": (1900, "sticky"), "clutter": (1100, "assisted"),
                "cabinet": (5700, "sticky"), "dusty": (5100, "sticky")}
    for name, pair in expected.items():
        spec = family_spec(name)
        assert (spec["max_steps"], spec["grasping_mode"]) == pair
        assert spec["controller_preset"] == "joint_position_raw"


@pytest.mark.parametrize("family", ["jar", "lid", "stack"])
def test_phase_one_assignments_and_policies_unchanged(family):
    old = json.loads((ROOT / "configs/maniguard/splits.json").read_text())
    assert load_splits()["families"][family] == old["families"][family]
    old_sources = json.loads((ROOT / "configs/maniguard/sources.json").read_text())
    assert load_sources()["policies"][family] == old_sources["policies"][family]


def test_every_condition_inherits_the_base_scene_partition():
    for family, spec in load_registry().items():
        partitions = load_splits()["families"][family]
        for split, tasks in partitions.items():
            for task in tasks:
                for condition in spec["conditions"]:
                    scene = f"{spec['benchmark_dir']}/{task}/{condition}"
                    assert task_split(family, scene) == split
                    assert scene_base_id(family, scene) == task


@pytest.mark.parametrize("family", ["clutter", "cabinet", "dusty"])
def test_new_partitions_are_reproducible_and_independent_of_listing_order(family):
    spec, splits = family_spec(family), load_splits()
    expected = splits["families"][family]
    assert deterministic_partitions(family, list(reversed(spec["task_ids"])),
                                    splits["seed"]) == expected
    assert [len(expected[key]) for key in ("engineering", "validation", "test")] == [4, 4, 12]


@pytest.mark.parametrize("family", ["cabinet", "dusty", "lid"])
def test_transfer_cannot_leak_into_training_or_reference_selection(family):
    parts = load_splits()["families"][family]
    for purpose in ("engineering", "train", "validation", "test"):
        scene = parts[purpose][0] + "/base"
        assert validate_scene_access(family, scene, "metadata") == purpose
        with pytest.raises(ValueError, match="excluded from development"):
            validate_scene_access(family, scene, purpose, frozen_evaluation=True)
        with pytest.raises(ValueError, match="freeze transfer"):
            validate_scene_access(family, scene, "transfer")
        assert validate_scene_access(family, scene, "transfer", frozen_evaluation=True) == purpose
    assert family_spec("lid")["phase1_outcomes_exposed"]
    assert not family_spec("cabinet")["phase1_outcomes_exposed"]


def test_development_partition_and_reserved_test_enforcement():
    parts = load_splits()["families"]["clutter"]
    training = parts["train"][0] + "/language"
    assert validate_scene_access("clutter", training, "train") == "train"
    with pytest.raises(ValueError, match="not permitted"):
        validate_scene_access("clutter", training, "validation")
    test = parts["test"][0] + "/base"
    with pytest.raises(ValueError, match="frozen evaluation"):
        validate_scene_access("clutter", test, "test")
    assert validate_scene_access("clutter", test, "test", frozen_evaluation=True) == "test"
    with pytest.raises(ValueError, match="not whole-family transfer"):
        validate_scene_access("clutter", test, "transfer", frozen_evaluation=True)


@pytest.mark.parametrize("scene", ["../task_0000/base", "task_0000/../base",
                                   "task_0000/unknown", "task_9999/base", ""])
def test_scene_paths_are_not_arbitrary_filesystem_paths(scene):
    with pytest.raises(ValueError):
        scene_base_id("jar", scene)


def test_registry_rejects_overlap_even_if_total_assignment_count_matches(tmp_path):
    for filename in ("families.json", "splits.json", "sources.json"):
        shutil.copyfile(CONFIG / filename, tmp_path / filename)
    path = tmp_path / "splits.json"
    splits = json.loads(path.read_text())
    parts = splits["families"]["clutter"]
    parts["test"][0] = parts["train"][0]
    path.write_text(json.dumps(splits))
    with pytest.raises(ValueError, match="overlapping"):
        validate_registry(tmp_path)


def _downloader():
    path = ROOT / "scripts/maniguard/phase2_download.py"
    spec = importlib.util.spec_from_file_location("phase2_download_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_verifies_content_and_never_overwrites_existing_destination(tmp_path, monkeypatch):
    module = _downloader()
    payload = b"public pinned checkpoint fixture"
    entry = {"remote": "7100/params/file", "relative": "7100/params/file",
             "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
             "git_blob_sha1": None}
    destination = tmp_path / "clutter"
    plan = {"destination": str(destination), "repo": "example/public", "revision": "a" * 40,
            "repo_type": "model", "files": [entry], "bytes": len(payload)}
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))
    record = module.download(plan, reserve_bytes=0)
    assert record["complete"] and not record["behavioral_evaluation_performed"]
    target = destination / entry["relative"]
    assert target.read_bytes() == payload
    assert module.download(plan, reserve_bytes=0) == record
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="integrity verification"):
        module.download(plan, reserve_bytes=0)
    assert target.read_bytes() == b"changed"


def test_public_metadata_refuses_gated_repository(monkeypatch):
    module = _downloader()
    raw = json.dumps({"sha": "a" * 40, "private": False, "gated": "auto"}).encode()
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(raw))
    with pytest.raises(ValueError, match="private or gated"):
        module.public_metadata("example/repo", "a" * 40, "model")
