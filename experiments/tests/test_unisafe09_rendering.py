"""A missing renderer must fail before it can produce accepted policy outcomes."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import unisafe09_rendering as rendering


def test_each_camera_must_have_real_image_content():
    cameras = np.zeros((2, 128, 128, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="Camera 0"):
        rendering.validate_cameras(cameras)
    cameras[0, 3, 4, 0] = 123
    with pytest.raises(RuntimeError, match="Camera 1"):
        rendering.validate_cameras(cameras)
    cameras[1, 2, 5, 2] = 210
    rendering.validate_cameras(cameras)
    with pytest.raises(ValueError, match="shape or dtype"):
        rendering.validate_cameras(cameras.astype(np.float32))


def test_vulkan_must_enumerate_the_actual_gpu(tmp_path, monkeypatch):
    config = tmp_path / "configs/unisafe09/vulkan-icd.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    monkeypatch.setenv("VK_DRIVER_FILES", "old")
    monkeypatch.setenv("VK_ICD_FILENAMES", "old")
    monkeypatch.setattr(
        rendering.subprocess, "run", lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="software Vulkan renderer", stderr=""
        )
    )
    with pytest.raises(RuntimeError, match="RTX 3090"):
        rendering.verify_renderer(tmp_path)
    monkeypatch.setattr(
        rendering.subprocess, "run", lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="deviceName = NVIDIA GeForce RTX 3090", stderr=""
        )
    )
    assert rendering.verify_renderer(tmp_path)["exit_code"] == 0
    assert rendering.os.environ["VK_DRIVER_FILES"] == str(config)
    assert rendering.os.environ["VK_ICD_FILENAMES"] == str(config)


def test_invalidated_completed_attempt_keeps_full_budget_charge(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_unisafe09.py"
    spec = importlib.util.spec_from_file_location("run_unisafe09_rendering_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    folder = tmp_path / "attempts/0000-engineering-00"
    folder.mkdir(parents=True)
    (folder / "intent.json").write_text(json.dumps({
        "maximum_controls": 340, "identity_shadow": True,
        "job": {"id": "engineering-00"}, "replacement": False,
    }))
    (folder / "summary.json").write_text(json.dumps({
        "status": "complete", "controls": 37, "identity_checks": 37,
    }))
    (folder / "invalidation.json").write_text(json.dumps({"reason": "blank cameras"}))
    row, = runner.inventory(tmp_path)
    assert not row["valid"]
    assert row["charged_controls"] == 340
    assert row["offline_queries"] == 340
    assert row["invalidation"]["reason"] == "blank cameras"
