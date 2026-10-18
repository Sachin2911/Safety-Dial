"""Provider mutation must remain unreachable until preservation and worker exit pass."""

import importlib
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts/maniguard"
sys.path.insert(0, str(SCRIPTS))
instance = importlib.import_module("followup_instance")


def forbidden(*args, **kwargs):
    raise AssertionError("Provider must not be called before closeout guards pass")


def test_wrong_instance_never_reaches_provider(monkeypatch):
    monkeypatch.setenv("CONTAINER_ID", "123")
    monkeypatch.setattr(instance, "request", forbidden)
    with pytest.raises(ValueError, match="identity"):
        instance.stop()


def test_pending_preservation_never_reaches_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTAINER_ID", str(instance.INSTANCE))
    monkeypatch.setattr(instance, "RUN", tmp_path)
    monkeypatch.setattr(instance, "request", forbidden)
    (tmp_path / "final-preservation.json").write_text(
        json.dumps({"ready_for_provider_stop": False})
    )
    with pytest.raises(ValueError, match="verified first"):
        instance.stop()


def test_changed_commit_never_reaches_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTAINER_ID", str(instance.INSTANCE))
    monkeypatch.setattr(instance, "RUN", tmp_path)
    monkeypatch.setattr(instance, "request", forbidden)
    receipt = {"ready_for_provider_stop": True, "git_commit": "wrong", "remote_main": "wrong"}
    (tmp_path / "final-preservation.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="differs"):
        instance.stop()


def test_remaining_gpu_process_never_reaches_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTAINER_ID", str(instance.INSTANCE))
    monkeypatch.setattr(instance, "RUN", tmp_path)
    monkeypatch.setattr(instance, "request", forbidden)
    head = "f" * 40
    receipt = {"ready_for_provider_stop": True, "git_commit": head, "remote_main": head}
    (tmp_path / "final-preservation.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(
        instance.subprocess,
        "check_output",
        lambda command, **kwargs: head if command[0] == "git" else "99999",
    )
    with pytest.raises(ValueError, match="GPU compute"):
        instance.stop()
