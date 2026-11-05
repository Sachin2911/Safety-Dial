"""Offline orchestration checks for per-session baseline/AP output isolation."""

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "LeWM_Experiment1/computeConfigs"
SPEC = importlib.util.spec_from_file_location("compute_storage_test", DIRECTORY / "compute.py")
compute = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compute)
common = importlib.import_module("compute_common")
worker = importlib.import_module("remote_worker")
store_module = importlib.import_module("hf_store")


class FakeVast:
    def __init__(self):
        self.calls = []

    def call(self, path, **kwargs):
        self.calls.append((path, kwargs))
        assert path == "/asks/123/"
        return {"success": True, "new_contract": 456}


@pytest.fixture
def controller(tmp_path):
    directory = tmp_path / "compute"
    shutil.copytree(DIRECTORY / "config", directory / "config")
    return compute.Controller(
        directory=directory,
        state_dir=tmp_path / "state",
        secrets={"GHCR_READ_TOKEN": "test-image-credential", "HF_TOKEN": "test-hub-credential"},
        api=FakeVast(),
    )


def up_args(profile):
    return SimpleNamespace(
        storage_profile=profile, hours=1, max_hourly=1, download_gb=1,
        upload_gb=1, budget_usd=2,
    )


@pytest.mark.parametrize("profile", ["baseline", "ap"])
def test_new_session_pins_destination_and_archives_retired_state(controller, monkeypatch, profile):
    previous = {
        "session_id": "retired-baseline", "instance_id": 42,
        "phase": "destroyed", "runs": ["original-run"],
    }
    controller.save(previous)
    monkeypatch.setattr(controller, "credentials_preflight", lambda: None)
    monkeypatch.setattr(controller, "search", lambda args: [{"offer_id": 123, "disk_gb": 200}])
    monkeypatch.setattr(controller, "prepare", lambda state: state)
    state = controller.up(up_args(profile))
    assert state["storage"]["repo_id"] == common.STORAGE_REPOSITORIES[profile]
    assert state["instance_id"] == 456
    assert len(controller.api.calls) == 1
    archive = controller.state_dir / "sessions/retired-baseline.json"
    assert json.loads(archive.read_text()) == previous
    assert controller.state()["storage"] == state["storage"]


def test_current_session_keeps_ap_destination_after_profile_file_changes(controller):
    storage = controller.select_storage("ap")
    controller.save({"phase": "ready", "session_id": "active-ap", "storage": storage})
    path = controller.directory / "config/storage-ap.json"
    path.write_text(json.dumps(controller.select_storage("baseline")))
    controller.state()
    assert controller.storage == storage
    with pytest.raises(common.ComputeError, match="different dataset"):
        controller.select_storage("ap")


def test_legacy_session_keeps_baseline_destination(controller):
    controller.save({"phase": "destroyed", "session_id": "legacy", "instance_id": 42})
    controller.storage = controller.select_storage("ap")
    controller.state()
    assert controller.storage["repo_id"] == common.STORAGE_REPOSITORIES["baseline"]


def test_live_session_blocks_new_rental_before_provider_call(controller):
    controller.save({"phase": "boot_failed", "session_id": "active"})
    with pytest.raises(common.ComputeError, match="already recorded"):
        controller.up(up_args("ap"))
    assert controller.api.calls == []


def test_worker_rejects_run_for_another_dataset(controller, monkeypatch):
    session = {"session_id": "ap-session", "storage": controller.select_storage("ap")}
    monkeypatch.setattr(worker, "read_json", lambda path: session)
    request = {
        "root": "/workspace/LeWM_Experiment1", "session_id": "ap-session", "action": "launch",
        "spec": {"storage": controller.select_storage("baseline")},
    }
    with pytest.raises(common.ComputeError, match="pinned destination"):
        worker.dispatch(request)


def test_receipt_verification_uses_the_session_destination(controller, monkeypatch):
    storage = controller.select_storage("ap")
    controller.save({"phase": "ready", "session_id": "ap-session", "storage": storage})
    controller.state()
    calls = []
    revision = "a" * 40
    manifest = b'{"run_id":"ap-run"}'

    class FakeHub:
        def __init__(self, base, token):
            self.token = token

        def call(self, path):
            calls.append(path)
            return {"private": True, "sha": revision}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            return manifest

    class Opener:
        def open(self, request, timeout):
            calls.append(request.full_url)
            return Response()

    monkeypatch.setattr(compute, "JsonApi", FakeHub)
    monkeypatch.setattr(compute, "build_opener", lambda *args: Opener())
    controller.verify_receipt({
        "run_id": "ap-run", "ready_to_destroy": True,
        "upload": {"revision": revision, "manifest_sha256": hashlib.sha256(manifest).hexdigest()},
    })
    assert all(common.STORAGE_REPOSITORIES["ap"] in call for call in calls)
    assert len(calls) == 2


@pytest.mark.parametrize("profile", ["baseline", "ap"])
def test_uploader_accepts_declared_destination(controller, monkeypatch, profile):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda **kw: object()))
    store = store_module.HubStore(controller.select_storage(profile), "test-hub-credential")
    assert store.repo == common.STORAGE_REPOSITORIES[profile]


@pytest.mark.parametrize("change", [
    {"repo_id": "other/dataset"}, {"private": False}, {"path_prefix": "elsewhere"},
    {"repo_type": "model"}, {"remote_runs_root": "/tmp/runs"}, {"branch": "other"},
])
def test_uploader_refuses_changed_destination_contract(controller, monkeypatch, change):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda **kw: object()))
    storage = {**controller.select_storage("ap"), **change}
    with pytest.raises(common.ComputeError, match="private storage"):
        store_module.HubStore(storage, "test-hub-credential")
