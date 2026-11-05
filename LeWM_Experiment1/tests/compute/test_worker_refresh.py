"""Offline checks for bounded read-back and idle-only versioned worker refresh."""

import base64
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "LeWM_Experiment1/computeConfigs"
SPEC = importlib.util.spec_from_file_location("compute_refresh_test", DIRECTORY / "compute.py")
compute = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compute)
common = importlib.import_module("compute_common")
hf_store = importlib.import_module("hf_store")


def make_store(monkeypatch, get):
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(get=get))

    def hub_url(repo, path, **kwargs):
        assert kwargs == {"repo_type": "dataset", "revision": "a" * 40}
        return "https://example.invalid/" + path

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_url=hub_url))
    store = object.__new__(hf_store.HubStore)
    store.repo, store.token = common.STORAGE_REPOSITORIES["ap"], "fake-unit-test-token"
    return store


def test_full_fresh_downloads_are_bounded_at_eight_and_all_responses_close(monkeypatch):
    barrier = threading.Barrier(8, timeout=3)
    lock = threading.Lock()
    active, maximum = 0, 0
    calls, closed = [], []
    files = {f"file-{index}": {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
             for index in range(16)}

    class Response:
        def __init__(self, url):
            self.url = url

        def __enter__(self):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            return self

        def __exit__(self, *args):
            nonlocal active
            with lock:
                active -= 1
                closed.append(self.url)

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            assert chunk_size == 1024 * 1024
            barrier.wait()
            yield b"a"
            yield b"bc"

    def get(url, **kwargs):
        assert kwargs["headers"] == {"Authorization": "Bearer fake-unit-test-token",
                                     "Cache-Control": "no-cache", "Accept-Encoding": "identity"}
        assert kwargs["stream"] is True and kwargs["timeout"] == (15, 60)
        with lock:
            calls.append(url)
        return Response(url)

    make_store(monkeypatch, get).verify("runs/finished", files, "a" * 40)
    assert maximum == 8 and active == 0
    assert sorted(calls) == sorted(closed)
    assert len(calls) == len(set(calls)) == 16


@pytest.mark.parametrize("body, failure", [
    (b"xyz", None), (b"a", None), (b"abcd", None),
    (b"abc", "http"), (b"abc", "stream"),
])
def test_readback_errors_propagate_and_close_the_response(monkeypatch, body, failure):
    closed = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

        def raise_for_status(self):
            if failure == "http":
                raise RuntimeError("synthetic HTTP failure")

        def iter_content(self, chunk_size):
            if failure == "stream":
                raise RuntimeError("synthetic stream failure")
            yield body

    store = make_store(monkeypatch, lambda *args, **kwargs: Response())
    with pytest.raises((common.ComputeError, RuntimeError)):
        store.verify("runs/finished", {"x": {"bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}}, "a" * 40)
    assert closed == [True]


def test_failed_verification_never_marks_final_backup_verified(tmp_path):
    root = tmp_path / "run"
    (root / ".control").mkdir(parents=True)
    storage = json.loads((DIRECTORY / "config/storage-ap.json").read_text())
    common.atomic_json(root / "run.json", {"run_id": "run", "storage": storage, "max_artifact_bytes": 100000})
    common.atomic_json(root / "result.json", {"status": "completed", "exit_code": 0})

    class Store:
        def __init__(self, storage, token):
            pass

        def commit(self, prefix, files, directory):
            return "a" * 40

        def verify(self, prefix, files, revision):
            raise common.ComputeError("synthetic read-back mismatch")

    with pytest.raises(common.ComputeError, match="read-back mismatch"):
        hf_store.sync_run(root, "fake-unit-test-token", final=True, store_factory=Store)
    journal = common.read_json(root / ".control/upload.json")
    assert journal["verified"] is False
    assert "verified_epoch" not in journal
    assert (root / "manifest.json").is_file()


@pytest.mark.parametrize("filename", ["storage.json", "storage-ap.json"])
def test_current_storage_profiles_still_reject_public_destinations(monkeypatch, filename):
    api = SimpleNamespace(repo_info=lambda *args, **kwargs: SimpleNamespace(private=False, sha="a" * 40))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda **kwargs: api))
    storage = json.loads((DIRECTORY / "config" / filename).read_text())
    store = hf_store.HubStore(storage, "fake-unit-test-token")
    with pytest.raises(common.ComputeError, match="public"):
        store.head()


@pytest.fixture
def controller(tmp_path):
    directory = tmp_path / "config-copy"
    shutil.copytree(DIRECTORY / "config", directory / "config")
    value = compute.Controller(directory=directory, state_dir=tmp_path / "state",
                               secrets={}, api=object())
    value.save({"phase": "ready", "session_id": "session", "instance_id": 42,
                "storage": value.select_storage("ap"), "runs": ["done"],
                "remote_tools": "/workspace/LeWM_Experiment1/.control/tools/old",
                "readiness": {"original": "preserved"}})
    return value


def finished_rows():
    return [{"run_id": "done", "job_alive": False, "supervisor_alive": False,
             "ready_to_destroy": True, "upload": {"verified": True,
              "revision": "a" * 40, "manifest_sha256": "b" * 64}}]


def mock_refresh_transport(controller, monkeypatch, *, lose_reply=False):
    calls = []
    monkeypatch.setattr(controller, "worker", lambda *args, **kwargs: {"runs": finished_rows()})
    monkeypatch.setattr(controller, "connected", lambda state: {"host": "fake", "port": 22})

    def ssh(state, endpoint, command, payload, **kwargs):
        calls.append(deepcopy(payload))
        assert command == [controller.policy["python"], "-c", compute.REFRESH_WORKERS]
        assert "pending_worker_refresh" not in payload["state"]
        if lose_reply and len(calls) == 1:
            raise compute.SSHUnavailable("synthetic lost reply")
        return json.dumps({"refreshed": True, "refresh": payload["refresh"],
                           "runs_verified": 1, "remote_receipt": "fake-receipt"})

    monkeypatch.setattr(controller, "ssh_call", ssh)
    return calls


def test_refresh_changes_only_version_metadata_and_preserves_previous_state(controller, monkeypatch):
    previous = controller.state()
    calls = mock_refresh_transport(controller, monkeypatch)
    reply = controller.refresh_workers()
    current = controller.state()
    assert reply["refreshed"] is True and len(calls) == 1
    for key in ("phase", "session_id", "instance_id", "storage", "runs", "readiness"):
        assert current[key] == previous[key]
    assert current["remote_tools"] != previous["remote_tools"]
    receipt = common.read_json(next((controller.state_dir / "worker-refreshes").glob("*.json")))
    assert receipt["previous_state"] == previous
    assert receipt["confirmation"] == reply
    assert len(current["worker_refresh_history"]) == 1
    assert controller.refresh_workers()["refreshed"] is False
    assert len(calls) == 1


def test_lost_refresh_reply_blocks_launch_and_retry_reuses_identity(controller, monkeypatch):
    calls = mock_refresh_transport(controller, monkeypatch, lose_reply=True)
    with pytest.raises(compute.SSHUnavailable):
        controller.refresh_workers()
    pending = controller.state()["pending_worker_refresh"]
    with pytest.raises(common.ComputeError, match="pending worker refresh"):
        controller.run(None)
    controller.refresh_workers()
    assert calls[0]["state"] == calls[1]["state"]
    assert calls[1]["refresh"] == pending
    assert "pending_worker_refresh" not in controller.state()


@pytest.mark.parametrize("problem", ["active", "unverified", "missing"])
def test_refresh_refuses_nonfinal_runs_before_deployment(controller, monkeypatch, problem):
    previous = controller.state()
    rows = finished_rows()
    if problem == "active":
        rows[0]["job_alive"] = True
    elif problem == "unverified":
        rows[0]["upload"]["verified"] = False
    else:
        rows = []
    monkeypatch.setattr(controller, "worker", lambda *args, **kwargs: {"runs": rows})
    with pytest.raises(common.ComputeError):
        controller.refresh_workers()
    assert controller.state() == previous
    assert not (controller.state_dir / "worker-refreshes").exists()


def remote_fixture(tmp_path):
    root = tmp_path / "remote"
    control = root / ".control"
    old = control / "tools/old"
    old.mkdir(parents=True)
    shutil.copyfile(DIRECTORY / "scripts/compute_common.py", old / "compute_common.py")
    (old / "remote_worker.py").write_text(
        "import json\ndef all_status(root):\n return json.loads((root/'.control/test-status.json').read_text())\n")
    state = {"session_id": "session", "instance_id": 42, "storage": {"pinned": "same"},
             "remote_tools": str(old), "runs": ["done"], "readiness": {"unchanged": True}}
    common.atomic_json(control / "session.json", state)
    common.atomic_json(control / "test-status.json", finished_rows())
    files = {name: base64.b64encode(b"VALUE = 1\n").decode() for name in compute.WORKER_FILES}
    target = str(control / "tools" / common.fingerprint(files))
    refresh = {"id": "test-refresh", "previous_remote_tools": str(old), "remote_tools": target}
    candidate = dict(state, remote_tools=target, worker_refresh_history=[refresh])
    payload = {"state": candidate, "files": files, "refresh": refresh}
    script = compute.REFRESH_WORKERS.replace("/workspace/LeWM_Experiment1", str(root))
    return root, state, payload, script


def execute_refresh(script, payload):
    return subprocess.run([sys.executable, "-c", script], input=json.dumps(payload),
                          text=True, capture_output=True, timeout=10)


def test_remote_refresh_is_idempotent_and_leaves_original_tools_unchanged(tmp_path):
    root, previous, payload, script = remote_fixture(tmp_path)
    old = Path(previous["remote_tools"])
    original = {path.name: path.read_bytes() for path in old.glob("*.py")}
    first = execute_refresh(script, payload)
    assert first.returncode == 0, first.stderr
    assert execute_refresh(script, payload).returncode == 0
    assert {path.name: path.read_bytes() for path in old.glob("*.py")} == original
    assert common.read_json(root / ".control/session.json") == payload["state"]
    receipt = common.read_json(root / ".control/worker-refreshes/test-refresh.json")
    assert receipt["previous_session"] == previous
    assert len(list((root / ".control/tools").iterdir())) == 2


@pytest.mark.parametrize("problem", ["active", "storage", "unverified", "missing", "syntax"])
def test_remote_guard_refuses_races_identity_changes_and_invalid_code(tmp_path, problem):
    root, previous, payload, script = remote_fixture(tmp_path)
    rows = finished_rows()
    if problem == "active":
        rows[0]["supervisor_alive"] = True
    elif problem == "storage":
        payload["state"]["storage"] = {"pinned": "different"}
    elif problem == "unverified":
        rows[0]["upload"]["verified"] = False
    elif problem == "missing":
        rows = []
    else:
        payload["files"]["hf_store.py"] = base64.b64encode(b"bad syntax !").decode()
        target = str(root / ".control/tools" / common.fingerprint(payload["files"]))
        payload["state"]["remote_tools"] = payload["refresh"]["remote_tools"] = target
    common.atomic_json(root / ".control/test-status.json", rows)
    assert execute_refresh(script, payload).returncode != 0
    assert common.read_json(root / ".control/session.json") == previous
    assert not Path(payload["state"]["remote_tools"]).exists()
