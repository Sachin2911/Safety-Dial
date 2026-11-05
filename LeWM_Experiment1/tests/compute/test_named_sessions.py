"""Offline checks that named rentals never replace the original default state."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "LeWM_Experiment1/computeConfigs"
SPEC = importlib.util.spec_from_file_location("compute_named_test", DIRECTORY / "compute.py")
compute = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compute)


@pytest.fixture
def state_root(tmp_path, monkeypatch):
    path = tmp_path / "state"
    monkeypatch.setattr(compute, "STATE_DIR", path)
    return path


def test_default_and_named_parser_routes_are_explicit(state_root):
    default = compute.parser().parse_args(["status"])
    named = compute.parser().parse_args(["--session", "ap-cube", "status"])
    assert default.session is None
    assert compute.session_state_dir(default.session) == state_root
    assert compute.session_state_dir(named.session) == state_root / "named/ap-cube"
    assert not state_root.exists()


@pytest.mark.parametrize("name", ["", ".", "..", "../default", "/tmp/other", "a/b", "a\\b", "x" * 97])
def test_unsafe_names_are_rejected_without_creating_state(state_root, name):
    with pytest.raises(compute.ComputeError):
        compute.session_state_dir(name)
    assert not state_root.exists()


@pytest.mark.parametrize("alias_parent", [True, False])
def test_named_state_cannot_alias_existing_state(state_root, tmp_path, alias_parent):
    state_root.mkdir()
    if alias_parent:
        (state_root / "named").symlink_to(tmp_path, target_is_directory=True)
    else:
        (state_root / "named").mkdir()
        (state_root / "named/ap-cube").symlink_to(state_root, target_is_directory=True)
    with pytest.raises(compute.ComputeError, match="alias"):
        compute.session_state_dir("ap-cube")


def test_two_fake_rentals_keep_identity_locks_host_keys_and_default_bytes_separate(state_root, monkeypatch):
    class Provider:
        def __init__(self):
            self.identities = []

        def call(self, path, **kwargs):
            assert path == "/asks/123/"
            identity = 100 + len(self.identities)
            self.identities.append(identity)
            return {"success": True, "new_contract": identity}

    provider = Provider()
    controllers = [compute.Controller(directory=DIRECTORY,
                                       state_dir=compute.session_state_dir(name),
                                       secrets={"GHCR_READ_TOKEN": "fake-image-token",
                                                "HF_TOKEN": "fake-hub-token"}, api=provider)
                   for name in (None, "ap-cube")]
    args = SimpleNamespace(storage_profile="ap", hours=2, max_hourly=0.6,
                           download_gb=60, upload_gb=2, budget_usd=1.5)
    for controller in controllers:
        monkeypatch.setattr(controller, "credentials_preflight", lambda: None)
        monkeypatch.setattr(controller, "search", lambda args: [{"offer_id": 123, "disk_gb": 200}])
        monkeypatch.setattr(controller, "prepare", lambda state: state)
    first = controllers[0].up(args)
    original_bytes = controllers[0].state_path.read_bytes()
    second = controllers[1].up(args)
    assert controllers[0].state_path.read_bytes() == original_bytes
    assert controllers[0].state()["instance_id"] == first["instance_id"] == 100
    assert controllers[1].state()["instance_id"] == second["instance_id"] == 101
    assert first["session_id"] != second["session_id"]
    assert first["storage"] == second["storage"]
    endpoint = {"host": "example.invalid", "port": 22}
    host_keys = [next(arg for arg in controller.ssh_arguments(state, endpoint)
                      if arg.startswith("UserKnownHostsFile="))
                 for controller, state in zip(controllers, (first, second))]
    assert host_keys[0] != host_keys[1]
    assert str(state_root / "named/ap-cube") in host_keys[1]
    assert controllers[0].state_dir / "controller.lock" != controllers[1].state_dir / "controller.lock"


@pytest.mark.parametrize("name", [None, "ap-cube"])
def test_main_constructs_controller_with_selected_state(state_root, monkeypatch, capsys, name):
    observed = []

    def constructor(*, state_dir):
        observed.append(state_dir)
        return SimpleNamespace(status=lambda: {"fake": True})

    monkeypatch.setattr(compute, "Controller", constructor)
    argv = ["compute.py", *(["--session", name] if name else []), "status"]
    monkeypatch.setattr(sys, "argv", argv)
    compute.main()
    assert observed == [compute.session_state_dir(name)]
    assert '"fake": true' in capsys.readouterr().out
