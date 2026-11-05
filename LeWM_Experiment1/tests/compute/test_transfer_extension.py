"""Fake-provider checks for scoped, explicitly requested transfer-cap increases."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import time

import pytest


ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "LeWM_Experiment1/computeConfigs"
SPEC = importlib.util.spec_from_file_location("compute_transfer_test", DIRECTORY / "compute.py")
compute = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compute)


class Provider:
    def __init__(self):
        self.rows = [{"id": 42, "label": "exact-session-label", "actual_status": "running"}]
        self.calls = []

    def call(self, path, **kwargs):
        self.calls.append((path, kwargs))
        assert path == "/instances/" and not kwargs
        return {"instances": self.rows}


@pytest.fixture
def controller(tmp_path, monkeypatch):
    directory = tmp_path / "config"
    shutil.copytree(DIRECTORY / "config", directory / "config")
    value = compute.Controller(directory=directory, state_dir=tmp_path / "state",
                               secrets={}, api=Provider())
    value.save({"phase": "ready", "session_id": "same-session", "instance_id": 42,
                "label": "exact-session-label", "upload_gb": 5.0, "download_gb": 200.0,
                "budget_usd": 10.0, "hours": 6.0, "deadline_epoch": time.time() + 10000,
                "storage": value.select_storage("ap"), "runs": ["original-run"],
                "remote_tools": "/workspace/LeWM_Experiment1/.control/tools/unchanged",
                "readiness": {"original": True},
                "offer": {"estimated_session_usd": 3.727274,
                          "upload_usd_per_gb": 0.005208333333333333,
                          "download_usd_per_gb": 0.0013020833333333333}})
    value.fake_rows = [{"run_id": "original-run", "job_alive": False, "supervisor_alive": False,
                        "ready_to_destroy": True, "upload": {"verified": True, "bytes": 4054102586,
                        "revision": "a" * 40, "manifest_sha256": "b" * 64}}]

    def worker(state, action, *, provider_instance):
        assert state["session_id"] == "same-session"
        assert action == "status" and provider_instance["id"] == 42
        return {"runs": deepcopy(value.fake_rows)}

    monkeypatch.setattr(value, "worker", worker)
    return value


def test_extension_preserves_all_other_limits_and_original_quote(controller):
    before = controller.state()
    reply = controller.extend_transfer(8)
    after = controller.state()
    assert after["upload_gb"] == 8
    assert after["estimated_session_usd"] == pytest.approx(3.746805)
    changed = {key for key in after if after[key] != before.get(key)}
    assert changed == {"upload_gb", "estimated_session_usd", "transfer_extensions"}
    assert reply["additional_outgoing_usd"] == pytest.approx(0.015625)
    assert reply["additional_readback_usd"] == pytest.approx(0.00390625)
    assert reply["remote_mutation"] is False
    receipt = json.loads(Path(reply["receipt"]).read_text())
    assert receipt["previous_state"] == before and receipt["phase"] == "applied"
    assert receipt["verified_runs"][0]["upload"]["bytes"] == 4054102586
    assert controller.api.calls == [("/instances/", {})]


def test_repeated_value_is_idempotent_and_later_extension_uses_same_original_quote(controller):
    controller.extend_transfer(8)
    before = controller.state()
    assert controller.extend_transfer(8)["changed"] is False
    assert controller.state() == before
    controller.extend_transfer(10)
    expected = 3.727274 + 5 * (0.005208333333333333 + 0.0013020833333333333)
    assert controller.state()["estimated_session_usd"] == round(expected, 6)
    assert len(controller.state()["transfer_extensions"]) == 2


@pytest.mark.parametrize("problem", ["budget", "decrease", "pending", "deadline", "absent",
                                     "label", "active", "unverified", "missing", "unpinned"])
def test_extension_fails_closed_without_state_mutation(controller, problem):
    state = controller.state()
    requested = 8
    if problem == "budget":
        state["budget_usd"] = 3.73
    elif problem == "decrease":
        requested = 4
    elif problem == "pending":
        state["pending_worker_refresh"] = {"pending": True}
    elif problem == "deadline":
        state["deadline_epoch"] = 1
    elif problem == "absent":
        controller.api.rows = []
    elif problem == "label":
        controller.api.rows[0]["label"] = "another-session"
    elif problem == "active":
        controller.fake_rows[0]["job_alive"] = True
    elif problem == "unverified":
        controller.fake_rows[0]["upload"]["verified"] = False
    elif problem == "missing":
        controller.fake_rows = []
    else:
        controller.fake_rows[0]["upload"].pop("manifest_sha256")
    controller.save(state)
    with pytest.raises(compute.ComputeError):
        controller.extend_transfer(requested)
    assert controller.state() == state
    assert not (controller.state_dir / "transfer-extensions").exists()


def test_reallocation_reduces_future_cap_without_changing_original_limits(controller):
    controller.extend_transfer(8)
    before = controller.state()
    result = controller.set_transfer_limit(6)
    after = controller.state()
    assert after["upload_gb"] == 6
    assert after["estimated_session_usd"] == 3.733784
    assert result["operation"] == "set-transfer-limit"
    assert result["additional_outgoing_usd"] < 0 and result["additional_readback_usd"] < 0
    for key in ("budget_usd", "hours", "deadline_epoch", "download_gb", "offer", "runs",
                "storage", "remote_tools", "readiness", "instance_id", "session_id"):
        assert after[key] == before[key]
    assert after["transfer_extensions"][:-1] == before["transfer_extensions"]
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["previous_state"] == before and receipt["phase"] == "applied"
    with pytest.raises(compute.ComputeError, match="cannot reduce"):
        controller.extend_transfer(5)
    assert controller.state() == after


@pytest.mark.parametrize("problem", ["equal_usage", "below_usage", "active", "unverified"])
def test_reallocation_rejects_unsafe_decreases_without_touching_state(controller, problem):
    controller.extend_transfer(8)
    before = controller.state()
    existing = sorted((controller.state_dir / "transfer-extensions").glob("*.json"))
    if problem == "equal_usage":
        controller.fake_rows[0]["upload"]["bytes"] = 6000000000
    elif problem == "below_usage":
        controller.fake_rows[0]["upload"]["bytes"] = 6000000001
    elif problem == "active":
        controller.fake_rows[0]["job_alive"] = True
    else:
        controller.fake_rows[0]["upload"]["verified"] = False
    with pytest.raises(compute.ComputeError):
        controller.set_transfer_limit(6)
    assert controller.state() == before
    assert sorted((controller.state_dir / "transfer-extensions").glob("*.json")) == existing


def test_named_reallocation_requires_explicit_cli_value():
    args = compute.parser().parse_args(["--session", "ap-cube", "set-transfer-limit", "--upload-gb", "2"])
    assert args.session == "ap-cube" and args.action == "set-transfer-limit" and args.upload_gb == 2
