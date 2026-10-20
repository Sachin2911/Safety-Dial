"""Coordinator recovery/preflight cases with all physical/provider calls replaced."""

import importlib.util
import json
import time
from pathlib import Path

import pytest
from test_maniguard_continuation_admission import plan, write

from helpers.maniguard_continuation_admission import ScreenLedger

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/maniguard_continuation_screen.py"


@pytest.fixture
def screen():
    spec = importlib.util.spec_from_file_location("screen_coordinator_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preparation_failure_charges_reserved_pair_and_stops_without_spawning(tmp_path, monkeypatch, screen):
    protocol = plan(tmp_path)
    protocol["closure_controller_file"] = "synthetic-unused"
    p = tmp_path / "protocol.json"
    write(p, protocol)
    monkeypatch.setattr(screen, "require_original_clearance", lambda value: {"synthetic": True})
    monkeypatch.setattr(screen, "read_resources", lambda value, run: time.time() + 20000)
    monkeypatch.setattr(screen, "policy_lease_path", lambda value, worker: tmp_path / f"lease-{worker}")
    monkeypatch.setattr(screen, "prepare_job", lambda *args: (_ for _ in ()).throw(ValueError("synthetic preflight")))
    spawned = []
    monkeypatch.setattr(screen.subprocess, "Popen", lambda *args, **kwargs: spawned.append(args))
    # Use a distant future cutoff; the test never accesses an actual provider or scene.
    protocol["physical_cutoff_utc"] = "2099-01-01T00:00:00Z"
    write(p, protocol)
    with pytest.raises(RuntimeError, match="charged invalid start"):
        screen.run_screen(protocol["run_directory"], p)
    state = json.loads((Path(protocol["run_directory"]) / "screen.json").read_text())
    assert len(state["attempts"]) == 2 and all(a["status"] == "invalid" for a in state["attempts"])
    assert not spawned
    with pytest.raises(ValueError, match="no automatic replacement"):
        screen.run_screen(protocol["run_directory"], p)
    assert len(json.loads((Path(protocol["run_directory"]) / "screen.json").read_text())["attempts"]) == 2


def test_reconcile_observes_existing_identity_and_never_relaunches(tmp_path, monkeypatch, screen):
    p = tmp_path / "protocol.json"
    protocol = plan(tmp_path)
    write(p, protocol)
    ledger = ScreenLedger(protocol["run_directory"], p)
    output = tmp_path / "output"
    output.mkdir()
    identity = {"pid": 123, "start_ticks": "synthetic"}
    write(output / "guardian.json", {"guardian": identity})
    write(output / "guardian-terminal.json", {"exit_code": 0, "technical_stop_reason": None,
                                             "finished_epoch": 2})
    attempt = {"protocol_episode_id": "native_only", "worker": 0, "status": "running",
               "output": str(output), "guardian_identity": identity, "deadline_epoch": time.time() + 20}
    ledger.state["attempts"] = [attempt]
    calls = []

    def current(pid):
        calls.append(pid)
        return identity if len(calls) == 1 else None

    monkeypatch.setattr(screen, "process_identity", current)
    monkeypatch.setattr(screen.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(screen, "policy_lease_path", lambda value, worker: tmp_path / "lease")
    monkeypatch.setattr(screen, "read_episode", lambda value: {"score": 0})
    monkeypatch.setattr(screen, "capture_service_records", lambda value: [])
    verified = []
    monkeypatch.setattr(screen, "record_verification", lambda path, receipt: verified.append(receipt))
    monkeypatch.setattr(screen.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("No relaunch"))
    try:
        screen.reconcile(ledger, [attempt])
        assert len(calls) == 2 and verified == [{"score": 0}]
        assert attempt["status"] == "completed"
        assert len(ledger.state["attempts"]) == 1
    finally:
        ledger.close()


def test_cached_screen_results_are_compared_not_silently_overwritten(tmp_path, monkeypatch, screen):
    monkeypatch.setattr(screen, "attribute_progress", lambda path, receipt: {"synthetic": True})
    first = {"score": 0}
    screen.record_verification(tmp_path, first)
    screen.record_verification(tmp_path, first)
    with pytest.raises(ValueError, match="existing screen result differs"):
        screen.record_verification(tmp_path, {"score": -2})
    assert json.loads((tmp_path / "verified-receipt.json").read_text()) == first


def test_failed_clearance_is_read_only_and_does_not_make_a_screen_directory(tmp_path, monkeypatch, screen, capsys):
    p = tmp_path / "draft.json"
    write(p, {"status": "cpu_preparation_not_authorized_for_launch"})
    run = tmp_path / "never-created"
    monkeypatch.setattr(screen.sys, "argv", [str(SCRIPT), "--protocol", str(p), "--run", str(run), "--check"])
    screen.main()
    result = json.loads(capsys.readouterr().out)
    assert not result["original_clearance_ready"] and not result["physical_dispatch"]
    assert result["reserved_starts"] == 0 and not run.exists()
