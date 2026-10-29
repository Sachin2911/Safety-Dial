"""Billing resumes and cleanup retries must remain safe without GPU execution."""

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe09_infra import FinancialBook


def test_resumed_segment_keeps_prior_charges_and_closes_the_new_segment(tmp_path):
    book = FinancialBook(tmp_path / "finance.json")
    offer = {
        "dph_total": 0.4,
        "driver_version": "580.178.04",
        "gpu_name": "RTX 3090",
        "num_gpus": 1,
        "cpu_ram": 128000,
        "cpu_cores_effective": 32,
        "disk_space": 200,
        "inet_down_cost": 0.005,
        "inet_up_cost": 0.005,
    }
    book.reserve("initial", "2026-10-25T10:00:00+00:00", offer)
    book.attach("initial", 99)
    book.stopped(99, "2026-10-25T10:30:00+00:00", status="exited")
    first = dict(book.data["allocations"][0])
    book.reserve_resume(99, "resume-1", "2026-10-25T12:00:00+00:00", "2026-10-25T22:00:00+00:00")
    with pytest.raises(ValueError, match="active or uncertain"):
        book.reserve_resume(
            99, "resume-2", "2026-10-25T12:01:00+00:00", "2026-10-25T22:00:00+00:00"
        )
    book.stopped(99, "2026-10-25T13:00:00+00:00", status="exited")
    assert book.data["allocations"][0] == first
    assert book.data["allocations"][1]["active_stop_verified_utc"] == "2026-10-25T13:00:00+00:00"
    assert book.data["allocations"][1]["transfer_estimate_usd"] == 0
    snapshot = book.snapshot(dt.datetime.fromisoformat("2026-10-25T13:00:00+00:00").timestamp())
    assert snapshot["active_estimate_usd"] == pytest.approx(0.6)
    assert snapshot["aggregate_hours"] == pytest.approx(1.5)
    assert snapshot["aggregate_transfer_estimate_usd"] == pytest.approx(0.95)
    assert snapshot["active_instances"] == []
    with pytest.raises(ValueError, match="extend"):
        book.reserve_resume(
            99, "too-late", "2026-10-26T10:00:00+00:00", "2026-10-26T22:00:00+00:00"
        )
    with pytest.raises(ValueError, match="verified stopped"):
        book.reserve_resume(
            101, "wrong-host", "2026-10-25T13:30:00+00:00", "2026-10-25T22:00:00+00:00"
        )


def load_guard(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/guard_unisafe09_approval.py"
    spec = importlib.util.spec_from_file_location("approval_guard_test", path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    rental = tmp_path / "progress/rental-unisafe09-20261025.json"
    rental.parent.mkdir()
    rental.write_text(
        json.dumps(
            {
                "instance_id": 99,
                "approval_pending": True,
                "allocation_deadline_utc": "2026-10-25T00:00:00+00:00",
            }
        )
    )
    backup = tmp_path / "outputs/unisafe09/control/approval-setup-backup.json"
    backup.parent.mkdir(parents=True)
    backup.write_text(json.dumps({"instance_id": 99, "no_simulator_attempts": True}))
    monkeypatch.setattr(
        guard.time,
        "time",
        lambda: dt.datetime.fromisoformat("2026-10-25T01:00:00+00:00").timestamp(),
    )
    return guard


def test_cleanup_rechecks_after_network_error_and_uncertain_destruction(tmp_path, monkeypatch):
    guard = load_guard(tmp_path, monkeypatch)
    provider_reads = []
    sleeps = []
    mutations = []

    def current():
        provider_reads.append(True)
        if len(provider_reads) == 1:
            raise RuntimeError("temporary provider error")
        if len(provider_reads) == 2:
            return [
                {
                    "id": 99,
                    "actual_status": "exited",
                    "cur_state": "stopped",
                    "intended_status": "stopped",
                }
            ]
        return []

    def run(*args, **kwargs):
        mutations.append(args)
        return SimpleNamespace(returncode=1 if len(mutations) == 1 else 0)

    monkeypatch.setattr(guard, "instances", current)
    monkeypatch.setattr(guard.subprocess, "run", run)
    monkeypatch.setattr(guard.time, "sleep", sleeps.append)
    guard.main()
    assert len(provider_reads) == 3 and len(mutations) == 2
    assert sleeps == [30, 30]
    state = json.loads(
        (tmp_path / "outputs/unisafe09/control/approval-guard-status.json").read_text()
    )
    assert state["status"] == "unused_setup_destroyed"


def test_cleanup_refuses_to_destroy_an_externally_resumed_host(tmp_path, monkeypatch):
    guard = load_guard(tmp_path, monkeypatch)
    monkeypatch.setattr(
        guard,
        "instances",
        lambda: [
            {
                "id": 99,
                "actual_status": "running",
                "cur_state": "running",
                "intended_status": "running",
            }
        ],
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Must not destroy externally resumed work")

    monkeypatch.setattr(guard.subprocess, "run", forbidden)
    guard.main()
    state = json.loads(
        (tmp_path / "outputs/unisafe09/control/approval-guard-status.json").read_text()
    )
    assert state["status"] == "external_resume_detected_no_destruction"
