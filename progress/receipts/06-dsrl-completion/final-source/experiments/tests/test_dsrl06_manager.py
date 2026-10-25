"""Rental acknowledgements cannot cause a duplicate purchase or premature disk deletion."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from helpers.dsrl06_infra import FinancialBook


@pytest.fixture
def manager(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/manage_dsrl06.py"
    spec = importlib.util.spec_from_file_location("dsrl06_manager_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "BOOK", tmp_path / "finance.json")
    monkeypatch.setattr(module, "RENTAL", tmp_path / "rental.json")
    monkeypatch.setattr(module, "verify_freeze", lambda _: None)
    monkeypatch.setattr(module, "start_local_watchdog", lambda: 12345)
    monkeypatch.setattr(
        module,
        "offers",
        lambda: [
            {
                "id": 99,
                "dph_total": 1.9,
                "driver_version": "580.173.02",
                "gpu_name": "RTX 4090",
                "num_gpus": 4,
                "cpu_ram": 257000,
                "cpu_cores_effective": 64,
                "inet_down_cost": 0.003,
                "inet_up_cost": 0.004,
            }
        ],
    )
    return module


def test_uncertain_create_reconciles_label_without_second_purchase(manager, monkeypatch):
    calls = []

    def ambiguous(args, **kwargs):
        calls.append(args)
        raise RuntimeError("acknowledgement lost")

    def current():
        rental = json.loads(manager.RENTAL.read_text())
        return [{"id": 42, "label": rental["label"], "actual_status": "running"}]

    monkeypatch.setattr(manager, "vast", ambiguous)
    monkeypatch.setattr(manager, "instances", current)
    manager.create(argparse.Namespace(plan="unused", offer=99))
    assert len(calls) == 1
    assert json.loads(manager.RENTAL.read_text())["instance_id"] == 42
    assert FinancialBook(manager.BOOK).snapshot()["active_instances"] == [42]


def test_rejected_create_and_verified_absence_release_reservation(manager, monkeypatch):
    monkeypatch.setattr(manager, "vast", lambda *_, **__: {"success": False})
    monkeypatch.setattr(manager, "instances", lambda: [])
    manager.create(argparse.Namespace(plan="unused", offer=99))
    snapshot = FinancialBook(manager.BOOK).snapshot()
    assert not snapshot["pending_labels"] and not snapshot["active_instances"]
    assert snapshot["active_estimate_usd"] == 0
    assert json.loads(manager.RENTAL.read_text())["status"] == "create_rejected_absent"


def test_acknowledged_id_persisted_before_provider_read(manager, monkeypatch):
    monkeypatch.setattr(manager, "vast", lambda *_, **__: {"success": True, "new_contract": 42})

    def current():
        rental = json.loads(manager.RENTAL.read_text())
        assert rental["instance_id"] == 42
        return [{"id": 42, "label": rental["label"]}]

    monkeypatch.setattr(manager, "instances", current)
    manager.create(argparse.Namespace(plan="unused", offer=99))


def test_destroy_rejects_crashed_but_billing_host(manager, monkeypatch, tmp_path):
    manager.RENTAL.write_text(json.dumps({"instance_id": 42}))
    marker = tmp_path / "verified.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setitem(
        sys.modules, "preserve_dsrl06", SimpleNamespace(verify_local=lambda _: None)
    )
    marker.write_text(
        json.dumps(
            {
                "instance_id": 42,
                "local_verified": True,
                "private_hf_verified": True,
                "pinned_revisions": {"dataset": "abc"},
                "manifest_path": str(manifest),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
        )
    )
    monkeypatch.setattr(
        manager,
        "instances",
        lambda: [
            {
                "id": 42,
                "actual_status": "exited",
                "cur_state": "running",
                "intended_status": "running",
            }
        ],
    )
    monkeypatch.setattr(manager, "vast", lambda *_: pytest.fail("Must not destroy"))
    with pytest.raises(ValueError, match="Stop and verify"):
        manager.destroy(argparse.Namespace(marker=marker))


def test_missing_preservation_prevents_destroy(manager, monkeypatch, tmp_path):
    manager.RENTAL.write_text(json.dumps({"instance_id": 42}))
    marker = tmp_path / "incomplete.json"
    marker.write_text(json.dumps({"instance_id": 42, "local_verified": True}))
    monkeypatch.setattr(manager, "vast", lambda *_: pytest.fail("Must not destroy"))
    with pytest.raises(ValueError, match="preservation is required"):
        manager.destroy(argparse.Namespace(marker=marker))


def test_verified_cleanup_confirms_cli_and_reconciles_lost_ack(manager, monkeypatch, tmp_path):
    manager.RENTAL.write_text(json.dumps({"instance_id": 42}))
    marker = tmp_path / "verified.json"
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    marker.write_text(
        json.dumps(
            {
                "instance_id": 42,
                "local_verified": True,
                "private_hf_verified": True,
                "pinned_revisions": {"dataset": "abc"},
                "manifest_path": str(manifest),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            }
        )
    )
    checks = []
    monkeypatch.setitem(
        sys.modules,
        "preserve_dsrl06",
        SimpleNamespace(verify_local=lambda _: checks.append("preserved")),
    )
    rows = iter(
        [
            [
                {
                    "id": 42,
                    "actual_status": "exited",
                    "cur_state": "stopped",
                    "intended_status": "stopped",
                }
            ],
            [],
        ]
    )
    monkeypatch.setattr(manager, "instances", lambda: next(rows))
    calls = []

    def cli(args):
        assert checks == ["preserved"]
        # Model the installed CLI: missing --yes aborts before the API mutation.
        assert args == ["destroy", "instance", "42", "--yes"]
        calls.append(args)
        raise RuntimeError("acknowledgement lost")

    monkeypatch.setattr(manager, "vast", cli)
    manager.destroy(argparse.Namespace(marker=marker))
    assert len(calls) == 1
    rental = json.loads(manager.RENTAL.read_text())
    assert rental["status"] == "destroyed" and rental["destroy_verified_utc"]


def test_duplicate_label_stops_both_hosts_without_second_create(manager, monkeypatch):
    calls = []

    def provider(args, **kwargs):
        calls.append(args)
        if args[:2] == ["create", "instance"]:
            return {"success": True, "new_contract": 11}
        assert args[:2] == ["stop", "instance"]
        return {"success": True}

    def current():
        label = json.loads(manager.RENTAL.read_text())["label"]
        return [{"id": number, "label": label} for number in (11, 12)]

    monkeypatch.setattr(manager, "vast", provider)
    monkeypatch.setattr(manager, "instances", current)
    with pytest.raises(RuntimeError, match="duplicate labels"):
        manager.create(argparse.Namespace(plan="unused", offer=99))
    assert len([call for call in calls if call[0] == "create"]) == 1
    assert {int(call[2]) for call in calls if call[0] == "stop"} == {11, 12}
    state = FinancialBook(manager.BOOK).snapshot()
    assert state["active_instances"] == [11, 12] and state["stop"]
