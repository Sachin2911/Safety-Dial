"""Financial restarts, overlapping allocations and secret-safe CLI failure behavior."""

import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from helpers.dsrl06_infra import FinancialBook, instances, provider_stopped, timestamp, vast


def offer(rate=2.0):
    return {
        "dph_total": rate,
        "driver_version": "580.173.02",
        "gpu_name": "RTX 4090",
        "num_gpus": 4,
        "cpu_ram": 257000,
        "cpu_cores_effective": 64,
        "inet_down_cost": 0.003,
        "inet_up_cost": 0.004,
    }


def date(hours):
    return (
        dt.datetime(2026, 10, 25, tzinfo=dt.timezone.utc) + dt.timedelta(hours=hours)
    ).isoformat()


def test_replacement_preserves_cost_and_aggregate_time(tmp_path):
    p = tmp_path / "book.json"
    b = FinancialBook(p)
    b.add(1, date(0), offer())
    b.stopped(1, date(0.5), status="exited")
    b = FinancialBook(p)
    b.add(2, date(1), offer())
    snap = b.snapshot(timestamp(date(3)))
    assert snap["active_estimate_usd"] == 5
    assert snap["aggregate_hours"] == 2.5
    assert snap["remaining_seconds"] == pytest.approx(3.5 * 3600)
    assert snap["active_instances"] == [2]


def test_money_stops_before_total_cap(tmp_path):
    b = FinancialBook(tmp_path / "book.json")
    b.add(1, date(0), offer(2.2))
    snap = b.snapshot(timestamp(date(5.9)))
    assert snap["stop"]
    assert snap["active_estimate_usd"] == pytest.approx(12.98)
    assert snap["total_estimate_usd"] < 15


def test_accidental_overlap_consumes_both_rates_and_hours(tmp_path):
    b = FinancialBook(tmp_path / "book.json")
    b.add(1, date(0), offer())
    b.add(2, date(0), offer())
    snap = b.snapshot(timestamp(date(1)))
    assert snap["aggregate_hours"] == 2
    assert snap["active_estimate_usd"] == 4
    assert snap["remaining_seconds"] == pytest.approx(2 * 3600)


def test_limits_cannot_change_on_resume(tmp_path):
    p = tmp_path / "book.json"
    FinancialBook(p)
    data = json.loads(p.read_text())
    data["limits"]["total_usd"] = 16
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="limits changed"):
        FinancialBook(p)


def test_pending_reservation_is_charged_before_create_ack(tmp_path):
    p = tmp_path / "book.json"
    book = FinancialBook(p)
    book.reserve("unique-task6", date(0), offer())
    snap = FinancialBook(p).snapshot(timestamp(date(0.5)))
    assert snap["active_estimate_usd"] == 1
    assert snap["active_instances"] == []
    assert snap["pending_labels"] == ["unique-task6"]
    book.attach("unique-task6", 71)
    snap = FinancialBook(p).snapshot(timestamp(date(0.5)))
    assert snap["active_instances"] == [71] and snap["active_estimate_usd"] == 1
    with pytest.raises(ValueError, match="existing allocation"):
        book.reserve("another-task6", date(0.5), offer())


def test_stale_book_cannot_erase_a_new_allocation(tmp_path):
    p = tmp_path / "book.json"
    first = FinancialBook(p)
    first.add(1, date(0), offer())
    stale = FinancialBook(p)
    first.add(2, date(1), offer())
    stale.stopped(1, date(1), status="exited")
    assert [r["instance_id"] for r in FinancialBook(p).data["allocations"]] == [1, 2]


def test_snapshot_refreshes_charges_and_allocations(tmp_path):
    p = tmp_path / "book.json"
    stale = FinancialBook(p)
    fresh = FinancialBook(p)
    fresh.add(1, date(0), offer())
    fresh.other_charges(3.0)
    snap = stale.snapshot(timestamp(date(1)))
    assert snap["active_instances"] == [1]
    assert snap["other_reserved_or_actual_usd"] == 3


def test_replacement_transfer_estimates_accumulate(tmp_path):
    b = FinancialBook(tmp_path / "book.json")
    o = offer()
    o["inet_down_cost"] = 0.016
    b.add(1, date(0), o)
    b.stopped(1, date(0.1), status="exited")
    b.add(2, date(0.2), o)
    snap = b.snapshot(timestamp(date(0.3)))
    assert snap["other_reserved_or_actual_usd"] == pytest.approx(3.264)


def test_error_dictionary_is_not_provider_absence():
    with patch("helpers.dsrl06_infra.vast", return_value={"error": "unauthorized"}):
        with pytest.raises(RuntimeError, match="absence is unproven"):
            instances()


def test_embedded_error_is_not_provider_absence():
    for response in ({"error": "bad", "instances": []}, {"success": False, "instances": []}):
        with patch("helpers.dsrl06_infra.vast", return_value=response):
            with pytest.raises(RuntimeError, match="absence is unproven"):
                instances()


def test_exited_process_with_running_intention_is_not_stopped():
    assert not provider_stopped(
        {"actual_status": "exited", "intended_status": "running", "cur_state": "running"}
    )
    assert provider_stopped(
        {"actual_status": "exited", "intended_status": "stopped", "cur_state": "stopped"}
    )


def test_known_rejected_absent_reservation_can_be_released(tmp_path):
    book = FinancialBook(tmp_path / "book.json")
    book.reserve("first", date(0), offer())
    book.cancel_rejected("first", absence_verified_utc=date(0.01))
    snapshot = book.snapshot(timestamp(date(1)))
    assert snapshot["active_estimate_usd"] == 0
    assert snapshot["pending_labels"] == []
    book.reserve("second", date(1), offer())
    book.attach("second", 3)
    with pytest.raises(ValueError, match="attached rental"):
        book.cancel_rejected("second", absence_verified_utc=date(1.01))


def test_cli_error_does_not_expose_raw_api_key():
    class Result:
        returncode = 1
        stdout = "https://example.invalid/?api_key=secret-canary"
        stderr = "secret-canary"

    with patch("subprocess.run", return_value=Result()), pytest.raises(RuntimeError) as error:
        vast(["show", "instances"])
    assert "secret-canary" not in str(error.value)


def test_unsupported_offer_rejected(tmp_path):
    b = FinancialBook(tmp_path / "book.json")
    with pytest.raises(ValueError, match="hourly rate"):
        b.add(1, date(0), offer(2.21))
    o = offer()
    o["inet_down_cost"] = 0.03
    with pytest.raises(ValueError, match="transfer"):
        b.add(1, date(0), o)


@pytest.mark.parametrize("acknowledged", [False, True])
def test_duplicate_label_charges_every_host_and_forces_stop(tmp_path, acknowledged):
    book = FinancialBook(tmp_path / "book.json")
    book.reserve("ambiguous", date(0), offer())
    if acknowledged:
        book.attach("ambiguous", 11)
    book.reconcile_duplicate_label("ambiguous", [11, 12, 12])
    book.reconcile_duplicate_label("ambiguous", [11, 12])
    state = book.snapshot(timestamp(date(1)))
    assert state["active_instances"] == [11, 12] and not state["pending_labels"]
    assert state["active_estimate_usd"] == 4 and state["aggregate_hours"] == 2
    assert state["stop"] and len(book.data["allocations"]) == 2
    book.stopped(11, date(1), status="stopped")
    assert book.snapshot(timestamp(date(1)))["stop"]
    book.stopped(12, date(1), status="stopped")
    state = book.snapshot(timestamp(date(1)))
    assert not state["active_instances"] and not state["stop"]


def test_watchdog_stops_duplicate_label_hosts_before_budget_deadline(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "dsrl06_watchdog_regression", Path(__file__).parents[1] / "scripts/dsrl06_watchdog.py"
    )
    watchdog = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watchdog)
    path = tmp_path / "finance.json"
    FinancialBook(path).reserve("ambiguous", dt.datetime.now(dt.timezone.utc).isoformat(), offer())
    stopped = set()

    def current():
        return [
            {
                "id": number,
                "label": "ambiguous",
                "actual_status": "exited" if number in stopped else "running",
                "intended_status": "stopped" if number in stopped else "running",
                "cur_state": "stopped" if number in stopped else "running",
            }
            for number in (11, 12)
        ]

    def stop(args):
        assert args[:2] == ["stop", "instance"]
        stopped.add(int(args[2]))

    monkeypatch.setattr(sys, "argv", ["guard", "--book", str(path)])
    monkeypatch.setattr(watchdog, "instances", current)
    monkeypatch.setattr(watchdog, "vast", stop)
    monkeypatch.setattr(watchdog.time, "sleep", lambda _: pytest.fail("Stop must finish now"))
    watchdog.main()
    assert stopped == {11, 12}
    assert not FinancialBook(path).snapshot()["active_instances"]
