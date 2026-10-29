"""Lease guards must close early and count transfers independently of the laptop."""

import datetime as dt
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.lewm17_remote import finance


def record():
    return {
        "created_utc": "2026-10-26T12:00:00+00:00",
        "deadline_utc": "2026-10-26T17:00:00+00:00",
        "offer": {"dph_total": 1.0, "inet_down_cost": 0.01, "inet_up_cost": 0.02},
    }


def at(hour, minute=0):
    return dt.datetime(2026, 10, 26, hour, minute, tzinfo=dt.timezone.utc).timestamp()


def test_closeout_precedes_hard_deadline():
    assert not finance(record(), at(16, 29), 0, 0)["closeout_due"]
    assert finance(record(), at(16, 30), 0, 0)["closeout_due"]
    assert not finance(record(), at(16, 59), 0, 0)["hard_stop_due"]
    assert finance(record(), at(17), 0, 0)["hard_stop_due"]


def test_network_cost_can_trigger_early_closeout():
    result = finance(record(), at(13), 300e9, 230e9)
    assert math.isclose(result["measured_transfer_usd"], 7.6)
    assert result["closeout_due"] and not result["hard_stop_due"]
    assert finance(record(), at(13), 300e9, 280e9)["hard_stop_due"]


def test_unmeasured_transfer_retains_reserve():
    result = finance(record(), at(13), 0, 0)
    assert result["compute_disk_usd"] == 1
    assert result["conservative_total_usd"] == 1.75
