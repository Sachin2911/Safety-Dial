"""The reserve transfer keeps charges, scientific choices and the $15 boundary."""

import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys

import pytest

from helpers.dsrl07_infra import FinancialBook
from helpers.dsrl07_reallocation import (
    ORIGINAL_LIMITS,
    REALLOCATED_LIMITS,
    reallocated_book_data,
    remote_other_reserve,
    verify_reserve_proof,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_dsrl07 import validate_amendment  # noqa: E402


def fixture():
    old = {
        "stage": "B",
        "money": ORIGINAL_LIMITS.copy(),
        "jobs": {"evaluation_jobs": ["native", "fixed_zero", "random", "dsrl"]},
        "caps": {"learning_updates": 90000},
        "source_sha256": {"runner.py": "before"},
        "stage_a_summary_sha256": "unchanged",
    }
    new = copy.deepcopy(old)
    new.update(
        money=REALLOCATED_LIMITS.copy(),
        source_sha256={"runner.py": "after"},
        supersedes_sha256="old",
        amendment={"type": "financial_reserve_reallocation"},
    )
    receipt = {
        "schema": "dsrl07-reserve-reallocation-v1",
        "repair": "Retain all 48 evaluations within the existing $15 cap",
        "old_freeze_sha256": "old",
        "new_freeze_sha256": "new",
        "old_limits": ORIGINAL_LIMITS.copy(),
        "new_limits": REALLOCATED_LIMITS.copy(),
        "hard_cumulative_usd": 15,
        "reserve_proof": {
            "aggregate_transfer_estimate_usd": 1.6284594104,
            "stopped_disk_rate_usd_hour": 0.04,
            "stopped_disk_hours": 2,
            "other_actual_usd": 0,
        },
    }
    return old, new, receipt


def test_reallocation_retains_full_transfer_and_disk_envelope():
    old, new, receipt = fixture()
    validate_amendment(old, new, receipt, "old", "new")
    assert verify_reserve_proof(receipt["reserve_proof"]) == pytest.approx(1.7084594104)
    assert REALLOCATED_LIMITS["active_usd"] + REALLOCATED_LIMITS["other_reserve_usd"] == 15


@pytest.mark.parametrize("field", ["jobs", "caps", "stage_a_summary_sha256"])
def test_financial_receipt_cannot_change_science(field):
    old, new, receipt = fixture()
    new[field] = "changed"
    with pytest.raises(ValueError, match="scientific freeze"):
        validate_amendment(old, new, receipt, "old", "new")


@pytest.mark.parametrize("field", ["total_usd", "aggregate_hours", "rate_usd_hour"])
def test_financial_receipt_cannot_raise_authorized_limits(field):
    old, new, receipt = fixture()
    new["money"][field] += 1
    with pytest.raises(ValueError, match="financial envelope"):
        validate_amendment(old, new, receipt, "old", "new")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("aggregate_transfer_estimate_usd", 1.8),
        ("stopped_disk_hours", 1.9),
        ("stopped_disk_rate_usd_hour", 0.03),
        ("other_actual_usd", float("nan")),
    ],
)
def test_insufficient_or_invalid_reserve_is_rejected(field, value):
    _, _, receipt = fixture()
    receipt["reserve_proof"][field] = value
    with pytest.raises(ValueError):
        verify_reserve_proof(receipt["reserve_proof"])


def test_migration_retains_every_allocation_and_rejects_stale_or_understated_proof():
    _, _, receipt = fixture()
    before = {
        "limits": ORIGINAL_LIMITS.copy(),
        "allocations": [{"instance_id": 7, "transfer_estimate_usd": 1.6284594104}],
        "other_actual_usd": 0,
        "historical_receipt_sha256": "retained",
    }
    original = json.dumps(before).encode()
    receipt["old_book_sha256"] = hashlib.sha256(original).hexdigest()
    result = reallocated_book_data(before, original, receipt)
    assert result["allocations"] == before["allocations"]
    assert result["historical_receipt_sha256"] == "retained"
    assert before["limits"] == ORIGINAL_LIMITS
    with pytest.raises(ValueError, match="changed since"):
        reallocated_book_data(before, original + b" ", receipt)
    receipt["reserve_proof"]["aggregate_transfer_estimate_usd"] = 1.5
    with pytest.raises(ValueError, match="understates"):
        reallocated_book_data(before, original, receipt)


def test_dollar_and_extra_charge_guards_remain_binding(tmp_path):
    book = FinancialBook(tmp_path / "finance.json")
    offer = {**book.data["allocations"][0]["offer"], "dph_total": 1.8503703703703704}
    created = "2026-10-25T07:45:06.606764+00:00"
    book.add(57632188, created, offer)
    start = dt.datetime.fromisoformat(created).timestamp()
    first = book.snapshot(start)
    deadline = dt.datetime.fromisoformat(first["deadline_utc"]).timestamp()
    final = book.snapshot(deadline + 1)
    assert final["stop"]
    assert 14.9 <= final["total_estimate_usd"] < 15
    assert final["aggregate_hours"] < 8
    book.other_charges(2.5)
    revised = book.snapshot(start)
    assert dt.datetime.fromisoformat(revised["deadline_utc"]).timestamp() < deadline
    assert revised["other_reserved_or_actual_usd"] == 2.5


def test_remote_guard_requires_recorded_reallocation_and_keeps_old_default():
    assert remote_other_reserve({}) == 2
    with pytest.raises(ValueError, match="recorded reserve"):
        remote_other_reserve({"financial_limits": REALLOCATED_LIMITS})
    assert (
        remote_other_reserve(
            {"financial_limits": REALLOCATED_LIMITS, "reserve_reallocation_sha256": "a" * 64}
        )
        == 1.8
    )
