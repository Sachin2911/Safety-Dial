"""One recorded reserve transfer within the authorized cumulative $15 envelope."""

import hashlib
import json
import math

from helpers.dsrl06_infra import FinancialBook as OriginalFinancialBook

ORIGINAL_LIMITS = {**OriginalFinancialBook.limits(), "aggregate_hours": 8.0}
REALLOCATED_LIMITS = {
    **ORIGINAL_LIMITS,
    "active_usd": 13.2,
    "active_shutdown_usd": 13.1,
    "other_reserve_usd": 1.8,
}


def verify_reserve_proof(proof):
    """Keep the existing transfer envelope and two hours of stopped storage."""
    required = (
        "aggregate_transfer_estimate_usd",
        "stopped_disk_rate_usd_hour",
        "stopped_disk_hours",
        "other_actual_usd",
    )
    values = {key: float(proof[key]) for key in required}
    if not all(math.isfinite(value) and value >= 0 for value in values.values()):
        raise ValueError("Reserve proof has invalid financial quantities")
    if values["stopped_disk_hours"] < 2:
        raise ValueError("Retain at least two hours of stopped-disk allowance")
    if values["stopped_disk_rate_usd_hour"] < 0.04:
        raise ValueError("Retain the rounded-up 200-GB storage rate")
    reserved = (
        max(values["aggregate_transfer_estimate_usd"], values["other_actual_usd"])
        + values["stopped_disk_hours"] * values["stopped_disk_rate_usd_hour"]
    )
    if reserved > REALLOCATED_LIMITS["other_reserve_usd"]:
        raise ValueError("Transfer and stopped-disk allowance do not fit the revised reserve")
    return reserved


def verify_reallocation_receipt(receipt, old_hash, new_hash):
    if (
        not receipt
        or receipt.get("schema") != "dsrl07-reserve-reallocation-v1"
        or not receipt.get("repair")
        or receipt.get("old_freeze_sha256") != old_hash
        or receipt.get("new_freeze_sha256") != new_hash
        or receipt.get("old_limits") != ORIGINAL_LIMITS
        or receipt.get("new_limits") != REALLOCATED_LIMITS
        or receipt.get("hard_cumulative_usd") != 15
    ):
        raise ValueError("A matching explicit reserve-reallocation receipt is required")
    verify_reserve_proof(receipt["reserve_proof"])


def reallocated_book_data(data, original_bytes, receipt):
    """Return a migrated copy; the caller owns the book's exclusive file lock."""
    if data.get("limits") != ORIGINAL_LIMITS:
        raise ValueError("Only the original task-7 financial book can be migrated")
    if hashlib.sha256(original_bytes).hexdigest() != receipt.get("old_book_sha256"):
        raise ValueError("Financial book changed since the reserve proof was frozen")
    verify_reallocation_receipt(receipt, receipt["old_freeze_sha256"], receipt["new_freeze_sha256"])
    transfers = sum(
        row.get("transfer_actual_usd", row["transfer_estimate_usd"]) for row in data["allocations"]
    )
    proof = receipt["reserve_proof"]
    if (
        proof["aggregate_transfer_estimate_usd"] < transfers
        or proof["other_actual_usd"] < data["other_actual_usd"]
    ):
        raise ValueError("Reserve proof understates retained historical/current charges")
    result = json.loads(json.dumps(data))
    result["limits"] = REALLOCATED_LIMITS.copy()
    result["reserve_reallocation"] = {
        "old_freeze_sha256": receipt["old_freeze_sha256"],
        "new_freeze_sha256": receipt["new_freeze_sha256"],
        "old_book_sha256": receipt["old_book_sha256"],
    }
    return result


def remote_other_reserve(rental):
    limits = rental.get("financial_limits")
    if limits is None:
        return ORIGINAL_LIMITS["other_reserve_usd"]
    if limits != REALLOCATED_LIMITS or len(rental.get("reserve_reallocation_sha256", "")) != 64:
        raise ValueError("Remote guard requires the recorded reserve reallocation")
    return limits["other_reserve_usd"]
