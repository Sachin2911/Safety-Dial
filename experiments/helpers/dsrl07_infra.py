"""Task-7 allocation namespace; retain all task-6 spending in the same $15 cap."""

import hashlib
import json
from pathlib import Path

from helpers.dsrl06_infra import FinancialBook as PreviousFinancialBook
from helpers.dsrl06_infra import TEMPLATE as TEMPLATE
from helpers.dsrl06_infra import atomic_json as atomic_json
from helpers.dsrl06_infra import instances as instances
from helpers.dsrl06_infra import provider_stopped as provider_stopped
from helpers.dsrl06_infra import ssh_endpoint as ssh_endpoint
from helpers.dsrl06_infra import ssh_options as ssh_options
from helpers.dsrl06_infra import utc as utc
from helpers.dsrl06_infra import vast as vast

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "progress/receipts/07-dsrl-financial-history.json"


class FinancialBook(PreviousFinancialBook):
    @staticmethod
    def limits():
        return {**PreviousFinancialBook.limits(), "aggregate_hours": 8.0}

    def __init__(self, path):
        super().__init__(path)
        history = json.loads(HISTORY.read_text())
        digest = hashlib.sha256(HISTORY.read_bytes()).hexdigest()
        with self.transaction():
            if self.data.get("historical_receipt_sha256"):
                if self.data["historical_receipt_sha256"] != digest:
                    raise ValueError("Historical spending receipt changed")
                return
            if self.data["allocations"]:
                raise ValueError("Import historical costs before any new reservation")
            if any(not row["active_stop_verified_utc"] for row in history["allocations"]):
                raise ValueError("A historical allocation has no verified stop")
            self.data["allocations"] = history["allocations"]
            self.data["historical_receipt_sha256"] = digest
            self.data["schema"] = "dsrl07-finance-v1"
