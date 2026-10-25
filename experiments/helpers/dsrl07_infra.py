"""Task-7 allocation namespace; retain all task-6 spending in the same $15 cap."""

import hashlib
import json
import math
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
from helpers.dsrl07_reallocation import REALLOCATED_LIMITS

ROOT = Path(__file__).resolve().parents[2]
HISTORY = ROOT / "progress/receipts/07-dsrl-financial-history.json"


def stage_b_projection(episode_seconds, update_seconds, cpu_actor_seconds):
    """Reserve both preservation time and the queue's final-start allowance."""
    values = tuple(map(float, (episode_seconds, update_seconds, cpu_actor_seconds)))
    if not all(map(math.isfinite, values)) or min(values[:2]) <= 0 or values[2] < 0:
        raise ValueError("Projection requires finite positive episode/update durations")
    episode_seconds, update_seconds, cpu_actor_seconds = values
    # Eight three-episode training rounds and twelve four-arm evaluation blocks.
    collection = 20 * episode_seconds
    learning = 90000 * update_seconds
    actor = 3000 * cpu_actor_seconds
    # Every lane has already run during engineering. The queue reserves 25 minutes
    # for a cached episode plus 20 minutes for preservation at each new admission.
    # Merely adding 25 minutes after measured work could reject the last block.
    reserve = max(1500.0, 2700.0 - episode_seconds)
    return {
        "collection_rounds": 20,
        "collection_seconds": collection,
        "learning_seconds": learning,
        "evaluation_actor_seconds": actor,
        "final_reserve_seconds": reserve,
        "last_start_allowance_seconds": 2700,
        "total_seconds": collection + learning + actor + reserve,
    }


class FinancialBook(PreviousFinancialBook):
    @staticmethod
    def limits():
        return REALLOCATED_LIMITS.copy()

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
