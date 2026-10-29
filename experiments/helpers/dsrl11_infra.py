"""Fresh task-11 financial allocation and the preliminary full-schedule projection.

The task-6/7 ledger is historical and is neither imported nor charged here. Provider
helpers are reused unchanged; only the limits and the schedule projection are new.
"""

from __future__ import annotations

import math

from helpers.dsrl06_infra import FinancialBook as PreviousFinancialBook
from helpers.dsrl06_infra import TEMPLATE as TEMPLATE
from helpers.dsrl06_infra import atomic_json as atomic_json
from helpers.dsrl06_infra import instances as instances
from helpers.dsrl06_infra import provider_stopped as provider_stopped
from helpers.dsrl06_infra import ssh_endpoint as ssh_endpoint
from helpers.dsrl06_infra import ssh_options as ssh_options
from helpers.dsrl06_infra import timestamp as timestamp
from helpers.dsrl06_infra import utc as utc
from helpers.dsrl06_infra import vast as vast

# Sachin raised the cap from $15 to $18 at 15:58 UTC on 25 October 2026 so that an
# early rental, overlapping setup with workstation work, cannot block the frozen run.
LIMITS = {
    "total_usd": 18.0,
    "active_usd": 15.0,
    "active_shutdown_usd": 14.9,
    "other_reserve_usd": 3.0,
    "aggregate_hours": 8.0,
    "rate_usd_hour": 2.2,
}


class FinancialBook(PreviousFinancialBook):
    @staticmethod
    def limits() -> dict:
        return dict(LIMITS)

    def __init__(self, path):
        super().__init__(path)
        with self.transaction():
            if self.data.get("schema") == "dsrl06-finance-v1" and not self.data["allocations"]:
                self.data["schema"] = "dsrl11-finance-v1"
            if self.data["schema"] != "dsrl11-finance-v1":
                raise ValueError("The task-11 book holds only its fresh allocation")


def full_schedule_projection(
    timings: dict, rate_usd_hour: float, utd: int = 20, waiting_seconds: float = 0.0
) -> dict:
    """Conservative serial projection of every planned phase at one quoted rate.

    ``timings`` are seconds from task 7 (planning evidence, not admission evidence):
    preparation, cold and cached episode durations, three- and four-lane rounds, the
    per-update time and the local-copy reserve. ``waiting_seconds`` charges rental
    time spent waiting for workstation work after preparation.
    """
    values = {key: float(value) for key, value in timings.items()}
    if not all(math.isfinite(v) and v > 0 for v in values.values()):
        raise ValueError("Projection requires finite positive timings")
    if not 0 < rate_usd_hour <= LIMITS["rate_usd_hour"]:
        raise ValueError("Quoted rate exceeds the authorized hourly limit")
    phases = {
        "create_and_bootstrap": values["bootstrap"],
        "preparation_with_a1_overlap": values["preparation"],
        "a2_and_server_startup": values["numerical"],
        # E0-E3 cold on four lanes, E4-E7 cached, learner profile off the critical path.
        "engineering": values["cold_episode"] + values["four_lane_episode"],
        "bank": 3 * values["four_lane_episode"],
        "online_collection": 12 * values["three_lane_episode"],
        "final_learner_round": 750 * utd * values["update"] * 1.10,
        "evaluation": 10 * values["four_lane_episode"],
        "server_restarts_and_admission": values["restarts"],
        "local_copy_reserve": values["local_copy"],
        "workstation_wait": float(waiting_seconds),
    }
    seconds = sum(phases.values())
    hours = seconds / 3600
    active = hours * rate_usd_hour
    allowance = LIMITS["active_shutdown_usd"] / rate_usd_hour
    return {
        "rate_usd_hour": rate_usd_hour,
        "phases_seconds": phases,
        "total_seconds": seconds,
        "total_hours": hours,
        "projected_active_usd": active,
        "money_allowance_hours": allowance,
        "allowed_hours": min(allowance, LIMITS["aggregate_hours"]),
        "fits": hours <= min(allowance, LIMITS["aggregate_hours"]),
        "margin_hours": min(allowance, LIMITS["aggregate_hours"]) - hours,
    }
