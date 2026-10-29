"""Independent $10 financial scope for the authorized task 15 regrasp adapter rental."""

from helpers.dsrl06_infra import FinancialBook as BaseFinancialBook
from helpers.dsrl06_infra import TEMPLATE as TEMPLATE
from helpers.dsrl06_infra import atomic_json as atomic_json
from helpers.dsrl06_infra import instances as instances
from helpers.dsrl06_infra import provider_stopped as provider_stopped
from helpers.dsrl06_infra import ssh_endpoint as ssh_endpoint
from helpers.dsrl06_infra import ssh_options as base_ssh_options
from helpers.dsrl06_infra import utc as utc
from helpers.dsrl06_infra import vast as vast


def ssh_options(port):
    return [x.replace("/tmp/dsrl06-known-hosts", "/tmp/regrasp15-known-hosts")
            for x in base_ssh_options(port)]


class FinancialBook(BaseFinancialBook):
    @staticmethod
    def limits():
        return {"total_usd": 10.0, "active_usd": 8.0, "active_shutdown_usd": 7.9,
                "other_reserve_usd": 2.0, "aggregate_hours": 20.0, "rate_usd_hour": 0.65}

    @staticmethod
    def validate_offer(offer):
        if not 0 < float(offer["dph_total"]) <= 0.65:
            raise ValueError("Offer exceeds the task 15 hourly ceiling")
        if not str(offer["driver_version"]).startswith("580."):
            raise ValueError("Wrong driver family")
        if offer["gpu_name"] not in ("RTX 3090", "RTX 4090") or int(offer["num_gpus"]) != 1:
            raise ValueError("Task 15 requires one RTX 3090 or RTX 4090")
        if float(offer["cpu_ram"]) < 60000 or float(offer["cpu_cores_effective"]) < 12:
            raise ValueError("Insufficient allocated host resources")
        transfer = 150 * float(offer["inet_down_cost"]) + 40 * float(offer["inet_up_cost"])
        if transfer > 1.5:
            raise ValueError("Transfer allowance leaves insufficient disk reserve")

    @staticmethod
    def _row(instance_id, created_utc, offer):
        row = BaseFinancialBook._row(instance_id, created_utc, offer)
        row["transfer_estimate_usd"] = (
            150 * float(offer["inet_down_cost"]) + 40 * float(offer["inet_up_cost"])
        )
        return row
