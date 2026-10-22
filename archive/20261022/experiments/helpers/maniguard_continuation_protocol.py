"""Pure binding of a first four-start screen job to its declared protocol table."""

from pathlib import Path

from helpers.maniguard_continuation_handoff import MODES
from helpers.maniguard_continuation_service import ledger_path


def validate_plan(protocol):
    episodes = protocol["episodes"]
    if (len(episodes) != 4 or {row["condition"] for row in episodes} != MODES
            or len({row["id"] for row in episodes}) != 4):
        raise ValueError("The initial screen must declare its exact four reference jobs")
    budget = protocol["first_screen_budget"]
    if (budget["maximum_full_starts_including_partial_failures"] != 4
            or budget["automatic_replacements"] != 0):
        raise ValueError("The initial screen has four total starts and no automatic replacements")
    for row in episodes:
        if str(row["worker"]) not in protocol["workers"]:
            raise ValueError("The protocol job has no declared worker")
    if ({row["wave"] for row in episodes} != {0, 1}
            or any(len([r for r in episodes if r["wave"] == wave]) != 2
                   or len({r["worker"] for r in episodes if r["wave"] == wave}) != 2
                   for wave in (0, 1))):
        raise ValueError("Freeze exactly two waves, each using two distinct workers")


def validate_job(protocol, job):
    """Shared by future admission and independent physical evidence validation."""
    validate_plan(protocol)
    planned = [row for row in protocol["episodes"] if row["id"] == job["protocol_episode_id"]]
    if len(planned) != 1:
        raise ValueError("Job is absent from the frozen four-start table")
    planned = planned[0]
    mode = planned["condition"]
    if (job["condition"] != mode or job["worker"] != planned["worker"]
            or job["continuation"] != protocol["selector_parameters"][mode]
            or job["continuation_contract_sha256"] != protocol["selector_contract_sha256"][mode]
            or job["scene"] != protocol["scene"] or job["seed"] != protocol["seed"]):
        raise ValueError("Job condition, selector, case or worker differs from the frozen table")
    worker = protocol["workers"][str(planned["worker"])]
    expected_output = Path(protocol["run_directory"]).resolve() / "episodes" / (planned["id"] + "-a0")
    if Path(job["output"]).resolve() != expected_output:
        raise ValueError("Job output differs from the canonical screen start ledger")
    expected_lease = Path(protocol["service_lease_directory"]) / f"{worker['port']}.lock"
    if Path(job["service_lock_path"]) != expected_lease:
        raise ValueError("Job service lease differs from the declared shared authority")
    if (job["scene_asset_sha256"] != protocol["scene_asset_sha256"]
            or job["service_ledger_baseline"]["path"] != str(ledger_path(protocol, job["worker"]))):
        raise ValueError("The scene or service baseline differs from the declared source")
    for job_key, plan_key in (
        ("policy_port", "port"), ("simulation_gpu_uuid", "simulation_gpu_uuid"),
        ("policy_gpu_uuid", "policy_gpu_uuid"), ("renderer_vulkan_index", "renderer_vulkan_index"),
    ):
        if job[job_key] != worker[plan_key]:
            raise ValueError("Job device/service differs from its frozen worker: " + job_key)
    return planned
