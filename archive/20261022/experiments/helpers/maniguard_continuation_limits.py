"""Finite technical guards for the separate four-start engineering screen."""

import calendar
import math
import time

DEFAULT_LIMITS = {
    "policy_controls": 2000,
    "policy_requests": 250,
    "encoder_image_batches": 250,
    "encoder_images": 500,
    "encoder_text_encodings": 1,
    "initialization_environment_steps": 64,
    "total_environment_steps": 2064,
    "observed_initialization_physics_ticks": 3840,
    "initialization_wall_seconds": 600,
    "full_start_wall_seconds": 2700,
    "preservation_reserve_seconds": 900,
}
GUARDIAN_OVERRUN_SECONDS = 4  # At most one-second polling plus three-second TERM grace.


def validate_limits(limits):
    if set(limits) != set(DEFAULT_LIMITS):
        raise ValueError("Freeze every finite screen limit explicitly")
    if any(type(value) is not int or not 0 < value <= DEFAULT_LIMITS[key]
           for key, value in limits.items()):
        raise ValueError("Screen technical limits exceed the finite four-start prototype")
    if limits["policy_controls"] != 2000 or limits["policy_requests"] != 250:
        raise ValueError("Technical limits cannot shorten native Jar control/query horizons")


def counter_violations(record, limits):
    validate_limits(limits)
    mappings = {
        "policy_control_steps": "policy_controls",
        "native_policy_requests_admitted": "policy_requests",
        "encoder_image_attempts": "encoder_image_batches",
        "encoder_images": "encoder_images",
        "encoder_text_attempts": "encoder_text_encodings",
        "initialization_env_step_attempts": "initialization_environment_steps",
        "env_step_attempts": "total_environment_steps",
    }
    errors = []
    for counter, limit in mappings.items():
        value = record.get(counter, 0)
        if type(value) is not int or value < 0 or value > limits[limit]:
            errors.append(counter)
    ticks = record.get("initialization_physics_steps", record.get("physics_steps", 0))
    if not record.get("initialization_finished", False):
        ticks = record.get("physics_steps", 0)
    if type(ticks) is not int or ticks < 0 or ticks > limits["observed_initialization_physics_ticks"]:
        errors.append("observed_initialization_physics_ticks")
    return errors


def epoch(utc):
    # Match the pinned Python3.10 simulator's explicit, whole-second UTC format.
    return calendar.timegm(time.strptime(utc, "%Y-%m-%dT%H:%M:%SZ"))


def screen_capacity(protocol, attempts, *, now, actual_stop_epoch):
    """Reserve a complete timeout for each unfinished wave, with no elapsed credit."""
    limits = protocol["runtime_limits"]
    validate_limits(limits)
    if any(a["status"] == "invalid" for a in attempts):
        raise ValueError("An invalid screen start requires diagnosis, with no automatic replacement")
    if len(attempts) > 4 or len({a["protocol_episode_id"] for a in attempts}) != len(attempts):
        raise ValueError("Four-start screen ledger is duplicated or over budget")
    completed = {a["protocol_episode_id"] for a in attempts if a["status"] == "completed"}
    waves = {row["wave"] for row in protocol["episodes"] if row["id"] not in completed}
    cutoff = min(epoch(protocol["physical_cutoff_utc"]),
                 actual_stop_epoch - limits["preservation_reserve_seconds"])
    required = len(waves) * (limits["full_start_wall_seconds"] + GUARDIAN_OVERRUN_SECONDS)
    required += limits["preservation_reserve_seconds"]
    return {"remaining_waves": len(waves), "required_seconds": required,
            "available_seconds": cutoff - now, "fits": now + required <= cutoff,
            "physical_cutoff_epoch": cutoff}


def resource_contract(instance, clock, *, now, watchdog_live):
    """Validate sanitized own-instance/readback data; never access a credential here."""
    rate = instance.get("dph_total")
    if (instance.get("id") != 56777435 or instance.get("machine_id") != 54514
            or instance.get("actual_status") != "running" or instance.get("num_gpus") != 2
            or type(rate) not in (float, int) or not math.isfinite(rate) or not 0 < rate <= .65):
        raise ValueError("Actual same-instance hardware/status/rate does not match authorization")
    if (clock.get("instance_id") != 56777435 or clock.get("status") != "armed"
            or clock.get("action") != "stop_retaining_disk" or not watchdog_live
            or type(clock.get("checked_epoch")) not in (float, int)
            or not 0 <= now - clock["checked_epoch"] <= 65):
        raise ValueError("The independent stop clock is missing, stale or no longer live")
    stop = epoch(clock["deadline_utc"])
    if stop <= now:
        raise ValueError("The actual authorized resource deadline has elapsed")
    return stop
