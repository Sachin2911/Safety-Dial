"""No live services: finite counters, whole-screen time and CPU child lifecycles."""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

from helpers.maniguard_continuation_guardian import run_guarded
from helpers.maniguard_continuation_limits import (
    DEFAULT_LIMITS,
    counter_violations,
    resource_contract,
    screen_capacity,
    validate_limits,
)
from helpers.maniguard_latent_queue import process_identity


def test_native_horizon_cannot_be_shortened_or_extra_queries_added():
    for changed in ({"policy_controls": 100}, {"policy_requests": 251}):
        with pytest.raises(ValueError):
            validate_limits({**DEFAULT_LIMITS, **changed})


def test_initialization_ticks_are_frozen_at_completion_not_counted_as_policy_ticks():
    record = {"initialization_finished": True, "physics_steps": 12089,
              "initialization_physics_steps": 89}
    assert not counter_violations(record, DEFAULT_LIMITS)
    record["initialization_finished"] = False
    assert counter_violations(record, DEFAULT_LIMITS) == ["observed_initialization_physics_ticks"]
    assert counter_violations({"native_policy_requests_admitted": 251}, DEFAULT_LIMITS)


def test_whole_screen_capacity_counts_active_wave_fully_and_protects_reserve():
    protocol = {"runtime_limits": DEFAULT_LIMITS, "physical_cutoff_utc": "2026-10-20T19:30:00Z",
                "episodes": [{"id": str(i), "wave": i // 2} for i in range(4)]}
    from helpers.maniguard_continuation_limits import epoch

    cutoff = epoch(protocol["physical_cutoff_utc"])
    attempts = [{"protocol_episode_id": "0", "status": "running"}]
    full = screen_capacity(protocol, attempts, now=cutoff - 6308, actual_stop_epoch=cutoff + 1800)
    assert full["remaining_waves"] == 2 and full["required_seconds"] == 6308 and full["fits"]
    assert not screen_capacity(protocol, attempts, now=cutoff - 6307,
                               actual_stop_epoch=cutoff + 1800)["fits"]
    attempts[0]["status"] = "invalid"
    with pytest.raises(ValueError, match="no automatic replacement"):
        screen_capacity(protocol, attempts, now=cutoff - 10000, actual_stop_epoch=cutoff + 1800)


@pytest.mark.parametrize("change", ["rate", "stale", "dead_watchdog", "wrong_instance"])
def test_resource_gate_rejects_stale_or_unauthorized_actual_state(change):
    from helpers.maniguard_continuation_limits import epoch

    now = epoch("2026-10-20T18:00:00Z")
    instance = {"id": 56777435, "machine_id": 54514, "actual_status": "running",
                "num_gpus": 2, "dph_total": .6126}
    clock = {"instance_id": 56777435, "status": "armed", "action": "stop_retaining_disk",
             "checked_epoch": now, "deadline_utc": "2026-10-20T20:00:00Z"}
    assert resource_contract(instance, clock, now=now, watchdog_live=True) > now
    if change == "rate":
        instance["dph_total"] = .66
    elif change == "stale":
        clock["checked_epoch"] -= 66
    elif change == "wrong_instance":
        instance["id"] += 1
    with pytest.raises(ValueError):
        resource_contract(instance, clock, now=now, watchdog_live=change != "dead_watchdog")


@pytest.mark.parametrize("stop", ["startup", "full", "counter", "none"])
def test_guardian_bounds_cpu_child_and_keeps_inherited_lease(tmp_path, stop):
    limits = {**DEFAULT_LIMITS, "initialization_wall_seconds": 1, "full_start_wall_seconds": 2}
    ledger = {"initialization_finished": stop != "startup", "initialization_physics_steps": 0}
    if stop == "counter":
        ledger["native_policy_requests_admitted"] = 251
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))
    # Even the normal child proves that the lease survives both Python execs.
    with (tmp_path / "lease").open("a") as lease:
        command = [sys.executable, "-c", ("import os,sys,time; os.fstat(int(sys.argv[1])); "
                   "time.sleep(float(sys.argv[2]))"), str(lease.fileno()), "0" if stop == "none" else "10"]
        started = time.time() - (2.1 if stop == "full" else 1.1 if stop == "startup" else 0)
        code = run_guarded(command, output=tmp_path, limits=limits, started_epoch=started,
                           cutoff_epoch=time.time() + 20, lease_fd=lease.fileno(), poll_seconds=.01)
    terminal = json.loads((tmp_path / "guardian-terminal.json").read_text())
    assert (code == 0) is (stop == "none")
    assert (terminal["technical_stop_reason"] is None) is (stop == "none")
    physical_pid = json.loads((tmp_path / "physical-process.json").read_text())["physical_pid"]
    assert process_identity(physical_pid) is None


def test_child_does_not_survive_abrupt_guardian_loss(tmp_path):
    # A dedicated small guardian process, never a running study/service.
    code = """
import json,sys,time
from pathlib import Path
from helpers.maniguard_continuation_guardian import run_guarded
from helpers.maniguard_continuation_limits import DEFAULT_LIMITS
p=Path(sys.argv[1])
with (p/'lease').open('a') as lease:
 run_guarded([sys.executable,'-c','import time; time.sleep(30)'], output=p,
             limits=DEFAULT_LIMITS, started_epoch=time.time(), cutoff_epoch=time.time()+60,
             lease_fd=lease.fileno(), poll_seconds=.02)
"""
    guardian = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)])
    path = tmp_path / "physical-process.json"
    try:
        deadline = time.time() + 5
        while not path.exists() and time.time() < deadline:
            time.sleep(.01)
        child = json.loads(path.read_text())["physical"]
        assert process_identity(child["pid"]) == child
        os.kill(guardian.pid, signal.SIGKILL)
        guardian.wait(timeout=5)
        deadline = time.time() + 5
        while process_identity(child["pid"]) == child and time.time() < deadline:
            time.sleep(.01)
        assert process_identity(child["pid"]) is None
    finally:
        if guardian.poll() is None:
            guardian.kill()
            guardian.wait()
