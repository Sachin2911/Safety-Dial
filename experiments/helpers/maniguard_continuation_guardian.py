"""Keep per-start wall/counter limits independent of the screen coordinator."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from helpers.maniguard_continuation_limits import counter_violations, validate_limits
from helpers.maniguard_latent_queue import process_identity


def atomic_record(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


# Install in a fresh interpreter, avoiding preexec_fn with NumPy-loaded threads.
# https://docs.python.org/3.10/library/subprocess.html#popen-constructor
# https://man7.org/linux/man-pages/man2/PR_SET_PDEATHSIG.2const.html
PARENT_DEATH_EXEC = """
import ctypes, os, signal, sys
library = ctypes.CDLL(None, use_errno=True)
if library.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), 'Unable to arm guardian-death signal')
if os.getppid() != int(sys.argv[1]):
    os.kill(os.getpid(), signal.SIGKILL)
os.execv(sys.argv[2], sys.argv[2:])
"""


def _terminate_child(child):
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        child.wait()
        return
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()


def run_guarded(command, *, output, limits, started_epoch, cutoff_epoch, lease_fd, poll_seconds=1.):
    """Called only after a durable screen reservation and inherited service lease.

    The production command is fixed by the entry point. The injected command in
    focused tests is a small CPU child, never a simulator or model fixture.
    """
    validate_limits(limits)
    if not 0 < poll_seconds <= 1:
        raise ValueError("Guardian counter polling may not exceed one second")
    output = Path(output)
    parent = os.getpid()
    identity = process_identity(parent)
    atomic_record(output / "guardian.json", {"guardian": identity, "started_epoch": started_epoch})
    child = None
    reason = None
    deadline = min(started_epoch + limits["full_start_wall_seconds"], cutoff_epoch)
    try:
        child = subprocess.Popen([sys.executable, "-c", PARENT_DEATH_EXEC, str(parent), *command],
                                 start_new_session=True, pass_fds=(lease_fd,))
        atomic_record(output / "physical-process.json", {
            "guardian": identity, "physical": process_identity(child.pid), "physical_pid": child.pid,
            "spawned_epoch": time.time(), "command": command,
        })
        while child.poll() is None:
            now = time.time()
            record = None
            try:
                record = json.loads((output / "ledger.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                pass  # A missing/transient ledger is still bounded by startup wall time.
            if now >= deadline:
                reason = "full_start_wall_deadline"
            elif (not record or not record.get("initialization_finished", False)) and (
                    now >= started_epoch + limits["initialization_wall_seconds"]):
                reason = "initialization_wall_deadline"
            elif record and (counter_violations(record, limits)
                             or record.get("budget_violation_reasons")):
                reason = "recorded_technical_budget_violation"
            if reason is not None:
                _terminate_child(child)
                break
            time.sleep(poll_seconds)
        code = child.wait()
    except BaseException:
        if child is not None:
            _terminate_child(child)
        raise
    finally:
        if child is not None:
            atomic_record(output / "guardian-terminal.json", {
                "guardian": identity, "exit_code": child.poll(), "finished_epoch": time.time(),
                "started_epoch": started_epoch, "deadline_epoch": deadline,
                "technical_stop_reason": reason,
                "scope": "All starts remain charged. Forced-stop counters are observed lower bounds; "
                         "work after the last ledger and physics before attachment may be unknown.",
            })
    return code
