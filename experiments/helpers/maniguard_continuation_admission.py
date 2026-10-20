"""Durable four-start admission, original-study clearance and service exclusion."""

import fcntl
import hashlib
import json
import os
from pathlib import Path

from helpers.maniguard_continuation_guardian import atomic_record
from helpers.maniguard_continuation_limits import screen_capacity, validate_limits
from helpers.maniguard_continuation_protocol import validate_plan
from helpers.maniguard_latent_queue import process_identity

ROOT = Path(__file__).resolve().parents[2]
SERVICE_LEASE_DIRECTORY = ROOT / "runs/maniguard-policy-service-locks"


def policy_lease_path(protocol, worker):
    if Path(protocol["service_lease_directory"]).resolve() != SERVICE_LEASE_DIRECTORY.resolve():
        raise ValueError("The physical policy service has one canonical lease directory")
    return SERVICE_LEASE_DIRECTORY / f"{protocol['workers'][str(worker)]['port']}.lock"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_original_clearance(protocol, *, identity_reader=process_identity):
    """A successful design plus all physical restoration is required, not a flag."""
    original = Path(protocol["original_run"])
    raw = (original / "queue.json").read_bytes()
    queue = json.loads(raw)
    full = [a for a in queue["attempts"] if not a["engineering"]]
    valid = [a for a in full if a["status"] == "completed"]
    if (len(valid) != 60 or len(full) > 64
            or any(a["status"] not in ("completed", "invalid") for a in full)):
        raise ValueError("The original 60-episode comparison is not complete")
    for attempt in queue["attempts"]:
        identity = attempt.get("process_identity")
        if identity is not None and identity_reader(identity["pid"]) == identity:
            raise ValueError("An original physical process still owns a policy service")
    design_path = original / "analysis/design-verification.json"
    design = json.loads(design_path.read_text())
    if (design.get("physical_design_complete") is not True or design.get("verified_full_jobs") != 60
            or design.get("errors") != [] or design.get("pending") != []
            or design.get("protocol_sha256") != protocol["original_protocol_sha256"]
            or design.get("queue_sha256") != hashlib.sha256(raw).hexdigest()):
        raise ValueError("Original independent design verification is incomplete or stale")
    clearance_path = original / "original-comparison-clearance.json"
    clearance = json.loads(clearance_path.read_text())
    recovery_path = Path(protocol["original_physical_recovery_proof"])
    recovery = json.loads(recovery_path.read_text())
    names = {a["id"] for a in valid}
    if (clearance.get("protocol_sha256") != protocol["original_protocol_sha256"]
            or clearance.get("queue_sha256") != hashlib.sha256(raw).hexdigest()
            or clearance.get("design_verification_sha256") != file_hash(design_path)
            or clearance.get("physical_recovery_proof_sha256") != file_hash(recovery_path)
            or recovery.get("physical_receipts_reverified") is not True
            or recovery.get("all_archives_redownloaded_and_sha256_checked") is not True
            or len(recovery.get("episodes", [])) != 60
            or {row["attempt"] for row in recovery["episodes"]} != names):
        raise ValueError("Original remote physical preservation is incomplete or misbound")
    for row in recovery["episodes"]:
        receipt = json.loads((original / "episodes" / row["attempt"] / "verified-receipt.json").read_text())
        if (receipt.get("valid") is not True or receipt.get("full_episode") is not True
                or row["controller_sha256"] != receipt["controller_sha256"]
                or row["policy_controls"] != receipt["physical"]["steps"]
                or row["policy_queries"] != receipt["physical"]["ledger"]["policy_queries"]):
            raise ValueError("Original restored episode differs from its verified physical receipt")
    index = json.loads((original / "backups/index.json").read_text())
    backed = {name for stage in index["stages"].values() for name in stage["episodes"]}
    terminal = {a["id"] for a in queue["attempts"] if a["status"] in ("completed", "invalid")}
    if not terminal <= backed:
        raise ValueError("Some original starts, including failures, have no preservation stage")
    return {"clearance_sha256": file_hash(clearance_path),
            "design_sha256": file_hash(design_path), "recovery_sha256": file_hash(recovery_path)}


def acquire_service_lease(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle


def assert_service_lease(descriptor, path):
    expected, actual = Path(path).stat(), os.fstat(descriptor)
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise ValueError("Inherited lease is not the declared policy-service file")
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


class ScreenLedger:
    """A reservation is charged before file preparation or process creation."""

    def __init__(self, run, protocol_path):
        self.run, self.protocol_path = Path(run).resolve(), Path(protocol_path).resolve()
        self.protocol_bytes = self.protocol_path.read_bytes()
        self.protocol = json.loads(self.protocol_bytes)
        if self.run != Path(self.protocol["run_directory"]).resolve():
            raise ValueError("The frozen screen has one canonical run and start ledger")
        validate_plan(self.protocol)
        validate_limits(self.protocol["runtime_limits"])
        if self.protocol["status"] != "prospectively_frozen":
            raise ValueError("The CPU draft cannot dispatch physical work")
        if self.run == Path(self.protocol["original_run"]).resolve():
            raise ValueError("A new screen cannot reuse the original run or its start allowance")
        self.protocol_sha256 = hashlib.sha256(self.protocol_bytes).hexdigest()
        self.run.mkdir(parents=True, exist_ok=True)
        self.lock = acquire_service_lease(self.run / "coordinator.lock")
        self.path = self.run / "screen.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "schema": "maniguard-continuation-four-start-ledger-v1",
            "protocol_sha256": self.protocol_sha256, "attempts": [],
        }
        if self.state["protocol_sha256"] != self.protocol_sha256:
            raise ValueError("A reserved screen protocol cannot change")

    def save(self):
        atomic_record(self.path, self.state)

    def close(self):
        self.lock.close()

    def reserve_wave(self, wave, *, now, actual_stop_epoch):
        attempts = self.state["attempts"]
        if any(a["status"] not in ("completed", "invalid") for a in attempts):
            raise ValueError("Reconcile all surviving reservations before admitting another wave")
        capacity = screen_capacity(self.protocol, attempts, now=now, actual_stop_epoch=actual_stop_epoch)
        if not capacity["fits"]:
            raise ValueError("Complete remaining screen and preservation do not fit actual time")
        rows = [row for row in self.protocol["episodes"] if row["wave"] == wave]
        earlier = {row["id"] for row in self.protocol["episodes"] if row["wave"] < wave}
        done = {a["protocol_episode_id"] for a in attempts if a["status"] == "completed"}
        if not earlier <= done or len(rows) != 2 or len({r["worker"] for r in rows}) != 2:
            raise ValueError("Respect the declared two-worker wave and previous complete wave")
        used = {a["protocol_episode_id"] for a in attempts}
        if len(attempts) + len(rows) > 4 or any(row["id"] in used for row in rows):
            raise ValueError("Every initial reference has one charged start and no replacement")
        added = []
        for row in rows:
            output = self.run / "episodes" / (row["id"] + "-a0")
            added.append({"id": row["id"] + "-a0", "protocol_episode_id": row["id"],
                          "wave": wave, "worker": row["worker"], "condition": row["condition"],
                          "status": "reserved", "started_epoch": now, "output": str(output),
                          "protocol_sha256": self.protocol_sha256,
                          "deadline_epoch": min(now + self.protocol["runtime_limits"]["full_start_wall_seconds"],
                                                capacity["physical_cutoff_epoch"])})
        attempts.extend(added)
        self.save()  # fsync before source preparation or Popen can fail.
        return added
