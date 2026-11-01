"""NAV-only durable cumulative-counter journal, without model or simulator access.

A single writer fsyncs each changed snapshot before returning to the caller. Downstream
legacy snapshots are refreshed at action/reset boundaries and final/error completion.
A valid hash-chained prefix can reconstruct live.json on explicit stopped-worker resume;
this never adopts an unfinished episode as fitness or replays it.
"""

import hashlib
import json
import math
import numbers
import os
from pathlib import Path

SCHEMA = "lewm-progress-journal-v1"
BACKEND = "navigation-append-fsync-v1"


class JournalError(RuntimeError):
    pass


class JournalCorruption(JournalError):
    pass


class JournalCapExceeded(JournalError):
    pass


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _normalize(actions, resources, previous_actions, previous):
    if int(actions) != actions or actions < previous_actions:
        raise JournalError("Cumulative action count regressed or is invalid")
    actions = int(actions)
    merged = dict(previous)
    for key, value in (resources or {}).items():
        if isinstance(value, numbers.Integral):
            value = int(value)
        elif isinstance(value, numbers.Real):
            value = float(value)
        else:
            raise JournalError("Resource counters must be real scalars")
        if not math.isfinite(value) or value < previous.get(str(key), 0):
            raise JournalError("Cumulative resource counter regressed or is invalid")
        merged[str(key)] = value
    if "environment_actions" in (resources or {}) and merged["environment_actions"] != actions:
        raise JournalError("Action counter differs from the cumulative action argument")
    merged["environment_actions"] = actions
    return actions, merged


def _owner():
    pid = os.getpid()
    return {"pid": pid, "start_ticks": Path(f"/proc/{pid}/stat").read_text().split()[21]}


def _owner_live(owner):
    path = Path("/proc") / str(owner["pid"])
    try:
        stat = (path / "stat").read_text().split()
    except FileNotFoundError:
        return False
    return stat[2] != "Z" and stat[21] == str(owner["start_ticks"])


class ProgressJournal:
    def __init__(self, path, downstream, *, max_actions, metadata=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL, 0o600)
        self.downstream, self.max_actions = downstream, int(max_actions)
        if self.max_actions < 1:
            os.close(self.fd)
            raise JournalError("Positive episode action allowance required")
        self.actions, self.resources = 0, {"environment_actions": 0}
        self.last_requested = (self.actions, dict(self.resources))
        self.forwarded_actions = self.forwarded_resets = 0
        self.sequence, self.previous_digest = 0, None
        self.records = self.fsync_calls = self.downstream_forwards = 0
        self.closed = self.finished = self.failed = False
        self.write_error = None
        self.metadata = {
            **(metadata or {}),
            "backend": BACKEND,
            "max_actions": self.max_actions,
            "writer_source_sha256": digest(Path(__file__)),
            "owner": _owner(),
        }
        try:
            self._append("start", self.actions, self.resources, metadata=self.metadata)
        except BaseException:
            self.close()
            raise

    def _append(self, event, actions, resources, **extra):
        if self.closed or self.failed:
            raise JournalError("The journal is closed or failed; no later NN call is authorized")
        row = {
            "schema": SCHEMA,
            "seq": self.sequence,
            "event": event,
            "environment_actions": actions,
            "resources": resources,
            "previous_digest": self.previous_digest,
            **extra,
        }
        fingerprint = hashlib.sha256(_canonical(row)).hexdigest()
        encoded = _canonical({**row, "record_digest": fingerprint}) + b"\n"
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(self.fd, view)
                if written <= 0:
                    raise OSError("Incomplete journal write")
                view = view[written:]
            os.fsync(self.fd)
        except BaseException as exc:
            self.failed = True
            self.write_error = {"type": type(exc).__name__, "message": str(exc)}
            raise
        self.fsync_calls += 1
        self.records += 1
        self.sequence += 1
        self.previous_digest = fingerprint
        self.actions, self.resources = actions, dict(resources)

    def _forward(self, actions, resources):
        self.downstream(actions, dict(resources))
        self.downstream_forwards += 1
        self.forwarded_actions = actions
        self.forwarded_resets = int(resources.get("environment_resets", 0))

    def __call__(self, actions, resources=None):
        if self.finished or self.closed or self.failed:
            raise JournalError("A finished/failed journal cannot accept another progress event")
        actions, merged = _normalize(actions, resources, self.actions, self.resources)
        self.last_requested = (actions, merged)
        if actions != self.actions or merged != self.resources:
            self._append("snapshot", actions, merged, over_action_cap=actions > self.max_actions)
        if (
            actions != self.forwarded_actions
            or int(merged.get("environment_resets", 0)) != self.forwarded_resets
        ):
            self._forward(actions, merged)
        if actions > self.max_actions:
            raise JournalCapExceeded("Excess actions were journaled before rejecting the episode")

    def finish(self, *, status="complete"):
        if self.finished:
            return self.summary()
        if self.closed:
            raise JournalError("Cannot finalize a closed journal")
        actions, resources = self.last_requested
        if not self.failed:
            self._append("final", actions, resources, status=status)
        # Even a writer failure forwards its latest requested counters for conservative
        # failure accounting. No model/simulator call may follow a failed append.
        try:
            self._forward(actions, resources)
        finally:
            self.finished = True
        return self.summary()

    def close(self):
        if not self.closed:
            os.close(self.fd)
            self.closed = True

    def summary(self):
        return {
            "backend": BACKEND,
            "schema": SCHEMA,
            "path": str(self.path),
            "writer_source_sha256": self.metadata["writer_source_sha256"],
            "records": self.records,
            "fsync_calls": self.fsync_calls,
            "downstream_forwards": self.downstream_forwards,
            "last_record_digest": self.previous_digest,
            "failed": self.failed,
            "write_error": self.write_error,
            "finished": self.finished,
        }


def replay_progress_journal(path):
    """Validate every complete record; ignore only an unterminated final byte tail."""
    path = Path(path)
    sequence, previous_digest, last = 0, None, None
    metadata, tail_bytes = None, 0
    with path.open("rb") as stream:
        for encoded in stream:
            if not encoded.endswith(b"\n"):
                tail_bytes = len(encoded)
                break
            try:
                row = json.loads(encoded)
                fingerprint = row.pop("record_digest")
            except (ValueError, KeyError) as exc:
                raise JournalCorruption("A complete journal record is malformed") from exc
            if (
                row.get("schema") != SCHEMA
                or row.get("seq") != sequence
                or row.get("previous_digest") != previous_digest
                or hashlib.sha256(_canonical(row)).hexdigest() != fingerprint
            ):
                raise JournalCorruption("Journal sequence/hash chain differs")
            old_actions = 0 if last is None else last["environment_actions"]
            old_resources = {} if last is None else last["resources"]
            try:
                actions, resources = _normalize(
                    row["environment_actions"], row["resources"], old_actions, old_resources
                )
            except (KeyError, JournalError, TypeError, ValueError) as exc:
                raise JournalCorruption(
                    "Journal counters are not a valid monotonic prefix"
                ) from exc
            if resources != row["resources"]:
                raise JournalCorruption("A cumulative journal record dropped a counter")
            if sequence == 0:
                if row.get("event") != "start" or "metadata" not in row:
                    raise JournalCorruption("A journal must start with immutable metadata")
                metadata = row["metadata"]
            elif last["event"] == "final":
                raise JournalCorruption("A finalized journal cannot contain later events")
            last = {**row, "record_digest": fingerprint}
            previous_digest = fingerprint
            sequence += 1
    if last is None:
        raise JournalCorruption("No complete durable journal header is available")
    return {
        "path": str(path),
        "records": sequence,
        "last": last,
        "metadata": metadata,
        "truncated_tail_bytes": tail_bytes,
        "sha256": digest(path),
    }


def recover_progress_journals(output, *, require_stopped=True):
    """Rebuild derived live counters before explicit legacy-harness reconciliation.

    This does not create result.json, mark jobs complete or authorize episode replay.
    The unchanged harness still rejects started-only/failed jobs after recovery.
    """
    output = Path(output)
    plans = []
    for path in sorted((output / "jobs").glob("*/progress-journal.jsonl")):
        replay = replay_progress_journal(path)
        if require_stopped and _owner_live(replay["metadata"]["owner"]):
            raise JournalError("Refuse recovery while the recorded writer process remains live")
        if replay["metadata"].get("job_id") != path.parent.name:
            raise JournalCorruption("Journal job identity differs from its folder")
        expected_started = replay["metadata"].get("started_sha256")
        if expected_started is not None and (
            not (path.parent / "started.json").exists()
            or digest(path.parent / "started.json") != expected_started
        ):
            raise JournalCorruption("Immutable started-job identity differs from its journal")
        result = path.parent / "result.json"
        if result.exists():
            row = json.loads(result.read_text())
            if int(row["environment_actions"]) != replay["last"]["environment_actions"]:
                raise JournalCorruption("Final episode actions differ from its journal")
            for key, value in replay["last"]["resources"].items():
                if row["resources"].get(key, 0) != value:
                    raise JournalCorruption("Final episode resources differ from its journal")
        plans.append((path.parent / "live.json", replay))
    recovered = []
    for live_path, replay in plans:
        last = replay["last"]
        value = {
            "environment_actions": last["environment_actions"],
            "resources": last["resources"],
            "finalized": last["event"] == "final",
            "journal_recovery": {
                "journal_sha256": replay["sha256"],
                "records": replay["records"],
                "truncated_tail_bytes": replay["truncated_tail_bytes"],
                "does_not_authorize_episode_replay": True,
            },
        }
        temporary = live_path.with_suffix(".json.recovery-part")
        temporary.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        temporary.replace(live_path)
        recovered.append({"job_id": live_path.parent.name, **value["journal_recovery"]})
    return {
        "recovered_jobs": recovered,
        "new_nn_calls": 0,
        "new_simulator_actions": 0,
        "episode_outcomes_adopted": 0,
    }
