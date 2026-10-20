"""Separate, durable admission accounting for the deferred model-assisted study."""

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path


class WorkLedger:
    """Charge before work, including failures; output directories do not reset caps."""

    def __init__(self, path, limits, contract_sha256):
        if not limits or any(type(v) is not int or v < 0 for v in limits.values()):
            raise ValueError("Every work limit must be an explicit nonnegative integer")
        if not isinstance(contract_sha256, str) or not contract_sha256:
            raise ValueError("A frozen contract hash is required")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header = {"type": "contract", "schema": "maniguard-model-work-v1",
                       "limits": dict(limits), "contract_sha256": contract_sha256}
        with self._locked() as stream:
            rows = self._read(stream)
            if not rows:
                self._append(stream, self.header)
            elif rows[0] != self.header:
                raise ValueError("The canonical work ledger belongs to another contract")

    @contextmanager
    def _locked(self):
        with self.path.open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield stream
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _read(stream):
        stream.seek(0)
        text = stream.read()
        if text and not text.endswith("\n"):
            raise ValueError("Incomplete ledger tail requires explicit reconciliation")
        return [json.loads(line) for line in text.splitlines()]

    @staticmethod
    def _append(stream, row):
        stream.seek(0, os.SEEK_END)
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

    def admit(self, request_id, kind, units):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("A stable request identifier is required")
        if kind not in self.header["limits"] or type(units) is not int or units <= 0:
            raise ValueError("Unknown work kind or invalid units")
        with self._locked() as stream:
            rows = self._read(stream)
            if not rows or rows[0] != self.header:
                raise ValueError("Canonical ledger contract changed")
            admissions = [r for r in rows if r["type"] == "admission"]
            if any(r["id"] == request_id for r in admissions):
                raise ValueError("Request is already charged; reconcile instead of replaying")
            used = sum(r["units"] for r in admissions if r["kind"] == kind)
            if used + units > self.header["limits"][kind]:
                raise RuntimeError("Work allowance exhausted: " + kind)
            self._append(stream, {"type": "admission", "id": request_id, "kind": kind,
                                  "units": units, "started_epoch": time.time()})

    def finish(self, request_id, status):
        if status not in ("returned", "failed", "abandoned"):
            raise ValueError("Unknown terminal work status")
        with self._locked() as stream:
            rows = self._read(stream)
            if rows[0] != self.header:
                raise ValueError("Canonical ledger contract changed")
            if not any(r.get("id") == request_id and r["type"] == "admission" for r in rows):
                raise ValueError("Cannot finish work without its prior admission")
            if any(r.get("id") == request_id and r["type"] == "terminal" for r in rows):
                raise ValueError("Work already has a terminal record")
            self._append(stream, {"type": "terminal", "id": request_id,
                                  "status": status, "finished_epoch": time.time()})

    def summary(self):
        with self._locked() as stream:
            rows = self._read(stream)
        if not rows or rows[0] != self.header:
            raise ValueError("Canonical ledger contract changed")
        admissions = [r for r in rows if r["type"] == "admission"]
        terminals = {r["id"]: r["status"] for r in rows if r["type"] == "terminal"}
        used = {kind: sum(r["units"] for r in admissions if r["kind"] == kind)
                for kind in self.header["limits"]}
        return {"contract_sha256": self.header["contract_sha256"],
                "limits": self.header["limits"], "used": used,
                "remaining": {k: self.header["limits"][k] - v for k, v in used.items()},
                "unresolved": [r["id"] for r in admissions if r["id"] not in terminals],
                "terminal_status": terminals,
                "scope": "Admissions remain charged after failure or interruption."}
