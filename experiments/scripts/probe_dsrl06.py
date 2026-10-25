"""Accounted model-only startup and frozen decoder-authority probes for DSRL06.

There is no inference retry. After sending, a missing response is reconciled against
the server's shared SQLite ledger and lossless artifact, including across resumes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_protocol import decoder_vectors, evaluate_authority
from helpers.dsrl06_server import QueryLedger, RAW_KEYS, raw_observation_arrays

PROMPT = (
    "Close the lid of the hinged jar holding the can of beans, then carry the closed jar "
    "into the green goal sphere on the table."
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def observation(arrays):
    return {
        RAW_KEYS[0]: arrays["observation_image_left"],
        RAW_KEYS[1]: arrays["observation_wrist_image"],
        RAW_KEYS[2]: arrays["observation_state"],
        "prompt": str(arrays["prompt"]),
    }


class WebsocketTransport:
    """One connection and one send per call, with no client inference retry."""

    def __init__(self, host, port, timeout):
        self.uri = f"ws://{host}:{int(port)}"
        self.timeout = timeout

    def __call__(self, request):
        from openpi_client import msgpack_numpy
        from websockets.sync.client import connect

        with connect(
            self.uri, compression=None, max_size=None, ping_interval=None, open_timeout=30
        ) as connection:
            metadata = msgpack_numpy.unpackb(connection.recv(timeout=self.timeout))
            if metadata.get("dsrl06_schema") != "dsrl06-query-v1":
                raise ValueError("The endpoint is not the frozen DSRL06 policy server")
            connection.send(msgpack_numpy.packb(request))
            payload = connection.recv(timeout=self.timeout)
            if not isinstance(payload, bytes):
                raise RuntimeError("The server reported an inference exception")
            return msgpack_numpy.unpackb(payload)


class PendingQuery(RuntimeError):
    """A sent request has an unresolved server outcome and must never be resent."""


class FailedQuery(RuntimeError):
    """An accounted request failed, invalidating this frozen gate."""


class ProbeRunner:
    def __init__(
        self,
        *,
        ledger,
        record_root,
        output,
        freeze,
        transport,
        poll_seconds=2400,
        poll_interval=2,
        clock=time.monotonic,
        sleep=time.sleep,
    ):
        self.ledger = ledger
        self.record_root = Path(record_root)
        self.output = Path(output)
        self.transport = transport
        self.poll_seconds = poll_seconds
        self.poll_interval = poll_interval
        self.clock, self.sleep = clock, sleep
        serialized = json.dumps(freeze, sort_keys=True, separators=(",", ":"))
        self.run_id = hashlib.sha256(serialized.encode()).hexdigest()
        self.job_id = f"{freeze['kind']}_{self.run_id[:24]}"
        self.state_path = self.ledger.path.parent / f"{self.job_id}.state.json"
        # The shared database directory supplies the same lock even with another output path.
        lock_path = self.ledger.path.parent / f"{self.job_id}.lock"
        self.lock = lock_path.open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state = {"schema": "dsrl06-probe-state-v1", "freeze": freeze, "requests": {}}
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if self.state.get("freeze") != freeze:
                raise ValueError("A resumed probe cannot change its frozen inputs")
        self.attempts_before = self.counts()["reserved"]
        self.sends_this_invocation = 0

    def close(self):
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()

    def packet_id(self, index):
        return f"dsrl06:{self.run_id}:{index:03d}"

    def row(self, packet_id):
        with self.ledger.connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM queries WHERE packet_id=?", (packet_id,)).fetchall()
        if len(rows) > 1:
            raise FailedQuery("Duplicate model reservations for one frozen packet")
        return dict(rows[0]) if rows else None

    def counts(self):
        with self.ledger.connect() as db:
            rows = db.execute(
                "SELECT status,model_returned_utc,client_confirmed_utc FROM queries WHERE job_id=?",
                (self.job_id,),
            ).fetchall()
        return {
            "reserved": len(rows),
            "model_returned": sum(row[1] is not None for row in rows),
            "client_confirmed": sum(row[2] is not None for row in rows),
            "failed": sum(row[0] == "failed" for row in rows),
            "sends_this_invocation": getattr(self, "sends_this_invocation", 0),
        }

    def audit(self):
        with self.ledger.connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute(
                "SELECT * FROM queries WHERE job_id=? ORDER BY query_index,id", (self.job_id,)
            ).fetchall()
        return {
            "state_path": str(self.state_path),
            "requests": self.state["requests"],
            "reservations": [dict(row) for row in rows],
        }

    def _recover(self, row, envelope, inputs, expected_noise):
        if row["job_id"] != self.job_id or row["query_index"] != envelope["query_index"]:
            raise FailedQuery("Ledger row does not identify this frozen request")
        if row["mode"] != envelope["mode"] or row["purpose"] != envelope["purpose"]:
            raise FailedQuery("Ledger row changed the frozen inference route")
        if row["status"] == "failed":
            raise FailedQuery(f"Frozen model request failed: {row['error_type']}")
        if row["status"] != "returned":
            return None
        path = Path(row["record_path"])
        if not path.is_file():
            return None
        expected_path = self.record_root / self.job_id / f"query_{row['id']:08d}.npz"
        if path.resolve() != expected_path.resolve():
            raise FailedQuery("Returned artifact lies outside its frozen capture path")
        arrays = load_npz(path)
        meta = json.loads(str(arrays["meta"]))
        for key in ("job_id", "query_index", "mode", "purpose", "packet_id"):
            if meta.get(key) != envelope[key]:
                raise FailedQuery(f"Returned artifact identity differs: {key}")
        if meta.get("status") != "returned" or meta.get("query_id") != row["id"]:
            raise FailedQuery("Returned artifact is incomplete")
        for key, value in raw_observation_arrays(inputs).items():
            if not np.array_equal(arrays[key], value):
                raise FailedQuery("Probe artifact changed a frozen observation")
        if envelope["mode"] == "native_replay" or envelope.get("native_draw_replay"):
            if not np.array_equal(arrays["rng_before"], envelope["rng_before"]):
                raise FailedQuery("Native replay changed the saved policy key")
        if expected_noise is not None and not np.array_equal(arrays["noise"], expected_noise):
            raise FailedQuery("Probe artifact changed the frozen latent tensor")
        actions = arrays["actions"]
        if actions.shape != (16, 8) or not np.isfinite(actions).all():
            raise FailedQuery("Probe returned an invalid decoded action chunk")
        return arrays, {
            "query_id": row["id"],
            "packet_id": envelope["packet_id"],
            "record_path": str(path),
            "sha256": sha256(path),
            "mode": envelope["mode"],
            "query_index": envelope["query_index"],
            "model_returned": row["model_returned_utc"] is not None,
            "rpc_confirmed": row["client_confirmed_utc"] is not None,
        }

    def query(
        self,
        index,
        inputs,
        *,
        mode,
        noise=None,
        rng_before=None,
        step=0,
        native_draw_replay=False,
    ):
        ceiling = self.state["freeze"]["query_ceiling"]
        if not 0 <= index < ceiling or self.counts()["reserved"] > ceiling:
            raise FailedQuery("Frozen probe query allowance exceeded")
        envelope = {
            "mode": mode,
            "job_id": self.job_id,
            "query_index": index,
            "purpose": "probe" if self.state["freeze"]["kind"] == "authority" else "startup",
            "actor_version": -1,
            "rng_seed": 0,
            "step": int(step),
            "attempt_id": "1",
            "packet_id": self.packet_id(index),
        }
        if noise is not None:
            envelope["noise"] = np.asarray(noise, dtype=np.float32)
        if native_draw_replay:
            if mode != "explicit" or self.state["freeze"]["kind"] != "authority":
                raise FailedQuery("Native noise passthrough is an explicit model probe only")
            envelope["native_draw_replay"] = True
        if rng_before is not None:
            envelope["rng_before"] = np.asarray(rng_before, dtype=np.uint32)
        key = str(index)
        request_state = self.state["requests"].get(key)
        row = self.row(envelope["packet_id"])
        reply = None
        # A saved send intent or existing server reservation always takes the recovery path.
        if request_state is None and row is None:
            self.state["requests"][key] = {"packet_id": envelope["packet_id"], "sent": True}
            write_json(self.state_path, self.state)
            self.sends_this_invocation += 1
            try:
                reply = self.transport({**inputs, "_dsrl06": envelope})
            except Exception as error:  # Lost response never authorizes another send.
                self.state["requests"][key]["transport_error"] = type(error).__name__
                write_json(self.state_path, self.state)
        deadline = self.clock() + self.poll_seconds
        while True:
            row = self.row(envelope["packet_id"])
            recovered = self._recover(row, envelope, inputs, noise) if row else None
            if recovered is not None:
                arrays, reference = recovered
                if reply is not None:
                    receipt = reply.get("_dsrl06", {})
                    if receipt.get("query_id") != row["id"]:
                        raise FailedQuery("RPC response does not identify the reserved request")
                    if not np.array_equal(
                        np.asarray(reply["actions"], np.float32), arrays["actions"]
                    ):
                        raise FailedQuery("RPC response differs from the server artifact")
                    self.ledger.confirm(row["id"])
                    reference["rpc_confirmed"] = True
                reference["recovered_without_rpc_response"] = reply is None
                self.state["requests"][key] = {"packet_id": envelope["packet_id"], **reference}
                write_json(self.state_path, self.state)
                return arrays, reference
            if self.clock() >= deadline:
                raise PendingQuery(
                    f"Unresolved packet {envelope['packet_id']}; poll this run again without resending"
                )
            self.sleep(self.poll_interval)


def validate_spec(spec_path, candidate_path):
    spec = json.loads(Path(spec_path).read_text())
    labels, expected = decoder_vectors()
    vectors = np.load(candidate_path, allow_pickle=False)
    required = {
        "snapshots": 8,
        "queries_per_snapshot": 16,
        "native_replays_per_snapshot": 2,
        "bounded_candidates_per_snapshot": 14,
        "total_query_ceiling": 128,
        "identity_atol": 1e-5,
        "identity_rtol": 1e-5,
        "identity_requires_exact_converted_prefix_gripper": True,
        "executed_prefix": 8,
        "native_gripper_open_threshold": 0.01,
        "minimum_raw_open": 0.02,
        "maximum_raw_closed": 0.0,
        "minimum_sensitive_snapshots": 2,
        "minimum_distinct_sensitive_tasks": 2,
        "candidate_labels": labels,
    }
    if any(spec.get(key) != value for key, value in required.items()):
        raise ValueError("Gate specification differs from the implemented frozen authority rule")
    if vectors.dtype.str != "<f4" or not np.array_equal(vectors, expected):
        raise ValueError("Candidate file differs from the fourteen frozen float32 vectors")
    if spec["candidate_npy_sha256"] != sha256(candidate_path):
        raise ValueError("Candidate file hash differs from the frozen specification")
    if spec["candidate_float32_bytes_sha256"] != hashlib.sha256(vectors.tobytes()).hexdigest():
        raise ValueError("Candidate tensor hash differs from the frozen specification")
    return vectors, labels


def authority_inputs(panel_path, spec_path, candidate_path):
    vectors, labels = validate_spec(spec_path, candidate_path)
    panel = json.loads(Path(panel_path).read_text())
    panel = panel["snapshots"] if isinstance(panel, dict) else panel
    if len(panel) != 8 or Counter(x["task"] for x in panel) != Counter({1: 2, 4: 2, 10: 2, 11: 2}):
        raise ValueError("Authority panel must contain two snapshots from each frozen pool task")
    if len({x["snapshot_id"] for x in panel}) != 8:
        raise ValueError("Authority snapshot identifiers must be distinct")
    snapshots = []
    for descriptor in panel:
        path = Path(descriptor["record_path"])
        if sha256(path) != descriptor["sha256"]:
            raise ValueError("Authority snapshot artifact hash changed")
        arrays = load_npz(path)
        meta = json.loads(str(arrays["meta"]))
        if meta["status"] != "returned" or meta["mode"] != "native":
            raise ValueError("Authority snapshots must be factual native returned queries")
        raw_observation_arrays(observation(arrays))
        if arrays["noise"].shape != (16, 32) or arrays["rng_before"].shape != (2,):
            raise ValueError("Authority snapshot has no saved native latent/key")
        if arrays["actions"].shape != (16, 8) or not np.isfinite(arrays["actions"]).all():
            raise ValueError("Authority snapshot has no valid factual decoded chunk")
        snapshots.append((descriptor, arrays, meta))
    freeze = {
        "kind": "authority",
        "query_ceiling": 128,
        "panel_sha256": sha256(panel_path),
        "spec_sha256": sha256(spec_path),
        "candidate_npy_sha256": sha256(candidate_path),
        "snapshots": [{key: x[key] for key in ("task", "snapshot_id", "sha256")} for x in panel],
    }
    return freeze, snapshots, vectors, labels


def run_authority(runner, snapshots, vectors, labels):
    cases, measured = [], []
    for slot, (descriptor, factual, meta) in enumerate(snapshots):
        replays, candidates, references = [], [], []
        inputs = observation(factual)
        for replay in range(2):
            arrays, reference = runner.query(
                slot * 16 + replay,
                inputs,
                mode="native_replay",
                rng_before=factual["rng_before"],
                noise=None,
                step=meta["step"],
            )
            if not np.array_equal(arrays["noise"], factual["noise"]):
                raise FailedQuery("Original-path replay reconstructed a different native draw")
            replays.append(arrays["actions"])
            references.append({"label": f"native_replay_{replay}", **reference})
        for index, vector in enumerate(vectors):
            arrays, reference = runner.query(
                slot * 16 + 2 + index,
                inputs,
                mode="explicit",
                noise=np.repeat(vector[None], 16, axis=0),
                step=meta["step"],
            )
            candidates.append(arrays["actions"])
            references.append({"label": labels[index], **reference})
        case = {
            "task": descriptor["task"],
            "snapshot_id": descriptor["snapshot_id"],
            "reference": factual["actions"].tolist(),
            "native_replays": np.stack(replays).tolist(),
            "candidates": np.stack(candidates).tolist(),
        }
        measured.append(case)
        cases.append({**case, "source": descriptor, "queries": references})
        write_json(
            runner.output,
            {
                "schema": "dsrl06-authority-result-v1",
                "status": "running",
                "run_id": runner.run_id,
                "freeze": runner.state["freeze"],
                "cases": cases,
                "query_counts": runner.counts(),
                "query_audit": runner.audit(),
            },
        )
    gate = evaluate_authority(measured)
    if runner.counts()["reserved"] != 128:
        raise FailedQuery("A completed authority gate must contain exactly 128 reservations")
    return {
        "schema": "dsrl06-authority-result-v1",
        "status": "completed",
        "run_id": runner.run_id,
        "freeze": runner.state["freeze"],
        "candidate_labels": labels,
        "cases": cases,
        "gate": gate,
        "query_counts": runner.counts(),
        "query_audit": runner.audit(),
    }


def startup_freeze(host, port, queries, startup_id):
    if not 1 <= queries <= 3:
        raise ValueError("Startup permits one to three charged synthetic native queries")
    return {
        "kind": "startup",
        "query_ceiling": queries,
        "host": host,
        "port": port,
        "startup_id": startup_id,
        "synthetic_seed": 0,
    }


def run_startup(runner):
    rng = np.random.default_rng(0)
    references = []
    for index in range(runner.state["freeze"]["query_ceiling"]):
        inputs = {
            RAW_KEYS[0]: rng.integers(256, size=(256, 256, 3), dtype=np.uint8),
            RAW_KEYS[1]: rng.integers(256, size=(256, 256, 3), dtype=np.uint8),
            RAW_KEYS[2]: np.zeros(8, dtype=np.float32),
            "prompt": PROMPT,
        }
        _, reference = runner.query(index, inputs, mode="native")
        references.append(reference)
    return {
        "schema": "dsrl06-startup-result-v1",
        "status": "completed",
        "valid": True,
        "run_id": runner.run_id,
        "freeze": runner.state["freeze"],
        "queries": references,
        "query_counts": runner.counts(),
        "query_audit": runner.audit(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("startup", "authority"):
        command = commands.add_parser(name)
        command.add_argument("--host", default="127.0.0.1")
        command.add_argument("--port", type=int, default=8000)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--query-ledger", type=Path, required=True)
        command.add_argument("--record-root", type=Path, required=True)
        command.add_argument("--timeout", type=float, default=2400)
        command.add_argument("--poll-seconds", type=float, default=2400)
    startup = commands.choices["startup"]
    startup.add_argument("--queries", type=int, default=3)
    startup.add_argument("--startup-id", required=True, help="Unique frozen server-start identity")
    authority = commands.choices["authority"]
    authority.add_argument("--panel", type=Path, required=True)
    authority.add_argument("--spec", type=Path, required=True)
    authority.add_argument("--candidates", type=Path, required=True)
    args = parser.parse_args()
    prepared = None
    if args.command == "startup":
        freeze = startup_freeze(args.host, args.port, args.queries, args.startup_id)
    else:
        freeze, *prepared = authority_inputs(args.panel, args.spec, args.candidates)
    runner = ProbeRunner(
        ledger=QueryLedger(args.query_ledger),
        record_root=args.record_root,
        output=args.output,
        freeze=freeze,
        transport=WebsocketTransport(args.host, args.port, args.timeout),
        poll_seconds=args.poll_seconds,
    )
    try:
        result = run_startup(runner) if prepared is None else run_authority(runner, *prepared)
        code = 0 if result.get("valid", result.get("gate", {}).get("valid", False)) else 1
    except (FailedQuery, PendingQuery) as error:
        pending = isinstance(error, PendingQuery)
        previous = json.loads(args.output.read_text()) if args.output.exists() else {}
        result = {
            **previous,
            "status": "pending" if pending else "invalid",
            "run_id": runner.run_id,
            "freeze": freeze,
            "gate": {"valid": False, "admit": False, "reason": str(error)},
            "query_counts": runner.counts(),
            "query_audit": runner.audit(),
        }
        code = 2 if pending else 1
    finally:
        runner.close()
    write_json(args.output, result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"cases", "freeze", "query_audit"}
            }
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
