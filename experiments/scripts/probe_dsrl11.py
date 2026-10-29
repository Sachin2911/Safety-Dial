"""Accounted model-only probes for task 11: server startup and the A2 decoder checks.

The task-6 probe runner is reused unchanged for sending, durable reservation and
recovery without resending; only the ledger (fresh task-11 ceiling) and the residual
probe envelope are new. Run with the OpenPI interpreter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "scripts"))
from helpers.dsrl07_protocol import array_difference, decoder_comparison
from helpers.dsrl11_protocol import derive_seed, uniform_residual
from helpers.dsrl11_server import Dsrl11QueryLedger
from probe_dsrl06 import (
    FailedQuery,
    PendingQuery,
    ProbeRunner,
    WebsocketTransport,
    load_npz,
    observation,
    run_startup,
    startup_freeze,
    write_json,
)

INFLUENCE_VECTORS = 4
NATIVE_REPEATS = 2


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def influence_vectors() -> list[np.ndarray]:
    """Four fixed seeded residual vectors and their negatives, in that order."""
    vectors = [uniform_residual(derive_seed("a2-influence", k), 0) for k in range(4)]
    return [*vectors, *(-v for v in vectors)]


class ResidualProbeRunner(ProbeRunner):
    """Adds the residual probe envelope; send, reservation and recovery are inherited."""

    def residual_query(self, index, inputs, *, residual, rng_before):
        residual = np.asarray(residual, np.float32)
        if residual.shape != (32,) or not np.isfinite(residual).all():
            raise FailedQuery("A residual probe requires one finite 32-vector")
        ceiling = self.state["freeze"]["query_ceiling"]
        if not 0 <= index < ceiling or self.counts()["reserved"] > ceiling:
            raise FailedQuery("Frozen probe query allowance exceeded")
        envelope = {
            "mode": "residual",
            "job_id": self.job_id,
            "query_index": index,
            "purpose": "probe",
            "residual_source": "explicit",
            "residual_u": residual,
            "rng_before": np.asarray(rng_before, dtype=np.uint32),
            "actor_version": -1,
            "step": 0,
            "attempt_id": "1",
            "packet_id": self.packet_id(index),
        }
        key = str(index)
        row = self.row(envelope["packet_id"])
        reply = None
        if self.state["requests"].get(key) is None and row is None:
            self.state["requests"][key] = {"packet_id": envelope["packet_id"], "sent": True}
            write_json(self.state_path, self.state)
            self.sends_this_invocation += 1
            try:
                reply = self.transport({**inputs, "_dsrl06": envelope})
            except Exception as error:  # A lost response never authorizes another send.
                self.state["requests"][key]["transport_error"] = type(error).__name__
                write_json(self.state_path, self.state)
        deadline = self.clock() + self.poll_seconds
        while True:
            row = self.row(envelope["packet_id"])
            recovered = self._recover(row, envelope, inputs, None) if row else None
            if recovered is not None:
                arrays, reference = recovered
                if not np.array_equal(arrays["residual_u"], residual) or not np.array_equal(
                    arrays["rng_before"], envelope["rng_before"]
                ):
                    raise FailedQuery("Residual probe artifact changed its frozen inputs")
                if reply is not None:
                    receipt = reply.get("_dsrl06", {})
                    if receipt.get("query_id") != row["id"] or not np.array_equal(
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
                    f"Unresolved packet {envelope['packet_id']}; poll again without resending"
                )
            self.sleep(self.poll_interval)


def decoder_inputs(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "dsrl11-pool-inputs-v1" or len(manifest["snapshots"]) != 8:
        raise ValueError("A2 uses exactly the eight frozen saved pool observations")
    for row in manifest["snapshots"]:
        if digest(manifest_path.parent / row["file"]) != row["sha256"]:
            raise ValueError("A saved pool observation changed")
    return manifest, digest(manifest_path)


def run_decoder(runner, manifest, manifest_path, lane):
    result = {
        "schema": "dsrl11-decoder-diagnosis-v1",
        "lane": lane,
        "freeze": runner.state["freeze"],
        "cases": [],
    }
    vectors = influence_vectors() if lane == 0 else []
    index = 0
    for row in manifest["snapshots"]:
        factual = load_npz(manifest_path.parent / row["file"])
        obs = observation(factual)
        replays, references = [], []
        for _ in range(NATIVE_REPEATS):
            arrays, reference = runner.query(
                index, obs, mode="native_replay", rng_before=factual["rng_before"]
            )
            index += 1
            replays.append(arrays)
            references.append(reference)
        zero, reference = runner.residual_query(
            index, obs, residual=np.zeros(32, np.float32), rng_before=factual["rng_before"]
        )
        index += 1
        references.append(reference)
        case = {
            "snapshot_id": row["snapshot_id"],
            "task": row["task"],
            "native_repeat": decoder_comparison(replays[0]["actions"], replays[1]["actions"]),
            # Same key, same process: the residual route must reconstruct the native draw.
            "epsilon_equals_native_draw": bool(
                np.array_equal(zero["epsilon"], replays[0]["noise"])
            ),
            "zero_noise_equals_epsilon": bool(np.array_equal(zero["noise"], zero["epsilon"])),
            "zero_residual_vs_native": decoder_comparison(replays[0]["actions"], zero["actions"]),
            "historical_draw_reproduced": bool(
                np.array_equal(replays[0]["noise"], factual["noise"])
            ),
            "historical_actions": array_difference(
                factual["actions"], replays[0]["actions"], atol=1e-5, rtol=1e-5
            ),
            "native_actions": replays[0]["actions"].tolist(),
            "queries": references,
        }
        case["zero_residual_pass"] = bool(
            case["epsilon_equals_native_draw"]
            and case["zero_noise_equals_epsilon"]
            and case["zero_residual_vs_native"]["pass"]
        )
        if vectors:
            chunks = []
            for vector in vectors:
                arrays, reference = runner.residual_query(
                    index, obs, residual=vector, rng_before=factual["rng_before"]
                )
                index += 1
                case["queries"].append(reference)
                chunks.append(arrays["actions"])
            chunks = np.stack(chunks)
            arm = np.sqrt(np.mean((chunks[:, :8, :7] - zero["actions"][:8, :7]) ** 2, axis=(1, 2)))
            native_grip = zero["actions"][:8, 7] > 0.01
            case["influence"] = {
                "finite": bool(np.isfinite(chunks).all()),
                "arm_rms_from_zero_residual": arm.tolist(),
                "arm_influence": bool(np.max(arm) > 1e-5),
                "gripper_prefix_changes_descriptive": [
                    int(np.sum((c[:8, 7] > 0.01) != native_grip)) for c in chunks
                ],
            }
        result["cases"].append(case)
        result["query_counts"] = runner.counts()
        write_json(runner.output, result)
    cases = result["cases"]
    result["same_process_native_repeat_pass"] = all(c["native_repeat"]["pass"] for c in cases)
    result["zero_residual_pass"] = all(c["zero_residual_pass"] for c in cases)
    result["historical_draw_reproduced_all"] = all(c["historical_draw_reproduced"] for c in cases)
    if vectors:
        result["influence_finite"] = all(c["influence"]["finite"] for c in cases)
        result["influence_present"] = any(c["influence"]["arm_influence"] for c in cases)
    result["valid"] = bool(
        result["same_process_native_repeat_pass"]
        and result["zero_residual_pass"]
        and (not vectors or (result["influence_finite"] and result["influence_present"]))
    )
    result["status"] = "completed"
    result["query_counts"] = runner.counts()
    result["query_audit"] = runner.audit()
    return result


def run_parity_decode(runner, panel):
    """Decode GPU- and CPU-sampled residuals with one native key; require agreement."""
    cases, index = [], 0
    for case in panel["cases"]:
        inputs = observation(load_npz(case["record_path"]))
        decoded = []
        for name in ("gpu_residual", "cpu_residual"):
            arrays, reference = runner.residual_query(
                index,
                inputs,
                residual=np.asarray(case[name], np.float32),
                rng_before=np.asarray(case["rng_before"], np.uint32),
            )
            index += 1
            decoded.append(arrays["actions"])
        cases.append(
            {
                "record_path": case["record_path"],
                "residual_maximum_difference": float(
                    np.max(np.abs(np.subtract(case["gpu_residual"], case["cpu_residual"])))
                ),
                **decoder_comparison(decoded[0], decoded[1]),
            }
        )
        write_json(runner.output, {"schema": "dsrl11-parity-decode-v1", "cases": cases})
    return {
        "schema": "dsrl11-parity-decode-v1",
        "status": "completed",
        "cases": cases,
        "cases_passed": sum(case["pass"] for case in cases),
        "valid": bool(cases) and all(case["pass"] for case in cases),
        "query_counts": runner.counts(),
        "query_audit": runner.audit(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("startup", "decoder", "parity-decode"):
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
    startup.add_argument("--startup-id", required=True)
    decoder = commands.choices["decoder"]
    decoder.add_argument("--inputs", type=Path, required=True)
    decoder.add_argument("--lane", type=int, choices=range(4), required=True)
    decoder.add_argument("--operation", required=True)
    commands.choices["parity-decode"].add_argument("--panel", type=Path, required=True)
    args = parser.parse_args()
    panel = None
    if args.command == "startup":
        freeze = startup_freeze(args.host, args.port, args.queries, args.startup_id)
        freeze["experiment"] = "dsrl11"
    elif args.command == "parity-decode":
        panel = json.loads(args.panel.read_text())
        if panel.get("schema") != "dsrl11-parity-panel-v1" or not panel["cases"]:
            raise ValueError("Parity decoding needs the frozen engineering parity panel")
        freeze = {
            "kind": "authority",
            "experiment": "dsrl11",
            "operation": "parity-decode",
            "panel_sha256": digest(args.panel),
            "query_ceiling": 2 * len(panel["cases"]),
            "protocol": "gpu-versus-cpu-residual-same-native-key-v1",
        }
    else:
        manifest, manifest_hash = decoder_inputs(args.inputs)
        per_snapshot = NATIVE_REPEATS + 1 + (2 * INFLUENCE_VECTORS if args.lane == 0 else 0)
        freeze = {
            "kind": "authority",
            "experiment": "dsrl11",
            "operation": args.operation,
            "lane": args.lane,
            "input_manifest_sha256": manifest_hash,
            "query_ceiling": 8 * per_snapshot,
            "protocol": "two-native-replays-zero-residual-and-signed-influence-v1",
        }
    runner = ResidualProbeRunner(
        ledger=Dsrl11QueryLedger(args.query_ledger),
        record_root=args.record_root,
        output=args.output,
        freeze=freeze,
        transport=WebsocketTransport(args.host, args.port, args.timeout),
        poll_seconds=args.poll_seconds,
    )
    try:
        if args.command == "startup":
            result = run_startup(runner)
        elif args.command == "parity-decode":
            result = run_parity_decode(runner, panel)
        else:
            result = run_decoder(runner, manifest, args.inputs, args.lane)
        code = 0 if result.get("valid") else 1
    except (FailedQuery, PendingQuery) as error:
        pending = isinstance(error, PendingQuery)
        previous = json.loads(args.output.read_text()) if args.output.exists() else {}
        result = {
            **previous,
            "status": "pending" if pending else "invalid",
            "valid": False,
            "run_id": runner.run_id,
            "freeze": freeze,
            "reason": str(error),
            "query_counts": runner.counts(),
            "query_audit": runner.audit(),
        }
        code = 2 if pending else 1
    finally:
        runner.close()
    write_json(args.output, result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in {"cases", "freeze", "query_audit"}})
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
