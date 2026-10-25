"""Counted task-7 numerical diagnosis on immutable saved pool observations."""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.dsrl06_learner import LearnerBridge, LearnerConfig
from helpers.dsrl06_protocol import decoder_vectors
from helpers.dsrl06_server import QueryLedger
from helpers.dsrl07_protocol import array_difference, candidate_summary, decoder_comparison
from probe_dsrl06 import ProbeRunner, WebsocketTransport, load_npz, observation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inputs():
    manifest = ROOT / "outputs/dsrl07/inputs/manifest.json"
    data = json.loads(manifest.read_text())
    for row in [*data["snapshots"], data["profile"]]:
        if digest(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("Saved diagnostic input changed")
    return data, digest(manifest)


def decoder(args):
    data, input_hash = inputs()
    output = ROOT / "outputs/dsrl07" / f"decoder-{args.operation}-lane{args.lane}.json"
    freeze = {
        "kind": "authority",
        "experiment": "dsrl07",
        "operation": args.operation,
        "lane": args.lane,
        "input_manifest_sha256": input_hash,
        "query_ceiling": 144 if args.lane == 0 else 24,
        "protocol": "same-process-three-native-repeats-plus-explicit-draw-and-tiled-probes-v1",
    }
    runner = ProbeRunner(
        ledger=QueryLedger(ROOT / "outputs/dsrl07/queries.sqlite"),
        record_root=ROOT / "outputs/dsrl07/queries",
        output=output,
        freeze=freeze,
        transport=WebsocketTransport("127.0.0.1", 8000 + args.lane, 900),
        poll_seconds=900,
    )
    result = {
        "schema": "dsrl07-decoder-diagnosis-v1",
        "freeze": freeze,
        "started_utc": utc(),
        "cases": [],
    }
    labels, vectors = decoder_vectors()
    try:
        index = 0
        for row in data["snapshots"]:
            factual = load_npz(ROOT / row["path"])
            obs = observation(factual)
            replays, references = [], []
            for _ in range(3):
                arrays, ref = runner.query(
                    index, obs, mode="native_replay", rng_before=factual["rng_before"]
                )
                index += 1
                if not np.array_equal(arrays["noise"], factual["noise"]):
                    raise ValueError("Native draw reconstruction differs")
                replays.append(arrays["actions"])
                references.append(ref)
            case = {
                "snapshot_id": row["snapshot_id"],
                "task": row["task"],
                "native_replays": np.stack(replays).tolist(),
                "same_process": [decoder_comparison(replays[0], r) for r in replays[1:]],
                "historical_factual_comparison": decoder_comparison(factual["actions"], replays[0]),
                "queries": references,
            }
            if args.lane == 0:
                arrays, ref = runner.query(
                    index,
                    obs,
                    mode="explicit",
                    noise=factual["noise"],
                    rng_before=factual["rng_before"],
                    native_draw_replay=True,
                )
                index += 1
                case["native_draw_passthrough"] = decoder_comparison(replays[0], arrays["actions"])
                case["queries"].append(ref)
                candidates = []
                for vector in vectors:
                    arrays, ref = runner.query(
                        index, obs, mode="explicit", noise=np.repeat(vector[None], 16, axis=0)
                    )
                    index += 1
                    candidates.append(arrays["actions"])
                    case["queries"].append(ref)
                case["candidate_labels"] = labels
                case["candidates"] = np.stack(candidates).tolist()
                case["influence"] = candidate_summary(candidates)
            result["cases"].append(case)
            result["query_counts"] = runner.counts()
            atomic_json(output, result)
        result["same_process_pass"] = all(
            check["pass"] for case in result["cases"] for check in case["same_process"]
        )
        if args.lane == 0:
            result["native_draw_passthrough_pass"] = all(
                case["native_draw_passthrough"]["pass"] for case in result["cases"]
            )
            result["influence_present"] = any(
                c["influence"]["arm_influence"] or c["influence"]["gripper_sensitive"]
                for c in result["cases"]
            )
            sensitive = [c for c in result["cases"] if c["influence"]["gripper_sensitive"]]
            result["descriptive_gripper_sensitive_inputs"] = len(sensitive)
            result["descriptive_gripper_sensitive_tasks"] = sorted({c["task"] for c in sensitive})
        result["query_audit"] = runner.audit()
        result["completed_utc"] = utc()
        atomic_json(output, result)
    finally:
        runner.close()
    print(json.dumps({k: v for k, v in result.items() if k not in {"cases", "query_audit"}}))


def tree_difference(jax, left, right):
    ls, lt = jax.tree_util.tree_flatten_with_path(left)
    rs, rt = jax.tree_util.tree_flatten_with_path(right)
    if lt != rt:
        raise ValueError("State structures differ")
    rows = []
    for (path, a), (_, b) in zip(ls, rs):
        result = array_difference(a, b, atol=1e-6)
        result["path"] = jax.tree_util.keystr(path)
        rows.append(result)
    ordered = sorted(rows, key=lambda r: r["maximum_absolute_difference"] or 0, reverse=True)
    return {
        "pass": all(r["pass"] for r in rows),
        "finite": all(r["finite_same_shape"] for r in rows),
        "maximum_absolute_difference": ordered[0]["maximum_absolute_difference"],
        "maximum_path": ordered[0]["path"],
        "failed_leaves": [r for r in rows if not r["pass"]],
        "largest_differences": ordered[:12],
        "leaf_count": len(rows),
    }


def learner(args):
    modes = {
        "default": "",
        "deterministic": "--xla_gpu_deterministic_ops=true",
        "deterministic-no-autotune": "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0",
    }
    if "jax" in sys.modules:
        raise ValueError("Numerical flags must be set before JAX initialization")
    os.environ["XLA_FLAGS"] = modes[args.mode]
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    data, input_hash = inputs()
    directory = ROOT / "outputs/dsrl07/diagnostics" / args.operation
    output = directory / "result.json"
    if output.exists():
        print(json.dumps({"already_completed": str(output)}))
        return
    directory.mkdir(parents=True, exist_ok=True)
    started = directory / "started.json"
    if started.exists():
        raise ValueError("Unfinished diagnostic needs reconciliation and a fresh operation ID")
    atomic_json(started, {"utc": utc(), "mode": args.mode, "input_manifest_sha256": input_hash})
    bridge = LearnerBridge(
        ROOT / "outputs/dsrl07/learner",
        ROOT / "third_party/dsrl_pi0",
        data_root=ROOT / "outputs/dsrl07",
        config=LearnerConfig(),
    )
    bridge.initialize("engineering")
    backend = bridge.backend
    values = load_npz(ROOT / data["profile"]["path"])
    idx = np.arange(256) % len(values["pixels"])
    obs = {"pixels": values["pixels"][idx, ..., None], "state": values["states"][idx, ..., None]}
    batch = {
        "observations": obs,
        "next_observations": obs,
        "actions": np.zeros((256, 1, 32), np.float32),
        "next_actions": np.zeros((256, 1, 32), np.float32),
        "rewards": np.zeros(256, np.float32),
        "masks": np.zeros(256, np.float32),
        "discount": np.full(256, 0.9998**8, np.float32),
    }
    initial = backend.dumps()
    bridge.save(f"{args.operation}-before")
    comparisons, first_reference = [], None
    for pair in range(5):
        backend.loads(initial)
        reference = copy.copy(backend.agent)
        bridge.reserve_update("engineering", f"{args.operation}:reference:{pair}")
        begin = time.monotonic()
        info = reference.update(backend.freeze(batch))
        backend.jax.block_until_ready(info)
        reference_seconds = time.monotonic() - begin
        expected = {**reference._save_dict, "rng": reference._rng}
        bridge.reserve_update("engineering", f"{args.operation}:bridge:{pair}")
        begin = time.monotonic()
        backend.update(batch)
        bridge_seconds = time.monotonic() - begin
        actual = {**backend.agent._save_dict, "rng": backend.agent._rng}
        comparison = {
            "pair": pair,
            "source_vs_bridge": tree_difference(backend.jax, expected, actual),
            "reference_seconds": reference_seconds,
            "bridge_seconds": bridge_seconds,
        }
        if first_reference is not None:
            comparison["reference_vs_first_reference"] = tree_difference(
                backend.jax, first_reference, expected
            )
        else:
            first_reference = expected
        comparisons.append(comparison)
        atomic_json(directory / f"pair-{pair}.json", comparison)
    checkpoint = bridge.save(f"{args.operation}-after")
    result = {
        "schema": "dsrl07-learner-diagnosis-v1",
        "mode": args.mode,
        "flags": modes[args.mode],
        "input_manifest_sha256": input_hash,
        "jax_version": backend.jax.__version__,
        "devices": [str(x) for x in backend.jax.devices()],
        "comparisons": comparisons,
        "source_parity_pass": all(c["source_vs_bridge"]["pass"] for c in comparisons),
        "reference_repeatability_pass": all(
            c.get("reference_vs_first_reference", {"pass": True})["pass"] for c in comparisons
        ),
        "checkpoint": checkpoint,
        "status": bridge.status(),
        "completed_utc": utc(),
    }
    atomic_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"comparisons", "status"}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("decoder", "learner"))
    parser.add_argument("--operation", required=True)
    parser.add_argument("--lane", type=int, choices=range(4), default=0)
    parser.add_argument(
        "--mode",
        choices=("default", "deterministic", "deterministic-no-autotune"),
        default="default",
    )
    args = parser.parse_args()
    if Path(args.operation).name != args.operation or not args.operation:
        raise ValueError("Operation must be one path component")
    (decoder if args.kind == "decoder" else learner)(args)


if __name__ == "__main__":
    main()
