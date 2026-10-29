"""Workstation CPU integration of the task-11 learner bridge with the pinned author agent.

Every gradient update below is a workstation engineering charge. The receipt records
each one so the stage-A freeze carries them into the frozen accounting; nothing here
is scientific evidence, and the synthetic episodes never enter any replay on the rental.
"""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments/scripts"))
from helpers.dsrl06_infra import atomic_json, utc  # noqa: E402
from helpers.dsrl11_learner import (  # noqa: E402
    Dsrl11LearnerBridge,
    Dsrl11LearnerConfig,
    FrozenResidualActors,
    state_digest,
)
from helpers.dsrl11_protocol import derive_seed  # noqa: E402

CONFIG = Dsrl11LearnerConfig(updates_per_transition=5, critic_reduction="min")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def synthetic_episode(root, job_id, version, *, condition=None, count=1):
    rng = np.random.default_rng(derive_seed("preflight-episode", job_id))
    durations = np.array([8] * (count - 1) + [5], np.int32)
    arrays = {
        "pixels": rng.integers(0, 256, (count + 1, 64, 64, 3), dtype=np.uint8),
        "states": rng.standard_normal((count + 1, 8)).astype(np.float32),
        "residual": rng.uniform(-1, 1, (count, 32)).astype(np.float32),
        "rewards_r0": rng.uniform(-0.05, 0.05, count).astype(np.float32),
        "rewards_r1": rng.uniform(-0.05, 0.05, count).astype(np.float32),
        "discounts": (CONFIG.gamma**durations).astype(np.float32),
        "masks": np.r_[np.ones(count - 1), 0].astype(np.float32),
        "durations": durations,
        "query_steps": np.arange(count, dtype=np.int32) * 8,
        "actor_versions": np.full(count, version, np.int32),
    }
    path = root / "episodes" / f"{job_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return {
        "job_id": job_id,
        "path": str(path),
        "sha256": sha(path),
        "actor_version": version,
        "transitions": count,
        "condition": condition,
    }


def act_request(condition, version, index, job_id):
    rng = np.random.default_rng(derive_seed("preflight-observation", job_id, index))
    pixels = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    return {
        "condition": condition,
        "actor_version": version,
        "job_id": job_id,
        "query_index": index,
        "rng_seed": derive_seed("preflight-actor", job_id),
        "pixels_b64": base64.b64encode(pixels.tobytes()).decode(),
        "state": rng.standard_normal(8).astype(np.float32).tolist(),
    }


def ledger_updates(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return sum(row["event"] == "update_reserved" for row in rows)


def engineering(root, source):
    bridge = Dsrl11LearnerBridge(root / "engineering", source, CONFIG, data_root=root)
    bridge.initialize("engineering")
    episodes = [synthetic_episode(root, f"E{i}", -1, count=3) for i in range(4)]
    profile = bridge.profile({"updates": 2, "episodes": episodes})
    exported = bridge.export_actor(
        {"condition": "engineering", "name": "engineering-v1", "actor_version": 1}
    )
    actors = FrozenResidualActors(root / "cpu-actors", source, {"engineering": exported["path"]})
    requests = [act_request("engineering", 1, k, f"E{4 + k}") for k in range(4)]
    live = [np.asarray(bridge.act(r)["residual"]) for r in requests]
    cpu = [np.asarray(actors.act(r)["residual"]) for r in requests]
    difference = max(float(np.max(np.abs(a - b))) for a, b in zip(live, cpu, strict=True))
    if difference > 1e-5:
        raise ValueError(f"CPU actor export differs from the bridge actor: {difference}")
    return {
        "source_parity": profile["source_parity"],
        "updates_charged": profile["updates_charged"],
        "median_update_seconds": profile["median_update_seconds"],
        "exported_actor_version": exported["actor_version"],
        "cpu_actor_maximum_difference": difference,
        "ledger_updates": ledger_updates(bridge.ledger_path),
    }


def learning(root, source):
    bridge = Dsrl11LearnerBridge(root / "learning", source, CONFIG, data_root=root)
    status = bridge.initialize("learning")
    initial = [
        json.loads(line)
        for line in bridge.ledger_path.read_text().splitlines()
        if json.loads(line)["event"] == "initialized"
    ][0]["initial_state_sha256"]
    seeds = {name: state.backend.rng_seed for name, state in bridge.conditions.items()}
    bank = [synthetic_episode(root, f"B{i:02d}", -1) for i in range(12)]
    loaded = bridge.load_bank({"episodes": bank, "strict_safe_completions": 2})
    condition = "r1_prior"
    warm = [
        synthetic_episode(root, f"T_{condition}_{k}", -1, condition=condition) for k in range(3)
    ]
    bridge.append({"condition": condition, "phase": "warmup", "round_id": "w", "episodes": warm})
    batch = [
        synthetic_episode(root, f"T_{condition}_{k}", 0, condition=condition) for k in range(3, 6)
    ]
    appended = bridge.append(
        {"condition": condition, "phase": "batch_a", "round_id": "a", "episodes": batch}
    )
    trained = bridge.train({"condition": condition, "round_id": "a", "expected_new_transitions": 3})
    published = bridge.publish({"condition": condition, "round_id": "a"})
    request = act_request(condition, published["actor_version"], 0, f"T_{condition}_6")
    before = bridge.act(request)["residual"]
    saved = bridge.save("preflight")
    digest = state_digest(bridge.conditions[condition].backend)
    restored = Dsrl11LearnerBridge(root / "learning", source, CONFIG, data_root=root)
    restored.load(saved["path"])
    after = restored.act(request)["residual"]
    restored_digest = state_digest(restored.conditions[condition].backend)
    if restored_digest != digest or after != before:
        raise ValueError("Checkpoint restore changed the learner state or its actor")
    others = {
        name: state_digest(state.backend)
        for name, state in restored.conditions.items()
        if name != condition
    }
    if set(others.values()) != {initial[condition]}:
        raise ValueError("Updating one condition changed another condition's state")
    calibrations = bridge.conditions[condition].calibrations
    return {
        "initial_state_sha256": initial,
        "agent_rng_seeds": seeds,
        "bank_transitions": loaded["bank_transitions"],
        "credit": appended["update_credit"],
        "updates": trained["updates"],
        "calibration_labels": [c["label"] for c in calibrations],
        "calibration_finite": [c["finite"] for c in calibrations],
        "gate_fractions": [c["gate_fractions"] for c in calibrations],
        "published_actor_version": published["actor_version"],
        "checkpoint_manifest_sha256": saved["manifest_sha256"],
        "restore_state_identical": True,
        "restore_actor_identical": True,
        "untrained_conditions_unchanged": True,
        "parameter_counts": status["parameter_counts"],
        "ledger_updates": ledger_updates(bridge.ledger_path),
    }


def diagnostic(root, source, task7_root):
    import diagnose_dsrl11 as diagnose
    from helpers.dsrl11_learner import Dsrl11Backend, calibration_summary

    episodes, receipt = diagnose.task7_replay(task7_root)
    pool, fitting = diagnose.fitting_pool(episodes)
    calibration = diagnose.calibration_episodes(episodes)
    config = Dsrl11LearnerConfig(seed=11061, updates_per_transition=20, critic_reduction="min")
    backend = Dsrl11Backend(source, config, rng_seed=None)
    sampler = np.random.default_rng(derive_seed("preflight-a1", 0))
    infos = []
    for _ in range(2):
        rows = sampler.integers(0, len(pool["rewards"]), size=config.batch_size)
        infos.append(backend.update(diagnose.batch_at(pool, rows)))
    raw = backend.calibrate(calibration, derive_seed("a1-calibration-actions", 11061))
    summary = calibration_summary(raw, calibration, "min")
    return {
        "replay": receipt,
        "fitting_episodes": len(fitting),
        "calibration_transitions": summary["transitions"],
        "updates": len(infos),
        "finite": bool(summary["finite"])
        and all(np.isfinite(v) for info in infos for v in info.values()),
        "gate_fractions": summary["gate_fractions"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "third_party/dsrl_pi0")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dsrl11/preflight")
    parser.add_argument("--task7-root", type=Path, default=ROOT / "outputs/dsrl07/remote-final")
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        raise ValueError("The preflight is recorded once; its updates are already charged")
    root.mkdir(parents=True)
    begin = time.monotonic()
    import jax

    receipt = {
        "schema": "dsrl11-cpu-preflight-v1",
        "started_utc": utc(),
        "jax_version": jax.__version__,
        "devices": [str(d) for d in jax.devices()],
        "config": CONFIG.resolved(),
        "role": "Workstation engineering only; synthetic replay never reaches the rental",
    }
    atomic_json(root / "started.json", receipt)
    receipt["engineering"] = engineering(root, args.source)
    receipt["learning"] = learning(root, args.source)
    receipt["diagnostic"] = diagnostic(root, args.source, args.task7_root)
    receipt["workstation_updates"] = (
        receipt["engineering"]["ledger_updates"]
        + receipt["learning"]["ledger_updates"]
        + receipt["diagnostic"]["updates"]
    )
    receipt["elapsed_seconds"] = time.monotonic() - begin
    receipt["finished_utc"] = utc()
    atomic_json(root / "receipt.json", receipt)
    print(json.dumps({k: receipt[k] for k in ("workstation_updates", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
