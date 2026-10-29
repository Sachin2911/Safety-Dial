"""Task-11 stage-A diagnostics on the saved task-7 tiled-noise replay.

``critic-cell`` runs one approved A1 cell (reduction, update-to-data ratio, seed) on
the stage-C update schedule. ``task7-forward`` replays the preserved task-7 critic
checkpoints with forward passes only. ``static-audit`` checks the replay, rewards and
pinned learner source without JAX. No VLA or simulator call is made by any of them.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_protocol import GAMMA, SHAPING_SCALE
from helpers.dsrl11_protocol import CALIBRATION_INTERVAL, DIAGNOSTIC_SEEDS, derive_seed

CALIBRATION_EPISODES = ("T00", "T04", "T08", "T12", "T16", "T20")
FORWARD_CHECKPOINTS = ("round00", "round02", "episode12", "round04", "round05", "round06")
FORWARD_CHECKPOINTS = (*FORWARD_CHECKPOINTS, "episode24")
SCHEDULE = {"initial_reveal": 1500, "credit_per_phase": 750, "phases": 2}
LEARNER_SOURCE = (
    "third_party/dsrl_pi0/jaxrl2/agents/pixel_sac/pixel_sac_learner.py",
    "third_party/dsrl_pi0/jaxrl2/agents/pixel_sac/critic_updater.py",
    "third_party/dsrl_pi0/jaxrl2/agents/pixel_sac/actor_updater.py",
    "third_party/dsrl_pi0/jaxrl2/agents/pixel_sac/temperature_updater.py",
)


def utc():
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_event(path, event, **values):
    with Path(path).open("a") as stream:
        stream.write(json.dumps({"utc": utc(), "event": event, **values}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


# ---------------------------------------------------------------------------
# Saved task-7 replay.


def task7_replay(task7_root: Path):
    """Split the preserved final task-7 training replay into its 24 complete episodes."""
    checkpoint = Path(task7_root) / "learner/checkpoints/episode24"
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    for name in ("replay.npz", "state.json"):
        if digest(checkpoint / name) != manifest[name]:
            raise ValueError(f"Task-7 checkpoint file changed: {name}")
    state = json.loads((checkpoint / "state.json").read_text())
    with np.load(checkpoint / "replay.npz", allow_pickle=False) as source:
        data = {key: source[key].copy() for key in source.files}
    if state["replay"]["size"] != len(data["rewards"]):
        raise ValueError("Task-7 replay size differs from its checkpoint state")
    episodes, offset = [], 0
    for row in state["episodes"]:
        count = int(row["transitions"])
        rows = slice(offset, offset + count)
        if not np.array_equal(data["next_pixels"][rows][:-1], data["pixels"][rows][1:]):
            raise ValueError(f"{row['job_id']}: replay rows are not one contiguous episode")
        if not np.array_equal(data["next_states"][rows][:-1], data["states"][rows][1:]):
            raise ValueError(f"{row['job_id']}: replay states are not contiguous")
        episodes.append(
            {
                "job_id": row["job_id"],
                "order_index": row["order_index"],
                "warmup": row["warmup"],
                "actor_version": row["actor_version"],
                "descriptor_sha256": row["sha256"],
                "pixels": np.concatenate(
                    [data["pixels"][rows], data["next_pixels"][offset + count - 1][None]]
                ),
                "states": np.concatenate(
                    [data["states"][rows], data["next_states"][offset + count - 1][None]]
                ),
                "noise": data["noise"][rows],
                "next_noise": data["next_noise"][rows],
                "rewards": data["rewards"][rows],
                "discounts": data["discounts"][rows],
                "masks": data["masks"][rows],
            }
        )
        offset += count
    if offset != len(data["rewards"]):
        raise ValueError("Task-7 replay contains rows outside its recorded episodes")
    receipt = {
        "checkpoint": str(checkpoint.relative_to(ROOT))
        if checkpoint.is_relative_to(ROOT)
        else str(checkpoint),
        "manifest_sha256": digest(checkpoint / "manifest.json"),
        "replay_sha256": manifest["replay.npz"],
        "state_sha256": manifest["state.json"],
        "episodes": [e["job_id"] for e in episodes],
        "transitions": offset,
    }
    return episodes, receipt


def fitting_pool(episodes):
    fitting = [e for e in episodes if e["job_id"] not in CALIBRATION_EPISODES]
    if len(fitting) != 18:
        raise ValueError("A1 fits on exactly the other 18 task-7 episodes")
    keys = ("pixels", "next_pixels", "states", "next_states", "noise", "next_noise")
    pool = {key: [] for key in (*keys, "rewards", "masks", "discounts")}
    for episode in fitting:
        pool["pixels"].append(episode["pixels"][:-1])
        pool["next_pixels"].append(episode["pixels"][1:])
        pool["states"].append(episode["states"][:-1])
        pool["next_states"].append(episode["states"][1:])
        for key in ("noise", "next_noise", "rewards", "masks", "discounts"):
            pool[key].append(episode[key])
    return {key: np.concatenate(value) for key, value in pool.items()}, [
        e["job_id"] for e in fitting
    ]


def batch_at(pool, rows):
    return {
        "observations": {
            "pixels": pool["pixels"][rows][..., None],
            "state": pool["states"][rows][..., None],
        },
        "next_observations": {
            "pixels": pool["next_pixels"][rows][..., None],
            "state": pool["next_states"][rows][..., None],
        },
        "actions": pool["noise"][rows][:, None, :],
        "next_actions": pool["next_noise"][rows][:, None, :],
        "rewards": pool["rewards"][rows],
        "masks": pool["masks"][rows],
        "discount": pool["discounts"][rows],
    }


def calibration_episodes(episodes):
    from helpers.dsrl11_learner import calibration_set

    chosen = [e for e in episodes if e["job_id"] in CALIBRATION_EPISODES]
    if [e["job_id"] for e in chosen] != list(CALIBRATION_EPISODES):
        raise ValueError("Calibration episodes are T00, T04, T08, T12, T16 and T20")
    return calibration_set(chosen, list(CALIBRATION_EPISODES))


# ---------------------------------------------------------------------------
# A1 cell.


def critic_cell(args):
    if "jax" in sys.modules:
        raise ValueError("Numerical flags must be set before JAX initialization")
    os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("WANDB_MODE", "disabled")
    if args.seed not in DIAGNOSTIC_SEEDS:
        raise ValueError("A1 uses only diagnostic seeds 11061 and 11062")
    directory = Path(args.work_dir) / f"{args.reduction}-utd{args.utd}-seed{args.seed}"
    output = directory / "result.json"
    if output.exists():
        print(json.dumps({"already_completed": str(output)}))
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    started = directory / "started.json"
    if started.exists():
        raise ValueError("An unfinished A1 cell needs reconciliation and a recorded repair")
    from helpers.dsrl11_learner import (
        Dsrl11Backend,
        Dsrl11LearnerConfig,
        calibration_summary,
        json_safe,
        pause_status,
    )

    episodes, replay_receipt = task7_replay(Path(args.task7_root))
    pool, fitting_ids = fitting_pool(episodes)
    calibration = calibration_episodes(episodes)
    permutation = np.random.default_rng(derive_seed("a1-permutation", args.seed)).permutation(
        len(pool["rewards"])
    )
    sampler = np.random.default_rng(derive_seed("a1-replay", args.seed))
    config = Dsrl11LearnerConfig(
        seed=args.seed, updates_per_transition=args.utd, critic_reduction=args.reduction
    )
    total = SCHEDULE["credit_per_phase"] * SCHEDULE["phases"] * args.utd
    atomic_json(
        started,
        {
            "utc": utc(),
            "config": config.resolved(),
            "replay": replay_receipt,
            "planned_updates": total,
            "permutation_sha256": hashlib.sha256(permutation.tobytes()).hexdigest(),
        },
    )
    backend = Dsrl11Backend(Path(args.source), config, rng_seed=None)
    ledger = directory / "diagnostic-ledger.jsonl"
    key_seed = derive_seed("a1-calibration-actions", args.seed)
    history, updates, nonfinite = [], 0, None

    def measure(scheduled, label):
        raw = backend.calibrate(calibration, key_seed)
        summary = json_safe(calibration_summary(raw, calibration, args.reduction))
        summary.update(updates=updates, scheduled=scheduled, label=label, utc=utc())
        history.append(summary)
        append_event(ledger, "calibration", **summary)
        return summary

    begin = time.monotonic()
    measure(False, "initialization")
    revealed = SCHEDULE["initial_reveal"]
    for phase in range(SCHEDULE["phases"]):
        if phase:
            revealed += SCHEDULE["credit_per_phase"]
        for _ in range(SCHEDULE["credit_per_phase"] * args.utd):
            append_event(
                ledger,
                "update_reserved",
                phase="diagnostic",
                cell=directory.name,
                phase_update=updates + 1,
            )
            rows = permutation[sampler.integers(0, revealed, size=config.batch_size)]
            info = backend.update(batch_at(pool, rows))
            updates += 1
            if not all(math.isfinite(v) for v in info.values()):
                nonfinite = {"update": updates, "metrics": json_safe(info)}
                append_event(ledger, "nonfinite_stop", **nonfinite)
                break
            if updates % 100 == 0:
                append_event(ledger, "training_metrics", update=updates, **json_safe(info))
            if updates % CALIBRATION_INTERVAL == 0:
                measure(True, f"update{updates}")
        if nonfinite:
            break
    if updates % CALIBRATION_INTERVAL:
        measure(False, f"update{updates}:end")
    status = pause_status(history)
    result = {
        "schema": "dsrl11-a1-cell-v1",
        "created_utc": utc(),
        "cell": {"reduction": args.reduction, "utd": args.utd, "seed": args.seed},
        "config": config.resolved(),
        "replay": replay_receipt,
        "fitting_episodes": fitting_ids,
        "calibration_episodes": list(CALIBRATION_EPISODES),
        "exposure_transitions": revealed,
        "planned_updates": total,
        "gradient_updates": updates,
        "nonfinite": nonfinite,
        "pause": status,
        "admitted": bool(updates == total and not nonfinite and not status["pause"]),
        "final_policy": history[-1]["policy"],
        "final_learner_metrics": json_safe(info),
        "elapsed_seconds": time.monotonic() - begin,
        "calibrations": history,
        "jax_version": backend.jax.__version__,
        "devices": [str(device) for device in backend.jax.devices()],
        "gate": "Admitted iff all updates are finite and the stage-C pause rule (configured "
        "reduction of the ten heads outside [-0.30, 1.05] for more than 1% of calibration pairs "
        "at three successive post-update measurements) never fires for recorded or "
        "actor-sampled actions; the coordinator requires both seeds",
    }
    atomic_json(output, result)
    print(json.dumps({k: result[k] for k in ("cell", "gradient_updates", "admitted", "pause")}))
    return 0


# ---------------------------------------------------------------------------
# Task-7 forward replay.


def tree_statistics(jax, state):
    rows = {}
    for path, value in jax.tree_util.tree_flatten_with_path(state)[0]:
        array = np.asarray(value)
        if array.dtype.kind not in "fiu":
            continue
        rows[jax.tree_util.keystr(path)] = {
            "shape": list(array.shape),
            "finite": bool(np.isfinite(array).all()),
            "absolute_maximum": float(np.max(np.abs(array))) if array.size else 0.0,
        }
    return rows


def task7_forward(args):
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("MPLBACKEND", "Agg")
    from helpers.dsrl11_learner import (
        Dsrl11Backend,
        Dsrl11LearnerConfig,
        calibration_summary,
        json_safe,
    )

    task7 = Path(args.task7_root)
    episodes, replay_receipt = task7_replay(task7)
    calibration = calibration_episodes(episodes)
    config = Dsrl11LearnerConfig(seed=6060, updates_per_transition=20, critic_reduction="mean")
    backend = Dsrl11Backend(Path(args.source), config, rng_seed=None)
    jax = backend.jax
    rows = []
    for name in FORWARD_CHECKPOINTS:
        checkpoint = task7 / "learner/checkpoints" / name
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        for file, expected in manifest.items():
            if digest(checkpoint / file) != expected:
                raise ValueError(f"Task-7 checkpoint {name} changed: {file}")
        state = json.loads((checkpoint / "state.json").read_text())
        if state["config"]["author_kwargs"] != {
            **config.author_kwargs(),
            "hidden_dims": list(config.author_kwargs()["hidden_dims"]),
            "cnn_features": list(config.author_kwargs()["cnn_features"]),
            "cnn_strides": list(config.author_kwargs()["cnn_strides"]),
        }:
            raise ValueError("Forward replay must use the exact task-7 author configuration")
        backend.loads((checkpoint / "learner.msgpack").read_bytes())
        agent = backend.agent
        raw = backend.calibrate(calibration, derive_seed("task7-forward-actions"))
        summary = json_safe(calibration_summary(raw, calibration, "mean"))
        critic = np.concatenate([np.ravel(x) for x in jax.tree.leaves(agent._critic.params)])
        target = np.concatenate([np.ravel(x) for x in jax.tree.leaves(agent._target_critic_params)])
        rows.append(
            {
                "checkpoint": name,
                "manifest_sha256": digest(checkpoint / "manifest.json"),
                "training_updates": state["counts"]["training"],
                "actor_version": state["actor_version"],
                "train_state_steps": {
                    "critic": int(agent._critic.step),
                    "actor": int(agent._actor.step),
                    "temperature": int(agent._temp.step),
                },
                "temperature": float(
                    np.asarray(agent._temp.apply_fn({"params": agent._temp.params}))
                ),
                "target_minus_critic_rms": float(np.sqrt(np.mean((target - critic) ** 2))),
                "critic_parameter_rms": float(np.sqrt(np.mean(critic**2))),
                "state_finite": all(
                    row["finite"] for row in tree_statistics(jax, agent._save_dict).values()
                ),
                "calibration": summary,
            }
        )
        print(
            json.dumps(
                {
                    "checkpoint": name,
                    "updates": rows[-1]["training_updates"],
                    "recorded_q_mean": summary["recorded"]["heads"].get("mean"),
                    "fraction_outside_gate": summary["gate_fractions"],
                }
            ),
            flush=True,
        )
    result = {
        "schema": "dsrl11-task7-forward-v1",
        "created_utc": utc(),
        "method": "Forward passes of preserved task-7 checkpoints on calibration episodes; "
        "no gradient update, simulator or VLA call",
        "replay": replay_receipt,
        "calibration_episodes": list(CALIBRATION_EPISODES),
        "action_sets": {
            "recorded": "tiled-noise labels stored in the task-7 replay",
            "sampled": "one actor sample per observation from a fixed key",
        },
        "rows": rows,
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
    }
    atomic_json(args.output, result)
    return 0


# ---------------------------------------------------------------------------
# Static audit (no JAX).


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"Pinned source lacks {name}")


def source_audit():
    files = {name: digest(ROOT / name) for name in LEARNER_SOURCE}
    learner = ast.parse((ROOT / LEARNER_SOURCE[0]).read_text())
    critic = ast.parse((ROOT / LEARNER_SOURCE[1]).read_text())
    update = _function(learner, "_update_jit")
    calls = [n for n in ast.walk(update) if isinstance(n, ast.Call)]
    names = [getattr(c.func, "id", getattr(c.func, "attr", None)) for c in calls]
    critic_call = next(c for c in calls if getattr(c.func, "id", None) == "update_critic")
    soft = next(c for c in calls if getattr(c.func, "id", None) == "soft_target_update")
    update_critic = _function(critic, "update_critic")
    defaults = dict(
        zip(
            [a.arg for a in update_critic.args.args][-len(update_critic.args.defaults) :],
            [ast.literal_eval(d) for d in update_critic.args.defaults],
        )
    )
    target_assignments = [
        ast.unparse(node)
        for node in ast.walk(update_critic)
        if isinstance(node, (ast.Assign, ast.AugAssign))
        and "target_q"
        in ast.unparse(node.targets[0] if isinstance(node, ast.Assign) else node.target)
    ]
    return {
        "files": files,
        "update_jit_calls": names,
        "backup_entropy_default": defaults.get("backup_entropy"),
        "caller_passes_backup_entropy": any(
            k.arg == "backup_entropy" for k in critic_call.keywords
        ),
        "critic_reduction_passed": any(k.arg == "critic_reduction" for k in critic_call.keywords),
        "target_assignments": target_assignments,
        "soft_target_update_arguments": [ast.unparse(a) for a in soft.args],
        "per_transition_discount": any(
            "batch['discount']" in t or 'batch["discount"]' in t for t in target_assignments
        ),
        "supported_reductions": sorted(
            {
                node.comparators[0].value
                for node in ast.walk(update_critic)
                if isinstance(node, ast.Compare)
                and isinstance(node.comparators[0], ast.Constant)
                and ast.unparse(node.left) == "critic_reduction"
            }
        ),
    }


def replay_audit(task7_root: Path):
    episodes, receipt = task7_replay(task7_root)
    rows, failures = [], []
    for episode in episodes:
        path = task7_root / "episodes" / f"{episode['job_id']}.attempt1"
        validation = json.loads((path / "validation.json").read_text())
        transitions = validation["transitions"]
        with np.load(path / "learner.npz", allow_pickle=False) as source:
            stored = {key: source[key].copy() for key in source.files}
        if digest(path / "learner.npz") != episode["descriptor_sha256"]:
            failures.append(f"{episode['job_id']}: learner episode hash differs from replay index")
        for key in ("pixels", "states", "noise", "rewards", "discounts", "masks"):
            if not np.array_equal(stored[key], episode[key]):
                failures.append(f"{episode['job_id']}: replay {key} differs from admitted episode")
        durations = stored["durations"]
        expected_rewards = np.asarray([t["reward"] for t in transitions], np.float32)
        weight, shaped, base = 1.0, 0.0, 0.0
        for row in transitions:
            shaped += weight * row["reward"]
            base += weight * row["base_reward"]
            weight *= row["discount"]
        identity_error = shaped + SHAPING_SCALE * transitions[0]["phi_before"] - base
        checks = {
            "rewards_equal_validated_transitions": bool(
                np.array_equal(episode["rewards"], expected_rewards)
            ),
            "discount_is_gamma_to_duration": bool(
                np.allclose(episode["discounts"], GAMMA**durations, atol=3e-7, rtol=1e-7)
            ),
            "nonterminal_durations_eight": bool(np.all(durations[:-1] == 8)),
            "single_final_zero_mask": bool(
                np.all(episode["masks"][:-1] == 1) and episode["masks"][-1] == 0
            ),
            "terminal_potential_zero": transitions[-1]["phi_after"] == 0.0,
            "return_identity": abs(identity_error) <= 1e-5,
            # Task-7 warm-up used unbounded Gaussian labels; actor episodes are bounded.
            "action_label_domain": bool(
                np.isfinite(episode["noise"]).all()
                and (episode["warmup"] or np.max(np.abs(episode["noise"])) <= 1.000001)
            ),
            "next_noise_rule": bool(
                np.array_equal(episode["next_noise"][:-1], episode["noise"][1:])
                and np.array_equal(episode["next_noise"][-1], episode["noise"][-1])
            ),
            "potential_bounds": all(0 <= t["phi_before"] <= 1 for t in transitions),
            "reward_bounds": bool(
                np.all(episode["rewards"] >= -0.25) and np.all(episode["rewards"] <= 1.25)
            ),
        }
        descriptive = {
            "warmup_gaussian_labels": bool(episode["warmup"]),
            "maximum_absolute_label": float(np.max(np.abs(episode["noise"]))),
            "strict_safe_success": bool(validation["labels"]["strict_safe_requested"]),
            "steps": validation["steps"],
            "return_identity_error": float(identity_error),
        }
        rows.append({"job_id": episode["job_id"], **checks, **descriptive})
        for name, value in checks.items():
            if value is False:
                failures.append(f"{episode['job_id']}: {name}")
    return {
        "replay": receipt,
        "episodes": rows,
        "success_episodes": sum(r["strict_safe_success"] for r in rows),
        "failures": failures,
        "valid": not failures,
    }


def static_audit(args):
    result = {
        "schema": "dsrl11-static-audit-v1",
        "created_utc": utc(),
        "source": source_audit(),
        "task7_replay": replay_audit(Path(args.task7_root)),
    }
    source = result["source"]
    result["valid"] = bool(
        result["task7_replay"]["valid"]
        and source["backup_entropy_default"] is False
        and not source["caller_passes_backup_entropy"]
        and source["critic_reduction_passed"]
        and set(source["supported_reductions"]) == {"mean", "min"}
        and source["per_transition_discount"]
    )
    atomic_json(args.output, result)
    print(
        json.dumps({"valid": result["valid"], "failures": result["task7_replay"]["failures"][:5]})
    )
    return 0 if result["valid"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cell = commands.add_parser("critic-cell")
    cell.add_argument("--seed", type=int, required=True)
    cell.add_argument("--reduction", choices=("mean", "min"), required=True)
    cell.add_argument("--utd", type=int, choices=(5, 20), required=True)
    cell.add_argument("--work-dir", type=Path, required=True)
    forward = commands.add_parser("task7-forward")
    forward.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("static-audit")
    audit.add_argument("--output", type=Path, required=True)
    for command in (cell, forward, audit):
        command.add_argument(
            "--task7-root", type=Path, default=ROOT / "outputs/dsrl07/remote-final"
        )
        command.add_argument("--source", type=Path, default=ROOT / "third_party/dsrl_pi0")
    args = parser.parse_args()
    return {
        "critic-cell": critic_cell,
        "task7-forward": task7_forward,
        "static-audit": static_audit,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
