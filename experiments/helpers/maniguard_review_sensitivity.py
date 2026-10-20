"""Offline controller diagnostics on completed development and training traces only.

This module imports no simulator, policy client or encoder. Replaying a controller
on factual features measures numerical sensitivity, not a physical counterfactual.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

BLOCKS = {"bias": slice(0, 1), "robot": slice(1, 20), "overview": slice(20, 28),
          "wrist": slice(28, 36), "instruction": slice(36, 44)}
TRAINING_STAGES = {"development", "shared_initial", "search_generation_1", "search_generation_2"}
ROOT = Path(__file__).resolve().parents[2]


def matrix(value, shape, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Invalid {name}: expected finite {shape}")
    if np.any(np.abs(result) > 1):
        raise ValueError(f"Out-of-domain {name}")
    return result


def controller_hash(weights):
    return hashlib.sha256(weights.astype("<f4").tobytes()).hexdigest()


def rms(values):
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def sensitivity(features, weights, parent=None):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not len(features) or features.shape[1] != 44:
        raise ValueError("Require nonempty factual features with 44 columns")
    features = matrix(features, features.shape, "features")
    weights = matrix(weights, (4, 44), "weights")
    preactivation = features @ weights.T
    codes = np.tanh(preactivation)
    blocks = {}
    for name, columns in BLOCKS.items():
        contribution = features[:, columns] @ weights[:, columns].T
        removed = features.copy()
        removed[:, columns] = 0
        blocks[name] = {"preactivation_rms": rms(contribution),
                        "code_rms_change_on_block_removal": rms(
                            codes - np.tanh(removed @ weights.T))}
    result = {"feature_rows": len(features), "sha256": controller_hash(weights),
              "code_mean": codes.mean(axis=0).tolist(),
              "code_std": codes.std(axis=0).tolist(),
              "saturated_095_fraction": float(np.mean(np.abs(codes) >= .95)),
              "saturated_099_fraction": float(np.mean(np.abs(codes) >= .99)),
              "mean_tanh_derivative": float(np.mean(1 - codes**2)), "blocks": blocks}
    if parent is not None:
        parent = matrix(parent, (4, 44), "parent weights")
        difference = codes - np.tanh(features @ parent.T)
        result["parent_comparison"] = {
            "same_genotype": bool(np.array_equal(weights, parent)),
            "weight_rms_change": rms(weights - parent),
            "code_rms_change": rms(difference),
            "code_rms_change_by_channel": np.sqrt(np.mean(difference**2, axis=0)).tolist(),
            "arm_noise_offset_rms_change": .35 * float(np.sqrt(
                np.mean(np.sum(difference.astype(np.float64)**2, axis=1)) / 7)),
        }
    return result


def candidates(state):
    """Inspect already persisted cohorts, including unexecuted proposed children."""
    found = {}
    for cohort in state["cohorts"].values():
        for row in cohort:
            weights = matrix(row["weights"], (4, 44), "weights")
            item = {"id": row["id"], "parent": row.get("parent"), "weights": weights}
            if row["id"] in found:
                old = found[row["id"]]
                if not np.array_equal(old["weights"], weights) or old["parent"] != item["parent"]:
                    raise ValueError("A persisted candidate identity changed")
            found[row["id"]] = item
    if not found:
        raise ValueError("No persisted controller cohort")
    if any(row["parent"] is not None and row["parent"] not in found for row in found.values()):
        raise ValueError("A candidate parent is missing from the saved cohorts")
    return found


def trace_data(path):
    """Read only features, factual predicates and held-target goal observations."""
    digest = hashlib.sha256()
    features, query_steps, observed_steps, goal_steps = [], [], [], []
    predicates, first_doomed, held_steps = {}, None, []
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            if row["type"] == "latent_features":
                features.append(matrix(row["features"], (44,), "features"))
                query_steps.append(row["step"])
            elif row["type"] == "observed":
                observed_steps.append(row["step"])
                monitor = row["monitor"]
                if monitor.get("valid") is not True:
                    raise ValueError("Invalid recorded monitor")
                observation = monitor["observation"]
                if observation.get("doomed") is True and first_doomed is None:
                    first_doomed = row["step"]
                for key, value in observation["ap"].items():
                    if type(value) is not bool:
                        raise ValueError("Invalid recorded predicate")
                    stat = predicates.setdefault(key, {"samples": 0, "true_samples": 0,
                                                        "first_true_step": None,
                                                        "first_false_step": None})
                    stat["samples"] += 1
                    stat["true_samples"] += int(value)
                    first = "first_true_step" if value else "first_false_step"
                    if stat[first] is None:
                        stat[first] = row["step"]
            elif row["type"] == "goal":
                goal_steps.append(row["step"])
                held = row["detail"].get("held")
                if type(held) is not bool:
                    raise ValueError("Missing held-target evidence")
                if held:
                    held_steps.append(row["step"])
    if (not features or observed_steps != list(range(len(observed_steps)))
            or goal_steps != list(range(1, len(observed_steps)))
            or query_steps != sorted(set(query_steps))
            or any(step >= len(goal_steps) for step in query_steps)):
        raise ValueError("Incomplete or unordered factual training trace")
    summary = {"trace_sha256": digest.hexdigest(), "controls": len(goal_steps),
               "query_rows": len(features), "first_doomed_monitor_step": first_doomed,
               "held_target_control_samples": len(held_steps),
               "first_held_target_step": held_steps[0] if held_steps else None,
               "predicate_samples": predicates}
    return np.stack(features), summary


def analyze(run):
    run = Path(run).resolve()
    queue_bytes = (run / "queue.json").read_bytes()
    state_bytes = (run / "search/state.json").read_bytes()
    queue, state = json.loads(queue_bytes), json.loads(state_bytes)
    controllers = candidates(state)
    episodes, pools, excluded = [], {}, {}
    for attempt in queue["attempts"]:
        stage = attempt["stage"]
        if stage not in TRAINING_STAGES or attempt["status"] != "completed":
            excluded[stage] = excluded.get(stage, 0) + 1
            continue  # Never open independent-audit outputs, including incomplete ones.
        output = Path(attempt["output"])
        if not output.is_absolute():
            output = ROOT / output
        job = json.loads((output / "job.json").read_text())
        receipt = json.loads((output / "verified-receipt.json").read_text())
        if receipt["valid"] is not True or receipt["full_episode"] is not True:
            raise ValueError("Completed training episode lacks a full verification receipt")
        values, factual = trace_data(output / "trace.jsonl")
        with np.load(output / "controller.npz", allow_pickle=False) as data:
            weights = matrix(data["weights"], (4, 44), "episode weights")
        if (receipt["controller_sha256"] != controller_hash(weights)
                or receipt["controller_sha256"] != job["controller_sha256"]
                or factual["controls"] != receipt["physical"]["steps"]):
            raise ValueError("Stored training receipt and controller/trace disagree")
        family = job["family"]
        pools.setdefault(family, []).append(values)
        ledger = receipt["physical"]["ledger"]
        episodes.append({"attempt": attempt["id"], "stage": stage, "family": family,
                         "scene": job["scene"], "seed": job["seed"], **factual,
                         "score": receipt["score"], "outcome": receipt["outcome"],
                         "ever_contacted": receipt["physical"]["ever_contacted"],
                         "ever_grasped": receipt["physical"]["ever_grasped"],
                         "process_elapsed_seconds": attempt.get("elapsed_seconds"),
                         "policy_query_seconds": ledger.get("policy_query_seconds"),
                         "physics_seconds": ledger.get("physics_seconds"),
                         "encoder_seconds": ledger.get("encoder_seconds"),
                         "factual_controller": sensitivity(values, weights)})
    if not episodes:
        raise ValueError("No completed full development or training episodes")
    responses = {}
    for family, blocks in pools.items():
        factual = np.concatenate(blocks)
        responses[family] = []
        for row in controllers.values():
            parent = controllers.get(row["parent"])
            responses[family].append({"id": row["id"], "parent": row["parent"],
                                      **sensitivity(factual, row["weights"],
                                                    None if parent is None else parent["weights"])})
    return {"schema": "maniguard-review-sensitivity-v1",
            "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
            "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
            "episodes": episodes, "candidate_responses_by_family": responses,
            "excluded_attempt_counts_by_stage": excluded,
            "new_simulator_controls": 0, "new_policy_queries": 0, "new_encoder_calls": 0,
            "scope": "Factual development/training inputs only. Proposed child responses and block "
                     "removal measure numerical controller sensitivity, not decoded actions, "
                     "physical counterfactuals, semantic competence or safety. Queries within "
                     "episodes are correlated; pooled rows weight longer trajectories more. "
                     "Existing physical verification receipts are consumed, not independently "
                     "recertified. Timing categories can overlap and are not additive."}
