"""Frozen recorded-data transitions, excluding independent physical checking."""

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_STAGES = {"development", "shared_initial", "search_generation_1", "search_generation_2"}
STAGE_NAMES = ["prerequisite", "held", "in_goal", "upright", "supported"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError("Invalid " + name)
    return result


@dataclass
class Transitions:
    current: np.ndarray
    previous: np.ndarray
    history_valid: np.ndarray
    actions: np.ndarray
    following: np.ndarray
    stages: np.ndarray
    risk: np.ndarray
    risk_eligible: np.ndarray
    episode: np.ndarray
    task: np.ndarray
    step: np.ndarray

    def __len__(self):
        return len(self.current)

    def inputs(self):
        return np.concatenate([self.current, self.previous, self.history_valid[:, None],
                               self.actions.reshape(len(self), -1)], axis=1)

    def subset(self, indices):
        return Transitions(**{f.name: getattr(self, f.name)[indices] for f in fields(self)})


def freeze_manifest(run, config, output):
    """Freeze exact files and whole-task partitions before fitting any predictor."""
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Preserve the existing manifest; use a new cohort directory")
    raw = (run / "queue.json").read_bytes()
    queue = json.loads(raw)
    split_tasks = config["partitions"]
    groups = [set(tasks) for tasks in split_tasks.values()]
    if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i)):
        raise ValueError("Whole tasks cannot overlap fitting, calibration and diagnostic splits")
    entries = []
    for attempt in queue["attempts"]:
        # Stage exclusion precedes opening any output, including its job header.
        if attempt["stage"] not in ALLOWED_STAGES or attempt["status"] != "completed":
            continue
        path = Path(attempt["output"])
        path = path if path.is_absolute() else ROOT / path
        job = json.loads((path / "job.json").read_text())
        if job["family"] != config["family"]:
            continue
        task = job["scene"].split("/")[0]
        split = next((s for s, tasks in split_tasks.items() if task in tasks), None)
        if split is None or attempt["stage"] not in config["allowed_stages_by_split"][split]:
            continue
        receipt = json.loads((path / "verified-receipt.json").read_text())
        if receipt["valid"] is not True or receipt["full_episode"] is not True:
            raise ValueError("Only previously verified full episodes may enter the corpus")
        names = ["trace.jsonl", "job.json", "verified-receipt.json", "controller.npz"]
        entries.append({"episode": attempt["id"], "task": task, "split": split,
                        "stage": attempt["stage"], "output": str(path),
                        "controls": receipt["physical"]["steps"],
                        "safe_requested_completion": receipt["outcome"]["S_req"],
                        "sha256": {name: digest(path / name) for name in names}})
    if any(not any(e["split"] == split for e in entries) for split in split_tasks):
        raise ValueError("Every declared task partition needs recorded episodes")
    source_paths = [ROOT / "experiments/helpers" / ("maniguard_model_" + n + ".py")
                    for n in ("budget", "data", "transition", "control")]
    source_paths.append(ROOT / "experiments/scripts/maniguard_model_followup.py")
    source_paths.append(ROOT / "experiments/helpers/maniguard_latent.py")
    result = {"schema": "maniguard-model-corpus-v1", "run": str(run),
              "queue_sha256": hashlib.sha256(raw).hexdigest(), "configuration": config,
              "canonical_ledger": str(output.parent / "work.jsonl"),
              "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
              "entries": entries,
              "reused_acquisition_controls": sum(e["controls"] for e in entries),
              "new_simulator_controls": 0, "new_decoder_requests": 0, "new_encoder_images": 0,
              "scope": ("Previously exposed development/training tasks; "
                        "no independent checking data.")}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def _episode(entry, config):
    if entry["stage"] not in ALLOWED_STAGES:
        raise ValueError("Independent checking is excluded even in an edited manifest")
    directory = Path(entry["output"])
    for name, expected in entry["sha256"].items():
        if digest(directory / name) != expected:
            raise ValueError("Recorded corpus file changed: " + str(directory / name))
    job = json.loads((directory / "job.json").read_text())
    if job["family"] != config["family"] or job["scene"].split("/")[0] != entry["task"]:
        raise ValueError("Recorded task identity differs from its manifest")
    features, actions, observations, goals = {}, {}, {}, {}
    with (directory / "trace.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            kind, step = row["type"], row.get("step")
            if kind == "latent_features":
                if step in features:
                    raise ValueError("Duplicate feature observation")
                features[step] = finite(row["features"], (44,), "recorded actor features")
            elif kind == "executed":
                if (step in actions or row["action_scope"]
                        != "environment_input_before_controller_output_clipping"):
                    raise ValueError("Ambiguous executed action history")
                actions[step] = finite(row["action"], (8,), "executed action")
            elif kind == "observed":
                if step in observations or row["monitor"]["valid"] is not True:
                    raise ValueError("Invalid or duplicate passive observation")
                observations[step] = row["monitor"]["observation"]
            elif kind == "goal":
                if step in goals:
                    raise ValueError("Duplicate factual goal observation")
                goals[step] = row["detail"]
    controls = entry["controls"]
    if (sorted(actions) != list(range(controls))
            or sorted(observations) != list(range(controls + 1))
            or sorted(goals) != list(range(1, controls + 1))
            or sorted(features) != list(range(0, controls, 8))):
        raise ValueError("Recorded causal trajectory is incomplete")
    horizon = config["prefix_controls"]
    rows = []
    for step in sorted(features):
        end = step + horizon
        if end not in features:
            continue  # Do not invent an encoded terminal or future VLA observation.
        old = step - horizon
        ap, goal = observations[end]["ap"], goals[end]
        labels = [ap[config["predicates"]["prerequisite"]], goal["held"], goal["intersects"],
                  ap[config["predicates"]["upright"]], ap[config["predicates"]["supported"]]]
        if any(type(value) is not bool for value in labels):
            raise ValueError("Missing factual stage supervision")
        prior_doom = any(observations[t]["doomed"] for t in range(step + 1))
        risk = any(observations[t]["doomed"] for t in range(step + 1, end + 1))
        rows.append((features[step], features.get(old, features[step]), old in features,
                     np.stack([actions[t] for t in range(step, end)]), features[end],
                     labels, risk, not prior_doom, entry["episode"], entry["task"], step))
    limit = config["max_rows_per_episode"]
    if len(rows) > limit:
        indices = np.linspace(0, len(rows) - 1, limit, dtype=int)
        rows = [rows[i] for i in indices]
    return rows


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    config, datasets = manifest["configuration"], {}
    task_splits = {}
    for entry in manifest["entries"]:
        if task_splits.setdefault(entry["task"], entry["split"]) != entry["split"]:
            raise ValueError("A task crosses frozen split boundaries")
        if entry["task"] not in config["partitions"][entry["split"]]:
            raise ValueError("Task is outside its declared partition")
        datasets.setdefault(entry["split"], []).extend(_episode(entry, config))
    result = {}
    for split, rows in datasets.items():
        if not rows:
            raise ValueError("No complete query-prefix transitions in " + split)
        arrays = [np.asarray(column) for column in zip(*rows, strict=True)]
        result[split] = Transitions(**dict(zip([f.name for f in fields(Transitions)],
                                              arrays, strict=True)))
    return manifest, result
