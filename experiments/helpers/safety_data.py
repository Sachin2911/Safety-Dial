"""Offline safety examples from frozen LeWM imagination and dense recorded positions."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from helpers.safety_model import (
    FORMAT,
    Constraint,
    SafetySpec,
    chunk_features,
    file_digest,
    pusht_constraints,
    scaler_digest,
)

SPLITS = ("train", "calibration", "test")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_partitions(entries):
    """Relabelings/overlapping windows of one original source cannot cross splits."""
    groups = {}
    for entry in entries:
        group, split = entry["source_group"], entry["split"]
        if not isinstance(group, str) or not group or split not in SPLITS:
            raise ValueError("Every trace needs an original-source group and an explicit split")
        if group in groups and groups[group] != split:
            raise ValueError(f"Original source group crosses splits: {group}")
        groups[group] = split
    return groups


def trace_arrays(path, entry, spec):
    with np.load(path, allow_pickle=False) as data:
        frames = data[entry.get("frames_key", "frames")]
        actions = data[entry.get("actions_key", "actions")].astype(np.float32)
        dense = data[entry.get("dense_key", "dense")][:, entry["position_columns"]]
    n, substeps = len(actions), entry["substeps_per_action"]
    if not isinstance(substeps, int) or substeps < 1 or dense.shape != (1 + n * substeps, 2):
        raise ValueError("Dense trace must include the initial point and every physics substep")
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected uint8 RGB frames in NHWC format")
    if len(frames) != n + 1 or actions.shape != (n, spec.action_dim):
        raise ValueError("Frames, actions and dense trace are not aligned")
    if not np.isfinite(actions).all() or not np.isfinite(dense).all():
        raise ValueError("Trace contains nonfinite values")
    return frames, actions, dense


@torch.inference_mode()
def encode_frames(model, frames, device, counters, batch_size=32):
    from helpers.safecem10_planner import image_transform

    transform, encoded = image_transform(), []
    for start in range(0, len(frames), batch_size):
        pixels = torch.from_numpy(frames[start : start + batch_size].copy()).permute(0, 3, 1, 2)
        pixels = transform(pixels).to(device)
        encoded.append(model.encode({"pixels": pixels[:, None]})["emb"][:, -1])
        counters["encoder_frames"] += len(pixels)
    return torch.cat(encoded)


@torch.inference_mode()
def prepare_examples(model, spec, entries, action_mean, action_scale, root, counters):
    """No simulator calls. Explicit manifest controls sources, partitions and windows."""
    from helpers.safecem10_planner import image_transform

    source_partitions(entries)
    if scaler_digest(action_mean, action_scale) != spec.action_scaler_sha256:
        raise ValueError("Action normalizer differs from the model specification")
    device = next(model.parameters()).device
    model.eval().requires_grad_(False)
    mean = np.asarray(action_mean, dtype=np.float32)
    scale = np.asarray(action_scale, dtype=np.float32)
    if mean.shape != (spec.action_dim,) or scale.shape != mean.shape or not (scale > 0).all():
        raise ValueError("Invalid action normalizer")
    records, targets, groups, splits, depths, kinds = [], [], [], [], [], []
    for entry in entries:
        path = Path(root) / entry["trace"]
        if file_digest(path) != entry["sha256"]:
            raise ValueError(f"Trace hash differs from manifest: {path}")
        frames, actions, dense = trace_arrays(path, entry, spec)
        constraints = (
            pusht_constraints(entry["pusht_case"])
            if "pusht_case" in entry
            else [Constraint(**x) for x in entry["constraints"]]
        )
        if "pusht_case" in entry and spec.environment != "pusht":
            raise ValueError("PushT shape mapping used for a different environment")
        embeddings = encode_frames(model, frames, device, counters)
        if embeddings.shape[-1] != spec.latent_dim:
            raise ValueError("Frozen encoder dimensions differ from the specification")
        starts = entry["window_starts"]
        if not starts or len(set(starts)) != len(starts):
            raise ValueError("Predeclare unique window starts")
        for start in starts:
            if not isinstance(start, int) or start < 0 or start + spec.action_block > len(actions):
                raise ValueError("Window start cannot supply a full action chunk")
            h = min(spec.max_horizon, (len(actions) - start) // spec.action_block)
            length = h * spec.action_block
            normalized = (actions[start : start + length] - mean) / scale
            candidates = torch.as_tensor(normalized, device=device).reshape(1, 1, h, -1)
            pixels = torch.from_numpy(frames[start].copy()).permute(2, 0, 1)
            pixels = image_transform()(pixels).to(device)[None, None, None]
            info = {"pixels": pixels, "emb": embeddings[start][None, None, None]}
            model.rollout(info, candidates)
            counters["model_batch_calls"] += 1
            counters["model_candidate_rollouts"] += 1
            counters["predictor_steps"] += h
            previous_raw = actions[start - 1] if start else np.zeros(spec.action_dim)
            previous_action = torch.as_tensor((previous_raw - mean) / scale, device=device)
            features = chunk_features(
                spec,
                embeddings[start][None],
                embeddings[max(0, start - 1)][None],
                info["predicted_emb"],
                candidates,
                previous_action[None].float(),
                candidates.new_tensor([float(start > 0)]),
                constraints,
            )
            records.append(features.reshape(-1, spec.input_dim).cpu().numpy())
            for depth in range(h):
                begin = (start + depth * spec.action_block) * entry["substeps_per_action"]
                end = begin + spec.action_block * entry["substeps_per_action"] + 1
                positions = torch.as_tensor(dense[begin:end], dtype=torch.float32)
                for constraint in constraints:
                    targets.append(float(constraint.clearance(positions, spec.body_radius).min()))
                    groups.append(entry["source_group"])
                    splits.append(entry["split"])
                    depths.append(depth + 1)
                    kinds.append(constraint.kind)
    if not records:
        raise ValueError("No complete action chunks")
    return {
        "features": np.concatenate(records).astype(np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "groups": np.asarray(groups),
        "splits": np.asarray(splits),
        "depths": np.asarray(depths),
        "kinds": np.asarray(kinds),
    }


def save_examples(path, arrays, spec, provenance):
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    write_json(
        path.with_suffix(".json"),
        {
            "format": FORMAT,
            "spec": asdict(spec),
            "sha256": file_digest(path),
            "provenance": provenance,
        },
    )


def load_examples(path):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["format"] != FORMAT or metadata["sha256"] != file_digest(path):
        raise ValueError("Safety dataset hash/format differs from its metadata")
    spec = SafetySpec(**metadata["spec"])
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    n = len(arrays["targets"])
    if not n or arrays["features"].shape != (n, spec.input_dim):
        raise ValueError("Invalid feature shape")
    if any(len(arrays[key]) != n for key in ("groups", "splits", "depths", "kinds")):
        raise ValueError("Dataset columns have different lengths")
    if not np.isfinite(arrays["features"]).all() or not np.isfinite(arrays["targets"]).all():
        raise ValueError("Nonfinite training data")
    source_partitions(
        [
            {"source_group": str(group), "split": str(split)}
            for group, split in zip(arrays["groups"], arrays["splits"])
        ]
    )
    if not np.isin(arrays["kinds"], ("circle", "box", "capsule", "workspace")).all():
        raise ValueError("Unsupported constraint labels")
    if not ((arrays["depths"] >= 1) & (arrays["depths"] <= spec.max_horizon)).all():
        raise ValueError("Invalid prediction depths")
    return arrays, spec, metadata
