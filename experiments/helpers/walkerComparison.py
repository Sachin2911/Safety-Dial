"""Integrity checks for the matched Walker pretraining/adaptation comparison."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from helpers.walkerProtocol import file_sha256
from helpers.walkerValidation import upload_reference


def completed_s4(report, seed):
    if report.get("status") != "complete" or report.get("acquisition_mode") != "prospective":
        raise ValueError("S5 comparison requires a completed prospective S4 report")
    if not report.get("development_gate", {}).get("go"):
        raise ValueError("S4 development decomposition gate did not pass")
    budgets = report.get("planned_budgets", [])
    seeds = report.get("acquisition_seeds", [])
    if not budgets or sorted(set(budgets)) != budgets or seed not in seeds:
        raise ValueError("Missing planned budgets or requested acquisition seed")
    for acquisition_seed in seeds:
        curves = []
        for arm in ("random", "boundary"):
            curve = report.get("adaptation", {}).get(arm, {}).get(str(acquisition_seed), [])
            if [point["budget"] for point in curve] != budgets:
                raise ValueError(f"Incomplete S4 curve: {arm}, seed {acquisition_seed}")
            for point in curve:
                if len(set(point["selected_ids"])) != point["budget"]:
                    raise ValueError("Selected candidate count differs from budget")
            curves.append(curve)
        if [point["charged_steps"] for point in curves[0]] != [point["charged_steps"] for point in curves[1]]:
            raise ValueError("S4 acquisition arms do not have matched charged costs")
    return int(budgets[-1])


def probe_sample_plan(lengths, max_frames=30000, stride=2, seed=0):
    """One common frame list and whole-episode split for both representations."""
    if max_frames < 1 or stride < 1:
        raise ValueError("Probe frame count and stride must be positive")
    plan, got = [], 0
    for episode, length in enumerate(lengths):
        indices = np.arange(0, int(length), stride).tolist()
        if not indices:
            continue
        plan.append({"episode": episode, "indices": indices})
        got += len(indices)
        if got >= max_frames:
            break
    episodes = [part["episode"] for part in plan]
    if len(episodes) < 2:
        raise ValueError("Fresh probe validation requires at least two source episodes")
    validation = set(np.random.default_rng(seed).choice(episodes,
                     size=max(1, int(len(episodes) * 0.2)), replace=False).tolist())
    for part in plan:
        part["role"] = "validation" if part["episode"] in validation else "training"
    return {"parts": plan, "n_frames": got, "seed": seed, "stride": stride,
            "training_episodes": [e for e in episodes if e not in validation],
            "validation_episodes": sorted(validation)}


def pinned_input(path, filenames, *, allow_local=False):
    path = Path(path)
    reference = upload_reference(path)
    if not allow_local and (reference is None or not reference.get("revision")):
        raise ValueError(f"Pinned Hugging Face receipt required for {path}")
    return {"path": str(path.resolve()), "hf": reference,
            "sha256": {name: file_sha256(path / name) for name in filenames}}


def load_predictor_variant(base_model, checkpoint, base_sha256):
    """Load only the declared predictor-side update into its exact frozen base."""
    import torch

    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    if manifest["data"]["model_sha256"] != base_sha256:
        raise ValueError("Predictor update belongs to a different base representation")
    update = torch.load(checkpoint / "weights.pt", map_location="cpu", weights_only=True)
    prefixes = {"action_encoder", "predictor", "pred_proj"}
    allowed = {k for k in base_model.state_dict() if k.split(".")[0] in prefixes}
    if set(update) != allowed:
        raise ValueError("Predictor checkpoint must contain exactly predictor-side modules and no encoder changes")
    result = copy.deepcopy(base_model)
    missing, unexpected = result.load_state_dict(update, strict=False)
    if unexpected or set(missing) != set(base_model.state_dict()) - allowed:
        raise ValueError("Incompatible predictor update")
    return result.eval()


def verify_matched_training(model_a, model_b, data_b, budget, selection_sha256):
    """Require the exact A/B intervention, including optimizer count and normalizers."""
    from omegaconf import OmegaConf

    model_a, model_b, data_b = Path(model_a), Path(model_b), Path(data_b)
    a = json.loads((model_a / "manifest.json").read_text())
    b = json.loads((model_b / "manifest.json").read_text())
    intervention = json.loads((data_b / "manifest.json").read_text())
    if intervention["data"]["selection_sha256"] != selection_sha256:
        raise ValueError("LeWM-B used different acquired experience")
    if a["data"]["sha256"] != intervention["data"]["setA_sha256"]:
        raise ValueError("LeWM-A ordinary data does not match the B intervention")
    if b["data"]["sha256"] != intervention["data"]["setB_sha256"]:
        raise ValueError("LeWM-B was not trained on the declared set B")
    if file_sha256(data_b / "setB.h5") != b["data"]["sha256"]:
        raise ValueError("Set B changed after model training")
    metrics = intervention["metrics"]
    if metrics["ordinary_steps_replaced"] != budget * 100 or metrics["branch_steps_added"] != budget * 100:
        raise ValueError("Pretraining replacement count differs from largest S4 additional budget")
    if metrics["total_A_steps"] != metrics["total_B_steps"]:
        raise ValueError("Pretraining data sizes are not equal")
    config_a = OmegaConf.to_container(OmegaConf.load(model_a / "config.yaml"))
    config_b = OmegaConf.to_container(OmegaConf.load(model_b / "config.yaml"))
    for key in ("batch", "lr", "wd", "warmup", "epochs", "seed", "clip_stride", "model"):
        if config_a[key] != config_b[key]:
            raise ValueError(f"Pretraining recipes differ: {key}")
    state_a = json.loads((model_a / "train_state.json").read_text())
    state_b = json.loads((model_b / "train_state.json").read_text())
    if state_a["step"] != state_b["step"]:
        raise ValueError("A/B optimizer budgets differ")
    if json.loads((model_a / "config.json").read_text()) != json.loads((model_b / "config.json").read_text()):
        raise ValueError("A/B architectures differ")
    with np.load(model_a / "scalers.npz") as za, np.load(model_b / "scalers.npz") as zb:
        for key in ("action_mean", "action_std"):
            np.testing.assert_array_equal(za[key], zb[key])
    return {"optimizer_steps_each": state_a["step"], "pretraining_steps_each": metrics["total_A_steps"],
            "additional_experience_steps": budget * 100,
            "same_architecture_normalization_and_initial_seed": True}
