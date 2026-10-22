"""Compact executed-action dynamics and factual correction-error calibration.

All preprocessing and regressors are fitted on whole fitting episodes only. Audit
trajectories are first opened after artifacts and calibration have been frozen.
"""

import json
import time
from pathlib import Path

import numpy as np

from helpers.maniguard_followup_budget import atomic, clock
from helpers.maniguard_part3_data import load_group
from helpers.maniguard_part3_protocol import criteria
from helpers.maniguard_phase3_protocol import sha


def ridge(x, y, alpha=1.0):
    x = np.asarray(x, np.float64)
    design = np.column_stack([x, np.ones(len(x))])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[-1, -1] = 0
    return np.linalg.solve(design.T @ design + penalty, design.T @ np.asarray(y, np.float64))


def linear(x, weights):
    return np.column_stack([x, np.ones(len(x))]) @ weights


def randomized_pca(x, dimensions=32, seed=20261020):
    """Deterministic randomized PCA, 32 oversamples and two power iterations."""
    rng = np.random.default_rng(seed)
    width = min(dimensions + 32, min(x.shape))
    omega = rng.normal(size=(x.shape[1], width)).astype(np.float32)
    q = np.linalg.qr(x @ omega, mode="reduced")[0]
    for _ in range(2):
        q = np.linalg.qr(x @ (x.T @ q), mode="reduced")[0]
    _, singular, vt = np.linalg.svd(q.T @ x, full_matrices=False)
    return vt[:dimensions].T.astype(np.float32), singular[:dimensions]


def fit_representation(episodes):
    v = np.concatenate([e["visual"] for e in episodes])
    robot = np.concatenate([e["robot"] for e in episodes])
    mean, scale = v.mean(0), np.maximum(v.std(0), 1e-5)
    standardized = (v - mean) / scale
    projection, singular = randomized_pca(standardized)
    state = np.column_stack([standardized @ projection, robot])
    state_mean, state_scale = state.mean(0), np.maximum(state.std(0), 1e-4)
    random = np.random.default_rng(20261022).normal(
        size=(v.shape[1], 32), scale=1 / np.sqrt(32),
    ).astype(np.float32)
    return {"visual_mean": mean, "visual_scale": scale, "projection": projection,
            "random_projection": random, "state_mean": state_mean, "state_scale": state_scale,
            "robot_mean": robot.mean(0), "robot_scale": np.maximum(robot.std(0), 1e-4),
            "pca_explained_fraction": np.asarray(float(np.sum(singular ** 2) /
                                                     np.sum(standardized ** 2)))}


def state_for(episode, artifacts, representation="pca"):
    visual = (episode["visual"] - artifacts["visual_mean"]) / artifacts["visual_scale"]
    robot = episode["robot"]
    if representation == "pca":
        x = np.column_stack([visual @ artifacts["projection"], robot])
        return ((x - artifacts["state_mean"]) / artifacts["state_scale"]).astype(np.float32)
    if representation == "random":
        visual = visual @ artifacts["random_projection"]
    return np.column_stack([visual, (robot - artifacts["robot_mean"]) /
                            artifacts["robot_scale"]]).astype(np.float32)


def design_for(episode, artifacts):
    x = state_for(episode, artifacts)
    u = (episode["action"] - artifacts["action_mean"]) / artifacts["action_scale"]
    # Omit each episode's initial transition, which has no recorded causal predecessor.
    indices = np.arange(1, len(x) - 1)
    design = np.column_stack([x[indices], x[indices - 1], u[indices - 1], u[indices]])
    return design.astype(np.float32), (x[indices + 1] - x[indices]).astype(np.float32)


def network(input_dim=98, output_dim=41):
    import torch
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 128), torch.nn.ReLU(), torch.nn.Linear(128, output_dim),
    )


def train(features, directory, device="cuda", epochs=30):
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    fit = load_group(features, "fit")
    if len(fit) < criteria()["minimum_full_episodes"]["fit"]:
        raise ValueError("Insufficient fitting episodes")
    artifacts = fit_representation(fit)
    actions = np.concatenate([e["action"] for e in fit])
    artifacts.update(action_mean=actions.mean(0), action_scale=np.maximum(actions.std(0), 1e-3))
    all_x, all_y = [], []
    for e in fit:
        x, y = design_for(e, artifacts)
        all_x.append(x)
        all_y.append(y)
    x, y = np.concatenate(all_x), np.concatenate(all_y)
    artifacts["dynamics_ridge"] = ridge(x, y)
    # Shuffling executed actions on fitting records tests action dependence without
    # exposing calibration/audit targets. State/history columns remain correctly aligned.
    shuffled = x.copy()
    permutation = np.random.default_rng(20261023).permutation(len(x))
    shuffled[:, -8:] = shuffled[permutation, -8:]
    artifacts["action_shuffled_ridge"] = ridge(shuffled, y)
    labels = np.concatenate([e["target"] for e in fit])
    representation_seconds = time.monotonic() - started
    for name in ("pca", "raw", "random"):
        observations = np.concatenate([state_for(e, artifacts, name) for e in fit])
        artifacts["readout_" + name] = ridge(observations, labels)
    visual_only = np.concatenate([state_for(e, artifacts)[:, :32] for e in fit])
    artifacts["readout_visual_only"] = ridge(visual_only, labels)
    # Readout labels never enter the dynamics inputs or loss.
    np.savez_compressed(directory / "artifacts.npz", **artifacts)
    ledger = {"fit_episodes": [e["episode_id"] for e in fit], "fit_transitions": len(x),
              "epochs_each": epochs, "members": 5, "optimizer_steps": 0,
              "training_member_examples": 0, "member_losses": [],
              "representation_seconds": representation_seconds,
              "started_at": time.time(), "device": device, "torch": torch.__version__,
              "pca_explained_fraction": float(artifacts["pca_explained_fraction"])}
    weights = []
    for member in range(5):
        torch.manual_seed(20261021 + member)
        rng = np.random.default_rng(20261021 + member)
        sampled_episodes = rng.integers(0, len(fit), size=len(fit))
        boot_x = np.concatenate([all_x[i] for i in sampled_episodes])
        boot_y = np.concatenate([all_y[i] for i in sampled_episodes])
        tx, ty = torch.as_tensor(boot_x, device=device), torch.as_tensor(boot_y, device=device)
        model = network(x.shape[1], y.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(epochs):
            if clock()["physical_remaining"] < 45:
                raise RuntimeError("Model fitting reached cleanup reserve")
            order = rng.permutation(len(boot_x))
            total = 0.0
            model.train()
            for offset in range(0, len(order), 256):
                index = torch.as_tensor(order[offset:offset + 256], device=device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(tx[index])
                loss = torch.nn.functional.mse_loss(prediction, ty[index])
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(index)
                ledger["optimizer_steps"] += 1
                ledger["training_member_examples"] += len(index)
            ledger["member_losses"].append({"member": member, "epoch": epoch,
                                             "mse": total / len(order)})
            atomic(directory / "training-ledger.json", ledger)
        weights.append({k: v.detach().cpu() for k, v in model.state_dict().items()})
        ledger.setdefault("episode_bootstraps", []).append(sampled_episodes.tolist())
    torch.save(weights, directory / "ensemble.pt")
    ledger.update(seconds=time.monotonic() - started,
                  peak_gpu_bytes=int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0)
    atomic(directory / "training-ledger.json", ledger)
    atomic(directory / "trained-artifact-receipt.json", {
        "status": "frozen_before_calibration_and_forecast_audit",
        "files": {p.name: sha(p) for p in directory.iterdir() if p.is_file()},
        "features_manifest_sha256": sha(Path(features) / "corpus.json"),
    })
    return ledger


class Predictor:
    def __init__(self, directory, device="cpu"):
        import torch
        self.directory = Path(directory)
        self.device = device
        with np.load(self.directory / "artifacts.npz") as data:
            self.artifacts = {key: data[key].copy() for key in data.files}
        self.models = []
        for state in torch.load(self.directory / "ensemble.pt", map_location="cpu", weights_only=True):
            model = network().to(device).eval()
            model.load_state_dict(state)
            self.models.append(model)
        self.member_calls = 0
        self.member_examples = 0

    def residual(self, x, previous, prior_action, action):
        import torch
        a = self.artifacts
        design = np.column_stack([x, previous,
                                  (prior_action - a["action_mean"]) / a["action_scale"],
                                  (action - a["action_mean"]) / a["action_scale"]]).astype(np.float32)
        tensor = torch.as_tensor(design, device=self.device)
        with torch.inference_mode():
            predictions = np.stack([m(tensor).cpu().numpy() for m in self.models])
        self.member_calls += len(self.models)
        self.member_examples += len(self.models) * len(design)
        return predictions

    def rollout(self, x, previous, prior_action, actions):
        import torch
        a = self.artifacts
        current = np.repeat(x[None], 5, axis=0).copy()
        prev = np.repeat(previous[None], 5, axis=0).copy()
        uprev = prior_action
        trajectory = []
        for index in range(actions.shape[1]):
            results = []
            for i, model in enumerate(self.models):
                design = np.column_stack([
                    current[i], prev[i], (uprev - a["action_mean"]) / a["action_scale"],
                    (actions[:, index] - a["action_mean"]) / a["action_scale"],
                ]).astype(np.float32)
                with torch.inference_mode():
                    delta = model(torch.as_tensor(design, device=self.device)).cpu().numpy()
                results.append(current[i] + delta)
                self.member_calls += 1
                self.member_examples += len(design)
            prev, current = current, np.stack(results)
            uprev = actions[:, index]
            trajectory.append(current.copy())
        return np.stack(trajectory, axis=2)  # member, window, horizon, coordinates


def windows(episodes, artifacts):
    rows = {key: [] for key in ("x", "previous", "prior_action", "actions", "future",
                               "target", "initial_target", "correction", "episode", "scene", "step")}
    for episode in episodes:
        x = state_for(episode, artifacts)
        for t in np.flatnonzero(episode["query"]):
            if t < 1 or t + 4 >= len(x):
                continue
            rows["x"].append(x[t])
            rows["previous"].append(x[t - 1])
            rows["prior_action"].append(episode["action"][t - 1])
            rows["actions"].append(episode["action"][t:t + 4])
            rows["future"].append(x[t + 1:t + 5])
            rows["target"].append(episode["target"][t + 4])
            rows["initial_target"].append(episode["target"][t])
            rows["correction"].append(float(episode["correction"][t:t + 4].sum()))
            rows["episode"].append(episode["episode_id"])
            rows["scene"].append(episode["scene"])
            rows["step"].append(t)
    if not rows["x"]:
        raise ValueError("No complete four-action query-start windows")
    return {key: np.asarray(value) for key, value in rows.items()}


def ridge_rollout(w, artifacts, key):
    x, previous, uprev = w["x"].copy(), w["previous"].copy(), w["prior_action"]
    out = []
    for index in range(4):
        u = w["actions"][:, index]
        design = np.column_stack([x, previous,
                                  (uprev - artifacts["action_mean"]) / artifacts["action_scale"],
                                  (u - artifacts["action_mean"]) / artifacts["action_scale"]])
        next_x = x + linear(design, artifacts[key])
        previous, x, uprev = x, next_x, u
        out.append(x)
    return np.stack(out, axis=1)


def empirical_quantile(values, fraction):
    if len(values) == 0:
        raise ValueError("Missing calibration support")
    return float(np.quantile(values, fraction))


def trust_scores(disagreement, correction, calibration):
    bins = np.asarray(calibration["bins"])
    indices = np.clip(np.digitize(correction, bins) - 1, 0, len(bins) - 2)
    conditional = np.asarray(calibration["error_quantiles"])[indices]
    score = conditional * (0.5 + disagreement / calibration["disagreement_median_scale"])
    return disagreement, score


def calibrate(features, directory, device="cpu"):
    directory = Path(directory)
    predictor = Predictor(directory, device)
    episodes = load_group(features, "calibration")
    fit = load_group(features, "fit")
    w = windows(episodes, predictor.artifacts)
    predicted = predictor.rollout(w["x"], w["previous"], w["prior_action"], w["actions"])
    mean = predicted.mean(0)
    error = np.sqrt(np.mean((mean[:, 3, :32] - w["future"][:, 3, :32]) ** 2, axis=1))
    disagreement = np.mean(np.var(predicted[:, :, 3, :32], axis=0), axis=1)
    bins = [0, 0.5, 1, 2, 4, 8, 1000000]
    indices = np.clip(np.digitize(w["correction"], bins) - 1, 0, len(bins) - 2)
    unconditional = empirical_quantile(error, 0.8)
    quantiles, support = [], []
    for i in range(len(bins) - 1):
        selected = error[indices == i]
        support.append(len(selected))
        quantiles.append(empirical_quantile(selected, 0.8) if len(selected) >= 30 else unconditional)
    fw = windows(fit, predictor.artifacts)
    changing = np.sqrt(np.mean((fw["future"][:, 3, :32] - fw["x"][:, :32]) ** 2, axis=1))
    calibration = {"schema": "maniguard-part3-calibration-v1", "status": "frozen",
                   "episodes": [e["episode_id"] for e in episodes], "windows": len(error),
                   "bins": bins, "error_quantiles": quantiles, "bin_windows": support,
                   "unconditional_error_quantile": unconditional,
                   "minimum_bin_windows": 30, "fallback": "unconditional .8 quantile",
                   "nominal_acceptance_fraction": 0.5,
                   "disagreement_median_scale": max(empirical_quantile(disagreement, 0.5), 1e-8),
                   "fitting_changing_threshold": empirical_quantile(changing, 0.5),
                   "correction_support": [float(w["correction"].min()), float(w["correction"].max())],
                   "state_norm_limit": empirical_quantile(np.linalg.norm(w["x"], axis=1), 0.95)}
    m, c = trust_scores(disagreement, w["correction"], calibration)
    calibration.update(M_threshold=empirical_quantile(m, 0.5),
                       C_threshold=empirical_quantile(c, 0.5),
                       M_nominal_actual_fraction=float(np.mean(m <= np.quantile(m, 0.5))),
                       C_nominal_actual_fraction=float(np.mean(c <= np.quantile(c, 0.5))),
                       member_calls=predictor.member_calls, member_examples=predictor.member_examples)
    np.savez_compressed(directory / "calibration-windows.npz", error=error,
                        disagreement=disagreement, correction=w["correction"], episode=w["episode"])
    atomic(directory / "calibration.json", calibration)
    atomic(directory / "forecast-freeze.json", {
        "status": "sealed_before_opening_forecast_audit", "at": time.time(),
        "criteria": criteria(),
        "files": {name: sha(directory / name) for name in
                  ("artifacts.npz", "ensemble.pt", "calibration.json", "calibration-windows.npz")},
    })
    return calibration


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    out = np.empty(len(values), float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        out[order[start:end]] = (start + end - 1) / 2
        start = end
    return out


def correlation(a, b):
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def rmse_per_window(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


def check_gate(metrics, threshold=None):
    c = criteria() if threshold is None else threshold
    support = {
        "audit_episodes": min(metrics["audit_episodes"], metrics.get("audit_episodes_with_windows", metrics["audit_episodes"])) >= c["minimum_full_episodes"]["audit"],
        "changing_windows": metrics["changing_windows"] >= c["minimum_changing_windows_audit"],
        "near_calibration_episodes": metrics["near_calibration_episodes"] >= c["minimum_near_threshold_episodes_calibration"],
        "near_audit_episodes": metrics["near_audit_episodes"] >= c["minimum_near_threshold_episodes_audit"],
        "crossing_audit_episodes": metrics["crossing_audit_episodes"] >= c["minimum_upright_crossing_episodes_audit"],
        "varying_progress_windows": metrics["varying_progress_windows"] >= c["minimum_progress_varying_windows"],
        "crossing_forecast_windows": metrics.get("crossing_windows", int(metrics["false_reassurance_fraction"] is not None)) > 0,
    }
    accuracy = {
        "one_step_beats_persistence": metrics["persistence_improvement_1"] >= c["minimum_persistence_improvement_fraction_1_and_4"],
        "four_step_beats_persistence": metrics["persistence_improvement_4"] >= c["minimum_persistence_improvement_fraction_1_and_4"],
        "action_dependence": metrics["shuffled_improvement_4"] >= c["minimum_action_shuffled_improvement_fraction_4"],
        "ridge_reference": metrics["relative_ridge_error_4"] <= c["maximum_relative_ridge_error_4"],
        "actual_readout": metrics["actual_margin_mae_deg"] <= c["maximum_actual_latent_margin_mae_deg"],
        "near_forecast_readout": metrics["near_margin_mae_deg"] is not None and metrics["near_margin_mae_deg"] <= c["maximum_predicted_latent_near_margin_mae_deg"],
        "progress_readout": metrics["progress_mae_m"] <= c["maximum_predicted_progress_mae_m"],
        "false_reassurance": metrics["false_reassurance_fraction"] is not None and metrics["false_reassurance_fraction"] <= c["maximum_false_reassurance_fraction"],
        "ranking": metrics["progress_ranking_spearman"] is not None and metrics["progress_ranking_spearman"] >= c["minimum_progress_ranking_spearman"],
        "M_acceptance": metrics["M_acceptance"] >= c["minimum_actual_acceptance_fraction_audit_each"],
        "C_acceptance": metrics["C_acceptance"] >= c["minimum_actual_acceptance_fraction_audit_each"],
        "latency": metrics["screen_seconds_per_16"] <= c["maximum_screening_seconds_per_16_candidates"],
    }
    return {"eligible": all(support.values()) and all(accuracy.values()),
            "status": "eligible" if all(support.values()) and all(accuracy.values()) else
                      "inconclusive_support" if not all(support.values()) else "failed_accuracy",
            "support": support, "accuracy": accuracy,
            "failed_checks": [key for key, value in {**support, **accuracy}.items() if not value]}


def audit(features, directory, device="cpu"):
    directory = Path(directory)
    freeze = json.loads((directory / "forecast-freeze.json").read_text())
    if freeze["criteria"] != criteria():
        raise ValueError("Forecast criteria changed")
    for name, digest in freeze["files"].items():
        if sha(directory / name) != digest:
            raise ValueError("Frozen predictive artifact changed")
    predictor = Predictor(directory, device)
    a = predictor.artifacts
    episodes = load_group(features, "audit")
    calibration_episodes = load_group(features, "calibration")
    w = windows(episodes, a)
    started = time.monotonic()
    predicted = predictor.rollout(w["x"], w["previous"], w["prior_action"], w["actions"])
    elapsed = time.monotonic() - started
    # Matched action-dependence negative control: the same trained ensemble sees
    # a deterministically permuted four-action sequence from another audit window.
    # Images/history and factual future targets stay aligned. These deliberately
    # mispaired forecasts are not claimed as counterfactual physical outcomes.
    permutation = np.random.default_rng(20261029).permutation(len(w["actions"]))
    shuffled_ensemble = predictor.rollout(
        w["x"], w["previous"], w["prior_action"], w["actions"][permutation],
    ).mean(0)
    mean = predicted.mean(0)
    error1 = rmse_per_window(mean[:, 0, :32], w["future"][:, 0, :32])
    error4 = rmse_per_window(mean[:, 3, :32], w["future"][:, 3, :32])
    persistence1 = rmse_per_window(w["x"][:, :32], w["future"][:, 0, :32])
    persistence4 = rmse_per_window(w["x"][:, :32], w["future"][:, 3, :32])
    ridge_prediction = ridge_rollout(w, a, "dynamics_ridge")
    shuffled_prediction = ridge_rollout(w, a, "action_shuffled_ridge")
    ridge_error = rmse_per_window(ridge_prediction[:, 3, :32], w["future"][:, 3, :32])
    shuffled_error = rmse_per_window(shuffled_prediction[:, 3, :32], w["future"][:, 3, :32])
    shuffled_ensemble_error = rmse_per_window(shuffled_ensemble[:, 3, :32], w["future"][:, 3, :32])
    calibration = json.loads((directory / "calibration.json").read_text())
    changing = persistence4 > calibration["fitting_changing_threshold"]
    # Report full and changing windows separately; gate the latter to prevent static wins.
    selected = changing if changing.any() else np.ones(len(error4), bool)
    actual_head = linear(w["future"][:, 3], a["readout_pca"])
    forecast_head = linear(mean[:, 3], a["readout_pca"])
    near = w["target"][:, 0] <= criteria()["near_threshold_margin_deg"]
    crossings = w["target"][:, 0] < 0
    disagreement = np.mean(np.var(predicted[:, :, 3, :32], axis=0), axis=1)
    score_m, score_c = trust_scores(disagreement, w["correction"], calibration)
    support = ((w["correction"] >= calibration["correction_support"][0]) &
               (w["correction"] <= calibration["correction_support"][1]) &
               (np.linalg.norm(w["x"], axis=1) <= calibration["state_norm_limit"]))
    accepted_m = (score_m <= calibration["M_threshold"]) & support
    accepted_c = (score_c <= calibration["C_threshold"]) & support
    varying = np.abs(w["target"][:, 1] - w["initial_target"][:, 1]) > 0.001
    current_head = linear(w["x"], a["readout_pca"])
    progress_rank = correlation((forecast_head[:, 1] - current_head[:, 1])[varying],
                                (w["target"][:, 1] - w["initial_target"][:, 1])[varying])
    def episode_count(group, threshold):
        return sum(bool(np.any(e["target"][:, 0] <= threshold)) for e in group)

    def improvement(error, reference):
        return float(1 - np.mean(error[selected]) / max(np.mean(reference[selected]), 1e-12))

    metrics = {
        "audit_episodes": len(episodes), "audit_windows": len(error4),
        "audit_episodes_with_windows": len(set(w["episode"].tolist())),
        "changing_windows": int(changing.sum()), "varying_progress_windows": int(varying.sum()),
        "near_calibration_episodes": episode_count(calibration_episodes, 10),
        "near_audit_episodes": episode_count(episodes, 10),
        "crossing_audit_episodes": episode_count(episodes, -1e-12),
        "near_windows": int(near.sum()), "crossing_windows": int(crossings.sum()),
        "persistence_improvement_1": improvement(error1, persistence1),
        "persistence_improvement_4": improvement(error4, persistence4),
        "shuffled_improvement_4": improvement(error4, shuffled_ensemble_error),
        "action_shuffle_seed": 20261029,
        "action_shuffle_scope": "same ensemble, mispaired executed-action prefixes within audit partition",
        "relative_ridge_error_4": float(np.mean(error4[selected]) / max(np.mean(ridge_error[selected]), 1e-12)),
        "actual_margin_mae_deg": float(np.mean(np.abs(actual_head[:, 0] - w["target"][:, 0]))),
        "near_margin_mae_deg": float(np.mean(np.abs(forecast_head[near, 0] - w["target"][near, 0]))) if near.any() else None,
        "progress_mae_m": float(np.mean(np.abs(forecast_head[:, 1] - w["target"][:, 1]))),
        "false_reassurance_fraction": float(np.mean(forecast_head[crossings, 0] > 5)) if crossings.any() else None,
        "progress_ranking_spearman": progress_rank,
        "M_acceptance": float(accepted_m.mean()), "C_acceptance": float(accepted_c.mean()),
        "M_trusted_rmse": float(error4[accepted_m].mean()) if accepted_m.any() else None,
        "C_trusted_rmse": float(error4[accepted_c].mean()) if accepted_c.any() else None,
        "screen_seconds_per_16": elapsed / len(error4) * 16 * 32,
        "member_calls": predictor.member_calls, "member_examples": predictor.member_examples,
        "rollout_seconds": elapsed, "scenes": sorted(set(w["scene"].tolist())),
        "readout_comparison": {}, "per_episode": [],
    }
    for name in ("pca", "raw", "random"):
        observed = np.concatenate([state_for(e, a, name) for e in episodes])
        target = np.concatenate([e["target"] for e in episodes])
        head = linear(observed, a["readout_" + name])
        metrics["readout_comparison"][name] = {
            "margin_mae_deg": float(np.mean(np.abs(head[:, 0] - target[:, 0]))),
            "progress_mae_m": float(np.mean(np.abs(head[:, 1] - target[:, 1]))),
        }
    visual_head = linear(w["future"][:, 3, :32], a["readout_visual_only"])
    metrics["readout_comparison"]["pca_visual_only"] = {
        "margin_mae_deg": float(np.mean(np.abs(visual_head[:, 0] - w["target"][:, 0]))),
        "progress_mae_m": float(np.mean(np.abs(visual_head[:, 1] - w["target"][:, 1]))),
    }
    for episode in episodes:
        mask = w["episode"] == episode["episode_id"]
        metrics["per_episode"].append({
            "episode_id": episode["episode_id"], "scene": episode["scene"],
            "windows": int(mask.sum()), "near_windows": int((mask & near).sum()),
            "four_step_rmse": float(error4[mask].mean()) if mask.any() else None,
            "persistence_rmse": float(persistence4[mask].mean()) if mask.any() else None,
            "correction_mean": float(w["correction"][mask].mean()) if mask.any() else None,
        })
    for label, values in (("ensemble_1", error1), ("ensemble_4", error4),
                          ("persistence_1", persistence1), ("persistence_4", persistence4),
                          ("ridge_4", ridge_error), ("action_shuffled_ridge_4", shuffled_error),
                          ("action_shuffled_ensemble_4", shuffled_ensemble_error)):
        metrics[label + "_rmse"] = float(values.mean())
        metrics[label + "_changing_rmse"] = float(values[selected].mean())
    # Benchmark the actual acquisition code, including controller rollouts, on a
    # prospective 16-genotype grid. It creates no physical fitness or policy calls.
    from helpers.maniguard_part3_acquisition import screen
    benchmark_pool = [{"genome": [i // 4, i % 4, 4, 0], "parent": None, "birth": i}
                      for i in range(16)]
    benchmark = screen(benchmark_pool, directory, device)
    atomic(directory / "acquisition-engineering.json", benchmark)
    metrics["screen_seconds_per_16"] = benchmark["seconds"]
    metrics["member_calls"] += benchmark["member_calls"]
    metrics["member_examples"] += benchmark["member_examples"]
    result = {"schema": "maniguard-part3-forecast-audit-v1", "at": time.time(),
              "criteria": criteria(), "metrics": metrics, "gate": check_gate(metrics),
              "scope": "factual query-start prefixes on one held-out task; no optimizer superiority or counterfactual causal ranking claim"}
    np.savez_compressed(directory / "forecast-windows.npz", error1=error1, error4=error4,
                        persistence4=persistence4, ridge4=ridge_error, shuffled4=shuffled_ensemble_error,
                        shuffled_ridge4=shuffled_error, action_permutation=permutation,
                        disagreement=disagreement, correction=w["correction"],
                        actual_target=w["target"], forecast_target=forecast_head,
                        actual_latent_head=actual_head, accepted_m=accepted_m, accepted_c=accepted_c,
                        episode=w["episode"], scene=w["scene"], step=w["step"], changing=changing)
    atomic(directory / "forecast-audit.json", result)
    return result
