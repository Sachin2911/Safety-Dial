"""Bounded direct event predictors with fit-only preprocessing and episode bootstrap."""

import time
from types import SimpleNamespace

import numpy as np
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from helpers.maniguard_part4_data import causal_design
from helpers.maniguard_part4_protocol import CONFIG, atomic, digest, read


def capped_windows(episode, cap=256):
    y = episode["y"]
    pos, neg = np.flatnonzero(y), np.flatnonzero(y == 0)
    if len(pos) > cap:
        raise ValueError("Positive window count exceeds the prospectively fixed cap")
    count = min(len(neg), cap - len(pos))
    chosen_neg = neg[np.linspace(0, len(neg) - 1, count, dtype=int)] if count else neg[:0]
    index = np.concatenate([pos, chosen_neg])
    if not len(index):
        return index, np.empty(0)
    inclusion = np.concatenate([np.ones(len(pos)), np.full(count, len(neg) / max(count, 1))])
    return index, inclusion / inclusion.sum()


def training_pool(episodes, designs, cap=256):
    x, y, weight = [], [], []
    for episode, design in zip(episodes, designs, strict=True):
        idx, w = capped_windows(episode, cap)
        if len(idx):
            x.append(design[idx])
            y.append(episode["y"][idx])
            weight.append(w)
    if not x:
        raise ValueError("No eligible training windows")
    return np.concatenate(x), np.concatenate(y), np.concatenate(weight)


def balance_weights(y, weights):
    positive = float(np.average(y, weights=weights))
    if not 0 < positive < 1:
        raise ValueError("Both training classes are required")
    result = weights / np.where(y, 2 * positive, 2 * (1 - positive))
    return result * len(result) / result.sum()


def mlp_fit(x, y, weights, seed, updates=500, initial=None):
    rng = np.random.default_rng(seed)
    if initial is None:
        params = [rng.normal(0, np.sqrt(2 / (x.shape[1] + 64)), (x.shape[1], 64)),
                  np.zeros(64), rng.normal(0, np.sqrt(2 / 65), (64, 1)), np.zeros(1)]
    else:
        params = [p.copy() for p in initial]
    starting = [p.copy() for p in params]
    m, v = [np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]
    probability = weights / weights.sum()
    losses = []
    for step in range(1, updates + 1):
        index = rng.choice(len(x), 256, p=probability)
        bx, by = x[index], y[index]
        hidden = np.tanh(bx @ params[0] + params[1])
        logits = (hidden @ params[2] + params[3]).ravel()
        residual = (expit(logits) - by)[:, None] / len(bx)
        dh = (residual @ params[2].T) * (1 - hidden * hidden)
        gradients = [bx.T @ dh + .001 * params[0], dh.sum(0),
                     hidden.T @ residual + .001 * params[2], residual.sum(0)]
        for i, gradient in enumerate(gradients):
            m[i] = .9 * m[i] + .1 * gradient
            v[i] = .999 * v[i] + .001 * gradient * gradient
            params[i] -= .001 * (m[i] / (1 - .9**step)) / (np.sqrt(v[i] / (1 - .999**step)) + 1e-8)
        if step == 1 or step % 50 == 0:
            losses.append({"update": step,
                           "minibatch_loss": float(np.mean(np.logaddexp(0, logits) - by * logits))})
    return params, starting, losses


def mlp_probability(x, parameters):
    return expit((np.tanh(x @ parameters[0] + parameters[1]) @ parameters[2]
                  + parameters[3]).ravel())


def fit_predictors(episodes, directory):
    directory.mkdir(parents=True, exist_ok=False)
    began = time.time()
    spec = read(CONFIG / "offline-protocol.json")["models"]
    fit = [e for e in episodes if e["group"] == "fit"]
    pca_rows = np.concatenate([e["visual"][np.linspace(
        0, len(e["visual"]) - 1, min(256, len(e["visual"])), dtype=int)] for e in fit])
    pca = PCA(n_components=32, svd_solver="randomized", random_state=20261040).fit(pca_rows)
    designs = [causal_design(e, pca) for e in fit]
    x, y, w = training_pool(fit, designs)
    scaler = StandardScaler().fit(x, sample_weight=w)
    x = scaler.transform(x)
    balanced = balance_weights(y, w)
    reference = LogisticRegression(C=1, max_iter=1000, random_state=20261040).fit(
        x, y, sample_weight=balanced)
    visual = [causal_design(e, pca, visual_only=True) for e in fit]
    vx, vy, vw = training_pool(fit, visual)
    visual_scaler = StandardScaler().fit(vx, sample_weight=vw)
    visual_reference = LogisticRegression(C=1, max_iter=1000, random_state=20261040).fit(
        visual_scaler.transform(vx), vy, sample_weight=balance_weights(vy, vw))
    arrays = {"pca_mean": pca.mean_, "pca_components": pca.components_,
              "scale_mean": scaler.mean_, "scale_std": scaler.scale_,
              "visual_mean": visual_scaler.mean_, "visual_std": visual_scaler.scale_,
              "logistic_coef": reference.coef_[0], "logistic_intercept": reference.intercept_,
              "visual_coef": visual_reference.coef_[0],
              "visual_intercept": visual_reference.intercept_}
    by_task = {task: [i for i, e in enumerate(fit) if e["task"] == task]
               for task in sorted({e["task"] for e in fit})}
    members = []
    for member, seed in enumerate(spec["seeds"]):
        rng = np.random.default_rng(seed)
        selection = [int(i) for ids in by_task.values()
                     for i in rng.choice(ids, len(ids), replace=True)]
        mx, my, mw = training_pool([fit[i] for i in selection], [designs[i] for i in selection])
        parameters, initial, losses = mlp_fit(scaler.transform(mx), my,
                                             balance_weights(my, mw), seed)
        for k, (value, start) in enumerate(zip(parameters, initial, strict=True)):
            arrays[f"member_{member}_{k}"] = value
            arrays[f"initial_{member}_{k}"] = start
        members.append({"seed": seed, "donor_episodes": [fit[i]["episode_id"] for i in selection],
                        "updates": 500, "batch_size": 256, "loss_curve": losses,
                        "sampled_windows": 500 * 256})
    np.savez_compressed(directory / "predictors.npz", **arrays)
    ledger = {"schema": "maniguard-part4-training-v1", "started": began,
              "seconds": time.time() - began, "members": members,
              "pca_rows": len(pca_rows), "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
              "fitting_episodes": [e["episode_id"] for e in fit],
              "predictors_sha256": digest(directory / "predictors.npz"),
              "protocol_sha256": digest(CONFIG / "offline-protocol.json"),
              "new_physical_controls": 0, "new_encoder_calls": 0,
              "gradient_updates": 2500, "bootstrap_member_examples": 5 * 500 * 256,
              "logistic_iterations": int(reference.n_iter_[0]),
              "visual_logistic_iterations": int(visual_reference.n_iter_[0]),
              "architecture_selection": "one frozen MLP and two frozen logistic references; no retuning"}
    atomic(directory / "training-ledger.json", ledger)
    return Predictor(directory), ledger


class Predictor:
    def __init__(self, directory):
        with np.load(directory / "predictors.npz", allow_pickle=False) as value:
            self.a = {k: value[k].copy() for k in value.files}
        # The frozen non-whitened PCA transform is explicit numerical data. Do
        # not reconstruct a partly fitted sklearn estimator or load a pickle.
        self.pca = SimpleNamespace(n_components_=len(self.a["pca_components"]),
                                   transform=self.transform_visual)

    def transform_visual(self, visual):
        return (visual - self.a["pca_mean"]) @ self.a["pca_components"].T

    def predict(self, episode):
        a = self.a
        x = (causal_design(episode, self.pca) - a["scale_mean"]) / a["scale_std"]
        vx = (causal_design(episode, self.pca, True) - a["visual_mean"]) / a["visual_std"]
        members = np.asarray([mlp_probability(x, [a[f"member_{m}_{k}"] for k in range(4)])
                              for m in range(5)])
        return {"primary": members.mean(0), "uncertainty": members.std(0), "members": members,
                "logistic": expit(x @ a["logistic_coef"] + a["logistic_intercept"]),
                "visual": expit(vx @ a["visual_coef"] + a["visual_intercept"])}


def calibration_rule(raw, y, episode_weights, disagreement, target=.05):
    if len(np.unique(y)) != 2:
        raise ValueError("Calibration has insufficient class support")
    logits = logit(np.clip(raw, 1e-8, 1 - 1e-8))[:, None]
    platt = LogisticRegression(C=1, max_iter=1000).fit(logits, y, sample_weight=(
        episode_weights * len(y) / episode_weights.sum()))
    p = platt.predict_proba(logits)[:, 1]
    negatives = y == 0
    values = np.unique(p[negatives])
    candidates = np.r_[values, np.nextafter(values[-1], np.inf)]
    # Sort once; strict nextafter handling makes all tied scores explicit.
    order = np.argsort(p[negatives])
    scores, weights = p[negatives][order], episode_weights[negatives][order]
    cumulative = np.r_[0, np.cumsum(weights)]
    positions = np.searchsorted(scores, candidates, side="left")
    fpr = (cumulative[-1] - cumulative[positions]) / cumulative[-1]
    index = int(np.flatnonzero(fpr <= target + 1e-12)[0])
    threshold = float(candidates[index])
    return {"slope": float(platt.coef_[0, 0]), "intercept": float(platt.intercept_[0]),
            "threshold": threshold, "calibration_negative_window_false_alarm": float(fpr[index]),
            "disagreement_cutoff": float(np.quantile(disagreement[negatives], .25)),
            "low_risk_cutoff": .1, "calibration_windows": len(y),
            "calibration_positive_windows": int(y.sum()), "probability": p}


def calibrated(raw, rule):
    return expit(rule["slope"] * logit(np.clip(raw, 1e-8, 1 - 1e-8)) + rule["intercept"])


def event_metrics(episode, p, u, rule, horizon=8, deadline=4, history=4):
    steps, tau = episode["steps"], episode["onset"]
    result = {"episode_id": episode["episode_id"], "task": episode["task"],
              "onset": tau, "event_objects": episode.get("event_objects", []),
              "eligible_event": False, "timely_warning": None, "confident_miss": False,
              "quality": 0.0, "event_exclusion": None}
    alarms = p >= rule["threshold"]
    negative = episode["y"] == 0
    result["negative_windows"] = int(negative.sum())
    result["false_alarm_windows"] = int((negative & alarms).sum())
    result["episode_false_alarm"] = bool((negative & alarms).any()) if negative.any() else None
    if tau is None:
        result["event_exclusion"] = "no_upright_event"
        return result
    if tau < horizon + history:
        result["event_exclusion"] = "initial_violation" if tau == 0 else "insufficient_pre_event_history"
        return result
    expected = np.arange(tau - horizon, tau - deadline + 1)
    index = np.flatnonzero(np.isin(steps, expected))
    if len(index) != len(expected):
        result["event_exclusion"] = "incomplete_recorded_future_for_warning_window"
        return result
    warned = bool(alarms[index].any())
    at_deadline = int(index[-1])
    low = bool(p[at_deadline] <= rule["low_risk_cutoff"])
    agreement = bool(u[at_deadline] <= rule["disagreement_cutoff"])
    confident = not warned and low and agreement
    prior_alarms = steps[(steps < tau) & alarms]
    timely_alarms = steps[index][alarms[index]]
    lead = int(tau - timely_alarms[0]) if len(timely_alarms) else None
    result.update(eligible_event=True, timely_warning=warned, confident_miss=confident,
                  probability_at_deadline=float(p[at_deadline]),
                  disagreement_at_deadline=float(u[at_deadline]),
                  late_warning=bool(not warned and np.any(prior_alarms > tau - deadline)),
                  warning_lead_actions=lead, low_risk=low, low_disagreement=agreement,
                  quality=float(1 - p[at_deadline]) if confident else 0.0)
    return result


def discrimination(y, p):
    if len(np.unique(y)) != 2:
        return {"auroc": None, "auprc": None, "prevalence": float(np.mean(y)) if len(y) else None}
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)), "prevalence": float(y.mean())}


def repair_predictor(baseline_directory, episodes, directory, arm):
    """Matched offline updates from identical trained baselines, never audit data.

    The caller supplies historical fitting episodes, every valid shared initial
    trajectory and every valid discovery for this arm. Confirmation and calibration
    trajectories remain excluded. This conditional stage is uncalled when the
    premise gate fails; unit-test execution is not physical repair evidence.
    """
    if arm not in ("common", "R", "U", "E", "D"):
        raise ValueError("Unknown repair arm")
    for e in episodes:
        if e.get("role") not in ("historical_fit", "shared_initial", "acquisition"):
            raise ValueError("Repair donor is not permitted fitting/acquisition data")
        if e["role"] == "acquisition" and (arm == "common" or e.get("method") != arm):
            raise ValueError("An arm cannot use another method's private discoveries")
        if e.get("group") in ("calibration", "development", "audit", "confirmation"):
            raise ValueError("Held-out groups cannot enter repair fitting")
        if e.get("valid") is False:
            raise ValueError("Invalid physical proposals are not training trajectories")
    baseline = Predictor(baseline_directory)
    a = {k: v.copy() for k, v in baseline.a.items()}
    designs = [causal_design(e, baseline.pca) for e in episodes]
    by_task = {task: [i for i, e in enumerate(episodes) if e["task"] == task]
               for task in sorted({e["task"] for e in episodes})}
    members = []
    for member, seed in enumerate(read(CONFIG / "offline-protocol.json")["models"]["seeds"]):
        rng = np.random.default_rng(seed)
        selection = [int(i) for ids in by_task.values()
                     for i in rng.choice(ids, len(ids), replace=True)]
        x, y, w = training_pool([episodes[i] for i in selection], [designs[i] for i in selection])
        x = (x - a["scale_mean"]) / a["scale_std"]
        initial = [baseline.a[f"member_{member}_{k}"] for k in range(4)]
        parameters, _, losses = mlp_fit(x, y, balance_weights(y, w), seed, initial=initial)
        for k, value in enumerate(parameters):
            a[f"initial_{member}_{k}"] = initial[k].copy()
            a[f"member_{member}_{k}"] = value
        members.append({"seed": seed, "updates": 500, "loss_curve": losses,
                        "bootstrap": [episodes[i]["episode_id"] for i in selection]})
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "predictors.npz", **a)
    atomic(directory / "repair-ledger.json", {
        "arm": arm, "baseline_sha256": digest(baseline_directory / "predictors.npz"),
        "preprocessing": "baseline PCA/scalers unchanged", "members": members,
        "donors": [{k: e.get(k) for k in ("episode_id", "task", "role", "method")} for e in episodes],
        "missing_valid_data": "no cross-arm replacement; episode-normalized capped contributions",
        "logistic_references": "baseline references preserved; primary comparison updates only MLP copies",
        "gradient_updates": 2500, "examples": 5 * 500 * 256,
        "predictors_sha256": digest(directory / "predictors.npz"), "audit_accessed": False,
    })
    return Predictor(directory)
