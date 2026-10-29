"""Shared 36-coefficient selector and transparent optimizer updates for task 17."""

import numpy as np

MODES = {
    "tworoom": [
        (2, 1, 0, 0),
        (2, 2, 0.5, 0.5),
        (2, 2, 0.5, 0),
        (3, 1, 0.25, 0.5),
        (2, 1, 0.25, 0.5),
        (3, 2, 0.5, 0.5),
    ],
    "reacher": [
        (3, 1, 0.25, 0.5),
        (3, 1, 0.5, 0),
        (2, 1, 0.25, 0.5),
        (3, 2, 0.5, 0.5),
        (2, 2, 0.5, 0),
        (5, 1, 0, 0),
    ],
}
MARGINS = {"tworoom": [4.0, 6.0], "reacher": [0.01, 0.015]}
FEATURE_KEYS = {"current_cost", "improvement", "prediction_error", "previous_feasible", "replan"}


def features_vector(features, env, margin):
    if set(features) != FEATURE_KEYS:
        raise ValueError("Unexpected task-side feature keys")
    if margin not in MARGINS[env]:
        raise ValueError("Requested clearance is outside the frozen panel")
    x = np.array(
        [
            1.0,
            np.clip(features["improvement"], -1, 1),
            np.clip(features["prediction_error"], 0, 2) / 2,
            np.clip(features["previous_feasible"], 0, 1),
            min(features["replan"], 9) / 9,
            margin / max(MARGINS[env]),
        ]
    )
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite selector observation")
    return x


def probabilities(weights, x):
    w = np.asarray(weights, dtype=float)
    if w.shape != (6, 6) or not np.isfinite(w).all() or np.abs(w).max() > 3.00000001:
        raise ValueError("Selector must have 36 finite coefficients bounded by 3")
    logits = w @ x
    p = np.exp(logits - logits.max())
    return p / p.sum()


def select_mode(features, weights, env, margin, rng, stochastic):
    x = features_vector(features, env, margin)
    p = probabilities(weights, x)
    mode = int(rng.choice(6, p=p)) if stochastic else int(p.argmax())
    return {
        "x": x.tolist(),
        "probabilities": p.tolist(),
        "mode": mode,
        "planning_mode": list(MODES[env][mode]),
        "stochastic": bool(stochastic),
    }


def mode_params(env, mode):
    h, variance, retain, correlate = MODES[env][mode]
    return {"horizon": h, "variance": variance, "retain": retain, "correlate": correlate}


def anchors(env):
    result = []
    for mode in range(6):
        w = np.zeros((6, 6))
        w[mode, 0] = 3.0
        result.append(w)
    # At d/max = 2/3 and 1 this switches modes at 5/6.
    # Task-16 winning margin routing and its reversed routing are shared anchors.
    low, high = (1, 2) if env == "tworoom" else (0, 1)
    for a, b in [(low, high), (high, low)]:
        w = np.zeros((6, 6))
        w[:, 0] = -3
        w[a, 0], w[a, 5] = 2.5, -3
        w[b, 0], w[b, 5] = -2.5, 3
        result.append(w)
    return result


def utility(result):
    return float(result["requested_safe_completion"]) - 0.1 * result["J"] / 50


def reinforce_gradient(episodes, baseline):
    """Mean terminal-reward score gradient; no privileged reward or critic."""
    gradient = np.zeros((6, 6))
    entropies = []
    for reward, decisions in episodes:
        score = np.zeros((6, 6))
        for decision in decisions:
            p = np.asarray(decision["probabilities"])
            onehot = np.eye(6)[decision["mode"]]
            score += np.outer(onehot - p, decision["x"])
            entropies.append(float(-(p * np.log(np.maximum(p, 1e-30))).sum()))
        gradient += (reward - baseline) * score
    return gradient / max(1, len(episodes)), float(np.mean(entropies)) if entropies else 0.0


def es_gradient(directions, rewards, sigma=0.3):
    # Candidates are +eps, -eps for each independent Gaussian direction.
    difference = np.asarray(rewards)[::2] - np.asarray(rewards)[1::2]
    return np.einsum("i,ijk->jk", difference, directions) / (2 * sigma * len(directions))
