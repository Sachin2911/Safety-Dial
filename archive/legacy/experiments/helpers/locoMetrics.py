"""Gate 0 statistics, matched-pair construction, and the trivial baselines.

Two ideas here are load-bearing for the project, and both replace something weaker.

**Disjointness lift replaces "does cost ever fire".** The claim is that a filter seeded from the
velocity-cost channel carries no information about falling. The test for that is not whether cost
is ever nonzero, it is whether cost is *predictive of an imminent fall*. Lift is the ratio of
P(cost in the window before a fall) to P(cost in a random window of the same length). Lift near
1.0 means the channels are independent, which is what the transfer experiment needs. Lift of 3
means the supervised baseline gets Mode B nearly for free and the headline comparison is void.

**Matched pairs replace population AUROC.** Torso height and pitch decode from a raw 32x32
grayscale frame at R^2 = 0.99, and `terminated` is a threshold on exactly those two numbers. So
AUROC against the whole population is close to free: a two-line baseline scores ~0.98 and the
result says nothing. A matched pair holds pose almost fixed and varies only recoverability, so
whatever separates the pair has to live in the velocities and the dynamics, which is precisely
what a world-model rollout is for. Matched-pair AUROC, reported as a margin over the strongest
trivial baseline, is the honest number.
"""

from __future__ import annotations

import numpy as np

# Walker2d and Hopper share an observation layout: concat(qpos[1:], clip(qvel, -10, 10)).
# So obs[0] is torso height and obs[1] is pitch. qpos carries the unclipped truth.
QPOS_Z, QPOS_PITCH = 1, 2


def episode_metrics_loco(cols: dict[str, np.ndarray]) -> dict:
    """Per-episode summary. `cols` is one episode's worth of the collection schema.

    "Fell" is read from the `healthy` column where present, not from `terminated`. Under
    termination-disabled collection `terminated` is all-zero by construction, and even under the
    benchmark's own termination it is offset one row from the actual failure. See
    `locoCollect.rollout_episode`.
    """
    n = len(cols["reward"])
    healthy = cols.get("healthy")
    if healthy is not None and n:
        fell_flag = bool((healthy == 0).any())
    else:
        fell_flag = bool(cols["terminated"][-1]) if n else False
    terminated = bool(cols["terminated"][-1]) if n else False
    truncated = bool(cols["truncated"][-1]) if n else False
    return {
        "length": n,
        "episode_return": float(cols["reward"].sum()),
        # A fall is a health violation, never truncation. The time limit is not a safety event.
        "fell": fell_flag,
        "terminated_flag": terminated,
        "truncated": truncated,
        "cost_sum": int(cols["cost"].sum()),
        "cost_rate": float(cols["cost"].mean()) if n else 0.0,
        # Mode A confined to the pre-failure phase. Post-failure rows exist only because
        # collection deliberately continues past the benchmark's cutoff, and a sliding fallen
        # robot trips the velocity threshold far more often than an upright one, so pooling
        # them makes the Mode A rate mostly a measurement of Mode B.
        "cost_rate_prefail": (
            float(cols["cost"][: int(np.argmax(healthy == 0))].mean())
            if (healthy is not None and n and (healthy == 0).any()
                and int(np.argmax(healthy == 0)) > 0)
            else (float(cols["cost"].mean()) if n else 0.0)
        ),
        "n_prefail": (
            int(np.argmax(healthy == 0)) if (healthy is not None and n and (healthy == 0).any())
            else n
        ),
        "mean_x_velocity": float(cols["x_velocity"].mean()) if n else 0.0,
        "max_x_velocity": float(cols["x_velocity"].max()) if n else 0.0,
        "distance": float(cols["qpos"][-1, 0] - cols["qpos"][0, 0]) if n else 0.0,
        "min_z": float(cols["qpos"][:, QPOS_Z].min()) if n else 0.0,
        "max_abs_pitch": float(np.abs(cols["qpos"][:, QPOS_PITCH]).max()) if n else 0.0,
    }


def disjointness_lift(
    cost: np.ndarray,
    terminated: np.ndarray,
    ep_offset: np.ndarray,
    ep_len: np.ndarray,
    window: int = 125,
    rng: np.random.Generator | None = None,
    n_control: int = 2000,
) -> dict:
    """P(cost in the `window` steps before a failure) / P(cost in a matched control window).

    Both halves must be drawn from the SAME phase of the episode, and this is easy to get
    catastrophically wrong. Under termination-disabled collection each episode carries roughly
    200 synthetic post-failure rows, and a fallen Walker2d slides: measured cost rate is 0.00068
    on pre-failure rows against 0.01950 on post-failure rows, 29x higher. An earlier version
    anchored the numerator at the first failure row but still drew control windows uniformly
    over the full episode, so about 66% of control ground was post-failure and the denominator
    was really measuring cost-caused-by-the-fall. That reported lift 0.22 (a clean PASS, and
    below 1.0 it even reads as cost being anti-correlated with failure) where the correct value
    on the same pool is two orders of magnitude higher. The gate was certified backwards.

    So: controls are confined to each episode's pre-failure phase, and their lengths are drawn
    from the numerator's own realised window lengths, since a short episode gives the numerator
    less than `window` steps of exposure and matching that matters at these rates.
    """
    rng = rng or np.random.default_rng(0)
    pre_hits, control_hits = [], []
    pre_lengths = []

    # Per episode: the first failure row, and the length of the usable pre-failure phase.
    phases = []
    for off, ln in zip(ep_offset, ep_len):
        off, ln = int(off), int(ln)
        if not ln:
            continue
        seg = terminated[off : off + ln]
        first = int(np.argmax(seg)) if seg.any() else ln  # ln means "never failed"
        phases.append((off, ln, first))
        if seg.any():
            start = max(off, off + first + 1 - window)
            end = off + first + 1
            pre_hits.append(bool(cost[start:end].any()))
            pre_lengths.append(end - start)

    if not pre_lengths:
        return {"window": window, "n_falls": 0, "n_control": 0, "p_cost_before_fall": 0.0,
                "p_cost_random_window": 0.0, "lift": float("nan")}

    # Control windows: pre-failure ground only, length-matched to the numerator.
    usable = [(off, first) for off, ln, first in phases if first >= 2]
    for _ in range(n_control):
        if not usable:
            break
        off, first = usable[int(rng.integers(len(usable)))]
        w = int(pre_lengths[int(rng.integers(len(pre_lengths)))])
        w = min(w, first)
        if w < 1:
            continue
        s0 = off + int(rng.integers(0, max(1, first - w + 1)))
        control_hits.append(bool(cost[s0 : s0 + w].any()))

    p_pre = float(np.mean(pre_hits)) if pre_hits else 0.0
    p_ctl = float(np.mean(control_hits)) if control_hits else 0.0
    if p_ctl > 0:
        lift = p_pre / p_ctl
    elif p_pre > 0:
        lift = float("inf")
    else:
        lift = float("nan")  # no cost anywhere: undefined, not a pass
    return {
        "window": window,
        "n_falls": len(pre_hits),
        "n_control": len(control_hits),
        "mean_pre_window_len": float(np.mean(pre_lengths)),
        "p_cost_before_fall": p_pre,
        "p_cost_random_window": p_ctl,
        "lift": float(lift),
    }


def build_matched_pairs(
    qpos: np.ndarray,
    labels: np.ndarray,
    idx: np.ndarray,
    z_tol: float = 0.02,
    pitch_tol: float = 0.02,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair each irrecoverable state with a recoverable one at near-identical pose.

    `labels` is True for recoverable. `idx` indexes into `qpos` for the labelled subset.
    Returns `(irrecoverable_idx, recoverable_idx)`, aligned, each recoverable state used once.

    If very few pairs exist, that is itself the Phase 0 verdict: it means irrecoverability really
    is a pose threshold in this environment, and no latent method can add anything.
    """
    rng = rng or np.random.default_rng(0)
    pos = idx[~labels]  # irrecoverable
    neg = idx[labels]  # recoverable
    if len(pos) == 0 or len(neg) == 0:
        return np.zeros(0, int), np.zeros(0, int)

    neg_z = qpos[neg, QPOS_Z]
    neg_p = qpos[neg, QPOS_PITCH]
    used = np.zeros(len(neg), dtype=bool)

    a_out, b_out = [], []
    order = rng.permutation(len(pos))
    for i in pos[order]:
        dz = np.abs(neg_z - qpos[i, QPOS_Z])
        dp = np.abs(neg_p - qpos[i, QPOS_PITCH])
        cand = np.where((dz < z_tol) & (dp < pitch_tol) & ~used)[0]
        if not len(cand):
            continue
        # Nearest in pose among the admissible ones, so the pair is as tight as possible.
        j = cand[np.argmin(dz[cand] + dp[cand])]
        used[j] = True
        a_out.append(int(i))
        b_out.append(int(neg[j]))
    return np.asarray(a_out, int), np.asarray(b_out, int)


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Rank-based AUROC with tie handling. `positive` is the class scored higher when correct."""
    scores = np.asarray(scores, float)
    positive = np.asarray(positive, bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within ties, so a constant score gives exactly 0.5 rather than 0 or 1.
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = np.arange(i + 1, j + 2).mean()
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def paired_auroc(score_irrec: np.ndarray, score_rec: np.ndarray) -> float:
    """Fraction of matched pairs ranked correctly, ties counting a half.

    This is AUROC restricted to the matched pairing, which is the right statistic because the
    pairs are not independent draws: each one deliberately holds pose fixed.
    """
    d = np.asarray(score_irrec, float) - np.asarray(score_rec, float)
    if not len(d):
        return float("nan")
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def trivial_baselines(qpos: np.ndarray, qvel: np.ndarray, obs: np.ndarray | None = None) -> dict:
    """Scores that any competent detector must beat, higher meaning "more irrecoverable".

    These exist to make the result falsifiable. If the latent score cannot beat `neg_z` on
    matched pairs, the project has measured the torso height and nothing else.
    """
    out = {
        "neg_z": -qpos[:, QPOS_Z],
        "abs_pitch": np.abs(qpos[:, QPOS_PITCH]),
        "pose_2d": -qpos[:, QPOS_Z] + np.abs(qpos[:, QPOS_PITCH]),
        "qvel_norm": np.abs(qvel).max(axis=1),
    }
    if obs is not None:
        out["obs_norm"] = np.linalg.norm(obs, axis=1)
    return out


def gate0_table(episodes: list[dict], cost_rate_bounds=(0.01, 0.08)) -> dict:
    """Aggregate the Gate 0 statistics and evaluate each pass condition.

    Gate 0 gates the DATA REGIME, not the environment. A converged expert policy does not fall,
    on this benchmark or any other, and nobody deploys a runtime safety filter on a policy that
    never fails. So the question is never "does this environment produce failures", it is "does
    the collected mixture place enough mass on the competent-but-fallible regime".
    """
    n = len(episodes)
    fell = np.array([e["fell"] for e in episodes])
    steps = np.array([e["length"] for e in episodes])
    costs = np.array([e["cost_sum"] for e in episodes])
    pre_n = np.array([e.get("n_prefail", e["length"]) for e in episodes])
    pre_cost = np.array([e.get("cost_rate_prefail", 0.0) * e.get("n_prefail", e["length"])
                         for e in episodes])
    rets = np.array([e["episode_return"] for e in episodes])
    total_steps = int(steps.sum())

    stats = {
        "n_episodes": n,
        "n_steps": total_steps,
        "fall_rate": float(fell.mean()) if n else 0.0,
        "n_falls": int(fell.sum()),
        "cost_rate": float(costs.sum() / total_steps) if total_steps else 0.0,
        "cost_rate_prefail": float(pre_cost.sum() / pre_n.sum()) if pre_n.sum() else 0.0,
        "n_cost_steps": int(costs.sum()),
        "n_cost_steps_prefail": int(pre_cost.sum()),
        "mean_return": float(rets.mean()) if n else 0.0,
        "mean_length": float(steps.mean()) if n else 0.0,
    }
    stats["checks"] = {
        "fall_rate >= 0.25": stats["fall_rate"] >= 0.25,
        "n_falls >= 2000": stats["n_falls"] >= 2000,
        f"cost_rate in {cost_rate_bounds}": (
            cost_rate_bounds[0] <= stats["cost_rate"] <= cost_rate_bounds[1]
        ),
        "n_cost_steps >= 10000": stats["n_cost_steps"] >= 10000,
    }
    return stats


def format_table(rows: list[dict], cols: list[str] | None = None) -> str:
    """Small fixed-width table formatter, so reports read the same in a terminal and a notebook."""
    if not rows:
        return "(empty)"
    cols = cols or list(rows[0].keys())
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in cols}
    out = [" | ".join(c.ljust(widths[c]) for c in cols)]
    out.append("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        out.append(" | ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))
    return "\n".join(out)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)
