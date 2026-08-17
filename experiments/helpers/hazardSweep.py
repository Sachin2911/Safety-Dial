"""
Additions for the hazard-avoidance sweep.

Import alongside linProbeHelpers:

    import helpers.hazardSweep as hs
    importlib.reload(hs)
    from helpers.hazardSweep import *

What this fixes, in order of importance:

  1. TASK METRIC. `success` is binary and currently False everywhere, so the
     task axis of your Pareto front is constant. episode_metrics() records
     max coverage, final block error and out-of-bounds steps instead.

  2. BOX PLACEMENT. box_from_baseline() puts the hazard on the path the
     lambda=0 planner actually takes, and calibrate_box() checks the baseline
     violation fraction lands in a usable range. Guessing coordinates is how
     you ended up with a box the baseline never enters.

  3. MARGIN. margin_from_imagination() derives the cushion from the measured
     p95 radial probe error at rollout depth, not 2x the per-coordinate RMSE.

  4. COST. Model loading moved out of the episode loop (was reloading LeWM
     every run), and torch/numpy seeded per episode for reproducibility.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. instrumentation
# ---------------------------------------------------------------------------

def make_on_step_v2(frames, states, rewards, infos_keys=None):
    """Like make_on_step, but also records per-step reward.

    Push-T's reward is coverage-based, which gives you the continuous task
    axis that binary `success` cannot. If your swm version exposes coverage
    under a different key, run inspect_world_info() once to find it.
    """
    def on_step(w):
        frame = np.asarray(w.infos["pixels"][0, -1]).copy()
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        frames.append(frame)
        states.append(w.infos["state"][0, 0].copy())

        r = getattr(w, "rewards", None)
        if r is not None:
            rewards.append(float(np.asarray(r).ravel()[0]))
        elif "coverage" in w.infos:
            rewards.append(float(np.asarray(w.infos["coverage"]).ravel()[0]))
        else:
            rewards.append(np.nan)

        if infos_keys is not None and not infos_keys:
            infos_keys.extend(list(w.infos.keys()))

    return on_step


def inspect_world_info(world):
    """Run once. Tells you which fields carry coverage / reward / termination.

    You need this to answer why success is False when the block lands 6px from
    the goal -- either the criterion includes the pusher position, or the
    out-of-bounds state is breaking termination.
    """
    print("infos keys :", list(world.infos.keys()))
    for attr in ("rewards", "terminateds", "truncateds"):
        v = getattr(world, attr, None)
        print(f"{attr:12s}:", None if v is None else np.asarray(v).ravel()[:4])


def out_of_bounds_stats(states, lo=0.0, hi=512.0):
    """Your lambda=0 baseline ends at pusher (413, -559). That is outside the
    arena and is what triggers the observation-space warning. Track it: any
    episode with oob steps is not a valid data point for the front.
    """
    xy = states[:, 0:2]
    oob = (xy < lo).any(axis=1) | (xy > hi).any(axis=1)
    return {
        "oob_any": bool(oob.any()),
        "oob_steps": int(oob.sum()),
        "oob_frac": float(oob.mean()),
        "max_abs_pos": float(np.abs(xy).max()),
        "max_abs_vel": float(np.abs(states[:, 5:7]).max()),
    }


def episode_metrics(states, rewards, goal_state, hazard_box, terminated):
    """Everything one episode contributes to the front."""
    from helpers.linProbeHelpers import real_violation_stats

    rewards = np.asarray(rewards, dtype=float)
    block_err = float(np.linalg.norm(states[-1, 2:4] - goal_state[2:4]))
    theta_err = float(abs(np.arctan2(
        np.sin(states[-1, 4] - goal_state[4]),
        np.cos(states[-1, 4] - goal_state[4]),
    )))

    m = {
        "success": bool(terminated),
        "max_coverage": float(np.nanmax(rewards)) if rewards.size else np.nan,
        "final_coverage": float(rewards[-1]) if rewards.size else np.nan,
        "final_block_err_px": block_err,
        "final_theta_err_rad": theta_err,
    }
    m.update(real_violation_stats(states, hazard_box, entity="pusher"))
    m.update(out_of_bounds_stats(states))
    return m


# ---------------------------------------------------------------------------
# 2. hazard geometry, derived rather than guessed
# ---------------------------------------------------------------------------

def box_from_baseline(states, half_width=30.0, half_height=30.0, frac=0.5,
                      entity="pusher"):
    """Centre a box on the baseline path at a given fraction along the episode.

    frac=0.5 puts it mid-episode, avoiding the start (trivially avoidable) and
    the endgame (where blocking it just breaks the task).
    """
    col = slice(0, 2) if entity == "pusher" else slice(2, 4)
    xy = states[:, col]
    c = xy[int(np.clip(frac, 0.0, 0.999) * len(xy))]
    return (float(c[0] - half_width), float(c[0] + half_width),
            float(c[1] - half_height), float(c[1] + half_height))


def calibrate_box(states, target=(0.15, 0.50), half_sizes=(20, 30, 40, 50),
                  fracs=(0.35, 0.5, 0.65), entity="pusher", verbose=True):
    """Search box placements for one the baseline actually enters.

    Returns candidates whose baseline violation fraction lands in `target`.
    Too low and there is no hazard to avoid (your current situation, 0.00 at
    every lambda); too high and no safe route exists and every lambda fails.
    """
    from helpers.linProbeHelpers import real_violation_stats

    out = []
    for f in fracs:
        for h in half_sizes:
            box = box_from_baseline(states, h, h, f, entity)
            v = real_violation_stats(states, box, entity)["frac_violating"]
            out.append({"frac": f, "half": h, "box": box, "baseline_viol": v,
                        "usable": target[0] <= v <= target[1]})
    out.sort(key=lambda d: -d["baseline_viol"])

    if verbose:
        print(f"{'frac':>5} {'half':>5} {'baseline viol':>14}  box")
        for d in out:
            flag = "  <-- usable" if d["usable"] else ""
            print(f"{d['frac']:>5.2f} {d['half']:>5.0f} {d['baseline_viol']:>14.2f}  "
                  f"{tuple(round(b) for b in d['box'])}{flag}")
    return out


def margin_from_imagination(true_xy, imag_xy, t=5, q=95.0, verbose=True):
    """Margin from the p95 RADIAL probe error at rollout depth t.

    Your PROBE_MARGIN = 2 * 13.23 used the per-coordinate RMSE. Radial error is
    ~sqrt(2) larger, and the tail is what breaches a margin, so a quantile is
    the defensible choice. Pass the true_xy / imag_xy arrays from the
    imagination-vs-reality cell.
    """
    err = np.linalg.norm(true_xy[:, t] - imag_xy[:, t], axis=1)
    stats = {
        "rms": float(np.sqrt((err ** 2).mean())),
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
        "p99": float(np.percentile(err, 99)),
        "max": float(err.max()),
    }
    if verbose:
        print(f"radial probe error at t={t}: " +
              "  ".join(f"{k}={v:.1f}" for k, v in stats.items()))
    return float(np.percentile(err, q)), stats


def detour_feasible(start_xy, goal_xy, hazard_box, margin, px_per_step, n_steps):
    """Is a safe route reachable inside the step budget?

    If this returns False, every plan violates and the lambda sweep measures
    nothing. Run it before every sweep.
    """
    x0, x1, y0, y1 = hazard_box
    x0, y0, x1, y1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin
    budget = px_per_step * n_steps
    s, g = np.asarray(start_xy, float), np.asarray(goal_xy, float)
    best = min(np.linalg.norm(np.array(c) - s) + np.linalg.norm(g - np.array(c))
               for c in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)])
    return {"direct": float(np.linalg.norm(g - s)), "best_detour": float(best),
            "budget": float(budget), "feasible": bool(best < budget),
            "slack_px": float(budget - best)}


def estimate_px_per_step(states):
    """Median pusher displacement per env step, for detour_feasible."""
    d = np.linalg.norm(np.diff(states[:, 0:2], axis=0), axis=1)
    return float(np.median(d))


# ---------------------------------------------------------------------------
# 3. episode runner and sweep
# ---------------------------------------------------------------------------

def run_episode(
    base_model,          # load ONCE outside; was reloaded every episode
    probe,
    hazard_box,
    lam,
    *,
    swm,
    process,
    transform,
    start_state,
    goal_state,
    margin,
    n_substeps=5,
    seed=0,
    num_samples=300,
    n_steps=30,
    topk=30,
    var_scale=1.0,
    horizon=5,
    receding_horizon=1,
    action_block=5,
    eval_budget=50,
    keep_frames=False,
):
    from stable_worldmodel.solver import CEMSolver
    from helpers.linProbeHelpers import (
        HazardAugmentedCostModel, make_on_step, summarize_cem_history,
    )

    # CEMSolver(seed=...) does not control every RNG -- cell 38 and cell 39
    # disagreed at the same nominal seed. Seed globally too.
    torch.manual_seed(seed)
    np.random.seed(seed)

    cost_model = HazardAugmentedCostModel(
        base_model, probe, hazard_box,
        lam=lam, margin=margin, n_substeps=n_substeps,
    ).to("cuda").eval()

    solver = CEMSolver(
        model=cost_model, batch_size=1, num_samples=num_samples,
        var_scale=var_scale, n_steps=n_steps, topk=topk,
        device="cuda", seed=seed,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(horizon=horizon,
                              receding_horizon=receding_horizon,
                              action_block=action_block),
        process=process, transform=transform,
    )
    world = swm.World("swm/PushT-v1", num_envs=1, image_shape=(224, 224),
                      max_episode_steps=2 * eval_budget)
    world.set_policy(policy)
    world.reset(seed=seed, options={"state": start_state, "goal_state": goal_state})

    frames, states, rewards = [], [], []
    world._run(max_steps=eval_budget, mode="wait",
               on_step=make_on_step_v2(frames, states, rewards))

    states = np.stack(states)
    out = {"lam": lam, "seed": seed}
    out.update(episode_metrics(states, rewards, goal_state, hazard_box,
                               bool(world.terminateds[0])))
    out["solve_diagnostics"] = summarize_cem_history(cost_model, n_steps=n_steps,
                                                     verbose=False)
    out["states"] = states
    if keep_frames:
        out["frames"] = frames
    return out


def sweep_lambda(lams, seeds=(0, 1, 2, 3, 4), verbose=True, **kw):
    """Stage A: the front. 7 lambdas x 5 seeds ~= 35 episodes ~= 6 minutes."""
    rows = []
    for lam in lams:
        for s in seeds:
            r = run_episode(lam=lam, seed=s, **kw)
            rows.append(r)
        if verbose:
            g = [r for r in rows if r["lam"] == lam]
            print(f"lam={lam:<8g} "
                  f"viol={np.mean([r['frac_violating'] for r in g]):.2f}"
                  f"+-{np.std([r['frac_violating'] for r in g]):.2f}  "
                  f"cov={np.nanmean([r['max_coverage'] for r in g]):.3f}  "
                  f"blockerr={np.mean([r['final_block_err_px'] for r in g]):.1f}px  "
                  f"succ={np.mean([r['success'] for r in g]):.1f}  "
                  f"oob={np.mean([r['oob_any'] for r in g]):.1f}")
    return rows


def plot_front(rows, x="frac_violating", y="max_coverage"):
    """Mean +- std per lambda. Error bars are the point of running seeds."""
    lams = sorted({r["lam"] for r in rows})
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for lam in lams:
        g = [r for r in rows if r["lam"] == lam]
        xs, ys = [r[x] for r in g], [r[y] for r in g]
        ax.errorbar(np.mean(xs), np.nanmean(ys),
                    xerr=np.std(xs), yerr=np.nanstd(ys),
                    fmt="o", capsize=3)
        ax.annotate(f"λ={lam:g}", (np.mean(xs), np.nanmean(ys)),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("fraction of steps in hazard")
    ax.set_ylabel("max coverage")
    ax.set_title("Penalty-CEM front (mean ± std over seeds)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
