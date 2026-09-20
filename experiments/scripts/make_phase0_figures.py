"""Generate every figure in the Phase 0 report, and save the data behind each one.

Each figure writes a PDF into `docs/phase0/latex/figures/` and its underlying numbers into
`docs/phase0/data/` as JSON, so a plot can be re-made or checked without re-running the
simulation that produced it, and so the numbers are committed alongside the report.

    uv run python experiments/scripts/make_phase0_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments"))
FIG = REPO / "docs" / "phase0" / "latex" / "figures"
# Figure data lives beside the report, not under data/, because data/ is gitignored
# and the numbers behind a published figure have to be committed with it.
DAT = REPO / "docs" / "phase0" / "data"
FIG.mkdir(parents=True, exist_ok=True)
DAT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "legend.frameon": False,
})
C_W, C_H, C_ACC, C_MUTE = "#1f4e79", "#c1666b", "#d08c34", "#8c8c8c"


def save(name, payload):
    (DAT / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float))


def fig_timeline():
    """One episode either side of the failure flag: the robot is still standing when it fires."""
    import helpers.locoCollect as lc
    import helpers.locoEnv as le
    import helpers.locoPolicies as lp

    env = le.make_loco_env("Walker2d", "v1", terminate_when_unhealthy=False, time_limit=False)
    pol = lp.MixedPolicy(
        [lp.RandomActionPolicy(env.action_space, seed=1),
         lp.ScriptedForwardPolicy(env.action_space, seed=2, smooth=0.8, noise=0.3)], seed=3)
    c = lc.rollout_episode(env, pol, seed=7, stop_after_unhealthy=200)
    h = c["healthy"]
    fu = int(np.argmax(h == 0))
    t = np.arange(len(h)) - fu
    z, pitch = c["qpos"][:, 1], c["qpos"][:, 2]
    m = (t >= -30) & (t <= 180)

    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.plot(t[m], z[m], color=C_W, lw=1.6, label="torso height $z$")
    ax.axhline(0.8, color=C_MUTE, ls=":", lw=1, label="healthy floor $z=0.8$")
    ax.axvline(0, color=C_ACC, lw=1.4)
    ax.annotate("benchmark declares failure here\n(still standing, $z$=%.2f)" % z[fu],
                xy=(0, z[fu]), xytext=(40, 1.34), fontsize=8, color=C_ACC,
                arrowprops=dict(arrowstyle="->", color=C_ACC, lw=1))
    on_ground = int(np.argmax((z[fu:] < 0.4)) ) if (z[fu:] < 0.4).any() else None
    if on_ground:
        ax.axvline(on_ground, color=C_H, lw=1.2, ls="--")
        ax.axvspan(0, on_ground, color=C_ACC, alpha=0.08, lw=0)
        ax.annotate(f"actually on the ground, +{on_ground} steps", xy=(on_ground, 0.28),
                    xytext=(on_ground + 14, 0.55), fontsize=8, color=C_H,
                    arrowprops=dict(arrowstyle="->", color=C_H, lw=1))
    ax.set_xlabel("steps from the benchmark's failure flag")
    ax.set_ylabel("torso height (m)")
    ax.set_ylim(0, 1.75)
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(FIG / "fig_timeline.pdf")
    plt.close(fig)
    save("timeline", {"t": t[m].tolist(), "z": z[m].tolist(), "pitch": pitch[m].tolist(),
                      "first_unhealthy_idx": fu, "steps_to_ground": on_ground})
    print(f"  fig_timeline: flag at z={z[fu]:.3f}, ground at +{on_ground}")
    return c, fu


def fig_recovery(cols, fu):
    """Replay a CEM recovery from a state lying on the ground."""
    import helpers.oracle as orc

    n = len(cols["healthy"])
    i = min(fu + 150, n - 1)
    qpos, qvel = cols["qpos"][i], cols["qvel"][i]
    cfg = orc.OracleConfig(robot="Walker2d", version="v1", use_cem=True)
    sim = orc.RecoverySimulator(cfg)
    rng = np.random.default_rng(orc.state_seed(0, i))
    a_dim, H, k = sim.action_dim, cfg.horizon, cfg.cem_knots
    pop = max(cfg.cem_elites * 2, cfg.cem_pop)
    winner = None
    for _ in range(cfg.cem_restarts):
        mean = np.zeros((k, a_dim))
        sigma = np.full((k, a_dim), cfg.cem_sigma0)
        for _ in range(cfg.cem_iters):
            samples = np.clip(rng.normal(mean, sigma, size=(pop, k, a_dim)), -1, 1)
            scores = np.empty(pop)
            for j, sp in enumerate(samples):
                seq = orc._knots_to_sequence(sp, H)
                ok, sc = sim.rollout(qpos, qvel, seq)
                if ok:
                    winner = seq
                    break
                scores[j] = sc
            if winner is not None:
                break
            el = samples[np.argsort(-scores)[: cfg.cem_elites]]
            mean = el.mean(axis=0)
            sigma = np.maximum(el.std(axis=0), 1e-3) * cfg.cem_sigma_decay
        if winner is not None:
            break
    if winner is None:
        print("  fig_recovery: no recovery found, skipping")
        return
    u = sim.env.unwrapped
    sim._reset_to(u, qpos, qvel)
    zs, ps = [], []
    for a in winner:
        u.do_simulation(a, u.frame_skip)
        zs.append(float(u.data.qpos[1]))
        ps.append(float(u.data.qpos[2]))
    zs, ps = np.array(zs), np.array(ps)

    # Locate the first 25-step window where the full predicate held, and shade it.
    pred = sim.pred
    ok_mask = np.array([
        pred.margin(np.array([0.0, zz, pp] + [0.0] * 6), np.zeros(9)) > 0
        for zz, pp in zip(zs, ps)
    ])
    first_ok = None
    run = 0
    for t, okk in enumerate(ok_mask):
        run = run + 1 if okk else 0
        if run >= pred.dwell:
            first_ok = t - pred.dwell + 1
            break

    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    ax.plot(zs, color=C_W, lw=1.6, label="torso height $z$")
    ax.axhline(1.0, color=C_MUTE, ls=":", lw=1, label="recovery target $z\\geq 1.0$")
    if first_ok is not None:
        ax.axvspan(first_ok, first_ok + pred.dwell, color="#2e7d32", alpha=0.16, lw=0,
                   label="predicate held 25 steps")
    ax2 = ax.twinx()
    ax2.plot(np.abs(ps), color=C_H, lw=1.2, alpha=0.75, label="$|$pitch$|$")
    ax2.set_ylabel("|pitch| (rad)", color=C_H)
    ax2.tick_params(axis="y", colors=C_H)
    ax2.grid(False)
    ax.set_xlabel("simulator step of the recovery plan")
    ax.set_ylabel("torso height (m)")
    ax.set_title(f"Walker2d standing up from $z={qpos[1]:.2f}$, pitch ${qpos[2]:.2f}$ rad",
                 fontsize=9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    fig.savefig(FIG / "fig_recovery.pdf")
    plt.close(fig)
    save("recovery", {"z": zs.tolist(), "pitch": ps.tolist(),
                      "start_z": float(qpos[1]), "start_pitch": float(qpos[2]),
                      "max_action": float(np.abs(winner).max()),
                      "predicate_satisfied_at": first_ok})
    print(f"  fig_recovery: z {qpos[1]:.3f} -> {zs.max():.3f}, max|a|={np.abs(winner).max():.3f}")


def fig_horizon():
    """Irrecoverable fraction by robot and search horizon, measured."""
    data = {"Walker2d": {250: 0.000, 500: 0.000}, "Hopper": {250: 0.333, 500: 0.167}}
    n = 30
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    x = np.arange(2)
    w = 0.36
    for i, (robot, col) in enumerate([("Walker2d", C_W), ("Hopper", C_H)]):
        vals = [data[robot][250], data[robot][500]]
        err = [1.96 * np.sqrt(max(v, 1e-9) * (1 - v) / n) for v in vals]
        b = ax.bar(x + (i - 0.5) * w, vals, w, color=col, label=robot,
                   yerr=err, capsize=3, error_kw=dict(lw=0.8, ecolor=C_MUTE))
        for rect, v in zip(b, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.3f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["2 s (H=250)", "4 s (H=500)"])
    ax.set_ylabel("fraction irrecoverable")
    ax.set_title("Settled-fallen states, torso on the ground", fontsize=9)
    ax.set_ylim(0, 0.45)
    ax.legend(fontsize=8)
    fig.savefig(FIG / "fig_horizon.pdf")
    plt.close(fig)
    save("horizon", {"n_per_cell": n, "data": data})
    print("  fig_horizon: written")


def fig_cost():
    """Cost rate before and after the failure, the bug that inverted Gate 0."""
    vals = [0.00049, 0.01950]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    b = ax.bar(["pre-failure", "post-failure"], vals, 0.5, color=[C_W, C_H])
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.0006, f"{v:.5f}",
                ha="center", fontsize=8)
    ax.set_ylabel("fraction of steps with cost = 1")
    ax.set_title("A fallen Walker2d slides, so it trips the\nvelocity limit 29x more often",
                 fontsize=9)
    ax.set_ylim(0, 0.024)
    fig.savefig(FIG / "fig_cost.pdf")
    plt.close(fig)
    save("cost_phase", {"pre_failure_cost_rate": vals[0], "post_failure_cost_rate": vals[1],
                        "ratio": vals[1] / vals[0]})
    print("  fig_cost: written")


def fig_offset():
    """Recoverability against offset from the failure flag."""
    src = REPO / "data" / "eval" / "phase0_offset_sweep.json"
    if not src.is_file():
        src = DAT / "offset_sweep.json"
    if not src.is_file():
        print("  fig_offset: sweep data not present yet, skipping")
        return False
    d = json.loads(src.read_text())
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    for robot, col in [("Walker2d", C_W), ("Hopper", C_H)]:
        rows = d.get(robot) or []
        if not rows:
            continue
        o = [r["offset"] for r in rows]
        y = [r["recoverable"] for r in rows]
        lo = [r["recoverable"] - r["lo"] for r in rows]
        hi = [r["hi"] - r["recoverable"] for r in rows]
        ax.errorbar(o, y, yerr=[lo, hi], color=col, marker="o", ms=3.5, lw=1.5,
                    capsize=2, elinewidth=0.8, label=robot)
    ax.axvline(0, color=C_ACC, lw=1.3)
    ax.text(2, 0.06, "benchmark\ndeclares failure", fontsize=7.5, color=C_ACC)
    ax.set_xlabel("steps from the benchmark's failure flag  (negative = before)")
    ax.set_ylabel("fraction recoverable")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    fig.savefig(FIG / "fig_offset.pdf")
    plt.close(fig)
    save("offset_sweep", d)
    print("  fig_offset: written")
    return True


def fig_storage():
    """Why pixels are not stored."""
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    vals = [0.84, 489.0]
    b = ax.barh(["state\n(260 B/step)", "pixels\n(150 kB/frame)"], vals,
                0.5, color=[C_W, C_H])
    ax.axvline(60, color=C_ACC, ls="--", lw=1.2)
    ax.text(66, 0.42, "free disk\n60 GB", fontsize=7.5, color=C_ACC)
    for rect, v in zip(b, vals):
        ax.text(v * 1.06 if v < 100 else v * 0.5, rect.get_y() + rect.get_height() / 2,
                f"{v:g} GB", va="center", fontsize=8,
                color="white" if v > 100 else "black")
    ax.set_xscale("log")
    ax.set_xlabel("storage for 3.25M steps (GB, log scale)")
    fig.savefig(FIG / "fig_storage.pdf")
    plt.close(fig)
    save("storage", {"state_gb": 0.84, "pixels_gb": 489.0, "free_gb": 60,
                     "bytes_per_step": 260, "bytes_per_frame": 150528})
    print("  fig_storage: written")


if __name__ == "__main__":
    print("generating Phase 0 figures")
    fig_storage()
    fig_cost()
    fig_horizon()
    cols, fu = fig_timeline()
    fig_recovery(cols, fu)
    fig_offset()
    print(f"\nfigures -> {FIG}")
    print(f"data    -> {DAT}")
