"""Phase 0 triage: can this environment support the project's claim?

The cheapest experiment that can kill the project, run before any world-model code exists. It
answers three questions at once.

**1. Where does recoverability actually collapse, relative to where the benchmark says it does?**
This is the paper's Figure 1. Safety-Gymnasium declares failure when Walker2d's torso pitch
passes 1.0 rad, at which point the robot is typically still standing at z ~ 1.09; it does not
reach the ground for roughly another 60 steps. Measuring recoverability across that window is
what turns "termination is not irreversibility" from an assertion into a number.

**2. Is the problem non-trivial?** Torso height and pitch decode from a raw 32x32 grayscale frame
at R^2 = 0.99, and the benchmark's failure flag is a threshold on exactly those two numbers. So
there is a live risk that "predict irreversible failure" is solved by a two-line baseline, making
the result true but vacuous. Matched pairs (near-identical pose, differing recoverability) are the
honest test, and the trivial baselines are scored on them here.

**3. Is there a window for a filter to act in?** The lead time from first-irrecoverable to
failure bounds what any runtime filter can do, however good its detector.

Collection runs with termination DISABLED and continues past the benchmark's cutoff, because
under the benchmark's own termination the episode stops one step BEFORE the first unhealthy
state, so every recorded row is healthy and the dataset contains no failures at all. See
`locoCollect.rollout_episode`.

Usage:
    uv run python experiments/scripts/run_triage.py                  # full, with CEM
    uv run python experiments/scripts/run_triage.py --no-cem         # cheap, biased proxy
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import helpers.locoCollect as lc  # noqa: E402
import helpers.locoEnv as le  # noqa: E402
import helpers.locoMetrics as lm  # noqa: E402
import helpers.locoPolicies as lp  # noqa: E402
import helpers.oracle as orc  # noqa: E402

#: Offsets relative to the first unhealthy step. Negative is before the benchmark's cutoff.
OFFSET_BANDS = [-60, -30, -15, -8, -4, -2, 0, 5, 10, 20, 40, 80, 150]


def collect_pool(robot, version, n_episodes, seed, past):
    """A crude mixed policy is enough here. Trained policies are a Phase 1 concern."""
    env = le.make_loco_env(robot, version, terminate_when_unhealthy=False, time_limit=False)
    mix = lp.MixedPolicy(
        [
            lp.RandomActionPolicy(env.action_space, seed=seed),
            lp.ScriptedForwardPolicy(env.action_space, seed=seed + 1, smooth=0.8, noise=0.3),
            lp.ScriptedForwardPolicy(env.action_space, seed=seed + 2, smooth=0.5, noise=0.6),
        ],
        weights=[0.3, 0.4, 0.3],
        seed=seed + 3,
    )
    episodes, metrics = [], []
    for e in range(n_episodes):
        cols = lc.rollout_episode(env, mix, seed=seed + 1000 + e, stop_after_unhealthy=past)
        cols["episode_idx"][:] = e
        episodes.append(cols)
        metrics.append(lm.episode_metrics_loco(cols))
    return episodes, metrics


def flatten(episodes):
    keys = ("qpos", "qvel", "observation", "action", "cost", "healthy", "terminated",
            "step_idx", "episode_idx")
    cat = {k: np.concatenate([c[k] for c in episodes]) for k in keys}
    lens = np.array([len(c["reward"]) for c in episodes], dtype=int)
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
    # Signed steps-to-failure: positive before the first unhealthy step, 0 at it, negative after.
    # -10**6 marks an episode that never failed.
    stf = []
    for c in episodes:
        n = len(c["reward"])
        unhealthy = c["healthy"] == 0
        if unhealthy.any():
            first = int(np.argmax(unhealthy))
            stf.append(first - np.arange(n))
        else:
            stf.append(np.full(n, -(10**6)))
    return cat, lens, offs, np.concatenate(stf).astype(np.int64)


def stratified_sample(stf, n_total, rng):
    """Sample across offset bands either side of the failure moment.

    Uniform sampling would be dominated by mid-episode states far from any failure, and the
    whole question lives in the window around the boundary.
    """
    picks = []
    per = max(1, n_total // (len(OFFSET_BANDS) + 1))
    for off in OFFSET_BANDS:
        pool = np.where(stf == -off)[0] if off < 0 else np.where(stf == -off)[0]
        if len(pool):
            picks.append(rng.choice(pool, min(per, len(pool)), replace=False))
    far = np.where((stf > 100) | (stf == -(10**6)))[0]  # benign, far from any failure
    if len(far):
        picks.append(rng.choice(far, min(per, len(far)), replace=False))
    return np.unique(np.concatenate(picks)) if picks else np.zeros(0, int)


_WORKER = {}


def _init_worker(cfg):
    _WORKER["sim"] = orc.RecoverySimulator(cfg)
    _WORKER["cfg"] = cfg


def _label_one(args):
    qpos, qvel, seed, continuation = args
    r = orc.label_state(_WORKER["sim"], qpos, qvel, int(seed), _WORKER["cfg"],
                        continuation=continuation)
    return r["recoverable"], r["tier"], r["margin"], r["n_rollouts"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="Walker2d")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--n-label", type=int, default=1300)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--no-cem", action="store_true",
                    help="T0+T1+T2 only. Much cheaper, but a biased proxy: cheap search calls a "
                         "leaning-but-standing robot irrecoverable when CEM can recover it.")
    ap.add_argument("--past", type=int, default=200,
                    help="steps to continue past the benchmark's cutoff")
    ap.add_argument("--horizon", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "eval" / "triage"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Phase 0 triage: {args.robot} {args.version} ===\n", flush=True)

    t0 = time.time()
    episodes, metrics = collect_pool(args.robot, args.version, args.episodes, args.seed,
                                     args.past)
    cat, lens, offs, stf = flatten(episodes)
    n_failed = sum(1 for m in metrics if m["fell"])
    print(f"[1/5] {len(episodes)} episodes, {lens.sum()} steps, {time.time() - t0:.0f}s", flush=True)
    print(f"      {n_failed}/{len(episodes)} went unhealthy; "
          f"{int((cat['healthy'] == 0).sum())} post-failure rows recorded "
          f"(under benchmark termination these would not exist)", flush=True)

    gate = lm.gate0_table(metrics)
    print(f"      fall_rate {gate['fall_rate']:.3f}  cost_rate {gate['cost_rate']:.4f}  "
          f"mean_len {gate['mean_length']:.1f}", flush=True)

    failure_flag = (cat["healthy"] == 0).astype(np.uint8)
    lift = lm.disjointness_lift(cat["cost"], failure_flag, offs, lens, window=125, rng=rng)
    print(f"      disjointness lift (125 steps before failure): {lift['lift']:.3f}  "
          f"[P(cost|pre-failure)={lift['p_cost_before_fall']:.3f} vs "
          f"P(cost|random)={lift['p_cost_random_window']:.3f}]  need <= 2.0", flush=True)

    idx = stratified_sample(stf, args.n_label, rng)
    tiers_used = "T0+T1+T2" if args.no_cem else "T0+T1+T2+T3(CEM)"
    print(f"\n[2/5] labelling {len(idx)} states, {tiers_used}, H={args.horizon}, "
          f"{args.workers} workers", flush=True)

    cfg = orc.OracleConfig(robot=args.robot, version=args.version,
                           horizon=args.horizon, use_cem=not args.no_cem)
    seeds = [orc.state_seed(cat["episode_idx"][i], cat["step_idx"][i]) for i in idx]
    ep_end = {e: int(offs[e] + lens[e]) for e in range(len(lens))}
    payload = [
        (cat["qpos"][i], cat["qvel"][i], sd,
         cat["action"][i : min(i + args.horizon, ep_end[int(cat["episode_idx"][i])])])
        for i, sd in zip(idx, seeds)
    ]

    t1 = time.time()
    ctx = mp.get_context("fork")  # the oracle never renders, so no GL context is inherited
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        results = pool.map(_label_one, payload, chunksize=1)
    dt = time.time() - t1

    recoverable = np.array([r[0] for r in results], dtype=bool)
    tiers = [r[1] for r in results]
    print(f"      done in {dt / 60:.1f} min ({dt / max(len(idx), 1) * 1000:.0f} ms/state)",
          flush=True)
    print(f"      recoverable overall {recoverable.mean():.3f}  "
          f"tiers {dict(zip(*np.unique(tiers, return_counts=True)))}", flush=True)

    # --- Figure 1: recoverability against offset from the benchmark's failure flag ----------
    band = -stf[idx]  # offset relative to the first unhealthy step
    print("\n[3/5] recoverability vs offset from the benchmark's failure declaration")
    rows = []
    for off in OFFSET_BANDS:
        m = band == off
        if not m.sum():
            continue
        tc = dict(zip(*np.unique(np.array(tiers)[m], return_counts=True)))
        rows.append({"offset": off, "n": int(m.sum()),
                     "recoverable": float(recoverable[m].mean()),
                     "resolved_by": max(tc, key=tc.get) if tc else ""})
    far_m = band <= -1000
    if far_m.sum():
        rows.append({"offset": "far", "n": int(far_m.sum()),
                     "recoverable": float(recoverable[far_m].mean()), "resolved_by": "V1 benign"})
    print(lm.format_table(rows, ["offset", "n", "recoverable", "resolved_by"]))
    print("      offset 0 is the step the benchmark calls failure. Negative = before it.")

    # --- oracle validation checks that this pool supports ----------------------------------
    print("\n[4/5] oracle validation")
    v1 = far_m | (band <= -30)
    if v1.sum():
        print(f"      V1 benign recoverable      : {recoverable[v1].mean():.3f} "
              f"(n={int(v1.sum())}, target >= 0.99)")
    settled = band >= 100
    if settled.sum():
        print(f"      V2 settled-fallen irrecov. : {1 - recoverable[settled].mean():.3f} "
              f"(n={int(settled.sum())}, target >= 0.98)")

    per_ep = {}
    for k, i in enumerate(idx):
        if recoverable[k]:
            continue
        e = int(cat["episode_idx"][i])
        per_ep[e] = min(per_ep.get(e, 10**6), int(band[k]))
    lead = np.array(sorted(-v for v in per_ep.values() if v > -10**5)) if per_ep else None
    if lead is not None and len(lead):
        print(f"      lead time (steps from first irrecoverable to the benchmark's flag): "
              f"median {np.median(lead):.0f}, p25 {np.percentile(lead, 25):.0f}, "
              f"p75 {np.percentile(lead, 75):.0f}   (need median >= 10)")

    # --- matched pairs and the trivial baselines -------------------------------------------
    a_idx, b_idx = lm.build_matched_pairs(cat["qpos"], recoverable, idx, rng=rng)
    print(f"\n[5/5] matched pairs: {len(a_idx)} "
          f"(from {int((~recoverable).sum())} irrecoverable, {int(recoverable.sum())} recoverable)")

    base = lm.trivial_baselines(cat["qpos"][idx], cat["qvel"][idx], cat["observation"][idx])
    brows = []
    for name, score in base.items():
        row = {"baseline": name, "population_auroc": lm.auroc(score, ~recoverable)}
        if len(a_idx):
            sa = lm.trivial_baselines(cat["qpos"][a_idx], cat["qvel"][a_idx],
                                      cat["observation"][a_idx])[name]
            sb = lm.trivial_baselines(cat["qpos"][b_idx], cat["qvel"][b_idx],
                                      cat["observation"][b_idx])[name]
            row["matched_auroc"] = lm.paired_auroc(sa, sb)
        brows.append(row)
    print()
    print(lm.format_table(brows))

    best = max((r.get("matched_auroc") or 0.0) for r in brows) if brows else 0.0
    print("\n=== VERDICT ===")
    print(f"strongest trivial baseline on matched pairs: {best:.3f}")
    if len(a_idx) < 200:
        print("  PIVOT: too few matched pairs. Irrecoverability is a pose threshold here.")
    elif best > 0.90:
        print("  PIVOT: pose alone separates the pairs. A latent method cannot add anything.")
    elif best <= 0.75:
        print("  PROCEED: pose does not separate matched pairs, so the signal is in the")
        print("           dynamics, which is what a world-model rollout is for.")
    else:
        print("  MARGINAL: tighten the pose tolerance or add impulse-perturbation pairs.")
    if lead is not None and len(lead) and np.median(lead) < 10:
        print("  WARNING: median lead time under 10 steps. Keep H1/H2, drop H3 to a stretch goal.")

    summary = {
        "robot": args.robot, "version": args.version, "gate0": gate, "lift": lift,
        "n_labelled": int(len(idx)), "use_cem": not args.no_cem, "horizon": args.horizon,
        "recoverability_by_offset": rows, "baselines": brows,
        "n_matched_pairs": int(len(a_idx)),
        "lead_time_median": float(np.median(lead)) if lead is not None and len(lead) else None,
        "oracle_seconds": dt,
    }
    tag = f"{args.robot}_{args.version}" + ("_nocem" if args.no_cem else "")
    (out_dir / f"triage_{tag}.json").write_text(json.dumps(summary, indent=2, default=float))
    np.savez_compressed(
        out_dir / f"labels_{tag}.npz",
        idx=idx, recoverable=recoverable, tier=np.array(tiers, dtype="S4"), offset=band,
        qpos=cat["qpos"][idx], qvel=cat["qvel"][idx], matched_a=a_idx, matched_b=b_idx,
    )
    print(f"\nwrote {out_dir}/triage_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
