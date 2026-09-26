#!/usr/bin/env python3
"""S4: the minimum study on Walker2d (walker2d.md): four-source decomposition for the speed
and health rules, then predictor-side adaptation with random versus boundary acquisition at
two budgets with several acquisition seeds. Uses the S3 probes and LeWM-A.

    uv run python experiments/scripts/walker_s4_study.py --model runs/<lewm-a> --probes runs/<probes> [--budgets 128 512] [--seeds 0 1 2]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers.threads import apply_torch, pin_threads  # noqa: E402

pin_threads()

import numpy as np  # noqa: E402
import torch  # noqa: E402

apply_torch()

from helpers.dialMetrics import auc_dial, clearance_error_stats, cluster_bootstrap, fsa, margin_for_acceptance  # noqa: E402
from helpers.hfStore import HFStore  # noqa: E402
from helpers.locoData import RenderContext  # noqa: E402
from helpers.locoEnv import make_loco_env  # noqa: E402
from helpers.poseProbes import load_probe  # noqa: E402
from helpers.predictorAdapt import AdaptConfig, ClipSet, adapt, predictor_side_state  # noqa: E402
from helpers.runManifest import build_manifest, make_run_id, write_manifest  # noqa: E402
from helpers.walkerBank import HORIZON_STEPS, WalkerBank, build_walker_bank, endpoint_targets, interp_steps, sample_roots  # noqa: E402
from helpers.walkerLewm import FRAMESKIP, HISTORY, WalkerImaginer, load_walker_model  # noqa: E402
from helpers.walkerRules import HORIZON_BLOCKS, health_clearance, speed_clearance  # noqa: E402

DATA = REPO_ROOT / "data" / "study" / "walker2d"
RESULTS = REPO_ROOT / "docs" / "mainPlan" / "results" / "s4"
RULES = ("speed", "health")


def rule_clearance(rule, h, p, v):
    return speed_clearance(v) if rule == "speed" else health_clearance(h, p)


class RootCache:
    def __init__(self, im: WalkerImaginer, ctx: RenderContext):
        self.im, self.ctx, self.cache = im, ctx, {}

    def get(self, root):
        if root.root_id not in self.cache:
            frames = self.ctx.render_many(root.history_qpos, root.history_qvel)
            self.cache[root.root_id] = (self.im.encode(frames), frames)
        return self.cache[root.root_id]


def analyse(bank: WalkerBank, im: WalkerImaginer, probe, ctx: RenderContext, *, sources_all: bool = True) -> list[dict]:
    """Per branch: min clearance per rule per source. Sources: dense, endpoint, real_readout, imagined."""
    cache = RootCache(im, ctx)
    rows = []
    for ri, root in enumerate(bank.roots):
        idx = bank.indices_for_root(ri)
        if len(idx) == 0:
            continue
        z_hist, _ = cache.get(root)
        tapes = np.stack([bank.h5["tape"][int(j)] for j in idx]).astype(np.float64)
        z_imag = im.rollout(z_hist, root.history_actions, tapes.reshape(len(idx), HORIZON_BLOCKS, FRAMESKIP, 6))
        n, K, D = z_imag.shape
        pred = probe.predict(z_imag.reshape(-1, D)).reshape(n, K, 3)
        y0 = probe.predict(z_hist[-1:])[0]
        for a, j in enumerate(idx):
            j = int(j)
            qp, xv = bank.h5["qpos"][j], bank.h5["x_velocity"][j]
            truth = endpoint_targets(qp, xv)  # (K, 3)
            root_true = np.array([qp[0, 1], qp[0, 2], root.meta.get("x_velocity", 0.0)])
            row = {"bank": bank.dir.name, "branch": j, "root": ri, "kind": bank.h5["kind"][j].decode() if isinstance(bank.h5["kind"][j], bytes) else str(bank.h5["kind"][j])}
            for rule in RULES:
                dense = rule_clearance(rule, qp[1:, 1], qp[1:, 2], xv)
                row[f"cmin_dense_{rule}"] = float(dense.min())
                ends = np.vstack([root_true, truth])
                row[f"cmin_endpoint_{rule}"] = float(rule_clearance(rule, interp_steps(ends[:, 0]), interp_steps(ends[:, 1]), interp_steps(ends[:, 2])).min())
                imag = np.vstack([y0, pred[a]])
                row[f"cmin_imagined_{rule}"] = float(rule_clearance(rule, interp_steps(imag[:, 0]), interp_steps(imag[:, 1]), interp_steps(imag[:, 2])).min())
                if sources_all:
                    real = np.vstack([y0, probe.predict(im.encode(bank.h5["frames"][j][1:]))])
                    row[f"cmin_real_readout_{rule}"] = float(rule_clearance(rule, interp_steps(real[:, 0]), interp_steps(real[:, 1]), interp_steps(real[:, 2])).min())
            row["fell"] = bool((qp[1:, 1] < 0.8).any())
            row["speed_max"] = float(xv.max())
            rows.append(row)
    return rows


def table(rows, rule, m, sources):
    u = np.array([r[f"cmin_dense_{rule}"] < 0 for r in rows])
    root = np.array([r["root"] for r in rows])
    out = {"n": int(len(rows)), "n_unsafe": int(u.sum())}
    for s in sources:
        c = np.array([r[f"cmin_{s}_{rule}"] for r in rows])
        out[s] = fsa(c, u, m)
    if "imagined" in sources:
        c = np.array([r[f"cmin_imagined_{rule}"] for r in rows])
        out["imagined"]["ci"] = cluster_bootstrap(lambda c, u: fsa(c, u, m)["fsa"], root, n_boot=300, c=c, u=u)
        out["auc_dial_imagined"] = auc_dial(c, u)
        out["clearance_error_imagined"] = clearance_error_stats(c, np.array([r[f"cmin_dense_{rule}"] for r in rows]))
        if all(f"cmin_endpoint_{rule}" in r and f"cmin_real_readout_{rule}" in r for r in rows):
            acc = {s: np.array([r[f"cmin_{s}_{rule}"] >= m for r in rows]) for s in ("endpoint", "real_readout", "imagined")}
            fs4 = acc["imagined"] & u
            out["attribution"] = {"n_false_safe_imagined": int(fs4.sum()), "temporal": int((fs4 & acc["endpoint"]).sum()), "readout": int((fs4 & ~acc["endpoint"] & acc["real_readout"]).sum()), "imagination": int((fs4 & ~acc["endpoint"] & ~acc["real_readout"]).sum())}
    return out


def clips_from_bank(bank: WalkerBank, im: WalkerImaginer, ctx: RenderContext, indices, scaler) -> ClipSet:
    cache = RootCache(im, ctx)
    Z, A = [], []
    mean, std = scaler
    for j in indices:
        j = int(j)
        root = bank.roots[int(bank.h5["root_index"][j])]
        z_hist, _ = cache.get(root)
        z_branch = im.encode(bank.h5["frames"][j][1:])
        blocks = np.concatenate([root.history_actions, bank.h5["tape"][j].astype(np.float64).reshape(HORIZON_BLOCKS, FRAMESKIP, 6)], 0)
        Z.append(torch.cat([z_hist, z_branch]).cpu().numpy())
        A.append(((blocks - mean) / std).reshape(len(blocks), -1).astype(np.float32))
    return ClipSet(np.stack(Z).astype(np.float32), np.stack(A), {"source": "walker_branches", "n": len(Z)})


def replay_clips(h5_path: Path, im: WalkerImaginer, ctx: RenderContext, rng, n_clips: int, scaler) -> ClipSet:
    """Ordinary set-A windows: HISTORY + HORIZON_BLOCKS endpoint frames and the blocks between."""
    from helpers.walkerBank import iter_episodes

    mean, std = scaler
    T = HISTORY + HORIZON_BLOCKS
    span = (T - 1) * FRAMESKIP + 1
    Z, A = [], []
    for ei, ep in iter_episodes(h5_path):
        n = len(ep["qpos"])
        if n < span:
            continue
        for s in rng.choice(np.arange(0, n - span + 1), size=min(2, n - span + 1), replace=False):
            idx = s + np.arange(T) * FRAMESKIP
            frames = ctx.render_many(ep["qpos"][idx], ep["qvel"][idx])
            Z.append(im.encode(frames).cpu().numpy())
            blocks = np.asarray(ep["action"][s : s + (T - 1) * FRAMESKIP], float).reshape(T - 1, FRAMESKIP, 6)
            A.append(((blocks - mean) / std).reshape(T - 1, -1).astype(np.float32))
            if len(Z) >= n_clips:
                break
        if len(Z) >= n_clips:
            break
    return ClipSet(np.stack(Z).astype(np.float32), np.stack(A), {"source": "setA_replay", "n": len(Z)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--dev-roots", type=int, default=24)
    ap.add_argument("--test-roots", type=int, default=96)
    ap.add_argument("--tapes", type=int, default=8)
    ap.add_argument("--budgets", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--replay-clips", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--band", type=float, default=0.5)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    device = "cuda"
    run_id = make_run_id("walker2d", "s4", n=args.n)
    model, scaler = load_walker_model(Path(args.model), device)
    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    im = WalkerImaginer(model, scaler, device)
    probe, _ = load_probe(Path(args.probes) / "walker_mlp.pt", device)
    ctx = RenderContext()
    env = make_loco_env("Walker2d", "v1", render=True, terminate_when_unhealthy=False, six_tuple=True)
    env.reset(seed=0)
    rng = np.random.default_rng(20261020)

    # ---- banks (whole episodes split by parity into dev/test/adaptation roles) --------------
    banks = {}
    for name, n_roots, kind, filt in (("dev", args.dev_roots, "representative", lambda e: e % 4 == 0), ("test", args.test_roots, "representative", lambda e: e % 4 == 1),
                                     ("stress", args.test_roots, "stress", lambda e: e % 4 == 1), ("acq", args.test_roots, "representative", lambda e: e % 4 in (2, 3))):
        bank_dir = data_dir / f"bank_{name}"
        if (bank_dir / "roots.json").is_file():
            banks[name] = WalkerBank(bank_dir)
            continue
        shutil.rmtree(bank_dir, ignore_errors=True)
        roots = sample_roots(data_dir / "roots.h5", rng, n_roots, kind=kind, episode_filter=filt, prefix=name)
        if name == "stress" and len(roots) < n_roots:
            roots += sample_roots(data_dir / "roots.h5", rng, n_roots - len(roots), kind="representative", episode_filter=filt, prefix="stress2")
        banks[name] = build_walker_bank(bank_dir, roots, rng, env, n_tapes=args.tapes, stress=(name == "stress"), seed=20261020)
        print(f"[s4] bank {name}: {len(banks[name].roots)} roots, {len(banks[name])} branches")

    # ---- decomposition ---------------------------------------------------------------------
    report = {"run_id": run_id, "model": args.model, "probes": args.probes, "decomposition": {}, "adaptation": {}}
    rows_by_bank = {n: analyse(b, im, probe, ctx) for n, b in banks.items() if n != "acq"}
    srcs = ["dense", "endpoint", "real_readout", "imagined"]
    target_ar, m_matched = {}, {}
    for rule in RULES:
        c_dev = np.array([r[f"cmin_imagined_{rule}"] for r in rows_by_bank["dev"]])
        target_ar[rule] = float((c_dev >= 0).mean())
        m_matched[rule] = margin_for_acceptance(c_dev, target_ar[rule])
        report["decomposition"][rule] = {n: {"at_m0": table(rows, rule, 0.0, srcs), "at_matched": table(rows, rule, m_matched[rule], srcs), "margin_matched": m_matched[rule]} for n, rows in rows_by_bank.items()}
        for n in rows_by_bank:
            t = report["decomposition"][rule][n]["at_m0"]
            print(f"[s4] {rule:6s} {n:6s} @m=0: " + " | ".join(f"{s} FSA {t[s]['fsa']:.3f} AR {t[s]['acceptance_rate']:.2f}" for s in srcs) + f" | attribution {t.get('attribution')}")
    (RESULTS / "decomposition.json").write_text(json.dumps(report["decomposition"], indent=1, default=float) + "\n")

    # ---- adaptation: random vs boundary ----------------------------------------------------
    replay = replay_clips(data_dir / "setA.h5", im, ctx, rng, args.replay_clips, scaler)
    acq = banks["acq"]
    cfg = AdaptConfig(modules="predictor_side", loss="teacher_forced", lr=5e-5, steps=args.steps, ctx=HISTORY)

    def evaluate(m):
        imx = WalkerImaginer(m, scaler, device)
        out = {}
        for n in ("dev", "test", "stress"):
            rows = analyse(banks[n], imx, probe, ctx, sources_all=False)
            out[n] = {}
            for rule in RULES:
                c = np.array([r[f"cmin_imagined_{rule}"] for r in rows])
                u = np.array([r[f"cmin_dense_{rule}"] < 0 for r in rows])
                mm = margin_for_acceptance(np.array([r[f"cmin_imagined_{rule}"] for r in rows]), target_ar[rule]) if n == "dev" else out["dev"][rule]["margin"]
                out[n][rule] = {"margin": float(mm), "at_matched": fsa(c, u, mm), "auc_dial": auc_dial(c, u), "clearance_error": clearance_error_stats(c, np.array([r[f"cmin_dense_{rule}"] for r in rows]))}
        return out

    report["adaptation"]["no_update"] = evaluate(model)
    pool_rows = analyse(acq, im, probe, ctx, sources_all=False)  # scores with the unadapted model (round 0)
    for seed in args.seeds:
        srng = np.random.default_rng(seed)
        for arm in ("random", "boundary"):
            chosen: list[int] = []
            m_cur = model
            curve = []
            remaining = [r["branch"] for r in pool_rows]
            for budget in args.budgets:
                need = budget - len(chosen)
                if arm == "random":
                    pick = list(srng.choice(remaining, size=need, replace=False))
                else:
                    scored = analyse(acq, WalkerImaginer(m_cur, scaler, device), probe, ctx, sources_all=False)
                    key = {r["branch"]: min(abs(r["cmin_imagined_speed"] - m_matched["speed"]) / 0.5, abs(r["cmin_imagined_health"] - m_matched["health"]) / args.band) for r in scored}
                    pick = sorted(remaining, key=lambda b: key[b] + srng.uniform(0, 1e-3))[:need]
                chosen += [int(b) for b in pick]
                remaining = [b for b in remaining if b not in set(pick)]
                clips = clips_from_bank(acq, im, ctx, chosen, scaler)
                m_cur, log = adapt(base_sd, model, clips, replay, AdaptConfig(**dict(cfg.to_dict(), seed=seed)), device=device, verbose=False)
                ev = evaluate(m_cur)
                curve.append({"budget": budget, "charged_steps": budget * HORIZON_STEPS, "eval": ev, "train_time_s": log["wall_clock_s"]})
                print(f"[s4] seed {seed} {arm:8s} budget {budget}: " + " | ".join(f"{rule} test FSA {ev['test'][rule]['at_matched']['fsa']:.3f} stress {ev['stress'][rule]['at_matched']['fsa']:.3f}" for rule in RULES))
            report["adaptation"].setdefault(arm, {})[str(seed)] = curve
            run_dir = REPO_ROOT / "runs" / f"{run_id}-{arm}-s{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(predictor_side_state(m_cur), run_dir / "weights.pt")
            write_manifest(run_dir, build_manifest(run_id=run_dir.name, kind="adapted", seeds={"seed": seed}, data={"arm": arm, "budget": args.budgets[-1]}, metrics=curve[-1]["eval"]["test"]))
            (run_dir / "README.md").write_text(f"# {run_dir.name}\n\nWalker2d LeWM-A predictor-side adaptation, {arm} arm, seed {seed}.\n")
            if not args.no_upload:
                HFStore().upload_run("walker2d", "adapted", run_dir, run_id=run_dir.name)
            (RESULTS / "adaptation.json").write_text(json.dumps(report["adaptation"], indent=1, default=float) + "\n")
    report["wall_clock_s"] = time.time() - t0
    (RESULTS / "study.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(f"[s4] done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
