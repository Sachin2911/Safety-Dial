#!/usr/bin/env python3
"""S3 go/no-go (18 October 2026): can the Walker2d LeWM be read out and does it imagine?

On real encoded frames, trajectory-split linear and MLP probes for (height, pitch, speed).
Pass if height and pitch reach R2 >= 0.9 and speed >= 0.8 (provisional), imagined errors
grow measurably but not uselessly with horizon, and real-readout decisions track dense
truth for both rules. Writes docs/mainPlan/results/s3/gate.json and pushes the probes to
<ns>/safetydial-walker2d:probes/<run_id>.

    uv run python experiments/scripts/walker_s3_gate.py --model runs/<lewm-a run dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers.threads import apply_torch, pin_threads  # noqa: E402

pin_threads()

import h5py  # noqa: E402
import hdf5plugin  # noqa: E402,F401
import numpy as np  # noqa: E402

apply_torch()

from helpers.dialMetrics import fsa  # noqa: E402
from helpers.hfStore import HFStore  # noqa: E402
from helpers.locoData import RenderContext  # noqa: E402
from helpers.locoEnv import make_loco_env  # noqa: E402
from helpers.poseProbes import ProbeSpec, fit_probe, save_probe, split_by_trajectory  # noqa: E402
from helpers.runManifest import build_manifest, make_run_id, write_manifest  # noqa: E402
from helpers.walkerLewm import FRAMESKIP, HISTORY, WalkerImaginer, load_walker_model  # noqa: E402
from helpers.walkerRules import HORIZON_BLOCKS, execute_branch, health_clearance, roots_from_episode, speed_clearance  # noqa: E402

DATA = REPO_ROOT / "data" / "study" / "walker2d"
RESULTS = REPO_ROOT / "docs" / "mainPlan" / "results" / "s3"


def episodes(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        ln, off = f["ep_len"][:], f["ep_offset"][:]
        for i in range(len(ln)):
            a, b = int(off[i]), int(off[i] + ln[i])
            yield i, {k: f[k][a:b] for k in ("qpos", "qvel", "action", "x_velocity", "healthy")}


def targets_at(ep, idx):
    xv = ep["x_velocity"]
    return np.stack([ep["qpos"][idx, 1], ep["qpos"][idx, 2], np.where(idx > 0, xv[np.maximum(idx - 1, 0)], 0.0)], 1).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe-frames", type=int, default=30000)
    ap.add_argument("--n-roots", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    run_id = make_run_id("walker2d", "probes", n=args.n)
    model, scaler = load_walker_model(Path(args.model), device)
    im = WalkerImaginer(model, scaler, device)
    ctx = RenderContext()
    rng = np.random.default_rng(args.seed)

    # ---- probes on real frames (probe set, split by episode) ---------------------------
    Z, Y, E = [], [], []
    got = 0
    for ei, ep in episodes(DATA / "probe.h5"):
        idx = np.arange(0, len(ep["qpos"]), 2)
        frames = ctx.render_many(ep["qpos"][idx], ep["qvel"][idx])
        Z.append(im.encode(frames).cpu().numpy())
        Y.append(targets_at(ep, idx))
        E.append(np.full(len(idx), ei))
        got += len(idx)
        if got >= args.probe_frames:
            break
    Z, Y, E = np.concatenate(Z), np.concatenate(Y), np.concatenate(E)
    tr, va = split_by_trajectory(E, rng, 0.2)
    print(f"[s3] probe frames {len(Z):,} from {len(np.unique(E))} episodes")
    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    probes, report = {}, {"model": args.model, "probes": {}}
    for kind in ("linear", "mlp"):
        p, st = fit_probe(Z[tr], Y[tr], Z[va], Y[va], ProbeSpec(target="walker", kind=kind, seed=args.seed), device=device, verbose=False)
        save_probe(p, st, run_dir / f"walker_{kind}.pt")
        probes[kind] = p
        v = st["val"]
        report["probes"][kind] = {k: v[k] for k in ("r2", "rmse", "height_r2", "pitch_r2", "speed_r2", "height_rmse", "pitch_rmse", "speed_rmse")}
        print(f"[s3] {kind:6s} val R2 height {v['height_r2']:.3f} pitch {v['pitch_r2']:.3f} speed {v['speed_r2']:.3f} | rmse {v['height_rmse']:.3f} m {v['pitch_rmse']:.3f} rad {v['speed_rmse']:.3f} m/s")
    probe = probes["mlp"]

    # ---- imagined error by horizon and decisions on roots ---------------------------------
    env = make_loco_env("Walker2d", "v1", render=True, terminate_when_unhealthy=False, six_tuple=True)
    env.reset(seed=0)
    rows = []
    err_by_block = {"height": [], "pitch": [], "speed": []}
    real_err = {"height": [], "pitch": [], "speed": []}
    n_done = 0
    for ei, ep in episodes(DATA / "roots.h5"):
        n = len(ep["qpos"])
        if n < (HISTORY - 1) * FRAMESKIP + HORIZON_BLOCKS * FRAMESKIP + 1:
            continue
        steps = rng.choice(np.arange((HISTORY - 1) * FRAMESKIP, n - HORIZON_BLOCKS * FRAMESKIP), size=min(2, n // 200 + 1), replace=False)
        for root in roots_from_episode(ep, steps, episode=ei):
            hist_frames = ctx.render_many(root.history_qpos, root.history_qvel)
            z_hist = im.encode(hist_frames)
            tapes = np.stack([root.policy_tape] + [np.clip(root.policy_tape + rng.normal(0, s, root.policy_tape.shape), -1, 1) for s in (0.1, 0.3)])
            z_imag = im.rollout(z_hist, root.history_actions, tapes.reshape(len(tapes), HORIZON_BLOCKS, FRAMESKIP, 6))
            for a, tape in enumerate(tapes):
                log = execute_branch(env, root.qpos, root.qvel, tape, render_every=FRAMESKIP)
                true_end = np.stack([log.qpos[FRAMESKIP::FRAMESKIP, 1], log.qpos[FRAMESKIP::FRAMESKIP, 2], log.x_velocity[FRAMESKIP - 1 :: FRAMESKIP]], 1)
                pred = probe.predict(z_imag[a])
                real = probe.predict(im.encode(log.frames[1:]))
                for k, i in (("height", 0), ("pitch", 1), ("speed", 2)):
                    err_by_block[k].append(np.abs(pred[:, i] - true_end[:, i]))
                    real_err[k].append(np.abs(real[:, i] - true_end[:, i]))
                # decisions: min clearance per source (endpoint-interpolated for readouts)
                c_true = {"speed": float(speed_clearance(log.x_velocity).min()), "health": float(health_clearance(log.qpos[1:, 1], log.qpos[1:, 2]).min())}
                c_real = {"speed": float(speed_clearance(real[:, 2]).min()), "health": float(health_clearance(real[:, 0], real[:, 1]).min())}
                c_imag = {"speed": float(speed_clearance(pred[:, 2]).min()), "health": float(health_clearance(pred[:, 0], pred[:, 1]).min())}
                rows.append({"root": root.root_id, "tape": a, **{f"true_{k}": v for k, v in c_true.items()}, **{f"real_{k}": v for k, v in c_real.items()}, **{f"imag_{k}": v for k, v in c_imag.items()}})
            n_done += 1
        if n_done >= args.n_roots:
            break
    report["imagined_abs_error_by_block"] = {k: np.stack(v).mean(0).tolist() for k, v in err_by_block.items()}
    report["real_readout_abs_error_by_block"] = {k: np.stack(v).mean(0).tolist() for k, v in real_err.items()}
    dec = {}
    for rule in ("speed", "health"):
        u = np.array([r[f"true_{rule}"] < 0 for r in rows])
        dec[rule] = {"n": int(len(rows)), "n_unsafe": int(u.sum()), "real_readout": fsa(np.array([r[f"real_{rule}"] for r in rows]), u, 0.0), "imagined": fsa(np.array([r[f"imag_{rule}"] for r in rows]), u, 0.0),
                     "real_agreement": float(np.mean((np.array([r[f"real_{rule}"] for r in rows]) < 0) == u))}
    report["decisions_m0"] = dec
    mlp = report["probes"]["mlp"]
    e = report["imagined_abs_error_by_block"]
    grows = all(e[k][-1] > e[k][0] for k in e)
    useful = e["height"][-1] < 0.3 and e["pitch"][-1] < 0.6
    report["gate"] = {"height_pass": mlp["height_r2"] >= 0.9, "pitch_pass": mlp["pitch_r2"] >= 0.9, "speed_pass": mlp["speed_r2"] >= 0.8,
                      "imagined_error_grows": bool(grows), "imagined_error_useful": bool(useful),
                      "real_readout_tracks_truth": {r: dec[r]["real_agreement"] >= 0.85 for r in dec}}
    g = report["gate"]
    g["health_rule_ok"] = bool(g["height_pass"] and g["pitch_pass"] and g["imagined_error_useful"])
    g["speed_rule_ok"] = bool(g["speed_pass"])
    g["go"] = bool(g["health_rule_ok"])
    print(f"[s3] GATE: {json.dumps(g)}")
    report["wall_clock_s"] = time.time() - t0
    (RESULTS / "gate.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    write_manifest(run_dir, build_manifest(run_id=run_id, kind="probes", seeds={"seed": args.seed}, data={"model": args.model, "n_probe_frames": int(len(Z))}, metrics=report["probes"], started_at=t0))
    (run_dir / "README.md").write_text(f"# {run_id}\n\nWalker2d readout probes (height, pitch, speed) on frozen latents of {args.model}. Not safety supervision.\n")
    if not args.no_upload:
        HFStore().upload_run("walker2d", "probes", run_dir, run_id=run_id)
    print(f"[s3] done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
