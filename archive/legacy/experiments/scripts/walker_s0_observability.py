#!/usr/bin/env python3
"""S0: can speed, height and pitch be read from stacks of three rendered frames?

For each candidate frameskip, roll out behaviour policies in the vendored env, render
224 px frames at block endpoints, and regress (height, pitch, speed) from a stack of three
consecutive endpoint frames with a small CNN, split by trajectory. If speed cannot be
recovered, the speed rule is dropped rather than feeding velocity in (walker2d.md).

    uv run python experiments/scripts/walker_s0_observability.py --frameskips 5 10 20
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

apply_torch()

from helpers.locoData import RenderContext, render_fingerprint  # noqa: E402
from helpers.locoEnv import make_loco_env  # noqa: E402
from helpers.locoPolicies import make_policy  # noqa: E402
from helpers.walkerAssets import list_exports, load_exported_actor  # noqa: E402

RESULTS = REPO_ROOT / "docs" / "mainPlan" / "results" / "s0"


def collect(env, policies, n_episodes, rng, max_steps=1000):
    """State-only rollouts: qpos, qvel, obs, x_velocity per step, per episode."""
    eps = []
    u = env.unwrapped
    for i in range(n_episodes):
        pol = policies[i % len(policies)]
        obs, info = env.reset(seed=int(rng.integers(1 << 30)))
        rows = {"qpos": [], "qvel": [], "obs": [], "xvel": []}
        for t in range(max_steps):
            rows["qpos"].append(u.data.qpos.copy())
            rows["qvel"].append(u.data.qvel.copy())
            rows["obs"].append(np.asarray(obs, np.float64).copy())
            a = pol.act(obs)
            obs, r, term, trunc, info = env.step(a)
            rows["xvel"].append(float(info["x_velocity"]))
            if term or trunc:
                break
        eps.append({k: np.asarray(v) for k, v in rows.items()})
    return eps


class SmallCNN(torch.nn.Module):
    def __init__(self, in_ch=9, out=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, 32, 5, 2, 2), torch.nn.ReLU(), torch.nn.Conv2d(32, 64, 5, 2, 2), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, 2, 1), torch.nn.ReLU(), torch.nn.Conv2d(64, 64, 3, 2, 1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(4), torch.nn.Flatten(), torch.nn.Linear(64 * 16, 256), torch.nn.ReLU(), torch.nn.Linear(256, out))

    def forward(self, x):
        return self.net(x)


def build_stacks(ctx, eps, frameskip, rng, max_per_ep=40, res=96):
    """Stacks of 3 endpoint frames (downsampled to res) -> targets at the last frame."""
    import torch.nn.functional as F

    X, Y, T = [], [], []
    for ei, ep in enumerate(eps):
        n = len(ep["qpos"])
        idx = np.arange(0, n - 2 * frameskip, frameskip)
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, size=min(max_per_ep, len(idx)), replace=False)
        for t in pick:
            frames = [ctx.render_state(ep["qpos"][t + k * frameskip], ep["qvel"][t + k * frameskip]) for k in range(3)]
            x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0  # (3, C, H, W)
            x = F.interpolate(x, size=(res, res), mode="area").reshape(-1, res, res)
            tt = t + 2 * frameskip
            X.append(x.numpy().astype(np.float16))
            Y.append([ep["obs"][tt][0], ep["obs"][tt][1], ep["xvel"][tt - 1] if tt - 1 < len(ep["xvel"]) else ep["xvel"][-1]])
            T.append(ei)
    return np.stack(X), np.asarray(Y, np.float32), np.asarray(T)


def fit_cnn(X, Y, tr, va, device, epochs=25, seed=0):
    torch.manual_seed(seed)
    net = SmallCNN(X.shape[1]).to(device)
    ym, ys = Y[tr].mean(0), Y[tr].std(0) + 1e-6
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.from_numpy(X[tr]).to(device)
    yt = torch.from_numpy((Y[tr] - ym) / ys).to(device)
    xv = torch.from_numpy(X[va]).to(device)
    best, best_pred = np.inf, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), 64):
            b = perm[i : i + 64]
            loss = torch.nn.functional.mse_loss(net(xt[b].float()), yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(xv[i : i + 256].float()) for i in range(0, len(xv), 256)]).cpu().numpy() * ys + ym
        mse = float(((pv - Y[va]) ** 2).mean())
        if mse < best:
            best, best_pred = mse, pv
    resid = Y[va] - best_pred
    r2 = 1 - (resid**2).sum(0) / np.maximum(((Y[va] - Y[va].mean(0)) ** 2).sum(0), 1e-9)
    return {"r2_height": float(r2[0]), "r2_pitch": float(r2[1]), "r2_speed": float(r2[2]), "rmse": np.sqrt((resid**2).mean(0)).tolist(), "n_train": int(len(tr)), "n_val": int(len(va))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frameskips", nargs="+", type=int, default=[5, 10, 20])
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    rng = np.random.default_rng(args.seed)
    env = make_loco_env("Walker2d", "v1", render=False)
    policies = [make_policy("random", env.action_space, seed=1), make_policy("scripted_forward", env.action_space, seed=2)]
    for p in list_exports():
        policies.append(load_exported_actor(p, env.action_space, seed=3))
        policies.append(load_exported_actor(p, env.action_space, seed=4, action_noise=0.1))
    print(f"[s0] policies: {[getattr(p, 'name', type(p).__name__) for p in policies]}")
    eps = collect(env, policies, args.episodes, rng)
    speeds = np.concatenate([e["xvel"] for e in eps])
    print(f"[s0] {len(eps)} episodes, {len(speeds):,} steps; speed p10/50/90 {np.percentile(speeds, [10, 50, 90]).round(2)}; frac > 2.3415: {(speeds > 2.3415).mean():.3f}")
    ctx = RenderContext()
    fp = render_fingerprint(ctx)
    frame = ctx.render_state(eps[0]["qpos"][0], eps[0]["qvel"][0])
    print(f"[s0] render fingerprint {fp}; frame mean {frame.mean():.1f} (black frames would be ~0)")
    report = {"render_fingerprint": fp, "frame_mean": float(frame.mean()), "n_episodes": len(eps), "n_steps": int(len(speeds)),
              "speed_percentiles": np.percentile(speeds, [10, 50, 90]).tolist(), "frac_over_threshold": float((speeds > 2.3415).mean()), "frameskips": {}}
    tr_eps = set(rng.choice(len(eps), size=int(0.8 * len(eps)), replace=False).tolist())
    for fs in args.frameskips:
        X, Y, T = build_stacks(ctx, eps, fs, rng)
        tr = np.where([t in tr_eps for t in T])[0]
        va = np.where([t not in tr_eps for t in T])[0]
        res = fit_cnn(X, Y, tr, va, device)
        res["dt_per_block_s"] = fs * 0.008
        report["frameskips"][str(fs)] = res
        print(f"[s0] frameskip {fs} ({fs * 0.008:.3f}s/block): R2 height {res['r2_height']:.3f} pitch {res['r2_pitch']:.3f} speed {res['r2_speed']:.3f} (n={res['n_train']}/{res['n_val']})")
    report["wall_clock_s"] = time.time() - t0
    (RESULTS / "observability.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[s0] wrote {RESULTS / 'observability.json'} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
