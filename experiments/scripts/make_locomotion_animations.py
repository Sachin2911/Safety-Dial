#!/usr/bin/env python3
"""Animate the Phase 0 recoverability result: falls, get-ups, and the ones that fail.

Every frame is the real MuJoCo renderer replaying a simulator state, so these are the same
trajectories the oracle scored, not a reconstruction. The overlay carries the numbers the
argument rests on: torso height, the benchmark's healthy band, and which phase of the episode
each frame belongs to.

Three clips, and the contrast between the last two is the point:

  walker2d_fall_and_getup   the robot walks, the benchmark declares failure while it is still
                            standing, it goes all the way to the ground, and then a CEM plan
                            stands it back up. This is why Walker2d cannot support the
                            proposal's experiment: nothing about the fall is irreversible.
  hopper_getup              a Hopper state the oracle recovered.
  hopper_irrecoverable      a Hopper state the oracle could not recover, showing its best
                            attempt failing. Hopper has genuine irreversibility; Walker2d
                            does not.

    uv run python experiments/scripts/make_locomotion_animations.py
    uv run python experiments/scripts/make_locomotion_animations.py --only walker2d_fall_and_getup
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

import helpers.locoCollect as lc  # noqa: E402
import helpers.locoEnv as le  # noqa: E402
import helpers.locoPolicies as lp  # noqa: E402
import helpers.oracle as orc  # noqa: E402

OUT = REPO_ROOT / "animations"
SIZE = 480
FPS = 50  # the sim runs at 125 Hz, so this is 0.4x real time and readable

WHITE, AMBER, RED, GREEN, GREY = (
    (245, 245, 245), (52, 140, 208), (107, 102, 193), (60, 140, 60), (150, 150, 150)
)  # BGR, because cv2


def band(img, text, y, colour=WHITE, scale=0.5, thick=1):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2,
                cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def annotate(frame, *, phase, phase_colour, step, z, pitch, healthy, z_range, note=None):
    """Draw the state readout. The healthy band is shown because it is the thing in question."""
    img = np.ascontiguousarray(frame[:, :, ::-1])  # RGB -> BGR
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 74), (24, 24, 24), -1)
    cv2.rectangle(img, (0, h - 30), (w, h), (24, 24, 24), -1)

    band(img, phase, 24, phase_colour, 0.62, 2)
    band(img, f"step {step:4d}   torso z = {z:5.3f}   pitch = {pitch:+6.3f} rad", 48, WHITE, 0.46)
    state = "healthy" if healthy else "UNHEALTHY (benchmark calls this failed)"
    band(img, state, 68, GREEN if healthy else RED, 0.44)
    band(img, note or f"healthy band: {z_range[0]} < z < {z_range[1]}", h - 10, GREY, 0.42)

    # A height gauge on the right, so the fall and the recovery are legible at a glance.
    x0, y0, y1 = w - 26, 86, h - 42
    cv2.rectangle(img, (x0, y0), (x0 + 12, y1), (60, 60, 60), -1)
    lo, hi = 0.0, 2.0
    def ypix(v):
        return int(y1 - (np.clip(v, lo, hi) - lo) / (hi - lo) * (y1 - y0))
    cv2.rectangle(img, (x0, ypix(z_range[1])), (x0 + 12, ypix(z_range[0])), (70, 110, 70), -1)
    cv2.rectangle(img, (x0, ypix(z)), (x0 + 12, y1), (200, 170, 90), -1)
    cv2.line(img, (x0 - 3, ypix(z)), (x0 + 15, ypix(z)), WHITE, 1)
    return img[:, :, ::-1]  # back to RGB


def cem_plan(sim, qpos, qvel, cfg, seed):
    """Return (sequence, recovered). On failure, the best-scoring sequence found."""
    rng = np.random.default_rng(seed)
    a_dim, H, k = sim.action_dim, cfg.horizon, cfg.cem_knots
    pop = max(cfg.cem_elites * 2, cfg.cem_pop)
    best_seq, best_score = None, -np.inf
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
                    return seq, True
                scores[j] = sc
                if sc > best_score:
                    best_score, best_seq = sc, seq
            el = samples[np.argsort(-scores)[: cfg.cem_elites]]
            mean = el.mean(axis=0)
            sigma = np.maximum(el.std(axis=0), 1e-3) * cfg.cem_sigma_decay
    return best_seq, False


def episode_to_fallen(robot, seed, past=140, render=True):
    """Roll out until the robot is well past the benchmark's cutoff.

    `render=False` skips building a GL context entirely. That matters: an earlier version of
    the Hopper search built a rendering env per candidate seed and abandoned it, and after a
    dozen or so abandoned EGL contexts the driver quietly started returning black frames rather
    than raising. It is the same silent-failure mode `locoData.RenderContext` guards against,
    and it is worth not reproducing in a script whose whole job is to produce pictures.
    """
    env = le.make_loco_env(robot, "v1", render=render, width=SIZE, height=SIZE,
                           terminate_when_unhealthy=False, time_limit=False)
    pol = lp.MixedPolicy(
        [lp.RandomActionPolicy(env.action_space, seed=seed),
         lp.ScriptedForwardPolicy(env.action_space, seed=seed + 1, smooth=0.8, noise=0.3)],
        weights=[0.35, 0.65], seed=seed + 2)
    cols = lc.rollout_episode(env, pol, seed=seed, stop_after_unhealthy=past)
    if not render:
        return env, cols, None
    u = env.unwrapped
    frames = []
    for qp, qv in zip(cols["qpos"], cols["qvel"]):
        u.set_state(qp.copy(), qv.copy())
        frames.append(env.render())
    return env, cols, frames


def zrange(env):
    return tuple(env.unwrapped._healthy_z_range)


def write(name, frames, gif_stride=3, gif_width=360):
    OUT.mkdir(parents=True, exist_ok=True)
    mp4 = OUT / f"{name}.mp4"
    imageio.mimsave(mp4, frames, fps=FPS, quality=7, macro_block_size=None)
    small = [cv2.resize(f, (gif_width, gif_width), interpolation=cv2.INTER_AREA)
             for f in frames[::gif_stride]]
    gif = OUT / f"{name}.gif"
    imageio.mimsave(gif, small, duration=gif_stride / FPS, loop=0)
    print(f"  {mp4.name}  {mp4.stat().st_size / 1e6:.2f} MB   "
          f"{gif.name}  {gif.stat().st_size / 1e6:.2f} MB   ({len(frames)} frames)")


def clip_fall_and_getup(robot="Walker2d", seed=7, offset=120):
    """Walk, fall past the flag, then stand back up under a CEM plan."""
    env, cols, frames = episode_to_fallen(robot, seed)
    healthy = cols["healthy"]
    if not (healthy == 0).any():
        print(f"  {robot}: never went unhealthy at seed {seed}, skipping")
        return
    fu = int(np.argmax(healthy == 0))
    zr = zrange(env)
    n = len(healthy)
    start_i = min(fu + offset, n - 1)

    out = []
    for t in range(max(0, fu - 45), start_i + 1):
        z, p = cols["qpos"][t, 1], cols["qpos"][t, 2]
        rel = t - fu
        if rel < 0:
            phase, col = "WALKING", GREEN
        elif rel == 0:
            phase, col = "BENCHMARK DECLARES FAILURE", AMBER
        elif rel < 12:
            phase, col = "BENCHMARK DECLARES FAILURE", AMBER
        else:
            phase, col = "FALLEN", RED
        note = ("the flag fires here, and the robot is still standing"
                if 0 <= rel < 12 else None)
        out.append(annotate(frames[t], phase=phase, phase_colour=col, step=rel,
                            z=z, pitch=p, healthy=bool(healthy[t]), z_range=zr, note=note))
    for _ in range(FPS // 2):
        out.append(out[-1])

    cfg = orc.OracleConfig(robot=robot, version="v1", use_cem=True)
    sim = orc.RecoverySimulator(cfg)
    seq, ok = cem_plan(sim, cols["qpos"][start_i], cols["qvel"][start_i], cfg,
                       orc.state_seed(seed, start_i))
    print(f"  {robot} get-up from z={cols['qpos'][start_i, 1]:.3f}: recovered={ok}")

    u = env.unwrapped
    sim._reset_to(u, cols["qpos"][start_i], cols["qvel"][start_i])
    pred = sim.pred
    run = 0
    recovered_at = None
    for t, a in enumerate(seq):
        u.do_simulation(a, u.frame_skip)
        z, p = float(u.data.qpos[1]), float(u.data.qpos[2])
        run = run + 1 if pred.margin(u.data.qpos, u.data.qvel) > 0 else 0
        # Latch. The oracle's verdict is "the predicate held for `dwell` consecutive steps",
        # which is a property of the state, not of where the plan happens to end. This plan
        # stands the robot up and then lets it topple again, so an unlatched flag would flip
        # back to "still trying" and the clip would contradict the label the oracle assigned.
        if recovered_at is None and run >= pred.dwell:
            recovered_at = t
        done = recovered_at is not None
        out.append(annotate(env.render(),
                            phase="RECOVERED" if done else "RECOVERY PLAN (CEM)",
                            phase_colour=GREEN if done else AMBER, step=t, z=z, pitch=p,
                            healthy=bool(u.is_healthy), z_range=zr,
                            note=("upright and settled for 25 consecutive steps" if done
                                  else "same action bounds any policy has")))
        if done and t > recovered_at + FPS // 2:
            break
    write(f"{robot.lower()}_fall_and_getup", out)


def clip_hopper(kind, seed_pool=range(40, 90), offset=150):
    """A Hopper state the oracle recovered, or one it could not.

    Two passes on purpose. The first finds a matching seed using a non-rendering env, so the
    search does not accumulate GL contexts; only the winning seed gets rendered.
    """
    cfg = orc.OracleConfig(robot="Hopper", version="v1", use_cem=True)
    sim = orc.RecoverySimulator(cfg)
    pred = sim.pred

    chosen = None
    for seed in seed_pool:
        env, cols, _ = episode_to_fallen("Hopper", seed, past=offset + 20, render=False)
        env.close()
        healthy = cols["healthy"]
        if not (healthy == 0).any():
            continue
        fu = int(np.argmax(healthy == 0))
        i = min(fu + offset, len(healthy) - 1)
        if i - fu < offset - 5:
            continue
        seq, ok = cem_plan(sim, cols["qpos"][i], cols["qvel"][i], cfg, orc.state_seed(seed, i))
        if (kind == "getup") == ok:
            chosen = (seed, i, seq, ok, cols["qpos"][i, 1])
            break
    if chosen is None:
        print(f"  Hopper {kind}: no matching state found in the seed pool")
        return
    seed, i, seq, ok, z0 = chosen
    print(f"  Hopper {kind}: seed {seed}, z={z0:.3f}, recovered={ok}")

    env, cols, frames = episode_to_fallen("Hopper", seed, past=offset + 20, render=True)
    zr = zrange(env)
    u = env.unwrapped
    sim._reset_to(u, cols["qpos"][i], cols["qvel"][i])
    out = []
    for _ in range(FPS // 3):
        out.append(annotate(frames[i], phase="FALLEN", phase_colour=RED, step=0,
                            z=cols["qpos"][i, 1], pitch=cols["qpos"][i, 2],
                            healthy=False, z_range=zr,
                            note=f"{offset} steps past the benchmark's flag"))
    run = 0
    recovered_at = None
    for t, a in enumerate(seq):
        u.do_simulation(a, u.frame_skip)
        z, p = float(u.data.qpos[1]), float(u.data.qpos[2])
        run = run + 1 if pred.margin(u.data.qpos, u.data.qvel) > 0 else 0
        if recovered_at is None and run >= pred.dwell:
            recovered_at = t
        done = recovered_at is not None
        if ok:
            phase = "RECOVERED" if done else "RECOVERY PLAN (CEM)"
            col = GREEN if done else AMBER
            note = ("upright and settled for 25 consecutive steps" if done
                    else "same action bounds any policy has")
        else:
            phase = "NO RECOVERY FOUND (best attempt)"
            col = RED
            note = "never upright and settled for 25 consecutive steps"
        out.append(annotate(env.render(), phase=phase, phase_colour=col, step=t, z=z,
                            pitch=p, healthy=bool(u.is_healthy), z_range=zr, note=note))
        if done and t > recovered_at + FPS // 2:
            break
    write(f"hopper_{'getup' if ok else 'irrecoverable'}", out)
    env.close()


CLIPS = {
    "walker2d_fall_and_getup": lambda: clip_fall_and_getup(),
    "hopper_getup": lambda: clip_hopper("getup"),
    "hopper_irrecoverable": lambda: clip_hopper("irrecoverable"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(CLIPS), action="append")
    args = ap.parse_args()
    for name in (args.only or sorted(CLIPS)):
        print(f"[{name}]")
        CLIPS[name]()
    print(f"\nwrote to {OUT}")
