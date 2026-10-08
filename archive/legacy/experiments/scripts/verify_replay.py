"""Verify that a stored `(qpos, qvel)` reproduces the trajectory it came from.

The whole storage design rests on this. Pixels are not stored, because 3.25M frames at 224x224
is 489 GB against 60 GB of free disk; instead state is stored at 264 bytes per step and frames
are re-rendered on demand. That is only sound if restoring the state and replaying the recorded
actions gives back exactly the recorded observations, and if rendering from a restored state
gives back exactly the same image.

Walker2d has `na == 0`, so `(qpos, qvel)` is the complete state with no actuator activations to
carry. `MujocoEnv.set_state` calls `mj_forward`, so the rendered frame is a pure function of
those two arrays.

Run:  uv run python experiments/scripts/verify_replay.py
Exits non-zero if any tolerance is missed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import helpers.locoCollect as lc  # noqa: E402
import helpers.locoEnv as le  # noqa: E402
import helpers.locoPolicies as lp  # noqa: E402

# `qpos`/`qvel` are stored as float64 and must match to machine precision: that is the invariant
# lazy rendering depends on. `observation` is stored as float32 to save 68 MB per million steps,
# so it can only be expected to match at float32 resolution (~1e-7 relative, and the observation
# carries values of order 10). Comparing it at 1e-10 would be comparing against the storage
# dtype, not against the physics.
OBS_TOL = 1e-5
QPOS_TOL = 1e-12


def main() -> int:
    ok = True
    env = le.make_loco_env("Walker2d", "v1", render=True, width=224, height=224)
    pol = lp.MixedPolicy(
        [
            lp.RandomActionPolicy(env.action_space, seed=0),
            lp.ScriptedForwardPolicy(env.action_space, seed=1, smooth=0.8, noise=0.3),
        ],
        seed=2,
    )

    print("=== replay exactness ===")
    worst_obs = worst_qpos = 0.0
    n_checked = 0
    for ep in range(5):
        cols = lc.rollout_episode(env, pol, seed=500 + ep)
        n = len(cols["reward"])
        if n < 12:
            continue
        for start in (0, n // 3, n // 2, max(0, n - 10)):
            u = env.unwrapped
            u.set_state(cols["qpos"][start].copy(), cols["qvel"][start].copy())
            span = min(20, n - start - 1)
            for k in range(span):
                env.step(cols["action"][start + k])
                j = start + k + 1
                worst_obs = max(worst_obs, float(np.abs(u._get_obs() - cols["observation"][j]).max()))
                worst_qpos = max(worst_qpos, float(np.abs(u.data.qpos - cols["qpos"][j]).max()))
                n_checked += 1

    print(f"  checked {n_checked} replayed transitions across 5 episodes")
    print(f"  max |obs  - recorded|  = {worst_obs:.3e}   (tolerance {OBS_TOL:.0e}, float32 storage)")
    print(f"  max |qpos - recorded|  = {worst_qpos:.3e}   (tolerance {QPOS_TOL:.0e}, float64)")
    if worst_obs > OBS_TOL or worst_qpos > QPOS_TOL:
        print("  FAIL: replay is not exact, so lazy rendering is unsound")
        ok = False
    else:
        print("  PASS")

    print("\n=== render determinism from restored state ===")
    cols = lc.rollout_episode(env, pol, seed=999)
    u = env.unwrapped
    frames = []
    for _ in range(3):
        u.set_state(cols["qpos"][5].copy(), cols["qvel"][5].copy())
        frames.append(env.render().copy())
    identical = all(np.array_equal(frames[0], f) for f in frames[1:])
    print(f"  3 renders of the same restored state identical: {identical}")

    # A different state must give a different frame, otherwise the check above is vacuous.
    u.set_state(cols["qpos"][min(40, len(cols["qpos"]) - 1)].copy(),
                cols["qvel"][min(40, len(cols["qvel"]) - 1)].copy())
    other = env.render().copy()
    differs = not np.array_equal(frames[0], other)
    print(f"  a different state renders differently:            {differs}")
    if not (identical and differs):
        print("  FAIL")
        ok = False
    else:
        print("  PASS")

    print("\n=== storage arithmetic ===")
    per_step = sum(
        np.dtype(dt).itemsize * (env.unwrapped.model.nq if n in ("qpos",) else
                                 env.unwrapped.model.nv if n in ("qvel",) else
                                 env.action_space.shape[0] if n == "action" else
                                 env.observation_space.shape[0] if n == "observation" else 1)
        for n, (dt, _) in lc.SCHEMA.items()
    )
    frame_bytes = 224 * 224 * 3
    print(f"  state schema: {per_step} bytes/step")
    print(f"  one frame   : {frame_bytes} bytes  ({frame_bytes / per_step:.0f}x larger)")
    print(f"  3.25M steps : {per_step * 3.25e6 / 1e9:.2f} GB state vs "
          f"{frame_bytes * 3.25e6 / 1e9:.0f} GB pixels")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
