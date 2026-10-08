"""Reference side of the environment equivalence check. Runs under Python 3.10.

This is the ONLY place real `safety-gymnasium` is used, and it is used as a tool rather than a
dependency: it emits reference trajectories to a `.npz`, and nothing else in the project imports
it. Run it through an ephemeral uv environment so the project venv and `uv.lock` are untouched:

    uv run --isolated --no-project --python 3.10 \
      --with "safety-gymnasium==1.0.0" --with "numpy<2" \
      python experiments/scripts/emit_reference_traj.py

Imports numpy and safety_gymnasium only. `helpers/equivCheck.py` is deliberately torch-free so
that the identical file loads here and in the project venv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers import equivCheck as ec  # noqa: E402


class UpstreamAdapter:
    """Wrap the real safety-gymnasium env in the interface `equivCheck.run_trial` expects.

    Built from the raw env class rather than through `safety_gymnasium.make`, so that no
    `SafeTimeLimit` wrapper truncates the tape. Divergence has to be measured over the whole
    1000 steps, and termination compared as an index.
    """

    def __init__(self, robot: str, version: str):
        from safety_gymnasium.tasks.safe_velocity import (
            safety_hopper_velocity_v0, safety_hopper_velocity_v1,
            safety_walker2d_velocity_v0, safety_walker2d_velocity_v1,
        )

        table = {
            ("Walker2d", "v0"): safety_walker2d_velocity_v0.SafetyWalker2dVelocityEnv,
            ("Walker2d", "v1"): safety_walker2d_velocity_v1.SafetyWalker2dVelocityEnv,
            ("Hopper", "v0"): safety_hopper_velocity_v0.SafetyHopperVelocityEnv,
            ("Hopper", "v1"): safety_hopper_velocity_v1.SafetyHopperVelocityEnv,
        }
        self.env = table[(robot, version)]()
        self.u = self.env
        self.reset_qpos = None
        self.reset_qvel = None

    def reset(self, seed=0):
        self.env.reset(seed=int(seed))
        self.reset_qpos = self.u.data.qpos.copy()
        self.reset_qvel = self.u.data.qvel.copy()

    def set_state(self, qpos, qvel):
        self.u.set_state(np.asarray(qpos, float), np.asarray(qvel, float))

    def step(self, action):
        obs, reward, cost, terminated, _, info = self.env.step(action)
        return (obs, float(reward), float(cost), bool(terminated),
                float(info["x_velocity"]), self.u.data.qpos.copy(), self.u.data.qvel.copy())

    def manifest(self):
        import gymnasium
        import mujoco

        m, opt = self.u.model, self.u.model.opt
        import hashlib
        import json
        import os

        path = getattr(self.u, "fullpath", None)
        xml_md5 = (hashlib.md5(open(path, "rb").read()).hexdigest()  # noqa: S324
                   if path and os.path.isfile(path) else None)
        scalars = {
            "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu), "na": int(m.na),
            "frame_skip": int(self.u.frame_skip), "dt": float(self.u.dt),
            "opt_timestep": float(opt.timestep), "opt_solver": int(opt.solver),
            "opt_iterations": int(opt.iterations), "opt_integrator": int(opt.integrator),
            "opt_tolerance": float(opt.tolerance),
            "body_mass": np.asarray(m.body_mass).tolist(),
            "dof_damping": np.asarray(m.dof_damping).tolist(),
            "jnt_range": np.asarray(m.jnt_range).tolist(),
            "actuator_gear": np.asarray(m.actuator_gear).tolist(),
        }
        return {
            "stack": "safety-gymnasium upstream",
            "velocity_threshold": float(self.u._velocity_threshold),
            "healthy_z_range": list(getattr(self.u, "_healthy_z_range", ())),
            "healthy_angle_range": list(getattr(self.u, "_healthy_angle_range", ())),
            "terminate_when_unhealthy": bool(
                getattr(self.u, "_terminate_when_unhealthy", False)
            ),
            "xml_md5": xml_md5,
            "model_scalars_json": json.dumps(scalars, sort_keys=True),
            "mujoco_version": mujoco.__version__,
            "gymnasium_version": gymnasium.__version__,
            "numpy_version": np.__version__,
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="Walker2d")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--horizon", type=int, default=ec.HORIZON)
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "equiv"))
    args = ap.parse_args()

    import safety_gymnasium  # noqa: F401  (import check, and it is the point of this script)

    adapter = UpstreamAdapter(args.robot, args.version)
    adapter.reset(seed=0)
    nq, nv = adapter.u.model.nq, adapter.u.model.nv
    a_dim = int(adapter.env.action_space.shape[0])

    specs = ec.build_specs(nq, nv, a_dim, args.horizon)
    print(f"[reference] {args.robot}-{args.version}: {len(specs)} trials x {args.horizon} steps")

    blobs = {}
    for i, spec in enumerate(specs):
        r = ec.run_trial(adapter, spec, args.horizon)
        for key in ("qpos", "qvel", "obs", "reward", "cost", "terminated", "x_velocity"):
            blobs[f"t{i}_{key}"] = r[key]
        blobs[f"t{i}_reset_qpos"] = r["reset_qpos"]
        blobs[f"t{i}_reset_qvel"] = r["reset_qvel"]
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(specs)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"ref_{args.robot}_{args.version}.npz"
    np.savez_compressed(out, n_trials=len(specs), horizon=args.horizon,
                        manifest=np.array(str(adapter.manifest())), **blobs)
    print(f"[reference] wrote {out}")
    import json
    (out_dir / f"ref_{args.robot}_{args.version}_manifest.json").write_text(
        json.dumps(adapter.manifest(), indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
