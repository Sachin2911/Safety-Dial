#!/usr/bin/env python3
"""Post-hoc diagnostic for the failed Gate 0 of the evolution study (not part of the gate).

Gate 0 asks whether the behaviour-cloned linear latent policy, which holds one action for the
10 steps of a model block (12.5 Hz), walks. It failed. This asks whether the action interface
alone explains it: from the same 256 evaluation states, replay open loop (no policy, no
cloning error) the recorded policies' own actions three ways: exactly (125 Hz), each block's
mean action held for the block, and each block's first action held (sample and hold).

    uv run python experiments/scripts/evo_gate0_hold_diagnostic.py --s2 RUN_ID
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np  # noqa: E402

from helpers.evoInputs import RESULTS_ROOT, fetch_inputs, load_stage_config  # noqa: E402
from helpers.evoRoots import evaluation_roots, rootset_digest  # noqa: E402
from helpers.evoStats import wilson  # noqa: E402
from helpers.locoEnv import make_loco_env  # noqa: E402
from helpers.walkerRules import FRAMESKIP, HORIZON_BLOCKS, execute_branch  # noqa: E402

VARIANTS = {
    "exact_125hz": lambda t: t,
    "block_mean_hold_12p5hz": lambda t: np.repeat(t.reshape(HORIZON_BLOCKS, FRAMESKIP, 6).mean(1), FRAMESKIP, 0),
    "first_action_hold_12p5hz": lambda t: np.repeat(t.reshape(HORIZON_BLOCKS, FRAMESKIP, 6)[:, 0], FRAMESKIP, 0),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--s2", required=True, help="stage 2 run id whose results folder receives the diagnostic")
    args = ap.parse_args()
    cfg = load_stage_config("stage2")
    paths = fetch_inputs(cfg.inputs, ["data", "banks"])
    ev = evaluation_roots(paths["banks"], paths["data"] / "roots.h5", seed=cfg.roots.evaluation_seed,
                          extra_per_episode=cfg.roots.extra_per_episode)
    env = make_loco_env("Walker2d", "v1", render=False, terminate_when_unhealthy=False, six_tuple=True)
    env.reset(seed=0)
    out = {"note": "post-hoc diagnostic of the failed Gate 0; not part of the gate", "evaluation_roots_digest": rootset_digest(ev)}
    for name, f in VARIANTS.items():
        v, p = [], []
        for r in ev.roots:
            log = execute_branch(env, r.qpos, r.qvel, np.clip(f(np.asarray(r.policy_tape, float)), -1, 1))
            v.append(log.unsafe()["health"])
            p.append(float(log.qpos[-1, 0] - log.qpos[0, 0]))
        k = int(np.sum(v))
        out[name] = {"violations": k, "n": len(v), "rate": k / len(v), "rate_wilson": wilson(k, len(v)), "progress_mean": float(np.mean(p))}
        print(name, out[name], flush=True)
    env.close()
    dest = RESULTS_ROOT / "stage2" / args.s2 / "diagnostic_hold.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
