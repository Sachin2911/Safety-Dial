"""Counted synthetic native-CEM parity checks while the benchmark data downloads."""

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from helpers.dsrl06_infra import atomic_json, utc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=("cube", "pusht"))
    args = parser.parse_args()
    out = ROOT / "outputs/lewm18"
    path = out / "synthetic-model-check.json"
    if path.exists():
        raise FileExistsError(path)
    sources = [Path(__file__), ROOT / "experiments/helpers/lewm18_planner.py"]
    atomic_json(
        out / "synthetic-model-check-freeze.json",
        {
            "utc": utc(),
            "seed": 180019,
            "inputs": "Zeros current image, ones goal image",
            "criterion": "Complete 25-action normalized plan agrees with upstream CEM within 1e-5",
            "source_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            },
        },
    )
    counters = collections.Counter()
    report = {"utc": utc(), "pass": False, "environments": {}}
    try:
        import gymnasium as gym
        import stable_worldmodel as swm
        import torch
        from helpers.lewm18_planner import CostModel, PlanningCEM
        from stable_worldmodel.solver import CEMSolver

        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)

        class NoCheck:
            def check(self, candidates):
                b, n = candidates.shape[:2]
                zero = torch.zeros(b, n, device=candidates.device)
                return zero, torch.zeros(b, n, 1, 2, device=candidates.device), zero

        for kind, dim in [("cube", 5), ("pusht", 2)]:
            if args.env and kind != args.env:
                continue
            checkpoint = ROOT / f"data/lewm18/stablewm/checkpoints/{kind}/lewm"
            model = swm.policy.AutoCostModel(str(checkpoint)).cuda().eval()
            model.requires_grad_(False)
            params = {"checked": False, "objective": "learned", "horizon": 5, "iterations": 30}
            cost = CostModel(model, NoCheck(), params, counters)
            adapter = PlanningCEM(cost, kind, params, 180019, "cuda")
            reference = CEMSolver(
                CostModel(model, NoCheck(), params, counters),
                num_samples=300,
                n_steps=30,
                topk=30,
                seed=180019,
                device="cuda",
            )
            info = {
                "pixels": torch.zeros(1, 1, 3, 224, 224),
                "goal": torch.ones(1, 1, 3, 224, 224),
                "action": torch.zeros(1, 1, dim),
            }
            plans, seconds = [], []
            for solver in (adapter, reference):
                solver.configure(
                    action_space=gym.spaces.Box(-1.0, 1.0, (1, dim)),
                    n_envs=1,
                    config=swm.PlanConfig(horizon=5, receding_horizon=1, action_block=5),
                )
                start = time.perf_counter()
                plans.append(solver.solve(dict(info))["actions"])
                if solver is reference:
                    counters["encoder_frames"] += 2
                torch.cuda.synchronize()
                seconds.append(time.perf_counter() - start)
            error = float((plans[0] - plans[1]).abs().max())
            report["environments"][kind] = {
                "max_error": error,
                "solve_seconds": seconds,
                "frozen_parameters": sum(p.numel() for p in model.parameters()),
            }
            assert error <= 1e-5, (kind, error)
            del model, cost, adapter, reference
            torch.cuda.empty_cache()
        report["pass"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        report["resources"] = dict(counters)
        report["finished_utc"] = utc()
        atomic_json(path, report)
        print(json.dumps(report), flush=True)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
