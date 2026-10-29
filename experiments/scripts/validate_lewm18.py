"""Pre-data engineering checks, with a frozen source record and counted interactions."""

import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
from helpers.dsrl06_infra import atomic_json, utc


def main():
    out = ROOT / "outputs/lewm18"
    index = 1
    while (out / f"validation-{index}-freeze.json").exists():
        index += 1
    files = [Path(__file__), ROOT / "experiments/helpers/lewm18_runtime.py"]
    atomic_json(
        out / f"validation-{index}-freeze.json",
        {
            "utc": utc(),
            "seed": 180018,
            "cube_actions": 10,
            "cube_preview_actions": 5,
            "pusht_actions": 3,
            "source_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
            },
            "criteria": "Cube integration/observation parity <=1e-10, pixels equal; "
            "preview restoration and subsequent parity <=1e-10; PushT PD dense error <=1e-6 px",
        },
    )
    counters = collections.Counter()
    report = {"utc": utc(), "pass": False}
    try:
        import mujoco
        import numpy as np
        from helpers.lewm18_runtime import DenseCube, GoalCubeEnv
        from helpers.safecem10_planner import pd_response
        from stable_worldmodel.envs.pusht.env import PushT

        rng = np.random.default_rng(180018)
        options = dict(
            env_type="single", ob_type="states", pixel_transparent_arm=False, width=224, height=224
        )
        reference, dense = GoalCubeEnv(**options), DenseCube(**options)
        for env in (reference, dense):
            np.random.seed(180018)
            env.np_random = np.random.default_rng(180018)
            counters["resets"] += 1
            env.reset(seed=180018)
        state_type = mujoco.mjtState.mjSTATE_INTEGRATION

        def state(env):
            vector = np.empty(mujoco.mj_stateSize(env._model, state_type))
            mujoco.mj_getState(env._model, env._data, vector, state_type)
            return vector

        errors = []
        for i, action in enumerate(rng.uniform(-0.2, 0.2, (10, 5))):
            if i == 5:
                before = state(dense)
                image_before = dense.render()
                counters["renders"] += 1
                preview = dense.preview(rng.uniform(-0.2, 0.2, (5, 5)))
                counters["counterfactual_physics_steps"] += len(preview)
                assert np.array_equal(state(dense), before)
                counters["renders"] += 1
                assert np.array_equal(dense.render(), image_before)
            observations = []
            for env in (reference, dense):
                counters["environment_actions"] += 1
                observations.append(env.step(action)[0])
                counters["executed_physics_steps"] += env._n_steps
            error = max(
                float(np.max(np.abs(state(reference) - state(dense)))),
                float(np.max(np.abs(observations[0] - observations[1]))),
            )
            errors.append(error)
            assert error <= 1e-10, f"Cube parity {i}: {error}"
        counters["renders"] += 2
        assert np.array_equal(reference.render(), dense.render())
        report["cube_max_error"] = max(errors)
        report["cube_render_equal"] = True
        reference.close()
        dense.close()

        push = PushT()
        counters["resets"] += 1
        observation, _ = push.reset(seed=180018)
        actual = []
        original = push.space.step

        def capture(dt):
            original(dt)
            actual.append(np.asarray(push.agent.position).copy())
            counters["executed_physics_steps"] += 1

        push.space.step = capture
        initial = observation["state"]
        targets = rng.uniform(100, 400, (3, 2))
        for action in targets:
            counters["environment_actions"] += 1
            push.step(action)
        c, gain = pd_response(3)
        expected = initial[:2] + c[:, None] * initial[-2:] + gain @ targets
        report["pusht_dense_error_px"] = float(np.max(np.abs(expected - actual)))
        counters["analytic_path_steps"] += len(expected)
        assert report["pusht_dense_error_px"] <= 1e-6
        push.close()
        report["pass"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        report["resources"] = dict(counters)
        report["finished_utc"] = utc()
        atomic_json(out / f"validation-{index}-result.json", report)
        print(json.dumps(report), flush=True)
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
