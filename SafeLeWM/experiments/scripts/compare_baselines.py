"""Frozen no-hazard comparison; all computation belongs on the authorised pod."""

import collections
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

import h5py
import hdf5plugin  # noqa: F401 -- registers the dataset's image compression filters
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.baseline_planner import make_planner, models, pixels  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/baseline-20261101/run-v2"


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare():
    OUT.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((OUT.parent / "data-receipt.json").read_text())
    assert (
        receipt["h5_sha256"] == "129a36aa93ea0de488d2bcc876e396de9e3907bf66c6aae6394e542ef6a6d623"
    )
    panel = []
    inputs = OUT / "inputs"
    inputs.mkdir()
    with h5py.File(ROOT / "data/tworoom.h5", "r") as data:
        lengths = np.asarray(data["ep_len"][:], dtype=int)
        offsets = np.asarray(data["ep_offset"][:], dtype=int)
        valid = np.concatenate(
            [
                np.arange(offset, offset + length - 100, dtype=int)
                for offset, length in zip(offsets, lengths)
                if length > 100
            ]
        )
        selected = np.sort(
            valid[np.random.default_rng(42).choice(len(valid) - 1, 50, replace=False)]
        )
        action = np.asarray(data["action"][:], dtype=np.float64)
        action = action[~np.isnan(action).any(axis=1)]
        stats = dict(mean=action.mean(0).tolist(), scale=action.std(0).tolist(), rows=len(action))
        expected = json.loads((ROOT / "configs/tworoom-scalers.json").read_text())["columns"][
            "action"
        ]
        for key in ["mean", "scale"]:
            np.testing.assert_allclose(stats[key], expected[key], rtol=1e-7, atol=1e-8)
        save(OUT / "scaler-audit.json", {"computed": stats, "saved": expected, "passed": True})
        for i, row in enumerate(selected):
            episode = int(np.searchsorted(offsets, row, side="right") - 1)
            path = inputs / f"dataset-{i:02d}.npz"
            np.savez_compressed(
                path,
                initial=data["pos_agent"][row],
                goal=data["pos_agent"][row + 100],
                start_image=data["pixels"][row],
                goal_image=data["pixels"][row + 100],
            )
            panel.append(
                dict(
                    id=path.stem,
                    panel="dataset",
                    file=str(path.relative_to(ROOT)),
                    sha256=digest(path),
                    source_episode=episode,
                    source_row=int(row),
                    goal_row=int(row + 100),
                    seed=42,
                    profiles=["paper"],
                )
            )
    for i in range(6, 12):
        paths = list((ROOT / "assets/baseline-inputs").glob(f"case-{i:02d}-*.npz"))
        assert len(paths) == 1
        path = paths[0]
        panel.append(
            dict(
                id=f"historical-{i:02d}",
                panel="historical",
                file=str(path.relative_to(ROOT)),
                sha256=digest(path),
                seed=303000 + i,
                profiles=["paper", "random05", "qualification"],
            )
        )
    for seed in [4201, 4202, 4203]:
        panel.append(
            dict(
                id=f"constructed-{seed}",
                panel="constructed",
                seed=seed,
                initial=[164.0, 90.0],
                goal=[164.0, 170.0],
                profiles=["paper", "random05", "qualification"],
            )
        )
    save(OUT / "panel.json", panel)
    return panel


@torch.inference_mode()
def preflight(pair):
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    current, original = pair
    z = np.load(next((ROOT / "assets/baseline-inputs").glob("case-06-*.npz")))
    image, goal = pixels(z["start_image"]), pixels(z["goal_image"])
    rng = torch.Generator(device="cuda").manual_seed(5100)
    report = {
        "model_parity": [],
        "environment_resets": 0,
        "environment_actions": 0,
        "candidate_model_steps": 0,
        "passed": False,
    }
    save(OUT / "preflight.json", report)
    for horizon in [2, 3, 5]:
        actions = torch.randn(1, 8, horizon, 10, device="cuda", generator=rng)
        costs = []
        for model in pair:
            info = {
                "pixels": image[:, None].expand(1, 8, -1, -1, -1, -1),
                "goal": goal[:, None].expand(1, 8, -1, -1, -1, -1),
                "action": torch.zeros(1, 8, 1, 10, device="cuda"),
            }
            report["candidate_model_steps"] += 8 * horizon
            costs.append(model.get_cost(info, actions).cpu().numpy())
        np.testing.assert_allclose(costs[0], costs[1], rtol=2e-5, atol=1e-5)
        report["model_parity"].append(
            dict(horizon=horizon, maximum_cost_difference=float(np.max(abs(costs[0] - costs[1]))))
        )
        save(OUT / "preflight.json", report)
    env = TwoRoomEnv()
    try:
        report["environment_resets"] += 1
        save(OUT / "preflight.json", report)
        env.reset(
            seed=5100,
            options={"state": np.array([164.0, 90.0]), "target_state": np.array([164.0, 170.0])},
        )
        report["environment_actions"] += 1
        save(OUT / "preflight.json", report)
        env.step(np.array([0.0, 2.0]))
        np.testing.assert_array_equal(env.agent_position.numpy(), [164.0, 95.0])
        report["passed"] = True
    finally:
        env.close()
        save(OUT / "preflight.json", report)
    return report


def episode(case, profile, pair, totals, deadline):
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    name = case["id"] + "_" + profile
    folder = OUT / name
    folder.mkdir()
    count = collections.Counter()
    frames, states, proposed, executed, rewards = [], [], [], [], []
    error, success, success_step = None, False, None
    started = time.monotonic()
    hooks = []
    env = None
    goal = np.array(case.get("goal", [0, 0]), dtype=float)
    start_mae = None
    try:
        if "file" in case:
            path = ROOT / case["file"]
            assert digest(path) == case["sha256"]
            with np.load(path) as z:
                initial, goal = z["initial"].copy(), z["goal"].copy()
                target_image, source_image = z["goal_image"].copy(), z["start_image"].copy()
        else:
            initial = np.asarray(case["initial"])
            target_image = source_image = None
        env = TwoRoomEnv()
        count["environment_resets"] += 1
        save(folder / "live.json", dict(status="reset_started", resources=dict(count)))
        env.reset(seed=case["seed"], options={"state": initial, "target_state": goal})
        np.testing.assert_allclose(env.agent_position.numpy(), initial, atol=1e-5, rtol=0)
        frame = env.render()
        count["render_frames"] += 1
        if source_image is not None:
            start_mae = float(np.abs(frame.astype(float) - source_image.astype(float)).mean())
            if start_mae > 2:
                raise ValueError(f"Start image mismatch: MAE {start_mae}")
        if target_image is None:
            target_image = env._target_img.permute(1, 2, 0).numpy().copy()
        states.append(env.agent_position.numpy().copy())
        frames.append(frame)
        planner = make_planner(ROOT, pair, profile, case["seed"], count)

        def encoder_hook(_module, args):
            count["actual_encoder_frames"] += int(args[0].shape[0])

        # Encoder is shared; attach one hook, not one per model.
        hooks.append(pair[0].encoder.register_forward_pre_hook(encoder_hook))
        pending = []
        for step in range(150):
            if time.monotonic() >= deadline:
                raise TimeoutError("Worker deadline")
            if totals["environment_actions"] + count["environment_actions"] >= 11550:
                raise RuntimeError("Episode action cap")
            if not pending:
                if (
                    totals["candidate_model_steps"] + count["candidate_model_steps"] + 15000
                    > 30000000
                ):
                    raise RuntimeError("Model-work cap")
                p, a = planner.plan(frames[-1], target_image)
                pending = list(zip(p, a))
            p, a = pending.pop(0)
            count["environment_actions"] += 1
            count["clipped_action_coordinates"] += int((np.abs(p - a) > 1e-7).sum())
            save(folder / "live.json", dict(status="action_started", resources=dict(count)))
            _, reward, terminated, truncated, _ = env.step(a)
            proposed.append(np.array(p))
            executed.append(np.array(a))
            rewards.append(float(reward))
            states.append(env.agent_position.numpy().copy())
            frames.append(env.render())
            count["render_frames"] += 1
            success = bool(terminated)
            if terminated or truncated:
                success_step = step + 1 if success else None
                break
    except Exception:
        error = traceback.format_exc()
    finally:
        for hook in hooks:
            hook.remove()
        if env is not None:
            env.close()
    xy = np.asarray(states)
    result = dict(
        name=name,
        case=case["id"],
        panel=case["panel"],
        profile=profile,
        success=success,
        success_step=success_step,
        success_by_100=bool(success and success_step <= 100),
        unfinished=not success,
        engaged=bool(len(xy) > 1 and np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() > 1),
        initial_goal_distance=float(np.linalg.norm(xy[0] - goal)) if len(xy) else None,
        final_goal_distance=float(np.linalg.norm(xy[-1] - goal)) if len(xy) else None,
        native_return=sum(rewards),
        start_image_mae=start_mae,
        seconds=time.monotonic() - started,
        resources=dict(count),
        error=error,
    )
    np.savez_compressed(
        folder / "trace.npz",
        states=xy,
        frames=frames,
        proposed=proposed,
        actions=executed,
        native_rewards=rewards,
        goal=goal,
    )
    save(folder / "result.json", result)
    totals.update(count)
    print(
        json.dumps({k: result[k] for k in ["name", "success", "success_step", "seconds", "error"]}),
        flush=True,
    )
    return result


def main():
    assert os.environ.get("RUNPOD_POD_ID") == "yvrg8ow7i9n9u6"
    from loguru import logger
    from stable_worldmodel.solver import CEMSolver
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    logger.disable("stable_worldmodel")
    torch.set_num_threads(4)
    torch.manual_seed(5100)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    panel = prepare()
    sources = [
        ROOT / "BaselineComparison.md",
        ROOT / "configs/tworoom-scalers.json",
        *sorted((ROOT / "experiments").rglob("*.py")),
        *sorted((ROOT / "assets/baseline-reference").rglob("*.py")),
        ROOT / "assets/baseline-reference/manifest.json",
        ROOT / "assets/baseline-reference/model-config.json",
    ]
    save(
        OUT / "freeze.json",
        dict(
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
            panel_sha256=digest(OUT / "panel.json"),
            upstream_sources={
                inspect.getfile(c): digest(inspect.getfile(c)) for c in [CEMSolver, TwoRoomEnv]
            },
        ),
    )
    pair = models(ROOT)
    preflight(pair)
    totals = collections.Counter()
    results = []
    deadline = time.monotonic() + 3600
    # Matched diagnostic panel first, then the prospectively selected dataset panel.
    ordered = [c for c in panel if c["panel"] != "dataset"] + [
        c for c in panel if c["panel"] == "dataset"
    ]
    for case in ordered:
        for profile in case["profiles"]:
            row = episode(case, profile, pair, totals, deadline)
            results.append(row)
            save(
                OUT / "summary.json",
                dict(results=results, resources=dict(totals), scheduled=77, finished=len(results)),
            )
            if row["error"]:
                return


if __name__ == "__main__":
    main()
