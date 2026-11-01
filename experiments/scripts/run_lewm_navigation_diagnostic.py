"""Prospectively frozen TwoRoom command-contract diagnostic on its declared rented pod."""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm
from stable_worldmodel.policy import AutoCostModel
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm16_planner import NoSafeAction
from helpers.lewm18_planner import PlanningCEM
from helpers.lewm_navigation_evolution import ActionContract, ProjectedCostModel
from helpers.lewm_proposal_pilot import write_json
from helpers.lewm_screen import image_stats
from helpers.portable_env import NativeEnv
from helpers.portable_motion import MotionChecker, clearance, load_model
from helpers.pusht_smoke import Scaler
from helpers.safecem10_planner import image_transform


class ZeroAdapter:
    def __init__(self):
        self.history = []

    def propose(self, features, current, target, checker, horizon):
        value = current.new_zeros(1, horizon, 10)
        self.history.append({"features": features, "proposal": value[0].cpu().tolist()})
        return value


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def trial(d, root, rt, job):
    folder = root / "episodes" / job["id"]
    folder.mkdir(parents=True, exist_ok=False)
    write_json(
        folder / "started.json",
        {**job, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()},
    )
    c = collections.Counter()
    env = NativeEnv("tworoom", c, folder / "interactions.jsonl")
    wm, head, normal, contract = rt
    cfg = d["configurations"][job["configuration"]]
    pair_source = d["sources"][job["pair"]]
    with np.load(ROOT / pair_source["path"], allow_pickle=False) as z:
        pair = dict(z)
    params = cfg["params"]
    checker = MotionChecker(head, [], d["margin"], c)
    cost = ProjectedCostModel(wm, checker, params, c, contract, project=cfg["project_candidates"])
    adapter = ZeroAdapter() if job.get("zero_adapter") else None
    solver = PlanningCEM(cost, "pusht", params, job["seed"], "cuda", proposal_adapter=adapter)
    policy = swm.policy.WorldModelPolicy(
        solver,
        swm.PlanConfig(horizon=5, receding_horizon=params["replan_blocks"], action_block=5),
        process={"action": normal},
        transform={"pixels": image_transform(), "goal": image_transform()},
    )
    policy.set_env(
        SimpleNamespace(
            num_envs=1,
            single_action_space=env.env.action_space,
            action_space=gym.spaces.Box(-1, 1, (1, 2), dtype=np.float64),
        )
    )
    states, actions, proposed, frames = [], [], [], []
    dense = []
    success, failure, error = False, None, None
    planning = 0.0
    start = time.perf_counter()
    try:
        torch.manual_seed(job["seed"])
        np.random.seed(job["seed"])
        observation = env.reset(job["seed"], pair["initial"], pair["goal"], pair["goal_image"])
        np.testing.assert_allclose(observation, pair["initial"], atol=1e-6, rtol=0)
        if env.success():
            raise RuntimeError("Already successful start")
        checker.observe(observation)
        states.append(observation.copy())
        dense.append(observation.copy())
        previous = np.zeros(2)
        for _ in range(job["cap"]):
            frame = env.render()
            image_stats(frame)
            frames.append(frame)
            before = time.perf_counter()
            try:
                command = policy.get_action(
                    {
                        "pixels": frame[None, None],
                        "goal": pair["goal_image"][None, None],
                        "action": previous[None, None],
                    }
                )[0]
            finally:
                torch.cuda.synchronize()
                planning += time.perf_counter() - before
            executed = contract.execute(command)
            observation, path = env.step(executed)
            checker.observe(observation)
            states.append(observation.copy())
            dense.extend(path)
            proposed.append(np.asarray(command).copy())
            actions.append(executed.copy())
            previous = executed if cfg["project_candidates"] else np.asarray(command)
            success = env.success()
            write_json(folder / "live.json", {"actions": len(actions), "resources": dict(c)})
            if success:
                break
            if time.perf_counter() - start > 300:
                raise TimeoutError("Episode time cap")
    except NoSafeAction as exc:
        failure = str(exc)
    except Exception:
        error = traceback.format_exc()
    finally:
        try:
            frames.append(env.render())
        finally:
            env.close()
    elapsed = time.perf_counter() - start
    positions = np.asarray(dense)
    actual_clear = (
        clearance(torch.as_tensor(positions), [], d["body_radius"]).numpy()
        if len(dense)
        else np.array([])
    )
    violation = bool((actual_clear < d["margin"] - 1e-6).any())
    pathlength = (
        float(np.linalg.norm(np.diff(np.asarray(states), axis=0), axis=1).sum()) if states else 0.0
    )
    flags = [i for i, frame in enumerate(frames) if image_stats(frame)["flagged"]]
    row = {
        **job,
        "environment": "tworoom",
        "condition": "none",
        "success": bool(success),
        "safe_completion": bool(success and not violation and not failure and not error),
        "margin_breach": violation,
        "arena_margin_breach": False,
        "unfinished": not bool(success),
        "engaged": len(actions) > 0 and pathlength > 1.0,
        "protected_body_path_length": pathlength,
        "actions": len(actions),
        "resources": dict(c),
        "planning_seconds": planning,
        "episode_seconds": elapsed,
        "error": error,
        "controller_failure": failure,
        "frame_flags": flags,
        "original_goal": pair["goal"].tolist(),
        "final_goal_error": float(np.linalg.norm(np.asarray(states[-1]) - pair["goal"]))
        if states
        else None,
        "action_contract": contract.record(),
        "candidate_projection": cfg["project_candidates"],
        "raw_native_action_coordinates_outside_bounds": int(
            sum(np.any((a < -1) | (a > 1)) for a in proposed)
        ),
    }
    np.savez_compressed(
        folder / "trace.npz",
        states=states,
        dense=dense,
        actions=actions,
        proposed=proposed,
        frames=frames,
        goal_image=pair["goal_image"],
    )
    write_json(folder / "replans.json", solver.replans)
    write_json(folder / "cost-history.json", cost.history)
    write_json(folder / "projection.json", cost.projection_history)
    if adapter is not None:
        write_json(folder / "zero-proposals.json", adapter.history)
    write_json(folder / "result.json", row)
    print(
        json.dumps(
            {k: row[k] for k in ("id", "safe_completion", "actions", "final_goal_error", "error")}
        ),
        flush=True,
    )
    if error:
        raise RuntimeError("Recorded technical failure: " + job["id"])
    if c.get("predictor_steps", 0) > d["maximum_predictor_steps_per_episode"]:
        raise RuntimeError("Frozen query ceiling exceeded")
    return row


def equivalence(root, state):
    jobs = [j for j in state["jobs"].values() if j["category"] == "engineering"]
    assert len(jobs) == 2
    traces = []
    for row in jobs:
        with np.load(root / "episodes" / row["id"] / "trace.npz", allow_pickle=False) as z:
            traces.append({k: z[k] for k in ("actions", "proposed", "states", "dense", "frames")})
    for key in traces[0]:
        np.testing.assert_array_equal(traces[0][key], traces[1][key])
    for name in (
        "predictor_steps",
        "model_candidate_rollouts",
        "model_batch_calls",
        "encoder_frames",
    ):
        assert jobs[0]["resources"][name] == jobs[1]["resources"][name]
    write_json(
        root / "engineering-equivalence.json",
        {"verified": True, "episodes": 2, "new_simulator_interactions": 0},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    d = json.loads(args.protocol.read_text())
    if os.environ.get("RUNPOD_POD_ID") != d["execution_pod_id"]:
        raise RuntimeError("Only the declared rented pod may run this diagnostic")
    torch.set_num_threads(4)
    if args.out.exists():
        raise FileExistsError("Never replay or overwrite an existing diagnostic")
    args.out.mkdir(parents=True)
    dependencies = d["source_files"]
    source = {name: sha(ROOT / name) for name in dependencies}
    assets = {name: sha(ROOT / name) for name in d["asset_sha256"]}
    if assets != d["asset_sha256"]:
        raise RuntimeError("Frozen asset differs")
    write_json(
        args.out / "source-freeze.json",
        {
            "protocol_sha256": sha(args.protocol),
            "sources": source,
            "assets": assets,
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "pod_id": d["execution_pod_id"],
        },
    )
    state = {"phase": "running", "jobs": {}, "active": None}
    write_json(args.out / "state.json", state)
    checkpoint = ROOT / d["world_model"]
    wm = (
        AutoCostModel(str(checkpoint)[: -len("_object.ckpt")], cache_dir=checkpoint.parent)
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    head = load_model(ROOT / d["motion_model"])
    if not head.calibration["ready"]:
        raise RuntimeError("Frozen safety head is not released")
    action = json.loads((ROOT / d["scalers"]).read_text())["columns"]["action"]
    normal = Scaler(action["mean"], action["scale"])
    contract = ActionContract(
        tuple(action["mean"]), tuple(action["scale"]), (-1.0, -1.0), (1.0, 1.0)
    )
    for job in d["jobs"]:
        if dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=15) >= dt.datetime.fromisoformat(
            d["deadline_utc"]
        ):
            raise RuntimeError("Preservation reserve reached")
        if (
            len(list((args.out / "episodes").glob("*/started.json")))
            >= d["maximum_all_episode_starts"]
        ):
            raise RuntimeError("Frozen start ceiling reached")
        state["active"] = job["id"]
        write_json(args.out / "state.json", state)
        row = trial(d, args.out, (wm, head, normal, contract), job)
        state["jobs"][job["id"]] = row
        state["active"] = None
        write_json(args.out / "state.json", state)
        if len(state["jobs"]) == 2:
            equivalence(args.out, state)
    state["phase"] = "complete"
    state["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(args.out / "state.json", state)


if __name__ == "__main__":
    main()
