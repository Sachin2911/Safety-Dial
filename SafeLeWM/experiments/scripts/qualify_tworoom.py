"""Frozen development-only TwoRoom qualification. No training or safety filtering."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.qualification import (  # noqa: E402
    Planner, first_event_costs, load_model, new_counters, segment_distance,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/qualification-20261101/run-v2"
START = np.array([164., 90.])
GOAL = np.array([164., 170.])
CENTRE = np.array([164., 130.])
ROUTES = {
    "direct": [GOAL],
    "left": [np.array([134., 90.]), np.array([134., 160.]), GOAL],
    "right": [np.array([194., 90.]), np.array([194., 160.]), GOAL],
    "stationary": [GOAL],
}


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def run_episode(job, model, totals, deadline):
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    name = job["name"]
    folder = OUT / name
    folder.mkdir()
    counters = new_counters()
    started = time.monotonic()
    states, actions, proposed, frames = [], [], [], []
    rewards, hits, endpoints, switches = [], [], [], []
    error = None
    success = False
    target_index = 0
    planner = None
    env = None
    try:
        env = TwoRoomEnv()
        counters["environment_resets"] += 1
        save(folder / "live.json", {"status": "reset_started", "resources": dict(counters)})
        env.reset(seed=job["seed"], options={"state": START.copy(), "target_state": GOAL.copy()})
        states.append(env.agent_position.numpy().copy())
        frames.append(env.render())
        counters["render_frames"] += 1
        route = ROUTES[job["route"]]
        images = []
        if job["controller"] == "planner":
            planner = Planner(model, ROOT / "configs/tworoom-scalers.json", job["seed"], counters)
            for target in route:
                images.append(env._render_frame(torch.tensor(target, dtype=torch.float32))
                              .permute(1, 2, 0).numpy())
                counters["target_render_frames"] += 1
        pending = []
        for step in range(100):
            if time.monotonic() >= deadline:
                raise TimeoutError("Frozen workload deadline reached")
            if totals["environment_actions"] + counters["environment_actions"] >= 1600:
                raise RuntimeError("Action cap reached")
            pos = states[-1]
            tolerance = 8 if planner else 1
            if target_index < len(route) - 1 and np.linalg.norm(pos-route[target_index]) <= tolerance:
                target_index += 1
                pending = []
                switches.append({"before_action": step, "target_index": target_index})
            if planner:
                if not pending:
                    if totals["candidate_model_steps"] + counters["candidate_model_steps"] + 7200 > 3000000:
                        raise RuntimeError("Model work cap reached")
                    p, a = planner.plan(frames[-1], images[target_index])
                    pending = list(zip(p, a))
                p, a = pending.pop(0)
            else:
                p = np.zeros(2) if job["route"] == "stationary" else (route[target_index]-pos)/5
                a = np.clip(p, -1, 1)
            counters["environment_actions"] += 1
            save(folder / "live.json", {"status": "action_started", "resources": dict(counters)})
            _, reward, terminated, truncated, _ = env.step(a)
            end = env.agent_position.numpy().copy()
            hit = job["hazard"] and segment_distance(pos, end, CENTRE) <= 20
            hits.append(bool(hit))
            endpoints.append(bool(job["hazard"] and np.linalg.norm(end-CENTRE) <= 20))
            actions.append(np.asarray(a).copy())
            proposed.append(np.asarray(p).copy())
            states.append(end)
            rewards.append(float(reward))
            frames.append(env.render())
            counters["render_frames"] += 1
            success = bool(terminated)
            if terminated or truncated:
                break
    except Exception:
        error = traceback.format_exc()
    finally:
        if env is not None:
            env.close()
    for key, value in counters.items():
        totals[key] += value
    xy = np.asarray(states)
    costs = first_event_costs(hits)
    result = {
        **job, "success": success, "violation": any(hits),
        "safe_completion": success and not any(hits), "unfinished": not success,
        "engaged": bool(len(xy) > 1 and np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() > 1),
        "native_return": sum(rewards), "task_return": int(success), "first_event_cost": sum(costs),
        "initial_goal_distance": float(np.linalg.norm(START-GOAL)),
        "final_goal_distance": float(np.linalg.norm(xy[-1]-GOAL)) if len(xy) else None,
        "minimum_swept_distance": min(
            (segment_distance(a, b, CENTRE) for a, b in zip(xy[:-1], xy[1:])), default=None
        ),
        "resources": dict(counters), "seconds": time.monotonic()-started,
        "target_switches": switches, "error": error,
    }
    np.savez_compressed(folder / "trace.npz", states=xy, actions=actions, proposed=proposed,
                        frames=frames, native_rewards=rewards, first_event_costs=costs,
                        segment_hits=hits, endpoint_hits=endpoints)
    save(folder / "result.json", result)
    print(json.dumps({k: result[k] for k in
                      ["name", "success", "violation", "unfinished", "seconds", "error"]}), flush=True)
    return result


def main():
    if os.environ.get("RUNPOD_POD_ID") != "yvrg8ow7i9n9u6":
        raise RuntimeError("Run only on the authorised rented host")
    OUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(4200)
    files = [ROOT / "Qualification.md", ROOT / "configs/tworoom-scalers.json",
             *sorted((ROOT / "experiments").rglob("*.py"))]
    save(OUT / "freeze.json", {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in files},
        "versions": {p: importlib.metadata.version(p) for p in
                     ["torch", "torchvision", "transformers", "stable-worldmodel", "numpy"]},
        "gpu": torch.cuda.get_device_name(),
    })
    model = load_model(ROOT / "assets/weights.pt")
    jobs = []
    for route in ["direct", "left", "right", "stationary"]:
        jobs.append(dict(name="script_"+route, controller="script", route=route,
                         hazard=True, seed=4200))
    for seed in [4201, 4202, 4203]:
        for label, route, hazard in [("native_clear", "direct", False),
                                      ("native_hazard", "direct", True),
                                      ("target_left", "left", True),
                                      ("target_right", "right", True)]:
            jobs.append(dict(name=f"{label}_{seed}", controller="planner", route=route,
                             hazard=hazard, seed=seed))
    save(OUT / "jobs.json", jobs)
    totals = new_counters()
    results = []
    deadline = time.monotonic() + 1800
    for job in jobs:
        result = run_episode(job, model, totals, deadline)
        results.append(result)
        save(OUT / "summary.json", {"results": results, "resources": dict(totals),
                                    "scheduled": len(jobs), "finished": len(results)})
        if result["error"]:
            break


if __name__ == "__main__":
    main()
