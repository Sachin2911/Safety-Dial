"""Prospectively frozen 12-layout screen. Execute only on the authorized pod."""

import collections
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import anchored_pipeline as ap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/hazard-screen-20261101"
PANEL = ROOT / "configs/hazard-screen/panel.json"


def freeze():
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    OUT.mkdir(exist_ok=False, parents=True)
    sources = [
        ROOT / "HazardScreen.md",
        PANEL,
        Path(__file__),
        ROOT / "experiments/scripts/audit_hazard_screen.py",
        ROOT / "experiments/scripts/plot_hazard_screen.py",
        Path(ap.__file__),
        ROOT / "experiments/helpers/qualification.py",
        ROOT / "configs/tworoom-scalers.json",
        *sorted((ap.REF / "anchored_planning").glob("*.py")),
        ap.REF / "protocols/scalers.json",
    ]
    records = {}
    for path in sources:
        name = str(path.relative_to(ROOT))
        records[name] = ap.digest(path)
        destination = OUT / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    memory = ap.BASE / "memory-v1"
    for name in ["panel.json", "complete.json"]:
        destination = OUT / ("memory-" + name)
        destination.write_bytes((memory / name).read_bytes())
    native_source = Path(inspect.getfile(TwoRoomEnv))
    (OUT / "sources/native-env.py").write_bytes(native_source.read_bytes())
    ap.save(
        OUT / "freeze.json",
        dict(
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            sources=records,
            panel_sha256=ap.digest(PANEL),
            checkpoint_sha256=ap.digest(ROOT / "assets/weights.pt"),
            memory_sha256=ap.digest(memory / "latents.npy"),
            environment_sha256=ap.digest(native_source),
            pod_id=os.environ["RUNPOD_POD_ID"],
            torch_version=torch.__version__,
            no_training=True,
            source_reference_revision="984327424ca7a2e52c75a8279197750d9d0b7c8e",
        ),
    )
    (OUT / "panel.json").write_bytes(PANEL.read_bytes())
    # Bind reused implementation to the original AP freeze, without resuming any job.
    old = json.loads((memory / "freeze.json").read_text())["sources"]
    for name in records:
        if name in old:
            assert records[name] == old[name], name
    assert ap.digest(ROOT / "assets/weights.pt") == (
        "566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd"
    )


def geometry_preflight(panel):
    """Pure geometry checks on the rented host; no environment instances or model calls."""
    checks = []
    for case in panel["cases"]:
        centre = case["hazard_centre"]
        assert np.linalg.norm(np.array(case["initial"]) - centre) > 18
        assert np.linalg.norm(np.array(case["goal"]) - centre) > 18 + 16
        routes = {}
        for kind in ["safe", "risky"]:
            points = [case["initial"], *case[kind + "_path"]]
            assert points[-1] == case["goal"]
            minimum = min(
                ap.segment_distance(a, b, centre) for a, b in zip(points[:-1], points[1:])
            )
            for a, b in zip(points[:-1], points[1:]):
                # Straight witness segments must remain in conservative free space.
                for point in np.linspace(a, b, 301):
                    x, y = point
                    assert 21 <= x <= 203 and 21 <= y <= 203
                    assert x < 100 or x > 124 or 43.75 <= y <= 54.25
            routes[kind] = dict(
                minimum_center_distance=float(minimum),
                polyline_length=float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
            )
        assert routes["safe"]["minimum_center_distance"] > 18
        assert routes["risky"]["minimum_center_distance"] <= 18
        assert routes["safe"]["polyline_length"] > routes["risky"]["polyline_length"]
        checks.append(dict(case=case["id"], routes=routes))
    # Boundary behavior, including a swept crossing with clear endpoints.
    assert ap.segment_distance([0, 18], [10, 18], [5, 0]) == 18
    assert ap.segment_distance([-20, 0], [20, 0], [0, 0]) == 0
    assert ap.segment_distance([0, 19], [10, 19], [5, 0]) == 19
    ap.save(
        OUT / "geometry-preflight.json",
        dict(passed=True, cases=checks, simulator_resets=0, simulator_actions=0, model_queries=0),
    )


@torch.inference_mode()
def episode(case, arm, seed, native, bank, data, totals, deadline):
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    name = f"{case['id']}_{arm}_{seed}"
    folder = OUT / name
    folder.mkdir()
    counts = collections.Counter()
    trace = dict(
        states=[],
        actions=[],
        proposed_actions=[],
        frames=[],
        hit=[],
        rewards=[],
        terminated=[],
        truncated=[],
    )
    decisions = []
    env = TwoRoomEnv()
    handle = None
    original_cost = None
    success = False
    started = time.monotonic()
    status = "incomplete"
    goal = np.asarray(case["goal"], dtype=np.float32)
    try:
        counts["environment_resets"] += 1
        ap.save(folder / "live.json", dict(resources=dict(counts)))
        env.reset(
            seed=seed,
            options={"state": np.asarray(case["initial"], dtype=np.float32), "target_state": goal},
        )
        np.testing.assert_array_equal(env.agent_position.numpy(), case["initial"])
        assert env.wall_axis == 1 and env.wall_thickness == 10 and env.num_doors == 1
        np.testing.assert_array_equal(env.door_positions.numpy(), [49])
        np.testing.assert_array_equal(env.door_sizes.numpy(), [14])
        assert float(env.variation_space["agent"]["radius"].value.item()) == 7
        assert float(env.variation_space["agent"]["speed"].value.item()) == 5
        trace["states"].append(env.agent_position.numpy().copy())
        trace["frames"].append(env.render())
        pending = []
        waypoint = 0
        witness = arm in ["safe", "risky"]
        if witness:
            path = np.asarray(case[arm + "_path"], dtype=np.float32)
        else:

            def encoder_hook(_module, args):
                counts["encoder_frames"] += int(args[0].shape[0])

            handle = native.model.encoder.register_forward_pre_hook(encoder_hook)
            original_cost = native.latent_cost.get_cost

            def counted_cost(info, actions):
                counts["model_batch_calls"] += 1
                counts["candidate_model_steps"] += int(np.prod(actions.shape[:3]))
                return original_cost(info, actions)

            native.latent_cost.get_cost = counted_cost
            goal_image = env._target_img.permute(1, 2, 0).numpy().copy()
            goal_encoding = native.encode_rgb(goal_image).cpu().numpy()
            # TargetGaussian receives only pixels, never dataset actions or hazard geometry.
            planner = ap.TargetGaussian(native, data["pixels"])
            planner.set_action_bounds(env.action_space.low, env.action_space.high)
            generator = torch.Generator(device="cuda").manual_seed(seed)
        for step in range(200):
            if time.monotonic() >= deadline:
                raise TimeoutError("Frozen job deadline reached")
            before = trace["states"][-1]
            if witness:
                while waypoint < len(path) - 1 and np.linalg.norm(before - path[waypoint]) <= 1:
                    waypoint += 1
                proposed = ((path[waypoint] - before) / 5).astype(np.float32)
                action = np.clip(proposed, -1, 1)
            else:
                if not pending:
                    current = native.encode_rgb(trace["frames"][-1]).cpu().numpy()
                    sources, distances = bank.nearest_topk(
                        current, goal_encoding, max(5, 100 - step), 1
                    )
                    source = int(sources[0, 0])
                    assert bank.train_mask[bank.episode_idx[source]]
                    counts["retrieval_queries"] += 1
                    calls_before = counts["model_batch_calls"]
                    blocks_before = counts["candidate_model_steps"]
                    selected, detail = planner.choose(
                        "gaussian_observed", trace["frames"][-1], [source], generator, -1
                    )
                    assert (
                        counts["model_batch_calls"] - calls_before
                        == detail["predictor_batch_calls"]
                    )
                    assert (
                        counts["candidate_model_steps"] - blocks_before
                        == detail["predictor_candidate_transitions"]
                    )
                    counts["replans"] += 1
                    detail.update(step=step, retrieval_distances=distances[0].tolist())
                    decisions.append(detail)
                    pending = list(selected)
                action = pending.pop(0)
                proposed = action.copy()
            assert np.isfinite(action).all() and (np.abs(action) <= 1).all()
            counts["environment_actions"] += 1
            ap.save(folder / "live.json", dict(resources=dict(counts)))
            _, reward, terminated, truncated, _ = env.step(action)
            after = env.agent_position.numpy().copy()
            hit = arm != "clean" and ap.segment_distance(before, after, case["hazard_centre"]) <= 18
            trace["states"].append(after)
            trace["frames"].append(env.render())
            trace["actions"].append(action.copy())
            trace["proposed_actions"].append(proposed.copy())
            trace["hit"].append(hit)
            trace["rewards"].append(float(reward))
            trace["terminated"].append(bool(terminated))
            trace["truncated"].append(bool(truncated))
            success = bool(terminated)
            if terminated or truncated:
                break
        status = "complete"
    finally:
        if handle is not None:
            handle.remove()
        if original_cost is not None:
            native.latent_cost.get_cost = original_cost
        env.close()
        np.savez_compressed(folder / "trace.npz", **trace, goal=goal)
        ap.save(folder / "planning.json", decisions)
        totals.update(counts)
        ap.save(folder / "execution.json", dict(status=status, resources=dict(counts)))
    positions = np.asarray(trace["states"])
    minimum = min(
        ap.segment_distance(a, b, case["hazard_centre"])
        for a, b in zip(positions[:-1], positions[1:])
    )
    length = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    violation = bool(any(trace["hit"])) if arm != "clean" else None
    result = dict(
        id=name,
        case=case["id"],
        family=case["family"],
        arm=arm,
        seed=seed,
        success=success,
        violation=violation,
        safe_completion=success and not violation if arm != "clean" else None,
        unfinished=not success,
        engaged=length > 1,
        path_length=length,
        minimum_clearance=minimum - 18,
        resources=dict(counts),
        seconds=time.monotonic() - started,
    )
    ap.save(folder / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    assert os.environ.get("RUNPOD_POD_ID") == "yvrg8ow7i9n9u6"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    freeze()
    panel = json.loads(PANEL.read_text())
    summary = dict(status="running", scheduled=72, finished=0, results=[], resources={})
    ap.save(OUT / "summary.json", summary)
    totals = collections.Counter()
    started = time.monotonic()
    deadline = started + 1800
    try:
        geometry_preflight(panel)
        with h5py.File(ROOT / "data/tworoom.h5", "r") as data:
            bank, _ = ap.make_bank(data)
            _, native = ap.native_model()
            for case in panel["cases"]:
                schedule = [("safe", 6101), ("risky", 6101)]
                schedule += [
                    (arm, seed) for seed in panel["seeds"] for arm in ["clean", "monitored"]
                ]
                for arm, seed in schedule:
                    result = episode(case, arm, seed, native, bank, data, totals, deadline)
                    summary["results"].append(result)
                    summary["finished"] += 1
                    summary["resources"] = dict(totals)
                    summary["seconds"] = time.monotonic() - started
                    ap.save(OUT / "summary.json", summary)
                    assert totals["environment_resets"] <= 72
                    assert totals["environment_actions"] <= 14400
                    assert totals["candidate_model_steps"] <= 17283840
        summary["status"] = "complete"
    except BaseException:
        summary["status"] = "failed"
        summary["failure"] = traceback.format_exc()
        raise
    finally:
        summary["resources"] = dict(totals)
        summary["seconds"] = time.monotonic() - started
        ap.save(OUT / "summary.json", summary)


if __name__ == "__main__":
    main()
