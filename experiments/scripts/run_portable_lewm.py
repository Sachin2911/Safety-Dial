"""Frozen learned-safety extension, trained and executed only on the authorized pod."""

import os

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import argparse
import collections
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.portable_env import NativeEnv
from helpers.portable_motion import TransitionModel, MotionChecker, clearance, load_model
from helpers.portable_motion import shape_constraints
from helpers.pusht_preset import KINDS, configurations
from helpers.pusht_smoke import Scaler
from helpers.safety_data import write_json
from helpers.safety_model import file_digest

CONFIG = ROOT / "configs/lewm_portable_20261030"
DESIGN = json.loads((CONFIG / "design.json").read_text())
OUTPUT = ROOT / "outputs/lewm-preset-expansion-20261030"
SOURCES = [
    Path(__file__),
    ROOT / "experiments/helpers/portable_motion.py",
    ROOT / "experiments/helpers/portable_env.py",
    ROOT / "experiments/helpers/lewm18_planner.py",
    ROOT / "experiments/helpers/cem_refinements.py",
    ROOT / "experiments/helpers/lewm16_planner.py",
    ROOT / "experiments/helpers/safety_model.py",
    ROOT / "experiments/helpers/pusht_preset.py",
    ROOT / "experiments/helpers/tworoom12_env.py",
    ROOT / "experiments/helpers/reacher13_env.py",
    ROOT / "experiments/helpers/cube14_env.py",
    ROOT / "experiments/helpers/lewm18_runtime.py",
    ROOT / "experiments/helpers/safecem10_planner.py",
]


def summary(rows):
    return {
        "episodes": len(rows),
        "safe_completions": sum(r["safe_completion"] for r in rows),
        "violations": sum(r["margin_breach"] for r in rows),
        "contacts": sum(r["hazard_violation"] for r in rows),
        "unfinished": sum(r["unfinished"] for r in rows),
        "engaged": sum(r["engaged"] for r in rows),
        "J": sum(r["actions"] if r["safe_completion"] else 150 for r in rows),
        "actions": sum(r["actions"] for r in rows),
        "completion_at": {
            str(limit): sum(r["safe_completion"] and r["actions"] <= limit for r in rows)
            for limit in (50, 100, 150)
        },
        "planning_seconds": sum(r["planning_seconds"] for r in rows),
        "resources": dict(
            sum((collections.Counter(r["resources"]) for r in rows), collections.Counter())
        ),
    }


def select(configs, results):
    expected = {(i, condition) for i in range(6) for condition in ("control", "hazard")}
    scores, eligible = {}, []
    base = sum(r["safe_completion"] for r in results["authors"] if r["condition"] == "control")
    for order, config in enumerate(configs):
        rows = results[config["id"]]
        if len(rows) != 12 or {(r["case_index"], r["condition"]) for r in rows} != expected:
            raise ValueError("Incomplete development panel")
        if any(r["split"] != "development" or r["error"] for r in rows):
            raise ValueError("Engineering errors cannot select parameters")
        h = summary([r for r in rows if r["condition"] == "hazard"])
        c = summary([r for r in rows if r["condition"] == "control"])
        scores[config["id"]] = {"hazard": h, "control": c}
        if h["violations"] or c["violations"] or c["safe_completions"] < base:
            continue
        eligible.append(
            ((-h["safe_completions"], -c["safe_completions"], h["J"] + c["J"], order), config)
        )
    if not eligible:
        raise ValueError("No eligible configuration; reserved tasks remain untouched")
    return {"selected": min(eligible, key=lambda pair: pair[0])[1], "development": scores}


def freeze(out, stage):
    files = SOURCES + list(CONFIG.glob("*.json"))
    hashes = {str(p.relative_to(ROOT)): file_digest(p) for p in files}
    target = out / (stage + "-freeze.json")
    if target.exists():
        if json.loads(target.read_text())["sources"] != hashes:
            raise RuntimeError("Frozen source changed: use a new repair namespace")
    else:
        write_json(target, {"epoch": time.time(), "sources": hashes, "design": DESIGN})
        for p in files:
            q = out / "source" / p.relative_to(ROOT)
            q.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, q)


def scaler(name):
    col = json.loads((CONFIG / (name + "-scalers.json")).read_text())["columns"]["action"]
    return Scaler(col["mean"], col["scale"])


def cube_data(split):
    suffix = "-val" if split == "val" else ""
    path = ROOT / f"data/lewm-preset-expansion/cube/cube-single-play-v0{suffix}.npz"
    with np.load(path) as z:
        data = {k: z[k] for k in ("qpos", "qvel", "terminals")}
    ends = np.flatnonzero(data["terminals"]) + 1
    starts = np.r_[0, ends[:-1]]
    data["states"] = np.c_[data["qpos"], data["qvel"]]
    data["episodes"] = list(zip(starts, ends))
    return data


def training_state(name, group, rng, dataset):
    if name == "cube":
        start, end = dataset["episodes"][group]
        # Different starts within the same source group remain grouped for calibration.
        return dataset["states"][rng.integers(start, max(start + 1, end - 26))].copy()
    if name == "reacher":
        return np.r_[rng.uniform(-2, 2, 2), rng.uniform(-0.5, 0.5, 2)]
    return np.array([rng.uniform(40, 85) + 100 * (group % 2), rng.uniform(35, 185)])


def train(name, out):
    dest = out / "motion"
    dest.mkdir(parents=True, exist_ok=False)
    freeze(dest, "training")
    c = collections.Counter()
    env = None
    started = time.perf_counter()
    try:
        env = NativeEnv(name, c, dest / "interactions.jsonl")
        normal = scaler(name)
        dataset = cube_data("train") if name == "cube" else None
        groups = {"train": range(80), "calibration": range(80, 92), "test": range(92, 104)}
        arrays = {}
        for split, indices in groups.items():
            before, after, paths, actions, labels = [], [], [], [], []
            for group in indices:
                for repeat in range(4):
                    seed = 301030 + group * 100 + repeat
                    rng = np.random.default_rng(seed)
                    observed = env.reset(seed, training_state(name, group, rng, dataset))
                    prev = np.zeros(env.env.action_space.shape)
                    spread = (0.5, 1.0, 2.0, 1.0)[repeat]
                    for _ in range(25):
                        a = 0.5 * prev + np.sqrt(0.75) * rng.normal(size=prev.shape) * spread
                        raw = normal.inverse_transform(a)
                        raw = np.clip(raw, env.env.action_space.low, env.env.action_space.high)
                        normalized = normal.transform(raw)
                        nxt, dense = env.step(raw)
                        before.append(observed.copy())
                        after.append(nxt.copy())
                        paths.append(dense.copy())
                        actions.append(normalized.copy())
                        labels.append(group)
                        observed, prev = nxt, normalized
            arrays[split] = {
                "before": np.asarray(before),
                "after": np.asarray(after),
                "paths": np.asarray(paths),
                "actions": np.asarray(actions),
                "groups": np.asarray(labels),
            }
            np.savez_compressed(dest / (split + ".npz"), **arrays[split])
            write_json(dest / "live.json", {"stage": split, "resources": dict(c)})
        d = arrays["train"]
        o, a, p = d["before"].shape[-1], d["actions"].shape[-1], d["paths"].shape[-2]
        angles = [] if name == "tworoom" else list(range(2 if name == "reacher" else 6))
        spec = {
            "environment": name,
            "observation_dim": o,
            "action_dim": a,
            "substeps": p,
            "max_horizon": 5,
            "action_block": 5,
            "angle_indices": angles,
            "body_radius": DESIGN["body_radius"][name],
            "units": DESIGN["units"][name],
            "normalized_action_low": normal.transform(env.env.action_space.low).tolist(),
            "normalized_action_high": normal.transform(env.env.action_space.high).tolist(),
        }
        torch.manual_seed(301030)
        model = TransitionModel(spec).cuda()
        x = torch.as_tensor(np.c_[d["before"], d["actions"]], dtype=torch.float32, device="cuda")
        y = torch.as_tensor(
            np.c_[
                d["after"] - d["before"],
                (d["paths"] - d["before"][:, None, -2:]).reshape(len(x), -1),
            ],
            dtype=torch.float32,
            device="cuda",
        )
        with torch.no_grad():
            model.xmean.copy_(x.mean(0))
            model.xscale.copy_(x.std(0).clamp_min(1e-4))
            model.ymean.copy_(y.mean(0))
            model.yscale.copy_(y.std(0).clamp_min(1e-5))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
        obs = x[:, :o]
        act = x[:, o:]
        losses = []
        for update in range(6000):
            idx = torch.randint(len(x), (512,), device="cuda")
            nxt, path = model.transition(obs[idx], act[idx])
            pred = torch.cat([nxt - obs[idx], (path - obs[idx, None, -2:]).flatten(1)], -1)
            loss = ((pred - y[idx]) / model.yscale).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if update % 500 == 0:
                losses.append([update, float(loss.detach())])
                print(
                    json.dumps({"environment": name, "update": update, "loss": losses[-1][1]}),
                    flush=True,
                )
        model.eval().requires_grad_(False)
        errors, stats = {}, {}
        with torch.inference_mode():
            for split in ("calibration", "test"):
                v = arrays[split]
                initial = torch.as_tensor(v["before"][::25], dtype=torch.float32, device="cuda")
                acts = torch.as_tensor(
                    v["actions"].reshape(-1, 25, a), dtype=torch.float32, device="cuda"
                )
                expected = torch.as_tensor(
                    v["paths"].reshape(-1, 25 * p, 2), dtype=torch.float32, device="cuda"
                )
                prediction = model(initial, acts)
                err = (prediction - expected).norm(dim=-1).amax(-1).cpu().numpy()
                g = v["groups"][::25]
                grouped = [float(err[g == group].max()) for group in np.unique(g)]
                errors[split] = grouped
                stats[split] = {
                    "group_errors": grouped,
                    "max_error": max(grouped),
                    "mean_trajectory_max_error": float(err.mean()),
                }
                np.savez_compressed(
                    dest / (split + "-predictions.npz"),
                    predicted=prediction.cpu().numpy(),
                    expected=expected.cpu().numpy(),
                )
        rank = int(np.ceil(13 * 0.9))
        allowance = max(sorted(errors["calibration"])[rank - 1], 1e-4)
        covered = sum(e <= allowance + 1e-6 for e in errors["test"])
        calibration = {
            "allowance": allowance,
            "test_groups_covered": covered,
            "test_groups": 12,
            "alpha": 0.1,
            "calibration_groups": 12,
            "ready": covered >= 11
            and allowance <= DESIGN["motion_training"]["max_allowance"][name],
            "interpretation": "Empirical independent group check; no deterministic safety guarantee.",
        }
        torch.save(
            {"spec": spec, "state_dict": model.state_dict(), "calibration": calibration},
            dest / "model.pt",
        )
        write_json(
            dest / "report.json",
            {
                "spec": spec,
                "calibration": calibration,
                "errors": stats,
                "training_updates": 6000,
                "training_losses": losses,
                "resources": dict(c),
                "elapsed_seconds": time.perf_counter() - started,
                "model_sha256": file_digest(dest / "model.pt"),
            },
        )
        print(
            json.dumps({"environment": name, "calibration": calibration, "resources": dict(c)}),
            flush=True,
        )
    except Exception:
        write_json(dest / "error.json", {"error": traceback.format_exc(), "resources": dict(c)})
        raise
    finally:
        if env is not None:
            env.close()
        write_json(dest / "resources.json", dict(c))


def layout(route, kind, strength, name):
    length = np.r_[0, np.linalg.norm(np.diff(route, axis=0), axis=1).cumsum()]
    centre = route[np.argmin(abs(length - length[-1] / 2))]
    forward = route[-1] - route[0]
    forward /= np.linalg.norm(forward)
    side = np.array([forward[1], -forward[0]])
    scale = 1 if name == "tworoom" else 0.003
    shape = {"kind": kind, "centre": centre.tolist(), "axis": side.tolist()}
    if kind == "island":
        shape["radius"] = (3 if strength == "mild" else 5) * scale
    elif kind == "bar":
        shape.update(half_length=(8 if strength == "mild" else 12) * scale, radius=2 * scale)
    else:
        shape.update(
            offset=((60 if strength == "mild" else 48) / 2 + 10) * scale,
            gap=(60 if strength == "mild" else 48) * scale,
            extent=90 * scale,
            band=2 * scale,
        )
    return shape


def prepare(name, out):
    freeze(out, "preparation")
    head = load_model(out / "motion/model.pt")
    if not head.calibration["ready"]:
        raise ValueError("Motion model failed release gate")
    c = collections.Counter()
    env = NativeEnv(name, c)
    cases, rejected = [], []
    (out / "inputs").mkdir(exist_ok=False)
    source = cube_data("val") if name == "cube" else None
    try:
        candidates = []
        if name == "cube":
            for episode, (begin, end) in enumerate(source["episodes"]):
                for row in range(int(begin), int(end) - 25, 25):
                    candidates.append((episode, row))
        else:
            candidates = [(index, 0) for index in range(12)]
        used = set()
        for episode, row in candidates:
            if len(cases) == 12:
                break
            if episode in used:
                continue
            if c["environment_resets"] >= 240:
                raise RuntimeError("Admission budget exhausted")
            index = len(cases)
            kind, strength = KINDS[index % 6]
            seed = 303000 + index
            if name == "tworoom":
                x = 164 if index < 6 else 60
                initial = np.array([x - 3 + index % 3, 43 + index % 4])
                goal = np.array([x + 3 - index % 3, 180 - index % 4])
                env.reset(seed, initial, goal)
                route = np.linspace(initial, goal, 26)
                image = np.asarray(env.info["goal"])
                if image.shape[0] == 3:
                    image = image.transpose(1, 2, 0)
                if image.max() <= 1:
                    image = (image * 255).astype(np.uint8)
            elif name == "reacher":
                elbow = 0.9 if index < 6 else 1.2
                initial = np.r_[[-0.9 + 0.02 * (index % 6), elbow], [0.0, 0.0]]
                goal = np.array([0.9 - 0.02 * (index % 6), elbow])
                env.reset(seed, initial, goal)
                qs = np.linspace(initial[:2], goal, 26)
                route = np.asarray([env.pose(np.r_[q, [0.0, 0.0]])[0] for q in qs])
                image = env.info["goal"]
            else:
                initial = source["states"][row].copy()
                future = source["states"][row + 25].copy()
                # qpos ends in the 7D free cube joint; xyz is the first three entries.
                nq = source["qpos"].shape[1]
                if np.linalg.norm(initial[nq - 7 : nq - 4] - future[nq - 7 : nq - 4]) <= 0.05:
                    continue
                env.reset(seed, initial)
                route = np.asarray([env.pose(s)[0] for s in source["states"][row : row + 26]])
                if np.linalg.norm(route[-1] - route[0]) < 0.12:
                    continue
                _, (image, goal) = env.pose(future, render=True)
                env.reset(seed, initial, goal, image)
            actual = env.observe()[-2:]
            route[0] = actual
            shape = layout(route, kind, strength, name)
            endpoints = torch.as_tensor(np.stack([actual, route[-1]]), dtype=torch.float32)
            dist = clearance(endpoints, shape_constraints(shape), DESIGN["body_radius"][name])
            reference_clearance = clearance(
                torch.as_tensor(route), shape_constraints(shape), DESIGN["body_radius"][name]
            )
            if float(reference_clearance.min()) >= 0:
                raise ValueError("Hazard must intersect the reference route")
            buffer = 2.0 if name == "tworoom" else 0.005
            threshold = DESIGN["margin"][name] + head.calibration["allowance"] + buffer
            if bool((dist < threshold).any()) or env.success():
                rejected.append(
                    {
                        "episode": int(episode),
                        "row": int(row),
                        "reason": "endpoint admission",
                        "clearance": dist.tolist(),
                    }
                )
                continue
            case = {
                "id": f"case-{index:02d}-{kind}-{strength}",
                "index": index,
                "split": "development" if index < 6 else "evaluation",
                "seed": seed,
                "source_episode": int(episode),
                "source_row": int(row),
                "shape": shape,
                "initial_clearance": float(dist[0]),
                "goal_clearance": float(dist[1]),
                "reference_route_minimum_clearance": float(reference_clearance.min()),
            }
            path = out / "inputs" / (case["id"] + ".npz")
            np.savez_compressed(
                path,
                initial=initial,
                goal=goal,
                goal_image=image,
                start_image=env.render(),
                reference_route=route,
            )
            case["sha256"] = file_digest(path)
            cases.append(case)
            used.add(episode)
        if len(cases) != 12:
            raise RuntimeError("Fewer than 12 geometrically admissible cases")
        wm = ROOT / f"data/lewm-preset-expansion/{name}/lewm_object.ckpt"
        write_json(
            out / "panel.json",
            {
                "cases": cases,
                "rejected": rejected,
                "configurations": configurations(DESIGN),
                "resources": dict(c),
                "world_model_sha256": file_digest(wm),
                "motion_model_sha256": file_digest(out / "motion/model.pt"),
                "sources": json.loads((out / "preparation-freeze.json").read_text())["sources"],
            },
        )
    finally:
        write_json(out / "preparation-resources.json", dict(c))
        write_json(out / "admission-record.json", {"cases": cases, "rejected": rejected})
        env.close()


def runtime(name, out):
    from stable_worldmodel.policy import AutoCostModel

    head = load_model(out / "motion/model.pt")
    path = ROOT / f"data/lewm-preset-expansion/{name}/lewm"
    wm = AutoCostModel(str(path), cache_dir=path.parent).to("cuda").eval().requires_grad_(False)
    panel = json.loads((out / "panel.json").read_text())
    if file_digest(Path(str(path) + "_object.ckpt")) != panel["world_model_sha256"]:
        raise RuntimeError("World model changed")
    if file_digest(out / "motion/model.pt") != panel["motion_model_sha256"]:
        raise RuntimeError("Safety model changed")
    return wm, head


def trial(name, out, rt, config, case, condition):
    import gymnasium as gym
    import stable_worldmodel as swm
    from helpers.lewm18_planner import CostModel, PlanningCEM
    from helpers.lewm16_planner import NoSafeAction
    from helpers.safecem10_planner import image_transform

    wm, head = rt
    folder = out / case["split"] / config["id"] / case["id"] / condition
    folder.mkdir(parents=True, exist_ok=False)
    write_json(
        folder / "started.json",
        {"case": case, "config": config, "condition": condition, "epoch": time.time()},
    )
    pair_path = out / "inputs" / (case["id"] + ".npz")
    if file_digest(pair_path) != case["sha256"]:
        raise RuntimeError("Case changed")
    with np.load(pair_path) as z:
        pair = dict(z)
    counters = collections.Counter()
    env = NativeEnv(name, counters, folder / "interactions.jsonl")
    params = config["params"]
    shape = case["shape"] if condition == "hazard" else {"kind": "none"}
    constraints = shape_constraints(shape)
    torch.manual_seed(case["seed"])
    np.random.seed(case["seed"])
    observed = env.reset(case["seed"], pair["initial"], pair["goal"], pair["goal_image"])
    checker = MotionChecker(head, constraints, DESIGN["margin"][name], counters)
    checker.observe(observed)
    cost = CostModel(wm, checker, params, counters)
    solver = PlanningCEM(cost, "pusht", params, case["seed"], "cuda")
    normal = scaler(name)
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
            action_space=gym.spaces.Box(-1, 1, (1, head.spec["action_dim"]), dtype=np.float64),
        )
    )
    states, actions, frames = [observed.copy()], [], []
    dense = [observed[-2:].copy()]
    previous = np.zeros(head.spec["action_dim"])
    error, failure, success = None, None, False
    started = time.perf_counter()
    planning_seconds = 0.0
    try:
        if env.success():
            raise RuntimeError("Already successful start")
        for _ in range(150):
            frame = env.render()
            frames.append(frame)
            before = time.perf_counter()
            action = policy.get_action(
                {
                    "pixels": frame[None, None],
                    "goal": pair["goal_image"][None, None],
                    "action": previous[None, None],
                }
            )[0]
            torch.cuda.synchronize()
            planning_seconds += time.perf_counter() - before
            observed, path = env.step(action)
            dense.extend(path)
            states.append(observed.copy())
            actions.append(action.copy())
            checker.observe(observed)
            previous = action
            success = env.success()
            write_json(folder / "live.json", {"actions": len(actions), "resources": dict(counters)})
            if success:
                break
            if time.perf_counter() - started > 900:
                raise TimeoutError("Episode wall-time cap")
    except NoSafeAction as exc:
        failure = str(exc)
    except Exception:
        error = traceback.format_exc()
    elapsed = time.perf_counter() - started
    frames.append(env.render())
    clear = clearance(torch.as_tensor(np.asarray(dense)), constraints, DESIGN["body_radius"][name])
    violation = bool((clear < DESIGN["margin"][name] - 1e-6).any())
    contact = bool((clear <= 0).any())
    pathlength = float(np.linalg.norm(np.diff(np.asarray(states)[:, -2:], axis=0), axis=1).sum())
    result = {
        "environment": name,
        "split": case["split"],
        "case_index": case["index"],
        "case_id": case["id"],
        "configuration": config["id"],
        "condition": condition,
        "success": bool(success),
        "safe_completion": bool(success and not violation and not error),
        "margin_breach": violation,
        "arena_margin_breach": False,
        "hazard_violation": contact,
        "unfinished": not bool(success),
        "actions": len(actions),
        "protected_body_path_length": pathlength,
        "engaged": pathlength > (1.0 if name == "tworoom" else 0.001),
        "minimum_clearance": float(clear.min()),
        "error": error,
        "controller_failure": failure,
        "planning_seconds": planning_seconds,
        "episode_seconds": elapsed,
        "resources": dict(counters),
    }
    np.savez_compressed(
        folder / "trace.npz",
        states=states,
        actions=actions,
        dense=dense,
        frames=frames,
        goal_image=pair["goal_image"],
    )
    write_json(folder / "replans.json", solver.replans)
    write_json(folder / "cost-history.json", cost.history)
    write_json(folder / "result.json", result)
    env.close()
    print(json.dumps(result), flush=True)
    return result


def compare(name, out, stage):
    freeze(out, "comparison")
    panel = json.loads((out / "panel.json").read_text())
    current = json.loads((out / "comparison-freeze.json").read_text())["sources"]
    if panel["sources"] != current:
        raise RuntimeError("Source changed after preparation")
    configs = panel["configurations"]
    rt = runtime(name, out)
    if stage == "tune":
        results = {}
        for config in configs:
            rows = []
            for case in panel["cases"][:6]:
                for condition in ("control", "hazard"):
                    rows.append(trial(name, out, rt, config, case, condition))
            results[config["id"]] = rows
        chosen = select(configs, results)
        write_json(out / "selection.json", chosen)
    else:
        chosen = json.loads((out / "selection.json").read_text())["selected"]
        write_json(
            out / "evaluation-freeze.json",
            {
                "selected": chosen,
                "epoch": time.time(),
                "selection_sha256": file_digest(out / "selection.json"),
            },
        )
        for config in (configs[0], chosen):
            # If authors is selected, reuse its reserved arm rather than duplicate it.
            if config["id"] == configs[0]["id"] and config is chosen:
                continue
            for case in panel["cases"][6:]:
                for condition in ("control", "hazard"):
                    folder = out / case["split"] / config["id"] / case["id"] / condition
                    if folder.exists() and chosen["id"] == "authors":
                        continue
                    trial(name, out, rt, config, case, condition)


def report(name, out):
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("*/**/result.json"))]
    counts = collections.Counter()
    for row in rows:
        counts.update(row["resources"])
    for p in (out / "motion/resources.json", out / "preparation-resources.json"):
        if p.exists():
            counts.update(json.loads(p.read_text()))
    selection = (
        json.loads((out / "selection.json").read_text())
        if (out / "selection.json").exists()
        else None
    )
    evaluations = {}
    if selection:
        for config in ("authors", selection["selected"]["id"]):
            evaluations[config] = {
                condition: summary(
                    [
                        r
                        for r in rows
                        if r["split"] == "evaluation"
                        and r["configuration"] == config
                        and r["condition"] == condition
                    ]
                )
                for condition in ("control", "hazard")
            }
    audit_errors = []
    for p in out.glob("*/**/cost-history.json"):
        plans = json.loads(p.read_text())
        work = sum(r["samples"] * r["horizon"] for r in plans)
        if work > 270000:
            audit_errors.append(str(p.relative_to(out)))
    panel = json.loads((out / "panel.json").read_text())
    paired_diagnostics = []
    for row in rows:
        if row["condition"] != "control":
            continue
        case = next(c for c in panel["cases"] if c["id"] == row["case_id"])
        folder = out / row["split"] / row["configuration"] / row["case_id"]
        with np.load(folder / "control/trace.npz") as z:
            points = torch.as_tensor(z["dense"])
        minimum = float(
            clearance(points, shape_constraints(case["shape"]), DESIGN["body_radius"][name]).min()
        )
        hazard = json.loads((folder / "hazard/result.json").read_text())
        paired_diagnostics.append(
            {
                "split": row["split"],
                "configuration": row["configuration"],
                "case": row["case_id"],
                "control_success": row["success"],
                "control_rescored_hazard_clearance": minimum,
                "hazard_safe_completion": hazard["safe_completion"],
                "deliberate_safe_adaptation": bool(
                    row["success"]
                    and minimum < DESIGN["margin"][name]
                    and hazard["safe_completion"]
                ),
            }
        )
    report = {
        "environment": name,
        "selection": selection,
        "evaluation": evaluations,
        "resources": dict(counts),
        "trials": len(rows),
        "errors": sum(bool(r["error"]) for r in rows),
        "budget_audit_errors": audit_errors,
        "paired_hazard_diagnostics": paired_diagnostics,
        "motion_report": json.loads((out / "motion/report.json").read_text()),
        "limitations": "One seed per case. Distinct source groups, not globally unseen WM data. XY protected-body safety only; empirical calibration, not formal safety. TwoRoom within-room panel, no door-crossing claim. Cube original object-position task. No evolutionary claim.",
    }
    write_json(out / "report.json", report)
    print(
        json.dumps(
            {k: report[k] for k in ("environment", "trials", "errors", "resources", "evaluation")}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["tworoom", "reacher", "cube"], required=True)
    parser.add_argument(
        "--stage", choices=["train", "prepare", "tune", "evaluate", "report"], required=True
    )
    args = parser.parse_args()
    if os.environ.get("RUNPOD_POD_ID") != DESIGN["pod_id"]:
        raise RuntimeError("Run only on the authorized remote pod")
    out = OUTPUT / args.env
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "train":
        train(args.env, out)
    elif args.stage == "prepare":
        prepare(args.env, out)
    elif args.stage in ("tune", "evaluate"):
        compare(args.env, out, args.stage)
    else:
        report(args.env, out)


if __name__ == "__main__":
    main()
