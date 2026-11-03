"""Frozen PushT and Cube screen stages. Execute only on the authorised pod.

Stages: inspect, panel, cache, engineering, long, hazard. Each stage freezes its
sources before the first simulator reset and records every reset, action, model
batch call, candidate model step and encoder frame.
"""

import argparse
import collections
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import manip as M  # noqa: E402

sys.path.insert(0, str(M.REF))

SEEDS_HAZARD = [6101, 6102]
SEED_LONG = 42
SEED_PANEL = 42
PANEL_DRAW = 60
LAYOUTS_PER_FAMILY = 4
FAMILIES = ["on_path", "near_path"]
SCAN_EPISODES = 4000


def task_out(task):
    return M.OUT / task


def open_data(task):
    return h5py.File(M.ROOT / M.TASKS[task]["h5"], "r")


def episode_index(data):
    lengths = np.asarray(data["ep_len"], np.int64)
    offsets = np.asarray(data["ep_offset"], np.int64)
    return lengths, offsets


def source_records(task):
    sources = [
        M.ROOT / "ManipulationScreen.md",
        Path(__file__),
        M.ROOT / "experiments/scripts/audit_manip.py",
        M.ROOT / "experiments/helpers/manip.py",
        M.ROOT / "experiments/helpers/qualification.py",
        *sorted((M.REF / "anchored_planning").glob("*.py")),
        M.REF / "protocols/scalers.json",
    ]
    records = {}
    for path in sources:
        if path.exists():
            records[str(path.relative_to(M.ROOT))] = M.digest(path)
    return records


def stage_folder(task, stage):
    """Next unused versioned folder; failed attempts are kept."""
    n = 1
    while (task_out(task) / f"{stage}-v{n}").exists():
        n += 1
    return task_out(task) / f"{stage}-v{n}"


def latest_complete(task, stage):
    folders = sorted(task_out(task).glob(f"{stage}-v*"), key=lambda f: int(f.name.rsplit("-v", 1)[1]))
    for folder in reversed(folders):
        summary = folder / "summary.json"
        if summary.exists() and json.loads(summary.read_text()).get("status") == "complete":
            return folder
    raise FileNotFoundError(f"no complete {stage} stage for {task}")


def freeze(task, stage, extra=None):
    import torch

    folder = stage_folder(task, stage)
    folder.mkdir(parents=True, exist_ok=False)
    records = source_records(task)
    for name in records:
        destination = folder / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((M.ROOT / name).read_bytes())
    env_source = None
    if stage in {"engineering", "long", "hazard"}:
        if task == "pusht":
            from stable_worldmodel.envs.pusht.env import PushT as EnvClass
        else:
            from stable_worldmodel.envs.ogbench.cube_env import CubeEnv as EnvClass
        env_source = Path(inspect.getfile(EnvClass))
        (folder / "sources/native-env.py").write_bytes(env_source.read_bytes())
    record = dict(
        utc=M.utc(), task=task, stage=stage, sources=records,
        checkpoint_sha256=M.digest(M.ROOT / M.TASKS[task]["weights"]),
        environment_sha256=M.digest(env_source) if env_source else None,
        pod_id=os.environ.get("RUNPOD_POD_ID"), torch_version=torch.__version__,
        renderer={k: os.environ.get(k) for k in ("MUJOCO_GL", "PYOPENGL_PLATFORM")},
        no_training=True, exclusion=M.exclusion(task), spec=M.TASKS[task],
    )
    record.update(extra or {})
    M.save(folder / "freeze.json", record)
    return folder


# ----------------------------------------------------------------------------- inspect

def inspect_stage(task):
    folder = task_out(task)
    folder.mkdir(parents=True, exist_ok=True)
    with open_data(task) as data:
        lengths, offsets = episode_index(data)
        report = dict(
            utc=M.utc(), task=task, rows=int(len(data["pixels"])), episodes=int(len(lengths)),
            keys={k: dict(shape=list(data[k].shape), dtype=str(data[k].dtype)) for k in data.keys()},
            ep_len=dict(min=int(lengths.min()), max=int(lengths.max()), mean=float(lengths.mean()),
                        at_least_horizon_plus_one=int((lengths >= M.TASKS[task]["horizon"] + 1).sum())),
            offsets_contiguous=bool(np.all(offsets[1:] == offsets[:-1] + lengths[:-1])),
            receipt=json.loads((M.OUT / f"data-receipt-{task}.json").read_text()),
        )
    M.save(folder / "inspect.json", report)
    print(json.dumps(report["ep_len"]), flush=True)


# ----------------------------------------------------------------------------- panel

def initially_successful(task, data, start, goal):
    if task == "pusht":
        s, g = np.asarray(data["state"][start]), np.asarray(data["state"][goal])
        pos = np.linalg.norm(g[:4] - s[:4])
        angle = abs(g[4] - s[4])
        angle = min(angle, 2 * np.pi - angle)
        return bool(pos < 20 and angle < np.pi / 9)
    s, g = np.asarray(data["privileged_block_0_pos"][start]), np.asarray(data["privileged_block_0_pos"][goal])
    return bool(np.linalg.norm(g - s) <= 0.04)


def panel_stage(task):
    spec = M.TASKS[task]
    folder = task_out(task)
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / "panel.json").exists()
    horizon = spec["horizon"]
    with open_data(task) as data:
        lengths, offsets = episode_index(data)
        episode_of = np.repeat(np.arange(len(lengths)), lengths)
        step_idx = np.arange(len(episode_of)) - offsets[episode_of]
        # Official evaluator rule: starts with at least `horizon` steps remaining, seeded draw, sorted.
        valid = np.nonzero(step_idx <= lengths[episode_of] - horizon - 1)[0]
        generator = np.random.default_rng(SEED_PANEL)
        drawn = np.sort(valid[generator.choice(len(valid) - 1, size=PANEL_DRAW, replace=False)])
        stage_a = []
        rejected = []
        for row in drawn:
            row = int(row)
            if initially_successful(task, data, row, row + horizon):
                rejected.append(row)
                continue
            stage_a.append(dict(id=f"{task}-long-{len(stage_a):02d}", episode=int(episode_of[row]),
                                start_row=row, goal_row=row + horizon, horizon=horizon, seed=SEED_LONG))
            if len(stage_a) == 50:
                break
        assert len(stage_a) == 50, len(stage_a)
        used = {c["episode"] for c in stage_a}
        layouts = {family: [] for family in FAMILIES}
        development = {family: None for family in FAMILIES}
        scanned = 0
        eligibility = collections.Counter()
        for episode in range(min(SCAN_EPISODES, len(lengths))):
            if all(len(layouts[f]) >= LAYOUTS_PER_FAMILY and development[f] for f in FAMILIES):
                break
            if episode in used:
                continue
            start = int(offsets[episode])
            if task == "pusht":
                if lengths[episode] < horizon + 1:
                    eligibility["short_episode"] += 1
                    continue
                goal = start + horizon
                states = np.asarray(data["state"][start:goal + 1], np.float64)
                if initially_successful(task, data, start, goal):
                    eligibility["initially_successful"] += 1
                    continue
                candidates = {f: M.pusht_layout(states, f, spec) for f in FAMILIES}
            else:
                flags = np.asarray(data["success"][start:start + int(lengths[episode])]).astype(bool).reshape(-1)
                hits = np.nonzero(flags)[0]
                if len(hits) == 0:
                    eligibility["no_success"] += 1
                    continue
                goal = start + int(hits[0])
                case_horizon = goal - start
                if not spec["min_horizon"] <= case_horizon <= spec["max_horizon"]:
                    eligibility["horizon_out_of_range"] += 1
                    continue
                pinch = np.asarray(data["proprio_effector_pos"][start:goal + 1], np.float64)
                cube = np.asarray(data["privileged_block_0_pos"][start:goal + 1], np.float64)
                candidates = {f: M.cube_layout(pinch, cube, f, spec) for f in FAMILIES}
            scanned += 1
            placed = False
            for family in FAMILIES:
                layout = candidates[family]
                if layout is None:
                    eligibility[f"{family}_ineligible"] += 1
                    continue
                record = dict(episode=episode, start_row=start, goal_row=goal, horizon=goal - start, layout=layout)
                if len(layouts[family]) < LAYOUTS_PER_FAMILY:
                    record["id"] = f"{task}-{family}-{len(layouts[family])}"
                    layouts[family].append(record)
                elif development[family] is None:
                    record["id"] = f"{task}-{family}-dev"
                    development[family] = record
                else:
                    continue
                used.add(episode)
                placed = True
                break
            if not placed:
                eligibility["unused"] += 1
        assert all(len(layouts[f]) == LAYOUTS_PER_FAMILY for f in FAMILIES), {f: len(layouts[f]) for f in FAMILIES}
        assert all(development[f] is not None for f in FAMILIES)
        excluded = sorted(used)
        eligible = [i for i in range(len(lengths)) if lengths[i] > 5 and i not in used]
        memory = eligible[:spec["memory_episodes"]]
        assert len(memory) == spec["memory_episodes"]
        panel = dict(
            utc=M.utc(), task=task, seed=SEED_PANEL, draw=PANEL_DRAW, horizon=horizon,
            stage_a=stage_a, rejected_initial_success_rows=rejected,
            stage_b=[r for f in FAMILIES for r in layouts[f]], development=[development[f] for f in FAMILIES],
            episodes_scanned=scanned, eligibility=dict(eligibility),
            memory_episodes=memory, excluded_episodes=excluded,
            dataset_sha256=json.loads((M.OUT / f"data-receipt-{task}.json").read_text())["dataset"].get("sha256"),
        )
    M.save(folder / "panel.json", panel)
    print(json.dumps(dict(stage_a=len(stage_a), stage_b={f: len(layouts[f]) for f in FAMILIES},
                          scanned=scanned, eligibility=dict(eligibility))), flush=True)


# ----------------------------------------------------------------------------- cache

def cache_stage(task, shard=0, shards=1):
    """Encode retrieval latents for memory-episode rows only, sharded over worker processes.

    Rows of episodes outside the memory are never eligible for retrieval and stay zero.
    Shard 0 creates the memmap; every shard writes its own completion record; the
    `cache-finish` stage hashes the result once all shards are done.
    """
    import torch
    from anchored_planning.model import encode_retrieval

    folder = task_out(task) / "memory-v1"
    panel = json.loads((task_out(task) / "panel.json").read_text())
    with open_data(task) as data:
        lengths, offsets = episode_index(data)
        n = int(lengths.sum())
        path = folder / "latents.npy"
        if shard == 0:
            folder.mkdir(parents=True, exist_ok=False)
            M.save(folder / "panel.json", dict(memory_episodes=panel["memory_episodes"],
                                               excluded_episodes=panel["excluded_episodes"],
                                               precision="bfloat16 encoder, float16 cache",
                                               rows_encoded="memory episodes only; other rows zero",
                                               shards=shards))
            output = np.lib.format.open_memmap(path, mode="w+", dtype="float16", shape=(n, 192))
            output.flush()
            del output
            (folder / "created.json").write_text(json.dumps(dict(utc=M.utc(), rows=n)) + "\n")
        else:
            waited = 0
            while not (folder / "created.json").exists():
                time.sleep(2)
                waited += 2
                assert waited < 300, "shard 0 did not create the cache"
        output = np.lib.format.open_memmap(path, mode="r+")
        assert output.shape == (n, 192)
        adapter, _ = M.native_model(task)
        episodes = [e for i, e in enumerate(panel["memory_episodes"]) if i % shards == shard]
        total = int(sum(lengths[e] for e in episodes))
        done = 0
        started = time.monotonic()
        batch = 256
        for count, episode in enumerate(episodes):
            start, stop = int(offsets[episode]), int(offsets[episode] + lengths[episode])
            for a in range(start, stop, batch):
                b = min(a + batch, stop)
                with torch.inference_mode():
                    output[a:b] = encode_retrieval(adapter, data["pixels"][a:b], torch.device("cuda"))
                done += b - a
            if count % 100 == 0 or count == len(episodes) - 1:
                output.flush()
                M.save(folder / f"progress-shard{shard}.json",
                       dict(encoded_frames=done, total=total, episodes=count + 1, seconds=time.monotonic() - started))
                print(f"shard {shard}: {done}/{total}", flush=True)
            if time.monotonic() - started > 7200:
                raise TimeoutError("Cache encoding deadline")
        output.flush()
        del output
    M.save(folder / f"shard{shard}-of{shards}.json", dict(frames=done, episodes=len(episodes),
                                                         seconds=time.monotonic() - started))


def cache_finish(task):
    folder = task_out(task) / "memory-v1"
    memory = json.loads((folder / "panel.json").read_text())
    shards = memory["shards"]
    records = [json.loads((folder / f"shard{i}-of{shards}.json").read_text()) for i in range(shards)]
    latents = np.load(folder / "latents.npy", mmap_mode="r")
    with open_data(task) as data:
        lengths, offsets = episode_index(data)
    frames = sum(r["frames"] for r in records)
    assert frames == int(sum(lengths[e] for e in memory["memory_episodes"]))
    # Spot check: memory rows are non-zero, excluded rows are zero.
    probe = memory["memory_episodes"][0]
    assert np.abs(latents[offsets[probe]:offsets[probe] + 3].astype(np.float32)).sum() > 0
    other = memory["excluded_episodes"][0]
    assert np.abs(latents[offsets[other]:offsets[other] + 3].astype(np.float32)).sum() == 0
    M.save(folder / "complete.json", dict(frames=frames, seconds=max(r["seconds"] for r in records),
                                          shards=records, cache_sha256=M.digest(folder / "latents.npy")))
    print(json.dumps(dict(frames=frames, sha256=M.digest(folder / "latents.npy"))), flush=True)


def make_bank(task, data):
    import torch
    from anchored_planning.observation_bank import ObservationBank

    folder = task_out(task) / "memory-v1"
    done = json.loads((folder / "complete.json").read_text())
    assert M.digest(folder / "latents.npy") == done["cache_sha256"]
    memory = json.loads((folder / "panel.json").read_text())
    lengths, offsets = episode_index(data)
    bank = ObservationBank(dict(lengths=lengths, offsets=offsets), np.load(folder / "latents.npy", mmap_mode="r"),
                           np.array(memory["memory_episodes"]), BLOCK_STEPS, np.ones(3, np.float32),
                           torch.device("cuda"))
    assert not set(memory["memory_episodes"]) & set(memory["excluded_episodes"])
    return bank, memory


BLOCK_STEPS = 5


# ----------------------------------------------------------------------------- episodes

def run_episode(task, folder, case, arm, seed, data, native, bank, totals, deadline, hazard=None, witness=None):
    """One episode of any arm. Witness arms use the recording and scripted waypoints only."""
    import torch
    from anchored_planning.planners import TargetGaussian

    spec = M.TASKS[task]
    name = f"{case['id']}_{arm}_{seed}"
    out = folder / name
    out.mkdir()
    counts = collections.Counter()
    arrays = M.case_arrays(task, data, case["start_row"], case["goal_row"])
    goal_image = np.asarray(data["pixels"][case["goal_row"]])
    trace = dict(states=[], points=[], actions=[], proposed_actions=[], frames=[], hit=[],
                 rewards=[], terminated=[], truncated=[], success=[], phase=[])
    decisions = []
    env = M.make_env(task)
    handle = None
    original_cost = None
    success = False
    witness_invalid = None
    status = "incomplete"
    started = time.monotonic()
    budget = spec["budget"]
    monitored = arm in {"monitored", "replay", "scripted"} and hazard is not None
    centre = np.asarray(hazard["hazard_centre"]) if hazard is not None else None
    try:
        counts["environment_resets"] += 1
        M.save(out / "live.json", dict(resources=dict(counts)))
        _, info = M.reset_case(env, task, arrays, seed)
        phys = M.physical(env, task)
        frame = np.ascontiguousarray(env.render())
        discrepancy = float(np.abs(frame.astype(float) - np.asarray(data["pixels"][case["start_row"]]).astype(float)).mean())
        trace["states"].append(phys["state"])
        trace["points"].append(np.asarray(phys["points"]))
        trace["frames"].append(frame)
        success = M.is_success(env, task, info, arrays)
        initial_success = success
        pending = []
        checked = False
        if arm in {"final", "ap", "clean", "monitored"}:
            def encoder_hook(_module, args):
                counts["encoder_frames"] += int(args[0].shape[0])

            handle = native.model.encoder.register_forward_pre_hook(encoder_hook)
            original_cost = native.latent_cost.get_cost

            def counted_cost(info_, actions):
                counts["model_batch_calls"] += 1
                counts["candidate_model_steps"] += int(np.prod(actions.shape[:3]))
                return original_cost(info_, actions)

            native.latent_cost.get_cost = counted_cost
            goal_encoding = native.encode_rgb(goal_image).cpu().numpy()

            class Images:
                def __getitem__(self, index):
                    return goal_image if int(index) == -1 else data["pixels"][int(index)]

            planner = TargetGaussian(native, Images())
            planner.set_action_bounds(env.action_space.low, env.action_space.high)
            generator = torch.Generator(device="cuda").manual_seed(seed)
        else:
            recorded = np.asarray(data["action"][case["start_row"]:case["goal_row"]], np.float32)
            plan = witness_plan(task, arm, case, hazard, recorded, spec)
            plan_index = 0
            phase_steps = 0
            waypoint_steps = 0
            release_steps = 0
        step = 0
        while step < budget and not success:
            if time.monotonic() >= deadline:
                raise TimeoutError("Frozen job deadline reached")
            before_points = trace["points"][-1]
            if arm in {"final", "ap", "clean", "monitored"}:
                phase = "planner"
                if not pending:
                    if arm == "final":
                        sources = [-1]
                        retrieved = None
                    else:
                        current = native.encode_rgb(trace["frames"][-1]).cpu().numpy()
                        remaining = max(5, case["horizon"] - step)
                        sources_array, distances = bank.nearest_topk(current, goal_encoding, remaining, 1)
                        sources = sources_array[0].tolist()
                        retrieved = distances[0].tolist()
                        assert bank.train_mask[bank.episode_idx[sources[0]]]
                        counts["retrieval_queries"] += 1
                    calls_before, blocks_before = counts["model_batch_calls"], counts["candidate_model_steps"]
                    selected, detail = planner.choose(
                        "gaussian_final" if arm == "final" else "gaussian_observed",
                        trace["frames"][-1], sources, generator, -1)
                    assert counts["model_batch_calls"] - calls_before == detail["predictor_batch_calls"]
                    assert counts["candidate_model_steps"] - blocks_before == detail["predictor_candidate_transitions"]
                    counts["replans"] += 1
                    detail.update(step=step, retrieval_distances=retrieved, sources=sources)
                    decisions.append(detail)
                    pending = list(selected)
                action = np.asarray(pending.pop(0), np.float32)
                proposed = action.copy()
            else:
                if plan_index >= len(plan):
                    break
                kind, payload = plan[plan_index]
                if kind == "replay":
                    phase = "replay"
                    if phase_steps >= len(payload):
                        plan_index += 1
                        phase_steps = 0
                        continue
                    action = np.asarray(payload[phase_steps], np.float32)
                    phase_steps += 1
                elif kind == "goto":
                    phase = "scripted"
                    target, settle, checked = payload
                    reached = M.at_waypoint(task, env, target, spec) if settle else _near_waypoint(task, env, target, spec)
                    if reached:
                        plan_index += 1
                        waypoint_steps = 0
                        continue
                    waypoint_steps += 1
                    if waypoint_steps > spec["settle_steps"] + 40:
                        witness_invalid = f"waypoint {plan_index} not reached"
                        break
                    action = M.witness_action(task, env, target, hold_closed=True)
                elif kind == "release":
                    phase = "release"
                    if release_steps >= payload:
                        plan_index += 1
                        continue
                    release_steps += 1
                    target = np.asarray(M.physical(env, task)["points"][0])
                    action = M.witness_action(task, env, target, hold_closed=False)
                elif kind == "hold":
                    phase = "hold"
                    if phase_steps >= payload:
                        plan_index += 1
                        phase_steps = 0
                        continue
                    phase_steps += 1
                    action = np.zeros(spec["action_dim"], np.float32)
                proposed = action.copy()
            assert np.isfinite(action).all() and (np.abs(action) <= 1 + 1e-6).all(), action
            action = np.clip(action, -1, 1).astype(np.float32)
            counts["environment_actions"] += 1
            if counts["environment_actions"] % 20 == 0:
                M.save(out / "live.json", dict(resources=dict(counts)))
            _, reward, terminated, truncated, info = env.step(action)
            phys = M.physical(env, task)
            after_points = np.asarray(phys["points"])
            hit = bool(monitored and M.monitor_hit(task, before_points, after_points, centre))
            success = M.is_success(env, task, info, arrays)
            trace["states"].append(phys["state"])
            trace["points"].append(after_points)
            trace["frames"].append(np.ascontiguousarray(env.render()))
            trace["actions"].append(action)
            trace["proposed_actions"].append(proposed)
            trace["hit"].append(hit)
            trace["rewards"].append(float(reward))
            trace["terminated"].append(bool(terminated))
            trace["truncated"].append(bool(truncated))
            trace["success"].append(bool(success))
            trace["phase"].append(phase)
            step += 1
            if arm == "scripted" and phase == "scripted" and checked:
                witness_invalid = witness_invalid or witness_check(task, env, trace, case, hazard, spec)
            if task == "pusht" and bool(terminated):
                success = True
            if (terminated or truncated) and not success:
                break
        status = "complete"
    except BaseException:
        status = "failed"
        (out / "traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        if handle is not None:
            handle.remove()
        if original_cost is not None:
            native.latent_cost.get_cost = original_cost
        env.close()
        np.savez_compressed(out / "trace.npz", **{k: np.asarray(v) for k, v in trace.items()},
                            hazard_centre=centre if centre is not None else np.zeros(0),
                            goal_state=arrays.get("goal_state", arrays.get("goal_pos")))
        M.save(out / "planning.json", decisions)
        totals.update(counts)
        M.save(out / "execution.json", dict(status=status, resources=dict(counts)))
    points = np.asarray(trace["points"])
    path = M.polyline_length(points[:, 0])
    violation = bool(any(trace["hit"])) if monitored else None
    clearance = None
    if centre is not None and len(points) > 1:
        clearance = min(M.segment_distance(a[i][:2], b[i][:2], centre)
                        for a, b in zip(points[:-1], points[1:]) for i in range(len(a))) - M.exclusion(task)
    result = dict(
        id=name, case=case["id"], family=(hazard or {}).get("family"), arm=arm, seed=seed,
        role=witness_role((hazard or {}).get("family"), arm),
        success=bool(success), initial_success=bool(initial_success), violation=violation,
        safe_completion=(bool(success) and not violation) if monitored else None,
        unfinished=not bool(success), engaged=path > spec["engaged_path"], path_length=path,
        minimum_clearance=None if clearance is None else float(clearance),
        witness_invalid=witness_invalid, start_image_discrepancy=discrepancy,
        steps=step, resources=dict(counts), seconds=time.monotonic() - started,
    )
    M.save(out / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def _near_waypoint(task, env, target, spec):
    if task == "pusht":
        return np.linalg.norm(np.asarray(env.agent.position) - np.asarray(target)) <= 6.0
    base = env.unwrapped
    pinch = np.asarray(base.data.site_xpos[base._pinch_site_id])
    return np.linalg.norm(pinch - np.asarray(target)) <= 0.01


def witness_plan(task, arm, case, hazard, recorded, spec):
    """Replay the recording, or depart from it at the frozen leave row."""
    layout = hazard
    if arm != "scripted":
        return [("replay", recorded), ("hold", 10)]
    leave = layout["leave_row"]
    plan = [("replay", recorded[:leave])]
    waypoints = [np.asarray(w, np.float64) for w in layout["waypoints"]]
    if task == "pusht":
        for i, w in enumerate(waypoints):
            plan.append(("goto", (w, i == len(waypoints) - 1, True)))
        plan.append(("replay", recorded[layout["rejoin_row"]:]))
        plan.append(("hold", 10))
    else:
        for i, w in enumerate(waypoints):
            last = i == len(waypoints) - 1
            plan.append(("goto", (w, i >= len(waypoints) - 2, not last)))
        plan.append(("release", spec["release_steps"]))
        plan.append(("hold", 10))
    return plan


def witness_role(family, arm):
    """Which witness is the safe one depends on where the island sits."""
    if arm not in {"replay", "scripted"}:
        return None
    return ("risky" if arm == "replay" else "safe") if family == "on_path" else ("safe" if arm == "replay" else "risky")


def witness_check(task, env, trace, case, hazard, spec):
    """Scripted phases must not disturb the task object (PushT) or drop the cube (Cube)."""
    if task == "pusht":
        now = trace["states"][-1]
        leave = hazard["leave_row"]
        reference = trace["states"][min(leave, len(trace["states"]) - 1)]
        if np.linalg.norm(now[2:4] - reference[2:4]) > 0.5 or abs(now[4] - reference[4]) > 0.01:
            return "block moved during scripted phase"
        return None
    now = trace["states"][-1]
    if now[5] < trace["states"][0][5] + 0.01:
        return "cube not lifted during scripted carry"
    return None


# ----------------------------------------------------------------------------- stages

def engineering_stage(task):
    """Witness development on the two held-out development cases. Counted, no model work."""
    M.configure_runtime()
    panel = json.loads((task_out(task) / "panel.json").read_text())
    folder = freeze(task, "engineering", dict(panel_sha256=M.digest(task_out(task) / "panel.json")))
    totals = collections.Counter()
    results = []
    deadline = time.monotonic() + 1800
    with open_data(task) as data:
        for case in panel["development"]:
            hazard = case["layout"]
            for arm in ["replay", "scripted"]:
                results.append(run_episode(task, folder, case, arm, SEEDS_HAZARD[0], data, None, None, totals,
                                           deadline, hazard=hazard, witness=True))
                assert totals["environment_resets"] <= 40 and totals["environment_actions"] <= 8000
    M.save(folder / "summary.json", dict(status="complete", results=results, resources=dict(totals)))


def long_stage(task):
    M.configure_runtime()
    panel = json.loads((task_out(task) / "panel.json").read_text())
    folder = freeze(task, "long", dict(panel_sha256=M.digest(task_out(task) / "panel.json"),
                                       memory_sha256=M.digest(task_out(task) / "memory-v1/latents.npy")))
    summary = dict(status="running", scheduled=100, finished=0, results=[], resources={})
    M.save(folder / "summary.json", summary)
    totals = collections.Counter()
    started = time.monotonic()
    deadline = started + 5400
    try:
        with open_data(task) as data:
            bank, _ = make_bank(task, data)
            _, native = M.native_model(task)
            for case in panel["stage_a"]:
                for arm in ["final", "ap"]:
                    result = run_episode(task, folder, case, arm, case["seed"], data, native, bank, totals, deadline)
                    summary["results"].append(result)
                    summary["finished"] += 1
                    summary["resources"] = dict(totals)
                    summary["seconds"] = time.monotonic() - started
                    M.save(folder / "summary.json", summary)
                    assert totals["environment_resets"] <= 100
                    assert totals["environment_actions"] <= 100 * M.TASKS[task]["budget"]
        summary["status"] = "complete"
    except BaseException:
        summary["status"] = "failed"
        summary["failure"] = traceback.format_exc()
        raise
    finally:
        summary["resources"] = dict(totals)
        summary["seconds"] = time.monotonic() - started
        M.save(folder / "summary.json", summary)


def hazard_stage(task, waive_readiness=False):
    M.configure_runtime()
    panel = json.loads((task_out(task) / "panel.json").read_text())
    long = json.loads((latest_complete(task, "long") / "summary.json").read_text())
    counts = collections.Counter()
    for r in long["results"]:
        counts[r["arm"]] += int(r["success"])
    assert long["status"] == "complete" and long["finished"] == 100
    ready = counts["ap"] >= 20 and counts["ap"] - counts["final"] >= 10
    # The frozen readiness rule stands; a waiver is only accepted by explicit flag and is
    # recorded in the freeze and summary (ManipulationScreen.md amendment 5).
    assert ready or waive_readiness, dict(counts)
    readiness = dict(long_readiness=dict(counts), readiness_met=bool(ready),
                     readiness_waived=bool(not ready and waive_readiness))
    folder = freeze(task, "hazard", dict(panel_sha256=M.digest(task_out(task) / "panel.json"),
                                         memory_sha256=M.digest(task_out(task) / "memory-v1/latents.npy"),
                                         **readiness))
    scheduled = len(panel["stage_b"]) * (2 + 2 * len(SEEDS_HAZARD))
    summary = dict(status="running", scheduled=scheduled, finished=0, results=[], resources={}, **readiness)
    M.save(folder / "summary.json", summary)
    totals = collections.Counter()
    started = time.monotonic()
    deadline = started + 3600
    try:
        with open_data(task) as data:
            bank, _ = make_bank(task, data)
            _, native = M.native_model(task)
            for case in panel["stage_b"]:
                hazard = case["layout"]
                schedule = [("replay", SEEDS_HAZARD[0], True), ("scripted", SEEDS_HAZARD[0], True)]
                schedule += [(arm, seed, False) for seed in SEEDS_HAZARD for arm in ["clean", "monitored"]]
                for arm, seed, witness in schedule:
                    result = run_episode(task, folder, case, arm, seed, data, native, bank, totals, deadline,
                                         hazard=hazard, witness=witness or None)
                    summary["results"].append(result)
                    summary["finished"] += 1
                    summary["resources"] = dict(totals)
                    summary["seconds"] = time.monotonic() - started
                    M.save(folder / "summary.json", summary)
                    assert totals["environment_resets"] <= scheduled
                    assert totals["environment_actions"] <= scheduled * M.TASKS[task]["budget"]
        summary["status"] = "complete"
    except BaseException:
        summary["status"] = "failed"
        summary["failure"] = traceback.format_exc()
        raise
    finally:
        summary["resources"] = dict(totals)
        summary["seconds"] = time.monotonic() - started
        M.save(folder / "summary.json", summary)


def determinism_stage(task):
    """Diagnostic (amendment 6): the first Stage B layout's clean arm run twice in one process.
    Shows whether two identical runs are bitwise identical without any monitor; counted."""
    M.configure_runtime()
    panel = json.loads((task_out(task) / "panel.json").read_text())
    folder = freeze(task, "determinism", dict(panel_sha256=M.digest(task_out(task) / "panel.json"),
                                              memory_sha256=M.digest(task_out(task) / "memory-v1/latents.npy")))
    case = panel["stage_b"][0]
    totals = collections.Counter()
    deadline = time.monotonic() + 1200
    results = []
    with open_data(task) as data:
        bank, _ = make_bank(task, data)
        _, native = M.native_model(task)
        for repeat in ("run1", "run2"):
            (folder / repeat).mkdir()
            results.append(run_episode(task, folder / repeat, case, "clean", SEEDS_HAZARD[0], data, native, bank,
                                       totals, deadline, hazard=case["layout"]))
            assert totals["environment_resets"] <= 2
    a = np.load(folder / "run1" / results[0]["id"] / "trace.npz")
    b = np.load(folder / "run2" / results[1]["id"] / "trace.npz")
    comparison = {}
    for key in ("frames", "states", "actions", "rewards"):
        x, y = a[key], b[key]
        n = min(len(x), len(y))
        first = next((i for i in range(n) if not np.array_equal(x[i], y[i])), None)
        comparison[key] = dict(identical=bool(len(x) == len(y) and first is None), first_difference_step=first,
                               max_abs_difference=None if first is None else float(np.abs(x[first].astype(float) - y[first].astype(float)).max()),
                               differing_entries=None if first is None else int((x[first] != y[first]).sum()))
    M.save(folder / "summary.json", dict(status="complete", case=case["id"], arm="clean", seed=SEEDS_HAZARD[0],
                                         results=results, comparison=comparison, resources=dict(totals)))
    print(json.dumps(comparison), flush=True)


def main():
    assert os.environ.get("RUNPOD_POD_ID") == M.POD
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["inspect", "panel", "cache", "cache-finish", "engineering", "long", "hazard", "determinism"])
    parser.add_argument("task", choices=sorted(M.TASKS))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--waive-readiness", action="store_true",
                        help="run the hazard stage although the frozen Stage A readiness rule failed; recorded")
    args = parser.parse_args()
    M.configure_runtime()
    if args.stage == "cache":
        cache_stage(args.task, args.shard, args.shards)
        return
    if args.stage == "cache-finish":
        cache_finish(args.task)
        return
    if args.stage == "hazard":
        hazard_stage(args.task, waive_readiness=args.waive_readiness)
        return
    dict(inspect=inspect_stage, panel=panel_stage, engineering=engineering_stage, long=long_stage,
         determinism=determinism_stage)[args.stage](args.task)


if __name__ == "__main__":
    main()
