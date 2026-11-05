"""Compare passive recording with the native environment on two memory cases/task."""

import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
import time

from common import (Ledger, TASKS, assets, configuration, configure, provenance, publish,
                    read, serial, sha, write)
from instrumentation import Observer, Recorder, publish_failure


def instrumentation_hashes():
    root = Path(__file__).parent
    return {name: sha(root / name) for name in ("common.py", "instrumentation.py",
                                               "engineering.py", "evaluate.py")}


def cem_smoke(config, task, asset, protocol, data, probe, observer, ledger, attempt, ap_root):
    """Two planning calls on a memory case; proposed actions are never executed."""
    import numpy as np
    import torch
    from anchored_planning.model import encode_retrieval, load_model
    from anchored_planning.native_adapter import NativeAdapter
    from anchored_planning.observation_bank import ObservationBank
    from anchored_planning.planners import TargetGaussian

    identity = f"engineering-cem:{task}:attempt{attempt}"
    ledger.reserve(identity, "engineering", predictor_calls=62, predictor_candidate_blocks=18004)
    observer.case, observer.case_limits = Counter(), {
        "predictor_calls": 62, "predictor_candidate_blocks": 18004}
    completed = False
    timings = []
    try:
        device = torch.device("cuda:0")
        model = load_model(asset["checkpoint"], device)
        context = {"task": task, "device": device, "model": model}
        native = NativeAdapter(None, context, task,
                               {"native_action_stats": read(ap_root / "protocols/scalers.json")},
                               ap_root)
        bank = ObservationBank({"lengths": np.asarray(data["ep_len"], np.int64),
                                "offsets": np.asarray(data["ep_offset"], np.int64)},
                               np.load(asset["latent_cache"], mmap_mode="r", allow_pickle=False),
                               np.asarray(protocol["bank_train_episodes"][task], np.int64),
                               5, np.ones(3, np.float32), device)
        gaussian = TargetGaussian(native, data["pixels"])
        gaussian.set_action_bounds(probe["low"], probe["high"])
        with torch.inference_mode():
            current = encode_retrieval(model, probe["rgb"][None], device)
            goal = encode_retrieval(model, np.asarray(data["pixels"][probe["goal"]])[None], device)
            support, _ = bank.nearest_topk(current, goal, protocol["studies"][task]["horizon"], 8)
            for arm in ("gaussian_final", "gaussian_observed"):
                before = dict(observer.case)
                generator = torch.Generator(device=device).manual_seed(900001 + TASKS.index(task))
                torch.cuda.synchronize()
                start = time.monotonic()
                proposed, detail = gaussian.choose(arm, probe["rgb"], support[0], generator,
                                                   probe["goal"])
                torch.cuda.synchronize()
                elapsed = time.monotonic() - start
                calls = observer.case["predictor_calls"] - before.get("predictor_calls", 0)
                blocks = observer.case["predictor_candidate_blocks"] - before.get(
                    "predictor_candidate_blocks", 0)
                if (calls, blocks) != (31, 9002):
                    raise AssertionError("Engineering CEM work differs from the native contract")
                if not np.isfinite(proposed).all():
                    raise AssertionError("Engineering CEM proposed non-finite actions")
                timings.append({"arm": arm, "seconds": elapsed, "predictor_calls": calls,
                                "predictor_candidate_blocks": blocks,
                                "proposed_not_executed": proposed, "detail": detail})
        completed = True
        return serial({"memory_episode": probe["episode"], "goal_row": probe["goal"],
                       "optimizations": timings, "counts": dict(observer.case),
                       "additional_simulator_actions": 0})
    finally:
        ledger.finish(identity, observer.case, completed)
        publish("metrics/cem-smoke.json", {"completed": completed, "timings": timings,
                                           "counts": observer.case})
        observer.case, observer.case_limits = None, {}


def main(task):
    config = configuration()
    ap_root, runtime = configure(config, task)
    runtime.check_simulator_versions(task)
    from anchored_planning import environments as envs
    from anchored_planning.lifecycle import install_cube_renderer_cleanup
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    from types import SimpleNamespace

    if task == "cube":
        install_cube_renderer_cleanup(SimpleNamespace(np=np, runtime=envs))
    asset = assets(config, task)
    protocol = read(ap_root / "protocols/main.json")
    phase = "engineering"
    ledger = Ledger(config)
    observer = Observer(config, task)
    scratch = Path(os.environ["LEWM_RUN_DIR"]) / ".control/engineering"
    observer.install()
    attempt = int(config.get("engineering_attempt", 0))
    comparisons = []
    probe, numerical = None, None
    publish("artifacts/provenance.json", provenance(config, ap_root))
    try:
        with h5py.File(asset["dataset"], "r") as data:
            horizon = protocol["studies"][task]["horizon"]
            memory = sorted(protocol["bank_train_episodes"][task])
            # Goal row start+H and action rows start..start+9 only need this
            # many rows; adding ten to H excludes valid 101-row TwoRoom episodes.
            episodes = [e for e in memory if int(data["ep_len"][e]) > max(horizon, 9)][:2]
            if len(episodes) != 2:
                raise ValueError("Insufficient separate engineering memory episodes")
            forbidden = {r["episode"] for r in protocol["studies"][task]["confirm"]}
            if set(episodes) & forbidden:
                raise ValueError("Engineering touched a scientific query episode")
            for case_index, episode in enumerate(episodes):
                start = int(data["ep_offset"][episode])
                goal = start + horizon
                row = {"episode": episode, "global_start": start, "global_goal": goal,
                       "environment_seed": 600001 + TASKS.index(task) * 1000 + case_index,
                       "record_id": f"engineering-{task}-{case_index}"}
                states = {i: np.asarray(data[asset["state_key"]][i], np.float32)
                          for i in (start, goal)}
                actions = np.asarray(data["action"][start:start + 10], np.float32)
                paired = []
                for recorded in (False, True):
                    identity = f"engineering:{task}:{case_index}:{recorded}:attempt{attempt}"
                    ledger.reserve(identity, phase, resets=1, policy_actions=10)
                    observer.case, observer.phase = Counter(), "setup"
                    observer.add("resets_attempted")
                    env, completed = None, False
                    try:
                        env, obs, info, true_goal = envs._make_live(
                            {"task": task, "arrays": {"states": states}}, row)
                        observer.add("resets_completed")
                        initial = dict(observation=obs, info=info, true_goal=true_goal,
                                       identity=row, action_budget=10)
                        if recorded:
                            env = Recorder(env, task, row["record_id"] + f"-{attempt}", initial,
                                           np.asarray(data["pixels"][goal]), observer, scratch,
                                           config, video=True)
                        states_seen, images = [], []
                        rgb = np.ascontiguousarray(env.render())
                        images.append(hashlib.sha256(rgb.tobytes()).hexdigest())
                        if probe is None:
                            probe = {"rgb": rgb.copy(), "goal": goal, "episode": episode,
                                     "low": env.action_space.low.copy(),
                                     "high": env.action_space.high.copy()}
                        for index, raw in enumerate(actions):
                            action = np.clip(raw, env.action_space.low, env.action_space.high)
                            observer.phase = "policy"
                            if not recorded:
                                observer.guard()
                                observer.add("policy_actions_attempted")
                            result = env.step(action)
                            if not recorded:
                                observer.add("policy_actions_completed")
                            obs, reward, terminated, truncated, info = result
                            states_seen.append(serial({"observation": obs, "reward": reward,
                                "terminated": terminated, "truncated": truncated,
                                "info": info, "snapshot": envs._snapshot(task, env),
                                "success": envs._success_now(task, obs, info, true_goal,
                                                             False, False)}))
                            if (index + 1) % 5 == 0:
                                images.append(hashlib.sha256(np.ascontiguousarray(
                                    env.render()).tobytes()).hexdigest())
                            if terminated or truncated:
                                break
                        paired.append({"transitions": states_seen, "render_hashes": images})
                        completed = True
                    finally:
                        if env is not None:
                            env.close()
                        ledger.finish(identity, observer.case, completed)
                        publish(f"metrics/attempt-{case_index}-{int(recorded)}.json",
                                {"identity": identity, "completed": completed,
                                 "counts": observer.case})
                        observer.case = None
                if paired[0] != paired[1]:
                    raise AssertionError("Recorder changed native transitions, renders, or scoring")
                comparisons.append({"identity": row, "actions": len(paired[0]["transitions"]),
                                    "reserved_actions_per_replay": 10,
                                    "native_equals_recorded": True, "comparison": paired[0]})
            if config.get("engineering_cem_smoke", True):
                numerical = cem_smoke(config, task, asset, protocol, data, probe, observer,
                                      ledger, attempt, ap_root)
        gate = {"task": task, "passed": True, "comparisons": comparisons,
                "numerical_smoke": numerical,
                "counts": dict(observer.total), "source_hashes": instrumentation_hashes(),
                "ap_revision": config.get("ap_revision"), "versions": runtime.versions(),
                "source_run": Path(os.environ["LEWM_RUN_DIR"]).name,
                "elapsed_seconds": time.monotonic() - observer.started}
        write(Path(config["asset_root"]) / "gates" / (task + ".json"), gate)
        publish("artifacts/engineering-gate.json", gate)
    except Exception as error:
        publish_failure(observer, error)
        raise
    finally:
        observer.close()
        publish("metrics/accounting.json", {"counts": observer.total,
                "elapsed_seconds": time.monotonic() - observer.started})
        publish("metrics/study-attempt-ledger.json", read(ledger.path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=TASKS)
    main(parser.parse_args().task)
