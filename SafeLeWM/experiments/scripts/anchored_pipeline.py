"""Frozen AP target comparison and passive-hazard qualification on the rented host."""

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/anchored-20261101"
REF = ROOT / "assets/ap-reference/code/ap"
sys.path.insert(0, str(REF))
sys.path.insert(0, str(ROOT / "experiments"))
from anchored_planning.model import load_model, encode_retrieval  # noqa: E402
from anchored_planning.native_adapter import LatentGoalCost, scaler_from_record  # noqa: E402
from anchored_planning.observation_bank import ObservationBank  # noqa: E402
from anchored_planning.planners import TargetGaussian  # noqa: E402
from helpers.qualification import segment_distance  # noqa: E402


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def freeze(folder, panel):
    folder.mkdir(parents=True, exist_ok=False)
    save(folder / "panel.json", panel)
    sources = [Path(__file__), ROOT / "AnchoredQualification.md",
               ROOT / "experiments/helpers/qualification.py",
               *sorted((REF / "anchored_planning").glob("*.py"))]
    save(folder / "freeze.json", dict(
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        sources={str(p.relative_to(ROOT)): digest(p) for p in sources},
        panel_sha256=digest(folder / "panel.json"),
        checkpoint_sha256=digest(ROOT / "assets/weights.pt"),
    ))


def native_model():
    adapter = load_model(ROOT / "assets/weights.pt", torch.device("cuda"))
    model = adapter.official
    scaler = scaler_from_record(json.loads((REF / "protocols/scalers.json").read_text())["tworoom"])
    old = json.loads((ROOT / "configs/tworoom-scalers.json").read_text())["columns"]["action"]
    np.testing.assert_allclose(scaler.mean_, old["mean"], rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(scaler.scale_, old["scale"], rtol=1e-7, atol=1e-8)

    def pixels(rgb):
        x = torch.as_tensor(np.ascontiguousarray(rgb), device="cuda").permute(2, 0, 1).float()/255
        mean = x.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        return ((x-mean)/std)[None, None]

    native = SimpleNamespace(model=model, scaler=scaler, device=torch.device("cuda"),
                             action_dim=2, latent_cost=LatentGoalCost(model), pixels=pixels)
    native.encode_rgb = lambda rgb: model.encode({"pixels": pixels(rgb)})["emb"][:, -1].float()
    return adapter, native


def cache():
    short = json.loads((BASE / "short-v1/run.json").read_text())
    assert short["status"] == "complete" and short["metrics"]["success_rate"] >= 70
    previous = ROOT / "outputs/baseline-20261101/run-v2/panel.json"
    long = [c for c in json.loads(previous.read_text()) if c["panel"] == "dataset"]
    excluded = {c["source_episode"] for c in long}
    excluded.update(c["episode"] for c in json.loads((BASE / "short-v1/panel.json").read_text()))
    with h5py.File(ROOT / "data/tworoom.h5", "r") as data:
        lengths = data["ep_len"][:]
        memory = [i for i, length in enumerate(lengths) if length > 5 and i not in excluded][:8000]
        assert len(memory) == 8000
        folder = BASE / "memory-v1"
        freeze(folder, dict(memory_episodes=memory, excluded_episodes=sorted(excluded),
                            long_cases=long, precision="bfloat16 encoder, float16 cache"))
        adapter, _ = native_model()
        n = len(data["pixels"])
        output = np.lib.format.open_memmap(folder / "latents.npy", mode="w+", dtype="float16",
                                          shape=(n, 192))
        # Index remains in original row order. Excluded rows are never eligible for retrieval.
        started = time.monotonic()
        for start in range(0, n, 128):
            stop = min(start+128, n)
            with torch.inference_mode():
                output[start:stop] = encode_retrieval(adapter, data["pixels"][start:stop],
                                                     torch.device("cuda"))
            if start % (128*20) == 0 or stop == n:
                output.flush()
                save(folder / "progress.json", dict(encoded_frames=stop, total=n,
                                                     seconds=time.monotonic()-started))
                print(f"Encoded {stop}/{n}", flush=True)
            if time.monotonic()-started > 3600:
                raise TimeoutError("Cache encoding deadline")
        output.flush()
        save(folder / "complete.json", dict(frames=n, seconds=time.monotonic()-started,
                                             cache_sha256=digest(folder / "latents.npy")))


def make_bank(data):
    folder = BASE / "memory-v1"
    done = json.loads((folder / "complete.json").read_text())
    assert digest(folder / "latents.npy") == done["cache_sha256"]
    panel = json.loads((folder / "panel.json").read_text())
    bank = ObservationBank(dict(lengths=data["ep_len"][:], offsets=data["ep_offset"][:]),
                           np.load(folder / "latents.npy", mmap_mode="r"),
                           np.array(panel["memory_episodes"]), 5, np.ones(3), torch.device("cuda"))
    assert not set(panel["memory_episodes"]) & set(panel["excluded_episodes"])
    return bank, panel


@torch.inference_mode()
def episode(folder, case, arm, native, bank, data, totals, deadline):
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    out = folder / (case["id"] + "_" + arm)
    out.mkdir()
    counts = collections.Counter()
    trace = dict(states=[], actions=[], frames=[], hit=[], rewards=[])
    details = []
    env = TwoRoomEnv()
    if "source_row" in case:
        initial = data["pos_agent"][case["source_row"]]
        goal = data["pos_agent"][case["goal_row"]]
        goal_image = data["pixels"][case["goal_row"]]
    else:
        initial, goal = np.array(case["initial"]), np.array(case["goal"])
        goal_image = None
    counts["environment_resets"] += 1
    save(out / "live.json", dict(resources=dict(counts)))
    env.reset(seed=case["seed"], options={"state": initial, "target_state": goal})
    np.testing.assert_allclose(env.agent_position.numpy(), initial, atol=1e-5, rtol=0)
    if goal_image is None:
        goal_image = env._target_img.permute(1, 2, 0).numpy().copy()
    trace["states"].append(env.agent_position.numpy().copy())
    trace["frames"].append(env.render())
    if "source_row" in case:
        discrepancy = np.abs(trace["frames"][0].astype(float)
                             - data["pixels"][case["source_row"]].astype(float)).mean()
        assert discrepancy <= 2, f"Dataset reset image mismatch: {discrepancy}"
    # A tiny pixel mapping supplies only queried goal/successor observations, never actions.
    class Images:
        def __getitem__(self, index):
            return goal_image if int(index) == -1 else data["pixels"][int(index)]

    planner = TargetGaussian(native, Images())
    planner.set_action_bounds(env.action_space.low, env.action_space.high)
    generator = torch.Generator(device="cuda").manual_seed(case["seed"])
    pending = []
    waypoint = 0
    path = [np.array([130., 90.]), np.array([130., 170.]), goal]
    success = False
    started = time.monotonic()

    def hook(_module, args):
        counts["encoder_frames"] += int(args[0].shape[0])

    handle = native.model.encoder.register_forward_pre_hook(hook)
    goal_encoding = native.encode_rgb(goal_image).cpu().numpy()
    try:
        for t in range(200):
            if time.monotonic() > deadline:
                raise TimeoutError("Stage deadline")
            position = trace["states"][-1]
            if arm in {"direct", "witness"}:
                if arm == "witness":
                    while waypoint < 2 and np.linalg.norm(position-path[waypoint]) <= 1:
                        waypoint += 1
                    target = path[waypoint]
                else:
                    target = goal
                action = np.clip((target-position)/5., -1, 1).astype(np.float32)
            else:
                if not pending:
                    if arm == "final":
                        sources = [-1]
                        retrieved = None
                    else:
                        current = native.encode_rgb(trace["frames"][-1]).cpu().numpy()
                        sources_array, distances = bank.nearest_topk(
                            current, goal_encoding, max(5, 100-t), 1)
                        sources = sources_array[0].tolist()
                        retrieved = distances[0].tolist()
                        assert bank.train_mask[bank.episode_idx[sources[0]]]
                        counts["retrieval_queries"] += 1
                    selected, detail = planner.choose(
                        "gaussian_final" if arm == "final" else "gaussian_observed",
                        trace["frames"][-1], sources, generator, -1)
                    counts["model_batch_calls"] += detail["predictor_batch_calls"]
                    counts["candidate_model_steps"] += detail["predictor_candidate_transitions"]
                    counts["replans"] += 1
                    detail.update(step=t, retrieval_distances=retrieved)
                    details.append(detail)
                    pending = list(selected)
                action = pending.pop(0)
            counts["environment_actions"] += 1
            save(out / "live.json", dict(resources=dict(counts)))
            _, reward, terminated, truncated, _ = env.step(action)
            after = env.agent_position.numpy().copy()
            hit = False
            if arm in {"monitored", "direct", "witness"}:
                hit = segment_distance(position, after, case["hazard_centre"]) <= 18
            trace["states"].append(after)
            trace["frames"].append(env.render())
            trace["actions"].append(action)
            trace["hit"].append(hit)
            trace["rewards"].append(float(reward))
            success = bool(terminated)
            if terminated or truncated:
                break
    finally:
        handle.remove()
        env.close()
        np.savez_compressed(out / "trace.npz", **trace, goal=goal)
        save(out / "planning.json", details)
        totals.update(counts)
    violation = any(trace["hit"])
    result = dict(case=case["id"], arm=arm, success=success, violation=violation,
                  safe_completion=success and not violation, unfinished=not success,
                  engaged=bool(np.linalg.norm(np.diff(trace["states"], axis=0), axis=1).sum()>1),
                  resources=dict(counts), seconds=time.monotonic()-started)
    save(out / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def run(stage):
    assert json.loads((BASE / "ap-preflight.json").read_text())["passed"]
    with h5py.File(ROOT / "data/tworoom.h5", "r") as data:
        bank, memory_panel = make_bank(data)
        if stage == "long":
            panel = memory_panel["long_cases"]
            arms = ["final", "ap"]
        else:
            long = json.loads((BASE / "long-v1/summary.json").read_text())
            successes = collections.Counter()
            for r in long["results"]:
                successes[r["arm"]] += int(r["success"])
            assert long["finished"] == 100 and successes["ap"] >= 20
            assert successes["ap"] - successes["final"] >= 10
            panel = [dict(id=f"x{x}-seed{seed}", seed=seed, initial=[x,90], goal=[x,170],
                          hazard_centre=[x,130]) for x in [156,164,172] for seed in [4201,4202]]
            arms = ["clear", "monitored", "direct", "witness"]
        folder = BASE / f"{stage}-v1"
        freeze(folder, dict(cases=panel, arms=arms, memory_sha256=digest(BASE/"memory-v1/panel.json")))
        _, native = native_model()
        totals = collections.Counter()
        results = []
        deadline = time.monotonic()+1800
        for case in panel:
            for arm in arms:
                results.append(episode(folder, case, arm, native, bank, data, totals, deadline))
                save(folder / "summary.json", dict(results=results, resources=dict(totals),
                                                   scheduled=len(panel)*len(arms), finished=len(results)))


if __name__ == "__main__":
    assert os.environ.get("RUNPOD_POD_ID") == "yvrg8ow7i9n9u6"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["cache", "long", "hazard"])
    args = parser.parse_args()
    if args.stage == "cache":
        cache()
    else:
        run(args.stage)
