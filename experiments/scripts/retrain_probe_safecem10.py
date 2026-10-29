"""Retrain the pusher probe on simulator frames that cover the arena, plus a wider dataset sample.

Stage B of task 10 showed that the September probe misreads the pusher by up to 200 px in
the start corner and by 15 to 40 px along the wall, regions the expert data barely visits.
Probe v2 keeps the architecture and is trained on (a) frames rendered through the same
World pipeline the planner sees, with the pusher uniform over the arena and the block near
the episode start, along the push, or anywhere, and (b) dataset frames from runs spread over
the whole file. Both probes are evaluated on held-out dataset runs, held-out simulator
frames and the real stage A and B closed-loop states (re-rendered from the saved simulator
states, never used for training). The imagined-error quantiles are recomputed on the same
1500 dataset snippets as the preparation step, for the v1 and v2 probes with and without
anchoring, to give the stage B2 dials in uncertainty units.

Run once on the rental after the preparation step. Never overwrites probe-v2.pt.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments/scripts"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
from helpers.dsrl06_infra import atomic_json, utc
from helpers.safecem10_protocol import design, digest

DATA = ROOT / "data/safecem10"
OUT = ROOT / "outputs/safecem10"
os.environ.setdefault("STABLEWM_HOME", str(DATA / "stablewm"))
REGIONS = ("corner", "wall", "other")
_WORLD = None


def region_of(xy):
    import numpy as np

    xy = np.asarray(xy, dtype=float)
    corner = np.linalg.norm(xy - np.array([40.0, 40.0]), axis=1) < 80.0
    wall = (xy[:, 1] >= 150.0) & (xy[:, 1] <= 270.0) & ~corner
    return np.where(corner, "corner", np.where(wall, "wall", "other"))


def sample_env_states(n, cfg, spec, rng):
    """Pusher uniform over the arena; block from the frozen mixture; zero velocities."""
    import numpy as np

    env = spec["environment"]
    lo, hi = cfg["pusher_range"]
    mix = cfg["block_mixture"]
    names = list(mix)
    weights = np.array([mix[k]["weight"] for k in names], dtype=float)
    kinds = rng.choice(len(names), size=n, p=weights / weights.sum())
    states = np.zeros((n, 7))
    states[:, 0:2] = rng.uniform(lo, hi, size=(n, 2))
    sx, sy = env["start_block"]
    theta0 = env["block_theta"]
    for i, k in enumerate(kinds):
        kind = names[k]
        m = mix[kind]
        if kind == "start":
            bx, by = rng.normal([sx, sy], m["xy_sd"])
            th = theta0 + rng.normal(0.0, m["theta_sd"])
        elif kind == "push_line":
            bx = rng.uniform(*m["x_range"]) + rng.normal(0.0, m["xy_sd"])
            by = sy + rng.normal(0.0, m["xy_sd"])
            th = theta0 + rng.normal(0.0, m["theta_sd"])
        else:
            bx, by = rng.uniform(*m["xy_range"], size=2)
            th = rng.uniform(0.0, 2 * np.pi)
        states[i, 2:5] = (bx, by, th)
    return states


def _render_init(spec):
    global _WORLD
    import stable_worldmodel as swm

    env = spec["environment"]
    _WORLD = swm.World(
        env["env_name"],
        num_envs=1,
        image_shape=(env["image_size"], env["image_size"]),
        max_episode_steps=int(env["max_episode_steps"]),
    )


def _inner_env(world):
    envs = world.envs
    for path in (("envs",), ("env", "envs"), ("unwrapped", "envs")):
        obj = envs
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return obj[0].unwrapped
        except (AttributeError, IndexError, TypeError):
            continue
    return None


def _render_chunk(payload):
    """Render states through the World pipeline; returns uint8 HWC frames and overlap flags."""
    import numpy as np

    states, goal_state, seed0, overlap_px = payload
    frames, overlaps = [], []
    for i, state in enumerate(states):
        _WORLD.reset(seed=int(seed0 + i), options={"state": state, "goal_state": goal_state})
        frame = np.asarray(_WORLD.infos["pixels"][0, -1])
        if frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[-1] != 3:
            frame = np.transpose(frame, (1, 2, 0))
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        frames.append(frame.copy())
        inner = _inner_env(_WORLD)
        flag = None
        if inner is not None and overlap_px is not None:
            try:
                p = (float(state[0]), float(state[1]))
                flag = bool(min(s.point_query(p).distance for s in inner.block.shapes) < overlap_px)
            except AttributeError:
                flag = None
        overlaps.append(flag)
    return np.stack(frames), overlaps


def render_embeddings(states, spec, workers, seed0, model, device, overlap_px=None, chunk=400):
    """Render states through the World pipeline in worker processes and encode each chunk as
    it arrives, so only the 192-d embeddings stay in memory. Order follows `states`."""
    import multiprocessing as mp

    import numpy as np
    from prepare_safecem10 import encode_frames

    env = spec["environment"]
    goal = np.array(
        [*env["goal_pusher"], *env["goal_block"], env["block_theta"], 0.0, 0.0], dtype=float
    )
    starts = list(range(0, len(states), chunk))
    payloads = [(states[i : i + chunk], goal, seed0 + i, overlap_px) for i in starts]
    Z = np.empty((len(states), 192), dtype=np.float32)
    overlaps = [None] * len(states)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_render_init, initargs=(spec,)) as pool:
        for i, (frames, flags) in zip(starts, pool.imap(_render_chunk, payloads)):
            Z[i : i + len(frames)] = encode_frames(model, frames, device)
            overlaps[i : i + len(frames)] = flags
    return Z, overlaps


def dataset_embeddings(h5_path, cfg, model, device, runs_per_batch=5):
    """Encode dataset runs spread over the whole file, run by run; keeps embeddings only."""
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    from prepare_safecem10 import encode_frames

    L = cfg["dataset_run_length"]
    with h5py.File(h5_path, "r") as f:
        total = len(f["pixels"])
        starts = np.linspace(0, total - L, cfg["dataset_runs"], dtype=int)
        Z = np.empty((len(starts) * L, 192), dtype=np.float32)
        xy = np.empty((len(starts) * L, 2), dtype=np.float32)
        run = np.repeat(np.arange(len(starts)), L)
        for b in range(0, len(starts), runs_per_batch):
            group = starts[b : b + runs_per_batch]
            frames = np.concatenate([f["pixels"][s : s + L] for s in group])
            Z[b * L : b * L + len(frames)] = encode_frames(model, frames, device)
            xy[b * L : b * L + len(frames)] = np.concatenate(
                [f["state"][s : s + L, 0:2] for s in group]
            )
    return Z, xy, run, int(total)


def closed_loop_states(n, rng):
    import numpy as np

    rows = []
    for path in sorted((OUT / "attempts").glob("[AB]-*/states.npz")):
        with np.load(path) as blob:
            rows.append(blob["states"])
    states = np.concatenate(rows)
    pick = rng.choice(len(states), size=min(n, len(states)), replace=False)
    return states[np.sort(pick)], len(rows), int(len(states))


def errors(probe, Z, Y, device):
    import numpy as np
    import torch

    with torch.inference_mode():
        pred = probe(torch.from_numpy(Z).to(device)).cpu().numpy()
    return np.linalg.norm(pred - Y, axis=1)


def summary(err, regions=None):
    import numpy as np

    def stats(e):
        if len(e) == 0:
            return {"n": 0}
        return {
            "n": int(len(e)),
            "median": float(np.median(e)),
            "p90": float(np.percentile(e, 90)),
            "p95": float(np.percentile(e, 95)),
            "mean": float(np.mean(e)),
        }

    out = {"all": stats(err)}
    if regions is not None:
        for r in REGIONS:
            out[r] = stats(err[regions == r])
    return out


def train(Z_env, Y_env, Z_ds, Y_ds, cfg, device):
    import numpy as np
    import torch

    from helpers.safecem10_planner import MLPProbe

    Z = np.concatenate([np.repeat(Z_env, cfg["env_oversample"], axis=0), Z_ds])
    Y = np.concatenate([np.repeat(Y_env, cfg["env_oversample"], axis=0), Y_ds])
    Zt = torch.from_numpy(Z).to(device)
    Yt = torch.from_numpy(Y.astype(np.float32)).to(device)
    torch.manual_seed(cfg["seed"])
    probe = MLPProbe(in_dim=Z.shape[1], hidden=tuple(cfg["hidden"])).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = int(np.ceil(len(Z) / cfg["batch"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["epochs"] * steps_per_epoch, eta_min=cfg["lr"] * 0.01
    )
    losses = []
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(len(Z), device=device)
        running = 0.0
        for i in range(0, len(Z), cfg["batch"]):
            b = idx[i : i + cfg["batch"]]
            loss = torch.nn.functional.mse_loss(probe(Zt[b]), Yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += float(loss.detach()) * len(b)
        losses.append(running / len(Z))
    return probe.eval(), losses, int(len(Z))


def imagination_variants(model, h5_path, spec, probes, device, n_snippets):
    """Imagined radial error at each depth on the preparation's snippets, per probe, with
    and without anchoring the current reading at the true current position."""
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    import torch

    from helpers.safecem10_planner import IMAGENET_MEAN, IMAGENET_STD
    from helpers.safecem10_runtime import load_scalers

    cfg = spec["imagination"]
    block = spec["planner"]["action_block"]
    depth = cfg["depth_blocks"]
    span = block * depth
    process, _ = load_scalers()
    action = process["action"]
    rng = np.random.default_rng(cfg["seed"])
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    keys = [f"{name}_{mode}" for name in probes for mode in ("raw", "anchored")]
    err = {k: np.full((n_snippets, depth), np.nan) for k in keys}
    with h5py.File(h5_path, "r") as f:
        lengths = f["ep_len"][:]
        offsets = f["ep_offset"][:]
        eligible = np.flatnonzero(lengths > span + 1)
        picks = rng.choice(eligible, size=cfg["n_snippets"], replace=True)[:n_snippets]
        for i, ep in enumerate(picks):
            start = int(offsets[ep]) + int(rng.integers(0, lengths[ep] - span - 1))
            frames = f["pixels"][start : start + span + 1]
            states = f["state"][start : start + span + 1]
            acts = np.asarray(f["action"][start : start + span], dtype=np.float64)
            acts = action.transform(acts).reshape(1, 1, depth, block * 2)
            with torch.inference_mode():
                x = torch.from_numpy(frames[0:1]).to(device).permute(0, 3, 1, 2).float() / 255.0
                x = ((x - mean) / std).unsqueeze(0).unsqueeze(0)
                info = model.rollout(
                    {"pixels": x}, torch.as_tensor(acts, device=device, dtype=x.dtype)
                )
                emb = info["predicted_emb"][0, 0].float()
                truth = states[block::block, 0:2]
                for name, probe in probes.items():
                    xy = probe(emb).cpu().numpy()
                    err[f"{name}_raw"][i] = np.linalg.norm(xy[1 : depth + 1] - truth, axis=1)
                    anchored = xy - xy[0:1] + states[0:1, 0:2]
                    err[f"{name}_anchored"][i] = np.linalg.norm(
                        anchored[1 : depth + 1] - truth, axis=1
                    )
    variants = {}
    for k, e in err.items():
        variants[k] = {
            "radial_quantiles_px": {
                str(q): float(np.percentile(e[:, depth - 1], q)) for q in cfg["quantiles"]
            },
            "per_depth": [
                {
                    "depth": d + 1,
                    "p50": float(np.percentile(e[:, d], 50)),
                    "p90": float(np.percentile(e[:, d], 90)),
                    "p95": float(np.percentile(e[:, d], 95)),
                }
                for d in range(depth)
            ],
        }
    return variants, err


def main():
    import numpy as np
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6, help="rendering processes")
    parser.add_argument("--smoke", action="store_true", help="tiny run into a scratch path")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Retraining requires the rental GPU")
    device = "cuda"
    spec = design()
    cfg = dict(spec["probe_v2"])
    n_snippets = spec["imagination"]["n_snippets"]
    tag = "probe-v2"
    if args.smoke:
        cfg.update(
            {
                "env_frames": 400,
                "dataset_runs": 20,
                "dataset_run_length": 50,
                "epochs": 2,
                "closed_loop_eval_states": 200,
            }
        )
        n_snippets = 40
        tag = "probe-v2-smoke"
    probe_path = DATA / f"{tag}.pt"
    report_path = OUT / f"{tag}.json"
    if probe_path.exists() or report_path.exists():
        raise FileExistsError(f"{probe_path.name} already exists; never overwrite a probe")
    timings = {}
    rng = np.random.default_rng(cfg["seed"])

    from prepare_safecem10 import fetch_dataset

    from helpers.safecem10_runtime import load_probe, load_world_model

    model = load_world_model(device)
    tick = time.time()
    n_env = int(cfg["env_frames"])
    candidates = sample_env_states(int(n_env * 1.25), cfg, spec, rng)
    Z_all, overlaps = render_embeddings(
        candidates, spec, args.workers, 10_000_000, model, device, cfg["reject_overlap_px"]
    )
    usable = np.array([flag is not True for flag in overlaps])
    keep = np.flatnonzero(usable)[:n_env]
    if len(keep) < n_env:
        raise RuntimeError("Too few non-overlapping simulator frames")
    env_states = candidates[keep]
    Z_env = Z_all[keep]
    del Z_all
    overlap_known = sum(flag is not None for flag in overlaps)
    cl_states, cl_files, cl_total = closed_loop_states(cfg["closed_loop_eval_states"], rng)
    Z_cl, _ = render_embeddings(cl_states, spec, args.workers, 20_000_000, model, device, None)
    timings["render_and_encode_env_seconds"] = time.time() - tick
    tick = time.time()
    h5_path = fetch_dataset(spec)
    Z_ds, ds_xy, ds_run, ds_total = dataset_embeddings(h5_path, cfg, model, device)
    timings["dataset_read_and_encode_seconds"] = time.time() - tick

    env_hold = rng.random(len(Z_env)) < cfg["env_holdout_fraction"]
    ds_hold = (ds_run % cfg["dataset_holdout_every"]) == 0
    Y_env = env_states[:, 0:2].astype(np.float32)
    tick = time.time()
    probe_v2, losses, n_train = train(
        Z_env[~env_hold], Y_env[~env_hold], Z_ds[~ds_hold], ds_xy[~ds_hold], cfg, device
    )
    timings["train_seconds"] = time.time() - tick
    probe_v1, v1_stats = load_probe(device)
    probes = {"v1": probe_v1, "v2": probe_v2}

    evaluation, per_sample = {}, {}
    sets = {
        "dataset_holdout_runs": (Z_ds[ds_hold], ds_xy[ds_hold]),
        "env_holdout": (Z_env[env_hold], Y_env[env_hold]),
        "closed_loop_stage_AB": (Z_cl, cl_states[:, 0:2].astype(np.float32)),
    }
    for set_name, (Z, Y) in sets.items():
        regions = region_of(Y)
        evaluation[set_name] = {}
        for name, probe in probes.items():
            e = errors(probe, Z, Y, device)
            evaluation[set_name][name] = summary(e, regions)
            per_sample[f"{set_name}__{name}"] = e
        per_sample[f"{set_name}__xy"] = Y

    tick = time.time()
    variants, imag_err = imagination_variants(model, h5_path, spec, probes, device, n_snippets)
    timings["imagination_seconds"] = time.time() - tick

    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in probe_v2.state_dict().items()},
            "in_dim": 192,
            "hidden": list(cfg["hidden"]),
            "stats": {
                "env_holdout": evaluation["env_holdout"]["v2"]["all"],
                "dataset_holdout_runs": evaluation["dataset_holdout_runs"]["v2"]["all"],
                "closed_loop_stage_AB": evaluation["closed_loop_stage_AB"]["v2"]["all"],
            },
            "recipe": cfg,
            "utc": utc(),
        },
        probe_path,
    )
    imagination_v2 = {
        "utc": utc(),
        "n_snippets": int(n_snippets),
        "depth_blocks": spec["imagination"]["depth_blocks"],
        "probe": "v2",
        "check": "anchored",
        "radial_quantiles_px": variants["v2_anchored"]["radial_quantiles_px"],
        "per_depth": variants["v2_anchored"]["per_depth"],
        "variants": variants,
    }
    atomic_json(OUT / f"imagination-{tag.replace('probe-', '')}.json", imagination_v2)
    np.savez_compressed(
        OUT / f"{tag}-eval.npz",
        **per_sample,
        **{f"imagination__{k}": v for k, v in imag_err.items()},
    )
    report = {
        "utc": utc(),
        "probe_path": str(probe_path.relative_to(ROOT)),
        "probe_sha256": digest(probe_path),
        "recipe": cfg,
        "counts": {
            "env_frames": int(len(Z_env)),
            "env_holdout": int(env_hold.sum()),
            "env_candidates_rendered": int(len(candidates)),
            "env_overlap_rejected": int(sum(flag is True for flag in overlaps)),
            "env_overlap_checked": int(overlap_known),
            "dataset_frames": int(len(Z_ds)),
            "dataset_holdout_frames": int(ds_hold.sum()),
            "dataset_total_frames": ds_total,
            "train_rows_after_oversampling": n_train,
            "closed_loop_states": int(len(Z_cl)),
            "closed_loop_attempt_files": cl_files,
            "closed_loop_states_available": cl_total,
        },
        "v1_preparation_stats": v1_stats,
        "train_loss_by_epoch": losses,
        "evaluation": evaluation,
        "imagination_depth5_quantiles": {k: v["radial_quantiles_px"] for k, v in variants.items()},
        "timings": timings,
        "gpu": torch.cuda.get_device_name(),
    }
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "status": "probe_v2_ready",
                "closed_loop": {k: v["all"] for k, v in evaluation["closed_loop_stage_AB"].items()},
                "imagination": report["imagination_depth5_quantiles"],
            }
        )
    )


if __name__ == "__main__":
    main()
