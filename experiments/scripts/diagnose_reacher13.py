"""Planner diagnosis for task 13 (Reacher) before any hazard is designed.

Task 12 showed that a LeWM planner can be unable to reach goals the design assumes. This
script measures, with the frozen model and LeWM's planner settings and no hazard:
  1. success on dataset start/goal pairs 25 steps apart (LeWM's own protocol);
  2. success on fixed fingertip sweeps across the +x axis of growing size;
  3. whether our goal rendering matches the dataset's frames for the same state;
  4. that the exact arm model replays every executed step.
Writes outputs/reacher13/diagnosis/. Remote only (needs the GPU, checkpoint and dataset).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("STABLEWM_HOME", str(ROOT / "data/reacher13/stablewm"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.reacher13_protocol import design

OUT = ROOT / "outputs/reacher13/diagnosis"
DATA = ROOT / "data/reacher13"


def dataset_state_columns(f):
    """(qpos, qvel) readers for the dataset, whichever columns hold the arm state."""
    if "qpos" in f and "qvel" in f:
        return (lambda i: f["qpos"][i]), (lambda i: f["qvel"][i]), "qpos/qvel"
    if "observation" in f and f["observation"].shape[1] == 6:
        return (
            (lambda i: f["observation"][i][..., 0:2]),
            (lambda i: f["observation"][i][..., 4:6]),
            "observation[0:2], observation[4:6]",
        )
    if "state" in f and f["state"].shape[1] >= 4:
        return (lambda i: f["state"][i][..., 0:2]), (lambda i: f["state"][i][..., 2:4]), "state"
    raise ValueError(f"No arm-state columns among {list(f.keys())}")


def dataset_pairs(h5_path, n, offset, seed):
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np

    rng = np.random.default_rng(seed)
    pairs = []
    with h5py.File(h5_path, "r") as f:
        qpos, qvel, source = dataset_state_columns(f)
        lengths = np.asarray(f["ep_len"][:])
        offsets = np.asarray(f["ep_offset"][:])
        eligible = np.flatnonzero(lengths > offset + 1)
        for ep in rng.choice(eligible, size=n, replace=False):
            t = int(rng.integers(0, lengths[ep] - offset - 1))
            row = int(offsets[ep]) + t
            pairs.append(
                {
                    "episode": int(ep),
                    "t": t,
                    "row": row,
                    "start_state": [*map(float, qpos(row)), *map(float, qvel(row))],
                    "target_qpos": [float(x) for x in qpos(row + offset)],
                }
            )
        frames = {}
        if "pixels" in f:
            for k in (0, 1):
                frames[k] = np.asarray(f["pixels"][pairs[k]["row"]])
    return pairs, source, frames


def sweep_pairs(spec, sweeps, q2):
    out = []
    for half in sweeps:
        out.append(
            {"start_state": [half, q2, 0.0, 0.0], "target_qpos": [-half, q2], "sweep_rad": half}
        )
    return out


def free_job(spec, seed, pair, tag, **planner):
    params = dict(spec["planner"])
    params.update(planner)
    job = {
        "id": f"D-{tag}-{seed}",
        "stage": "D",
        "layout": None,
        "seed": int(seed),
        "mode": "penalty",
        "lam": 0.0,
        "clearances": [0.0] * int(params["horizon"]),
        "planner": params,
        "start_state": pair["start_state"],
        "target_qpos": pair["target_qpos"],
    }
    return job


def main():
    import numpy as np
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--sweeps", type=float, nargs="*", default=[0.25, 0.4, 0.55, 0.7])
    parser.add_argument("--q2", type=float, default=0.9)
    parser.add_argument("--sweep-seeds", type=int, default=4)
    parser.add_argument("--samples", type=int, default=None, help="override CEM samples")
    parser.add_argument("--topk", type=int, default=None, help="override CEM elites")
    parser.add_argument("--tag", default="lewm")
    args = parser.parse_args()
    spec = design()
    OUT.mkdir(parents=True, exist_ok=True)
    h5_path = next((DATA / "processed").rglob("*.h5"))
    offset = int(spec["environment"]["goal_offset_steps"])
    pairs, source, frames = dataset_pairs(h5_path, args.pairs, offset, seed=2026102513)
    from helpers.reacher13_runtime import Runtime

    runtime = Runtime(device="cuda", spec=spec)
    # Rendering check: our frame of a dataset state against the dataset's own frame.
    from helpers.reacher13_env import GoalReacherEnv

    render = {}
    env = GoalReacherEnv()
    for k, frame in frames.items():
        state = np.asarray(pairs[k]["start_state"])
        env.reset(seed=0, options={"state": state})
        ours = np.asarray(env.render())
        theirs = frame if frame.shape[-1] == 3 else np.moveaxis(frame, 0, -1)
        Image.fromarray(ours.astype(np.uint8)).save(OUT / f"render-ours-{k}.png")
        Image.fromarray(theirs.astype(np.uint8)).save(OUT / f"render-dataset-{k}.png")
        diff = np.abs(ours.astype(float) - theirs.astype(float))
        render[k] = {"mean_abs_diff": float(diff.mean()), "max_abs_diff": float(diff.max())}
    overrides = {}
    if args.samples:
        overrides["num_samples"] = args.samples
    if args.topk:
        overrides["topk"] = args.topk
    rows = []
    stream = (OUT / f"results-{args.tag}.jsonl").open("a")

    def episode(job, extra):
        tick = time.perf_counter()
        outcome = runtime.run(job)
        m = outcome["metrics"]
        row = {
            "id": job["id"],
            **extra,
            "seconds": time.perf_counter() - tick,
            **{k: m[k] for k in m if k not in ("planner", "execution")},
            "path": outcome["states"][:, 4:6].round(4).tolist(),
        }
        rows.append(row)
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in ("id", "success", "J", "final_joint_error_rad", "seconds")
                    if k in row
                }
            ),
            flush=True,
        )

    for i, pair in enumerate(pairs):
        job = free_job(spec, 100 + i, pair, f"{args.tag}-data{i:02d}", **overrides)
        episode(job, {"kind": "dataset", **pair})
    for pair in sweep_pairs(spec, args.sweeps, args.q2):
        for s in range(args.sweep_seeds):
            tag = f"{args.tag}-sweep{pair['sweep_rad']:g}"
            episode(free_job(spec, 200 + s, pair, tag, **overrides), {"kind": "sweep", **pair})
    data = [r for r in rows if r["kind"] == "dataset"]
    summary = {
        "utc": utc(),
        "tag": args.tag,
        "planner": {**spec["planner"], **overrides},
        "state_source": source,
        "render_check": render,
        "dataset_success": float(np.mean([r["success"] for r in data])) if data else None,
        "dataset_start_target_m": [r["start_target_distance_m"] for r in data],
        "sweeps": {
            f"{half:g}": {
                "success": float(
                    np.mean([r["success"] for r in rows if r.get("sweep_rad") == half])
                ),
                "start_target_m": next(
                    r["start_target_distance_m"] for r in rows if r.get("sweep_rad") == half
                ),
            }
            for half in args.sweeps
        },
        "max_step_m": float(max(r["max_step_m"] for r in rows)),
        "median_step_m": float(np.median([r["median_step_m"] for r in rows])),
        "exact_replay_max_m": float(max(r["exact_replay_error_m"]["max"] for r in rows)),
        "episodes": len(rows),
    }
    atomic_json(OUT / f"summary-{args.tag}.json", summary)
    print(
        json.dumps({"summary": {k: v for k, v in summary.items() if k != "dataset_start_target_m"}})
    )


if __name__ == "__main__":
    main()
