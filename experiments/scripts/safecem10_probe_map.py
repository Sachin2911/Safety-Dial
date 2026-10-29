"""Read-only diagnostic: dataset pusher coverage and probe error on env-rendered frames."""

import json
import os
import sys

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"
sys.path.insert(0, "experiments")
import numpy as np
import torch
import h5py
import hdf5plugin  # noqa: F401
from PIL import Image
from helpers.safecem10_runtime import load_world_model, load_probe
from helpers.safecem10_planner import IMAGENET_MEAN, IMAGENET_STD, make_state
import stable_worldmodel as swm

out = "outputs/safecem10/diag"
os.makedirs(out, exist_ok=True)
torch.set_num_threads(2)
dev = "cpu"
model = load_world_model(dev)
probe, _ = load_probe(dev)
mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


def read(frames):
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    x = (x - mean) / std
    with torch.inference_mode():
        z = model.encode({"pixels": x.unsqueeze(0)})["emb"][0]
        return probe(z.float()).numpy()


rep = {}
with h5py.File("data/safecem10/processed/pusht_expert_train.h5", "r") as f:
    st = f["state"][:, 0:2]
    rep["dataset_pusher_pct"] = {
        str(q): np.percentile(st, q, axis=0).round(1).tolist()
        for q in (0.1, 1, 5, 50, 95, 99, 99.9)
    }
    rep["frac_within_40px_of_start"] = float((np.hypot(st[:, 0] - 40, st[:, 1] - 40) < 40).mean())
    rep["frac_x_lt_60_and_y_lt_60"] = float(((st[:, 0] < 60) & (st[:, 1] < 60)).mean())
    rep["frac_in_wall_vicinity"] = float(
        ((st[:, 1] > 120) & (st[:, 1] < 260) & (st[:, 0] < 330)).mean()
    )
    H, xe, ye = np.histogram2d(st[:, 0], st[:, 1], bins=8, range=[[0, 512], [0, 512]])
    rep["coverage_8x8_per_mille_rows_x_cols_y"] = (1000 * H / H.sum()).round(1).tolist()
    blk = f["state"][:, 2:5]
    rep["dataset_block_pct"] = {
        str(q): np.percentile(blk, q, axis=0).round(2).tolist() for q in (1, 50, 99)
    }
    idx = np.linspace(0, len(st) - 1, 6).astype(int)
    dframes = np.stack([f["pixels"][i] for i in idx])
    dstates = f["state"][idx][:, 0:2]
rep["dataset_frames_probe_error_px"] = (
    np.linalg.norm(read(dframes) - dstates, axis=1).round(1).tolist()
)
Image.fromarray(dframes[2]).save(f"{out}/dataset_frame.png")

w = swm.World("swm/PushT-v1", num_envs=1, image_shape=(224, 224), max_episode_steps=100)
goal = make_state((250, 310), (360, 310), np.pi / 4)
grid = list(range(20, 500, 40))
frames, truth = [], []
for py in grid:
    for px in grid:
        w.reset(
            seed=0,
            options={"state": make_state((px, py), (300, 310), np.pi / 4), "goal_state": goal},
        )
        frames.append(np.asarray(w.infos["pixels"][0, 0]).copy())
        truth.append(np.asarray(w.infos["state"][0, 0][:2]).copy())
frames = np.stack(frames)
truth = np.stack(truth)
err = np.linalg.norm(read(frames) - truth, axis=1).reshape(len(grid), len(grid))
rep["env_probe_error_grid_rows_y_cols_x"] = {
    "coords": grid,
    "error_px": err.round(0).astype(int).tolist(),
}
w.reset(seed=0, options={"state": make_state((40, 40), (300, 310), np.pi / 4), "goal_state": goal})
start_frame = np.asarray(w.infos["pixels"][0, 0]).copy()
Image.fromarray(start_frame).save(f"{out}/env_start_frame.png")
rep["env_start_probe"] = read(start_frame[None])[0].round(1).tolist()
Image.fromarray(np.asarray(w.infos["goal"][0, 0]).copy()).save(f"{out}/env_goal_frame.png")
json.dump(rep, open(f"{out}/probe-map.json", "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "coverage_8x8_per_mille_rows_x_cols_y"}))
print("coverage 8x8 (per mille; rows = x bins of 64 px, cols = y bins):")
for row in rep["coverage_8x8_per_mille_rows_x_cols_y"]:
    print("  ", row)
