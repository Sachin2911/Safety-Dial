"""Encode the authors' demonstrations with the frozen world model into a retrieval library.

Runs on the GPU without the simulator. For every episode file the recorded
observations are passed through the frozen encoder and the recurrent posterior
(mode, no sampling) along the recorded actions, exactly as the online controller
tracks its filter latent. Features are stored per step with the actions that follow.

    /venv/main/bin/python SafeBlock/experiments/encode_library.py \
        --dataset /workspace/datasets/unisafe/extracted_success/successonly \
        --output SafeBlock/data/safeblock/library/success_only.npz
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from probe_common import atomic_json, project_path, utc  # noqa: E402

from helpers.unisafe09_model import load_frozen  # noqa: E402


def episode_observation(episode):
    T = len(episode["action"])
    obs = {
        "front_cam": episode["front_cam"],
        "wrist_cam": episode["wrist_cam"],
        "eef_pos": episode["eef_pos"],
        "eef_quat": episode["eef_quat"],
        "is_first": episode["is_first"],
        "is_last": episode["is_last"],
        "is_terminal": episode["is_terminal"],
        "failure": episode["failure"],
        "success": episode["success"],
    }
    for key, value in obs.items():
        if len(value) != T:
            raise ValueError(f"Field {key} has {len(value)} rows for {T} actions")
    # The recurrent state starts from zeros at the first frame regardless of the
    # stored flag (obs_step with no previous state), so force the flag and record
    # the anomaly instead of failing.
    anomalies = []
    if obs["is_first"][0] != 1.0:
        anomalies.append("is_first_missing")
        obs["is_first"] = obs["is_first"].copy()
        obs["is_first"][0] = 1.0
    if not np.allclose(episode["action"][0], 0.0):
        anomalies.append("nonzero_first_action")
    if (obs["is_first"][1:] > 0.5).any():
        anomalies.append("is_first_inside_episode")
    return obs, anomalies


@torch.inference_mode()
def encode_episode(wm, episode, device):
    obs, anomalies = episode_observation(episode)
    data = wm.preprocess(obs)
    embed = wm.encoder(data)
    actions = torch.tensor(episode["action"], device=device, dtype=torch.float32)
    feats = []
    state = None
    for t in range(len(actions)):
        state, _ = wm.dynamics.obs_step(
            state, actions[t : t + 1], embed[t : t + 1], data["is_first"][t : t + 1], sample=False
        )
        feats.append(wm.dynamics.get_feat(state)[0])
    return torch.stack(feats), anomalies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output = project_path(args.output)
    files = sorted(args.dataset.glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit("No episode files found")
    started = time.perf_counter()
    agent, _, config, provenance = load_frozen(device=args.device)
    wm = agent._wm
    feats, actions, eef, success, failure, ep, step = [], [], [], [], [], [], []
    episodes = []
    for index, path in enumerate(files):
        with np.load(path) as loaded:
            episode = {k: loaded[k] for k in loaded.files}
        f, anomalies = encode_episode(wm, episode, args.device)
        T = f.shape[0]
        first_success = np.nonzero(episode["success"] > 0.5)[0]
        episodes.append(
            {
                "index": index,
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "length": int(T),
                "start": int(sum(e["length"] for e in episodes)),
                "first_success_step": int(first_success[0]) if len(first_success) else -1,
                "failure_steps": int((episode["failure"] > 0.5).sum()),
                "final_eef_pos": episode["eef_pos"][-1].tolist(),
                "anomalies": anomalies,
            }
        )
        feats.append(f.to(torch.float16).cpu().numpy())
        actions.append(episode["action"].astype(np.float32))
        eef.append(episode["eef_pos"].astype(np.float32))
        success.append((episode["success"] > 0.5).astype(np.bool_))
        failure.append((episode["failure"] > 0.5).astype(np.bool_))
        ep.append(np.full(T, index, dtype=np.int32))
        step.append(np.arange(T, dtype=np.int32))
        if index % 100 == 0:
            print(f"[{utc()}] {index + 1}/{len(files)} episodes, {time.perf_counter() - started:.0f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        feats=np.concatenate(feats),
        actions=np.concatenate(actions),
        eef_pos=np.concatenate(eef),
        success=np.concatenate(success),
        failure=np.concatenate(failure),
        ep=np.concatenate(ep),
        step=np.concatenate(step),
        ep_start=np.array([e["start"] for e in episodes], dtype=np.int64),
        ep_len=np.array([e["length"] for e in episodes], dtype=np.int64),
    )
    manifest = {
        "utc": utc(),
        "dataset": str(args.dataset),
        "episodes": len(episodes),
        "steps": int(sum(e["length"] for e in episodes)),
        "feature_dim": int(feats[0].shape[1]),
        "with_success_flag": int(sum(e["first_success_step"] >= 0 for e in episodes)),
        "with_failure_flag": int(sum(e["failure_steps"] > 0 for e in episodes)),
        "with_anomalies": int(sum(bool(e["anomalies"]) for e in episodes)),
        "encoding": "frozen encoder + posterior mode along recorded actions (sample=False)",
        "model_provenance": provenance,
        "elapsed_seconds": time.perf_counter() - started,
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "episode_table": episodes,
    }
    atomic_json(args.output.with_suffix(".json"), manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "episode_table"}, indent=2))


if __name__ == "__main__":
    main()
