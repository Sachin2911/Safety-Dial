"""Audit release geometry and inspect learned reward/continuation on a verified replay."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from helpers.unisafe_controller import FrozenController
from helpers.unisafe_model import load_frozen
from helpers.unisafe_source import save_implementation


def reduced_success(state):
    c = np.asarray(state["cube_positions"])

    def stacked(i, j):
        d = c[i] - c[j]
        height = abs(d[2])
        return np.linalg.norm(d[:2]) < 0.05 and abs(height - 0.03) < 0.01 and height > 0.014

    return bool(stacked(2, 0) and not stacked(1, 0) and not stacked(2, 1) and c[1, 2] >= 0.13)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--episode", default="seed-001-filtered")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    geometry = []
    for path in sorted(args.baseline.glob("seed-*-*")):
        rows = [json.loads(line) for line in (path / "steps.jsonl").read_text().splitlines()]
        summary = json.loads((path / "summary.json").read_text())
        first = next((i for i, r in enumerate(rows) if reduced_success(r["after"])), None)
        closed, openings = 0, []
        for i, r in enumerate(rows):
            if r["proposed_action"][-1] < 0:
                closed += 1
            else:
                if closed:
                    openings.append({
                        "step": i, "prior_closed_commands": closed,
                        "middle_base_xy_m": r["before"]["middle_base_xy_distance_m"],
                        "eef_middle_distance_m": r["before"]["eef_middle_distance_m"],
                        "finger_positions": r["before"]["gripper_pos"],
                        "reduced_success_before_opening": reduced_success(r["before"]),
                    })
                closed = 0
        geometry.append({
            "episode": path.name, "mode": summary["mode"], "outcome": summary["outcome"],
            "first_reduced_success_step": first, "openings": openings,
        })
    episode = args.baseline / args.episode
    rows = [json.loads(line) for line in (episode / "steps.jsonl").read_text().splitlines()]
    arrays = np.load(episode / "observations.npz")
    summary = json.loads((episode / "summary.json").read_text())
    settings = json.loads((args.baseline / "settings.json").read_text())
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    agent, safety, _, provenance = load_frozen(filter_step=settings["filter_step"])
    save_implementation(args.output)
    controller = FrozenController(agent, safety, settings)
    torch.manual_seed(summary["policy_seed"])
    heads, error = [], 0.0
    for t, reference in enumerate(rows):
        obs = {
            "front_cam": torch.as_tensor(arrays["cameras"][t, 0:1], device="cuda"),
            "wrist_cam": torch.as_tensor(arrays["cameras"][t, 1:2], device="cuda"),
            "eef_pos": torch.as_tensor(arrays["eef_pos"][t:t + 1], device="cuda"),
            "eef_quat": torch.as_tensor(arrays["eef_quat_wxyz"][t:t + 1], device="cuda"),
            **{k: torch.tensor([v], device="cuda") for k, v in {
                "is_first": t == 0, "is_last": False, "is_terminal": False,
                "failure": reference["before"]["physical_failure"],
                "success": reference["before"]["physical_success"],
            }.items()},
        }
        _, actual = controller.act(obs, summary["mode"], diagnostic=t % settings["diagnostic_interval"] == 0)
        for key in ("proposed_action", "executed_normalized_action", "uncertainty", "next_q1", "next_q2"):
            delta = float(np.abs(np.asarray(actual[key]) - np.asarray(reference[key])).max())
            error = max(error, delta)
            assert delta < 1e-6, (t, key, delta)
        assert actual["intervened"] == reference["intervened"]
        with torch.random.fork_rng(devices=[0]):
            values = {"step": t, "time_before_action_s": t * 0.05}
            for name, latent in (("task", controller.task_latent), ("filter", controller.filter_latent)):
                feat = agent._wm.dynamics.get_feat(latent)
                values[name + "_predicted_reward"] = agent._wm.heads["reward"](feat).mode().item()
                values[name + "_continuation_probability"] = agent._wm.heads["cont"](feat).mean.item()
            heads.append(values)
    result = {
        "status": "complete", "geometry_audit": geometry, "head_episode": args.episode,
        "heads": heads, "maximum_replay_error": error,
        "model_provenance": provenance, "query_counts_replay": dict(controller.queries),
        "additional_model_queries": {"reward_head": 2 * len(rows), "continuation_head": 2 * len(rows)},
        "environment_steps": 0, "training_updates": 0, "elapsed_seconds": time.perf_counter() - started,
        "paper_source": "https://arxiv.org/html/2505.00779v2#A4.T12",
        "limits": "Reduced success removes only the released code's 7 cm distance component while retaining its other geometry thresholds. The paper table omits that component; this does not establish the exact historical checkpoint training predicate. Learned heads are predictions, not causes or physical outcome measurements.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Replay error", error, flush=True)
    for t in (20, 30, 40, 41, 42, 43, 44, 45, 50, 60, 100):
        print(heads[t], flush=True)


if __name__ == "__main__":
    main()
