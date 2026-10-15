"""Replay a recorded rollout exactly and isolate stochastic controller decisions."""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from helpers.unisafe_controller import FrozenController
from helpers.unisafe_model import load_frozen
from helpers.unisafe_source import save_implementation


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    settings = json.loads((args.episode.parent / "settings.json").read_text())
    summary = json.loads((args.episode / "summary.json").read_text())
    rows = [json.loads(line) for line in (args.episode / "steps.jsonl").read_text().splitlines()]
    arrays = np.load(args.episode / "observations.npz")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    agent, safety, _, provenance = load_frozen(filter_step=settings["filter_step"])
    save_implementation(args.output)
    controller = FrozenController(agent, safety, settings)
    wm = agent._wm
    torch.manual_seed(summary["policy_seed"])
    probes, probe_queries = [], Counter()
    max_replay_error = 0.0
    selected_steps = {30, 35, 40, 41, 42, 43, 44, 45, 49, 50, 51, 52, 60, 100, 200, 299}
    for t, reference in enumerate(rows):
        obs = {
            "front_cam": torch.as_tensor(arrays["cameras"][t, 0:1], device="cuda"),
            "wrist_cam": torch.as_tensor(arrays["cameras"][t, 1:2], device="cuda"),
            "eef_pos": torch.as_tensor(arrays["eef_pos"][t : t + 1], device="cuda"),
            "eef_quat": torch.as_tensor(arrays["eef_quat_wxyz"][t : t + 1], device="cuda"),
            **{
                k: torch.tensor([value], device="cuda")
                for k, value in {
                    "is_first": t == 0,
                    "is_last": False,
                    "is_terminal": False,
                    "failure": reference["before"]["physical_failure"],
                    "success": reference["before"]["physical_success"],
                }.items()
            },
        }
        _, actual = controller.act(
            obs, summary["mode"], diagnostic=t % settings["diagnostic_interval"] == 0
        )
        for key in (
            "proposed_action",
            "executed_normalized_action",
            "uncertainty",
            "next_q1",
            "next_q2",
        ):
            error = float(np.max(np.abs(np.asarray(actual[key]) - np.asarray(reference[key]))))
            max_replay_error = max(max_replay_error, error)
            if error > 1e-6:
                raise AssertionError(f"Replay mismatch step {t} {key}: {error}")
        assert actual["intervened"] == reference["intervened"]
        if t not in selected_steps:
            continue
        # Additional diagnostics must not advance the replay's random streams.
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(100000 + t)
            n = args.samples
            state = {
                k: v.repeat(n, *([1] * (v.ndim - 1))) for k, v in controller.filter_latent.items()
            }
            proposal = torch.tensor([actual["proposed_action"]], device="cuda").repeat(n, 1)
            imagined = wm.dynamics.img_step(state, proposal, sample=True)
            feat = wm.dynamics.get_feat(imagined)
            backup, _ = safety.select_action(feat, eval_mode=True)
            q = torch.minimum(safety.critic1(feat, backup), safety.critic2(feat, backup)).flatten()
            modal = {k: v[:1] for k, v in imagined.items()}
            modal["stoch"] = wm.dynamics.get_dist(modal).mode()
            modal_feat = wm.dynamics.get_feat(modal)
            modal_action, _ = safety.select_action(modal_feat, eval_mode=True)
            modal_q = torch.minimum(
                safety.critic1(modal_feat, modal_action), safety.critic2(modal_feat, modal_action)
            ).item()
            task_state = {
                k: v.repeat(n, *([1] * (v.ndim - 1))) for k, v in controller.task_latent.items()
            }
            task_state["stoch"] = wm.dynamics.get_dist(task_state).sample()
            task_actions = agent._task_behavior.actor(wm.dynamics.get_feat(task_state)).mode()
            modal_task = {k: v[:1] for k, v in task_state.items()}
            modal_task["stoch"] = wm.dynamics.get_dist(modal_task).mode()
            modal_gripper = (
                2
                * agent._task_behavior.actor(wm.dynamics.get_feat(modal_task)).mode()[0, -1].item()
            )
            probe_queries.update(
                imagined_step=n, safety_actor=n + 1, safety_critic=2 * (n + 1), task_actor=n + 1
            )
            probes.append(
                {
                    "step": t,
                    "time_before_action_s": t * 0.05,
                    "samples": n,
                    "actual_gripper_proposal": actual["proposed_action"][-1],
                    "actual_intervention": actual["intervened"],
                    "actual_q": actual["next_min_q"],
                    "fixed_input_q_quantiles": np.quantile(
                        q.cpu().numpy(), [0, 0.05, 0.5, 0.95, 1]
                    ).tolist(),
                    "fixed_input_q_std": q.std().item(),
                    "fixed_input_value_trigger_fraction": (q < settings["reachability_threshold"])
                    .float()
                    .mean()
                    .item(),
                    "fixed_input_uncertainty_trigger": actual["uncertainty_trigger"],
                    "modal_next_q": modal_q,
                    "fixed_posterior_gripper_open_fraction": (task_actions[:, -1] >= 0)
                    .float()
                    .mean()
                    .item(),
                    "fixed_posterior_gripper_quantiles": np.quantile(
                        (2 * task_actions[:, -1]).cpu().numpy(), [0, 0.05, 0.5, 0.95, 1]
                    ).tolist(),
                    "modal_posterior_gripper": modal_gripper,
                }
            )
    result = {
        "status": "complete",
        "episode": str(args.episode),
        "replayed_frames": len(rows),
        "maximum_replay_error": max_replay_error,
        "probes": probes,
        "query_counts_replay": dict(controller.queries),
        "query_counts_additional_individual_samples": dict(probe_queries),
        "environment_steps": 0,
        "training_updates": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "model_provenance": provenance,
        "limits": "Conditional sampling at fixed observed histories; offline probes are not physical counterfactual outcomes.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Exact replay maximum error:", max_replay_error)
    for p in probes:
        print(
            p["step"],
            "open probability",
            p["fixed_posterior_gripper_open_fraction"],
            "value-trigger probability",
            p["fixed_input_value_trigger_fraction"],
            "modal Q",
            p["modal_next_q"],
        )


if __name__ == "__main__":
    main()
