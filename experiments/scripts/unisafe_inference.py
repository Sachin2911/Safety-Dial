"""Frozen UNISafe inference on bundled recordings, without starting a simulator.

Outputs per-frame traces, reconstruction/prediction images, plots and measured costs.
Recorded action[t] led to observation[t]; outgoing transition t uses action[t+1].
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from PIL import Image, ImageDraw

from helpers.unisafe_model import load_frozen
from helpers.unisafe_source import ROOT, save_implementation, sha256


def image_uint8(value):
    return np.clip(value.detach().cpu().numpy() * 255, 0, 255).astype(np.uint8)


def compare_run(output, reference, episodes):
    """Require identical saved arrays and numerical traces, excluding elapsed time."""
    for episode in episodes:
        name = Path(episode["episode"]).stem
        with (
            np.load(output / name / "features_actions.npz", allow_pickle=False) as actual,
            np.load(reference / name / "features_actions.npz", allow_pickle=False) as expected,
        ):
            assert actual.files == expected.files
            for key in actual.files:
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=f"{name}/{key}")
        with (output / name / "trace.csv").open() as stream:
            actual_rows = list(csv.DictReader(stream))
        with (reference / name / "trace.csv").open() as stream:
            expected_rows = list(csv.DictReader(stream))
        assert len(actual_rows) == len(expected_rows)
        for actual, expected in zip(actual_rows, expected_rows, strict=True):
            actual.pop("inference_seconds_including_diagnostics")
            expected.pop("inference_seconds_including_diagnostics")
            assert actual == expected, f"Numerical trace mismatch: {name} frame {actual['frame']}"
    return {
        "reference": str(reference.resolve()),
        "result": "bitwise arrays and exact numerical traces match",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compare-to", type=Path, help="Require identical numerical outputs to a prior run"
    )
    parser.add_argument("--filter-step", type=int, default=200000)
    args = parser.parse_args()
    output = args.output or ROOT / "runs/unisafe" / datetime.now(UTC).strftime(
        "inference-%Y%m%dT%H%M%SZ"
    )
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    agent, safety, config, provenance = load_frozen(filter_step=args.filter_step)
    save_implementation(output)
    wm = agent._wm
    torch.cuda.reset_peak_memory_stats()
    results = []
    for source in sorted((ROOT / "third_party/UNISafe/log").glob("failure_*.npz")):
        with np.load(source, allow_pickle=False) as archive:
            episode = {key: archive[key] for key in archive.files}
        length = len(episode["action"])
        destination = output / source.stem
        destination.mkdir()
        latent = None
        rows, proposed, latents = [], [], []
        selected = set(np.linspace(0, length - 2, 6, dtype=int).tolist())
        observations_finite = all(np.isfinite(x).all() for x in episode.values())
        assert observations_finite
        for t in range(length):
            obs = {key: value[t : t + 1].copy() for key, value in episode.items()}
            torch.cuda.synchronize()
            before = time.perf_counter()
            with torch.inference_mode():
                data = wm.preprocess(obs)
                embed = wm.encoder(data)
                action_in = torch.as_tensor(episode["action"][t : t + 1], device=config.device)
                latent, _prior = wm.dynamics.obs_step(
                    latent,
                    action_in if latent is not None else None,
                    embed,
                    data["is_first"],
                    sample=False,
                )
                feat = wm.dynamics.get_feat(latent)
                decoded = {k: v.mode()[:, 0] for k, v in wm.heads["decoder"](feat[:, None]).items()}
                failure = wm.heads["failure"](feat).mean
                task_action = agent._task_behavior.actor(feat).mode()
                # Matches the evaluator's nominal gripper processing. Diagnostics
                # do not apply safety replacement or execute any proposed action.
                task_action = task_action.clone()
                task_action[:, -1] *= 2
                uq = agent._disag_ensemble.intrinsic_reward_penn(torch.cat((feat, task_action), -1))
                next_latent = wm.dynamics.img_step(latent, task_action, sample=False)
                next_feat = wm.dynamics.get_feat(next_latent)
                backup, _ = safety.select_action(next_feat, eval_mode=True)
                q1 = safety.critic1(next_feat, backup)
                q2 = safety.critic2(next_feat, backup)
                mse = (
                    sum((decoded[k] - data[k]).square().mean() for k in ("front_cam", "wrist_cam"))
                    / 2
                )
                row = {
                    "frame": t,
                    "failure_label": float(episode["failure"][t]),
                    "success_label": float(episode["success"][t]),
                    "is_last": float(episode["is_last"][t]),
                    "failure_prediction": failure.item(),
                    "proposal_uncertainty": uq.item(),
                    "proposal_next_q1": q1.item(),
                    "proposal_next_q2": q2.item(),
                    "proposal_next_min_q": min(q1.item(), q2.item()),
                    "reconstruction_mse": mse.item(),
                }
                predicted = None
                if t < length - 1:
                    outgoing = torch.as_tensor(
                        episode["action"][t + 1 : t + 2], device=config.device
                    )
                    recorded_next = wm.dynamics.img_step(latent, outgoing, sample=False)
                    predicted = {
                        k: v.mode()[:, 0]
                        for k, v in wm.heads["decoder"](
                            wm.dynamics.get_feat(recorded_next)[:, None]
                        ).items()
                    }
                    next_observed = {
                        key: torch.as_tensor(episode[key][t + 1 : t + 2], device=config.device)
                        / 255
                        for key in ("front_cam", "wrist_cam")
                    }
                    row["recorded_next_image_mse"] = (
                        sum(
                            (predicted[key] - next_observed[key]).square().mean().item()
                            for key in next_observed
                        )
                        / 2
                    )
                else:
                    row["recorded_next_image_mse"] = None
                proposed.append(task_action.cpu().numpy()[0])
                latents.append(feat.cpu().numpy()[0])
                if t in selected:
                    panel = Image.new("RGB", (128 * 4, 128 * 2 + 24), "white")
                    draw = ImageDraw.Draw(panel)
                    for x, title in enumerate(
                        ("Observed t", "Reconstruction", "Observed t+1", "Predicted t+1")
                    ):
                        draw.text((x * 128 + 3, 5), title, fill="black")
                    for y, key in enumerate(("front_cam", "wrist_cam")):
                        images = [
                            episode[key][t].astype(np.uint8),
                            image_uint8(decoded[key][0]),
                            episode[key][t + 1].astype(np.uint8),
                            image_uint8(predicted[key][0]),
                        ]
                        for x, arr in enumerate(images):
                            panel.paste(Image.fromarray(arr), (128 * x, 24 + 128 * y))
                    panel.save(destination / f"frame-{t:03d}.png")
                assert torch.isfinite(feat).all() and torch.isfinite(task_action).all()
            torch.cuda.synchronize()
            row["inference_seconds_including_diagnostics"] = time.perf_counter() - before
            assert all(np.isfinite(v) for v in row.values() if v is not None), row
            rows.append(row)
        with (destination / "trace.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(
            destination / "features_actions.npz",
            features=np.stack(latents),
            proposed_actions=np.stack(proposed),
            recorded_actions=episode["action"],
        )
        fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, layout="constrained")
        axes[0].plot([r["failure_prediction"] for r in rows], label="Predicted failure")
        axes[0].step(range(length), episode["failure"], label="Recorded label", where="post")
        axes[0].legend()
        axes[1].plot([r["proposal_uncertainty"] for r in rows])
        axes[1].axhline(
            config.ood_threshold, color="red", linestyle="--", label="Saved training threshold"
        )
        axes[1].set_ylabel("Uncertainty")
        axes[1].legend()
        axes[2].plot([r["proposal_next_min_q"] for r in rows])
        axes[2].set_ylabel("Next-state safety value")
        axes[3].plot([r["reconstruction_mse"] for r in rows], label="Reconstruction")
        axes[3].plot([r["recorded_next_image_mse"] for r in rows], label="One-step prediction")
        axes[3].legend()
        axes[3].set_ylabel("Image MSE [0,1]")
        axes[3].set_xlabel("Recorded frame (no physical event timing inferred)")
        fig.suptitle(source.stem + ": frozen inference, posterior mode")
        fig.savefig(destination / "traces.png", dpi=150)
        plt.close(fig)
        latencies = [r["inference_seconds_including_diagnostics"] for r in rows]
        result = {
            "episode": source.name,
            "frames": length,
            "sha256": sha256(source),
            "latency_median_s": float(np.median(latencies)),
            "latency_p95_s": float(np.percentile(latencies, 95)),
            "label_positive_frames": int(episode["failure"].sum()),
            "query_counts": {
                "encoder": length,
                "posterior_update": length,
                "imagined_step": 2 * length - 1,
                "decoder": 2 * length - 1,
                "task_actor": length,
                "failure_head": length,
                "uncertainty_ensemble": length,
                "safety_actor": length,
                "safety_critic": 2 * length,
            },
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    summary = {
        "status": "passed",
        "provenance": provenance,
        "seed": args.seed,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "latent_sampling": "posterior and imagined mode (sample=False); diagnostic convention",
        "action_alignment": "action[t] led to obs[t]; one-step prediction uses action[t+1]",
        "safety_value_scope": "hypothetical task proposal from recorded posterior; no intervention executed",
        "timing_scope": "batch 1 CUDA synchronization; includes reconstruction, prediction and selected image saving; no warmup exclusion",
        "physical_event_timing": "unknown; upstream labels include pre-failure relabeling and terminal observation substitution",
        "simulator_interactions": 0,
        "training_updates": 0,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "elapsed_seconds": time.perf_counter() - started,
        "episodes": results,
    }
    if args.compare_to is not None:
        summary["repeatability"] = compare_run(output, args.compare_to, results)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Results:", output, flush=True)


if __name__ == "__main__":
    main()
