"""Describe all native check episodes without fitting a controller or a threshold."""

import argparse
import collections
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import cv2

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts/todo2"))
from helpers.tracking import aligned_arm, events, load_recording, requested_sequence
from timeline import story
from frames import label


def stats(values):
    values = np.asarray(values)
    return ({"n": int(values.size), "mean": float(np.mean(values)),
             "median": float(np.median(values)), "p90": float(np.quantile(values, .9)),
             "max": float(np.max(values))} if values.size else {"n": 0})


def rms(values):
    return np.sqrt(np.mean(np.square(values), axis=-1))


def describe(data, meta, row):
    event = events(data, meta)
    ap = {name: data["ap"][:, i] for i, name in enumerate(meta["ap_names"])}
    v = event["violation"]
    clause = (None if v is None else "jar_upright" if not ap["jar_upright"][v]
              else "jar_not_dropped" if ap["jar_dropped"][v] else "close_before_lift")
    narrative = story({**event, "contact": row["first_contact_step"],
                       "success": row["success"], "counted_violation": row["counted_violation"],
                       "violation_clause": clause})
    metrics = {"events": event, "story": narrative, "violation_clause": clause,
               "safe_requested": bool(row["success"] and not row["counted_violation"]
                                      and requested_sequence(event)),
               "requested_success": bool(row["success"] and requested_sequence(event)),
               "lag_error_rad": []}
    contact = event["contact"]
    n = row["steps"]
    for lag in range(9):
        target, actual, steps = aligned_arm(data, meta, lag=lag)
        error = np.abs(target - actual)
        # Both endpoints precede contact for this reference, even at positive lag.
        pre = steps + lag < (contact if contact is not None else n + 1)
        metrics["lag_error_rad"].append({
            "lag_steps": lag, "lag_seconds": lag / meta["action_frequency"],
            "all": stats(error), "pre_contact": stats(error[pre]),
            "post_contact": stats(error[~pre]),
            "per_joint_mean": np.mean(error, axis=0).tolist(),
        })
    target, before, steps = aligned_arm(data, meta, 0)
    _, after, _ = aligned_arm(data, meta, 1)
    variation = np.vstack([np.zeros((1, 7)), np.diff(target, axis=0)])
    windows = {"all": (0, n), "last_200_steps": (max(0, n - 200), n)}
    if contact is not None:
        windows["pre_contact"] = (0, contact)
        windows["post_contact"] = (contact, n)
    if v is not None:
        windows["two_seconds_before_violation"] = (max(0, v - 40), v)
    if event["closed"] is not None:
        windows["two_seconds_after_closure"] = (event["closed"], min(n, event["closed"] + 40))
    metrics["windows"] = {}
    for name, (lo, hi) in windows.items():
        mask = (steps >= lo) & (steps < hi)
        metrics["windows"][name] = {
            "start_step": lo, "end_step_exclusive": hi,
            "requested_move_rms_rad": stats(rms(target[mask] - before[mask])),
            "realized_move_rms_rad": stats(rms(after[mask] - before[mask])),
            "post_action_error_rms_rad": stats(rms(target[mask] - after[mask])),
            "target_change_rms_rad": stats(rms(variation[mask])),
            "gripper_open_command_fraction": float(np.mean(data["exec_clipped"][mask, -1] > 0))
            if mask.any() else None,
        }
    metrics["arm_clipped_action_fraction"] = float(np.mean(
        np.any(data["exec_ctrl"][:, :7] != data["exec_clipped"][:, :7], axis=1)))
    return metrics


def plot_episode(data, meta, row, metrics, directory):
    target, before, steps = aligned_arm(data, meta, 0)
    _, after, _ = aligned_arm(data, meta, 1)
    hz = meta["action_frequency"]
    time = steps / hz
    state_time = data["step"] / hz
    fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True, layout="constrained")
    axes[0].plot(time, rms(target - before), label="before action", linewidth=.8)
    axes[0].plot(time, rms(target - after), label="after action", linewidth=.8)
    axes[0].set_ylabel("Arm tracking error\nRMS across joints (rad)")
    axes[1].plot(time, rms(target - before), label="requested move", linewidth=.8)
    axes[1].plot(time, rms(after - before), label="realized move", linewidth=.8)
    axes[1].set_ylabel("Arm step motion\nRMS across joints (rad)")
    axes[2].plot(state_time, data["jar_joint_q"], linewidth=.9, label="lid hinge")
    axes[2].set_ylabel("Lid joint angle (rad)")
    axes[3].plot(state_time, data["jar_pos"][:, 2] - data["jar_pos"][0, 2],
                 label="jar height change", linewidth=.9)
    axes[3].plot(state_time, np.linalg.norm(data["eef_pos"] - data["eef_pos"][0], axis=1),
                 label="end-effector displacement", linewidth=.9)
    axes[3].set_ylabel("Displacement (m)")
    axes[4].plot(state_time, np.mean(data["gripper_q"], axis=1),
                 label="mean finger joint", linewidth=.9)
    axes[4].set_ylabel("Finger position (m)")
    ax_grip = axes[4].twinx()
    ax_grip.plot(time, data["exec_clipped"][:, -1], color="gray", alpha=.4,
                 label="gripper command")
    ax_grip.set_ylabel("Command (-1 close, +1 open)")
    for i, name in enumerate(meta["ap_names"]):
        axes[5].step(state_time, data["ap"][:, i].astype(int) * .7 + i,
                     where="post", linewidth=.9, label=name)
    axes[5].set_yticks(np.arange(len(meta["ap_names"])), meta["ap_names"], fontsize=8)
    axes[5].set_ylabel("Monitor propositions\nbase=false, raised=true")
    axes[5].set_xlabel("Simulation time since first policy action (s), 20 Hz")
    for ax in axes:
        ax.grid(alpha=.15)
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        for event in ("contact", "closed", "grasp", "lift", "violation"):
            step = metrics["events"][event]
            if step is not None:
                ax.axvline(step / hz, color="black", alpha=.18, linewidth=.6)
    event_text = ", ".join(f"{k}={v}" for k, v in metrics["events"].items())
    fig.suptitle(f"Task {row['scene_name']} | seed {row['seed']} | {metrics['story']}\n"
                 f"Native commands and realized motion | event steps: {event_text}", fontsize=11)
    fig.savefig(directory / "timeline.png", dpi=125)
    plt.close(fig)
    fig, axes = plt.subplots(7, 1, figsize=(12, 11), sharex=True, layout="constrained")
    q = data["q"][:, meta["arm_control_idx"]]
    for i, ax in enumerate(axes):
        ax.plot(time, target[:, i], label="commanded target", linewidth=.8)
        ax.plot(state_time, q[:, i], label="measured joint", linewidth=.8, alpha=.8)
        ax.set_ylabel(f"Joint {i + 1}\n(rad)")
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Simulation time (s), target at issue time, state at observation time")
    fig.suptitle(f"Task {row['scene_name']} | seed {row['seed']} | absolute joint targets", fontsize=12)
    fig.savefig(directory / "joint_targets.png", dpi=125)
    plt.close(fig)


def frame_sheet(folder, row, metrics, directory):
    event = metrics["events"]
    end = row["steps"]
    contact = event["contact"] if event["contact"] is not None else 0
    key = next(((event[name], name) for name in ("violation", "lift")
                if event[name] is not None), (end, "end"))
    closed = ((event["closed"], "closed") if event["closed"] is not None
              else (min(contact + 100, end), "contact+100"))
    columns = [(0, "start"), (contact, "contact"),
               (min(contact + 40, end), "contact+40"), closed, key, (end, "end")]
    strips = []
    for camera in ("main", "wrist"):
        path = next(folder.rglob(f"base_{camera}.mp4"))
        cap = cv2.VideoCapture(str(path))
        frames = {}
        wanted = {step for step, _ in columns}
        for step in range(end + 1):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"{path}: missing frame {step}")
            if step in wanted:
                frames[step] = frame
        cap.release()
        strips.append(np.hstack([
            label(frames[step], [f"seed {row['seed']} {row['scene_name']} {camera}",
                                 f"{name}: step {step}"])
            for step, name in columns
        ]))
    cv2.imwrite(str(directory / "event_frames.png"), np.vstack(strips))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/tracking04/remote")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/tracking04/analysis")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    episodes = []
    for validated in sorted(args.root.glob("episodes/*/validation.json")):
        folder = validated.parent
        validation = json.loads(validated.read_text())
        row = validation["row"]
        data, meta = load_recording(next(folder.glob("recordings/*.npz")))
        job = json.loads((folder / "job.json").read_text())
        metrics = describe(data, meta, row)
        directory = args.out / folder.name
        directory.mkdir(exist_ok=True)
        plot_episode(data, meta, row, metrics, directory)
        frame_sheet(folder, row, metrics, directory)
        episode = {**job, **metrics, "row": row, "queries": validation["queries"],
                   "recorder_overhead_s": validation["recorder_overhead_s"]}
        (directory / "metrics.json").write_text(json.dumps(episode, indent=2) + "\n")
        episodes.append(episode)
    counts = collections.Counter()
    for e in episodes:
        r = e["row"]
        cls = ("unsafe_" if r["counted_violation"] else "safe_")
        cls += "success" if r["success"] else "failure"
        counts[cls] += 1
        counts["engaged"] += r["first_contact_step"] is not None
        counts["safe_requested"] += e["safe_requested"]
        counts["requested_success"] += e["requested_success"]
        counts["policy_steps"] += r["steps"]
        counts["queries"] += e["queries"]
    summary = {"n": len(episodes), "counts": counts,
               "stories": collections.Counter(e["story"] for e in episodes), "episodes": episodes}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
