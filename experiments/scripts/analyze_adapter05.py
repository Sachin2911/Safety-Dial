"""Describe every adapter attempt, separating engineering checks from scored episodes."""

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from analyze_tracking import describe, frame_sheet, plot_episode, stats
from helpers.tracking import load_recording


def read_events(path):
    if not path.exists():
        return []
    # A crash can leave a partial final line; keep every complete entry and report it.
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            rows.append({"event": "truncated_line"})
    return rows


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, c - h), min(1, c + h)]


def totals(episodes):
    counts = collections.Counter(
        {
            key: 0
            for key in (
                "safe_success",
                "unsafe_success",
                "safe_failure",
                "unsafe_failure",
                "safe_requested",
                "requested_success",
                "raw_violation",
                "counted_violation",
                "unfinished",
                "engaged",
                "policy_steps",
                "queries",
            )
        }
    )
    for episode in episodes:
        row = episode["row"]
        cls = "unsafe_" if row["counted_violation"] else "safe_"
        cls += "success" if row["success"] else "failure"
        counts[cls] += 1
        for key in ("safe_requested", "requested_success"):
            counts[key] += bool(episode[key])
        counts["raw_violation"] += bool(row["ltl_violated"])
        counts["counted_violation"] += bool(row["counted_violation"])
        counts["unfinished"] += not row["success"]
        counts["engaged"] += row["first_contact_step"] is not None
        counts["policy_steps"] += row["steps"]
        counts["queries"] += episode["queries"]
    n = len(episodes)
    return {
        "n": n,
        "counts": dict(counts),
        "nominal_wilson_95": {
            key: wilson(counts[key], n)
            for key in (
                "safe_requested",
                "safe_success",
                "counted_violation",
                "unfinished",
                "engaged",
            )
        },
    }


def paired(episodes, arm):
    cells = {(e["task"], e["seed"], e["arm"]): e for e in episodes}
    pairs = [
        (e, cells[(e["task"], e["seed"], "native")])
        for e in episodes
        if e["arm"] == arm and (e["task"], e["seed"], "native") in cells
    ]
    metrics = {}
    for metric in ("safe_requested", "counted_violation", "unfinished", "engaged"):

        def value(e):
            if metric == "safe_requested":
                return e[metric]
            if metric == "unfinished":
                return not e["row"]["success"]
            if metric == "engaged":
                return e["row"]["first_contact_step"] is not None
            return e["row"][metric]

        by_task = collections.defaultdict(list)
        for treated, native in pairs:
            by_task[treated["task"]].append(int(value(treated)) - int(value(native)))
        task_means = np.array([np.mean(v) for v in by_task.values()])
        if len(task_means):
            rng = np.random.default_rng(91405)
            draws = rng.choice(task_means, (10000, len(task_means)), replace=True).mean(axis=1)
            metrics[metric] = {
                "difference": float(task_means.mean()),
                "task_cluster_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
            }
    return {"paired_cells": len(pairs), "metrics": metrics}


def activity_plot(commands, data, job, directory):
    t = np.arange(len(commands)) / 20
    coefficients = np.array([e["coefficients"] for e in commands])
    correction = np.array([e["correction"] for e in commands])
    raw = np.array([e["native_command"] for e in commands])[:, :7]
    measured = np.array([e["q"] for e in commands])
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, layout="constrained")
    for i, name in enumerate(("position error", "velocity", "reference acceleration")):
        axes[0].plot(t, coefficients[:, i], label=name, linewidth=0.9)
    axes[0].set_ylabel("Coefficient")
    axes[1].plot(t, correction, linewidth=0.6)
    axes[1].axhline(0.10, color="gray", linestyle=":")
    axes[1].axhline(-0.10, color="gray", linestyle=":")
    axes[1].set_ylabel("Correction (rad)")
    axes[2].plot(t, np.abs(raw - measured).mean(axis=1), label="native target error", linewidth=0.9)
    axes[2].plot(
        t,
        np.abs(data["exec_clipped"][:, :7] - measured).mean(axis=1),
        label="executed target error",
        linewidth=0.9,
    )
    axes[2].set_ylabel("Mean pre-action\nabsolute error (rad)")
    axes[2].set_xlabel("Simulation time (s)")
    for ax in axes:
        ax.grid(alpha=0.15)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    fig.suptitle(f"{job['arm']} | task {job['task']} | seed {job['seed']} | {job['stage']}")
    fig.savefig(directory / "adapter_activity.png", dpi=125)
    plt.close(fig)


def accounting(root):
    attempts = []
    for folder in sorted(root.glob("episodes/*")):
        job_path = folder / "job.json"
        if not job_path.exists():
            continue
        job = json.loads(job_path.read_text())
        events = read_events(folder / "adapter.jsonl")
        counts = collections.Counter(e["event"] for e in events)
        recorder = [e for path in folder.rglob("*.journal.jsonl") for e in read_events(path)]
        probes = re.findall(
            r"model queries made by this probe: (\d+)",
            (folder / "reseed.log").read_text() if (folder / "reseed.log").exists() else "",
        )
        attempts.append(
            {
                **job,
                "folder": folder.name,
                "validated": (folder / "validation.json").exists(),
                "warmup_attempts": counts["warmup_attempt"],
                "warmup_returned": counts["warmup_applied"],
                "policy_commands": counts["command"],
                "policy_returned": counts["applied"],
                "recorded_policy_queries": sum(e["event"] == "proposal" for e in recorder),
                "policy_query_attempts": counts["query_attempt"],
                "policy_query_returns": counts["query_returned"],
                "reseed_probe_queries": (folder / "reseed.log")
                .read_text()
                .count("adapter05_query_attempt")
                if (folder / "reseed.log").exists()
                else 0,
                "reseed_final_report": int(probes[-1]) if probes else None,
                "truncated_journal_lines": counts["truncated_line"],
            }
        )
    startup_queries = 0
    incomplete_probes = []
    for path in root.glob("logs/**/probe*.log"):
        queries = re.findall(r"model queries made by this probe: (\d+)", path.read_text())
        if queries:
            startup_queries += path.read_text().count("adapter05_query_attempt")
        else:
            startup_queries += path.read_text().count("adapter05_query_attempt")
            incomplete_probes.append(str(path.relative_to(root)))
    return {
        "attempts": attempts,
        "startup_probe_queries": startup_queries,
        "incomplete_probe_logs": incomplete_probes,
        "query_note": "Query attempts and returns are journaled separately; interrupted calls remain charged as attempts.",
        "controls": {
            key: sum(a[key] for a in attempts)
            for key in ("warmup_attempts", "warmup_returned", "policy_commands", "policy_returned")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/adapter05/remote")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/adapter05/analysis")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    episodes = []
    for validated in sorted(args.root.glob("episodes/*/validation.json")):
        folder = validated.parent
        validation = json.loads(validated.read_text())
        job = json.loads((folder / "job.json").read_text())
        row = validation["row"]
        data, meta = load_recording(next(folder.glob("recordings/*.npz")))
        commands = [e for e in read_events(folder / "adapter.jsonl") if e["event"] == "command"]
        # Evaluate tracking against the policy's uncorrected target, not the adapter's target.
        reference_data = {**data, "exec_clipped": np.array([e["native_command"] for e in commands])}
        metrics = describe(reference_data, meta, row)
        metrics["tracking_reference"] = (
            "Native absolute joint target before adapter and action clipping"
        )
        metrics["arm_clipped_action_fraction"] = float(
            np.mean(np.any(data["exec_ctrl"][:, :7] != data["exec_clipped"][:, :7], axis=1))
        )
        metrics["correction_abs_rad"] = stats(np.abs([e["correction"] for e in commands]))
        metrics["saturated_joint_fraction"] = float(np.mean([e["saturated"] for e in commands]))
        metrics["final_coefficients"] = commands[-1]["next_coefficients"]
        directory = args.out / folder.name
        directory.mkdir(exist_ok=True)
        title_row = {**row, "scene_name": job["arm"] + " " + row["scene_name"]}
        plot_episode(reference_data, meta, title_row, metrics, directory)
        frame_sheet(folder, title_row, metrics, directory)
        activity_plot(commands, data, job, directory)
        episode = {**job, **metrics, "row": row, "queries": validation["queries"]}
        (directory / "metrics.json").write_text(json.dumps(episode, indent=2) + "\n")
        episodes.append(episode)
    scored = [e for e in episodes if e["stage"] == "pilot"]
    summary = {
        "engineering": totals([e for e in episodes if e["stage"] == "engineering"]),
        "pilot": {
            arm: totals([e for e in scored if e["arm"] == arm])
            for arm in ("native", "fixed", "adaptive")
        },
        "paired_vs_native": {arm: paired(scored, arm) for arm in ("fixed", "adaptive")},
        "uncertainty_note": "Exploratory panel, six tasks and two seeds. Wilson intervals assume "
        "independent episodes; paired bootstrap resamples tasks. Neither is confirmatory evidence "
        "or establishes zero risk when no violations occur. Matching seeds are not exact replays.",
        "accounting": accounting(args.root),
        "episodes": episodes,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in ("episodes", "accounting")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
