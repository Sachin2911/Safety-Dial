"""Inspect saved baseline evidence. This script never imports or steps a simulator."""

import collections
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/baseline-20261101/run-v2"


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    panel = json.loads((OUT / "panel.json").read_text())
    groups, checks = {}, []
    for r in summary["results"]:
        key = r["panel"] + "/" + r["profile"]
        group = groups.setdefault(key, collections.Counter())
        group["episodes"] += 1
        for field in ["success", "success_by_100", "unfinished", "engaged"]:
            group[field] += int(r[field])
        group.update(r["resources"])
        with np.load(OUT / r["name"] / "trace.npz") as z:
            xy, actions, proposed = z["states"], z["actions"], z["proposed"]
            assert len(xy) == len(actions) + 1
            assert len(actions) == r["resources"]["environment_actions"]
            assert np.array_equal(np.clip(proposed, -1, 1), actions)
            assert r["success"] == bool(np.linalg.norm(xy[-1] - z["goal"]) < 16)
            assert z["native_rewards"].sum() == r["native_return"] == 0
            checks.append(r["name"])
    colors = dict(paper="royalblue", random05="darkorange", qualification="purple")
    for subset, rows, cols, filename in [
        ([c for c in panel if c["panel"] != "dataset"], 3, 3, "diagnostic-paths.png"),
        ([c for c in panel if c["panel"] == "dataset"], 5, 10, "dataset-paths.png"),
    ]:
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 3.3))
        for ax, case in zip(axes.flat, subset):
            labels = []
            first = True
            for profile in case["profiles"]:
                name = case["id"] + "_" + profile
                path = OUT / name / "trace.npz"
                if not path.exists():
                    continue
                r = json.loads((path.parent / "result.json").read_text())
                with np.load(path) as z:
                    if first:
                        ax.imshow(z["frames"][0])
                        ax.scatter(*z["goal"], marker="*", color="green", s=70)
                        first = False
                    xy = z["states"]
                    ax.plot(xy[:, 0], xy[:, 1], color=colors[profile], lw=1, label=profile)
                    ax.scatter(*xy[-1], color=colors[profile], s=12)
                labels.append(
                    f"{profile}: {int(r['success'])}/{r['resources']['environment_actions']}"
                )
            ax.set_title(case["id"] + "\n" + ", ".join(labels), fontsize=7)
            ax.set_xlim(0, 224)
            ax.set_ylim(224, 0)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(
            "No-hazard baseline: completion/actions; green star = original goal", fontsize=12
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(OUT / filename, dpi=130)
        plt.close(fig)
    preflight = json.loads((OUT / "preflight.json").read_text())
    totals = dict(summary["resources"])
    for key in ["environment_actions", "environment_resets", "candidate_model_steps"]:
        totals[key] = totals.get(key, 0) + preflight[key]
    audit = dict(
        groups=groups,
        audited_episodes=len(checks),
        scheduled=77,
        episode_resources=summary["resources"],
        totals_with_preflight=totals,
        preflight=preflight,
    )
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    manifest = {
        str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(OUT.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
