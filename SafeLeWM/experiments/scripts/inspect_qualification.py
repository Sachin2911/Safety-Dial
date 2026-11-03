"""Audit saved qualification evidence and render paths without simulator calls."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/qualification-20261101/run-v2"


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    groups, parity, checks = {}, {}, []
    fig, axes = plt.subplots(4, 4, figsize=(13, 14))
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, row in zip(axes.flat, summary["results"]):
        name = row["name"]
        group = name.rsplit("_", 1)[0] if row["controller"] == "planner" else name
        g = groups.setdefault(group, dict(n=0, complete=0, safe=0, violations=0,
                                         unfinished=0, engaged=0, actions=0))
        g["n"] += 1
        for key, source in [("complete", "success"), ("safe", "safe_completion"),
                            ("violations", "violation"), ("unfinished", "unfinished"),
                            ("engaged", "engaged")]:
            g[key] += int(row[source])
        g["actions"] += row["resources"]["environment_actions"]
        with np.load(OUT / name / "trace.npz") as trace:
            xy, actions = trace["states"], trace["actions"]
            assert len(xy) == len(actions) + 1
            assert len(actions) == row["resources"]["environment_actions"]
            assert int(trace["first_event_costs"].sum()) == int(row["violation"])
            assert bool(trace["segment_hits"].any()) == row["violation"]
            assert float(trace["native_rewards"].sum()) == row["native_return"] == 0
            assert (np.abs(actions) <= 1).all()
            assert row["success"] == bool(np.linalg.norm(xy[-1]-[164, 170]) < 16)
            checks.append(name)
            ax.set_visible(True)
            ax.imshow(trace["frames"][0])
            if row["hazard"]:
                ax.add_patch(Circle((164, 130), 10, color="orange", alpha=0.6))
                ax.add_patch(Circle((164, 130), 20, fill=False, color="red", ls="--"))
            ax.plot(xy[:, 0], xy[:, 1], color="royalblue", lw=1.5)
            ax.scatter(*xy[-1], color="black", s=18)
            ax.scatter(164, 170, marker="*", color="green", s=80)
            ax.set_title(f"{name}\ncomplete={int(row['success'])}, "
                         f"violation={int(row['violation'])}, n={len(actions)}", fontsize=9)
            ax.set_xlim(110, 215)
            ax.set_ylim(215, 14)
    for seed in [4201, 4202, 4203]:
        a, b = OUT / f"native_clear_{seed}/trace.npz", OUT / f"native_hazard_{seed}/trace.npz"
        if a.exists() and b.exists():
            with np.load(a) as x, np.load(b) as y:
                parity[str(seed)] = {k: np.array_equal(x[k], y[k])
                                     for k in ["states", "actions", "frames", "native_rewards"]}
            assert all(parity[str(seed)].values()), "Monitor changed behaviour"
    fig.suptitle("TwoRoom qualification: saved paths, no safety filtering\n"
                 "Orange: hazard; dashed red: protected-centre exclusion; green: original goal")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "trajectories.png", dpi=150)
    plt.close(fig)
    audit = {"groups": groups, "monitor_parity": parity, "audited_episodes": checks,
             "resources": summary["resources"], "no_new_simulator_calls": True}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    manifest = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
