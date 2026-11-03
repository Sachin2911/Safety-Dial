"""Audit and plot the saved fixed-baseline evidence without simulator interactions."""

import collections
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fixed-baseline-20261101/run-v1"
OLD = ROOT / "outputs/baseline-20261101/run-v2"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    panel = json.loads((OUT / "panel.json").read_text())
    freeze = json.loads((OUT / "freeze.json").read_text())
    assert digest(OUT / "panel.json") == freeze["panel_sha256"]
    for name, expected in freeze["sources"].items():
        assert digest(ROOT / name) == expected
    assert summary["finished"] == summary["scheduled"] == len(panel) == 50
    old_rows = {
        r["case"]: r
        for r in json.loads((OLD / "summary.json").read_text())["results"]
        if r["panel"] == "dataset"
    }
    groups = collections.defaultdict(collections.Counter)
    totals = collections.Counter()
    paired = collections.Counter()
    fig, axes = plt.subplots(5, 10, figsize=(25, 13))
    for case, row, ax in zip(panel, summary["results"], axes.flat):
        assert row["case"] == case["id"] and row["profile"] == "random05"
        assert row["error"] is None
        assert digest(ROOT / case["file"]) == case["sha256"]
        with np.load(OUT / row["name"] / "trace.npz") as z:
            xy, actions, proposed = z["states"], z["actions"], z["proposed"]
            goal = z["goal"]
            n = len(actions)
            assert 1 <= n <= 150 and len(xy) == len(z["frames"]) == n + 1
            assert np.isfinite(actions).all() and np.isfinite(xy).all()
            assert np.array_equal(np.clip(proposed, -1, 1), actions)
            distances = np.linalg.norm(xy[1:] - goal, axis=1)
            success = bool(distances[-1] < 16)
            assert not (distances[:-1] < 16).any()
            assert success == row["success"] and (success or n == 150)
            assert row["success_step"] == (n if success else None)
            assert row["success_by_100"] == (success and n <= 100)
            assert row["unfinished"] == (not success)
            engaged = bool(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum() > 1)
            assert engaged == row["engaged"]
            assert row["native_return"] == z["native_rewards"].sum() == 0
            counters = row["resources"]
            assert counters["environment_actions"] == n
            assert counters["environment_resets"] == 1
            assert counters["replans"] == math.ceil(n / 5)
            assert counters["candidate_model_steps"] == counters["replans"] * 9000
            assert counters["model_batch_calls"] == counters["replans"] * 30
            assert counters["model_candidate_rollouts"] == counters["replans"] * 4500
            assert counters["clipped_action_coordinates"] == (abs(proposed - actions) > 1e-7).sum()
            with np.load(ROOT / case["file"]) as source:
                np.testing.assert_allclose(xy[0], source["initial"], rtol=0, atol=1e-5)
                np.testing.assert_array_equal(goal, source["goal"])
                assert np.abs(z["frames"][0].astype(float) - source["start_image"]).mean() <= 2
            room = "cross_room" if (xy[0, 0] < 112) != (goal[0] < 112) else "same_room"
            for key in ["all", room]:
                groups[key]["episodes"] += 1
                for field in ["success", "success_by_100", "unfinished", "engaged"]:
                    groups[key][field] += int(row[field])
            paired[f"paper_{int(old_rows[case['id']]['success'])}_random05_{int(success)}"] += 1
            totals.update(counters)
            ax.imshow(z["frames"][0])
            ax.plot(xy[:, 0], xy[:, 1], color="darkorange", lw=1)
            ax.scatter(*goal, color="green", marker="*", s=45)
            ax.scatter(*xy[-1], color="darkorange", s=15)
            ax.set_title(f"{case['id']}: {int(success)}/{n}\n{room}", fontsize=8)
            ax.set_xlim(0, 224)
            ax.set_ylim(224, 0)
            ax.set_xticks([])
            ax.set_yticks([])
    assert dict(totals) == summary["resources"]
    assert totals["environment_actions"] <= 7500
    assert totals["candidate_model_steps"] <= 13500000
    assert totals["environment_resets"] == 50
    fig.suptitle("Fixed random-05, no hazards: completion/actions; green star = original goal")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "trajectories.png", dpi=130)
    plt.close(fig)
    p, n, z = groups["all"]["success"] / 50, 50, 1.959963984540054
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    audit = dict(
        passed=True,
        audited_episodes=50,
        groups=dict(groups),
        paired_outcomes=dict(paired),
        unique_source_episodes=len({c["source_episode"] for c in panel}),
        descriptive_wilson_95=[centre - half, centre + half],
        interval_caveat="Exploratory development cases; independence is not established.",
        resources=dict(totals),
        episode_loop_seconds=summary["seconds"],
        upstream_replication="Unresolved; previous paper profile is a reconstruction.",
        audit_script_sha256=digest(Path(__file__)),
    )
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    manifest = {
        str(p.relative_to(OUT)): digest(p)
        for p in sorted(OUT.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
