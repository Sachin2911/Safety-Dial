"""Render audited Part 1 outcomes, lineage, all-attempt costs and matched video index."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.maniguard_phase3_budget import atomic, clock
from helpers.maniguard_phase3_protocol import sha
from helpers.maniguard_phase3_results import COSTS, audit
from helpers.maniguard_phase3_search import COMPARISON


def build(directory, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=False)
    attempts = []
    videos = []
    for receipt_path in sorted((directory / "attempts").rglob("process.json")):
        attempt = receipt_path.parent
        record = audit(attempt / "data")
        process_path = attempt / "process.json"
        process = json.loads(process_path.read_text()) if process_path.exists() else None
        record["process"] = process
        record["attempt_id"] = str(attempt.relative_to(directory / "attempts"))
        if process is None or process["status"] != "completed" or process.get("returncode") != 0:
            record.update(valid=False, engineering_valid=False, U=None)
        attempts.append(record)
        for path in sorted((attempt / "data").rglob("*.mp4")):
            videos.append(
                {
                    "attempt_id": attempt.name,
                    "scene": record.get("scene"),
                    "seed": record.get("seed"),
                    "dial": record.get("dial"),
                    "purpose": record.get("purpose"),
                    "path": str(path.relative_to(directory)),
                    "sha256": sha(path),
                    "bytes": path.stat().st_size,
                }
            )
    totals = {}
    unknown = {}
    for key in COSTS:
        values = [a["costs"].get(key) for a in attempts]
        totals[key] = sum(x for x in values if x is not None)
        unknown[key] = sum(x is None for x in values)
    costs = {
        "recorded_sums": totals,
        "attempts_with_unknown_cost": unknown,
        "allocation_snapshot": clock(),
        "physics_scope": "Post-app-launch PhysX subscription. Pre-subscription ticks unknown.",
        "invalid_cost_rule": "Persisted counters and complete trace records give lower bounds.",
        "deduplication": "Each raw attempt directory once; ignore derived audit/report copies.",
        "component_time_rule": "Overlapping server/simulator durations are not allocated GPU-hours.",
    }
    planned = directory / "comparison-plan.json"
    schedule = json.loads(planned.read_text())["jobs"] if planned.exists() else []
    states = []
    for job in schedule:
        found = [a for a in attempts if a.get("episode_id") == job["episode_id"]]
        states.append(
            {
                **job,
                "status": "not_started"
                if not found
                else "valid_completed"
                if found[0]["valid"]
                else "invalid_or_incomplete",
            }
        )
    validation = [a for a in attempts if a.get("purpose") == "validation" and a["valid"]]
    categories = ("safe complete", "unsafe complete", "safe unfinished", "unsafe unfinished")
    colors = ("#228833", "#cc6677", "#88ccee", "#aa4499")

    def category(row):
        return (0 if row["C_req"] else 2) + int(row["V"])

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    summary = []
    for scene, axis in zip(COMPARISON, axes[0], strict=True):
        counts = np.array(
            [
                [
                    sum(
                        a["scene"] == scene and a["dial"] == d and category(a) == k
                        for a in validation
                    )
                    for d in (0, 0.5, 1)
                ]
                for k in range(4)
            ]
        )
        base = np.zeros(3)
        for k, label in enumerate(categories):
            axis.bar([0, 1, 2], counts[k], bottom=base, label=label, color=colors[k])
            base += counts[k]
        axis.set(
            xticks=[0, 1, 2],
            xticklabels=["0", "0.5", "1"],
            xlabel="Dial strength",
            ylabel="Audited episode counts",
            title=scene,
            ylim=(0, max(1, max(base)) + 0.5),
        )
        for x, n in enumerate(base):
            axis.text(x, n + 0.05, f"n={int(n)}", ha="center")
    for dial in (0, 0.5, 1):
        rows = [a for a in validation if a["dial"] == dial]
        summary.append(
            {
                "dial": dial,
                "n": len(rows),
                "S_req": sum(a["S_req"] for a in rows),
                "C_req": sum(a["C_req"] for a in rows),
                "V": sum(a["V"] for a in rows),
                "unfinished": sum(not a["C_req"] for a in rows),
                "categories": dict(Counter(categories[category(a)] for a in rows)),
                "queries": sum(a["costs"]["policy_queries"] for a in rows),
                "process_seconds": sum(a["process"]["elapsed_seconds"] for a in rows),
            }
        )
    if validation:
        for key, label, color in (
            ("S_req", "Safe requested completion", "#228833"),
            ("V", "Raw physical violation", "#cc6677"),
        ):
            axes[1, 0].plot(
                [x["dial"] for x in summary],
                [x[key] / x["n"] if x["n"] else np.nan for x in summary],
                "o",
                label=label,
                color=color,
            )
        for row in summary:
            axes[1, 0].annotate(f"n={row['n']}", (row["dial"], 1.02), ha="center")
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, "No audited full comparison episodes", ha="center", va="center")
    axes[1, 0].set(
        xlabel="Dial strength",
        ylabel="Observed proportion",
        ylim=(-0.05, 1.15),
        xticks=[0, 0.5, 1],
        title="Measured points only",
    )
    axes[1, 1].axis("off")
    lines = [
        f"d={r['dial']}: n={r['n']}, queries={r['queries']}, process={r['process_seconds']:.1f}s"
        for r in summary
    ]
    lines += [
        f"Attempts: {len(attempts)}; valid full comparison: {len(validation)}",
        f"Allocated GPU hours at audit: {costs['allocation_snapshot']['allocated_gpu_hours']:.3f}",
    ]
    axes[1, 1].text(0.02, 0.9, "\n".join(lines), va="top", fontsize=10)
    axes[0, 1].legend(fontsize=8)
    figure.savefig(output / "dial-outcomes.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 4), layout="constrained")
    measured = []
    for path in sorted(directory.glob("population-*-result.json")):
        pop = json.loads(path.read_text())
        for slot, candidate in enumerate(pop["population"]):
            pair = pop["outcomes"][slot]
            completed_at = max(a["process"]["finished_at"] for a in pair)
            cumulative = sum(
                a["costs"].get("physics_steps") or 0
                for a in attempts
                if a.get("process")
                and a["process"].get("finished_at", float("inf")) <= completed_at
            )
            value = pop["ranking"]["scores"][slot]
            axis.plot(
                cumulative,
                value,
                "s" if candidate["kind"] == "elite_reevaluation" else "o",
                color="#4477aa",
            )
            measured.append(
                {
                    "population": path.name,
                    "slot": slot,
                    **candidate,
                    "fitness": value,
                    "cumulative_recorded_physics_ticks": cumulative,
                    "score_available_at": completed_at,
                }
            )
    if not measured:
        axis.text(
            0.5,
            0.5,
            "No complete audited search population",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set(
        xlabel="Cumulative recorded physics ticks (including engineering and failed attempts)",
        ylabel="Mean utility",
        title="Search lineage; square = physically re-evaluated elite",
    )
    figure.savefig(output / "search-lineage.png", dpi=160)
    plt.close(figure)
    atomic(output / "attempt-ledger.json", {"attempts": attempts, "costs": costs})
    atomic(output / "comparison.json", {"summary": summary, "planned_job_states": states})
    atomic(output / "lineage.json", measured)
    atomic(output / "videos.json", videos)
    matched = []
    for scene in COMPARISON:
        for seed in sorted({v["seed"] for v in videos if v["scene"] == scene}):
            block = [
                v
                for v in videos
                if v["scene"] == scene and v["seed"] == seed and v["purpose"] == "validation"
            ]
            matched.append(
                {
                    "scene": scene,
                    "seed": seed,
                    "videos": block,
                    "all_three_settings_present": {v["dial"] for v in block} == {0, 0.5, 1},
                }
            )
    atomic(output / "matched-videos.json", matched)
    manifest = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    atomic(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "attempts": len(attempts),
                "validation": len(validation),
                "summary": summary,
                "costs": costs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.directory.resolve(), args.output.resolve())
