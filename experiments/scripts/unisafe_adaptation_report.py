"""Export compact evidence and outcome figures from the completed UNISafe study."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    evaluation = args.evaluation
    paths = {
        "analysis": evaluation / "analysis/summary.json",
        "physical_audit": evaluation / "analysis/audit/summary.json",
        "plan": evaluation / "test-plan.json",
        "batch": evaluation / "test/summary.json",
        "handoff": evaluation / "handoff.json",
        "ledger": args.ledger,
    }
    records = {key: read(path) for key, path in paths.items()}
    analysis, plan, ledger = (records[key] for key in ("analysis", "plan", "ledger"))
    assert analysis["status"] == ledger["status"] == records["batch"]["status"] == "complete"
    assert analysis["prefix_audit_all_pass"] and len(analysis["prefix_checks"]) == 320
    assert all(row["pass"] for row in analysis["prefix_checks"])
    assert not ledger["incomplete_workers"] and records["batch"]["failed_job_count"] == 0
    assert len(plan["jobs"]) == records["batch"]["completed_episodes"] == 384
    for relative, expected in plan["frozen_sources_sha256"].items():
        assert digest(root / relative) == expected, relative
    for group in analysis["groups"].values():
        episodes = group["episodes_detail"]
        assert Counter(e["outcome"] for e in episodes) == group["outcomes"]
        assert len(episodes) == group["episodes"]
        assert sum(e["steps"] for e in episodes) == group["environment_steps"]
    assert sum(g["environment_steps"] for g in analysis["groups"].values()) == 47039
    # Independent exact-test cross-check against SciPy's binomial implementation.
    for contrast in analysis["contrasts"].values():
        for metric in ("safe_success", "physical_failure", "any_violation"):
            c = contrast[metric]
            wins, losses = c["candidate_only_positive"], c["reference_only_positive"]
            p = binomtest(wins, wins + losses).pvalue if wins + losses else 1.0
            assert abs(p - c["mcnemar_exact_two_sided_p"]) < 1e-14
    assert ledger["new_physical_completed"]["episodes"] == 1424
    assert ledger["new_physical_completed"]["environment_steps"] == 166697
    assert ledger["new_physical_completed"]["physics_steps"] == 833485

    ablation = {}
    for name in ("baseline", "evolution", "baseline-unfiltered", "evolution-unfiltered"):
        episodes = [e for e in analysis["groups"][name]["episodes_detail"] if e["seed"] < 1032]
        assert len(episodes) == 32
        ablation[name] = dict(Counter(e["outcome"] for e in episodes))
    failures = []
    for e in sorted(analysis["groups"]["evolution"]["episodes_detail"], key=lambda e: e["seed"]):
        if e["outcome"] == "success":
            continue
        directory = evaluation / f"test/evolution-seed{e['seed']}/episode-0000"
        paths[f"case_{e['seed']}_summary"] = directory / "summary.json"
        paths[f"case_{e['seed']}_trace"] = directory / "steps.jsonl"
        summary = read(directory / "summary.json")
        rows = [json.loads(line) for line in (directory / "steps.jsonl").read_text().splitlines()]
        failures.append({
            **e, "final_terms": summary["final_terms"], "final_state": summary["final_state"],
            "first_override_step": next((r["step"] for r in rows
                                         if r["adapter_gripper_override"]), None),
            "first_intervention_step": next((r["step"] for r in rows if r["intervened"]), None),
        })
    videos = {}
    for name in ("video-improvement", "video-failure"):
        paths[name] = evaluation / name / "provenance.json"
        videos[name] = read(paths[name])
        assert digest(evaluation / name / "comparison.mp4") == videos[name]["movie_sha256"]
    evidence = {
        "status": "complete_reviewed", "date": "2026-10-16",
        "scope": "Selected controllers on the migrated simulator; no optimizer superiority claim.",
        "analysis": {k: v for k, v in analysis.items()
                     if k not in ("artifact_sha256", "prefix_checks")},
        "ablation_outcomes_first_32_seeds": ablation,
        "prefix_audit": {"count": 320, "all_pass": True},
        "physical_audit": {k: records["physical_audit"][k] for k in
                           ("status", "artifact_audit", "environment_steps", "physics_steps")},
        "frozen_test_contract": {k: v for k, v in plan.items() if k != "jobs"},
        "ledger": {k: v for k, v in ledger.items() if k != "workers"},
        "evolution_unfinished_and_violating_cases": failures,
        "videos": videos,
        "source_artifacts": {k: {"path": str(p), "sha256": digest(p)} for k, p in paths.items()},
        "exporter_sha256": digest(Path(__file__)),
        "export_cost": {"environment_steps": 0, "model_queries": 0, "training_updates": 0},
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "results.json").write_text(json.dumps(evidence, indent=2) + "\n")
    demo = {
        "workers": 1,
        "scope": "Recorded seed-1000 example, now known. A rerun is demonstration, not new holdout evidence.",
        "source_commit": plan["source_commit"],
        "frozen_sources_sha256": plan["frozen_sources_sha256"],
        "jobs": [j for j in plan["jobs"] if j["name"] in
                 ("baseline-seed1000", "evolution-seed1000")],
    }
    assert len(demo["jobs"]) == 2
    (args.output / "demonstration-plan.json").write_text(json.dumps(demo, indent=2) + "\n")

    colors = {"success": "#167d75", "physical_failure": "#e7a13e",
              "boundary_exit": "#a33948", "timeout": "#d6dbe3"}
    labels = {"success": "Success", "physical_failure": "Physical failure",
              "boundary_exit": "Boundary exit", "timeout": "Timeout"}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    panels = [
        (axes[0], ["baseline", "fixed", "random", "policy_gradient", "evolution"],
         ["Frozen baseline", "Fixed rule", "Random search", "Policy gradient", "Evolution"],
         {n: g["outcomes"] for n, g in analysis["groups"].items()}, 64,
         "Selected controllers with the filter"),
        (axes[1], list(ablation),
         ["Baseline, filtered", "Evolution, filtered", "Baseline, unfiltered",
          "Evolution, unfiltered"], ablation, 32, "Filter ablation on the same 32 starts"),
    ]
    for ax, names, ticks, groups, n, title in panels:
        left = [0] * len(names)
        for outcome, color in colors.items():
            values = [groups[name].get(outcome, 0) for name in names]
            ax.barh(ticks, values, left=left, height=.65, color=color, label=labels[outcome])
            for i, (start, count) in enumerate(zip(left, values, strict=True)):
                if count:
                    ax.text(start + count / 2, i, str(count), ha="center", va="center",
                            fontsize=9, color="white" if outcome in
                            ("success", "boundary_exit") else "#15202b")
            left = [a + b for a, b in zip(left, values, strict=True)]
        ax.set(xlim=(0, n), xlabel="Episodes", title=title)
        ax.invert_yaxis()
        ax.set_xticks(range(0, n + 1, n // 4))
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .065))
    fig.suptitle("UNISafe grasp maintenance improves completion; violations remain", fontsize=15)
    fig.text(.5, .025, "Parameters selected before testing. One controller per method; "
             "instantaneous benchmark termination.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .15, 1, .93), w_pad=2)
    fig.savefig(args.output / "outcomes.png", dpi=180)
    fig.savefig(args.output / "outcomes.svg")
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "episodes": 384, "prefix_checks": 320,
                      "environment_steps_added": 0, "independent_exact_test_check": "passed"}))


if __name__ == "__main__":
    main()
