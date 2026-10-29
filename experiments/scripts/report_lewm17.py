"""Write the task-17 report and exportable figures from audited final statistics."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs/lewm17/cache/matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from helpers.lewm17_protocol import OUT, write_json

LABELS = {
    "historical": "Task-16 fixed",
    "tuned": "Retuned fixed",
    "random": "Random search",
    "reinforce": "REINFORCE",
    "evolution": "Evolution",
}
COLORS = {
    "historical": "#838a91",
    "tuned": "#203b55",
    "random": "#e39c35",
    "reinforce": "#6e55a5",
    "evolution": "#158f85",
}


def percent(value):
    return f"{100 * value:.1f}"


def figures(stats):
    output = OUT / "analysis"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for axis, (env, margin) in zip(
        axes.flat, [("tworoom", "4"), ("tworoom", "6"), ("reacher", "0.01"), ("reacher", "0.015")]
    ):
        for index, method in enumerate(LABELS):
            data = stats["endpoint_intervals"][env][margin][method]
            axis.bar(index, 100 * data["estimate"], color=COLORS[method], alpha=0.75)
            means = np.array(data["run_means"]) * 100
            axis.scatter(
                np.linspace(index - 0.13, index + 0.13, len(means)),
                means,
                color="black",
                s=16,
                zorder=4,
            )
        axis.set_title(f"{env.title()}, clearance {margin} {'px' if env == 'tworoom' else 'm'}")
        axis.set_xticks(range(5), [LABELS[m] for m in LABELS], rotation=20, ha="right")
        axis.set_ylim(0, 105)
        axis.set_ylabel("Requested-safe completion (%)")
    fig.suptitle("Fresh final cases; dots show independent optimizer runs")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"safe-completion.{suffix}", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 3.5))
    for index, method in enumerate(("random", "reinforce", "tuned")):
        row = stats["evolution_minus"]["reacher"]["0.015"][method]
        point, (low, high) = row["estimate"], row["interval"]
        axis.errorbar(
            100 * point,
            index,
            xerr=[[100 * (point - low)], [100 * (high - point)]],
            fmt="o",
            color=COLORS[method],
            capsize=4,
        )
    axis.axvline(0, color="black", linewidth=0.8)
    axis.axvline(10, color="grey", linewidth=0.8, linestyle="--")
    axis.set_yticks(
        range(3), [f"Evolution minus {LABELS[m]}" for m in ("random", "reinforce", "tuned")]
    )
    axis.set_xlabel("Paired safe-completion difference (percentage points)")
    axis.set_title("Reacher 1.5 cm: 98.33% crossed run/case bootstrap intervals")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"primary-contrasts.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, env in zip(axes, ("tworoom", "reacher")):
        for method in ("random", "reinforce", "evolution"):
            for replicate in range(5):
                history = json.loads(
                    (
                        OUT
                        / "optimizers"
                        / f"{env}-{method}-r{replicate}"
                        / "learning-history.json"
                    ).read_text()
                )
                resource = "analytic_path_steps" if env == "tworoom" else "checker_physics_steps"
                axis.plot(
                    [h["actual_resources"][resource] / 1e6 for h in history],
                    [h["mean_utility"] for h in history],
                    color=COLORS[method],
                    alpha=0.5,
                    label=LABELS[method] if replicate == 0 else None,
                )
        axis.set_title(env.title())
        axis.set_xlabel("Consumed checker steps, millions (including anchors)")
        axis.set_ylabel("Training batch utility")
        axis.legend(fontsize=8)
    fig.suptitle(
        "Descriptive training curves; not validation or equal-resource efficiency estimates"
    )
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"training-resources.{suffix}", dpi=180)
    plt.close(fig)


def main():
    stats = json.loads((OUT / "analysis/statistics.json").read_text())
    assert json.loads((ROOT / "progress/receipts/17-final-audit.json").read_text())["passed"]
    figures(stats)
    lines = ["# Task 17: LeWM evolutionary-controller comparison", "", "26 October 2026.", ""]
    primary = stats["final_tables"]["reacher"]["0.015"]
    scores = ", ".join(
        f"{LABELS[m]} {primary[m]['requested_safe_completion']}/{primary[m]['n']}"
        for m in ("evolution", "random", "reinforce", "tuned")
    )
    conclusion = (
        "The frozen practical completion-gain criterion passed. Safety and scope limitations below still apply."
        if stats["practical_completion_gain_gate"]
        else "The frozen criterion for an evolutionary completion advantage was not met."
    )
    lines += [
        (
            f"All {stats['episodes']:,} admitted episodes completed, including thirty independent optimizer runs and "
            f"{stats['final_episodes']:,} final episodes. {conclusion} On the primary Reacher 1.5 cm endpoint: {scores}."
        ),
        "",
        "## Design and execution",
        "",
        "The optimizers are established methods: [Salimans et al., ES](https://arxiv.org/abs/1703.03864) and [Williams, REINFORCE](https://doi.org/10.1007/BF00992696). The frozen task-17 recipe specifies this implementation; applying them here is not an algorithmic novelty claim.",
        "",
        (
            "The compact alternative was frozen before optimizer outcomes: five independent runs per method and environment, "
            "six rounds, and the first twelve fresh final cases per environment. Each run paid for 24 deterministic common-anchor "
            "episodes, 144 stochastic training episodes and 24 deterministic validation episodes. The final panel used two requested "
            "clearances and two fresh planner seeds. Fixed references were each evaluated once per case/clearance/seed, not copied five times."
        ),
        "",
        (
            "The initial p90-per-episode scheduling estimate rejected both proposed workloads. Before training, the scheduler packed "
            "individual runs across two lanes per GPU and used the measured all-mode mean with a 50% timing contingency, a further "
            "40% evaluation imbalance allowance, and 30 minutes for startup/reporting. This admitted the predeclared compact workload "
            "and rejected the full workload. The initial rejection receipt is retained. No outcomes determined panel size or optimizer settings."
        ),
        "",
        (
            "All adaptive methods used the same 36 bounded coefficients, six observation inputs and six planning modes. LeWM's "
            "18,034,478 parameters per environment, scalers, five-action executable block, 50-action limit and exact checker stayed frozen. "
            "Clearance was an input, never an output. The task-side selector used no true position, target geometry or candidate endpoint. "
            "The exact safety checker retained its declared state and dynamics access. Training used softmax sampling; selection and final "
            "deployment used deterministic argmax. The fixed tuning and historical-information anchors are distinct comparators."
        ),
        "",
        (
            "Training used requested-safe completion minus 0.1 times failure-capped J/50. Random search re-evaluated its incumbent with seven "
            "fresh candidates on every common batch. Evolution used four antithetic Gaussian directions. REINFORCE used only its current "
            "24 trajectories and a running scalar baseline. Every method had the same four-finalist validation allowance and a shared anchor fallback. "
            "Unsafe candidates were disqualified before selection. The final parameters were frozen before final case deserialization and evaluation."
        ),
        "",
        (
            "Panels were generated from disjoint task-17 RNG streams using task-16 geometry ranges and endpoint validity checks. No case was filtered "
            "by success. Final geometry was present only in its sealed file; source integrity checks hashed that file without deserializing it in "
            "training. Final access and the complete parameter selection have separate receipts. Geometric validity does not prove 50-action feasibility."
        ),
        "",
        "A pre-final seed audit found that the first proposed final planner pair overlapped the last replicate's training seeds. Before any final-case access or evaluation, a [recorded amendment](17-final-planner-seed-amendment.json) replaced it with 179710 and 179711, disjoint from every planned nonfinal planner seed. Panels, counts, selection and inference rules stayed fixed.",
        "",
        "## Final outcomes",
        "",
        "Counts below retain failures, censored attempts and every accepted case. Movement defines engagement, not progress toward the goal.",
        "",
        "| Environment | Clearance | Method | Safe completion | Original violations | Margin breaches | Unfinished | Engaged | Censored | Mean capped J |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for env, margins in stats["final_tables"].items():
        for margin, table in margins.items():
            for method, label in LABELS.items():
                r, n = table[method], table[method]["n"]
                values = [
                    f"{r[k]}/{n}"
                    for k in (
                        "requested_safe_completion",
                        "any_violation",
                        "margin_breach",
                        "unfinished",
                        "engaged",
                        "censored",
                    )
                ]
                lines.append(
                    f"| {env} | {margin} | {label} | "
                    + " | ".join(values)
                    + f" | {r['J_mean']:.2f} |"
                )
    lines += [
        "",
        "## Contemporary protocol context",
        "",
        "This small development panel is descriptive context, separate from the reserved final comparison.",
        "",
        "| Environment | Clearance | Context method | Requested-safe | Benchmark-safe | Violations | Margin breaches | Unfinished | Engaged |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for env, margins in stats["context_tables"].items():
        for margin, conditions in margins.items():
            for condition, row in conditions.items():
                values = [
                    f"{row[k]}/{row['n']}"
                    for k in (
                        "requested_safe_completion",
                        "benchmark_safe_completion",
                        "any_violation",
                        "margin_breach",
                        "unfinished",
                        "engaged",
                    )
                ]
                lines.append(f"| {env} | {margin} | {condition} | " + " | ".join(values) + " |")
    lines += [
        "",
        (
            "Benchmark-safe completion is also retained separately in the machine-readable statistics. The requested margins are "
            "4 and 6 pixels in TwoRoom, and 1 and 1.5 cm in Reacher. The checker enforced the full requested margin in every checked episode."
        ),
        "",
        "## Primary paired comparisons",
        "",
        (
            "The practical target was a ten-percentage-point completion gain over each of random search, REINFORCE and the newly tuned fixed "
            "reference. The intervals resample whole optimizer runs and whole case blocks independently while retaining paired methods and both "
            "planner seeds. The three primary contrasts use 98.33% intervals; other descriptive intervals use 95%."
        ),
        "",
        "| Evolution minus | Difference, percentage points | 98.33% interval |",
        "| --- | ---: | --- |",
    ]
    for method in ("random", "reinforce", "tuned"):
        r = stats["evolution_minus"]["reacher"]["0.015"][method]
        lines.append(
            f"| {LABELS[method]} | {percent(r['estimate'])} | [{percent(r['interval'][0])}, {percent(r['interval'][1])}] |"
        )
    lines += [
        "",
        (
            "Five optimizer repeats and twelve case blocks have limited power. A ceiling can make the bootstrap interval degenerate. "
            "Zero breaches in n episodes gives an approximate 3/n episode-wise upper bound, not a zero-risk guarantee; repeated cases and trained "
            "controllers further limit independence. No transfer, general safety or algorithmic novelty claim follows from this campaign."
        ),
        "",
        "## Every optimizer run",
        "",
        "Method labels identify the complete selection procedure. When validation selects an anchor or the initial coefficients, the deployed controller was not produced by optimizer updates. The provenance column makes this visible; gains from a shared anchor are not evidence that evolution discovered a better controller.",
        "",
        "| Environment | Method | Run | Selected candidate | Final safe completion, low margin | Final safe completion, high margin |",
        "| --- | --- | ---: | --- | ---: | ---: |",
    ]
    for r in stats["optimizer_runs"]:
        env, method, rep = r["env"], r["method"], r["replicate"]
        ms = ("4", "6") if env == "tworoom" else ("0.01", "0.015")
        values = [stats["endpoint_intervals"][env][m][method]["run_means"][rep] for m in ms]
        lines.append(
            f"| {env} | {LABELS[method]} | {rep} | {r['selected_label']} | {percent(values[0])}% | {percent(values[1])}% |"
        )
    diagnostics = {}
    for method in ("random", "reinforce", "evolution"):
        histories = [
            json.loads(p.read_text())
            for p in (OUT / "optimizers").glob(f"*-{method}-r*/learning-history.json")
        ]
        flat = [r for history in histories for r in history]
        diagnostics[method] = {
            "rounds": len(flat),
            "coefficient_update_steps": 0 if method == "random" else len(flat),
            "world_model_gradient_updates": 0,
            "mean_reward_variance": float(np.mean([r["reward_variance"] for r in flat])),
            "mean_entropy": float(np.mean([r["entropy"] for r in flat])),
            "nonzero_coefficient_updates": sum(r["coefficient_change_l2"] > 1e-12 for r in flat),
            "mean_gradient_norm": float(np.mean([r["gradient_norm_before_clip"] for r in flat])),
            "mode_counts": np.sum([r["mode_counts"] for r in flat], axis=0).tolist(),
        }
    write_json(OUT / "analysis/learner-diagnostics.json", diagnostics)
    lines += [
        "",
        "Validation selected shared anchors in eight of ten evolution runs and eight of ten REINFORCE runs. Both optimizers changed coefficients in all sixty rounds and exercised all six modes, so the negative result is not explained by an inactive update loop. The gradient and allocation tests passed. The small validation panel, six-round search and train/deploy sampling difference limit what this result says about either optimizer beyond this frozen recipe.",
        "",
        "## Learning diagnostics and resources",
        "",
        "Round-to-round training utility also changes with the common case batch. These curves describe the search trajectories and do not estimate held-out deployment improvement. Per-episode execution seconds and per-run elapsed wall_seconds, including model startup and result writing, are retained separately from the resource counters in the machine-readable evidence.",
        "",
        (
            "The analytical REINFORCE gradient passed a finite-difference expected-reward test, and complete synthetic runs verified all three "
            "optimizers' charged allocations. Live diagnostics record reward variance, entropy, coefficient changes, gradient norms and actual selected "
            "modes for every round. No reward, step size, panel or stopping rule was changed after optimizer outcomes."
        ),
        "",
        "| Method | Completed rounds | Mean reward variance | Mean entropy | Nonzero coefficient updates | Mean gradient norm |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, d in diagnostics.items():
        lines.append(
            f"| {LABELS[method]} | {d['rounds']} | {d['mean_reward_variance']:.5f} | {d['mean_entropy']:.3f} | {d['nonzero_coefficient_updates']} | {d['mean_gradient_norm']:.3f} |"
        )
    lines += [
        "",
        (
            "REINFORCE and evolution each execute one coefficient-update step per completed round; there are no world-model gradient updates. Random search does not calculate a gradient; its stored zero gradient/entropy diagnostics are placeholders. Selection and round "
            "checkpoints retain the actual candidate weights, scores and mode counts."
        ),
        "",
        "| Resource, all attempts | Count |",
        "| --- | ---: |",
    ]
    for name, count in stats["all_attempt_resources"].items():
        lines.append(f"| {name.replace('_', ' ')} | {count:,} |")
    lines.append(
        f"| Frozen model batch calls, reconstructed from recorded iterations | {stats['model_batch_query_calls_derived']:,} |"
    )
    lines += [
        "",
        (
            "Equal episode counts were not treated as equal simulator interaction work. Each optimizer run had the same enforced action, predictor "
            "and checker ceilings, including anchors and validation. Early completion and selected horizons changed actual consumption. The resource "
            "vector and learning curves expose those differences. Saved replan records distinguish proposed CEM means and checked selected blocks; state traces record actual commands. Analysis separately reconstructs simulator-bounded controls with the frozen [-1,1] clipping, not joint torques. Validation was not repeated at common consumed-resource checkpoints, so this study "
            "does not support an equal-work or search-efficiency claim. Counterfactual Reacher physics and TwoRoom analytic path steps are charged "
            "separately from executed actions. Episode reset counts exclude implicit simulator construction/forward operations, which are not controlled "
            "physics integrations. Native engineering parity probes had additional counted predictor allowance beyond the base ticket reservation."
        ),
        "",
        "Shared verification also performed 14,499,085 synthetic analytic path checks, derived from recorded test-suite executions and fixed solver loop shapes. These used dummy costs: zero live simulator integrations and zero frozen-LeWM model queries. They are separate from the measured episode resource vector above; see the [synthetic test-work receipt](17-synthetic-test-work.json).",
        "",
        "## Validation, preservation and rental closeout",
        "",
        (
            "All 48 engineering episodes passed native identity, constant-mode parity, rendering and dense checker replay. Seventeen focused implementation "
            "tests passed. The final audit independently recomputed dense violations, completion labels, action/physics counts, optimizer resource totals, "
            "selection ordering and the complete final job set. No failed rollout starts occurred in the admitted campaign. Historical task-16 files and "
            "all other task allocations were left unchanged."
        ),
        "",
        (
            "Every completed batch was uploaded to the existing private dataset `Sachioster/maniguard-jar-replication-20261022`, under "
            "`17_lewm17_campaign/57809511/`. Each archive was downloaded at its pinned revision and hash-verified. Source archives, all attempts, model "
            "identities, panels, optimizer checkpoints, reports and the Odyssey are included. Credentials, caches and downloaded frozen weights are excluded."
        ),
        "",
        (
            "The final completion-status receipt records the last observed conservative cost and verified backup. The authorized finish command hands "
            "terminal preservation and self-destruction to the independent guard. A finish request or accepted provider deletion request is not external "
            "absence verification. Final absence and posted billing must be checked from another device or the Vast console."
        ),
        "",
        (
            "Reproduce with `experiments/scripts/analyze_lewm17.py` and `experiments/scripts/report_lewm17.py` using `/venv/main/bin/python` on the restored "
            "private snapshot. See [statistics](17-final-statistics.json), [audit](17-final-audit.json), [comparison freeze](17-optimizer-comparison.json), "
            "[final freeze](17-final-evaluation.json), and the [campaign](../17-lewm-evolution-campaign.md). Exportable figures are under "
            "[safe completion](../../outputs/lewm17/analysis/safe-completion.pdf), [primary contrasts](../../outputs/lewm17/analysis/primary-contrasts.pdf), and [training resources](../../outputs/lewm17/analysis/training-resources.pdf)."
        ),
        "",
    ]
    (ROOT / "progress/receipts/17-lewm-evolution-results.md").write_text("\n".join(lines))
    print(conclusion)


if __name__ == "__main__":
    main()
