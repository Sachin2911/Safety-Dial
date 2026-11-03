"""Render audited development results on their rented host; no new rollouts."""

import hashlib
import json
import os
from pathlib import Path
import socket
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/safelewm-development-closeout-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRICS = ["safe_completion", "success", "violation", "unfinished", "engaged"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(groups):
    lines = [
        "| Policy | Safe completion | Completion | Violations | Unfinished | Engaged |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, g in groups.items():
        lines.append("| " + name + " | " + " | ".join(f"{g[k]}/{g['n']}" for k in METRICS) + " |")
    return lines


def dial(base, out, analysis, audit):
    title = "Safety-dial library: prospective development check"
    curves = analysis["curves"]
    methods = [
        ("main", "CMA-ES", "#246b8b"),
        ("random", "Random search", "#c17a32"),
        ("ppo", "PPO", "#6f5b97"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True, sharey="row")
    native = analysis["groups"]["shared-native"]
    multi = []
    missing = 0
    exceeded = 0
    available = 0
    for col, rid in enumerate(["R0", "R1", "R2"]):
        for suffix, label, color in methods:
            key = f"{rid}-head-{suffix}"
            curve = curves[key]
            points = curve["points"]
            selected = {p["candidate"] for p in points if p["available"]}
            if len(selected) > 1:
                multi.append((key, curve))
            missing += sum(not p["available"] for p in points)
            available += sum(p["available"] for p in points)
            exceeded += sum(p.get("empirical_target_exceeded", False) for p in points)
            x = [100 * p["epsilon"] for p in points]
            for row, metric in enumerate(["violation", "safe_completion"]):
                y = [100 * p[metric] / p["n"] if p["available"] else np.nan for p in points]
                axes[row, col].plot(x, y, "o-", label=label, color=color, markersize=4)
        axes[0, col].plot(
            [0, 100], [0, 100], "--", color="#999999", linewidth=0.8, label="Requested limit"
        )
        axes[0, col].set_title(rid)
        for row, metric in enumerate(["violation", "safe_completion"]):
            axes[row, col].axhline(
                100 * native[metric] / native["n"], color="#333333", linestyle=":", label="Native"
            )
            axes[row, col].set_ylim(-2, 102)
            axes[row, col].set_xticks([0, 25, 50, 75, 100])
        axes[1, col].set_xlabel("Requested violation limit (%)")
    axes[0, 0].set_ylabel("Observed violations (%)")
    axes[1, 0].set_ylabel("Safe completion (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    stem = "17-safety-dial-library"
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    monotonic = sum(c["nondecreasing_observed_risk"] for _, c in multi)
    synopsis = dict(
        multi_policy_curves=len(multi),
        nondecreasing_multi_policy_curves=monotonic,
        missing_target_assignments=missing,
        available_target_assignments=available,
        empirical_exceedances=exceeded,
        independent_target_trials=False,
    )
    lines = [
        "# " + title,
        "",
        "All 772 episode records passed audit: 16 engineering and 756 development episodes.",
        "The 12 development layouts were previously observed. This is not a fresh final test.",
        "",
        f"Of {len(multi)} curves selecting more than one policy, {monotonic} had nondecreasing observed violation counts as the requested limit increased.",
        f"There were {missing}/45 unavailable target assignments. Among {available} available assignments, {exceeded} exceeded their requested empirical violation limit on development evaluation.",
        "Several targets reuse the same evaluated actor. These target counts are not independent trials or a calibration test.",
        "",
        f"![Development risk and safe completion](graphs/{stem}.png)",
        "",
        "## Every target assignment",
        "",
        "| Repeat / optimizer | Limit | Policy | Safe | Complete | Violated | Unfinished | Engaged | Empirical limit exceeded |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, curve in curves.items():
        for p in curve["points"]:
            if not p["available"]:
                lines.append(f"| {key} | {p['epsilon']} | unavailable | - | - | - | - | - | - |")
            else:
                counts = " | ".join(f"{p[k]}/{p['n']}" for k in METRICS)
                lines.append(
                    f"| {key} | {p['epsilon']} | {p['candidate']} | {counts} | {p['empirical_target_exceeded']} |"
                )
    lines += [
        "",
        "## Unique policies and references",
        "",
        *table(analysis["groups"]),
        "",
        "## Interpretation and accounting",
        "",
        "All 17 selected actors were locked before evaluation, alongside shared native and three original untrained references. PPO actors were evaluated deterministically. R2 was trained on L4 and evaluated here on the same 4090 as R0/R1.",
        "The library was acquired with the original weighted utility, not trained to the requested risk limits. Its previous acquisition cost was 360,000 actions, separately charged. No new optimizer or cache was trained.",
        "A flat or nonmonotonic curve and an unavailable target remain results. Neither monotonicity nor a low empirical violation count proves calibrated risk or reduced manual tuning.",
        "",
        f"Execution: {analysis['seconds']:.1f} seconds. New simulator actions: {analysis['resources']['environment_actions']}; resets: {analysis['resources']['environment_resets']}.",
        "",
        "Frozen protocol: [SafetyDialLibrary.md](../SafetyDialLibrary.md). All rental shutdowns remain manual.",
    ]
    return stem, title, lines, synopsis


def evolution(base, out, analysis, audit):
    title = "CMA population size under a matched action budget"
    groups = analysis["groups"]
    names = [
        ("untrained", "Untrained", "#7c8890"),
        ("cma8", "CMA population 8", "#246b8b"),
        ("cma4", "CMA population 4", "#4d8d71"),
        ("random", "Random search", "#c17a32"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), sharey=True)
    for ax, rid in zip(axes, ["E0", "E1"]):
        for index, (arm, label, color) in enumerate(names):
            g = groups[f"{rid}-{arm}"]
            ax.bar(
                index - 0.14,
                g["safe_completion"],
                width=0.26,
                color=color,
                label="Safe completion" if index == 0 else None,
            )
            ax.bar(
                index + 0.14,
                g["violation"],
                width=0.26,
                color=color,
                alpha=0.35,
                label="Any violation" if index == 0 else None,
            )
            ax.text(
                index - 0.14,
                g["safe_completion"] + 0.4,
                str(g["safe_completion"]),
                ha="center",
                fontsize=8,
            )
            ax.text(
                index + 0.14, g["violation"] + 0.4, str(g["violation"]), ha="center", fontsize=8
            )
        ax.axhline(
            groups["shared-native"]["safe_completion"],
            color="#333333",
            linestyle="--",
            label="Native safe",
        )
        ax.set_xticks(range(4), ["Untrained", "CMA8", "CMA4", "Random"])
        ax.set_title(rid)
        ax.set_ylim(0, 39)
    axes[0].set_ylabel("Episodes / 36")
    axes[1].legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8
    )
    fig.tight_layout()
    stem = "18-evolution-budget"
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"{stem}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    summary = json.loads((base / "run-v1/summary.json").read_text())
    searches = summary["searches"]
    lines = [
        "# " + title,
        "",
        "Two development repeats compared CMA populations four and eight with parameter random search.",
        "Every optimizer used exactly 40,000 actual training/selection actions per repeat, 240,000 total.",
        "",
        f"**Frozen development advancement gate: {'passed' if analysis['development_gate_passed'] else 'failed'}.**",
        "Passing requires CMA4 to exceed both CMA8 and random by at least two safe completions in each repeat, without higher violations than CMA8 in either repeat.",
        "",
        *table(groups),
        "",
        f"![Matched-budget population comparison](graphs/{stem}.png)",
        "",
        "## Search work",
        "",
        "| Search | Actions | Complete CMA updates | Eligible candidates | Quota-cut candidates |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, s in searches.items():
        lines.append(
            f"| {key} | {s['actions']} | {s['updates']} | {s['eligible']} | {s['partial']} |"
        )
    lines += [
        "",
        "## Limits and accounting",
        "",
        "The 12 validation layouts and three planner seeds are previously observed development data. This is not a new held-out comparison. All three nominees were frozen within repeat before validation.",
        "Native is shared. The 324 validation episodes are repeated evaluations on the same layouts; their denominators are not independent tasks. All 16 native/neutral engineering checks passed.",
        "Only population size changed between CMA arms: diagonal adaptation, sigma 0.5, explicit CSA, 132 head coordinates and the original utility stayed fixed. A positive development gate would require new confirmation before a reliable superiority claim; failure does not authorize another sweep.",
        "",
        f"Audited records: {audit['episodes']}. Simulator actions: {analysis['resources']['environment_actions']}; resets: {analysis['resources']['environment_resets']}; execution: {analysis['seconds']:.1f} seconds.",
        "",
        "Frozen protocol: [EvolutionBudgetStudy.md](../EvolutionBudgetStudy.md). Rental shutdown remains manual.",
    ]
    synopsis = dict(
        development_gate_passed=analysis["development_gate_passed"],
        searches=searches,
        groups=groups,
    )
    return stem, title, lines, synopsis


def main():
    assert sys.argv[1:] in [["dial"], ["evolution"]]
    kind = sys.argv[1]
    host = "075cb160c9b3" if kind == "dial" else "8cf3a706cdd1"
    assert socket.gethostname() == host
    name = "safety-dial-library-20261102" if kind == "dial" else "evolution-budget-20261102"
    base = ROOT / "outputs" / name
    audit = json.loads((base / "run-v1/audit.json").read_text())
    analysis = json.loads((base / "run-v1/analysis.json").read_text())
    assert (
        audit["passed"]
        and json.loads((base / "run-v1/summary.json").read_text())["status"] == "complete"
    )
    out = base / "closeout"
    out.mkdir(exist_ok=False)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    stem, title, lines, synopsis = (dial if kind == "dial" else evolution)(
        base, out, analysis, audit
    )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "closeout-summary.json").write_text(
        json.dumps(dict(title=title, **synopsis), indent=2) + "\n"
    )
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    provenance = dict(
        source_sha256=digest(Path(__file__)),
        analysis_sha256=digest(base / "run-v1/analysis.json"),
        audit_sha256=digest(base / "run-v1/audit.json"),
        additional_simulator_actions=0,
        host=host,
        matplotlib=matplotlib.__version__,
        numpy=np.__version__,
    )
    (out / "figure-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    files = {
        p.name: dict(sha256=digest(p), bytes=p.stat().st_size)
        for p in sorted(out.iterdir())
        if p.is_file()
    }
    (out / "manifest.json").write_text(json.dumps(dict(files=files), indent=2) + "\n")
    print(json.dumps(dict(study=name, figure=stem, files=len(files))), flush=True)


if __name__ == "__main__":
    main()
