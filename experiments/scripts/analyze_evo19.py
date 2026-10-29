"""Analyze task 19: CMA-ES search trajectory and the evolved-versus-native confirmation.

Reads outputs/evo19/search/state.json (the orchestrator's record of every job and its
validated outcome) and writes outputs/evo19/analysis/{summary.json,summary.md,figure.png}.
Search episodes are training data; only the confirmation panel estimates effects.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "outputs/evo19/search/state.json"
OUT = ROOT / "outputs/evo19/analysis"
ARMS = ("confirm_native", "confirm_evolved", "confirm_handset")


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def tally(jobs):
    valid = [j for j in jobs if j.get("outcome")]
    o = [j["outcome"] for j in valid]
    n = len(valid)
    safe = sum(x["strict_safe"] for x in o)
    return {
        "episodes": len(jobs), "valid": n, "invalid": len(jobs) - n,
        "strict_safe": safe, "strict_safe_rate": round(safe / n, 4) if n else None,
        "strict_safe_wilson95": wilson(safe, n),
        "success": sum(x["success"] for x in o),
        "raw_violation": sum(x["raw_violation"] for x in o),
        "post_gate_tip": sum(x["post_gate_tip"] for x in o),
        "post_gate_stall": sum(x["post_gate_stall"] for x in o),
        "ever_grasped": sum(x["ever_grasped"] for x in o),
        "gate_opened": sum(x["closure_gate_opened"] for x in o),
        "unfinished": sum(x["unfinished"] for x in o),
        "engaged": sum(x["engaged"] for x in o),
        "projected_chunks": sum((j.get("adapter") or {}).get("projected", 0) for j in valid),
        "mean_elapsed_s": round(float(np.mean([j["elapsed_s"] for j in valid])), 1) if n else None,
    }


def category(o):
    if o["strict_safe"]:
        return "strict_safe"
    if o["raw_violation"]:
        return "violation"
    return "stall_after_closure" if o["closure_gate_opened"] else "never_closed"


def contrast(pairs, native, evolved):
    """One-sided exact tests: evolved more strict safe, and evolved fewer violations."""
    n = len(pairs)
    out = []
    for key, better in (("strict_safe", "evolved"), ("raw_violation", "native")):
        a = [bool(native[p]["outcome"][key]) for p in pairs]
        b = [bool(evolved[p]["outcome"][key]) for p in pairs]
        ka, kb = sum(a), sum(b)
        # Table rows: the arm expected to have more events first.
        table = [[kb, n - kb], [ka, n - ka]] if better == "evolved" else [[ka, n - ka],
                                                                          [kb, n - kb]]
        fisher = stats.fisher_exact(table, alternative="greater").pvalue
        # scipy's barnard_exact treats each column as one binomial sample, so the arms
        # go in columns; the earlier row orientation conditioned on outcome totals.
        barnard = stats.barnard_exact([list(c) for c in zip(*table)],
                                      alternative="greater").pvalue
        only_b = sum(y and not x for x, y in zip(a, b))
        only_a = sum(x and not y for x, y in zip(a, b))
        hits, total = (only_b, only_a + only_b) if better == "evolved" else (only_a,
                                                                              only_a + only_b)
        mcnemar = (stats.binomtest(hits, total, 0.5, alternative="greater").pvalue
                   if total else None)
        out.append({
            "endpoint": key, "paired_task_seeds": n, "native": ka, "evolved": kb,
            "native_rate": round(ka / n, 4), "evolved_rate": round(kb / n, 4),
            "native_wilson95": wilson(ka, n), "evolved_wilson95": wilson(kb, n),
            "difference_evolved_minus_native": round((kb - ka) / n, 4),
            "fisher_one_sided_p": round(float(fisher), 5),
            "barnard_one_sided_p": round(float(barnard), 5),
            "pairs_only_native": only_a, "pairs_only_evolved": only_b,
            "exact_mcnemar_one_sided_p": None if mcnemar is None else round(float(mcnemar), 5),
            "by_task": {str(t): {"pairs": sum(p[0] == t for p in pairs),
                                 "native": sum(bool(native[p]["outcome"][key])
                                               for p in pairs if p[0] == t),
                                 "evolved": sum(bool(evolved[p]["outcome"][key])
                                                for p in pairs if p[0] == t)}
                        for t in sorted({p[0] for p in pairs})}})
    return out[0], out[1]


def main():
    state = json.loads(STATE.read_text())
    jobs = list(state["jobs"].values())
    done = [j for j in jobs if j["status"] == "done"]
    by_kind = defaultdict(list)
    for job in done:
        by_kind[job["kind"]].append(job)
    stages = {k: [j for j in by_kind["search"] if j.get("stage", 1) == k] for k in (1, 2, 3, 4)}
    summary = {"generations": [], "search": tally(by_kind["search"]),
               **{f"search_stage{k}": tally(v) for k, v in stages.items()},
               "stage_closes": {k: state.get(f"stage{k}") for k in (1, 2, 3)},
               "extension": state.get("extension"),
               "evolved": state.get("evolved")}
    rows_by_stage = [(1, r) for r in state["generations"]] + [
        (k, r) for k in (2, 3, 4) for r in state.get(f"generations{k}", [])]
    for stage, row in rows_by_stage:
        gen_jobs = [state["jobs"][j] for j in row["jobs"]]
        summary["generations"].append({
            "stage": stage, "generation": row["generation"],
            "mean_settings": row["mean_settings"],
            "sigma": row["sigma"], "scores": row.get("scores"),
            "strict_safe": sum(bool(j.get("outcome") and j["outcome"]["strict_safe"])
                               for j in gen_jobs),
            "violations": sum(bool(j.get("outcome") and j["outcome"]["raw_violation"])
                              for j in gen_jobs),
            "complete": all(j["status"] == "done" for j in gen_jobs),
            "told": "scores" in row})
    # Descriptive speed-limit bands over all valid search episodes (training data).
    bands = [(0.0, 0.06), (0.06, 0.09), (0.09, 0.13), (0.13, 0.2), (0.2, 1.0)]
    summary["search_speed_bands"] = []
    for low, high in bands:
        for stage, group in ((1, stages[1]), (2, stages[2])):
            inside = [j for j in group if j.get("outcome")
                      and low <= j["settings"]["speed_limit_mps"] < high]
            if inside:
                summary["search_speed_bands"].append({"stage": stage, "band_mps": [low, high],
                                                      **tally(inside)})
    for arm in ARMS:
        summary[arm] = tally(by_kind[arm])
    # Descriptive response landscape over all search episodes (training data, not effects):
    # logistic fits of strict safe completion and raw violation on the standardized genes.
    rows = [j for j in stages[1] + stages[2]
            if j.get("outcome") and j["outcome"]["closure_gate_opened"]]
    if len(rows) >= 20:  # projection stages only (the shield has no speed limit)
        x = np.array([[math.log(j["settings"]["speed_limit_mps"]), j["settings"]["near_radius_m"],
                       j["settings"]["retreat_weight"]] for j in rows])
        z = (x - x.mean(0)) / x.std(0)
        design = np.column_stack([np.ones(len(z)), z])
        landscape = {"episodes_with_gate_open": len(rows), "genes": ["log speed limit",
                     "near radius", "retreat weight"], "standardized": True}
        for name in ("strict_safe", "raw_violation", "post_gate_stall"):
            y = np.array([float(j["outcome"][name]) for j in rows])
            beta = np.zeros(design.shape[1])
            for _ in range(50):  # ridge-stabilized Newton steps
                p = 1 / (1 + np.exp(-design @ beta))
                w = p * (1 - p) + 1e-9
                hessian = design.T @ (design * w[:, None]) + 0.5 * np.eye(design.shape[1])
                beta += np.linalg.solve(hessian, design.T @ (y - p) - 0.5 * beta)
            landscape[name] = {"rate": round(float(y.mean()), 4),
                               "coefficients": [round(float(b), 3) for b in beta]}
        summary["search_landscape"] = landscape
    # Primary contrast on task-seed pairs where both arms have a valid episode; amendment 6
    # also reports the original 72-pair panel (seeds 193000..193017) alone.
    native = {(j["task"], j["seed"]): j for j in by_kind["confirm_native"] if j.get("outcome")}
    evolved = {(j["task"], j["seed"]): j for j in by_kind["confirm_evolved"] if j.get("outcome")}
    pairs = sorted(set(native) & set(evolved))
    if pairs:
        summary["primary"], summary["secondary_violations"] = contrast(pairs, native, evolved)
        panel = [p for p in pairs if p[1] < 193018]
        if panel:
            summary["panel72_primary"], summary["panel72_violations"] = contrast(
                panel, native, evolved)
        summary["outcome_mix"] = {
            arm: dict(sorted(Counter(category(table[p]["outcome"]) for p in pairs).items()))
            for arm, table in (("native", native), ("evolved", evolved))}
        summary["violation_timing"] = {
            arm: dict(Counter(("before_closure" if table[p]["outcome"]["first_closed_step"] is None
                               or table[p]["outcome"]["violation_step"]
                               < table[p]["outcome"]["first_closed_step"] else "after_closure")
                              for p in pairs if table[p]["outcome"]["raw_violation"]))
            for arm, table in (("native", native), ("evolved", evolved))}
        summary["evolved_shield_activity"] = {
            "episodes_with_retrace": sum(bool((evolved[p].get("adapter") or {}).get("projected"))
                                         for p in pairs),
            "retrace_chunks": sum((evolved[p].get("adapter") or {}).get("projected", 0)
                                  for p in pairs)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["| Arm | Valid | Strict safe | Wilson 95% | Violations | Tips | Stalls | Grasped |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for arm in ("search_stage1", "search_stage2", "search_stage3", "search_stage4") + ARMS:
        t = summary[arm]
        if not t["episodes"]:
            continue
        lines.append(f"| {arm} | {t['valid']}/{t['episodes']} | {t['strict_safe']} | "
                     f"{t['strict_safe_wilson95']} | {t['raw_violation']} | {t['post_gate_tip']} | "
                     f"{t['post_gate_stall']} | {t['ever_grasped']} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if "primary" in summary:
        for key in ("primary", "secondary_violations", "outcome_mix", "violation_timing",
                    "evolved_shield_activity"):
            print(key, json.dumps(summary[key]))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        for stage, marker in ((1, "o--"), (2, "s--")):
            gens = [r for r in summary["generations"] if r["stage"] == stage]
            if not gens:
                continue
            g = [r["generation"] for r in gens]
            axes[0].plot(g, [r["mean_settings"]["speed_limit_mps"] for r in gens], marker,
                         label=f"stage {stage}")
            axes[1].plot(g, [r["mean_settings"]["near_radius_m"] for r in gens], marker,
                         label=f"stage {stage}")
        axes[0].set(title="Projection stages: mean speed limit (m/s)", xlabel="generation")
        axes[1].set(title="Projection stages: mean near radius (m)", xlabel="generation")
        for ax in axes[:2]:
            ax.legend(frameon=False, fontsize=8)
        gens3 = [r for r in summary["generations"] if r["stage"] == 3]
        if gens3:
            g = [r["generation"] for r in gens3]
            twin = axes[2].twinx()
            axes[2].plot(g, [r["mean_settings"]["tilt_limit_deg"] for r in gens3], "o-",
                         color="#2f6db5", label="stage 3 tilt limit (deg)")
            twin.plot(g, [100 * r["mean_settings"]["retreat_m"] for r in gens3], "s-",
                      color="#c9853a", label="stage 3 retreat (cm)")
            axes[2].set(title="Stage 3 shield: mean genes", xlabel="generation", ylim=(0, 16))
            twin.set(ylim=(0, 13))
            axes[2].legend(frameon=False, fontsize=8, loc="upper left")
            twin.legend(frameon=False, fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(OUT / "search.png", dpi=150)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for stage, marker in ((1, "o--"), (2, "s--"), (3, "o-")):
            told = [r for r in summary["generations"] if r["stage"] == stage and r["told"]]
            if told:
                ax.plot([r["generation"] for r in told],
                        [r["strict_safe"] / len(r["scores"]) for r in told], marker,
                        label=f"stage {stage}")
        ax.set(title="Strict safe fraction per generation", xlabel="generation", ylim=(0, 1))
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "search-success.png", dpi=150)
        arms = [(a, label) for a, label in (("confirm_native", "native"),
                                            ("confirm_evolved", "evolved (CMA-ES mean)"),
                                            ("confirm_handset", "hand-set (x0)"))
                if summary[a]["valid"]]
        if arms:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
            for i, (arm, label) in enumerate(arms):
                t = summary[arm]
                rate = t["strict_safe_rate"]
                low, high = t["strict_safe_wilson95"]
                axes[0].bar(i, rate, color=("#8a8f98", "#2f6db5", "#c9853a")[i])
                axes[0].errorbar(i, rate, yerr=[[rate - low], [high - rate]], color="black",
                                 capsize=4)
                axes[0].text(i, high + 0.02, f"{t['strict_safe']}/{t['valid']}", ha="center")
                parts = [t["strict_safe"], t["raw_violation"], t["post_gate_stall"],
                         t["valid"] - t["gate_opened"] - (t["raw_violation"] - t["post_gate_tip"])]
                bottom = 0
                for value, colour, name in zip(parts, ("#3a9d5d", "#c0392b", "#e6b33a", "#bbbbbb"),
                                               ("strict safe", "violation", "post-gate stall",
                                                "never closed")):
                    axes[1].bar(i, value / t["valid"], bottom=bottom, color=colour,
                                label=name if i == 0 else None)
                    bottom += value / t["valid"]
            for ax in axes:
                ax.set_xticks(range(len(arms)), [label for _, label in arms])
            axes[0].set(title="Strict safe completion (Wilson 95%)", ylim=(0, 1))
            axes[1].set(title="Outcome mix", ylim=(0, 1.0))
            axes[1].legend(frameon=False, fontsize=8, loc="upper right")
            fig.tight_layout()
            fig.savefig(OUT / "confirmation.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
