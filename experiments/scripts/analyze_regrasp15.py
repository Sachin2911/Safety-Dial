"""Task 15 outcomes: the frozen stage C choice (--decide) and the final analysis.

Reads only validated episodes from the queue ledgers. The stage C arm is the stage B
closure-gated selection arm with the most strict safe completions, ties broken by fewer
raw-violation episodes and then by the 0.10 m/s limit; projection-0.10 replaces it if
that arm had at least three more violation episodes than native. The primary contrast
is stage C strict safe completion, selected arm minus native: one-sided exact Barnard
test at alpha 0.05, Wilson intervals and a task-stratified bootstrap. Stages A and B
are descriptive. Nothing pools across stages or with earlier tasks.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/regrasp15"
DESIGN = ROOT / "configs/regrasp15/design.json"
SELECTION_ARMS = ("select-0.10", "select-0.06")
OUTCOMES = ("strict_safe", "success", "raw_violation", "counted_violation", "unfinished",
            "engaged", "ever_grasped", "closure_gate_opened", "post_gate_tip",
            "post_gate_stall", "post_gate_success")
ACTIVITY = ("gated", "vetoes", "closest_passing", "least_excess", "projected",
            "candidate_decodes")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episodes(stage):
    path = OUT / f"attempts_{stage}.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(x) for x in path.read_text().splitlines() if x]
    return [r for r in rows if r["event"] == "validated"]


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def summarize(rows):
    by_arm = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
    summary = {}
    for arm, members in sorted(by_arm.items()):
        n = len(members)
        counts = {key: sum(bool(r["outcome"][key]) for r in members) for key in OUTCOMES}
        opened = counts["closure_gate_opened"]
        summary[arm] = {
            "episodes": n,
            **counts,
            "strict_safe_wilson95": wilson(counts["strict_safe"], n),
            "post_gate_conversion": counts["post_gate_success"] / opened if opened else None,
            "activity": {key: sum(r["adapter"][key] for r in members) for key in ACTIVITY},
            "by_task": {
                str(task): {
                    "episodes": sum(r["task"] == task for r in members),
                    "strict_safe": sum(r["outcome"]["strict_safe"] for r in members
                                       if r["task"] == task),
                    "raw_violation": sum(r["outcome"]["raw_violation"] for r in members
                                         if r["task"] == task),
                }
                for task in sorted({r["task"] for r in members})
            },
        }
    return summary


def decide():
    design = json.loads(DESIGN.read_text())
    stage_b = summarize(episodes("stage_b"))
    native = stage_b.get("native")
    ranked = sorted(
        (arm for arm in SELECTION_ARMS if arm in stage_b),
        key=lambda arm: (-stage_b[arm]["strict_safe"], stage_b[arm]["raw_violation"],
                         arm != "select-0.10"),
    )
    if not ranked or native is None:
        raise SystemExit("Stage B has no validated native or selection episodes")
    chosen = ranked[0]
    rule = "most_strict_safe_selection_arm"
    if stage_b[chosen]["raw_violation"] >= native["raw_violation"] + 3:
        chosen, rule = "project-0.10", "selection_arm_violations_exceeded_native_by_3"
    decision = {
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "arm": chosen,
        "rule": rule,
        "ranking": ranked,
        "stage_b": {arm: {k: stage_b[arm][k] for k in ("episodes", "strict_safe",
                                                         "raw_violation", "success")}
                    for arm in stage_b},
        "design_sha256": sha256(DESIGN),
        "evidence_sha256": {name: sha256(OUT / name)
                            for name in ("attempts_stage_a.jsonl", "attempts_stage_b.jsonl")},
        "stage_c_episodes_per_arm": len(design["stage_c_seeds"]) * len(design["stage_c_tasks"]),
    }
    (OUT / "stage-c-decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision))


def bootstrap(arm_rows, native_rows, replicates=10_000, seed=15):
    rng = np.random.default_rng(seed)
    tasks = sorted({r["task"] for r in arm_rows} & {r["task"] for r in native_rows})
    arm = {t: np.array([r["outcome"]["strict_safe"] for r in arm_rows if r["task"] == t],
                       dtype=float) for t in tasks}
    nat = {t: np.array([r["outcome"]["strict_safe"] for r in native_rows if r["task"] == t],
                       dtype=float) for t in tasks}
    n_arm = sum(len(v) for v in arm.values())
    n_nat = sum(len(v) for v in nat.values())
    draws = np.empty(replicates)
    for i in range(replicates):
        a = sum(rng.choice(arm[t], size=len(arm[t])).sum() for t in tasks)
        b = sum(rng.choice(nat[t], size=len(nat[t])).sum() for t in tasks)
        draws[i] = a / n_arm - b / n_nat
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def contrast(rows, arm):
    from scipy.stats import barnard_exact

    arm_rows = [r for r in rows if r["arm"] == arm]
    native_rows = [r for r in rows if r["arm"] == "native"]
    s_arm = sum(r["outcome"]["strict_safe"] for r in arm_rows)
    s_nat = sum(r["outcome"]["strict_safe"] for r in native_rows)
    n_arm, n_nat = len(arm_rows), len(native_rows)
    if not n_arm or not n_nat:
        return None
    # Columns are the two samples, rows success and failure; "greater": p_arm > p_native.
    table = [[s_arm, s_nat], [n_arm - s_arm, n_nat - s_nat]]
    test = barnard_exact(table, alternative="greater")
    return {
        "arm": arm,
        "arm_strict_safe": [s_arm, n_arm],
        "native_strict_safe": [s_nat, n_nat],
        "difference": s_arm / n_arm - s_nat / n_nat,
        "barnard_one_sided_p": float(test.pvalue),
        "significant_at_0.05": bool(test.pvalue < 0.05),
        "arm_wilson95": wilson(s_arm, n_arm),
        "native_wilson95": wilson(s_nat, n_nat),
        "difference_bootstrap95_by_task": bootstrap(arm_rows, native_rows),
    }


def final():
    analysis = {"utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    stage0 = OUT / "stage0/summary.json"
    if stage0.exists():
        analysis["stage0"] = json.loads(stage0.read_text())
    for stage in ("stage_a", "stage_b", "stage_c"):
        analysis[stage] = summarize(episodes(stage))
    decision_path = OUT / "stage-c-decision.json"
    if decision_path.exists():
        decision = json.loads(decision_path.read_text())
        analysis["stage_c_decision"] = decision
        analysis["primary"] = contrast(episodes("stage_c"), decision["arm"])
    (OUT / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({k: analysis.get(k) for k in ("primary",)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decide", action="store_true")
    args = parser.parse_args()
    if args.decide:
        decide()
    else:
        final()


if __name__ == "__main__":
    main()
