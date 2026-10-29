"""Write manuscript tables directly from the audited task-18 aggregate record."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = {
    "native": "Native",
    "canonical": "Original checked",
    "fixed": "Tuned fixed",
    "feedback": "Feedback",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = json.loads((args.root / "analysis/summary.json").read_text())
    assert not report["missing_result_attempts"]
    assert set(report["environments"]) == {"cube", "pusht"}
    rows = report["tables"]
    paper = ROOT / "docs/research/honours-gecco"
    lines = [
        r"\section{PushT and Cube: Complete Matched Comparisons}\label{app:extension}",
        "Table~\\ref{tab:extension-all} gives every final comparison. Each cell uses",
        "the same 12 cases and three planner seeds. A boundary breach and an unfinished",
        "task can occur in the same attempt, so those columns are not a partition.",
        r"\begin{table*}[t]",
        r"\caption{Task-18 final results, 36 attempts per cell. Safe completion, requested-margin",
        "breaches, unfinished attempts and engagement are reported together. Lower capped",
        "actions is better; all unsafe or unfinished attempts cost 50 actions.}",
        r"\label{tab:extension-all}\centering\small",
        r"\begin{tabular}{llrrrrr}\toprule",
        r"Task and margin & Planner & Safe & Breaches & Unfinished & Moved & Capped actions \\",
        r"\midrule",
    ]
    numerical = []
    for kind in ("pusht", "cube"):
        margins = sorted({r["margin"] for r in rows if r["stage"] == "H" and r["env"] == kind})
        for margin in margins:
            label = f"PushT, {margin:g} px" if kind == "pusht" else f"Cube, {margin * 100:g} cm"
            for method, name in NAMES.items():
                r = next(
                    r
                    for r in rows
                    if (r["stage"], r["env"], r["margin"], r["method"])
                    == ("H", kind, margin, method)
                )
                assert r["n"] == 36 and r["error"] == 0
                lines.append(
                    f"{label} & {name} & {r['safe']} & {r['margin_breach']} & "
                    f"{r['unfinished']} & {r['engaged']} & {r['mean_J_safe']:.2f} " + r"\\"
                )
                numerical.append(
                    {
                        k: r[k]
                        for k in (
                            "env",
                            "margin",
                            "method",
                            "n",
                            "safe",
                            "margin_breach",
                            "unfinished",
                            "engaged",
                            "mean_J_safe",
                        )
                    }
                )
                label = ""
            lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [
        r"\end{tabular}",
        r"\end{table*}",
        "",
        r"\begin{table}[t]",
        r"\caption{Serial planning time at the highest clearance, two prespecified cases per",
        "method with a separate common seed. Seconds per solve includes a failed final",
        "solve. These descriptive timings do not estimate a population speed advantage.}",
        r"\label{tab:extension-time}\centering\small",
        r"\begin{tabular}{llrr}\toprule",
        r"Task & Planner & Seconds/solve & Safe / $n$ \\ \midrule",
    ]
    for kind in ("pusht", "cube"):
        for method, name in NAMES.items():
            r = next(r for r in rows if (r["stage"], r["env"], r["method"]) == ("T", kind, method))
            assert r["n"] == 2 and r["error"] == 0
            lines.append(
                f"{'PushT' if kind == 'pusht' else 'Cube'} & {name} & "
                f"{r['planning_seconds_per_replan']:.3f} & {r['safe']}/2 " + r"\\"
            )
    lines += [r"\bottomrule\end{tabular}", r"\end{table}", ""]
    lines += [
        r"\paragraph{All charged work, including the repair.}",
        "Each valid task cohort contains 8 engineering, 216 fixed-development,",
        "72 feedback-development, 360 unique final, 12 context and 8 serial-timing",
        "attempts. Together they contain 1,352 attempts. The excluded original",
        "PushT cohort adds 676. Preliminary failed checks, model warm-ups and",
        "reset work are also charged in the following totals.",
        r"\begin{table}[t]\centering\small",
        r"\caption{Task-18 work across both valid cohorts, the excluded PushT cohort and preliminary checks.}",
        r"\begin{tabular}{lr}\toprule Quantity & Count \\ \midrule",
    ]
    work = report["all_charged_work"]
    for label, key in (
        ("Executed actions, excluding reset", "environment_actions_excluding_reset"),
        ("Reset stabilization actions", "reset_stabilization_actions"),
        ("Environment resets", "environment_resets"),
        ("All physics integrations", "physics_integration_steps_including_reset_and_previews"),
        ("Of which counterfactual previews", "counterfactual_physics_steps"),
        ("Learned predictor steps", "predictor_steps"),
        ("Model batch calls", "model_batch_calls"),
    ):
        lines.append(f"{label} & {work[key]:,} " + r"\\")
    lines += [
        r"\bottomrule\end{tabular}\end{table}",
        "Original PushT and pre-data reset work is reconstructed from the pinned",
        "reset implementation; the repaired cohort measures it directly. Cube's",
        "fifty reset physics steps per attempt are already included in its physics",
        "counter and are not added twice. Analytic candidate checks are separate",
        "from true physics integrations; the complete counters remain in the dataset.",
        "",
    ]
    (paper / "lewm18-tables.tex").write_text("\n".join(lines))
    (args.root / "analysis/paper-table-validation.json").write_text(
        json.dumps(
            {
                "source": "analysis/summary.json",
                "final_cells": numerical,
                "timing_cells": 8,
                "method": "Tables generated directly from audited aggregates",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"final_cells": len(numerical), "timing_cells": 8}))


if __name__ == "__main__":
    main()
