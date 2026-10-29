"""Create a short LaTeX report from a complete panel and a reviewed interpretation."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

LABELS = {
    "d00_g100": "N: native",
    "d15_g100": "E: delay 15",
    "d40_g100": "F: delay 40",
    "reference_random": "R: earlier random-search setting",
    "history_only": "H: history-only delay",
    "grasp_latch": "L: grasp latch",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tex(text):
    special = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(special.get(char, char) for char in str(text))


def label(condition):
    if condition in LABELS:
        return LABELS[condition]
    delay, gain = condition.split("_g")
    return f"Delay {int(delay[1:])}, translation {int(gain) / 100:g}"


def checked_data(analysis, validation, freeze, freeze_sha, accounting):
    if not validation["passed"] or validation["freeze_sha256"] != freeze_sha:
        raise ValueError("The report requires validation of this exact freeze")
    stage = freeze["stage"]
    if stage not in {"B", "C"} or analysis["stage"] != stage:
        raise ValueError("Only a complete performance stage can be reported")
    rows = validation["episodes"]
    expected = {job["id"] for job in freeze["jobs"]}
    actual = [row["job"]["id"] for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("The full frozen panel is required")
    if set(analysis["conditions"]) != {row["condition"] for row in rows}:
        raise ValueError("Analysis and validation conditions differ")
    if not accounting["all_attempts_terminal"] or not accounting["all_caps_respected"]:
        raise ValueError("Final accounting must cover terminal attempts within the caps")
    attempted = Counter(
        row["job"] for row in accounting["attempts"] if row["stage"] == stage
    )
    for cid, summary in analysis["conditions"].items():
        group = [row for row in rows if row["condition"] == cid]
        n = 16 if stage == "B" else 64
        if len(group) != n or summary["valid"] != n:
            raise ValueError("Condition sample size differs from the frozen panel")
        if summary["attempted"] != sum(attempted[row["job"]["id"]] for row in group):
            raise ValueError("Attempted denominator differs from accounting")
        if summary["outcomes"] != dict(Counter(row["outcome"] for row in group)):
            raise ValueError("Outcome counts differ from the validated traces")
        for key, flag in (("C_count", "safe_success"), ("V_count", "violation")):
            if summary[key] != sum(row[flag] for row in group):
                raise ValueError(f"{key} differs from validation")
        if abs(summary["J"] - sum(row["J"] for row in group) / n) > 1e-12:
            raise ValueError("Time aggregate differs from validation")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("analysis", "validation", "freeze", "accounting", "interpretation", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--figure", type=Path, help="Optional figure placed below the table")
    args = parser.parse_args()
    inputs = {name: getattr(args, name) for name in (
        "analysis", "validation", "freeze", "accounting", "interpretation"
    )}
    if args.figure:
        inputs["figure"] = args.figure.resolve(strict=True)
    analysis, validation, freeze, accounting = (
        json.loads(inputs[name].read_text())
        for name in ("analysis", "validation", "freeze", "accounting")
    )
    rows = checked_data(analysis, validation, freeze, sha(args.freeze), accounting)
    interpretation = args.interpretation.read_text().strip()
    if not interpretation or len(interpretation.split()) > 220:
        raise ValueError("Supply a reviewed plain-text interpretation of at most 220 words")
    if any(char in interpretation for char in "\u2013\u2014"):
        raise ValueError("Use plain hyphens in the interpretation")
    stage = freeze["stage"]
    conditions = analysis["conditions"]
    phase = "Exploratory mechanism screen" if stage == "B" else "Independent confirmation"
    table = []
    for cid in sorted(conditions):
        row = conditions[cid]
        counts = row["outcomes"]
        table.append(
            f"{tex(label(cid))} & {row['attempted']}/{row['valid']} & "
            f"{row['C_count']} & {counts.get('physical_failure', 0)} & "
            f"{counts.get('boundary_exit', 0)} & {counts.get('timeout', 0)} & "
            f"{row['J']:.2f} " + r"\\"
        )
    motion = [row["engagement_means"]["eef_path_length_m"] for row in conditions.values()]
    engagement = (
        f"Mean end-effector travel ranged from {100 * min(motion):.1f} to "
        f"{100 * max(motion):.1f} cm across settings. Motion is engagement, not completion."
    )
    if stage == "C":
        continuation = "; ".join(
            f"{label(cid)} {row['sustained_count']}/{row['valid']}"
            for cid, row in sorted(conditions.items())
        )
        continuation = (
            "Sustained safe completion through 40 further controls: " + continuation + "."
        )
    else:
        continuation = (
            "This screen has no post-success continuation. The engineering continuation "
            "check is not a performance estimate."
        )
    invalid = sum(row["invalid"] for row in accounting["stages"].values())
    accounting_text = (
        f"Across all retained stages: {len(accounting['attempts'])} attempts, including "
        f"{invalid} infrastructure-invalid starts; "
        f"{accounting['known_completed_controls']:,} observed controls and "
        f"{accounting['explicit_controls_conservatively_charged']:,} conservatively charged. "
        f"Recorded online/shadow controller-query bundles: "
        f"{accounting['recorded_controller_query_bundles_started']:,}/"
        f"{accounting['recorded_shadow_query_bundles_started']:,}. "
        f"At least {accounting['source_derived_backend_warmup_steps_lower_bound']:,} "
        "backend warmup updates are counted separately; full internal-physics totals "
        "remain unmeasured. "
        "All attempts remain in the ledger; invalid starts are excluded from valid denominators."
    )
    paragraphs = "\n\n".join(tex(part) for part in interpretation.split("\n\n"))
    source = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=1.9cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,booktabs,parskip,graphicx}
\usepackage[hidelinks]{hyperref}
\hypersetup{pdftitle={UNISafe block-plucking follow-up},pdfauthor={Sachin Mohan}}
\setlength{\parskip}{6pt}
\setlength{\parindent}{0pt}
\pagestyle{empty}
\begin{document}
{\LARGE\bfseries UNISafe block-plucking follow-up}\par
{\large PHASE}\par
{\small Sachin Mohan \enspace | \enspace 25 October 2026}

INTERPRETATION

The frozen policy and learned safety filter were tested on the released block-plucking
task at 20 Hz, with a 15-second horizon and the full released success predicate.
The settings change release delay, grasp sensing or translation before filtering.
Each setting uses the same SEEDS new initial-condition seeds, in frozen randomized blocks.
No model weights or evolutionary optimizer were updated.

\begin{center}
\small
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Setting & A/V & Complete & Failure & Boundary & Timeout & $J$ (s) \\
\midrule
TABLE
\bottomrule
\end{tabular}
\end{center}
{\small A/V: attempted/valid episodes. Complete means safe benchmark completion.
$J$ is mean completion time with 15 seconds assigned to each failure or unfinished
episode. Failure and boundary counts together give the violation count.}

FIGURE
ENGAGEMENT

CONTINUATION

{\small ACCOUNTING}

\vfill
{\footnotesize These results concern one simulated task. Evolutionary advantage and
real-robot transfer require separate evidence. The full response map, paired uncertainty
analysis where applicable, raw traces and attempt ledger are retained with task 9.
Evidence: \path{ANALYSIS_PATH} and matching
validation/accounting receipts.}
\end{document}
"""
    replacements = {
        "PHASE": tex(phase), "INTERPRETATION": paragraphs,
        "SEEDS": str(len({row["seed"] for row in rows})), "TABLE": "\n".join(table),
        "ENGAGEMENT": tex(engagement), "CONTINUATION": tex(continuation),
        "ACCOUNTING": tex(accounting_text), "STAGE": stage,
        "ANALYSIS_PATH": str(args.analysis),
        "FIGURE": (r"\begin{center}\includegraphics[width=\linewidth]{"
                   + str(Path(os.path.relpath(inputs["figure"], args.output.parent)))
                   + r"}\end{center}\vspace{-4pt}") if args.figure else "",
    }
    for token, value in replacements.items():
        source = source.replace(token, value)
    if args.output.suffix != ".tex":
        raise ValueError("Output must be LaTeX; compile and visually inspect the PDF separately")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source)
    manifest = {
        "stage": stage, "inputs": {str(path): sha(path) for path in inputs.values()},
        "script_sha256": sha(Path(__file__)), "latex_sha256": sha(args.output),
        "requires_pdf_render_and_visual_review": True,
    }
    args.output.with_suffix(".inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"stage": stage, "output": str(args.output)}))


if __name__ == "__main__":
    main()
