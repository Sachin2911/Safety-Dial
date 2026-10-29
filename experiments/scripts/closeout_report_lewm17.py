"""Write scientific completion records and the Odyssey before the final verified backup."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm17_protocol import OUT, write_json
from helpers.lewm17_remote import finance, rental, utc


def main():
    stats = json.loads((OUT / "analysis/statistics.json").read_text())
    audit = json.loads((ROOT / "progress/receipts/17-final-audit.json").read_text())
    assert audit["passed"] and audit["all_attempts_complete"] == 7800
    assert (ROOT / "progress/receipts/17-lewm-evolution-results.md").exists()
    primary = stats["final_tables"]["reacher"]["0.015"]
    conclusion = (
        "The frozen practical completion-gain criterion passed."
        if stats["practical_completion_gain_gate"]
        else "An evolutionary completion advantage was not established."
    )
    text_scores = ", ".join(
        f"{method} {primary[method]['requested_safe_completion']}/{primary[method]['n']}"
        for method in ("evolution", "random", "reinforce", "tuned")
    )
    note = f"""# LeWM evolution, task 17

26 October 2026. {conclusion}

The complete compact campaign ran 7,800 episodes. All three optimizers used the same
36-coefficient selector around frozen LeWM, with five independent runs per environment.
Training and validation allowances were identical. The selected controllers then faced
twelve fresh cases per environment, two clearances and two new planner seeds.

At the primary Reacher 1.5 cm clearance: {text_scores}.
The paired crossed run/case intervals and every optimizer run are in the
[results](../../../progress/receipts/17-lewm-evolution-results.md). The final denominator
includes every accepted case, unfinished episode and censored attempt. The native and
canonical context uses a separate development panel and is not pooled with final cases.

Actual simulator, checker and predictor consumption differs despite equal enforced
allowances. Validation was not scheduled at common consumed-resource checkpoints, so
these data cannot establish a search-efficiency advantage. Five optimizer repeats and
twelve case blocks give limited precision. Zero observed breaches is not zero risk.
The exact checker retains privileged state/dynamics access; the task-side controller
receives no true target geometry. No transfer or general safety claim is supported.

This campaign's result is separate from task 16's configuration lever and the earlier
ManiGuard studies. Historical evidence and other rentals were left unchanged. All raw
attempts, checkpoints, source freezes and reports are preserved in the existing private
research dataset under `17_lewm17_campaign/57809511/`. The independent guard owns final
preservation and rental self-destruction after the authorized finish request. External
provider absence and posted billing remain separate checks from another device.
"""
    (ROOT / "docs/research/research_notes/LeWM evolution task 17.md").write_text(note)
    path = ROOT / "progress/theOdessy.tex"
    source = path.read_text()
    if "\\section*{Task 17:" in source:
        raise FileExistsError("Task-17 Odyssey entry already exists")
    entry = r"""
\newpage
\section*{Task 17: Comparing evolutionary LeWM planning controllers}

\textbf{26 October 2026. Why.} Task 16 established a useful fixed planner
configuration, but no evolutionary advantage. This campaign compared parameter
random search, REINFORCE and antithetic evolution in one shared compact controller
class, with native, canonical and strong fixed references.

\textbf{What ran.} The prospectively admitted compact schedule completed all
7,800 episodes: 48 engineering, 72 context, 288 fixed tuning, 5,760 optimizer
anchor/training/validation episodes and 1,632 final episodes. Each method had five
independent runs per environment. All used 36 coefficients around frozen LeWM.
Twelve new final cases, two requested margins and two planner seeds were kept out
of training and selection. Clearances were never relaxed.

\textbf{Result.} @CONCLUSION@ At Reacher's 1.5 cm clearance, safe completion was
@EVOLUTION@ for evolution, @RANDOM@ for random search, @REINFORCE@ for REINFORCE
and @TUNED@ for newly tuned fixed control. The report retains every optimizer run,
paired run/case uncertainty, violations, unfinished episodes and engagement.
Five optimizer repeats and twelve case blocks limit precision; zero observed
breaches does not imply zero risk.

\textbf{Accounting.} Actual work included @ACTIONS@ executed actions,
@CHECKER@ counterfactual Reacher physics steps and @PREDICTOR@ learned predictor
steps, with TwoRoom analytic checks counted separately. Identical enforced ceilings
did not make consumed work identical. No equal-work search-efficiency claim is made.
All 48 live engineering episodes and seventeen focused tests passed. Every attempt,
optimizer checkpoint and source freeze is preserved privately. The independent guard
performs terminal preservation and self-destruction after the finish request;
external provider absence and posted billing require a separate check.

\textbf{Record.} Start at
\path{progress/receipts/17-lewm-evolution-results.md} for the complete report,
statistics, audit and private preservation receipts.
"""
    entry = entry.replace("@CONCLUSION@", conclusion)
    for method in ("evolution", "random", "reinforce", "tuned"):
        row = primary[method]
        entry = entry.replace(
            f"@{method.upper()}@", f"{row['requested_safe_completion']}/{row['n']}"
        )
    resources = stats["all_attempt_resources"]
    for marker, key in (
        ("ACTIONS", "environment_actions"),
        ("CHECKER", "checker_physics_steps"),
        ("PREDICTOR", "predictor_steps"),
    ):
        entry = entry.replace(f"@{marker}@", f"{resources[key]:,}".replace(",", "{,}"))
    path.write_text(source.replace("\\end{document}", entry + "\n\\end{document}"))
    path = ROOT / "progress/README.md"
    source = path.read_text()
    start, end = source.index("- **Task 17,"), source.index("- **Task 15,")
    source = source[:start] + source[end:]
    marker = "## Latest completed campaign\n\n"
    summary = f"""- **Task 17, LeWM evolutionary-controller comparison: experiment complete.**
  All 7,800 admitted episodes and thirty optimizer runs completed. {conclusion}
  Primary Reacher 1.5 cm safe completion: {text_scores}.
  All evidence is privately preserved; the independent guard handles terminal
  preservation and rental self-destruction. External absence and posted billing
  require a separate observation. See [results](receipts/17-lewm-evolution-results.md),
  [statistics](receipts/17-final-statistics.json), [audit](receipts/17-final-audit.json)
  and [completion status](receipts/17-completion-status.json).

"""
    source = source.replace(marker, marker + summary)
    marker = "\nFull per-run evidence for every closed task"
    row = (
        "\n| 17, LeWM optimizer comparison | "
        + conclusion
        + " All 7,800 admitted episodes completed | [Results](receipts/17-lewm-evolution-results.md), "
        "[audit](receipts/17-final-audit.json), [completion status](receipts/17-completion-status.json) |\n"
    )
    source = source.replace(marker, row.lstrip("\n") + marker)
    path.write_text(source)
    path = ROOT / "progress/receipts/README.md"
    source = path.read_text()
    start = source.index("Task 17")
    end = source.index("\n\nFor task 5", start)
    source = (
        source[:start]
        + """Task 17's experiment is complete: [results](17-lewm-evolution-results.md),
[statistics](17-final-statistics.json), [audit](17-final-audit.json),
[optimizer freeze](17-optimizer-comparison.json), [final freeze](17-final-evaluation.json),
[engineering audit](17-engineering-final-audit.json), [implementation audit](17-implementation-audit.json)
and [completion status](17-completion-status.json). Private snapshots are pinned by
`17-preservation-*` receipts. The independent guard handles terminal preservation and
self-destruction; provider absence and posted billing need an external observation."""
        + source[end:]
    )
    path.write_text(source)
    record = rental()
    status = {
        "utc": utc(),
        "scientific_campaign": "complete",
        "episodes": 7800,
        "optimizer_runs": 30,
        "final_audit_passed": True,
        "conclusion": conclusion,
        "financial_cap_usd": 10,
        "last_observed_accounting": finance(record),
        "instance_id": record["instance_id"],
        "deadline_utc": record["deadline_utc"],
        "latest_verified_backup_before_final_reporting": json.loads(
            (OUT / "control/preservation-latest.json").read_text()
        ),
        "next_authorized_action": "Compile Odyssey; verify final private snapshot; control_lewm17.py finish",
        "closeout_status": "Ready for independent guard handoff after final verified snapshot",
        "provider_delete_acknowledgement": "Not yet observed in this receipt",
        "external_provider_absence": "Pending external observation, not claimed",
        "posted_billing": "Pending external observation, not claimed",
    }
    write_json(ROOT / "progress/receipts/17-completion-status.json", status)
    print(conclusion)


if __name__ == "__main__":
    main()
