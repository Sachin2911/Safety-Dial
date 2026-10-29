"""Write audited task-18 paper additions with pinned media references."""

import argparse
import json
from pathlib import Path
import shutil

from package_lewm18_evidence import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = json.loads((args.root / "analysis/summary.json").read_text())
    uploads = json.loads((ROOT / "progress/receipts/18-datasets-preservation.json").read_text())
    revisions = {x["repo"]: x["revision"] for x in uploads["uploads"]}
    assert len(revisions) == 2 and report["unique_attempts"] == 1352
    paper = ROOT / "docs/research/honours-gecco"
    results, guide = [], []
    for kind, margin, title in (("pusht", 12, "PushT"), ("cube", 0.01, "Cube")):
        rows = {
            r["method"]: r
            for r in report["tables"]
            if (r["stage"], r["env"], r["margin"]) == ("H", kind, margin)
        }
        c, f, a, n = (rows[m] for m in ("canonical", "fixed", "feedback", "native"))

        def contrast(first, second, metric):
            return next(
                r
                for r in report["contrasts"]
                if (r["env"], r["margin"], r["first"], r["second"], r["metric"])
                == (kind, margin, first, second, metric)
            )

        effect = contrast("fixed", "canonical", "safe")
        speed = contrast("fixed", "canonical", "J_safe")
        adaptive = contrast("feedback", "fixed", "safe")
        lo, hi = (100 * v for v in effect["interval"])
        alo, ahi = (100 * v for v in adaptive["interval"])
        slo, shi = speed["interval"]
        clearance = "12-pixel" if kind == "pusht" else "1 cm"
        results += [
            rf"\paragraph{{{title}: fixed settings recover safe completion.}}",
            f"At the {clearance} clearance, original SafeCEM completes safely in {c['safe']}/36",
            f"attempts, tuned fixed settings in {f['safe']}/36 and feedback in {a['safe']}/36.",
            f"Their requested-margin breach counts are {c['margin_breach']}, {f['margin_breach']} and {a['margin_breach']};",
            f"unfinished counts are {c['unfinished']}, {f['unfinished']} and {a['unfinished']};",
            f"engagement is {c['engaged']}/36, {f['engaged']}/36 and {a['engaged']}/36, respectively.",
            f"Native planning has {n['safe']}/36 safe completions, {n['margin_breach']}/36 breaches",
            f"and {n['unfinished']}/36 unfinished attempts.",
            "",
            f"The fixed-minus-original gain is {100 * effect['estimate']:.1f} percentage points,",
            rf"with a descriptive 95\% paired case-bootstrap interval of $[{lo:.1f},{hi:.1f}]$.",
            f"Mean failure-capped actions fall from {c['mean_J_safe']:.2f} to {f['mean_J_safe']:.2f};",
            rf"the difference is ${speed['estimate']:.2f}$ actions, interval $[{slo:.2f},{shi:.2f}]$.",
            f"Feedback minus fixed is {100 * adaptive['estimate']:+.1f} percentage points",
            rf"with interval $[{alo:.1f},{ahi:.1f}]$.",
            "",
        ]
        activity = next(
            x for x in report["feedback_activity"] if (x["env"], x["margin"]) == (kind, margin)
        )
        results += [
            f"Recorded replans depart from the base settings in {activity['episodes_changed']}/36",
            f"feedback attempts ({activity['changed_solves']}/{activity['recorded_solves']} recorded solves).",
            "This descriptive audit does not count unlogged failed final prefix solves.",
            "",
        ]
        guide += [
            rf"\textbf{{{title}, {clearance} clearance.}} Original SafeCEM finishes safely",
            f"in {c['safe']} of 36 attempts; tuned fixed settings in {f['safe']} of 36;",
            f"feedback in {a['safe']} of 36. Their breach counts are {c['margin_breach']},",
            f"{f['margin_breach']} and {a['margin_breach']}. Tuned fixed settings leave {f['unfinished']} attempts unfinished.",
            "",
        ]
    results += [
        "These intervals keep the three seeds for each case together and are",
        "conditional on the selected settings. Identical observed case effects",
        "can produce a zero-width empirical bootstrap range; this is not evidence",
        "of population equivalence or zero uncertainty. Feedback has not established",
        "an additional safe-completion benefit on these panels.",
        "",
        "The separate Cube native context check completes 10/12, but seven pairs",
        "already meet the object goal at reset. Among the five remaining pairs,",
        "three complete. This easier unfiltered panel must not be substituted",
        "for the reserved moving-object cases. Appendix~\\ref{app:extension}",
        "contains every margin, the serial timing results and the recorded examples.",
        "",
    ]
    (paper / "lewm18-results.tex").write_text("\n".join(results))
    guide += [
        r"\textbf{The lesson.} Better fixed settings help at the same safety requirement.",
        "Feedback has not shown an additional benefit. This extension does not test evolution.",
        "",
    ]
    (paper / "lewm18-guide-results.tex").write_text("\n".join(guide))
    shutil.copy2(args.root / "analysis/comparison.pdf", paper / "assets/lewm18-comparison.pdf")
    appendix = [
        r"\input{lewm18-tables.tex}",
        r"\input{lewm18-appendix-intro.tex}",
        r"\begin{figure*}[t]\centering",
        r"\includegraphics[width=.95\textwidth]{lewm18-comparison.pdf}",
        r"\caption{All task-18 final comparisons at matched clearances. Each point has 36",
        "attempts on twelve cases. Native observations are reused across margins.",
        "Capped actions include every failure. Overlapping fixed and feedback markers",
        "indicate similar outcomes, not absent data.}",
        r"\Description{Safe completion, requested-margin breaches and capped actions for four planners in PushT and Cube at three clearances.}",
        r"\end{figure*}",
    ]
    urls = []
    for kind, title in (("pusht", "PushT"), ("cube", "Cube")):
        repo = f"Sachioster/SafetyDial-{title}"
        base = (
            f"https://huggingface.co/datasets/{repo}/blob/{revisions[repo]}/campaigns/lewm18/media"
        )
        name = f"{kind}-case-0-high-margin-paper.png"
        shutil.copy2(args.root / "media" / name, paper / "assets" / name)
        appendix += [
            r"\begin{figure*}[t]\centering",
            rf"\includegraphics[width=\textwidth]{{{name}}}",
            rf"\caption{{{title}: final observations from the first reserved case and first planner seed,",
            "at the largest clearance. The goal image shows what completion requires.",
            "This prespecified example is illustrative; the aggregate tables include every case.}",
            rf"\Description{{Four {title} planners' final observations, outcomes and action counts beside their shared goal image.}}",
            r"\end{figure*}",
            rf"\paragraph{{{title} videos and full-case pictures.}}",
        ]
        for ci in (0, 1):
            url = f"{base}/{kind}-case-{ci}-high-margin.mp4"
            urls.append(url)
            appendix.append(rf"\href{{{url}}}{{Comparison video, case {ci + 1}}}.")
        for page in (1, 2):
            appendix.append(
                rf"\href{{{base}/{kind}-trajectory-gallery-{page}.png}}{{All-case gallery, page {page}}}."
            )
        appendix.append("")
    (paper / "lewm18-appendix.tex").write_text("\n".join(appendix))
    source = (paper / "honoursThesis.tex").read_text()
    if r"\input{lewm18-appendix}" not in source:
        source = source.replace(
            r"\input{visual-appendix}", "\\input{visual-appendix}\n\\input{lewm18-appendix}"
        )
    (paper / "honoursThesis.tex").write_text(source)
    (args.root / "analysis/paper-media-links.json").write_text(
        json.dumps({"pinned_videos": urls}, indent=2) + "\n"
    )
    print(json.dumps({"pinned_videos": len(urls)}))


if __name__ == "__main__":
    main()
