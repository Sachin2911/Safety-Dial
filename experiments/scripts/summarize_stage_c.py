"""Numbers for a stage C receipt (tasks 12 to 14), as Markdown on stdout: run facts, the
method table, pairwise comparisons, the best front settings on the final seeds and the
stage B baselines on the same layouts (stage B used seeds 200 to 209, the finals 400 to 409).

    summarize_stage_c.py --task reacher13 [--analysis DIR]
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
UNITS = {"tworoom12": ("px", 1.0), "reacher13": ("cm", 100.0), "cube14": ("cm", 100.0)}
PREFIX = {"tworoom12": "12-tworoom", "reacher13": "13-reacher", "cube14": "14-cube"}
LABELS = {"nsga2": "NSGA-II", "random": "random search", "grid": "grid search"}


def complete(remote, job_id):
    folders = [remote / "attempts" / job_id] + sorted((remote / "attempts").glob(f"{job_id}-*"))
    for folder in folders:
        path = folder / "result.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["status"] == "complete":
                return result
    return None


def stats(rows):
    safe = [bool(r["success"]) and not r["any_violation"] for r in rows]
    return {
        "n": len(rows),
        "success": float(np.mean([r["success"] for r in rows])),
        "safe_success": float(np.mean(safe)),
        "entries": int(sum(r["any_violation"] for r in rows)),
        "exposure": float(np.mean([r["exposure"] for r in rows])),
        "J": float(np.mean([r["J"] for r in rows])),
    }


def settings(params, unit, k):
    return (
        f"d0 {params['d0'] * k:.3g} {unit}, d1 {params['d1'] * k:+.3g} {unit} per block, "
        f"horizon {params['horizon']}, elite fraction {params['topk_frac']:.3g}, "
        f"tolerance {params['tol'] * k:.3g} {unit}"
    )


def cell(entry, key, digits=2):
    value = entry.get(key)
    return f"{value['mean']:.{digits}f} ± {value['sd']:.{digits}f}" if value else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(UNITS), required=True)
    parser.add_argument("--analysis", type=Path)
    args = parser.parse_args()
    ns, (unit, k) = args.task, UNITS[args.task]
    spec = json.loads((ROOT / f"configs/{ns}/design.json").read_text())
    stage = spec["stages"]["C"]
    remote = ROOT / f"outputs/{ns}/remote"
    analysis = args.analysis or ROOT / f"outputs/{ns}/analysis"
    summary = json.loads((analysis / "stageC-summary.json").read_text())
    b_summary = json.loads((ROOT / f"outputs/{ns}/analysis/stageB-summary.json").read_text())
    lines = []

    results = [
        json.loads(p.read_text()) for p in sorted((remote / "attempts").glob("C-*/result.json"))
    ]
    status = {}
    for r in results:
        status[r["status"]] = status.get(r["status"], 0) + 1
    markers = {}
    for name in ("start", "complete"):
        path = remote / f"stage-C-{name}.json"
        markers[name] = json.loads(path.read_text()).get("utc") if path.exists() else None
    freezes = sorted((ROOT / "progress/receipts").glob(f"{PREFIX[ns]}-C-v*.json"))
    lines += [
        f"- Freeze `{freezes[-1].name}`; stage C start {markers['start']}, complete "
        f"{markers['complete']}.",
        f"- Attempt records: {json.dumps(status)}; search layout {stage['search_layout']}, held-out "
        f"layout {stage['holdout_layout']}.",
        "",
        "| method | replicates | search HV | fresh-seed HV | held-out HV | best safe success, "
        "fresh | best safe success, held out |",
        "|---|---|---|---|---|---|---|",
    ]
    for method, entry in summary["methods"].items():
        lines.append(
            f"| {LABELS.get(method, method)} | {entry['replicates']} | "
            + " | ".join(
                cell(entry, key)
                for key in (
                    "search_hv",
                    "fresh_hv",
                    "holdout_hv",
                    "fresh_best_safe_success",
                    "holdout_best_safe_success",
                )
            )
            + " |"
        )
    lines += ["", "Pairwise (difference of means; wins out of replicate pairs):", ""]
    for name, item in summary["pairwise"].items():
        lines.append(
            f"- {name}: {item['difference_of_means']:+.2f}; {item['wins']} of {item['comparisons']}"
        )

    runs = summary["runs"]
    for key, lay in (("fresh", stage["search_layout"]), ("holdout", stage["holdout_layout"])):
        best = []
        for run in runs:
            for c, score in enumerate(run.get(f"{key}_safe_success_by_candidate") or []):
                best.append((score, run["method"], run["replicate"], c))
        if not best:
            continue
        top = max(score for score, *_ in best)
        winners = [b for b in best if b[0] == top]
        lines += ["", f"Best {key} settings (layout {lay}, safe success {top:.1f}):", ""]
        for _, method, rep, c in winners:
            final = json.loads((remote / f"stageC/{method}-r{rep}/final.json").read_text())
            ids = [j for j in final[f"{key}_job_ids"] if j.endswith(f"-c{c:02d}")]
            rows = [r for r in (complete(remote, j) for j in ids) if r]
            s = stats(rows)
            lines.append(
                f"- {LABELS.get(method, method)} r{rep} c{c:02d}: "
                f"{settings(rows[0]['job']['params'], unit, k)}; success {s['success']:.1f}, "
                f"safe success {s['safe_success']:.1f}, entries {s['entries']} of {s['n']}, "
                f"mean J {s['J']:.1f}, mean exposure {s['exposure']:.3f}"
            )

    lines += ["", "Stage B on the same layouts (seeds 200 to 209):", ""]
    for lay in (stage["search_layout"], stage["holdout_layout"]):
        groups = [g for g in b_summary["groups"] if g["layout"] == lay]
        free = next(g for g in groups if g["condition"] == "free")
        d0 = next(g for g in groups if g["condition"] == "safe" and g["dial"] == 0.0)
        best = b_summary["best_fixed_dial"][lay]
        lines.append(
            f"- {lay}: free safe success {free['safe_success_rate']:.1f} (J {free['J_mean']:.1f}); "
            f"checked d = 0 {d0['safe_success_rate']:.1f} (J {d0['J_mean']:.1f}); best fixed dial "
            f"{best['dial'] * k:g} {unit}: {best['safe_success_rate']:.1f} (J {best['J_mean']:.1f})"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
