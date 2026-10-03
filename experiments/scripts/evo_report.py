#!/usr/bin/env python3
"""Stage 7 of the evolution study (docs/evoPlan/README.md): figures, tables and the freeze.

Reads the committed stage results and writes, under docs/evoPlan/results/stage7/<run_id>/:
the cheating curves (best of N, and CMA-ES by generation for each population size); the
imagined against real scatter of stage 4; the noise x rule grid (real violation rate and real
return against the noise level, one chart each, never a second y axis); the decomposition of
the imagined-real gap into imagined dynamics, probe readout and temporal sampling; and the
tables (real violation rate with Wilson intervals and real return per configuration).

Colours follow the dataviz reference palette in fixed slot order (validated, adjacent pairs);
rules also differ by marker and are labelled directly, because three light slots sit below
3:1 contrast on the surface.

    uv run python experiments/scripts/evo_report.py --s2 ID --s4 ID --s5 ID --s6 ID [--no-upload]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from helpers.evoInputs import RESULTS_ROOT, EvoRun, load_stage_config  # noqa: E402
from helpers.runManifest import build_manifest  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # dataviz slots 1-5, fixed order
MARKERS = ["o", "s", "^", "D", "v"]
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # blue ramp 250..650
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def label_end(ax, x, y, text, color) -> None:
    """Direct label at a line's last point, in text ink beside a colour swatch marker."""
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", fontsize=7, color=INK, va="center")
    ax.plot([x], [y], marker="o", markersize=3, color=color)


def save(fig, out: Path, name: str) -> list[str]:
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    files = []
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
        files.append(f"{name}.{ext}")
    plt.close(fig)
    return files


def best_of_n_figure(s5: dict, out: Path) -> list[str]:
    curve = s5["best_of_n_curve"]
    ns = sorted(int(n) for n in curve)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    series = [("eval_real_violation_rate", "real (dense truth)", 0), ("eval_imagined_violation_rate", "imagined, evaluation roots", 1),
              ("fitness_violation_rate", "imagined, fitness roots", 2)]
    for key, label, i in series:
        y = [curve[str(n)][key] for n in ns]
        ax.plot(ns, y, color=SERIES[i], linewidth=1.5, marker=MARKERS[i], markersize=4)
        label_end(ax, ns[-1], y[-1], label, SERIES[i])
    ax.set_xscale("log", base=2)
    style(ax, "Best of N in imagination: violation rate of the pick", "N (policies scored, log scale)", "health violation rate")
    ax.legend([s[1] for s in series], fontsize=7, frameon=False, loc="upper left")
    return save(fig, out, "cheating_best_of_n")


def cmaes_figure(s5: dict, out: Path) -> list[str]:
    lams = sorted(s5["cmaes_curve"], key=int)
    fig, axes = plt.subplots(1, len(lams), figsize=(3.0 * len(lams), 3.2), sharey=True)
    for ax, lam in zip(np.atleast_1d(axes), lams, strict=True):
        rows = s5["cmaes_curve"][lam]["best"]
        gs = sorted(int(g) for g in rows)
        x = [max(g, 0.5) for g in gs]
        for key, label, i in (("real_violation_rate", "real", 0), ("imagined_violation_rate", "imagined", 1)):
            y = [rows[str(g)][key] for g in gs]
            ax.plot(x, y, color=SERIES[i], linewidth=1.5, marker=MARKERS[i], markersize=4, label=label)
            label_end(ax, x[-1], y[-1], label, SERIES[i])
        ax.set_xscale("symlog", linthresh=1)
        style(ax, f"CMA-ES, population {lam}", "generation", "violation rate (best so far)")
    np.atleast_1d(axes)[0].legend(fontsize=7, frameon=False, loc="upper left")
    return save(fig, out, "cheating_cmaes")


def transfer_figure(s4: dict, out: Path) -> list[str]:
    pol = s4["policies"]
    kinds = ["bc", "perturbation", "interpolation"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, (ik, rk, title) in zip(axes, (("imagined_return", "real_return", "Mean return (m per 0.8 s)"),
                                          ("imagined_violation_rate", "real_violation_rate", "Violation rate")), strict=True):
        for i, kind in enumerate(kinds):
            pts = [p for p in pol if p["kind"] == kind]
            ax.scatter([p[ik] for p in pts], [p[rk] for p in pts], s=18 if kind != "bc" else 40, color=SERIES[i],
                       marker=MARKERS[i], edgecolor=SURFACE, linewidth=0.8, label=kind, zorder=3)
        g = s4["gate1"]["return" if "return" in ik else "violation_rate"]
        style(ax, f"{title}: Spearman {g['point']:.2f} [{g['lo']:.2f}, {g['hi']:.2f}]", "imagined (k = 0)", "real")
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color=GRID, linewidth=1, zorder=1)
    axes[0].legend(fontsize=7, frameon=False)
    return save(fig, out, "transfer_scatter")


def grid_figures(s6: dict, out: Path) -> list[str]:
    cells = s6["cells"]
    rules = list(dict.fromkeys(c["rule"] for c in cells.values()))
    ks = sorted({c["k"] for c in cells.values()})
    files = []
    for key, ylabel, name in (("real_violation_rate", "real violation rate (IQM over seeds)", "grid_violation"),
                              ("real_return", "real return, m per 0.8 s (IQM over seeds)", "grid_return")):
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        for i, rule in enumerate(rules):
            pts = [cells[f"{rule}|{k}"][key] for k in ks]
            y = [p["point"] for p in pts]
            err = np.array([[p["point"] - p["lo"] for p in pts], [p["hi"] - p["point"] for p in pts]])
            ax.errorbar(ks, y, yerr=err, color=SERIES[i % len(SERIES)], marker=MARKERS[i % len(MARKERS)], markersize=4,
                        linewidth=1.5, capsize=2, label=rule)
            label_end(ax, ks[-1], y[-1], rule, SERIES[i % len(SERIES)])
        style(ax, ylabel.split(" (")[0].capitalize() + " against imagination noise", "noise level k (x sigma)", ylabel)
        ax.legend(fontsize=7, frameon=False)
        files += save(fig, out, name)
    return files


def decomposition_figure(s6: dict, out: Path) -> list[str]:
    """Gap = real - imagined(k=0) = (readout - imagined) + (real - readout), per rule at k = 0."""
    runs = [r for r in s6["runs"] if r["k"] == 0.0]
    rules = list(dict.fromkeys(r["rule"] for r in runs))
    comp = {"imagined dynamics (readout - imagined)": [], "probe and timing (real - readout)": []}
    for rule in rules:
        sel = [r for r in runs if r["rule"] == rule]
        comp["imagined dynamics (readout - imagined)"].append(np.mean([r["readout_violation_rate"] - r["imagined_k0_violation_rate"] for r in sel]))
        comp["probe and timing (real - readout)"].append(np.mean([r["real_violation_rate"] - r["readout_violation_rate"] for r in sel]))
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    x = np.arange(len(rules))
    w = 0.38
    for i, (label, vals) in enumerate(comp.items()):
        ax.bar(x + (i - 0.5) * w, vals, width=w - 0.02, color=SERIES[i], label=label)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(x, rules, fontsize=8)
    style(ax, "Where the imagined-real gap comes from (k = 0, final best)", "ranking rule", "violation-rate difference")
    ax.legend(fontsize=7, frameon=False)
    return save(fig, out, "gap_decomposition")


def tables_markdown(s6: dict) -> str:
    lines = ["| rule | k | real violation rate (pooled, Wilson 95%) | IQM real violation rate [95%] | IQM real return [95%] | IQM gap, own noise |",
             "|---|---|---|---|---|---|"]
    for c in s6["cells"].values():
        w = c["pooled_real_violation_wilson"]
        v, r, g = c["real_violation_rate"], c["real_return"], c["gap_own"]
        lines.append(f"| {c['rule']} | {c['k']:g} | {c['pooled_real_violation_rate']:.3f} [{w[0]:.3f}, {w[1]:.3f}] | "
                     f"{v['point']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] | {r['point']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] | {g['point']:.3f} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for s in ("s2", "s4", "s5", "s6"):
        ap.add_argument(f"--{s}", required=True, help=f"stage {s[1]} results run id")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_stage_config("stage7")
    run = EvoRun.create(cfg, "s7", args.run_id)
    s2 = json.loads((RESULTS_ROOT / "stage2" / args.s2 / "gate0.json").read_text())
    s4 = json.loads((RESULTS_ROOT / "stage4" / args.s4 / "gate1.json").read_text())
    s5 = json.loads((RESULTS_ROOT / "stage5-summary" / args.s5 / "cheating_curve.json").read_text())
    s6 = json.loads((RESULTS_ROOT / "stage6-summary" / args.s6 / "grid.json").read_text())
    files = []
    for fn, data in ((best_of_n_figure, s5), (cmaes_figure, s5), (transfer_figure, s4), (grid_figures, s6), (decomposition_figure, s6)):
        files += fn(data, run.run_dir)
    for f in files:
        (run.results_dir / f).write_bytes((run.run_dir / f).read_bytes())
    table = tables_markdown(s6)
    summary = {"run_id": run.run_id, "inputs": {"s2": args.s2, "s4": args.s4, "s5": args.s5, "s6": args.s6},
               "gate0": s2["gate0"], "gate1": s4["gate1"], "gate2": s5["gate2"], "stage6_generations": s5["stage6_generations"],
               "figures": files, "provisional": cfg.provisional, "wall_clock_s": time.time() - t0}
    run.write_json("summary.json", summary)
    (run.results_dir / "table_grid.md").write_text(table)
    (run.run_dir / "table_grid.md").write_text(table)
    manifest = build_manifest(run_id=run.run_id, kind="evo-s7", data=summary["inputs"], metrics={"figures": files}, started_at=t0)
    run.finish(manifest, upload=not args.no_upload, readme=f"# {run.run_id}\n\nStage 7 figures and tables of docs/evoPlan.\n")
    print(f"[evo-s7] wrote {len(files)} figure files and the grid table", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
