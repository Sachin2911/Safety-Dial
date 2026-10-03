#!/usr/bin/env python3
"""Stage 6 summary of the evolution study (docs/evoPlan/README.md): the fix grid.

For every CMA-ES run of the grid (noise level x ranking rule x seed) it reads the evaluated
final snapshot: the best-so-far policy's real violation rate (dense truth) and real return on
the 256 evaluation roots, its imagined violation rate at k = 0 and at the run's own noise level,
and the real-minus-imagined gaps. Per rule and noise level: interquartile mean over seeds with a
bootstrap interval and the pooled Wilson rate. At each noise level ROSARL is compared with Deb
and with each fixed penalty by Mann-Whitney U over seeds, Holm-corrected within each outcome.

    uv run python experiments/scripts/evo_grid_report.py --s6 RUN_ID [--no-upload]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np  # noqa: E402

from helpers.evoInputs import RUNS_ROOT, EvoRun, load_stage_config  # noqa: E402
from helpers.evoStats import bootstrap_ci, holm, iqm, mann_whitney, wilson  # noqa: E402
from helpers.runManifest import build_manifest  # noqa: E402

OUTCOMES = ("real_violation_rate", "real_return")


def run_summary(run_dir: Path, r: dict, g: int, which: int = 0) -> dict:
    with np.load(run_dir / "cma" / r["name"] / "eval" / f"g{g:04d}.npz") as z:
        e = {k: z[k] for k in z.files}
    real = e["real_violated"][which].astype(float)
    imag0 = e["imag0_violated"][which].astype(float)
    own = e["imag_own_violated"][which].astype(float).mean(-1) if "imag_own_violated" in e else imag0
    return {"name": r["name"], "rule": r["name"].split("-k")[0], "k": float(r["noise_k"]), "seed": int(r["seed"]),
            "real_violation_rate": float(real.mean()), "real_violations": int(real.sum()), "n_roots": int(real.size),
            "real_return": float(e["real_ret"][which].mean()), "imagined_k0_violation_rate": float(imag0.mean()),
            "imagined_own_violation_rate": float(own.mean()), "readout_violation_rate": float(e["readout_violated"][which].mean()),
            "imagined_k0_return": float(e["imag0_ret"][which].mean()),
            "gap_k0": float((real - imag0).mean()), "gap_own": float((real - own).mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--s6", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_stage_config("stage6")
    run = EvoRun.create(cfg, "s6-summary", args.run_id)
    rd = RUNS_ROOT / args.s6
    runs = json.loads((rd / "runs.json").read_text())
    G = int(cfg.grid.generations)
    rows = [run_summary(rd, r, G) for r in runs]
    rules = [str(r.name) for r in cfg.grid.rules]
    ks = [float(k) for k in cfg.grid.noise_levels]
    cells = {}
    for rule in rules:
        for k in ks:
            sel = [x for x in rows if x["rule"] == rule and x["k"] == k]
            n_v, n = sum(x["real_violations"] for x in sel), sum(x["n_roots"] for x in sel)
            cells[f"{rule}|{k}"] = {"rule": rule, "k": k, "n_seeds": len(sel),
                                    **{o: bootstrap_ci([x[o] for x in sel], iqm, n_boot=cfg.report.n_boot, seed=cfg.report.bootstrap_seed)
                                       for o in OUTCOMES + ("gap_own", "gap_k0", "imagined_own_violation_rate", "imagined_k0_violation_rate")},
                                    "pooled_real_violation_rate": n_v / n if n else float("nan"),
                                    "pooled_real_violation_wilson": wilson(n_v, n)}
    comparisons = []
    for k in ks:
        ros = [x for x in rows if x["rule"] == cfg.report.focus_rule and x["k"] == k]
        for other in rules:
            if other == cfg.report.focus_rule:
                continue
            oth = [x for x in rows if x["rule"] == other and x["k"] == k]
            for o in OUTCOMES:
                t = mann_whitney([x[o] for x in ros], [x[o] for x in oth])
                comparisons.append({"k": k, "versus": other, "outcome": o, **t,
                                    "iqm_focus": iqm([x[o] for x in ros]), "iqm_other": iqm([x[o] for x in oth])})
    for o in OUTCOMES:
        fam = [c for c in comparisons if c["outcome"] == o]
        for c, p in zip(fam, holm([c["p"] for c in fam]), strict=True):
            c["p_holm"] = float(p)
    report = {"run_id": run.run_id, "stage": 6, "s6_run": args.s6, "generations": G, "focus_rule": cfg.report.focus_rule,
              "cells": cells, "comparisons": comparisons, "runs": rows, "wall_clock_s": time.time() - t0}
    run.write_json("grid.json", report)
    manifest = build_manifest(run_id=run.run_id, kind="evo-s6-summary", data={"s6": args.s6},
                              metrics={"n_runs": len(rows)}, started_at=t0)
    run.finish(manifest, upload=not args.no_upload,
               readme=f"# {run.run_id}\n\nStage 6 summary of docs/evoPlan: noise x rule grid, per-cell IQM and Holm-corrected rule comparisons.\n")
    print(f"[evo-s6] summarised {len(rows)} runs in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
