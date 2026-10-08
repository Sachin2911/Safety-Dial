#!/usr/bin/env python3
"""Stage 5 summary of the evolution study (docs/evoPlan/README.md): the cheating curve.

Reads the stage 5a best-of-N result and the evaluated stage 5b CMA-ES snapshots. For every
population size and snapshot generation it pools seeds and reports, on the 256 evaluation
roots, the imagined (k = 0) and real violation rates of the best-so-far and distribution-mean
policies, the real readout, real return, and the real-minus-imagined gap, with the paired change
of the gap since generation 0 (source-episode cluster bootstrap). It then applies the rule
declared in configs/evo/stage5.yaml to fix the generation count of stage 6.

    uv run python experiments/scripts/evo_cheating_curve.py --s5a RUN_ID --s5b RUN_ID [--no-upload]
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
from helpers.evoStats import clustered_mean_ci, paired_gap_difference  # noqa: E402
from helpers.runManifest import build_manifest  # noqa: E402


def load_eval(run_dir: Path, name: str, g: int) -> dict:
    with np.load(run_dir / "cma" / name / "eval" / f"g{g:04d}.npz") as z:
        return {k: z[k] for k in z.files}


def choose_generations(diffs: dict, generations: list[int], fallback: int) -> int:
    """Smallest g > 0 whose gap change since g = 0 has a lower bound above zero at g and at
    every later snapshot; the fallback if there is none."""
    later = [g for g in generations if g > 0]
    for i, g in enumerate(later):
        if all(diffs[str(h)]["lo"] > 0 for h in later[i:]):
            return int(g)
    return int(fallback)


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--s5a", required=True)
    ap.add_argument("--s5b", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_stage_config("stage5")
    run = EvoRun.create(cfg, "s5-summary", args.run_id)
    s5a = json.loads((RUNS_ROOT / args.s5a / "best_of_n.json").read_text())
    rd = RUNS_ROOT / args.s5b
    runs = json.loads((rd / "runs.json").read_text())
    clusters = np.asarray(json.loads((rd / "evaluation_clusters.json").read_text())["source_episode"])
    gens = [int(g) for g in cfg.eval_generations]
    curve, choice_diffs = {}, {}
    for lam in cfg.cmaes.popsizes:
        names = [r["name"] for r in runs if r["popsize"] == int(lam)]
        per_g = {g: [load_eval(rd, n, g) for n in names] for g in gens}
        entry = {}
        for which, idx in (("best", 0), ("mean", 1)):
            rows = {}
            base_real = np.stack([e["real_violated"][idx] for e in per_g[0]]).astype(float)
            base_imag = np.stack([e["imag0_violated"][idx] for e in per_g[0]]).astype(float)
            for g in gens:
                real = np.stack([e["real_violated"][idx] for e in per_g[g]]).astype(float)
                imag = np.stack([e["imag0_violated"][idx] for e in per_g[g]]).astype(float)
                diff = paired_gap_difference(real, imag, base_real, base_imag, clusters,
                                             n_boot=cfg.gate2.n_boot, seed=cfg.gate2.bootstrap_seed)
                rows[str(g)] = {"real_violation_rate": float(real.mean()), "imagined_violation_rate": float(imag.mean()),
                                "readout_violation_rate": float(np.mean([e["readout_violated"][idx] for e in per_g[g]])),
                                "real_return": clustered_mean_ci(np.stack([e["real_ret"][idx] for e in per_g[g]]), clusters,
                                                                 n_boot=cfg.gate2.n_boot, seed=cfg.gate2.bootstrap_seed),
                                "imagined_return": float(np.mean([e["imag0_ret"][idx] for e in per_g[g]])),
                                "gap": float((real - imag).mean()), "gap_change_since_g0": diff}
            entry[which] = rows
            if int(lam) == int(cfg.stage6_generations.popsize) and which == cfg.stage6_generations.policy:
                choice_diffs = {g: rows[g]["gap_change_since_g0"] for g in rows}
        curve[str(lam)] = entry
    g6 = choose_generations(choice_diffs, gens, int(cfg.stage6_generations.fallback))
    report = {"run_id": run.run_id, "stage": 5, "s5a_run": args.s5a, "s5b_run": args.s5b, "gate2": s5a["gate2"],
              "best_of_n_curve": s5a["curve"], "cmaes_curve": curve, "stage6_generations": g6,
              "stage6_rule": "smallest g > 0 whose gap change since g = 0 (lambda 64, best so far) has a 95% lower bound > 0 at g and every later snapshot; else the fallback",
              "wall_clock_s": time.time() - t0}
    run.write_json("cheating_curve.json", report)
    manifest = build_manifest(run_id=run.run_id, kind="evo-s5-summary", data={"s5a": args.s5a, "s5b": args.s5b},
                              metrics={"gate2": s5a["gate2"], "stage6_generations": g6}, started_at=t0)
    run.finish(manifest, upload=not args.no_upload,
               readme=f"# {run.run_id}\n\nStage 5 summary of docs/evoPlan: cheating curves (best of N and CMA-ES) and the stage 6 generation count.\n")
    print(f"[evo-s5] Gate 2 {'PASS' if s5a['gate2']['pass'] else 'FAIL'}; stage 6 generations {g6}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
