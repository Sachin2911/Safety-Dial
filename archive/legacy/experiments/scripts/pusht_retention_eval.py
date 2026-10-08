#!/usr/bin/env python3
"""E0 baseline episode or fixed E2 goal-reaching retention, written to fresh paths."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.threads import pin_threads, apply_torch

pin_threads()
apply_torch()

from helpers.hfStore import HFStore
from helpers.pushtAssets import load_model, load_scalers
from helpers.pushtRetention import compare_retention, evaluate_retention, load_fixed_cases, make_cases
from helpers.runManifest import build_manifest, file_sha256, make_run_id, write_manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets-run", type=Path, default=ROOT / "runs/pusht-assets-20260926-1")
    ap.add_argument("--cases", type=Path, help="Exact E2 goal_retention_cases.json; required for repaired checkpoints")
    ap.add_argument("--repaired", type=Path, help="exact predictor checkpoint for paired retention")
    ap.add_argument("--modules", choices=["predictor_only", "predictor_side"], help="default from repaired config.yaml")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--blocks", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    if args.repaired and not args.cases:
        ap.error("Repaired checkpoint retention must reuse --cases from E2")
    if args.out.exists():
        ap.error("--out must be fresh")
    started = time.time()
    args.out.mkdir(parents=True)
    splits = json.loads((args.assets_run / "splits.json").read_text())
    if args.cases:
        cases, case_sha256 = load_fixed_cases(args.cases, splits)
        (args.out / "cases.json").write_bytes(args.cases.read_bytes())
    else:
        cases = make_cases(splits, args.seed, args.episodes)
        (args.out / "cases.json").write_text(json.dumps([r.to_dict() for r in cases], indent=2) + "\n")
        case_sha256 = file_sha256(args.out / "cases.json")
    model = load_model("cuda")
    process = load_scalers(args.assets_run / "scalers.npz")
    base = evaluate_retention(model, process, cases, n_blocks=args.blocks)
    if args.repaired:
        import torch
        from omegaconf import OmegaConf
        from pusht_e5_gated import validate_partial_weights

        modules = args.modules
        if modules is None:
            config = args.repaired.parent / "config.yaml"
            if not config.exists():
                config = args.repaired.parent / "config.json"
            modules = OmegaConf.load(config)["modules"]
        weights = torch.load(args.repaired, map_location="cpu", weights_only=True)
        model = validate_partial_weights(model, weights, modules)
        adapted = evaluate_retention(model, process, cases, n_blocks=args.blocks)
        result = {"base": base, "adapted": adapted, "comparison": compare_retention(base, adapted),
                  "repaired_weights_sha256": file_sha256(args.repaired), "modules": modules,
                  "summary": adapted["summary"], "ledger": {"base": base["ledger"], "adapted": adapted["ledger"]}}
    else:
        result = base
    run_id = make_run_id("pusht", "retention", "paired" if args.repaired else "baseline", n=args.n)
    result["run_id"] = run_id
    result["case_file_sha256"] = case_sha256
    result["n_blocks"] = args.blocks
    result["role"] = "baseline demonstration" if args.episodes < 20 else "fixed goal-retention baseline"
    (args.out / "retention.json").write_text(json.dumps(result, indent=2) + "\n")
    write_manifest(args.out, build_manifest(run_id=run_id, kind="retention", seeds={"seed": args.seed}, data={"case_file_sha256": case_sha256, "assets_run": str(args.assets_run), "source_episodes": [r.meta["episode"] for r in cases]}, costs=result["ledger"], metrics=result["summary"], started_at=started))
    (args.out / "config.yaml").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2) + "\n")
    (args.out / "README.md").write_text(f"# {run_id}\n\nReleased Push-T LeWM with nominal CEM (300 samples, 30 iterations), whole-T goal coverage and the shared arena guard. No hazards.\n")
    if not args.no_upload:
        HFStore().upload_run("pusht", "retention", args.out, run_id=run_id)
    print(f"[retention] done: {json.dumps(result['summary'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
