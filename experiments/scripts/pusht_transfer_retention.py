#!/usr/bin/env python3
"""E4 retention on actual final acquisition checkpoints and the frozen E2 goal cases.

Largest budget for every reported arm and acquisition seed, plus the E5-selected budget
when different. The released baseline runs once on the identical cases. Every checkpoint
is reported, including regressions. No acquisition or final-test results are rewritten.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.threads import apply_torch, pin_threads

pin_threads()
apply_torch()
import torch

from helpers.hfStore import HFStore
from helpers.pushtAssets import load_model, load_scalers
from helpers.pushtRetention import compare_retention, evaluate_retention, load_fixed_cases
from helpers.pushtTransferRetention import checkpoint_plan
from helpers.runManifest import build_manifest, file_sha256, make_run_id, validate_run_id, write_manifest
from scripts.pusht_e5_gated import validate_partial_weights


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acquisition-report", type=Path, required=True)
    ap.add_argument("--cases", type=Path, required=True, help="Exact frozen E2 case file")
    ap.add_argument("--checkpoint-root", type=Path, help="Default runs/<acquisition run_id>")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--run-id", type=validate_run_id)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    started = time.time()
    acquisition = json.loads(args.acquisition_report.read_text())
    e2_path = Path(acquisition["e2_report"])
    e2 = json.loads(e2_path.read_text())
    if e2.get("gate", {}).get("passes") is not True:
        raise ValueError("E2 did not pass")
    case_sha256 = file_sha256(args.cases)
    if case_sha256 != e2.get("goal_retention_cases_sha256"):
        raise ValueError("Transfer retention must reuse the exact frozen E2 cases")
    n_blocks = e2["goal_retention"]["n_blocks"]
    if n_blocks < 50:
        raise ValueError("E2 did not freeze a full retention horizon")
    assets = Path(e2["assets_run"])
    if assets.resolve() != Path(acquisition["assets_run"]).resolve():
        raise ValueError("E2/acquisition assets differ")
    splits = json.loads((assets / "splits.json").read_text())
    cases, _ = load_fixed_cases(args.cases, splits)
    plan = checkpoint_plan(acquisition, args.checkpoint_root or ROOT / "runs" / acquisition["run_id"])
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving existing transfer retention: {args.output_dir}")
    if args.preflight_only:
        print(json.dumps({"checkpoints": plan, "n_cases": len(cases), "n_blocks": n_blocks,
                          "case_file_sha256": case_sha256, "simulator_steps": 0}, indent=2))
        return 0
    args.output_dir.mkdir(parents=True)
    run_id = args.run_id or make_run_id("pusht", "transfer-retention", "e4")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output_dir / "config.yaml").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "cases.json").write_bytes(args.cases.read_bytes())
    model, process = load_model("cuda"), load_scalers(assets / "scalers.npz")
    base = evaluate_retention(model, process, cases, n_blocks=n_blocks)
    report = {"run_id": run_id, "status": "running", "acquisition_run": acquisition["run_id"],
        "case_file_sha256": case_sha256, "n_blocks": n_blocks, "checkpoint_plan": plan,
        "scope": "Every reported arm and seed at the largest budget, plus the E5-selected budget if different; intermediate curves have no goal-retention claim",
        "base": base, "checkpoints": {}, "ledger": {"base": base["ledger"]}}
    hf = None if args.no_upload else HFStore()
    for item in plan:
        output = args.output_dir / item["name"]
        output.mkdir()
        weights = torch.load(item["weights"], map_location="cpu", weights_only=True)
        adapted_model = validate_partial_weights(model, weights, acquisition["recipe"]["modules"])
        adapted = evaluate_retention(adapted_model, process, cases, n_blocks=n_blocks)
        result = {"base": base, "adapted": adapted, "comparison": compare_retention(base, adapted),
            "repaired_weights_sha256": item["weights_sha256"], "case_file_sha256": case_sha256,
            "n_blocks": n_blocks, "checkpoint": item}
        (output / "retention.json").write_text(json.dumps(result, indent=2) + "\n")
        report["checkpoints"][item["name"]] = {"comparison": result["comparison"],
            "weights_sha256": item["weights_sha256"], "report": str((output / "retention.json").resolve())}
        report["ledger"][item["name"]] = adapted["ledger"]
        (args.output_dir / "transfer_retention.json").write_text(json.dumps(report, indent=2) + "\n")
        del adapted_model
    report["status"] = "complete"
    report["all_checkpoints_pass"] = all(r["comparison"]["passes"] for r in report["checkpoints"].values())
    (args.output_dir / "transfer_retention.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(f"# {run_id}\n\nE4 goal retention on the exact frozen E2 source cases and maximum horizon. Final checkpoints of every acquisition arm and seed are evaluated; E5's selected budget is included when distinct. Raw censored futures are retained and early task completion needs independently verified whole-T goal coverage. This result does not extend to untested intermediate checkpoints.\n")
    write_manifest(args.output_dir, build_manifest(run_id=run_id, kind="retention",
        data={"acquisition_report_sha256": file_sha256(args.acquisition_report),
              "e2_report_sha256": file_sha256(e2_path), "case_file_sha256": case_sha256,
              "checkpoints": plan}, costs=report["ledger"],
        metrics={"n_checkpoints": len(plan), "all_checkpoints_pass": report["all_checkpoints_pass"]}, started_at=started))
    if hf:
        hf.upload_run("pusht", "retention", args.output_dir, run_id=run_id)
    print(f"[transfer-retention] complete: {len(plan)} checkpoints; all pass={report['all_checkpoints_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
