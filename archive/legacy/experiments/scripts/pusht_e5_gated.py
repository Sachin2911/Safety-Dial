#!/usr/bin/env python3
"""Run E5 only from validated repeatable acquisition evidence, preserving earlier runs.

Example (all paths explicit):
  uv run python experiments/scripts/pusht_e5_gated.py --acquisition-report <acquisition.json> \
    --repaired <weights.pt> --retention-report <retention.json> \
    --feasibility-report <development-feasibility.json> \
    --banks-dir <new-bank-directory> --output-dir <new-results>
Use --preflight-only to validate the gate, checkpoint identity and cases without GPU work.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers.threads import apply_torch, pin_threads

pin_threads()
import numpy as np
import torch

apply_torch()

from helpers.acquisitionSafety import MeteredEnv, require_new_paths
from helpers.branchBank import Bank
from helpers.dialMetrics import fsa
from helpers.hfStore import HFStore
from helpers.plannerAudit import AuditedNominalPlanner, AuditedSafePlannerT
from helpers.poseProbes import load_probe
from helpers.pushtAssets import load_model, load_scalers
from helpers.pushtClosedLoop import (closedloop_arm_specs, fixed_cases,
                                     penalty_reference_protocol, run_observed_episode)
from helpers.pushtGeometry import hazard_from_dict
from helpers.pushtLayouts import load_layouts
from helpers.pushtReplay import StepLedger
from helpers.pushtContactReplay import make_env
from helpers.pushtSourceFamilies import validate_geometric_bank
from helpers.runManifest import build_manifest, validate_run_id, write_manifest
from helpers.splitIntegrity import validate_bank_splits
from helpers.studyGates import require_closedloop_gate, require_goal_retention


def validate_partial_weights(base, state, modules):
    names = {"predictor"} if modules == "predictor_only" else {"action_encoder", "predictor", "pred_proj"}
    expected = {key for key in base.state_dict() if key.split(".")[0] in names}
    if set(state) != expected:
        raise ValueError("Repaired checkpoint must contain every and only the selected predictor module's keys")
    repaired = copy.deepcopy(base)
    repaired.load_state_dict(state, strict=False)
    return repaired.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--acquisition-report", type=Path, required=True)
    ap.add_argument("--repaired", type=Path, required=True)
    ap.add_argument("--feasibility-report", type=Path, required=True)
    ap.add_argument("--retention-report", type=Path, required=True)
    ap.add_argument("--banks-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--run-id", type=validate_run_id, help="Immutable local/remote identity; defaults to output directory basename")
    ap.add_argument("--acquisition-seed", default="0")
    ap.add_argument("--episodes-per-layout", type=int, default=10)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--case-seed", type=int, default=20261005)
    ap.add_argument("--lam", type=float, default=.05)
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    args.run_id = validate_run_id(args.run_id or args.output_dir.name)
    penalty_protocol = penalty_reference_protocol(args.lam)
    from helpers.pushtFeasibility import require_feasibility_report

    feasibility = require_feasibility_report(args.feasibility_report, args.banks_dir / "dev")
    report = json.loads(args.acquisition_report.read_text())
    gate = require_closedloop_gate(report)
    if args.acquisition_seed not in gate["paired_seeds"]:
        raise ValueError("Checkpoint acquisition seed is not in the qualifying paired comparison")
    selected = next(p for p in report["arms"]["boundary"][args.acquisition_seed]["curve"] if p["budget_added"] == gate["selected_budget"])
    digest = hashlib.sha256(args.repaired.read_bytes()).hexdigest()
    if selected.get("weights_sha256") != digest:
        raise ValueError("Repaired checkpoint does not match the passing acquisition budget's recorded SHA256")
    e2 = json.loads(Path(report["e2_report"]).read_text())
    retention = require_goal_retention(json.loads(args.retention_report.read_text()), digest,
        cases_path=e2["goal_retention_cases"], case_sha256=e2["goal_retention_cases_sha256"],
        n_blocks=e2["goal_retention"]["n_blocks"])
    if not selected.get("hf_revision") or not selected.get("hf_repo"):
        raise ValueError("The selected repaired checkpoint must have a pinned private Hugging Face reference")
    if Path(report["banks_dir"]).resolve() != args.banks_dir.resolve():
        raise ValueError("E5 must use the bank protocol recorded in acquisition")
    split_audit = validate_bank_splits({name: args.banks_dir / name for name in ("dev", "test", "stress")})
    bank = Bank(args.banks_dir / "test")
    validate_geometric_bank(bank)
    layouts = load_layouts(args.banks_dir / "test" / "layouts.json")[0]
    fixed = float(e2["arms"]["fixed_margin"]["margin"])
    dial = float(report["no_update"]["margin_matched_dev"])
    closedloop_arm_specs(None, None, dial=dial, fixed_margin=fixed)  # Validate before any output or GPU load.
    if args.episodes_per_layout != 10 or args.blocks <= 0 or args.samples < 30 or args.iterations <= 0:
        raise ValueError("The declared E5 design requires 10 episodes per layout and valid planner budgets")
    cases = fixed_cases(bank, layouts, episodes_per_layout=args.episodes_per_layout,
                        seed=args.case_seed, margin=max(dial, fixed))
    require_new_paths([args.output_dir])
    if args.preflight_only:
        print(json.dumps({"run_id": args.run_id, "gate": gate, "n_cases": len(cases), "weights_sha256": digest,
                          "gpu_loaded": False, "simulator_steps": 0}, indent=1))
        return 0
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "cases.json").write_text(json.dumps(cases, indent=1) + "\n")
    device = "cuda"
    base = load_model(device).eval()
    process = load_scalers(Path(report["assets_run"]) / "scalers.npz")
    probe, _ = load_probe(Path(report["probes_run"]) / (report["probe_name"] + ".pt"), device)
    state = torch.load(args.repaired, map_location="cpu", weights_only=True)
    repaired = validate_partial_weights(base, state, report["recipe"]["modules"])
    arm_specs = closedloop_arm_specs(base, repaired, dial=dial, fixed_margin=fixed)
    hf = HFStore()
    upstream = {"repaired": {"repo_id": selected["hf_repo"], "revision": selected["hf_revision"], "weights_sha256": digest},
                "assets": hf.reference_run("pusht", "assets", Path(report["assets_run"])),
                "probes": hf.reference_run("pusht", "probes", Path(report["probes_run"])),
                "banks": {name: hf.reference_run("pusht-banks", "banks", args.banks_dir / name) for name in ("dev", "test", "stress")}}
    (args.output_dir / "config.yaml").write_text(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, indent=1) + "\n")
    out = {"run_id": args.run_id, "acquisition_report": str(args.acquisition_report), "gate": gate, "weights_sha256": digest,
           "goal_retention": retention, "development_feasibility": feasibility, "upstream": upstream,
           "split_audit": split_audit, "cases": cases, "arms": {}, "dial": dial, "fixed_margin": fixed,
           "planner": {"samples": args.samples, "iterations": args.iterations, "penalty_lambda": args.lam,
                       "penalty_reference": penalty_protocol},
           "arm_protocol": {arm: {"model": "repaired" if model is repaired else "original",
                                  "planner": mode or "nominal", "margin": margin}
                            for arm, (model, mode, margin) in arm_specs.items()}}
    root_map = {root.root_id: root for root in bank.roots}
    ledger = StepLedger()
    env = MeteredEnv(make_env(), ledger, "closedloop")
    try:
        for arm, (model, mode, margin) in arm_specs.items():
            rows = []
            out["arms"][arm] = {"episodes": rows}
            env.category = arm
            for case in cases:
                kw = {"num_samples": args.samples, "n_steps": args.iterations}
                planner = (AuditedNominalPlanner(model, process, device, **kw) if mode is None else
                           AuditedSafePlannerT(model, process, device, probe=probe, hazard=hazard_from_dict(case["hazard"]),
                                        dial=margin, mode=mode, lam=args.lam, **kw))
                rows.append(run_observed_episode(env, planner, root_map[case["root_id"]], case, n_blocks=args.blocks))
                out["ledger"] = ledger.to_dict()
                (args.output_dir / "closedloop.json").write_text(json.dumps(out, indent=1) + "\n")
            out["arms"][arm]["unsafe_episode_bounds"] = fsa(np.ones(len(rows)), [r["unsafe_composite"] for r in rows],
                                                               0, censored=[r["censored"] for r in rows])
            out["arms"][arm]["mean_final_coverage"] = float(np.mean([r["final_coverage"] for r in rows]))
            out["arms"][arm]["all_infeasible_solves"] = sum(r["all_infeasible_solves"] for r in rows)
        out["status"] = "complete"
    except BaseException as exc:
        out["status"] = "incomplete"
        out["error_type"] = type(exc).__name__
        raise
    finally:
        env.close()
        out["ledger"] = ledger.to_dict()
        (args.output_dir / "closedloop.json").write_text(json.dumps(out, indent=1) + "\n")
        write_manifest(args.output_dir, build_manifest(run_id=args.run_id, kind="closedloop",
                       seeds={"case_seed": args.case_seed}, data={"checkpoint_sha256": digest, "upstream": upstream,
                        "acquisition_report_sha256": hashlib.sha256(args.acquisition_report.read_bytes()).hexdigest(),
                        "retention_report_sha256": hashlib.sha256(args.retention_report.read_bytes()).hexdigest()}, costs=ledger.to_dict()))
    revision = hf.upload_run("pusht", "closedloop", args.output_dir, run_id=args.run_id)
    print(json.dumps({"status": out["status"], "hf_repo": hf.repo_id("pusht"), "hf_revision": revision}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
