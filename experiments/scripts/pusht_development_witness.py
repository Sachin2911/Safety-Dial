#!/usr/bin/env python3
"""Replay bounded development candidates to witness feasible routes for frozen layouts.

This uses only development sources. No test root is evaluated or filtered. Failure means
there is no demonstrated witness in this bounded search, not that a route is impossible.
Run under the managed workflow before E2; no model or GPU is required.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.threads import pin_threads

pin_threads()
import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from helpers.acquisitionSafety import MeteredEnv
from helpers.branchBank import Bank
from helpers.hfStore import HFStore
from helpers.pushtAssets import H5_PATH
from helpers.pushtDevelopmentPlan import validate_development_plan
from helpers.pushtFeasibility import (
    MIN_GOAL_COVERAGE, MIN_WITNESSES, PROTOCOL, assess_witness, feasibility_gate,
    generator_identity, require_feasibility_report,
)
from helpers.pushtGeometry import polygons_from_env
from helpers.pushtLayouts import load_layouts
from helpers.pushtReplay import StepLedger, make_env, reset_root, run_actions
from helpers.pushtRetention import block_coverage
from helpers.runManifest import build_manifest, file_sha256, make_run_id, validate_run_id, write_manifest
from helpers.splitIntegrity import bank_identity, validate_bank_splits


def validate_witness_inputs(banks_dir, *, development_only=False, sampling_plan=None):
    """Check source reservations before any simulator work or output creation."""
    banks_dir = Path(banks_dir)
    if development_only:
        if sampling_plan is None:
            raise ValueError("Development-only witnesses require a frozen future sampling plan")
        if any((banks_dir / name).exists() for name in ("test", "stress")):
            raise ValueError("Prospective development witness creation requires absent final banks")
        binding = validate_development_plan(banks_dir / "dev", sampling_plan)
        plan = json.loads(Path(sampling_plan).read_text())
        counts = {role: {family: len(eps) for family, eps in families.items()}
                  for role, families in plan["source_roles"].items()}
        return {"passes": True, "scope": "Completed dev bank against all reserved final sources",
                "source_role_counts": counts, "final_banks_evaluated": False}, binding
    if sampling_plan is not None:
        raise ValueError("A sampling plan argument requires development-only witness mode")
    return validate_bank_splits({name: banks_dir / name for name in ("dev", "test", "stress")}), None


def candidate_tapes(bank, index, expert, rng, max_steps, perturbations):
    """Declared bounded proposal list; all executed attempts enter the ledger."""
    root = bank.roots[index]
    for j in bank.indices_for_root(index):
        yield f"recorded-dev-branch-{j}", bank.h5["tape"][j].reshape(-1, 2)
    ep = int(root.meta["episode"])
    offset, length = int(expert["ep_offset"][ep]), int(expert["ep_len"][ep])
    for skip in sorted({int(root.meta["t0"]), int(root.meta["t0"]) + int(root.meta.get("k", 0)) * 5}):
        tape = np.asarray(expert["action"][offset + skip:offset + min(length, skip + max_steps)])
        # Missing final actions in expert data are not executable controls.
        bad = np.flatnonzero(~np.isfinite(tape).all(axis=1))
        if len(bad):
            tape = tape[:bad[0]]
        if not len(tape):
            continue
        yield f"expert-continuation-t{skip}", tape
        for attempt in range(perturbations):
            sigma = (.02, .05, .1)[attempt % 3]
            yield f"expert-t{skip}-noise-{attempt}-sigma-{sigma}", np.clip(
                tape + rng.normal(0, sigma, tape.shape), -.6, .6)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--banks-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--development-only", action="store_true",
                    help="Validate completed development bank before generating frozen final banks")
    ap.add_argument("--sampling-plan", type=Path,
                    help="Required with --development-only; binds full future source/candidate plan")
    ap.add_argument("--expert-data", type=Path, default=H5_PATH)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--perturbations", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--run-id", type=validate_run_id)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    if args.max_steps < 25 or args.perturbations < 0:
        ap.error("Use at least 25 steps and a nonnegative bounded perturbation count")
    if args.development_only != (args.sampling_plan is not None):
        ap.error("--development-only and --sampling-plan must be supplied together")
    if args.output_dir.exists():
        raise FileExistsError(f"Preserving existing witnesses: {args.output_dir}")
    started = time.time()
    audit, frozen_plan = validate_witness_inputs(args.banks_dir,
        development_only=args.development_only, sampling_plan=args.sampling_plan)
    dev_dir = args.banks_dir / "dev"
    initial_bank_identity = bank_identity(dev_dir)
    bank = Bank(dev_dir)
    dev_roles = bank.manifest["data"]["source_roles"]["dev"]
    development = sorted({int(e) for episodes in dev_roles.values() for e in episodes})
    if any(int(root.meta["episode"]) not in development for root in bank.roots):
        raise ValueError("Witness bank includes a source outside the declared development role")
    layouts = {lay.root_id: lay for lay in load_layouts(dev_dir / "layouts.json")[0]
               if lay.family == "familiar"}
    run_id = args.run_id or make_run_id("pusht", "development-witness", "frozen")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    ledger = StepLedger()
    env = MeteredEnv(make_env(), ledger, "development_witness")
    rng = np.random.default_rng(args.seed)
    witnesses, attempts, accepted_episodes = [], [], set()
    with h5py.File(args.expert_data, "r") as expert:
        for i, root in enumerate(bank.roots):
            if int(root.meta["episode"]) in accepted_episodes:
                continue
            before = ledger.total
            context = reset_root(env, root, record_frames=False)
            local = polygons_from_env(env)
            nominal = run_actions(env, np.asarray(root.meta["nominal_plan"]).reshape(-1, 2), record_frames=False)
            if nominal.censored:
                attempts.append({"root_id": root.root_id, "kind": "nominal-route",
                                 "reason": "censored", "charged_steps": ledger.total - before})
                continue
            route = np.concatenate([context.prefix_log.states[:, 2:5], nominal.states[1:, 2:5]])
            attempts.append({"root_id": root.root_id, "kind": "nominal-route",
                             "charged_steps": ledger.total - before})
            for label, tape in candidate_tapes(bank, i, expert, rng, args.max_steps, args.perturbations):
                before = ledger.total
                reset_root(env, root, record_frames=False)
                log = run_actions(env, tape, record_frames=False)
                states = log.states[:log.executed_steps + 1]
                hits = [j for j, state in enumerate(states[1:], 1)
                        if block_coverage(state[2:5], root.goal_state[2:5], local) >= MIN_GOAL_COVERAGE]
                if not hits:
                    attempts.append({"root_id": root.root_id, "candidate": label,
                                     "reason": "goal not reached", "charged_steps": ledger.total - before})
                    continue
                end = hits[0]
                record = {"role": "development", "root": root.to_dict(),
                    "local_polygons": [p.tolist() for p in local],
                    "layout": layouts[root.root_id].to_dict(), "root_pose": context.state[2:5].tolist(),
                    "nominal_route_poses": route.tolist(), "states": states[:end + 1].tolist(),
                    "actions": np.asarray(tape[:end]).tolist(), "observed_steps": end,
                    "candidate": label, "replay": {"repeats": 2, "bitwise_equal": True}}
                # Check geometry/progress first; the replay flag is established below.
                check = assess_witness(record)
                if check["passes"]:
                    reset_root(env, root, record_frames=False)
                    repeat = run_actions(env, tape[:end], record_frames=False)
                    equal = repeat.executed_steps == end and all(np.array_equal(
                        getattr(log, field)[:end + 1], getattr(repeat, field)) for field in
                        ("states", "block_vel", "block_ang_vel", "n_contacts", "terminated", "truncated"))
                    record["replay"]["bitwise_equal"] = bool(equal)
                    check = assess_witness(record)
                attempts.append({"root_id": root.root_id, "candidate": label,
                    "assessment": check, "charged_steps": ledger.total - before})
                if check["passes"]:
                    witnesses.append(record)
                    accepted_episodes.add(int(root.meta["episode"]))
                    print(f"[witness] {root.root_id}: clearance={check['min_clearance']:.2f}px coverage={check['final_coverage']:.3f}", flush=True)
                    break
            if len(accepted_episodes) >= MIN_WITNESSES:
                break
    env.close()
    bank.h5.close()
    if bank_identity(dev_dir) != initial_bank_identity:
        raise ValueError("Development bank changed during witness generation")
    if frozen_plan is not None and validate_development_plan(dev_dir, args.sampling_plan) != frozen_plan:
        raise ValueError("Frozen future sampling plan changed during witness generation")
    report = {"run_id": run_id, "protocol": PROTOCOL, "generator_identity": generator_identity(),
        "development_bank_identity": initial_bank_identity, "development_episodes": development,
        "split_audit": audit, "witnesses": witnesses, "attempts": attempts, "ledger": ledger.to_dict(),
        "generation_phase": "before_final_banks" if args.development_only else "after_full_banks",
        **({"frozen_sampling_plan": frozen_plan} if frozen_plan is not None else {}),
        "gate": feasibility_gate(witnesses, development),
        "freeze_policy": "Validate the already frozen generator without changing or filtering final-test cases; failure requires diagnosis and fresh banks if generator tuning changes it"}
    output = args.output_dir / "witnesses.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    if report["gate"]["passes"]:
        require_feasibility_report(output, dev_dir)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.output_dir / "config.yaml").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(f"# {run_id}\n\nDevelopment-only exact simulator route witnesses for the frozen nominal-swept-route hazard generator. Dense whole-T clearance must stay positive, observed goal coverage must reach 0.90, and reset-and-prefix replay must agree bitwise. Every attempt is charged. Final test cases are never filtered. This finite search does not certify impossibility when it fails.\n")
    write_manifest(args.output_dir, build_manifest(run_id=run_id, kind="development-witness",
        seeds={"search": args.seed}, data={"expert_sha256": file_sha256(args.expert_data),
            "development_bank": report["development_bank_identity"], "generator": report["generator_identity"],
            **({"frozen_sampling_plan": frozen_plan} if frozen_plan is not None else {})},
        costs=ledger.to_dict(), metrics=report["gate"], started_at=started))
    if not args.no_upload:
        HFStore().upload_run("pusht-banks", "development-witness", args.output_dir, run_id=run_id)
    print(f"[witness] gate passes={report['gate']['passes']}; independent sources={len(accepted_episodes)}; charged steps={ledger.total}", flush=True)
    return 0 if report["gate"]["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
