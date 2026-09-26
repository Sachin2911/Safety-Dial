#!/usr/bin/env python3
"""Build immutable E1 banks using a frozen, finite, source-disjoint candidate plan.

Recovery uses --phase development, then --phase final after a passing development
feasibility report. Each source contributes at most one accepted root. Candidate retries
change no geometric/layout/domain filter and never inspect evaluated-method performance.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers.threads import apply_torch, pin_threads

pin_threads()
import numpy as np

apply_torch()

from helpers.acquisitionSafety import MeteredEnv, require_new_paths
from helpers.branchBank import build_root, execute_proposals
from helpers.hfStore import HFStore
from helpers.plannerAudit import AuditedNominalPlanner
from helpers.pushtAssets import H5_PATH, STABLEWM_HOME, load_model, load_scalers
from helpers.pushtBankSampling import (SAMPLING_PROTOCOL, ProgressBankWriter, atomic_json,
    candidate_schedule, exact_proposals, validate_source_roles, verify_plan_sources)
from helpers.pushtContactReplay import CONTACT_COUNTER, execute_tape, make_env, reset_root
from helpers.pushtLayouts import generate_layout
from helpers.pushtReplay import StepLedger
from helpers.pushtSourceFamilies import SOURCE_FAMILY_PROTOCOL, geometric_source_family
from helpers.runManifest import build_manifest, file_sha256, make_run_id, validate_run_id, write_manifest
from helpers.splitIntegrity import validate_bank_splits

ASSETS_RUN = REPO_ROOT / "runs" / "pusht-assets-20260926-1"
STUDY = REPO_ROOT / "data" / "study" / "pusht"
SOURCE_FILES = ["experiments/scripts/pusht_e1_banks.py", "experiments/scripts/pusht_development_witness.py"] + [
    f"experiments/helpers/{name}.py" for name in (
        "pushtBankSampling", "branchBank", "pushtLayouts", "pushtGeometry", "pushtSourceFamilies",
        "pushtReplay", "pushtContactReplay", "imagination", "plannerAudit", "pushtAssets",
        "acquisitionSafety", "runManifest", "splitIntegrity", "threads", "pushtDevelopmentPlan", "pushtFeasibility")]


class PlanningMeter:
    def __init__(self, planner):
        self.planner, self.calls, self.seconds = planner, 0, 0.0

    def plan(self, *args, **kwargs):
        self.calls += 1
        start = time.monotonic()
        try:
            return self.planner.plan(*args, **kwargs)
        finally:
            self.seconds += time.monotonic() - start


def nominal_route(env, root, ctx, ledger):
    env.category = "layout_route"
    reset_root(env, root, record_frames=False)
    log = execute_tape(env, np.asarray(root.meta["nominal_plan"]).reshape(-1, 2), record_frames=False)
    if log.censored:
        ledger.add("discarded_censored_layout_route", 0, branches=1)
        return None
    return np.concatenate([ctx.prefix_log.states[:, 2:5], log.states[1:, 2:5]], 0)


def build_bank(name, raw_env, base_planner, rng, candidates_by_family, targets, n_tapes, *,
               ledger, plan_path, study_dir, capture=None, roots_reuse=None):
    """Complete a bank or preserve explicit incomplete roots, branches and accounting."""
    bank_dir = study_dir / name
    require_new_paths([bank_dir])
    writer = ProgressBankWriter(bank_dir, bank_name=name, plan_path=plan_path)
    env, planner = MeteredEnv(raw_env, ledger, "root_generation"), PlanningMeter(base_planner)
    try:
        writer.checkpoint(ledger)
        accepted_episodes = set()
        for family, target in targets.items():
            accepted, contacts = 0, 0
            if capture is not None:
                capture[family] = []
            writer.progress.update({"family": family, "target_roots": target,
                "scheduled_candidates": len(candidates_by_family[family])})
            sequence = roots_reuse[family] if roots_reuse is not None else candidates_by_family[family]
            for cursor, candidate in enumerate(sequence):
                if accepted >= target:
                    break
                if roots_reuse is None and candidate["episode"] in accepted_episodes:
                    writer.progress["skipped_accepted_sources"] += 1
                    continue
                ledger_before = ledger.total
                brief = ({key: candidate[key] for key in ("episode", "t0", "goal_offset", "k", "pair_draw",
                          "candidate_round", "source_position", "candidate_index", "root_seed", "root_id")}
                         if roots_reuse is None else {"root_id": candidate[0].root_id, "episode": candidate[0].meta["episode"], "reused_test_root": True})
                writer.progress["pending_candidate"] = brief
                writer.progress["schedule_cursor"] = cursor
                writer.event({"event": "started", "candidate": brief, "ledger": ledger.to_dict()})
                writer.checkpoint(ledger)
                outcome = "accepted"
                root = ctx = lay_f = lay_h = None
                if roots_reuse is None:
                    pair = {**candidate, "start": np.asarray(candidate["start"]), "goal": np.asarray(candidate["goal"])}
                    env.category = "root_generation"
                    try:
                        root, ctx, _ = build_root(env, planner, pair, seed=candidate["root_seed"],
                            k=candidate["k"], root_id=candidate["root_id"])
                    except ValueError as exc:
                        if "censored or out-of-domain prefix" not in str(exc):
                            raise
                        outcome = "discarded_invalid_root"
                    if outcome == "accepted" and geometric_source_family(ctx.state, root.goal_state) != family:
                        outcome = "discarded_source_geometry"
                    if outcome == "accepted":
                        root.meta.update({"source_family": family, "source_family_protocol": SOURCE_FAMILY_PROTOCOL["version"],
                            "sampling_plan_sha256": writer.plan_hash,
                            **{key: candidate[key] for key in ("pair_draw", "candidate_round", "source_position", "candidate_index")}})
                        route = nominal_route(env, root, ctx, ledger)
                        if route is None:
                            outcome = "discarded_censored_layout_route"
                    if outcome == "accepted":
                        lay_f = generate_layout(rng, route, ctx.state[2:5], root.goal_state[2:5], family="familiar", root_id=root.root_id)
                        lay_h = None if name == "dev" else generate_layout(rng, route, ctx.state[2:5], root.goal_state[2:5], family="heldout", root_id=root.root_id)
                        if lay_f is None or (name != "dev" and lay_h is None):
                            outcome = "root_discarded_no_layout"
                else:
                    root, ctx, lay_f, lay_h = candidate
                if outcome == "accepted":
                    props, proposal_audit = exact_proposals(rng, root, ctx, n_tapes, stress=name == "stress",
                        hazard_centre=lay_f.shape.centre if name == "stress" else None)
                    writer.event({"event": "proposals", "candidate": brief, "audit": proposal_audit})
                    for label in ("arena_rejected", "duplicate_rejected"):
                        ledger.add("proposals_" + label, 0, branches=sum(row["outcome"] == label for row in proposal_audit["trials"]))
                    ledger.add("proposals_attempted", 0, branches=len(proposal_audit["trials"]))
                    if not proposal_audit["complete"]:
                        if name == "stress":
                            raise RuntimeError(f"Frozen stress root {root.root_id} lacks {n_tapes} valid unique proposals within the declared limit")
                        outcome = "discarded_proposal_shortfall"
                if outcome == "accepted":
                    if root.meta["episode"] in accepted_episodes:
                        raise ValueError("A bank cannot accept two roots from the same source episode")
                    writer.add_root(root)
                    writer.layouts += [lay_f] + ([lay_h] if lay_h is not None else [])
                    # Preserve root/layout identity before querying any branch outcome.
                    writer.checkpoint(ledger)
                    env.category = "branch"
                    for proposal in props:
                        branches = execute_proposals(env, root, [proposal])
                        ledger.add("branch", 0, branches=len(branches))
                        writer.add_branches(branches)
                        writer.checkpoint(ledger)
                    accepted_episodes.add(root.meta["episode"])
                    accepted += 1
                    contacts += int(root.meta.get("in_contact_last_block") is True)
                    if capture is not None:
                        capture[family].append((root, ctx, lay_f, lay_h))
                    writer.progress["accepted_by_family"][family] = accepted
                    writer.progress.setdefault("pusher_contact_roots", {})[family] = contacts
                else:
                    writer.progress["rejections"][outcome] = writer.progress["rejections"].get(outcome, 0) + 1
                    if outcome != "discarded_censored_layout_route":
                        ledger.add(outcome, 0, branches=1)
                writer.progress["attempts_completed"] += 1
                writer.progress["planner"] = {"calls": planner.calls, "wall_seconds": planner.seconds}
                writer.progress.pop("pending_candidate", None)
                writer.event({"event": "completed", "candidate": brief, "outcome": outcome,
                    "charged_steps": ledger.total - ledger_before, "ledger": ledger.to_dict()})
                writer.checkpoint(ledger)
                if outcome == "accepted" or writer.progress["attempts_completed"] % 25 == 0:
                    print(f"[banks:{name}] {family} {accepted}/{target} roots; "
                          f"{writer.progress['attempts_completed']} attempts, {ledger.total} steps, "
                          f"{time.time() - writer.started:.0f}s", flush=True)
            if accepted != target:
                raise RuntimeError(f"Exhausted frozen candidate schedule for {name}/{family}: {accepted}/{target} roots")
        writer.checkpoint(ledger, status="generated")
        return writer, bank_dir, writer.layouts
    except BaseException as exc:
        writer.progress["planner"] = {"calls": planner.calls, "wall_seconds": planner.seconds}
        writer.fail(ledger, exc)
        raise


def make_plan(args, study_dir):
    splits_path = ASSETS_RUN / "splits.json"
    splits = json.loads(splits_path.read_text())
    excluded, exclusion_inputs = set(), []
    for old in [STUDY / name for name in ("dev", "test", "stress")] + args.exclude_bank:
        path = old / "roots.json"
        if path.is_file():
            exclusion_inputs.append(path)
            excluded.update(int(row["meta"]["episode"]) for row in json.loads(path.read_text())["roots"])
    eligible = [e for e in splits["roles"]["roots"] if e not in excluded]
    permutation = np.random.default_rng(args.seed + 1).permutation(eligible)
    dev, familiar, heldout = np.array_split(permutation, [len(permutation) // 3, 2 * len(permutation) // 3])
    roles = {"dev": {"familiar": list(map(int, dev))}, "test": {"familiar": list(map(int, familiar)), "heldout": list(map(int, heldout))}}
    validate_source_roles(roles)
    schedule = {}
    for role, families in roles.items():
        schedule[role] = {}
        for family, episodes in families.items():
            target = args.dev_roots if role == "dev" else args.test_roots
            if len(episodes) < target:
                raise ValueError(f"Too few distinct sources for requested {role}/{family} root count")
            seed = args.seed + (10_000 if role == "test" else 0) + (100_000 if family == "heldout" else 0)
            schedule[role][family] = candidate_schedule(H5_PATH, episodes, seed=seed)
            for candidate in schedule[role][family]:
                candidate.update({"root_seed": seed + candidate["candidate_index"],
                    "root_id": f"{role}-{family}-c{candidate['candidate_index']:05d}"})
    sources = {relative: file_sha256(REPO_ROOT / relative) for relative in SOURCE_FILES}
    for relative in sources:
        destination = study_dir / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
    checkpoint = STABLEWM_HOME / "checkpoints/pusht/lewm_object.ckpt"
    inputs = [splits_path, ASSETS_RUN / "scalers.npz", ASSETS_RUN / "hf_revision.txt", checkpoint, H5_PATH, *exclusion_inputs]
    plan = {"protocol_version": SAMPLING_PROTOCOL["version"], "sampling_protocol": SAMPLING_PROTOCOL,
        "seed": args.seed, "source_roles": roles, "candidate_schedule": schedule,
        "source_family_protocol": SOURCE_FAMILY_PROTOCOL, "source_snapshot_sha256": sources,
        "source_snapshot_policy": "Local only; never included in bank uploads. Exact candidate plan and source hashes are uploaded as experiment data.",
        "excluded_previously_evaluated_episodes": sorted(excluded),
        "settings": {key: getattr(args, key) for key in ("seed", "dev_roots", "test_roots", "dev_tapes", "test_tapes")},
        "input_sha256": {str(path.resolve()): file_sha256(path) for path in inputs},
        "planner": {"class": "AuditedNominalPlanner", "samples": 300, "iterations": 30,
            "change": "Audit and execute the exact returned sequence consistently on clean and legacy planner versions"},
        "recovery_interpretation": "Finite construction retry protocol after v2 single-candidate exhaustion; original source role assignment and all geometric/hazard filters retained"}
    plan_path = study_dir / "sampling-plan.json"
    atomic_json(plan_path, plan)
    return plan_path, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "final", "all"), default="all")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--feasibility-report", type=Path)
    parser.add_argument("--dev-roots", type=int, default=24)
    parser.add_argument("--dev-tapes", type=int, default=8)
    parser.add_argument("--test-roots", type=int, default=64)
    parser.add_argument("--test-tapes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--run-id", type=validate_run_id)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--exclude-bank", type=Path, action="append", default=[])
    args = parser.parse_args()
    study_dir, results_dir = args.output_dir.resolve(), args.results_dir.resolve()
    if study_dir == results_dir or study_dir in results_dir.parents or results_dir in study_dir.parents:
        parser.error("Bank and phase result directories must be disjoint")
    if min(args.dev_roots, args.test_roots, args.dev_tapes, args.test_tapes) <= 0:
        parser.error("Bank counts must be positive")
    run_id = args.run_id or make_run_id("pusht", "banks", n=args.n)
    require_new_paths([results_dir])
    if args.phase == "final":
        if args.plan is None or args.feasibility_report is None:
            parser.error("Final generation requires --plan and --feasibility-report")
        plan_path = args.plan.resolve()
        if plan_path != study_dir / "sampling-plan.json":
            parser.error("Final generation must reuse the development directory's exact frozen plan")
        require_new_paths([study_dir / "test", study_dir / "stress"])
        plan = json.loads(plan_path.read_text())
        verify_plan_sources(plan, plan_path, REPO_ROOT)
        from helpers.pushtFeasibility import require_feasibility_report
        feasibility = require_feasibility_report(args.feasibility_report, study_dir / "dev", sampling_plan=plan_path)
    else:
        if args.plan is not None or args.feasibility_report is not None:
            parser.error("A new development/all phase creates its own plan")
        require_new_paths([study_dir])
        study_dir.mkdir(parents=True)
        plan_path, plan = make_plan(args, study_dir)
        verify_plan_sources(plan, plan_path, REPO_ROOT, verify_inputs=False)
        feasibility = None
    results_dir.mkdir(parents=True)
    settings, plan_hash = plan["settings"], file_sha256(plan_path)
    ledgers = {name: StepLedger() for name in ("dev", "test", "stress")}
    report = {"status": "running", "phase": args.phase, "run_id": run_id, "sampling_plan": str(plan_path),
        "sampling_plan_sha256": plan_hash, "sampling_protocol": SAMPLING_PROTOCOL, "source_roles": plan["source_roles"],
        "source_family_protocol": SOURCE_FAMILY_PROTOCOL, "source_snapshot_sha256": plan["source_snapshot_sha256"],
        "development_feasibility": feasibility, "banks": {}, "hf_revisions": {},
        "excluded_previously_evaluated_episodes": plan["excluded_previously_evaluated_episodes"]}
    atomic_json(results_dir / "banks.json", report)
    started, env, banks = time.time(), None, {}
    previous_handler = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise InterruptedError(f"Bank generation interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        os.environ["STABLEWM_HOME"] = str(STABLEWM_HOME)
        model = load_model("cuda")
        planner = AuditedNominalPlanner(model, load_scalers(ASSETS_RUN / "scalers.npz"), "cuda", num_samples=300, n_steps=30)
        env = make_env()
        test_roots = {}
        phases = ["dev"] if args.phase == "development" else (["test", "stress"] if args.phase == "final" else ["dev", "test", "stress"])
        for name in phases:
            role = "dev" if name == "dev" else "test"
            targets = {"familiar": settings["dev_roots"]} if name == "dev" else {"familiar": settings["test_roots"], "heldout": settings["test_roots"]}
            # Each bank's RNG is independent of whether development ran in this process.
            bank_seed = settings["seed"] + {"dev": 0, "test": 10_000, "stress": 20_000}[name]
            writer, directory, layouts = build_bank(name, env, planner, np.random.default_rng(bank_seed),
                plan["candidate_schedule"][role], targets, settings["dev_tapes"] if name == "dev" else settings["test_tapes"],
                ledger=ledgers[name], plan_path=plan_path, study_dir=study_dir,
                capture=test_roots if name == "test" else None, roots_reuse=test_roots if name == "stress" else None)
            manifest = build_manifest(run_id=f"{run_id}-{name}", kind="bank", seeds={"rng": bank_seed},
                data={"contact_kind": "pusher_block", "contact_counter": CONTACT_COUNTER,
                    "assets_run": ASSETS_RUN.name, "assets_revision": (ASSETS_RUN / "hf_revision.txt").read_text().split()[2],
                    "source_roles": plan["source_roles"], "source_family_protocol": SOURCE_FAMILY_PROTOCOL,
                    "sampling_plan_sha256": plan_hash, "sampling_protocol": SAMPLING_PROTOCOL,
                    "source_snapshot_sha256": plan["source_snapshot_sha256"],
                    "excluded_previously_evaluated_episodes": plan["excluded_previously_evaluated_episodes"]},
                costs=ledgers[name].to_dict(), metrics={"n_layouts": len(layouts), "n_roots": len(writer.roots)}, started_at=started)
            manifest["status"] = "complete"
            writer.finish(ledgers[name], manifest)
            write_manifest(directory, manifest)
            (directory / "README.md").write_text(f"# {run_id}-{name}\n\nComplete Push-T {name} bank. Frozen sampling-plan.json records candidate construction; source snapshot remains local outside this bundle.\n")
            banks[name] = directory
            report["banks"][name] = {"path": str(directory), "status": "complete", "roots": manifest["metrics"]["n_roots"]}
            report["ledgers"] = {key: value.to_dict() for key, value in ledgers.items() if key in phases}
            atomic_json(results_dir / "banks.json", report)
        if args.phase != "development":
            report["split_integrity"] = validate_bank_splits({name: study_dir / name for name in ("dev", "test", "stress")})
        if not args.no_upload:
            store = HFStore()
            for name, directory in banks.items():
                # Only bank data (including candidate plan) is uploaded, never source_snapshot.
                report["hf_revisions"][name] = store.upload_run("pusht-banks", "banks", directory, run_id=f"{run_id}-{name}")
                atomic_json(results_dir / "banks.json", report)
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "incomplete"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if env is not None:
            env.close()
        signal.signal(signal.SIGTERM, previous_handler)
        report["ledgers"] = {name: ledger.to_dict() for name, ledger in ledgers.items()}
        report["wall_clock_s"] = time.time() - started
        atomic_json(results_dir / "banks.json", report)
    print(f"[banks] {args.phase} complete: {results_dir / 'banks.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
