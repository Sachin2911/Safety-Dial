#!/usr/bin/env python3
"""Recover only stress controls on all frozen v3 test roots; never regenerate roots.

The old failed stress queries remain paid history. All 128 replacement pools are
frozen before constructing an environment. The v3 banks and witness stay untouched.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))
import h5py
import numpy as np

from helpers.acquisitionSafety import MeteredEnv, copy_ledger, require_new_paths
from helpers.branchBank import Proposal, execute_proposals
from helpers.hfStore import HFStore
from helpers.pushtBankSampling import ProgressBankWriter, atomic_json, verify_plan_sources
from helpers.pushtContactReplay import CONTACT_COUNTER, make_env
from helpers.pushtFeasibility import require_feasibility_report
from helpers.pushtLayouts import load_layouts
from helpers.pushtReplay import Root, StepLedger
from helpers.runManifest import build_manifest, file_sha256, validate_run_id, write_manifest
from helpers.splitIntegrity import bank_identity, validate_bank_splits

TEST_ROOTS, DEV_ROOTS, TAPES = 128, 24, 16
FAILED_STRESS_STEPS = 13040
NEW_SOURCES = ("experiments/scripts/pusht_stress_recovery.py",
               "experiments/helpers/pushtStressRecovery.py")


def read_json(path):
    return json.loads(Path(path).read_text())


def tree_hashes(directory):
    """Reject links and bind every regular byte copied; never traverse an input link."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Expected an ordinary directory: {directory}")
    paths = sorted(directory.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ValueError(f"Symlinks are forbidden in preserved input trees: {directory}")
    return {str(p.relative_to(directory)): file_sha256(p) for p in paths if p.is_file()}


def validate_complete_bank(path, n_roots, tapes):
    blob, progress, manifest = (read_json(path / name) for name in
                                ("roots.json", "progress.json", "manifest.json"))
    if (blob.get("status") != "complete" or progress.get("status") != "complete"
            or manifest.get("status") != "complete" or blob.get("manifest") != manifest
            or blob["ledger"] != progress["ledger"] or blob["ledger"] != manifest["costs"]):
        raise ValueError(f"Source bank is not a coherently completed bank: {path}")
    roots = blob["roots"]
    if (len(roots) != n_roots or len({r["root_id"] for r in roots}) != n_roots
            or len({r["meta"]["episode"] for r in roots}) != n_roots
            or blob.get("n_branches") != n_roots * tapes):
        raise ValueError("Source bank root/branch counts differ from the fixed protocol")
    for name, digest in progress["snapshot_sha256"].items():
        if file_sha256(path / name) != digest:
            raise ValueError("Source completed-bank snapshot hash differs")
    with h5py.File(path / "branches.h5", "r") as h5:
        indices = h5["root_index"][:]
        if (len(indices) != n_roots * tapes or len(h5["tape"]) != len(indices)
                or set(map(int, indices)) != set(range(n_roots))
                or not np.all(np.bincount(indices) == tapes)):
            raise ValueError("Source H5 does not contain the exact tapes per frozen root")
    return blob


def validate_failed_stress(path, test_roots):
    """Read only failed construction metadata, never use stress outcomes for selection."""
    progress, roots = read_json(path / "progress.json"), read_json(path / "roots.json")
    events = [json.loads(line) for line in (path / "attempts.jsonl").read_text().splitlines()]
    completed = [e for e in events if e["event"] == "completed"]
    ledger = progress["ledger"]
    if (progress.get("status") != "incomplete" or roots.get("status") != "incomplete"
            or not events or events[-1].get("event") != "failed"
            or events[-1].get("ledger") != ledger or roots["ledger"] != ledger
            or ledger["total_steps"] != FAILED_STRESS_STEPS
            or sum(ledger["steps"].values()) != ledger["total_steps"]
            or sum(e["charged_steps"] for e in completed) != ledger["total_steps"]
            or progress["attempts_completed"] != len(completed)
            or roots["n_branches"] != progress["n_branches"]
            or ledger["branches"].get("branch", 0) != roots["n_branches"]):
        raise ValueError("Failed stress history is not the charged terminal v3 failure")
    if roots["roots"] != test_roots[:len(roots["roots"])]:
        raise ValueError("Failed stress roots differ from the fixed test roots")
    for name, digest in progress["snapshot_sha256"].items():
        if file_sha256(path / name) != digest:
            raise ValueError("Failed-stress snapshot hash differs")
    return {"path": str(path.resolve()), "ledger": ledger,
            "error": progress.get("error"), "roots": len(roots["roots"]),
            "branches": roots["n_branches"],
            "metadata_sha256": {name: file_sha256(path / name) for name in
                                ("roots.json", "progress.json", "attempts.jsonl", "layouts.json")},
            "paid_bank_sha256": file_sha256(path / "branches.h5")}


def preflight(source, witness_path):
    source, witness_path = Path(source).resolve(), Path(witness_path).resolve()
    plan_path = source / "sampling-plan.json"
    if plan_path.is_symlink():
        raise ValueError("Sampling plan must be a regular copied file")
    plan = read_json(plan_path)
    verify_plan_sources(plan, plan_path, REPO_ROOT, verify_inputs=False)
    if plan["settings"] != {"seed": 20260927, "dev_roots": DEV_ROOTS,
                            "test_roots": TEST_ROOTS // 2, "dev_tapes": 8, "test_tapes": TAPES}:
        raise ValueError("Recovery requires the original full v3 root/tape settings")
    dev = validate_complete_bank(source / "dev", DEV_ROOTS, 8)
    test = validate_complete_bank(source / "test", TEST_ROOTS, TAPES)
    witness = require_feasibility_report(witness_path, source / "dev", sampling_plan=plan_path)
    validate_bank_splits({"dev": source / "dev", "test": source / "test"})
    candidates = {c["root_id"]: (family, c) for family, rows in
                  plan["candidate_schedule"]["test"].items() for c in rows}
    family_counts = {family: 0 for family in ("familiar", "heldout")}
    for root in test["roots"]:
        family, candidate = candidates[root["root_id"]]
        meta = root["meta"]
        if (meta["source_family"] != family or meta["episode"] not in plan["source_roles"]["test"][family]
                or root["seed"] != candidate["root_seed"] or root["start_state"] != candidate["start"]
                or root["goal_state"] != candidate["goal"]
                or any(meta[k] != candidate[k] for k in
                       ("episode", "t0", "goal_offset", "k", "pair_draw", "candidate_round", "candidate_index"))):
            raise ValueError("Test root differs from the frozen candidate plan")
        family_counts[family] += 1
    if any(n != TEST_ROOTS // 2 for n in family_counts.values()):
        raise ValueError("Recovery requires all familiar and heldout roots")
    failed = validate_failed_stress(source / "stress", test["roots"])
    return {"source": source, "plan": plan, "plan_sha256": file_sha256(plan_path),
            "dev": dev, "test": test, "witness": witness, "failed_stress": failed,
            "identities": {name: bank_identity(source / name) for name in ("dev", "test")},
            "trees": {name: tree_hashes(source / name) for name in ("dev", "test", "source_snapshot")},
            "failed_stress_tree": tree_hashes(source / "stress"),
            "witness_sha256": file_sha256(witness_path),
            "witness_upload_receipt": (read_json(witness_path.parent / "hf_upload.json")
                if (witness_path.parent / "hf_upload.json").exists() else None)}


def copy_inputs(inputs, output, source_witness, feasibility_output, run_id):
    """Relocate only witness binding; all preserved bank/source bytes remain exact."""
    source = inputs["source"]
    output.mkdir(parents=True)
    for name, expected in inputs["trees"].items():
        shutil.copytree(source / name, output / name, copy_function=shutil.copy2)
        if tree_hashes(source / name) != expected or tree_hashes(output / name) != expected:
            raise ValueError("Preserved source bytes changed while copying")
    shutil.copy2(source / "sampling-plan.json", output / "sampling-plan.json")
    if file_sha256(output / "sampling-plan.json") != inputs["plan_sha256"]:
        raise ValueError("Copied root plan differs")
    for name, identity in inputs["identities"].items():
        if bank_identity(output / name) != identity:
            raise ValueError("Copied bank identity differs")
    if file_sha256(source_witness) != inputs["witness_sha256"]:
        raise ValueError("Source feasibility evidence changed while copying")
    witness = read_json(source_witness)
    relocated = deepcopy(witness)
    relocated["frozen_sampling_plan"]["path"] = str((output / "sampling-plan.json").resolve())
    relocated["relocation"] = {"original_report": str(Path(source_witness).resolve()),
        "original_report_sha256": inputs["witness_sha256"],
        "original_upload_receipt": inputs.get("witness_upload_receipt"),
        "new_witness_simulator_steps": 0,
        "original_chronology_retained": "generation_phase refers to the original witness, not this relocation",
        "original_sampling_plan": str((source / "sampling-plan.json").resolve()),
        "copied_sampling_plan": str((output / "sampling-plan.json").resolve()),
        "sampling_plan_sha256": inputs["plan_sha256"], "bank_identities": inputs["identities"],
        "changed_original_fields": ["frozen_sampling_plan.path"],
        "interpretation": "Same witnessed development bank and frozen generator; path relocation only"}
    feasibility_output.mkdir(parents=True)
    path = feasibility_output / "witnesses.json"
    atomic_json(path, relocated)
    binding = require_feasibility_report(path, output / "dev", sampling_plan=output / "sampling-plan.json")
    write_manifest(feasibility_output, build_manifest(run_id=run_id + "-witness", kind="development-witness",
        data={"relocation": relocated["relocation"], "report_sha256": file_sha256(path)},
        costs={"new_simulator_steps": 0, "inherited_witness_ledger": read_json(source_witness)["ledger"]},
        metrics={"gate": binding}))
    return binding


def freeze_controls(inputs, output):
    from helpers.pushtStressRecovery import (STRESS_RECOVERY_PROTOCOL, StressProposalShortfall,
                                            preflight_stress_roots)

    roots = [Root.from_dict(r) for r in inputs["test"]["roots"]]
    layouts, _ = load_layouts(output / "test" / "layouts.json")
    familiar = {layout.root_id: layout for layout in layouts if layout.family == "familiar"}
    with h5py.File(output / "test" / "branches.h5", "r") as h5:
        indices = h5["root_index"][:]
        states = np.stack([h5["states"][int(np.flatnonzero(indices == i)[0]), 0] for i in range(len(roots))])
    centres = np.stack([familiar[root.root_id].shape.centre for root in roots])
    # All random control construction finishes before make_env or any real reset.
    directory = output / "stress-controls"
    directory.mkdir()
    try:
        pools, plan = preflight_stress_roots(roots, states, centres, expected_roots=TEST_ROOTS)
    except StressProposalShortfall as exc:
        atomic_json(directory / "preflight-failure.json", {"status": "incomplete",
            "root_id": getattr(exc, "root_id", None), "audit": exc.audit, "simulator_steps": 0})
        raise
    arrays = directory / "tapes.npz"
    np.savez_compressed(arrays, tapes=np.stack([[p.tape for p in pool] for pool in pools]),
        kinds=np.asarray([[p.kind for p in pool] for pool in pools]),
        params=np.asarray([[json.dumps(p.params, sort_keys=True) for p in pool] for pool in pools]),
        root_ids=np.asarray([r.root_id for r in roots]))
    plan.update({"original_sampling_plan_sha256": inputs["plan_sha256"],
        "source_bank_identities": inputs["identities"], "failed_stress": inputs["failed_stress"],
        "tapes_npz_sha256": file_sha256(arrays),
        "source_sha256": {rel: file_sha256(REPO_ROOT / rel) for rel in NEW_SOURCES},
        "frozen_generator_source_sha256": inputs["plan"]["source_snapshot_sha256"],
        "new_simulator_steps_when_frozen": 0, "protocol": STRESS_RECOVERY_PROTOCOL})
    atomic_json(directory / "controls-plan.json", plan)
    return roots, layouts, directory


def load_frozen_controls(directory, roots):
    from helpers.branchBank import exits_arena
    from helpers.pushtStressRecovery import STRESS_RECOVERY_PROTOCOL

    plan = read_json(directory / "controls-plan.json")
    if (plan.get("complete") is not True or plan["protocol"] != STRESS_RECOVERY_PROTOCOL
            or plan["root_count"] != len(roots) or plan["branch_count"] != len(roots) * TAPES
            or file_sha256(directory / "tapes.npz") != plan["tapes_npz_sha256"]):
        raise ValueError("Frozen stress controls are incomplete or changed")
    for rel, digest in {**plan["source_sha256"], **plan["frozen_generator_source_sha256"]}.items():
        if file_sha256(REPO_ROOT / rel) != digest:
            raise ValueError("Stress implementation changed after controls were frozen")
    pools, digest = [], hashlib.sha256()
    with np.load(directory / "tapes.npz", allow_pickle=False) as arrays:
        tapes = arrays["tapes"]
        if tapes.dtype != np.float32 or tapes.shape != (len(roots), TAPES, 5, 5, 2):
            raise ValueError("Frozen control shape/dtype differs")
        if list(arrays["root_ids"]) != [r.root_id for r in roots]:
            raise ValueError("Frozen control root order differs")
        for i, root in enumerate(roots):
            record = plan["roots"][i]
            hashes = [hashlib.sha256(t.tobytes()).hexdigest() for t in tapes[i]]
            if (record["root_id"] != root.root_id or len(set(hashes)) != TAPES
                    or hashes != record["accepted_tape_sha256"]):
                raise ValueError("Frozen control hashes differ from prequery proposals")
            trials = [t for t in record["audit"]["trials"] if t["outcome"] == "accepted"]
            if len(trials) != TAPES or [t["tape_sha256"] for t in trials] != hashes:
                raise ValueError("Frozen trials differ from retained control hashes")
            pool = []
            for j, tape in enumerate(tapes[i]):
                kind, params = str(arrays["kinds"][i, j]), json.loads(str(arrays["params"][i, j]))
                if (kind != ("stress" if j < TAPES // 2 else "toward_hazard")
                        or trials[j]["kind"] != kind or trials[j]["params"] != params
                        or exits_arena(tape, np.asarray(record["state"])[:2])):
                    raise ValueError("Frozen stress slot kind or arena guard differs")
                digest.update(root.root_id.encode() + b"\0" + kind.encode() + b"\0" + tape.tobytes())
                pool.append(Proposal(tape.copy(), kind, params))
            pools.append(pool)
    if digest.hexdigest() != plan["pool_sha256"]:
        raise ValueError("Frozen full-pool digest differs")
    return pools, plan


def study_costs(inputs, new_ledger):
    ledger = StepLedger()
    for prefix, costs in (("original_dev_", inputs["dev"]["ledger"]),
                          ("original_test_", inputs["test"]["ledger"]),
                          ("failed_stress_", inputs["failed_stress"]["ledger"]),
                          ("original_witness_", inputs["witness_ledger"]),
                          ("new_stress_", new_ledger.to_dict())):
        copy_ledger(costs, ledger, prefix)
    return {"new_stress": new_ledger.to_dict(), "failed_stress": inputs["failed_stress"]["ledger"],
            "inherited_dev": inputs["dev"]["ledger"], "inherited_test": inputs["test"]["ledger"],
            "inherited_witness": inputs["witness_ledger"], "total_study": ledger.to_dict(),
            "accounting": "Inherited costs paid once previously; only new_stress is new simulator execution",
            "scope": "Accounted v3/v4 development, test, witness, failed stress and recovered stress construction only; excludes Walker and historical experiments",
            "unaccounted_history": {"v2_failed_bank_steps": None,
                "interpretation": "Unknown, not zero; the preserved v2 failure record remains separate"}}


def replay_stress(output, roots, layouts, pools, control_plan, controls_dir, inputs, run_id, ledger):
    writer = ProgressBankWriter(output / "stress", bank_name="stress", plan_path=output / "sampling-plan.json")
    # Experiment controls/provenance may upload; source byte snapshots never enter this bank.
    shutil.copytree(controls_dir, writer.dir / "controls")
    env = None
    try:
        for row in control_plan["roots"]:
            trials = row["audit"]["trials"]
            ledger.add("proposals_attempted", 0, branches=len(trials))
            for category in ("arena_rejected", "duplicate_rejected"):
                ledger.add("proposals_" + category, 0,
                           branches=sum(trial["outcome"] == category for trial in trials))
        writer.checkpoint(ledger)
        env = MeteredEnv(make_env(), ledger, "branch")
        for ri, (root, pool) in enumerate(zip(roots, pools, strict=True)):
            before = ledger.total
            candidate = {"root_id": root.root_id, "episode": root.meta["episode"], "reused_test_root": True,
                         "root_index": ri, "controls_plan_sha256": file_sha256(controls_dir / "controls-plan.json")}
            writer.progress["pending_candidate"] = candidate
            writer.event({"event": "started", "candidate": candidate, "ledger": ledger.to_dict()})
            writer.event({"event": "proposals", "candidate": candidate, "audit": control_plan["roots"][ri]["audit"]})
            writer.add_root(root)
            writer.layouts.extend(layout for layout in layouts if layout.root_id == root.root_id)
            writer.checkpoint(ledger)
            for proposal in pool:
                branches = execute_proposals(env, root, [proposal])
                ledger.add("branch", 0, branches=len(branches))
                writer.add_branches(branches)
                writer.checkpoint(ledger)
                # A paid trace is retained before reporting an unexpected replay root.
                if any(not np.array_equal(branch.log.states[0],
                           np.asarray(control_plan["roots"][ri]["state"], dtype=float))
                       for branch in branches):
                    raise ValueError("Replayed initial state differs from frozen saved test state")
            family = root.meta["source_family"]
            writer.progress["accepted_by_family"][family] = writer.progress["accepted_by_family"].get(family, 0) + 1
            writer.progress["attempts_completed"] += 1
            writer.progress.pop("pending_candidate")
            writer.event({"event": "completed", "candidate": candidate, "outcome": "accepted",
                          "charged_steps": ledger.total - before, "ledger": ledger.to_dict()})
            writer.checkpoint(ledger)
            print(f"[stress-recovery] {ri + 1}/{len(roots)} roots; {len(pool)} tapes; {ledger.total} new steps", flush=True)
        if (len(writer.roots) != TEST_ROOTS or writer.h5["tape"].shape[0] != TEST_ROOTS * TAPES
                or not np.all(np.bincount(writer.h5["root_index"][:]) == TAPES)):
            raise ValueError("Recovered stress bank did not preserve exact root/tape counts")
        writer.checkpoint(ledger, status="generated")
        splits = validate_bank_splits({name: output / name for name in ("dev", "test", "stress")})
        if writer.roots != inputs["test"]["roots"]:
            raise ValueError("Recovered stress roots changed the fixed test root identities")
        for name, identity in inputs["identities"].items():
            if bank_identity(output / name) != identity:
                raise ValueError("A preserved evaluation bank changed during stress replay")
        manifest = build_manifest(run_id=run_id + "-stress", kind="bank",
            seeds={"stress_proposals": 20280927}, data={"contact_kind": "pusher_block", "contact_counter": CONTACT_COUNTER,
                "source_roles": inputs["plan"]["source_roles"], "sampling_plan_sha256": inputs["plan_sha256"],
                "root_sampling_protocol": inputs["plan"]["sampling_protocol"], "stress_protocol": control_plan["protocol"],
                "controls_plan_sha256": file_sha256(controls_dir / "controls-plan.json"),
                "controls_pool_sha256": control_plan["pool_sha256"], "source_bank_identities": inputs["identities"],
                "source_sha256": control_plan["source_sha256"], "failed_stress": inputs["failed_stress"]},
            costs=ledger.to_dict(), metrics={"n_roots": len(roots), "n_branches": TEST_ROOTS * TAPES})
        manifest["status"] = "complete"
        writer.finish(ledger, manifest)
        write_manifest(writer.dir, manifest)
        return splits
    except BaseException as exc:
        if writer.h5.id.valid:
            writer.fail(ledger, exc)
        raise
    finally:
        if env is not None:
            env.close()



def archive_failed_stress(inputs, output, run_id):
    """Preserve paid incomplete experiment data separately; no replay or source bytes."""
    source, destination = inputs["source"] / "stress", output / "prior-failed-stress"
    if (source / "manifest.json").exists():
        raise ValueError("Expected the original failed bank without a final manifest")
    expected = inputs["failed_stress_tree"]
    if any(Path(name).suffix in {".py", ".pyc", ".patch", ".diff", ".sh"}
           or "source_snapshot" in Path(name).parts for name in expected):
        raise ValueError("Source bytes cannot enter the failed-bank data archive")
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    if tree_hashes(source) != expected or tree_hashes(destination) != expected:
        raise ValueError("Failed paid-history bytes changed while copying")
    manifest = build_manifest(run_id=run_id + "-prior-failed-stress", kind="prior-failed-stress",
        data={"origin": str(source), "origin_file_sha256": expected,
              "failed_stress": inputs["failed_stress"], "new_simulator_steps": 0,
              "interpretation": "Archive of incomplete paid history; never a completed evaluation bank"},
        costs=inputs["failed_stress"]["ledger"])
    manifest["status"] = "incomplete"
    write_manifest(destination, manifest)
    return {"path": str(destination), "status": "incomplete", "origin_file_sha256": expected,
            "manifest_sha256": file_sha256(destination / "manifest.json"),
            "run_id": manifest["run_id"], "new_simulator_steps": 0}


def run(args):
    output, results, feasibility_output = (Path(p).resolve() for p in
        (args.output_dir, args.results_dir, args.feasibility_output))
    paths = [output, results, feasibility_output]
    require_new_paths(paths)
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(paths) for b in paths[i + 1:]):
        raise ValueError("Recovery output sets must be separate nonnested fresh directories")
    source = Path(args.source_banks_dir).resolve()
    source_witness_dir = Path(args.source_feasibility_report).resolve().parent
    if any(path == protected or protected in path.parents
           for path in paths for protected in (source, source_witness_dir)):
        raise ValueError("Recovery outputs cannot be nested inside preserved input trees")
    inputs = preflight(args.source_banks_dir, args.source_feasibility_report)
    inputs["witness_ledger"] = read_json(args.source_feasibility_report)["ledger"]
    report = {"status": "preparing", "phase": "stress_only_recovery", "run_id": args.run_id,
              "source_banks_dir": str(inputs["source"]), "sampling_plan_sha256": inputs["plan_sha256"],
              "source_bank_identities": inputs["identities"], "failed_stress": inputs["failed_stress"],
              "banks": {}, "hf_revisions": {}, "no_model_loaded": True}
    results.mkdir(parents=True)
    ledger, started = StepLedger(), time.time()
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise InterruptedError(f"Stress recovery interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        report["development_feasibility"] = copy_inputs(inputs, output, args.source_feasibility_report,
                                                        feasibility_output, args.run_id)
        atomic_json(results / "banks.json", report)
        roots, layouts, controls_dir = freeze_controls(inputs, output)
        pools, controls = load_frozen_controls(controls_dir, roots)
        # Source binding is rechecked after pure planning, immediately before any simulation.
        verify_plan_sources(inputs["plan"], output / "sampling-plan.json", REPO_ROOT, verify_inputs=False)
        require_feasibility_report(feasibility_output / "witnesses.json", output / "dev",
                                   sampling_plan=output / "sampling-plan.json")
        for name, identity in inputs["identities"].items():
            if bank_identity(output / name) != identity:
                raise ValueError("Copied bank changed after controls were frozen")
        report["prior_failed_stress_archive"] = archive_failed_stress(inputs, output, args.run_id)
        atomic_json(results / "banks.json", report)
        store = None if args.no_upload else HFStore()
        if store is not None:
            report["hf_revisions"]["prior_failed_stress"] = store.upload_run(
                "pusht-banks", "prior-failed-stress", output / "prior-failed-stress",
                run_id=args.run_id + "-prior-failed-stress")
            atomic_json(results / "banks.json", report)
        report.update(status="running", controls_plan=str(controls_dir / "controls-plan.json"),
                      controls_plan_sha256=file_sha256(controls_dir / "controls-plan.json"),
                      controls_pool_sha256=controls["pool_sha256"], stress_protocol=controls["protocol"])
        atomic_json(results / "banks.json", report)
        report["split_integrity"] = replay_stress(output, roots, layouts, pools, controls, controls_dir,
                                                 inputs, args.run_id, ledger)
        report["banks"] = {name: {"path": str(output / name), "status": "complete",
                                  "roots": DEV_ROOTS if name == "dev" else TEST_ROOTS}
                           for name in ("dev", "test", "stress")}
        if store is not None:
            # The unchanged dev receipt remains byte-for-byte original, avoiding duplicate upload.
            report["dev_upstream"] = store.reference_run("pusht-banks", "banks", output / "dev")
            report["hf_revisions"]["dev"] = report["dev_upstream"]["revision"]
            atomic_json(results / "banks.json", report)
            for name in ("test", "stress"):
                directory = output / name
                if name == "test" and (directory / "hf_upload.json").exists():
                    revision = store.reference_run("pusht-banks", "banks", directory)["revision"]
                else:
                    revision = store.upload_run("pusht-banks", "banks", directory,
                                                run_id=read_json(directory / "manifest.json")["run_id"])
                report["hf_revisions"][name] = revision
                atomic_json(results / "banks.json", report)
            report["hf_revisions"]["development_witness"] = store.upload_run(
                "pusht-banks", "development-witness", feasibility_output, run_id=args.run_id + "-witness")
            atomic_json(results / "banks.json", report)
        report["status"] = "complete"
        report["costs"] = study_costs(inputs, ledger)
        atomic_json(results / "banks.json", report)
        write_manifest(results, build_manifest(run_id=args.run_id, kind="stress-recovery",
            data={"report_sha256": file_sha256(results / "banks.json"),
                  "source_bank_identities": inputs["identities"], "controls_plan_sha256": report["controls_plan_sha256"],
                  "source_sha256": controls["source_sha256"], "hf_revisions": report["hf_revisions"]},
            costs=report["costs"], started_at=started))
        return report
    except BaseException as exc:
        report.update(status="incomplete", error={"type": type(exc).__name__, "message": str(exc)},
                      costs=study_costs(inputs, ledger))
        atomic_json(results / "banks.json", report)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-banks-dir", type=Path, required=True)
    parser.add_argument("--source-feasibility-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--feasibility-output", type=Path, required=True, help="Fresh relocated witness bundle directory")
    parser.add_argument("--run-id", type=validate_run_id, required=True)
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()
    # The CLI is CPU replay only; importing this module does not alter other callers.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    report = run(args)
    print(json.dumps({"status": report["status"], "new_stress_steps": report["costs"]["new_stress"]["total_steps"],
                      "total_study_steps": report["costs"]["total_study"]["total_steps"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
