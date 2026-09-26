#!/usr/bin/env python3
"""Revalidate archived E0 reset/prefix/tapes with current masks, without loading a model.

Stored tapes are float32; this checks fresh independent executions of those tapes and
never claims numerical equality to the archived float64 execution. Outputs are fresh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np

from helpers.acquisitionSafety import MeteredEnv, require_new_paths
from helpers.branchBank import Bank, BankWriter, Branch
from helpers.e0Validation import compare_replays, live_geometry_check, timing_row
from helpers.hfStore import HFStore
from helpers.pushtGeometry import Disc, clearance_trace, draw_overlay, tile
from helpers.pushtReplay import (StepLedger, check_substep_equivalence, execute_branch, make_env,
                                reset_root, run_actions)
from helpers.runManifest import build_manifest, validate_run_id, write_manifest


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    out = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            out.update(chunk)
    return out.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-bank", type=Path, required=True)
    ap.add_argument("--bank-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--n-roots", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--min-contact", type=int, default=15)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    validate_run_id(args.run_id)
    if args.n_roots < 50 or args.repeats < 3 or args.min_contact < 15:
        ap.error("E0 needs at least 50 roots, three repeats and 15 mid-contact roots")
    source_path, bank_path, result_path = (p.resolve() for p in (args.source_bank, args.bank_dir, args.results_dir))
    if bank_path == result_path or bank_path in result_path.parents or result_path in bank_path.parents:
        ap.error("Bank and result directories must be disjoint")
    require_new_paths([bank_path, result_path])
    source = Bank(source_path)
    if len(source.roots) < args.n_roots:
        ap.error("Source bank does not contain the required number of roots")
    roots = source.roots[:args.n_roots]
    if len({root.root_id for root in roots}) != len(roots):
        ap.error("Source bank root identifiers must be unique")
    selected = {root.root_id for root in roots}
    result_path.mkdir(parents=True)
    writer = BankWriter(bank_path)
    ledger = StepLedger()
    env = MeteredEnv(make_env(), ledger, "replay")
    report = {"status": "running", "run_id": args.run_id, "source_bank": str(source_path),
              "source_hashes": {name: digest(source_path / name) for name in ("roots.json", "branches.h5")},
              "execution": "CPU physics and rendering only; no world-model queries",
              "tape_precision": "Replays stored float32 tapes, not archived float64 action originals",
              "hazard_rule": "whole-T clearance <= 0 at observed environment steps",
              "branches": [], "invalid_roots": [], "geometry": [], "timing": [], "substep_mirror": [], "overlays": []}
    traced, geometry_roots, substep_roots, contact_roots = set(), set(), set(), set()
    seen_roots = set()
    try:
        for index in range(len(source)):
            root = source.root_of(index)
            if root.root_id not in selected or root.root_id in report["invalid_roots"]:
                continue
            branch = source.branch(index)
            tape = branch["tape"].astype(float).reshape(-1, 2)
            logs, contexts = [], []
            env.category = "replay"
            try:
                for _ in range(args.repeats):
                    ctx, log = execute_branch(env, root, tape)
                    contexts.append(ctx)
                    logs.append(log)
                    ledger.add("replay", 0, branches=1)
            except ValueError as exc:
                if "censored or out-of-domain prefix" not in str(exc):
                    raise
                report["invalid_roots"].append(root.root_id)
                continue
            seen_roots.add(root.root_id)
            ref = logs[0]
            if (contexts[0].prefix_log.n_contacts[-5:] > 0).any():
                contact_roots.add(root.root_id)
            row = {"source_branch_index": index, "root_id": root.root_id, "kind": branch["kind"], **compare_replays(logs)}
            report["branches"].append(row)
            writer.add_root(root)
            writer.add_branches([Branch(root.root_id, tape.reshape(5, 5, 2), branch["kind"], branch["params"], ref)])
            midpoint = ref.states[min(12, ref.executed_steps), 2:4]
            hazard = Disc(float(midpoint[0] + 70), float(midpoint[1]), 30)
            report["timing"].append({"source_branch_index": index, **timing_row(ref, hazard)})
            if root.root_id not in geometry_roots:
                report["geometry"].append({"root_id": root.root_id,
                    **live_geometry_check(env, ref.states[:ref.executed_steps + 1:5, 2:5])})
                geometry_roots.add(root.root_id)
            if len(substep_roots) < 8 and root.root_id not in substep_roots:
                env.category = "substep_standard_and_prefix"
                result = check_substep_equivalence(env, root, tape)
                ledger.add("manual_physics_step_equivalents", len(tape))
                report["substep_mirror"].append({"root_id": root.root_id, "bitwise": result["bitwise"], "max_abs_diff": result["max_abs_diff"],
                    "substeps": result["substeps"].tolist(), "diagnostic_continues_past_done": True})
                substep_roots.add(root.root_id)
            if len(traced) < 10 and root.root_id not in traced and row["contact_steps"]:
                from PIL import Image, ImageDraw

                env.category = "contact_overlay_replay"
                reset_root(env, root, record_frames=False)
                detail = run_actions(env, tape, record_frames=True, frame_every=1)
                contact = int(np.flatnonzero(detail.n_contacts[:detail.executed_steps + 1] > 0)[0])
                indices = list(range(max(0, contact - 2), min(detail.executed_steps, contact + 3) + 1))
                panels, clearances = [], clearance_trace(detail.states[indices, 2:5], hazard)
                for t, clearance in zip(indices, clearances):
                    panel = draw_overlay(detail.frames[t], detail.states[t, 2:5], hazard,
                                         pusher_xy=detail.states[t, :2])
                    canvas = Image.fromarray(panel)
                    draw = ImageDraw.Draw(canvas)
                    draw.rectangle((0, 0, 224, 15), fill="white")
                    draw.text((2, 2), f"t={t} contacts={detail.n_contacts[t]} c={clearance:.2f}", fill="black")
                    panels.append(np.asarray(canvas))
                filename = f"contact_{len(traced):02d}_{root.root_id}.png"
                Image.fromarray(tile(panels, ncols=6)).save(result_path / filename)
                report["overlays"].append({"path": filename, "sha256": digest(result_path / filename),
                    "root_id": root.root_id, "source_branch_index": index, "observed_rows": indices,
                    "hazard": hazard.to_dict(), "clearance_px": clearances.tolist()})
                traced.add(root.root_id)
            if len(report["branches"]) % 25 == 0:
                print(f"[e0-revalidate] {len(report['branches'])} branches; {len(seen_roots)} roots; {ledger.total} steps", flush=True)
        checks = {"enough_roots": len(seen_roots) >= args.n_roots,
                  "enough_mid_contact_roots": len(contact_roots) >= args.min_contact,
                  "prefixes_valid": not report["invalid_roots"],
                  "all_bitwise": bool(report["branches"]) and all(row["bitwise"] and row["frames_equal"] for row in report["branches"]),
                  "geometry": bool(report["geometry"]) and all(row["passes"] for row in report["geometry"]),
                  "substep_mirror": len(substep_roots) == 8 and all(row["bitwise"] for row in report["substep_mirror"]),
                  "ten_contact_overlays": len(traced) == 10}
        report.update(status="complete", n_roots=len(seen_roots), n_mid_contact_roots=len(contact_roots),
                      n_censored=sum(row["censored"] for row in report["branches"]),
                      n_domain_exit=sum(row["observation_domain_exit"] for row in report["branches"]),
                      gate={"passes": all(checks.values()), "checks": checks},
                      visual_review={"status": "pending", "required": "Inspect all ten fresh contact overlays and record checks against their hashes"})
    except BaseException as exc:
        report.update(status="incomplete", error_type=type(exc).__name__, gate={"passes": False})
        raise
    finally:
        source.h5.close()
        env.close()
        report["ledger"] = ledger.to_dict()
        report["accounting"] = {"env_step_calls": ledger.total - ledger.counts.get("manual_physics_step_equivalents", 0),
                                "manual_physics_step_equivalents": ledger.counts.get("manual_physics_step_equivalents", 0),
                                "physics_substeps_per_step": 10, "reset_internal_physics": "not an environment transition"}
        manifest = build_manifest(run_id=args.run_id, kind="e0-revalidation", seeds={},
                    data={"source_bank": str(source_path), "source_hashes": report["source_hashes"]}, costs=ledger.to_dict())
        writer.finish(ledger, manifest)
        write_manifest(bank_path, manifest)
        write_manifest(result_path, manifest)
        write_json(result_path / "replay_report.json", report)
    if not args.no_upload:
        store = HFStore()
        report["hf_revision"] = store.upload_run("pusht-banks", "banks", bank_path, run_id=args.run_id)
        report["hf_repo"] = store.repo_id("pusht-banks")
        write_json(result_path / "replay_report.json", report)
        store.upload_run("pusht", "e0-revalidation", result_path, run_id=args.run_id)
    print(json.dumps({"gate": report["gate"], "visual_review": report["visual_review"], "results_dir": str(result_path)}))
    return 0 if report["gate"]["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
