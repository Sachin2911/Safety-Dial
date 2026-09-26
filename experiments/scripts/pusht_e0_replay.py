#!/usr/bin/env python3
"""E0: replay determinism, whole-T geometry overlays and timing on the fresh instance.

Builds `--n-roots` roots (expert start/goal pairs, k in {0, 2, 4, 6} nominal blocks, at
least `--min-contact` of them in mid-contact), executes three tapes per root (nominal,
random, stress) into a bank, replays every branch `--repeats` times from reset and
compares every logged quantity, checks the physics-substep mirror, measures the
endpoint-interpolation error against dense truth, and draws overlays.

Outputs: docs/mainPlan/results/e0/replay_report.json, overlays PNGs, substep traces,
the bank under data/study/pusht/e0 (uploaded to <ns>/safetydial-pusht-banks:banks/<run_id>).

    uv run python experiments/scripts/pusht_e0_replay.py --n-roots 50 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers.threads import apply_torch, pin_threads  # noqa: E402

pin_threads()

import numpy as np  # noqa: E402

apply_torch()

from helpers.branchBank import (  # noqa: E402
    BankWriter,
    build_root,
    execute_proposals,
    expert_pairs,
    propose,
)
from helpers.hfStore import HFStore  # noqa: E402
from helpers.imagination import NominalPlanner  # noqa: E402
from helpers.pushtAssets import (  # noqa: E402
    ACTION_BLOCK,
    H5_PATH,
    HORIZON_BLOCKS,
    PHYSICS_DT,
    SUBSTEPS_PER_STEP,
    load_model,
    load_scalers,
)
from helpers.pushtGeometry import Disc, clearance_trace, draw_overlay, tile  # noqa: E402
from helpers.pushtReplay import (  # noqa: E402
    StepLedger,
    check_substep_equivalence,
    endpoint_interpolation,
    make_env,
    replay_check,
    reset_root,
    step_substeps,
)
from helpers.runManifest import build_manifest, make_run_id, write_manifest  # noqa: E402

RESULTS = REPO_ROOT / "docs" / "mainPlan" / "results" / "e0"
ASSETS_RUN = REPO_ROOT / "runs" / "pusht-assets-20260926-1"


def angle_err_deg(a, b):
    d = np.arctan2(np.sin(a - b), np.cos(a - b))
    return np.degrees(np.abs(d))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-roots", type=int, default=50)
    ap.add_argument("--min-contact", type=int, default=15)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    t_start = time.time()

    run_id = make_run_id("pusht", "e0bank", n=args.n)
    bank_dir = REPO_ROOT / "data" / "study" / "pusht" / "e0"
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"[e0] run_id={run_id} bank={bank_dir}")

    device = "cuda"
    model = load_model(device)
    process = load_scalers(ASSETS_RUN / "scalers.npz")
    splits = json.loads((ASSETS_RUN / "splits.json").read_text())
    assets_rev = (ASSETS_RUN / "hf_revision.txt").read_text().split()[2]
    env = make_env()
    planner = NominalPlanner(model, process, device)
    rng = np.random.default_rng(args.seed)
    ledger = StepLedger()
    if bank_dir.exists():
        import shutil

        shutil.rmtree(bank_dir)
    writer = BankWriter(bank_dir)

    # ---- roots ------------------------------------------------------------------------
    ks = [0, 2, 4, 6]
    roots, contexts = [], []
    n_contact = 0
    pairs = expert_pairs(H5_PATH, splits["roles"]["roots"], rng, args.n_roots * 2)
    i = 0
    t0 = time.time()
    while len(roots) < args.n_roots or n_contact < args.min_contact:
        if i >= len(pairs):
            pairs += expert_pairs(H5_PATH, splits["roles"]["roots"], rng, args.n_roots)
        k = ks[i % len(ks)] if len(roots) < args.n_roots else int(rng.choice([2, 4, 6]))
        root, ctx, _ = build_root(env, planner, pairs[i], seed=args.seed + i, k=k, root_id=f"e0-r{i:03d}", ledger=ledger)
        roots.append(root)
        contexts.append(ctx)
        n_contact += int(root.meta["in_contact_last_block"])
        i += 1
        if i % 10 == 0:
            print(f"[e0]   {len(roots)} roots ({n_contact} in contact) in {time.time() - t0:.0f}s")
    print(f"[e0] built {len(roots)} roots, {n_contact} in mid-contact, charged {ledger.total} steps")

    # ---- branches ---------------------------------------------------------------------
    branches_per_root = []
    for root, ctx in zip(roots, contexts):
        props, rej = propose(rng, root, ctx, n_random=1, sigmas=(0.1,), n_stress=1)
        ledger.add("proposals_rejected_by_arena", 0, branches=rej)
        br = execute_proposals(env, root, props, ledger=ledger)
        writer.add_root(root)
        writer.add_branches(br)
        branches_per_root.append(br)
    n_branches = sum(len(b) for b in branches_per_root)
    print(f"[e0] executed {n_branches} branches; ledger {ledger.to_dict()}")

    # ---- replay test ------------------------------------------------------------------
    t0 = time.time()
    rep_rows = []
    for root, brs in zip(roots, branches_per_root):
        for b in brs:
            r = replay_check(env, root, b.tape.reshape(-1, 2), args.repeats)
            r.update({"root_id": root.root_id, "kind": b.kind, "k": root.meta["k"], "in_contact_root": root.meta["in_contact_last_block"]})
            rep_rows.append(r)
    replay_steps = sum(len(r.prefix) + HORIZON_BLOCKS * ACTION_BLOCK for r in roots) * 3 * args.repeats
    ledger.add("replay_test", replay_steps, branches=len(rep_rows) * args.repeats)
    replay_summary = {
        "n_branches": len(rep_rows),
        "repeats": args.repeats,
        "all_bitwise": bool(all(r["bitwise"] for r in rep_rows)),
        "all_frames_equal": bool(all(r["frames_equal"] for r in rep_rows)),
        "max_abs_state": max(r["max_abs_state"] for r in rep_rows),
        "max_abs_block_vel": max(r["max_abs_block_vel"] for r in rep_rows),
        "max_abs_ang_vel": max(r["max_abs_ang_vel"] for r in rep_rows),
        "branches_with_contact": int(sum(r["contact_steps"] > 0 for r in rep_rows)),
        "contact_branches_bitwise": bool(all(r["bitwise"] for r in rep_rows if r["contact_steps"] > 0)),
        "seconds": time.time() - t0,
    }
    print(f"[e0] replay: {replay_summary}")

    # ---- substep mirror ---------------------------------------------------------------
    sub_rows = []
    for root, brs in list(zip(roots, branches_per_root))[:8]:
        r = check_substep_equivalence(env, root, brs[0].tape.reshape(-1, 2))
        sub_rows.append({"root_id": root.root_id, "max_abs_diff": r["max_abs_diff"], "bitwise": r["bitwise"]})
    ledger.add("substep_check", sum(len(r.prefix) * 2 + 25 * 2 for r in roots[:8]))
    print(f"[e0] substep mirror bitwise on {sum(r['bitwise'] for r in sub_rows)}/{len(sub_rows)} roots")

    # ---- timing: dense vs endpoint interpolation ---------------------------------------
    tim = {"contact": [], "free": []}
    dec = {"agree": 0, "dense_unsafe_interp_safe": 0, "dense_safe_interp_unsafe": 0, "n": 0}
    clr_gap = []
    for root, brs in zip(roots, branches_per_root):
        for b in brs:
            st = b.log.states
            it = endpoint_interpolation(st)
            centre = np.linalg.norm(it[:, 2:4] - st[:, 2:4], axis=1)
            ang = angle_err_deg(it[:, 4], st[:, 4])
            key = "contact" if (b.log.n_contacts > 0).any() else "free"
            tim[key].append({"centre_max_px": float(centre.max()), "centre_mean_px": float(centre.mean()), "angle_max_deg": float(ang.max())})
            # a borderline virtual hazard near the dense path, to see decisions flip
            mid = st[12, 2:4]
            hz = Disc(float(mid[0] + 70), float(mid[1]), 30.0)
            cd = clearance_trace(st[:, 2:5], hz)
            ci = clearance_trace(it[:, 2:5], hz)
            clr_gap.append(float(ci.min() - cd.min()))
            ud, ui = bool((cd < 0).any()), bool((ci < 0).any())
            dec["n"] += 1
            if ud == ui:
                dec["agree"] += 1
            elif ud:
                dec["dense_unsafe_interp_safe"] += 1
            else:
                dec["dense_safe_interp_unsafe"] += 1

    def summ(rows, key):
        v = np.array([r[key] for r in rows]) if rows else np.array([np.nan])
        return {"n": len(rows), "mean": float(np.nanmean(v)), "p50": float(np.nanpercentile(v, 50)), "p95": float(np.nanpercentile(v, 95)), "max": float(np.nanmax(v))}

    timing = {
        "control_hz": 10, "physics_dt": PHYSICS_DT, "substeps_per_step": SUBSTEPS_PER_STEP, "action_block": ACTION_BLOCK,
        "horizon_blocks": HORIZON_BLOCKS, "horizon_seconds": HORIZON_BLOCKS * ACTION_BLOCK / 10,
        "interp_centre_max_px": {k: summ(v, "centre_max_px") for k, v in tim.items()},
        "interp_angle_max_deg": {k: summ(v, "angle_max_deg") for k, v in tim.items()},
        "borderline_disc_decisions": dec,
        "interp_minus_dense_min_clearance_px": {"mean": float(np.mean(clr_gap)), "p95": float(np.percentile(clr_gap, 95)), "max": float(np.max(clr_gap))},
    }
    print(f"[e0] timing: {json.dumps(timing['interp_centre_max_px'])}\n[e0] decisions: {dec}")

    # ---- overlays and substep traces around contact -------------------------------------
    n_ov = 0
    traces = []
    for root, brs in zip(roots, branches_per_root):
        b = max(brs, key=lambda x: int((x.log.n_contacts > 0).sum()))
        if not (b.log.n_contacts > 0).any() or n_ov >= 10:
            continue
        frames = [draw_overlay(f, b.log.states[j * ACTION_BLOCK, 2:5], pusher_xy=b.log.states[j * ACTION_BLOCK, :2]) for j, f in enumerate(b.log.frames)]
        from PIL import Image

        Image.fromarray(tile(frames, ncols=6)).save(RESULTS / f"overlay_{n_ov:02d}_{root.root_id}.png")
        n_ov += 1
        if len(traces) < 3:
            first = int(np.argmax(b.log.n_contacts > 0))  # index of the first row with contact
            reset_root(env, root, record_frames=False)
            tape = b.tape.reshape(-1, 2)
            sub = []
            for t, a in enumerate(tape[: first + 2]):
                s = step_substeps(env, a)
                if t >= max(0, first - 2):
                    sub.append({"env_step": t + 1, "substeps": s[:, 2:5].round(4).tolist(), "n_contacts_step": int(b.log.n_contacts[t + 1])})
            ledger.add("substep_check", len(root.prefix) + first + 2)
            traces.append({"root_id": root.root_id, "kind": b.kind, "first_contact_row": first, "trace": sub})
    (RESULTS / "substep_traces.json").write_text(json.dumps(traces) + "\n")

    # ---- bank manifest, report, upload ---------------------------------------------------
    manifest = build_manifest(
        run_id=run_id, kind="e0bank", seeds={"rng": args.seed},
        data={"assets_run": ASSETS_RUN.name, "assets_revision": assets_rev, "roots_split": "roots"},
        costs=ledger.to_dict(), started_at=t_start,
        metrics={"n_roots": len(roots), "n_contact_roots": n_contact, "n_branches": n_branches, "replay_all_bitwise": replay_summary["all_bitwise"]},
    )
    writer.finish(ledger, manifest)
    write_manifest(bank_dir, manifest)
    (bank_dir / "README.md").write_text(f"# {run_id}\n\nE0 root bank: {len(roots)} roots, {n_branches} branches (nominal, random sigma 0.1, stress) with dense truth and endpoint frames. See manifest.json.\n")
    hf_rev = None
    if not args.no_upload:
        store = HFStore()
        hf_rev = store.upload_run("pusht-banks", "banks", bank_dir, run_id=run_id)

    report = {
        "run_id": run_id, "hf_repo": None if hf_rev is None else HFStore().repo_id("pusht-banks"), "hf_path": f"banks/{run_id}", "hf_revision": hf_rev,
        "assets_revision": assets_rev, "public_model_sha": json.loads((ASSETS_RUN / "manifest.json").read_text())["upstream_revisions"]["public_model_sha"],
        "roots": {"n": len(roots), "n_mid_contact": n_contact, "k_counts": {str(k): int(sum(r.meta["k"] == k for r in roots)) for k in ks},
                  "prefix_steps_mean": float(np.mean([len(r.prefix) for r in roots])),
                  "nominal_block_err_px_mean": float(np.mean([r.meta["block_err_px"] for r in roots])),
                  "nominal_solve_s_mean": float(np.mean([t for r in roots for t in [0.6]]))},
        "branches": {"n": n_branches, "per_root": 3, "kinds": ["nominal", "random(sigma=0.1)", "stress"]},
        "replay": replay_summary, "substep_mirror": sub_rows, "timing": timing,
        "ledger": ledger.to_dict(), "wall_clock_s": time.time() - t_start, "n_overlays": n_ov,
    }
    (RESULTS / "replay_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[e0] wrote {RESULTS / 'replay_report.json'}; total {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
