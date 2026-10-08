#!/usr/bin/env python3
"""E1 step 1: train and freeze the block-pose and pusher probes on expert frames.

Frames come from whole expert episodes in the `probe` split (assets splits.json), read
contiguously; validation is by episode. Linear and small-MLP probes are fitted for
both targets; the choice of capacity is made on validation trajectories here and then
re-checked on development-bank frames in the decomposition script before freezing.

Outputs runs/<run_id>/ with the four probes, stats and a manifest; uploads to
<ns>/safetydial-pusht:probes/<run_id>.

    uv run python experiments/scripts/pusht_e1_probes.py --n-frames 20000
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

from helpers.hfStore import HFStore  # noqa: E402
from helpers.poseProbes import (  # noqa: E402
    ProbeSpec,
    collect_expert_latents,
    fit_probe,
    pose_to_target,
    pusher_to_target,
    save_probe,
    split_by_trajectory,
)
from helpers.pushtAssets import H5_PATH, load_model  # noqa: E402
from helpers.runManifest import validate_run_id, build_manifest, make_run_id, write_manifest  # noqa: E402

ASSETS_RUN = REPO_ROOT / "runs" / "pusht-assets-20260926-1"


def probe_bank_latents(model, splits, n_roots, n_tapes, rng, device, *, bank_dir, assets_run=ASSETS_RUN):
    """Build (once) a root bank from probe-split episodes and encode its endpoint frames."""
    from helpers.branchBank import Bank, BankWriter, build_root, execute_proposals, expert_pairs, propose
    from helpers.imagination import NominalPlanner, encode
    from helpers.pushtAssets import load_scalers
    from helpers.pushtReplay import StepLedger, make_env

    bank_dir = Path(bank_dir)
    if not (bank_dir / "roots.json").is_file():
        env = make_env()
        planner = NominalPlanner(model, load_scalers(assets_run / "scalers.npz"), device)
        episodes = splits["roles"]["probe"]
        pairs = expert_pairs(H5_PATH, episodes, rng, min(len(episodes), n_roots * 4))
        writer = BankWriter(bank_dir)
        ledger = StepLedger()
        for i, pair in enumerate(pairs):
            if len(writer.roots) >= n_roots:
                break
            try:
                root, ctx, _ = build_root(env, planner, pair, seed=777 + i, k=[0, 2, 4, 6][i % 4], root_id=f"probe-r{i:03d}", ledger=ledger)
            except ValueError as exc:
                if "censored or out-of-domain prefix" not in str(exc):
                    raise
                ledger.add("discarded_invalid_root", 0, branches=1)
                continue
            props, rej = propose(rng, root, ctx, n_random=n_tapes - 4, sigmas=(0.05, 0.1, 0.2), n_stress=3)
            ledger.add("proposals_rejected_by_arena", 0, branches=rej)
            writer.add_root(root)
            writer.add_branches(execute_proposals(env, root, props, ledger=ledger))
        writer.finish(ledger, {"bank": "probe_bank"})
    bank = Bank(bank_dir)
    Z, S, E = [], [], []
    for j in bank.valid_training_indices():
        b = bank.branch(j, frames=True)
        Z.append(encode(model, b["frames"], device).cpu().numpy())
        S.append(b["states"][::5])
        episode = int(bank.roots[int(b["root_index"])].meta["episode"])
        if episode not in set(splits["roles"]["probe"]):
            raise ValueError("Probe bank contains a source outside the probe role")
        E.append(np.full(len(b["frames"]), episode, dtype=np.int64))
    if not Z:
        raise ValueError("Probe bank contains no complete observation-valid branch frames")
    return np.concatenate(Z), np.concatenate(S), np.concatenate(E), {"n_roots": len(bank.roots), "n_branches": len(bank), "ledger": bank.ledger}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bank-roots", type=int, default=0, help="also train on branch frames from a probe-split root bank (contact-rich coverage)")
    ap.add_argument("--bank-tapes", type=int, default=8)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--run-id", type=validate_run_id, help="Explicit stable run ID for queued workflows")
    ap.add_argument("--bank-dir", type=Path, help="Fresh optional probe branch bank directory")
    ap.add_argument("--output-dir", type=Path, help="Fresh model directory, defaults to runs/<run_id>")
    ap.add_argument("--results-dir", type=Path, help="Fresh stats directory, defaults to <output-dir>/results")
    ap.add_argument("--assets-run", type=Path, default=ASSETS_RUN)
    args = ap.parse_args()
    t0 = time.time()
    run_id = args.run_id or make_run_id("pusht", "probes", n=args.n)
    run_dir = args.output_dir or REPO_ROOT / "runs" / run_id
    results_dir = args.results_dir or run_dir / "results"
    paths = [run_dir, results_dir]
    if args.bank_roots:
        if args.bank_dir is None:
            ap.error("--bank-dir is required when collecting probe branches")
        paths.append(args.bank_dir)
    for path in paths:
        if path.exists():
            ap.error(f"Use a fresh output path; preserving {path}")
    store = None if args.no_upload else HFStore()
    assets_reference = None if store is None else store.reference_run("pusht", "assets", args.assets_run)
    run_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    device = "cuda"
    model = load_model(device)
    splits = json.loads((args.assets_run / "splits.json").read_text())
    assets_rev = (args.assets_run / "hf_revision.txt").read_text().split()[2]
    rng = np.random.default_rng(args.seed)

    Z, S, E = collect_expert_latents(model, H5_PATH, splits["roles"]["probe"], args.n_frames, rng, device=device)
    print(f"[probes] {len(Z):,} frames from {len(np.unique(E))} episodes in {time.time() - t0:.0f}s")
    bank_info = None
    if args.bank_roots:
        # Contact-rich coverage: roots from PROBE-split episodes, executed tapes, endpoint frames.
        # The probe stays a fixed instrument trained on probe-split data only; the simulator steps
        # are charged to instrument building, not to any acquisition arm.
        Zb, Sb, Eb, bank_info = probe_bank_latents(model, splits, args.bank_roots, args.bank_tapes, rng, device, bank_dir=args.bank_dir, assets_run=args.assets_run)
        Z, S, E = np.concatenate([Z, Zb]), np.concatenate([S, Sb]), np.concatenate([E, Eb])
        print(f"[probes] + {len(Zb):,} branch frames from {bank_info['n_roots']} probe-split roots ({bank_info['ledger']['total_steps']} charged steps)")
        write_manifest(args.bank_dir, build_manifest(run_id=f"{run_id}-bank", kind="probe-bank",
            data={"assets": assets_reference, "source_role": "probe"}, costs=bank_info["ledger"]))
        (args.bank_dir / "README.md").write_text("# Probe branch bank\n\nSource episode identities match the original expert frames; descendants stay in one internal probe split.\n")
        if store is not None:
            store.upload_run("pusht-banks", "banks", args.bank_dir, run_id=f"{run_id}-bank")
            bank_info["hf_reference"] = store.reference_run("pusht-banks", "banks", args.bank_dir)
    tr, va = split_by_trajectory(E, rng, 0.2)
    assert not set(E[tr]) & set(E[va]), "Probe sources overlap across training and validation"
    (run_dir / "source_splits.json").write_text(json.dumps({
        "train_episodes": sorted(int(e) for e in set(E[tr])),
        "validation_episodes": sorted(int(e) for e in set(E[va])),
        "branch_source_policy": "original expert episode ID, shared with real expert frames",
    }, indent=2) + "\n")
    targets = {"block_pose": pose_to_target(S), "pusher": pusher_to_target(S)}
    results = {}
    for target, Y in targets.items():
        for kind in ("linear", "mlp"):
            spec = ProbeSpec(target=target, kind=kind, epochs=args.epochs, seed=args.seed)
            probe, stats = fit_probe(Z[tr], Y[tr], Z[va], Y[va], spec, device=device, verbose=False)
            stats["n_train_episodes"] = int(len(np.unique(E[tr])))
            stats["n_val_episodes"] = int(len(np.unique(E[va])))
            save_probe(probe, stats, run_dir / f"{target}_{kind}.pt")
            v = stats["val"]
            line = f"[probes] {target:10s} {kind:6s} val r2 {v['r2']:.4f} rmse {v['rmse']:.3f}"
            if "centre_p95_px" in v:
                line += f" centre p50 {v['centre_median_px']:.1f} p95 {v['centre_p95_px']:.1f}px"
            if "angle_p95_deg" in v:
                line += f" angle p50 {v['angle_median_deg']:.1f} p95 {v['angle_p95_deg']:.1f}deg"
            print(line)
            results[f"{target}_{kind}"] = {k: v[k] for k in v if k != "r2_per_dim"} | {"r2_per_dim": v["r2_per_dim"]}
    manifest = build_manifest(
        run_id=run_id, kind="probes", seeds={"rng": args.seed},
        data={"assets_run": str(args.assets_run), "assets_revision": assets_rev, "assets": assets_reference, "split": "probe", "n_frames": int(len(Z)),
              "n_episodes": int(len(np.unique(E))), "train_episodes": sorted(int(e) for e in set(E[tr])), "validation_episodes": sorted(int(e) for e in set(E[va])), "val_frac_by_episode": 0.2, "probe_bank": bank_info},
        metrics=results, started_at=t0,
    )
    write_manifest(run_dir, manifest)
    (run_dir / "README.md").write_text(f"# {run_id}\n\nPush-T readout probes (block pose x, y, sin, cos; pusher x, y) from frozen 192-d LeWM latents, trained on whole expert episodes of the probe split with an episode-level validation split. Linear and MLP variants. Not safety supervision. See manifest.json.\n")
    revision = None
    if store is not None:
        revision = store.upload_run("pusht", "probes", run_dir, run_id=run_id)
    report = {"run_id": run_id, "metrics": results, "manifest": manifest,
              "hf_revision": revision, "source_split_integrity": "disjoint original expert episodes"}
    (results_dir / "probes.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    write_manifest(results_dir, {**manifest, "hf_revision": revision})
    print(f"[probes] done in {time.time() - t0:.0f}s -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
