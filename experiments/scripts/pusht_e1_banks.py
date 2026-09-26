#!/usr/bin/env python3
"""E1 step 2: build the development, test and stress banks with frozen hazard layouts.

Roots come from expert start/goal pairs; source episodes are split into familiar and
held-out families (pushtLayouts.split_source_families). Each root gets one familiar and
one held-out hazard layout generated across its nominal route. Every bank stores dense
truth and endpoint frames for every branch (branchBank.BankWriter) and is uploaded to
<ns>/safetydial-pusht-banks:banks/<run_id>.

Banks (provisional sizes, pushT.md E1):
  dev      24 familiar roots x 8 tapes (nominal + random)
  test     --test-roots roots per source family x 16 tapes (nominal + random)
  stress   same roots as test x 16 tapes (T corners/edges + toward the familiar hazard)

    uv run python experiments/scripts/pusht_e1_banks.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers.threads import apply_torch, pin_threads  # noqa: E402

pin_threads()

import numpy as np  # noqa: E402

apply_torch()

from helpers.branchBank import BankWriter, build_root, execute_proposals, expert_pairs, propose  # noqa: E402
from helpers.hfStore import HFStore  # noqa: E402
from helpers.imagination import NominalPlanner  # noqa: E402
from helpers.pushtAssets import H5_PATH, load_model, load_scalers  # noqa: E402
from helpers.pushtLayouts import generate_layout, save_layouts, split_source_families  # noqa: E402
from helpers.pushtReplay import StepLedger, execute_tape, reset_root  # noqa: E402
from helpers.runManifest import build_manifest, make_run_id, write_manifest  # noqa: E402

ASSETS_RUN = REPO_ROOT / "runs" / "pusht-assets-20260926-1"
STUDY = REPO_ROOT / "data" / "study" / "pusht"
KS = [0, 2, 4, 6]


def nominal_route(env, root, ctx) -> np.ndarray:
    """Dense block poses over the prefix and the nominal branch (the route the planner takes)."""
    reset_root(env, root, record_frames=False)
    log = execute_tape(env, np.asarray(root.meta["nominal_plan"]).reshape(-1, 2), record_frames=False)
    return np.concatenate([ctx.prefix_log.states[:, 2:5], log.states[1:, 2:5]], 0)


def build_bank(name, env, planner, rng, pairs_by_family, n_roots_by_family, n_tapes, *, stress: bool, seed_base: int,
               ledger: StepLedger, min_contact_frac: float = 0.3, roots_reuse=None):
    bank_dir = STUDY / name
    if bank_dir.exists():
        shutil.rmtree(bank_dir)
    writer = BankWriter(bank_dir)
    layouts = []
    roots_out = []
    i = 0
    t0 = time.time()
    for family, n_roots in n_roots_by_family.items():
        built, n_contact = 0, 0
        pairs = pairs_by_family[family]
        while built < n_roots:
            if roots_reuse is not None:
                root, ctx, lay_f, lay_h = roots_reuse[family][built]
            else:
                # keep a share of roots in mid-contact: retry k>=2 builds when short
                k = KS[i % len(KS)]
                if built >= 0.6 * n_roots and n_contact < min_contact_frac * built:
                    k = int(rng.choice([2, 4, 6]))
                pair = pairs[i % len(pairs)]
                root, ctx, _ = build_root(env, planner, pair, seed=seed_base + i, k=k, root_id=f"{name}-{family}-r{i:03d}", ledger=ledger)
                root.meta["source_family"] = family
                route = nominal_route(env, root, ctx)
                ledger.add("layout_route", len(root.prefix) + 25)
                lay_f = generate_layout(rng, route, ctx.state[2:5], root.goal_state[2:5], family="familiar", root_id=root.root_id)
                lay_h = generate_layout(rng, route, ctx.state[2:5], root.goal_state[2:5], family="heldout", root_id=root.root_id)
                i += 1
                if lay_f is None or lay_h is None:
                    ledger.add("root_discarded_no_layout", 0, branches=1)
                    continue
            n_contact += int(root.meta["in_contact_last_block"])
            reset_root(env, root, record_frames=False)
            if stress:
                props, rej = propose(rng, root, ctx, n_random=0, n_stress=n_tapes // 2, hazard_centre=lay_f.shape.centre,
                                     n_toward_hazard=n_tapes - n_tapes // 2, include_nominal=False)
            else:
                props, rej = propose(rng, root, ctx, n_random=n_tapes - 1, sigmas=(0.05, 0.1, 0.2))
            ledger.add("proposals_rejected_by_arena", 0, branches=rej)
            br = execute_proposals(env, root, props, ledger=ledger)
            writer.add_root(root)
            writer.add_branches(br)
            layouts += [lay_f, lay_h]
            roots_out.setdefault(family, []).append((root, ctx, lay_f, lay_h)) if isinstance(roots_out, dict) else None
            built += 1
            if built % 8 == 0:
                print(f"[banks:{name}] {family} {built}/{n_roots} roots ({n_contact} in contact) {time.time() - t0:.0f}s")
    save_layouts(bank_dir / "layouts.json", layouts, {"bank": name})
    return writer, bank_dir, layouts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-roots", type=int, default=24)
    ap.add_argument("--dev-tapes", type=int, default=8)
    ap.add_argument("--test-roots", type=int, default=64, help="per source family")
    ap.add_argument("--test-tapes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    t_start = time.time()
    run_id = make_run_id("pusht", "banks", n=args.n)
    device = "cuda"
    model = load_model(device)
    process = load_scalers(ASSETS_RUN / "scalers.npz")
    splits = json.loads((ASSETS_RUN / "splits.json").read_text())
    assets_rev = (ASSETS_RUN / "hf_revision.txt").read_text().split()[2]
    env = __import__("helpers.pushtReplay", fromlist=["make_env"]).make_env()
    planner = NominalPlanner(model, process, device)
    rng = np.random.default_rng(args.seed)
    ledger = StepLedger()
    fam = split_source_families(splits["roles"]["roots"], args.seed)
    pairs_by_family = {f: expert_pairs(H5_PATH, eps, rng, 400) for f, eps in fam.items()}

    banks = {}
    # dev: familiar sources only
    w, d, lays = build_bank("dev", env, planner, rng, pairs_by_family, {"familiar": args.dev_roots}, args.dev_tapes, stress=False, seed_base=args.seed, ledger=ledger)
    banks["dev"] = (w, d, lays)
    # test: both source families; stress reuses the SAME roots and layouts
    global roots_out
    roots_out = {}
    import helpers  # noqa: F401

    test_roots = {}
    w, d, lays = build_bank_with_capture("test", env, planner, rng, pairs_by_family, {"familiar": args.test_roots, "heldout": args.test_roots}, args.test_tapes, seed_base=args.seed + 10_000, ledger=ledger, capture=test_roots)
    banks["test"] = (w, d, lays)
    w, d, lays = build_bank("stress", env, planner, rng, pairs_by_family, {"familiar": args.test_roots, "heldout": args.test_roots}, args.test_tapes, stress=True, seed_base=args.seed + 20_000, ledger=ledger, roots_reuse=test_roots)
    banks["stress"] = (w, d, lays)

    store = None if args.no_upload else HFStore()
    revs = {}
    for name, (writer, bank_dir, lays) in banks.items():
        manifest = build_manifest(run_id=f"{run_id}-{name}", kind="bank", seeds={"rng": args.seed},
                                  data={"assets_run": ASSETS_RUN.name, "assets_revision": assets_rev, "source_families": {k: len(v) for k, v in fam.items()}},
                                  costs=ledger.to_dict(), metrics={"n_layouts": len(lays)}, started_at=t_start)
        writer.finish(ledger, manifest)
        write_manifest(bank_dir, manifest)
        (bank_dir / "README.md").write_text(f"# {run_id}-{name}\n\nPush-T {name} bank with frozen familiar and held-out hazard layouts per root. See manifest.json.\n")
        if store is not None:
            revs[name] = store.upload_run("pusht-banks", "banks", bank_dir, run_id=f"{run_id}-{name}")
    out = REPO_ROOT / "docs" / "mainPlan" / "results" / "e1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "banks.json").write_text(json.dumps({"run_id": run_id, "hf_revisions": revs, "ledger": ledger.to_dict(), "source_families": fam,
                                               "sizes": {"dev": [args.dev_roots, args.dev_tapes], "test": [2 * args.test_roots, args.test_tapes], "stress": [2 * args.test_roots, args.test_tapes]},
                                               "wall_clock_s": time.time() - t_start}, indent=2) + "\n")
    print(f"[banks] done in {time.time() - t_start:.0f}s; ledger {ledger.to_dict()}")
    return 0


def build_bank_with_capture(name, env, planner, rng, pairs_by_family, n_roots_by_family, n_tapes, *, seed_base, ledger, capture):
    """Same as build_bank(stress=False) but records (root, ctx, layouts) per family for reuse."""
    bank_dir = STUDY / name
    if bank_dir.exists():
        shutil.rmtree(bank_dir)
    writer = BankWriter(bank_dir)
    layouts = []
    i = 0
    t0 = time.time()
    for family, n_roots in n_roots_by_family.items():
        built, n_contact = 0, 0
        pairs = pairs_by_family[family]
        capture[family] = []
        while built < n_roots:
            k = KS[i % len(KS)]
            if built >= 0.6 * n_roots and n_contact < 0.3 * built:
                k = int(rng.choice([2, 4, 6]))
            pair = pairs[i % len(pairs)]
            root, ctx, _ = build_root(env, planner, pair, seed=seed_base + i, k=k, root_id=f"{name}-{family}-r{i:03d}", ledger=ledger)
            root.meta["source_family"] = family
            route = nominal_route(env, root, ctx)
            ledger.add("layout_route", len(root.prefix) + 25)
            lay_f = generate_layout(rng, route, ctx.state[2:5], root.goal_state[2:5], family="familiar", root_id=root.root_id)
            lay_h = generate_layout(rng, route, ctx.state[2:5], root.goal_state[2:5], family="heldout", root_id=root.root_id)
            i += 1
            if lay_f is None or lay_h is None:
                ledger.add("root_discarded_no_layout", 0, branches=1)
                continue
            n_contact += int(root.meta["in_contact_last_block"])
            props, rej = propose(rng, root, ctx, n_random=n_tapes - 1, sigmas=(0.05, 0.1, 0.2))
            ledger.add("proposals_rejected_by_arena", 0, branches=rej)
            br = execute_proposals(env, root, props, ledger=ledger)
            writer.add_root(root)
            writer.add_branches(br)
            layouts += [lay_f, lay_h]
            capture[family].append((root, ctx, lay_f, lay_h))
            built += 1
            if built % 8 == 0:
                print(f"[banks:{name}] {family} {built}/{n_roots} roots ({n_contact} in contact) {time.time() - t0:.0f}s")
    save_layouts(bank_dir / "layouts.json", layouts, {"bank": name})
    return writer, bank_dir, layouts


if __name__ == "__main__":
    raise SystemExit(main())
