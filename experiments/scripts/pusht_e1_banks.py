#!/usr/bin/env python3
"""E1 step 2: build the development, test and stress banks with frozen hazard layouts.

Roots come from expert start/goal pairs, with source episodes split into disjoint roles
before geometric filtering. Root and goal block positions define checkerboard familiar
and held-out families. Development uses familiar hazards; each final root receives both
hazard families across its nominal route. Every bank stores dense
truth and endpoint frames for every branch (branchBank.BankWriter) and is uploaded to
<ns>/safetydial-pusht-banks:banks/<run_id>.

Banks (provisional sizes, pushT.md E1):
  dev      24 familiar roots x 8 tapes (nominal + random)
  test     --test-roots roots per source family x 16 tapes (nominal + random)
  stress   same roots as test x 16 tapes (T corners/edges + toward the familiar hazard)

    uv run python experiments/scripts/pusht_e1_banks.py --output-dir data/study/pusht/e1-clean-1
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

from helpers.branchBank import BankWriter, build_root, execute_proposals, expert_pairs, propose  # noqa: E402
from helpers.hfStore import HFStore  # noqa: E402
from helpers.imagination import NominalPlanner  # noqa: E402
from helpers.pushtAssets import H5_PATH, load_model, load_scalers  # noqa: E402
from helpers.pushtLayouts import generate_layout, save_layouts  # noqa: E402
from helpers.pushtReplay import StepLedger
from helpers.pushtContactReplay import make_env, execute_tape, reset_root  # noqa: E402
from helpers.runManifest import build_manifest, make_run_id, write_manifest  # noqa: E402
from helpers.splitIntegrity import validate_bank_splits
from helpers.pushtSourceFamilies import SOURCE_FAMILY_PROTOCOL, geometric_source_family  # noqa: E402

ASSETS_RUN = REPO_ROOT / "runs" / "pusht-assets-20260926-1"
STUDY = REPO_ROOT / "data" / "study" / "pusht"
KS = [0, 2, 4, 6]


def nominal_route(env, root, ctx, ledger) -> np.ndarray | None:
    """Dense block poses over the prefix and the nominal branch (the route the planner takes)."""
    reset_root(env, root, record_frames=False)
    log = execute_tape(env, np.asarray(root.meta["nominal_plan"]).reshape(-1, 2), record_frames=False)
    ledger.add("layout_route", len(root.prefix) + log.executed_steps)
    if log.censored:
        ledger.add("discarded_censored_layout_route", 0, branches=1)
        return None
    return np.concatenate([ctx.prefix_log.states[:, 2:5], log.states[1:, 2:5]], 0)


def build_bank(name, env, planner, rng, pairs_by_family, n_roots_by_family, n_tapes, *, stress: bool, seed_base: int,
               ledger: StepLedger, min_contact_frac: float = 0.3, roots_reuse=None, study_dir=STUDY):
    bank_dir = Path(study_dir) / name
    if bank_dir.exists():
        raise FileExistsError(f"Preserving existing bank: {bank_dir}")
    writer = BankWriter(bank_dir)
    layouts = []
    i = 0
    t0 = time.time()
    for family, n_roots in n_roots_by_family.items():
        built, n_contact, pair_cursor = 0, 0, 0
        pairs = pairs_by_family[family]
        while built < n_roots:
            if roots_reuse is not None:
                root, ctx, lay_f, lay_h = roots_reuse[family][built]
            else:
                # keep a share of roots in mid-contact: retry k>=2 builds when short
                k = KS[i % len(KS)]
                if built >= 0.6 * n_roots and n_contact < min_contact_frac * built:
                    k = int(rng.choice([2, 4, 6]))
                if pair_cursor >= len(pairs):
                    raise RuntimeError(f"Exhausted disjoint source trajectories for {name}/{family}")
                pair = pairs[pair_cursor]
                pair_cursor += 1
                try:
                    root, ctx, _ = build_root(env, planner, pair, seed=seed_base + i, k=k, root_id=f"{name}-{family}-r{i:03d}", ledger=ledger)
                except ValueError as exc:
                    if "censored or out-of-domain prefix" not in str(exc):
                        raise
                    ledger.add("discarded_invalid_root", 0, branches=1)
                    i += 1
                    continue
                if geometric_source_family(ctx.state, root.goal_state) != family:
                    ledger.add("discarded_source_geometry", 0, branches=1)
                    i += 1
                    continue
                root.meta["source_family"] = family
                root.meta["source_family_protocol"] = SOURCE_FAMILY_PROTOCOL["version"]
                route = nominal_route(env, root, ctx, ledger)
                if route is None:
                    i += 1
                    continue
                lay_f = generate_layout(rng, route, ctx.state[2:5], root.goal_state[2:5], family="familiar", root_id=root.root_id)
                lay_h = None if name == "dev" else generate_layout(rng, route, ctx.state[2:5], root.goal_state[2:5], family="heldout", root_id=root.root_id)
                i += 1
                if lay_f is None or (name != "dev" and lay_h is None):
                    ledger.add("root_discarded_no_layout", 0, branches=1)
                    continue
            n_contact += int(root.meta.get("in_contact_last_block") is True)
            if stress:
                props, rej = propose(rng, root, ctx, n_random=0, n_stress=n_tapes // 2, hazard_centre=lay_f.shape.centre,
                                     n_toward_hazard=n_tapes - n_tapes // 2, include_nominal=False)
            else:
                props, rej = propose(rng, root, ctx, n_random=n_tapes - 1, sigmas=(0.05, 0.1, 0.2))
            ledger.add("proposals_rejected_by_arena", 0, branches=rej)
            br = execute_proposals(env, root, props, ledger=ledger)
            writer.add_root(root)
            writer.add_branches(br)
            layouts += [lay_f] + ([lay_h] if lay_h is not None else [])
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
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="New bank directory; existing paths are never overwritten")
    ap.add_argument("--results-dir", type=Path,
                    help="New result directory, default <output-dir>/results")
    ap.add_argument("--exclude-bank", type=Path, action="append", default=[],
                    help="Additional previously inspected source trajectories to exclude")
    args = ap.parse_args()
    study_dir = args.output_dir.resolve()
    out = (args.results_dir or study_dir / "results").resolve()
    if study_dir == out:
        ap.error("The results directory must differ from the bank directory")
    for path in (study_dir, out):
        if path.exists():
            ap.error(f"Use a new output path; preserving {path}")
    t_start = time.time()
    run_id = make_run_id("pusht", "banks", n=args.n)
    device = "cuda"
    model = load_model(device)
    process = load_scalers(ASSETS_RUN / "scalers.npz")
    splits = json.loads((ASSETS_RUN / "splits.json").read_text())
    assets_rev = (ASSETS_RUN / "hf_revision.txt").read_text().split()[2]
    env = make_env()
    planner = NominalPlanner(model, process, device)
    rng = np.random.default_rng(args.seed)
    # Prior evaluated source trajectories are no longer untouched test material.
    excluded = set()
    prior_banks = [STUDY / name for name in ("dev", "test", "stress")] + args.exclude_bank
    for old in prior_banks:
        path = old / "roots.json"
        if path.is_file():
            excluded.update(int(r["meta"]["episode"]) for r in json.loads(path.read_text())["roots"])
    eligible = [e for e in splits["roles"]["roots"] if e not in excluded]
    # Randomness splits SOURCE EPISODES into disjoint roles only. It never defines
    # the geometric familiar/held-out label, which is checked at the actual root.
    perm = np.random.default_rng(args.seed + 1).permutation(eligible)
    dev_eps, test_f_eps, test_h_eps = np.array_split(perm, [len(perm) // 3, 2 * len(perm) // 3])
    role_episodes = {"dev": {"familiar": [int(e) for e in dev_eps]},
                     "test": {"familiar": [int(e) for e in test_f_eps],
                              "heldout": [int(e) for e in test_h_eps]}}
    pairs = {
        role: {family: expert_pairs(H5_PATH, eps, rng, len(eps))
               for family, eps in source_families.items()}
        for role, source_families in role_episodes.items()
    }
    ledgers = {name: StepLedger() for name in ("dev", "test", "stress")}
    banks = {}
    banks["dev"] = build_bank(
        "dev", env, planner, rng, pairs["dev"], {"familiar": args.dev_roots},
        args.dev_tapes, stress=False, seed_base=args.seed, ledger=ledgers["dev"],
        study_dir=study_dir,
    )
    test_roots = {}
    counts = {"familiar": args.test_roots, "heldout": args.test_roots}
    banks["test"] = build_bank_with_capture(
        "test", env, planner, rng, pairs["test"], counts, args.test_tapes,
        seed_base=args.seed + 10_000, ledger=ledgers["test"], capture=test_roots,
        study_dir=study_dir,
    )
    banks["stress"] = build_bank(
        "stress", env, planner, rng, pairs["test"], counts, args.test_tapes,
        stress=True, seed_base=args.seed + 20_000, ledger=ledgers["stress"],
        roots_reuse=test_roots, study_dir=study_dir,
    )

    store = None if args.no_upload else HFStore()
    revs = {}
    for name, (writer, bank_dir, lays) in banks.items():
        manifest = build_manifest(run_id=f"{run_id}-{name}", kind="bank", seeds={"rng": args.seed},
                                  data={"contact_kind": "pusher_block", "contact_counter": "post_solve_contact_points_summed_over_physics_substeps_v1", "assets_run": ASSETS_RUN.name, "assets_revision": assets_rev, "source_roles": role_episodes, "source_family_protocol": SOURCE_FAMILY_PROTOCOL, "excluded_previously_evaluated_episodes": sorted(excluded)},
                                  costs=ledgers[name].to_dict(), metrics={"n_layouts": len(lays)}, started_at=t_start)
        writer.finish(ledgers[name], manifest)
        write_manifest(bank_dir, manifest)
        (bank_dir / "README.md").write_text(f"# {run_id}-{name}\n\nPush-T {name} bank with frozen layouts; development uses familiar hazards only. See manifest.json.\n")
    split_report = validate_bank_splits({name: item[1] for name, item in banks.items()})
    for name, (_, bank_dir, _) in banks.items():
        if store is not None:
            revs[name] = store.upload_run("pusht-banks", "banks", bank_dir, run_id=f"{run_id}-{name}")
    out.mkdir(parents=True, exist_ok=False)
    (out / "banks.json").write_text(json.dumps({
        "run_id": run_id, "hf_revisions": revs,
        "ledgers": {name: ledger.to_dict() for name, ledger in ledgers.items()},
        "source_roles": role_episodes, "split_integrity": split_report,
        "excluded_previously_evaluated_episodes": sorted(excluded),
        "source_family_protocol": SOURCE_FAMILY_PROTOCOL,
        "transfer_interpretation": "Disjoint geometric root/goal cells, held out from adaptation and development",
        "wall_clock_s": time.time() - t_start,
    }, indent=2) + "\n")
    print(f"[banks] complete with disjoint source roles; report {out / 'banks.json'}")
    return 0


def build_bank_with_capture(name, env, planner, rng, pairs_by_family, n_roots_by_family, n_tapes, *, seed_base, ledger, capture, study_dir=STUDY):
    """Same as build_bank(stress=False) but records (root, ctx, layouts) per family for reuse."""
    bank_dir = Path(study_dir) / name
    if bank_dir.exists():
        raise FileExistsError(f"Preserving existing bank: {bank_dir}")
    writer = BankWriter(bank_dir)
    layouts = []
    i = 0
    t0 = time.time()
    for family, n_roots in n_roots_by_family.items():
        built, n_contact, pair_cursor = 0, 0, 0
        pairs = pairs_by_family[family]
        capture[family] = []
        while built < n_roots:
            k = KS[i % len(KS)]
            if built >= 0.6 * n_roots and n_contact < 0.3 * built:
                k = int(rng.choice([2, 4, 6]))
            if pair_cursor >= len(pairs):
                raise RuntimeError(f"Exhausted disjoint source trajectories for {name}/{family}")
            pair = pairs[pair_cursor]
            pair_cursor += 1
            try:
                root, ctx, _ = build_root(env, planner, pair, seed=seed_base + i, k=k, root_id=f"{name}-{family}-r{i:03d}", ledger=ledger)
            except ValueError as exc:
                if "censored or out-of-domain prefix" not in str(exc):
                    raise
                ledger.add("discarded_invalid_root", 0, branches=1)
                i += 1
                continue
            if geometric_source_family(ctx.state, root.goal_state) != family:
                ledger.add("discarded_source_geometry", 0, branches=1)
                i += 1
                continue
            root.meta["source_family"] = family
            root.meta["source_family_protocol"] = SOURCE_FAMILY_PROTOCOL["version"]
            route = nominal_route(env, root, ctx, ledger)
            if route is None:
                i += 1
                continue
            lay_f = generate_layout(rng, route, ctx.state[2:5], root.goal_state[2:5], family="familiar", root_id=root.root_id)
            lay_h = generate_layout(rng, route, ctx.state[2:5], root.goal_state[2:5], family="heldout", root_id=root.root_id)
            i += 1
            if lay_f is None or lay_h is None:
                ledger.add("root_discarded_no_layout", 0, branches=1)
                continue
            n_contact += int(root.meta.get("in_contact_last_block") is True)
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
