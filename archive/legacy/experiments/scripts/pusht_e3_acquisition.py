#!/usr/bin/env python3
"""E3 (and the E4 transfer grid): which experience repairs more at equal charged steps?

Arms choose from one common candidate pool per acquisition root (acquisition.py):
random, predicted boundary, learned optimistic-error risk (only with --learned), and the
retrospective oracle (reference only; its simulator cost is not comparable). All arms
share 128 common seed branches, then rounds of 64, 64, 128 and 256 branches reach the
budgets 64, 128, 256 and 512 additional branches. Every budget restarts adaptation from
the released weights with the recipe fixed in E2. Evaluation at every budget on the
dev (margin choice), test and stress banks, reported by hazard-layout family and by
start/goal source family (E4).

    uv run python experiments/scripts/pusht_e3_acquisition.py --seeds 0            # priority item 4
    uv run python experiments/scripts/pusht_e3_acquisition.py --seeds 0 1 2 --learned --oracle
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers.threads import apply_torch, pin_threads  # noqa: E402

pin_threads()

import numpy as np  # noqa: E402
import torch  # noqa: E402

apply_torch()

from helpers.acquisition import CandidatePool, RiskModel, feature_matrix, root_schedule, score_pool, select  # noqa: E402
from helpers.branchBank import Bank, BankWriter, Branch, build_root, expert_pairs  # noqa: E402
from helpers.decomposition import RootLatentCache, evaluate_model_on_bank, layouts_by_root, ordinary_motion  # noqa: E402
from helpers.dialMetrics import auc_dial, clearance_error_stats, cluster_bootstrap, fsa, margin_for_acceptance  # noqa: E402
from helpers.hfStore import HFStore  # noqa: E402
from helpers.imagination import Imaginer, NominalPlanner, blocks_to_model  # noqa: E402
from helpers.poseProbes import load_probe  # noqa: E402
from helpers.predictorAdapt import AdaptConfig, ClipSet, adapt, predictor_side_state  # noqa: E402
from helpers.pushtAssets import H5_PATH, load_model, load_scalers  # noqa: E402
from helpers.pushtGeometry import clearance_trace  # noqa: E402
from helpers.pushtLayouts import generate_layout, load_layouts, save_layouts  # noqa: E402
from helpers.pushtReplay import StepLedger  # noqa: E402
from helpers.pushtContactReplay import execute_tape, make_env, reset_root  # noqa: E402
from helpers.runManifest import build_manifest, file_sha256, make_run_id, validate_run_id, write_manifest
from helpers.acquisitionSafety import COLLECTION_VERSION, MeteredEnv, copy_ledger, enforce_gate, require_new_paths, validate_acquisition_cache
from helpers.splitIntegrity import bank_identity, inspect_bank_splits
from helpers.pushtAcquisitionReporting import (
    candidate_pool_identity, result_provenance, validate_reference, validate_reference_pool,
)
from helpers.studyGates import acquisition_repeatability_gate
from helpers.pushtSourceFamilies import SOURCE_FAMILY_PROTOCOL, geometric_source_family, validate_geometric_bank
from scripts.pusht_e2_repair import retention_metrics  # noqa: E402

ASSETS_RUN = REPO_ROOT / "runs" / "pusht-assets-20260926-1"
PROBES_RUN = REPO_ROOT / "runs" / "pusht-probes-20260926-1"
STUDY = REPO_ROOT / "data" / "study" / "pusht"
RESULTS = REPO_ROOT / "docs" / "mainPlan" / "results" / "e3"
KS = [0, 2, 4, 6]
ROUNDS = [64, 64, 128, 256]
SEED_BRANCHES = 128


def build_acq_roots(n_roots, seed, splits, model, process, device, bank_dir):
    """Build charged roots or reuse only a complete cache with matching provenance."""
    if bank_dir.exists():
        validate_acquisition_cache(bank_dir, n_roots=n_roots, seed=seed)
        validate_geometric_bank(Bank(bank_dir), expected_family="familiar")
        return Bank(bank_dir), load_layouts(bank_dir / "layouts.json")[0]
    ledger = StepLedger()
    env = MeteredEnv(make_env(), ledger, "root_generation_and_layout")
    planner = NominalPlanner(model, process, device)
    rng = np.random.default_rng(seed)
    episodes = splits["roles"]["reserve"][3000:8000]
    # Freeze the full pre-existing acquisition reservation before any simulation.
    # A 4-8% geometric acceptance rate cannot reliably fill 96 roots from 768 trials.
    n_candidates = len(episodes)
    if n_candidates < n_roots:
        raise ValueError("Insufficient disjoint source trajectories for acquisition roots")
    pairs = expert_pairs(H5_PATH, episodes, rng, n_candidates)
    if len(set(episodes)) != len(episodes):
        raise ValueError("Acquisition reservation must contain unique source episodes")
    writer = BankWriter(bank_dir, with_frames=False)
    plan = {"protocol": "full-reserved-sources-single-candidate-v1", "seed": seed,
            "requested_roots": n_roots, "max_accepted_roots_per_source": 1,
            "source_episodes": episodes, "source_family_protocol": SOURCE_FAMILY_PROTOCOL,
            "expert_h5_sha256": file_sha256(H5_PATH),
            "candidates": [dict(pair, candidate_index=i, root_seed=seed + i,
                                k=KS[i % len(KS)], root_id=f"acq-r{i:03d}")
                           for i, pair in enumerate(pairs)]}
    plan_path = bank_dir / "sampling-plan.json"
    plan_path.write_text(json.dumps(plan, indent=1,
                                   default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v) + "\n")
    plan_sha = file_sha256(plan_path)
    layouts, contexts, built = [], [], 0
    t0 = time.time()

    def snapshot(status, attempts):
        writer.h5.flush()
        blob = {"status": status, "roots": writer.roots, "ledger": ledger.to_dict(),
                "manifest": {"bank": "acq_roots", "seed": seed, "n_roots": n_roots,
                             "collection_version": COLLECTION_VERSION, "status": status,
                             "sampling_plan_sha256": plan_sha}}
        for name, value in (("roots.json", blob), ("progress.json", {
            "status": status, "attempts_completed": attempts, "accepted_roots": built,
            "scheduled_candidates": n_candidates, "requested_roots": n_roots,
            "ledger": ledger.to_dict(), "elapsed_seconds": time.time() - t0,
            "sampling_plan_sha256": plan_sha,
        })):
            path = bank_dir / name
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(value, indent=1) + "\n")
            temporary.replace(path)
        save_layouts(bank_dir / "layouts.json", layouts, {"bank": "acq_roots", "status": status})

    attempts = 0
    snapshot("running", attempts)
    try:
        for i, pair in enumerate(pairs):
            steps_before, roots_before = ledger.total, built
            try:
                try:
                    root, ctx, _ = build_root(env, planner, pair, seed=seed + i, k=KS[i % 4], root_id=f"acq-r{i:03d}")
                except ValueError as exc:
                    if "censored or out-of-domain prefix" not in str(exc):
                        raise
                    ledger.add("discarded_invalid_root", 0, branches=1)
                    continue
                if geometric_source_family(ctx.state, root.goal_state) != "familiar":
                    ledger.add("discarded_source_geometry", 0, branches=1)
                    continue
                root.meta["source_family"] = "familiar"
                root.meta["source_family_protocol"] = SOURCE_FAMILY_PROTOCOL["version"]
                reset_root(env, root, record_frames=False)
                log = execute_tape(env, np.asarray(root.meta["nominal_plan"]).reshape(-1, 2), record_frames=False)
                if log.censored:
                    ledger.add("discarded_censored_layout_route", 0, branches=1)
                    continue
                route = np.concatenate([ctx.prefix_log.states[:, 2:5], log.states[1:, 2:5]], 0)
                lay = generate_layout(rng, route, ctx.state[2:5], root.goal_state[2:5], family="familiar", root_id=root.root_id)
                if lay is None:
                    ledger.add("discarded_root", 0, branches=1)
                    continue
                writer.add_root(root)
                layouts.append(lay)
                contexts.append(ctx)
                built += 1
                if built % 16 == 0:
                    print(f"[e3] acq roots {built}/{n_roots} {time.time() - t0:.0f}s")
                if built == n_roots:
                    break
            finally:
                attempts += 1
                record = {"candidate_index": i, "episode": pair["episode"],
                          "root_seed": seed + i, "k": KS[i % len(KS)],
                          "accepted": built > roots_before,
                          "actual_steps": ledger.total - steps_before,
                          "cumulative_steps": ledger.total}
                with (bank_dir / "attempts.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                snapshot("running", attempts)
        if built != n_roots:
            raise RuntimeError(f"Only {built}/{n_roots} roots obtained; preserving incomplete cache")
        save_layouts(bank_dir / "layouts.json", layouts, {"bank": "acq_roots"})
        np.savez_compressed(bank_dir / "contexts.npz", frames=np.stack([c.frames for c in contexts]),
                            history_actions=np.stack([c.history_actions for c in contexts]),
                            root_ids=np.asarray([r["root_id"] for r in writer.roots]))
        writer.finish(ledger, {"bank": "acq_roots", "seed": seed, "n_roots": n_roots,
                               "collection_version": COLLECTION_VERSION, "status": "complete",
                               "sampling_plan_sha256": plan_sha})
        progress = json.loads((bank_dir / "progress.json").read_text())
        progress["status"] = "complete"
        (bank_dir / "progress.json").write_text(json.dumps(progress, indent=1) + "\n")
    except BaseException:
        if writer.h5.id.valid:
            snapshot("incomplete", attempts)
        if writer.h5.id.valid:
            writer.h5.close()
        (bank_dir / "incomplete_ledger.json").write_text(json.dumps(ledger.to_dict(), indent=1) + "\n")
        raise
    finally:
        env.close()
    validate_acquisition_cache(bank_dir, n_roots=n_roots, seed=seed)
    return Bank(bank_dir), layouts


def acquisition_context_cache(imaginer, bank):
    """Encode already paid root observations once. No simulator execution for scoring."""
    cache = RootLatentCache(imaginer, bank)
    with np.load(bank.dir / "contexts.npz", allow_pickle=False) as z:
        for ri in range(len(bank.roots)):
            frames = z["frames"][ri]
            cache.cache[ri] = (imaginer.encode(frames), z["history_actions"][ri].copy(), frames)
    return cache


def execute_candidates(env, bank, cands, ledger: StepLedger, category: str, *, on_branch=None) -> list[Branch]:
    """Meter actual attempted steps; do not infer the cost from successful branch count."""
    metered = MeteredEnv(env, ledger, category)
    out = []
    for c in cands:
        root = bank.roots[c.root_index]
        reset_root(metered, root, record_frames=False)
        log = execute_tape(metered, c.proposal.tape.reshape(-1, 2), record_frames=True)
        ledger.add(category, 0, branches=1)
        branch = Branch(root.root_id, c.proposal.tape, c.proposal.kind,
                        dict(c.proposal.params, key=list(c.key)), log)
        # Preserve the observed trace before any later query or budget-padding failure.
        out.append(branch)
        if on_branch is not None:
            on_branch(branch)
        # A terminated tape leaves part of its scheduled horizon unavailable. Spend
        # that allowance on explicitly discarded fresh-reset neutral queries. This
        # preserves equal ACTUAL paid budgets without using an unseen suffix as data.
        unused = len(log.actions) - log.executed_steps
        if unused:
            padding = MeteredEnv(env, ledger, category + "_discarded_padding")
            for _ in range(unused):
                padding.reset(seed=int(root.seed), options={"state": root.start_state, "goal_state": root.goal_state})
                padding.step(np.zeros(2))
            ledger.add(category + "_discarded_padding", 0, branches=unused)
        c.executed = True
    return out


class DataStore:
    """Executed branches and cached training targets; adding data never replays roots."""

    def __init__(self, path: Path, imaginer, contexts):
        require_new_paths([path])
        self.writer = BankWriter(path)
        self.path, self.imaginer, self.contexts = path, imaginer, contexts
        self.clips: list[ClipSet] = []
        self.true_c: dict[tuple, float] = {}

    def add(self, bank, branches, lay):
        roots = {r.root_id: (ri, r) for ri, r in enumerate(bank.roots)}
        latents, actions = [], []
        for b in branches:
            ri, root = roots[b.root_id]
            self.writer.add_root(root)
            hz = lay[b.root_id]["familiar"]
            if not b.log.censored:
                self.true_c[tuple(b.params["key"])] = float(clearance_trace(b.log.states[:, 2:5], hz).min())
            if not b.log.valid_for_training:
                continue
            z_hist, hist_blocks, _ = self.contexts.get(ri)
            z_branch = self.imaginer.encode(b.log.frames).cpu().numpy()
            latents.append(np.concatenate([z_hist[:-1].cpu().numpy(), z_branch]))
            blocks = np.concatenate([hist_blocks, b.tape.astype(np.float64)])
            actions.append(blocks_to_model(self.imaginer.process, blocks).numpy())
        self.writer.add_branches(branches)
        if latents:
            self.clips.append(ClipSet(np.asarray(latents, np.float32), np.asarray(actions, np.float32),
                                      {"source": "charged_acquisition", "n": len(latents)}))

    def clipset(self) -> ClipSet:
        if not self.clips:
            raise ValueError("No fully observed in-domain acquisition clips are available for training")
        return ClipSet.concat(self.clips)


def evaluate_all(model, process, probe, banks, caches, device, target_ar):
    im = Imaginer(model, process, device)
    rows = {}
    for n, (b, lays) in banks.items():
        caches[n].imaginer = im
        rows[n] = evaluate_model_on_bank(n, b, lays, im, probe, cache=caches[n])
        exits = {}
        for row in rows[n]:
            branch = row["branch"]
            if branch not in exits:
                xy = b.h5["states"][branch, :, :2]
                exits[branch] = bool((xy < 0).any() or (xy > 512).any() or not np.isfinite(xy).all())
            row["arena_exit"] = exits[branch]
            row["unsafe"] = row.get("unsafe_composite", row["cmin_dense"] <= 0 or row["arena_exit"])
            row.setdefault("censored", False)
    m = margin_for_acceptance(np.array([r["cmin_imagined"] for r in rows["dev"]]), target_ar)
    out = {"margin_matched_dev": float(m)}
    for n, rr in rows.items():
        u = np.array([r["unsafe"] for r in rr])
        c = np.array([r["cmin_imagined"] for r in rr])
        censored = np.asarray([r["censored"] for r in rr], bool)
        ct = np.asarray([r["cmin_dense"] for r in rr])
        root = np.array([r["root"] for r in rr])
        out[n] = {"at_matched": fsa(c, u, m, censored=censored), "at_m0": fsa(c, u, 0.0, censored=censored),
                  "auc_dial": auc_dial(c, u, censored=censored),
                  "clearance_error": clearance_error_stats(c[~censored], ct[~censored]) if (~censored).any() else {"n": 0},
                  "ordinary_motion": ordinary_motion(rr), "n_censored_rows": int(censored.sum()),
                  "fsa_matched_ci": cluster_bootstrap(lambda c, u, censored: fsa(c, u, m, censored=censored)["fsa"],
                                                       root, n_boot=300, c=c, u=u, censored=censored)}
        out[n]["n_arena_exit_rows"] = int(sum(r["arena_exit"] for r in rr))
        out[n]["hazard_only_at_matched"] = fsa(c, ct <= 0, m, censored=censored)
        # Descriptive grid only: episode subsets do not establish shifted starts/goals.
        src = {r.root_id: r.meta.get("source_family", "familiar") for r in banks[n][0].roots}
        grid = {}
        for lf in ("familiar", "heldout"):
            for sf in ("familiar", "heldout"):
                sub = [r for r in rr if r["layout"] == lf and src[banks[n][0].roots[r["root"]].root_id] == sf]
                if sub:
                    grid[f"layout={lf}/source={sf}"] = fsa(np.array([r["cmin_imagined"] for r in sub]), np.array([r["unsafe"] for r in sub]), m, censored=np.asarray([r["censored"] for r in sub]))
        out[n]["transfer_grid"] = grid
    return out, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--arms", nargs="+", choices=["random", "boundary", "learned", "oracle"], default=["random", "boundary"])
    ap.add_argument("--learned", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--diagnostic", action="store_true", help="One seed and one 64-branch random/boundary check; never a main-study claim")
    ap.add_argument("--acq-roots", type=int, default=96)
    ap.add_argument("--pool-per-root", type=int, default=40)
    ap.add_argument("--band", type=float, default=10.0)
    ap.add_argument("--recipe", default=None, help="JSON AdaptConfig; default: recorded E2 recipe")
    ap.add_argument("--run-id", type=validate_run_id)
    ap.add_argument("--data-dir", type=Path, help="Fresh acquisition data directory")
    ap.add_argument("--checkpoint-dir", type=Path, help="Fresh checkpoint parent directory")
    ap.add_argument("--budgets", nargs="+", type=int, default=[64, 128, 256, 512],
                    help="Predeclared full-study additional branch totals")
    ap.add_argument("--feasibility-report", type=Path, help="Exact passing development witness recorded by E2")
    ap.add_argument("--e2-report", type=Path, default=REPO_ROOT / "docs/mainPlan/results/e2/repair.json")
    ap.add_argument("--banks-dir", type=Path)
    ap.add_argument("--assets-run", type=Path)
    ap.add_argument("--probes-run", type=Path)
    ap.add_argument("--clips-dir", type=Path)
    ap.add_argument("--reference-report", type=Path, help="Completed normal study binding an oracle-only reference")
    ap.add_argument("--acq-roots-dir", type=Path, help="Optional complete charged root cache")
    ap.add_argument("--output-dir", type=Path, help="New result directory; existing paths are refused")
    ap.add_argument("--preflight-only", action="store_true", help="Validate gates, splits and paths without loading a model or spending simulator steps")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    t_start = time.time()
    arms = list(args.arms) + (["learned"] if args.learned else []) + (["oracle"] if args.oracle else [])
    reference = None
    if args.reference_report:
        if arms != ["oracle"] or not args.acq_roots_dir or not args.acq_roots_dir.exists():
            raise ValueError("A separate oracle reference requires only the oracle arm and an existing root cache")
        reference = json.loads(args.reference_report.read_text())
    e2 = json.loads(args.e2_report.read_text())
    gate = enforce_gate(e2, diagnostic=args.diagnostic, arms=arms, seeds=args.seeds)
    args.banks_dir = args.banks_dir or Path(e2.get("banks_dir", STUDY))
    args.assets_run = args.assets_run or Path(e2.get("assets_run", ASSETS_RUN))
    args.probes_run = args.probes_run or Path(e2.get("probes_run", PROBES_RUN))
    args.clips_dir = args.clips_dir or Path(e2.get("clips_dir", STUDY / "clips"))
    for name in ("assets_run", "probes_run", "clips_dir", "banks_dir"):
        if not args.diagnostic and (name not in e2 or Path(e2[name]).resolve() != getattr(args, name).resolve()):
            raise ValueError(f"Normal E3 requires the exact {name} recorded by the chosen E2 run")
    if args.acq_roots <= 0 or args.pool_per_root < 4 or args.band <= 0:
        raise ValueError("Positive root count/band and at least 4 proposals per root are required")
    if args.acq_roots_dir and args.acq_roots_dir.exists():
        validate_acquisition_cache(args.acq_roots_dir, n_roots=args.acq_roots, seed=20261003)
    if args.budgets != [64, 128, 256, 512]:
        raise ValueError("The qualified protocol fixes additional budgets at 64, 128, 256 and 512 branches")
    rounds = [64] if args.diagnostic else ROUNDS
    feasibility = None
    if not args.diagnostic:
        from helpers.pushtFeasibility import require_feasibility_report

        recorded = e2.get("development_feasibility", {})
        if args.feasibility_report is None:
            if not recorded.get("report"):
                raise ValueError("Qualified acquisition requires E2's recorded development feasibility witness")
            args.feasibility_report = Path(recorded["report"])
        if (Path(recorded.get("report", "")).resolve() != args.feasibility_report.resolve()
                or recorded.get("sha256") != file_sha256(args.feasibility_report)):
            raise ValueError("Acquisition feasibility witness differs from the passing E2 report")
        feasibility = require_feasibility_report(args.feasibility_report, args.banks_dir / "dev")
    cfg = AdaptConfig(**(json.loads(args.recipe) if args.recipe else e2["chosen_recipe"]))
    if cfg.steps <= 0 or cfg.batch_size <= 0 or not 0 <= cfg.replay_frac <= 1:
        raise ValueError("Invalid adaptation recipe")
    bank_paths = {n: args.banks_dir / n for n in ("dev", "test", "stress")}
    split_audit = inspect_bank_splits(bank_paths)
    if not split_audit["passes"] and not args.diagnostic:
        raise ValueError(f"Evaluation roles overlap: {split_audit}")
    dev_layouts = load_layouts(bank_paths["dev"] / "layouts.json")[0]
    transfer_valid = split_audit["passes"] and all(layout.family == "familiar" for layout in dev_layouts)
    if not transfer_valid and not args.diagnostic:
        raise ValueError("Held-out layouts appear in development or source roles overlap; rebuild banks")
    run_id = args.run_id or make_run_id("pusht", "acq-diagnostic" if args.diagnostic else "acq", n=args.n)
    result_dir = args.output_dir or RESULTS / run_id
    data_dir = args.data_dir or STUDY / run_id
    checkpoint_dir = args.checkpoint_dir or REPO_ROOT / "runs" / run_id
    require_new_paths([result_dir, data_dir, checkpoint_dir])
    if args.preflight_only:
        print(json.dumps({"gate": gate, "split_errors": split_audit["errors"], "rounds": rounds,
                          "result_dir": str(result_dir), "data_dir": str(data_dir),
                          "checkpoint_dir": str(checkpoint_dir), "model_loaded": False,
                          "simulator_steps": 0}, indent=1))
        return 0
    result_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True)
    report = {"run_id": run_id, "gate": gate, "split_audit": split_audit,
              "transfer_interpretation": "descriptive_episode_subsets_only", "recipe": cfg.to_dict(),
              "arms": {}, "rounds": rounds, "seed_branches": SEED_BRANCHES,
              "acquisition_seeds": args.seeds, "candidate_pool_identities": {},
              "evaluation_bank_identity": {name: bank_identity(path) for name, path in bank_paths.items()},
              "charged_step_definition": "root generation plus cached history collection plus common seed plus acquired branches; evaluation recorded separately",
              "e2_report": str(args.e2_report.resolve()), "e2_report_sha256": file_sha256(args.e2_report),
              "development_feasibility": feasibility, "checkpoint_dir": str(checkpoint_dir.resolve()),
              "data_dir": str(data_dir.resolve()), "budgets_added": [sum(rounds[:i + 1]) for i in range(len(rounds))],
              "assets_run": str(args.assets_run), "probes_run": str(args.probes_run),
              "clips_dir": str(args.clips_dir), "banks_dir": str(args.banks_dir),
              "probe_name": e2.get("probe_name", "block_pose_mlp"),
              "candidate_pool": "nominal plan plus bounded Gaussian perturbations; stress remains a separate evaluation bank",
              "censored_budget_policy": "Unused scheduled horizon steps are paid fresh-reset neutral queries, discarded from training"}
    if reference is not None:
        report["oracle_reference"] = {"report": str(args.reference_report.resolve()),
            "sha256": file_sha256(args.reference_report), "run_id": reference.get("run_id"),
            "interpretation": "Retrospective data-usefulness reference; whole-pool cost is not comparable to prospective acquisition"}
    input_paths = [args.assets_run / "scalers.npz", args.assets_run / "splits.json",
                   args.probes_run / (report["probe_name"] + ".pt"),
                   args.clips_dir / "replay.npz", args.clips_dir / "retention.npz"]
    input_paths += [REPO_ROOT / "experiments" / relative for relative in (
        "scripts/pusht_e3_acquisition.py", "helpers/acquisition.py", "helpers/imagination.py",
        "helpers/decomposition.py", "helpers/predictorAdapt.py", "helpers/poseProbes.py",
        "helpers/pushtAcquisitionReporting.py", "helpers/dialMetrics.py",
        "scripts/pusht_e2_repair.py")]
    report["input_sha256"] = {str(path.resolve()): file_sha256(path) for path in input_paths}
    active_store, active_ledger, active_oracle = None, None, None
    try:
        device = "cuda"
        model = load_model(device)
        base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model_digest = hashlib.sha256()
        for name, value in sorted(base_sd.items()):
            model_digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
            model_digest.update(value.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
        report["input_sha256"]["loaded_base_model_state_dict"] = model_digest.hexdigest()
        process = load_scalers(args.assets_run / "scalers.npz")
        splits = json.loads((args.assets_run / "splits.json").read_text())
        probe, _ = load_probe(args.probes_run / (report["probe_name"] + ".pt"), device)
        imaginer = Imaginer(model, process, device)
        replay = ClipSet.load(args.clips_dir / "replay.npz")
        retention = ClipSet.load(args.clips_dir / "retention.npz")
        banks = {n: (Bank(path), load_layouts(path / "layouts.json")[0]) for n, path in bank_paths.items()}
        if not args.diagnostic:
            for name, (bank, _) in banks.items():
                validate_geometric_bank(bank, expected_family="familiar" if name == "dev" else None)
            report["transfer_interpretation"] = "Disjoint geometric root/goal cells, held out from adaptation and development"
            report["source_family_protocol"] = SOURCE_FAMILY_PROTOCOL
        evaluation_ledger = StepLedger()
        caches = {n: RootLatentCache(imaginer, b) for n, (b, _) in banks.items()}
        for n, cache in caches.items():
            cache.env = MeteredEnv(cache.env, evaluation_ledger, n + "_history")
        acq_dir = args.acq_roots_dir or data_dir / "acq_roots"
        acq_bank, acq_layouts = build_acq_roots(args.acq_roots, 20261003, splits, model, process, device, acq_dir)
        report["acquisition_root_identity"] = dict(bank_identity(acq_dir), **{
            "contexts.npz": file_sha256(acq_dir / "contexts.npz")})
        if reference is not None:
            validate_reference(reference, report)
        acquisition_splits = inspect_bank_splits(bank_paths | {"acquisition": acq_dir})
        # A diagnostic may reuse overlapping development/test rows, never acquisition rows.
        acq_episodes = {r.meta["episode"] for r in acq_bank.roots}
        if any(acq_episodes & {r.meta["episode"] for r in b.roots} for b, _ in banks.values()):
            raise ValueError("Acquisition and evaluation source trajectories overlap")
        report["acquisition_split_audit"] = acquisition_splits
        lay = layouts_by_root(acq_layouts)
        acq_contexts = acquisition_context_cache(imaginer, acq_bank)
        env = make_env()
        dev_rows0 = evaluate_model_on_bank("dev", banks["dev"][0], banks["dev"][1], imaginer, probe, cache=caches["dev"])
        target_ar = float(np.mean([r["cmin_imagined"] >= 0 for r in dev_rows0]))
        base_eval, base_rows = evaluate_all(model, process, probe, banks, caches, device, target_ar)
        baseline_rows_path = result_dir / "no-update-rows.npz"
        np.savez_compressed(baseline_rows_path, rows=json.dumps(base_rows))
        report["no_update_rows"] = {"file": baseline_rows_path.name,
            "sha256": hashlib.sha256(baseline_rows_path.read_bytes()).hexdigest()}
        report.update(target_acceptance_rate_dev=target_ar, no_update=base_eval,
                      root_collection_ledger=acq_bank.ledger, evaluation_ledger=evaluation_ledger.to_dict())
        report["no_update_retention"] = retention_metrics(model, Imaginer, process, probe, retention, device)
        report["limitations"] = ["Final selected weights need a separate 20-case goal-retention report before E5"]
        if args.diagnostic:
            report["limitations"].append("Legacy diagnostic episode subsets do not establish geometric transfer")
        comparison_rows = {}
        hf = None if args.no_upload else HFStore()
        report["upstream_revisions"] = {} if hf is None else {
            "assets": hf.reference_run("pusht", "assets", args.assets_run),
            "probes": hf.reference_run("pusht", "probes", args.probes_run),
            "evaluation_banks": {name: hf.reference_run("pusht-banks", "banks", path)
                                 for name, path in bank_paths.items()},
        }
        if hf and not args.acq_roots_dir:
            write_manifest(acq_dir, build_manifest(run_id=f"{run_id}-roots", kind="bank", seeds={"root_seed": 20261003},
                           data={"n_roots": len(acq_bank.roots)}, costs=acq_bank.ledger))
            report["root_bank_hf_revision"] = hf.upload_run("pusht-banks", "banks", acq_dir, run_id=f"{run_id}-roots")
        if hf:
            report["upstream_revisions"]["acquisition_roots"] = hf.reference_run("pusht-banks", "banks", acq_dir)
        for seed in args.seeds:
            pool = CandidatePool.build(np.random.default_rng(1000 + seed), acq_bank, args.pool_per_root,
                                       n_stress=0, n_toward=0, layouts_by_root=lay)
            schedule = root_schedule(pool, SEED_BRANCHES + sum(rounds), np.random.default_rng(2000 + seed))
            seed_schedule = schedule[:SEED_BRANCHES]
            seed_cands = select("random", pool, SEED_BRANCHES, np.random.default_rng(3000 + seed), schedule=seed_schedule)
            pool_identity = candidate_pool_identity(pool, schedule, seed_cands)
            if reference is not None:
                validate_reference_pool(reference, seed, pool_identity)
            report["candidate_pool_identities"][str(seed)] = pool_identity
            (result_dir / "acquisition.json").write_text(json.dumps(report, indent=1) + "\n")
            common_ledger, oracle_ledger = StepLedger(), StepLedger()
            active_ledger = common_ledger
            if "oracle" in arms:
                def cumulative_oracle_ledger():
                    combined = StepLedger()
                    copy_ledger(acq_bank.ledger, combined, "common_")
                    copy_ledger(common_ledger.to_dict(), combined)
                    copy_ledger(oracle_ledger.to_dict(), combined)
                    return combined
                active_oracle = OraclePoolProgress(data_dir / f"oracle-pool-s{seed}",
                                                   acq_bank, cumulative_oracle_ledger)
            seed_branches = execute_candidates(env, acq_bank, seed_cands, common_ledger, "seed",
                                               on_branch=active_oracle.add if active_oracle else None)
            oracle_truth, oracle_by_key = {}, {}
            if "oracle" in arms:
                rest = list(pool.unexecuted())
                active_ledger = oracle_ledger
                oracle_branches = execute_candidates(env, acq_bank, rest, oracle_ledger, "oracle_pool",
                                                     on_branch=active_oracle.add)
                for c in rest:
                    c.executed = False
                pool_ledger = StepLedger()
                copy_ledger(acq_bank.ledger, pool_ledger, "common_")
                copy_ledger(common_ledger.to_dict(), pool_ledger)
                copy_ledger(oracle_ledger.to_dict(), pool_ledger)
                pool_path = data_dir / f"oracle-pool-s{seed}"
                all_pool_branches = seed_branches + oracle_branches
                pool_gate = preserve_oracle_pool(pool_path, acq_bank, all_pool_branches, pool_ledger,
                                                run_id=run_id, seed=seed, hf=hf, writer=active_oracle.writer)
                active_oracle = None
                report.setdefault("oracle_pool_results", {})[str(seed)] = pool_gate
                if not pool_gate["gate"]["passes"]:
                    report["status"] = "oracle_undefined_censored_pool"
                    report["evaluation_ledger"] = evaluation_ledger.to_dict()
                    report["wall_clock_s"] = time.time() - t_start
                    (result_dir / "oracle_gate.json").write_text(json.dumps(pool_gate, indent=1) + "\n")
                    (result_dir / "acquisition.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
                    write_result_manifest(report, result_dir, t_start)
                    env.close()
                    return 2
                oracle_truth = {tuple(b.params["key"]): float(clearance_trace(b.log.states[:, 2:5], lay[b.root_id]["familiar"]).min()) for b in oracle_branches}
                oracle_by_key = {tuple(b.params["key"]): b for b in oracle_branches}
            for arm in arms:
                # Ranking randomness cannot alter the paired root or prefix schedule.
                arm_rng = np.random.default_rng(4000 + seed)
                seed_keys = {c.key for c in seed_cands}
                for c in pool.candidates:
                    c.executed = c.key in seed_keys
                ledger = StepLedger()
                copy_ledger(acq_bank.ledger, ledger, "common_")
                copy_ledger(common_ledger.to_dict(), ledger)
                if arm == "oracle":
                    copy_ledger(oracle_ledger.to_dict(), ledger)
                active_ledger = ledger
                store = DataStore(data_dir / f"{arm}-s{seed}", imaginer, acq_contexts)
                active_store = store
                store.add(acq_bank, seed_branches, lay)
                paired_cfg = AdaptConfig(**dict(cfg.to_dict(), seed=cfg.seed + seed))
                m_cur, seed_log = adapt(base_sd, model, store.clipset(), replay, paired_cfg, device=device, verbose=False)
                curve, total_added, training_total, scoring_total = [], 0, seed_log["wall_clock_s"], 0.0
                report["arms"].setdefault(arm, {})[str(seed)] = {"curve": curve, "root_schedule": schedule,
                    "seed_training_time_s": seed_log["wall_clock_s"], "oracle_pool_steps": oracle_ledger.total if arm == "oracle" else None}
                risk = RiskModel()
                for n_round in rounds:
                    im_cur = Imaginer(m_cur, process, device)
                    acq_contexts.imaginer = im_cur
                    scoring_start = time.time()
                    score_pool(pool, im_cur, probe, lay, acq_bank, cache=acq_contexts)
                    scoring_total += time.time() - scoring_start
                    ledger.add("model_queries", 0, branches=len(pool.candidates))
                    if arm == "learned":
                        ex = [c for c in pool.candidates if c.executed and c.key in store.true_c]
                        if not ex:
                            raise ValueError("No fully observed optimistic-error targets for the learned selector")
                        risk.fit(feature_matrix(ex), np.array([c.features["c_hat"] - store.true_c[c.key] for c in ex]))
                    true_err = ({c.key: c.features["c_hat"] - oracle_truth[c.key] for c in pool.unexecuted()} if arm == "oracle" else None)
                    batch_schedule = schedule[SEED_BRANCHES + total_added:SEED_BRANCHES + total_added + n_round]
                    chosen = select(arm, pool, n_round, arm_rng, margin=base_eval["margin_matched_dev"],
                                    band=args.band, risk_model=risk, true_errors=true_err, schedule=batch_schedule)
                    if arm == "oracle":
                        branches = [oracle_by_key[c.key] for c in chosen]
                        for c in chosen:
                            c.executed = True
                    else:
                        branches = execute_candidates(env, acq_bank, chosen, ledger, "branch")
                    store.add(acq_bank, branches, lay)
                    total_added += len(branches)
                    m_cur, log = adapt(base_sd, model, store.clipset(), replay, paired_cfg, device=device, verbose=False)
                    training_total += log["wall_clock_s"]
                    ev, rows = evaluate_all(m_cur, process, probe, banks, caches, device, target_ar)
                    point = {"budget_added": total_added, "charged_steps": ledger.total, "ledger": ledger.to_dict(), "eval": ev,
                        "n_trainable_branches": len(store.clipset()),
                        "n_queried_branches": len(pool.candidates) if arm == "oracle" else SEED_BRANCHES + total_added,
                        "n_retained_branches": SEED_BRANCHES + total_added,
                        "n_censored_round": sum(b.log.censored for b in branches),
                        "n_invalid_training_round": sum(not b.log.valid_for_training for b in branches),
                        "discarded_padding_steps": sum(v for k, v in ledger.counts.items() if "discarded_padding" in k),
                        "retention": retention_metrics(m_cur, Imaginer, process, probe, retention, device),
                        "train_time_s": log["wall_clock_s"], "training_time_cumulative_s": training_total,
                        "scoring_time_cumulative_s": scoring_total,
                        "selected_keys": [list(c.key) for c in chosen],
                        "chosen_kinds": {k: int(sum(c.proposal.kind == k for c in chosen)) for k in ("nominal", "random", "stress", "toward_hazard")},
                        "chosen_c_hat_mean": float(np.mean([c.features["c_hat"] for c in chosen])),
                        "chosen_true_c_mean": (float(np.mean([store.true_c[c.key] for c in chosen if c.key in store.true_c]))
                                               if any(c.key in store.true_c for c in chosen) else None)}
                    curve.append(point)
                    comparison_rows[arm, seed, total_added] = rows
                    budget_id = f"{run_id}-{arm}-s{seed}-b{total_added}"
                    budget_dir = checkpoint_dir / budget_id
                    budget_dir.mkdir()
                    torch.save(predictor_side_state(m_cur, cfg.modules), budget_dir / "weights.pt")
                    point["weights_sha256"] = hashlib.sha256((budget_dir / "weights.pt").read_bytes()).hexdigest()
                    (budget_dir / "config.json").write_text(json.dumps(paired_cfg.to_dict(), indent=1) + "\n")
                    rows_path = result_dir / f"{arm}-s{seed}-b{total_added}-rows.npz"
                    np.savez_compressed(rows_path, rows=json.dumps(rows))
                    point["evaluation_rows"] = {"file": rows_path.name,
                        "sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest()}
                    write_manifest(budget_dir, build_manifest(run_id=budget_id, kind="adapted", seeds={"acq_seed": seed, "train_seed": paired_cfg.seed},
                        data={"arm": arm, "budget_added": total_added, "diagnostic": args.diagnostic, "assets_run": str(args.assets_run),
                              "probes_run": str(args.probes_run), "evaluation_banks": str(args.banks_dir)}, costs=ledger.to_dict(), metrics=ev["test"]["at_matched"]))
                    if hf:
                        point["hf_revision"] = hf.upload_run("pusht", "adapted", budget_dir, run_id=budget_id)
                        point["hf_repo"] = hf.repo_id("pusht")
                    (result_dir / "acquisition.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
                    print(f"[e3] {gate['interpretation']} seed {seed} {arm} +{total_added}: {ledger.total} steps, test FSA {ev['test']['at_matched']['fsa']:.3f}")
                store.writer.finish(ledger, {"arm": arm, "seed": seed, "run_id": run_id, "diagnostic": args.diagnostic})
                active_store = None
                write_manifest(store.path, build_manifest(run_id=f"{run_id}-{arm}-s{seed}-bank", kind="bank", seeds={"acq_seed": seed}, costs=ledger.to_dict()))
                if hf:
                    report["arms"][arm][str(seed)]["bank_hf_revision"] = hf.upload_run("pusht-banks", "banks", store.path, run_id=f"{run_id}-{arm}-s{seed}-bank")
        comp = {}
        if "random" in arms and "boundary" in arms:
            for bi in range(len(rounds)):
                budget = int(sum(rounds[:bi + 1]))
                per_seed = {}
                for seed in args.seeds:
                    a = report["arms"]["random"][str(seed)]["curve"][bi]
                    b = report["arms"]["boundary"][str(seed)]["curve"][bi]
                    if a["charged_steps"] != b["charged_steps"]:
                        raise AssertionError("Paired acquisition arms have unequal charged simulator costs")
                    ra, rb = comparison_rows["random", seed, budget]["test"], comparison_rows["boundary", seed, budget]["test"]
                    if [(r["root"], r["branch"], r["layout"]) for r in ra] != [(r["root"], r["branch"], r["layout"]) for r in rb]:
                        raise AssertionError("Evaluation rows are not paired")
                    ma, mb = a["eval"]["margin_matched_dev"], b["eval"]["margin_matched_dev"]
                    per_seed[str(seed)] = cluster_bootstrap(lambda ca, cb, u, censored: fsa(cb, u, mb, censored=censored)["fsa"] - fsa(ca, u, ma, censored=censored)["fsa"],
                        np.asarray([r["root"] for r in ra]), n_boot=1000,
                        ca=np.asarray([r["cmin_imagined"] for r in ra]), cb=np.asarray([r["cmin_imagined"] for r in rb]),
                        u=np.asarray([r["unsafe"] for r in ra]), censored=np.asarray([r["censored"] for r in ra]))
                comp[str(budget)] = {"paired_root_intervals_by_seed": per_seed}
        report["boundary_vs_random_test"] = comp
        report["evaluation_ledger"] = evaluation_ledger.to_dict()
        report["wall_clock_s"] = time.time() - t_start
        report["status"] = "diagnostic_complete" if args.diagnostic else "complete"
        report["closedloop_gate"] = acquisition_repeatability_gate(report)
        (result_dir / "closedloop_gate.json").write_text(json.dumps(report["closedloop_gate"], indent=1) + "\n")
        (result_dir / "acquisition.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
        make_figures(report, result_dir)
        write_result_manifest(report, result_dir, t_start)
        if "oracle" in arms:
            (result_dir / "oracle_gate.json").write_text(json.dumps({"gate": {"passes": True},
                "interpretation": "Whole-pool optimistic-error reference completed; simulator cost is not comparable"}, indent=1) + "\n")
        env.close()
        print(f"[e3] {report['status']}: {result_dir}")
        return 0
    except BaseException as exc:
        report["status"] = "incomplete"
        report["error_type"] = type(exc).__name__
        if active_oracle is not None:
            active_ledger = active_oracle.ledger()
            if active_oracle.writer.h5.id.valid:
                report["oracle_pool_failure"] = active_oracle.abort(type(exc).__name__)
        report["active_ledger"] = active_ledger.to_dict() if active_ledger else None
        if active_store is not None and active_store.writer.h5.id.valid:
            active_store.writer.finish(active_ledger or StepLedger(), {"run_id": run_id, "incomplete": True})
        (result_dir / "incomplete.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
        raise


def oracle_observation_gate(branches):
    """Full-future optimistic-error ranking is undefined if any paid tape is censored."""
    censored = [list(branch.params["key"]) for branch in branches if branch.log.censored]
    return {"gate": {"passes": bool(branches) and not censored},
            "n_queried_branches": len(branches), "n_censored": len(censored),
            "n_invalid_training": sum(not branch.log.valid_for_training for branch in branches),
            "censored_keys": censored,
            "interpretation": ("Whole-pool optimistic-error reference; simulator cost is not comparable"
                if branches and not censored else
                "Full-future optimistic errors are undefined for censored tapes. All queries are preserved and charged; no ceiling or efficiency claim is made.")}


class OraclePoolProgress:
    """Persist every completed paid trace before another query, including common seeds."""

    def __init__(self, path, bank, ledger):
        self.path, self.ledger = path, ledger
        self.writer = BankWriter(path)
        for root in bank.roots:
            self.writer.add_root(root)
        (path / "layouts.json").write_bytes((bank.dir / "layouts.json").read_bytes())
        self.snapshot("running")

    def snapshot(self, status, error_type=None):
        self.writer.h5.flush()
        record = {"status": status, "roots": self.writer.roots,
                  "n_branches": int(self.writer.h5["tape"].shape[0]),
                  "ledger": self.ledger().to_dict(),
                  "manifest": {"bank": "oracle_full_pool", "outcomes_filtered": False}}
        if error_type:
            record["error_type"] = error_type
        temporary = self.path / "roots.json.tmp"
        temporary.write_text(json.dumps(record, indent=1) + "\n")
        temporary.replace(self.path / "roots.json")
        return record

    def add(self, branch):
        self.writer.add_branches([branch])
        self.snapshot("running")

    def abort(self, error_type):
        record = self.snapshot("incomplete", error_type)
        self.writer.h5.close()
        return {"bank": str(self.path.resolve()), "stored_branches": record["n_branches"],
                "ledger": record["ledger"], "error_type": error_type,
                "interpretation": "All completed traces saved; any unfinished query has no complete outcome and its attempted steps remain charged"}


def preserve_oracle_pool(path, bank, branches, ledger, *, run_id, seed, hf, writer=None):
    """Save all paid outcomes before the oracle gate, including untrainable branches."""
    if writer is None:
        writer = BankWriter(path)
        for root in bank.roots:
            writer.add_root(root)
        writer.add_branches(branches)
    if writer.h5["tape"].shape[0] != len(branches):
        raise ValueError("Incremental oracle bank does not contain every queried branch")
    (path / "layouts.json").write_bytes((bank.dir / "layouts.json").read_bytes())
    writer.finish(ledger, {"bank": "oracle_full_pool", "status": "complete", "seed": seed,
                           "run_id": run_id, "outcomes_filtered": False})
    gate = oracle_observation_gate(branches)
    gate.update(bank=str(path.resolve()), bank_identity=bank_identity(path), ledger=ledger.to_dict())
    pool_id = f"{run_id}-oracle-pool-s{seed}"
    write_manifest(path, build_manifest(run_id=pool_id, kind="oracle-pool", seeds={"acquisition": seed},
        data={"all_paid_outcomes_retained": True, "observation_gate": gate}, costs=ledger.to_dict()))
    if hf:
        gate["hf_revision"] = hf.upload_run("pusht-banks", "banks", path, run_id=pool_id)
        gate["hf_repo"] = hf.repo_id("pusht-banks")
    return gate


def write_result_manifest(report, result_dir, started_at):
    provenance = result_provenance(report)
    write_manifest(result_dir, build_manifest(run_id=report["run_id"], kind="acquisition-report",
        seeds={"acquisition_seeds": report["acquisition_seeds"]},
        data=provenance, upstream_revisions=report.get("upstream_revisions", {}),
        costs={"root_collection": report.get("root_collection_ledger"),
               "evaluation": report.get("evaluation_ledger"),
               "accounting": "Per-arm ledger includes shared root costs; do not sum shared arm allocations as new physical work"},
        extra={"acquisition_report_sha256": file_sha256(result_dir / "acquisition.json"),
               "status": report["status"]}, started_at=started_at))


def make_figures(report, result_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    table_rows = []
    for ax, bank in zip(axes, ("test", "stress")):
        for arm, seeds in report["arms"].items():
            # Plot each seed at its own paid costs; do not average unequal x coordinates.
            for seed, values in seeds.items():
                xs = [point["charged_steps"] for point in values["curve"]]
                ys = [point["eval"][bank]["at_matched"]["fsa"] for point in values["curve"]]
                label = f"{arm}, seed {seed}" + (" (retrospective)" if arm == "oracle" else "")
                line, = ax.plot(xs, ys, marker="o", label=label)
                for point in values["curve"]:
                    metrics = point["eval"][bank]
                    counts, ci = metrics["at_matched"], metrics["fsa_matched_ci"]
                    finite = np.isfinite(counts["fsa"]) and np.isfinite(ci["lo"]) and np.isfinite(ci["hi"])
                    if finite:
                        ax.vlines(point["charged_steps"], ci["lo"], ci["hi"], color=line.get_color(), alpha=.55)
                    table_rows.append({"bank": bank, "arm": arm, "acquisition_seed": seed,
                        "additional_branches": point["budget_added"], "charged_steps": point["charged_steps"],
                        "queried_branches": point["n_queried_branches"], "retained_branches": point["n_retained_branches"],
                        "trainable_branches": point["n_trainable_branches"], "fsa": counts["fsa"],
                        "fsa_ci_lo": ci["lo"] if finite else None, "fsa_ci_hi": ci["hi"] if finite else None,
                        "bootstrap_usable": ci["n_boot"], "bootstrap_requested": 300,
                        "acceptance_rate": counts["acceptance_rate"], "accepted": counts["n_accepted"],
                        "false_safe": counts["n_false_safe"], "total": counts["n"],
                        "accepted_censored": counts.get("n_accepted_censored", 0),
                        "training_seconds": point["training_time_cumulative_s"],
                        "scoring_seconds": point["scoring_time_cumulative_s"]})
        ax.axhline(report["no_update"][bank]["at_matched"]["fsa"], color="k", ls="--", label="no update")
        ax.set_xlabel("charged simulator steps (roots + seed + acquired)")
        ax.set_ylabel("FSA at matched acceptance")
        ax.set_title(f"{bank} bank")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(result_dir / "fsa_vs_charged_steps.png", dpi=130)
    plt.close(fig)
    if table_rows:
        with (result_dir / "acquisition_table.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
            writer.writeheader()
            writer.writerows(table_rows)
    (result_dir / "README.md").write_text(
        "# Acquisition report\n\n"
        "The CSV gives counts, charged simulator steps, and model-scoring/training time beside every rate. "
        "Figure bars are 95% root-cluster bootstrap intervals at the development-calibrated margins. "
        "Seeds retain their own paid x-coordinates. Undefined full-data FSA has no plotted interval; "
        "usable bootstrap replicates do not resolve it. Baseline evaluation is in acquisition.json.\n\n"
        "An oracle arm is a retrospective data-usefulness reference charged for its entire queried pool. "
        "Its interaction cost is not comparable to prospective random/boundary acquisition. "
        "Shared root-generation costs are allocated to each arm and must not be added together as fresh physical queries.\n")


if __name__ == "__main__":
    raise SystemExit(main())
