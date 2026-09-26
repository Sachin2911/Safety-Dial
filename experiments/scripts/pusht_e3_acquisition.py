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

from helpers.acquisition import CandidatePool, RiskModel, feature_matrix, score_pool, select  # noqa: E402
from helpers.branchBank import Bank, BankWriter, Branch, build_root, expert_pairs  # noqa: E402
from helpers.decomposition import RootLatentCache, evaluate_model_on_bank, layouts_by_root, ordinary_motion  # noqa: E402
from helpers.dialMetrics import auc_dial, clearance_error_stats, cluster_bootstrap, fsa, margin_for_acceptance  # noqa: E402
from helpers.hfStore import HFStore  # noqa: E402
from helpers.imagination import Imaginer, NominalPlanner  # noqa: E402
from helpers.poseProbes import load_probe  # noqa: E402
from helpers.predictorAdapt import AdaptConfig, ClipSet, adapt, branch_clips, predictor_side_state  # noqa: E402
from helpers.pushtAssets import H5_PATH, load_model, load_scalers  # noqa: E402
from helpers.pushtGeometry import clearance_trace  # noqa: E402
from helpers.pushtLayouts import generate_layout, load_layouts, save_layouts  # noqa: E402
from helpers.pushtReplay import StepLedger, execute_tape, make_env, reset_root  # noqa: E402
from helpers.runManifest import build_manifest, make_run_id, write_manifest  # noqa: E402

ASSETS_RUN = REPO_ROOT / "runs" / "pusht-assets-20260926-1"
PROBES_RUN = REPO_ROOT / "runs" / "pusht-probes-20260926-1"
STUDY = REPO_ROOT / "data" / "study" / "pusht"
RESULTS = REPO_ROOT / "docs" / "mainPlan" / "results" / "e3"
RESULTS4 = REPO_ROOT / "docs" / "mainPlan" / "results" / "e4"
KS = [0, 2, 4, 6]
ROUNDS = [64, 64, 128, 256]
SEED_BRANCHES = 128


def build_acq_roots(n_roots, seed, splits, model, process, device):
    """Roots + familiar layouts for acquisition (adaptation split, disjoint from E2's bank)."""
    bank_dir = STUDY / "acq_roots"
    if (bank_dir / "roots.json").is_file():
        return Bank(bank_dir), load_layouts(bank_dir / "layouts.json")[0]
    env = make_env()
    planner = NominalPlanner(model, process, device)
    rng = np.random.default_rng(seed)
    pairs = expert_pairs(H5_PATH, splits["roles"]["reserve"][3000:8000], rng, n_roots * 3)
    writer = BankWriter(bank_dir, with_frames=False)
    ledger = StepLedger()
    layouts, i, built = [], 0, 0
    t0 = time.time()
    while built < n_roots:
        root, ctx, _ = build_root(env, planner, pairs[i], seed=seed + i, k=KS[i % 4], root_id=f"acq-r{i:03d}", ledger=ledger)
        reset_root(env, root, record_frames=False)
        log = execute_tape(env, np.asarray(root.meta["nominal_plan"]).reshape(-1, 2), record_frames=False)
        ledger.add("layout_route", len(root.prefix) + 25)
        route = np.concatenate([ctx.prefix_log.states[:, 2:5], log.states[1:, 2:5]], 0)
        lay = generate_layout(rng, route, ctx.state[2:5], root.goal_state[2:5], family="familiar", root_id=root.root_id)
        i += 1
        if lay is None:
            continue
        writer.add_root(root)
        layouts.append(lay)
        built += 1
        if built % 16 == 0:
            print(f"[e3] acq roots {built}/{n_roots} {time.time() - t0:.0f}s")
    writer.finish(ledger, {"bank": "acq_roots", "seed": seed})
    save_layouts(bank_dir / "layouts.json", layouts, {"bank": "acq_roots"})
    return Bank(bank_dir), layouts


def execute_candidates(env, bank, cands, ledger: StepLedger, category: str) -> list[Branch]:
    out = []
    for c in cands:
        root = bank.roots[c.root_index]
        reset_root(env, root, record_frames=False)
        log = execute_tape(env, c.proposal.tape.reshape(-1, 2), record_frames=True)
        ledger.add(category, len(root.prefix) + 25, branches=1)
        c.executed = True
        out.append(Branch(root.root_id, c.proposal.tape, c.proposal.kind, dict(c.proposal.params, key=list(c.key)), log))
    return out


class DataStore:
    """Executed branches for one (arm, seed): a bank on disk plus cached clips."""

    def __init__(self, path: Path, imaginer):
        import shutil

        if path.exists():
            shutil.rmtree(path)
        self.writer = BankWriter(path)
        self.path = path
        self.imaginer = imaginer
        self.clips: list[ClipSet] = []
        self.true_c: dict[tuple, float] = {}
        self.n = 0

    def add(self, bank, branches, lay):
        for b in branches:
            self.writer.add_root(next(r for r in bank.roots if r.root_id == b.root_id))
        self.writer.add_branches(branches)
        tmp = Bank.__new__(Bank)
        tmp.roots, tmp.h5, tmp.ledger, tmp.manifest = [__import__("helpers.pushtReplay", fromlist=["Root"]).Root.from_dict(r) for r in self.writer.roots], self.writer.h5, {}, {}
        idx = range(self.n, self.n + len(branches))
        self.clips.append(branch_clips(self.imaginer, tmp, idx))
        for j, b in zip(idx, branches):
            hz = lay[b.root_id]["familiar"]
            self.true_c[tuple(b.params["key"])] = float(clearance_trace(b.log.states[:, 2:5], hz).min())
        self.n += len(branches)

    def clipset(self) -> ClipSet:
        return ClipSet.concat(self.clips)


def evaluate_all(model, process, probe, banks, caches, device, target_ar):
    im = Imaginer(model, process, device)
    rows = {}
    for n, (b, lays) in banks.items():
        caches[n].imaginer = im
        rows[n] = evaluate_model_on_bank(n, b, lays, im, probe, cache=caches[n])
    m = margin_for_acceptance(np.array([r["cmin_imagined"] for r in rows["dev"]]), target_ar)
    out = {"margin_matched_dev": float(m)}
    for n, rr in rows.items():
        u = np.array([r["cmin_dense"] < 0 for r in rr])
        c = np.array([r["cmin_imagined"] for r in rr])
        root = np.array([r["root"] for r in rr])
        out[n] = {"at_matched": fsa(c, u, m), "at_m0": fsa(c, u, 0.0), "auc_dial": auc_dial(c, u), "clearance_error": clearance_error_stats(c, np.array([r["cmin_dense"] for r in rr])),
                  "ordinary_motion": ordinary_motion(rr), "fsa_matched_ci": cluster_bootstrap(lambda c, u: fsa(c, u, m)["fsa"], root, n_boot=300, c=c, u=u)}
        # E4 grid: layout family x source family
        src = {r.root_id: r.meta.get("source_family", "familiar") for r in banks[n][0].roots}
        grid = {}
        for lf in ("familiar", "heldout"):
            for sf in ("familiar", "heldout"):
                sub = [r for r in rr if r["layout"] == lf and src[banks[n][0].roots[r["root"]].root_id] == sf]
                if sub:
                    grid[f"layout={lf}/source={sf}"] = fsa(np.array([r["cmin_imagined"] for r in sub]), np.array([r["cmin_dense"] < 0 for r in sub]), m)
        out[n]["transfer_grid"] = grid
    return out, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--arms", nargs="+", default=["random", "boundary"])
    ap.add_argument("--learned", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--acq-roots", type=int, default=96)
    ap.add_argument("--pool-per-root", type=int, default=40)
    ap.add_argument("--band", type=float, default=10.0)
    ap.add_argument("--recipe", default=None, help="json AdaptConfig; default: E2 chosen recipe")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    t_start = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    RESULTS4.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id("pusht", "acq", n=args.n)
    device = "cuda"
    model = load_model(device)
    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    process = load_scalers(ASSETS_RUN / "scalers.npz")
    splits = json.loads((ASSETS_RUN / "splits.json").read_text())
    probe, _ = load_probe(PROBES_RUN / "block_pose_mlp.pt", device)
    imaginer = Imaginer(model, process, device)
    if args.recipe:
        cfg = AdaptConfig(**json.loads(args.recipe))
    else:
        cfg = AdaptConfig(**json.loads((REPO_ROOT / "docs" / "mainPlan" / "results" / "e2" / "repair.json").read_text())["chosen_recipe"])
    print(f"[e3] recipe {cfg.to_dict()}")
    replay = ClipSet.load(STUDY / "clips" / "replay.npz")
    banks = {n: (Bank(STUDY / n), load_layouts(STUDY / n / "layouts.json")[0]) for n in ("dev", "test", "stress")}
    caches = {n: RootLatentCache(imaginer, b) for n, (b, _) in banks.items()}
    arms = list(args.arms) + (["learned"] if args.learned else []) + (["oracle"] if args.oracle else [])

    # ---- roots, pool, seed branches (common) ---------------------------------------------
    acq_bank, acq_layouts = build_acq_roots(args.acq_roots, 20261003, splits, model, process, device)
    lay = layouts_by_root(acq_layouts)
    env = make_env()
    dev_rows0 = evaluate_model_on_bank("dev", banks["dev"][0], banks["dev"][1], imaginer, probe, cache=caches["dev"])
    target_ar = float(np.mean([r["cmin_imagined"] >= 0 for r in dev_rows0]))  # dev AR of the unadapted model at m=0
    base_eval, _ = evaluate_all(model, process, probe, banks, caches, device, target_ar)
    report = {"run_id": run_id, "recipe": cfg.to_dict(), "target_acceptance_rate_dev": target_ar, "no_update": base_eval, "arms": {}, "rounds": ROUNDS, "seed_branches": SEED_BRANCHES}
    print(f"[e3] no-update test FSA@matched {base_eval['test']['at_matched']['fsa']:.3f} (AR {base_eval['test']['at_matched']['acceptance_rate']:.2f})")

    for seed in args.seeds:
        rng = np.random.default_rng(1000 + seed)
        pool = CandidatePool.build(rng, acq_bank, args.pool_per_root, n_stress=6, n_toward=6, layouts_by_root=lay)
        print(f"[e3] seed {seed}: pool {len(pool.candidates)} candidates over {len(acq_bank.roots)} roots")
        common_ledger = StepLedger()
        seed_cands = select("random", pool, SEED_BRANCHES, rng)
        seed_branches = execute_candidates(env, acq_bank, seed_cands, common_ledger, "seed")
        oracle_truth = None
        if "oracle" in arms:
            oracle_ledger = StepLedger()
            rest = [c for c in pool.candidates if not c.executed]
            oracle_branches = execute_candidates(env, acq_bank, rest, oracle_ledger, "oracle_pool")
            for c in rest:
                c.executed = False
            oracle_truth = {tuple(b.params["key"]): float(clearance_trace(b.log.states[:, 2:5], lay[b.root_id]["familiar"]).min()) for b in oracle_branches}
            oracle_by_key = {tuple(b.params["key"]): b for b in oracle_branches}
        for arm in arms:
            seed_ids = {id(c) for c in seed_cands}
            for c in pool.candidates:
                c.executed = id(c) in seed_ids
            ledger = StepLedger()
            ledger.add("seed", common_ledger.total, branches=len(seed_cands))
            store = DataStore(STUDY / f"acq-{arm}-s{seed}", imaginer)
            store.add(acq_bank, seed_branches, lay)
            m_cur, _ = adapt(base_sd, model, store.clipset(), replay, AdaptConfig(**dict(cfg.to_dict(), seed=cfg.seed + seed)), device=device, verbose=False)
            curve = []
            total_added = 0
            risk = RiskModel()
            for r_i, n_round in enumerate(ROUNDS):
                im_cur = Imaginer(m_cur, process, device)
                cache_acq = RootLatentCache(im_cur, acq_bank)
                score_pool(pool, im_cur, probe, lay, acq_bank, cache=cache_acq)
                ledger.add("model_queries", 0, branches=len(pool.unexecuted()))
                if arm == "learned":
                    ex = [c for c in pool.candidates if c.executed]
                    X = feature_matrix(ex)
                    y = np.array([c.features["c_hat"] - store.true_c[c.key] for c in ex])
                    risk.fit(X, y)
                true_err = None
                if arm == "oracle":
                    true_err = {c.key: c.features["c_hat"] - oracle_truth[c.key] for c in pool.unexecuted() if c.key in oracle_truth}
                chosen = select(arm, pool, n_round, rng, margin=base_eval["margin_matched_dev"], band=args.band, risk_model=risk, true_errors=true_err)
                if arm == "oracle":
                    branches = []
                    for c in chosen:
                        b = oracle_by_key[c.key]
                        c.executed = True
                        ledger.add("branch", len(acq_bank.roots[c.root_index].prefix) + 25, branches=1)
                        branches.append(b)
                else:
                    branches = execute_candidates(env, acq_bank, chosen, ledger, "branch")
                store.add(acq_bank, branches, lay)
                total_added += len(branches)
                m_cur, log = adapt(base_sd, model, store.clipset(), replay, AdaptConfig(**dict(cfg.to_dict(), seed=cfg.seed + seed)), device=device, verbose=False)
                ev, _ = evaluate_all(m_cur, process, probe, banks, caches, device, target_ar)
                point = {"budget_added": total_added, "charged_steps": ledger.total, "ledger": ledger.to_dict(), "eval": ev, "train_time_s": log["wall_clock_s"],
                         "chosen_kinds": {k: int(sum(c.proposal.kind == k for c in chosen)) for k in ("nominal", "random", "stress", "toward_hazard")},
                         "chosen_c_hat_mean": float(np.mean([c.features["c_hat"] for c in chosen])), "chosen_true_c_mean": float(np.mean([store.true_c[c.key] for c in chosen]))}
                curve.append(point)
                print(f"[e3] seed {seed} {arm:8s} +{total_added:3d} ({ledger.total} steps): test FSA@matched {ev['test']['at_matched']['fsa']:.3f} stress {ev['stress']['at_matched']['fsa']:.3f} AUC {ev['test']['auc_dial']:.3f} motion {ev['test']['ordinary_motion'].get('disp_ratio', float('nan')):.2f} [{log['wall_clock_s']:.0f}s]")
            report["arms"].setdefault(arm, {})[str(seed)] = {"curve": curve, "oracle_pool_steps": oracle_ledger.total if arm == "oracle" else None}
            run_dir = REPO_ROOT / "runs" / f"{run_id}-{arm}-s{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(predictor_side_state(m_cur, cfg.modules), run_dir / "weights.pt")
            write_manifest(run_dir, build_manifest(run_id=run_dir.name, kind="adapted", seeds={"acq_seed": seed}, data={"arm": arm, "budget_added": total_added}, costs=ledger.to_dict(), metrics=curve[-1]["eval"]["test"]["at_matched"]))
            (run_dir / "README.md").write_text(f"# {run_dir.name}\n\nE3 {arm} arm, acquisition seed {seed}, largest budget. Predictor-side modules only.\n")
            if not args.no_upload:
                HFStore().upload_run("pusht", "adapted", run_dir, run_id=run_dir.name)
            store.writer.finish(ledger, {"arm": arm, "seed": seed})
            (RESULTS / "acquisition.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    # ---- paired comparison boundary vs random at each budget ----------------------------
    comp = {}
    if "random" in report["arms"] and "boundary" in report["arms"]:
        for b_i in range(len(ROUNDS)):
            rs = [report["arms"]["random"][s]["curve"][b_i]["eval"]["test"]["at_matched"]["fsa"] for s in report["arms"]["random"]]
            bs = [report["arms"]["boundary"][s]["curve"][b_i]["eval"]["test"]["at_matched"]["fsa"] for s in report["arms"]["boundary"]]
            comp[str(report["arms"]["random"][list(report["arms"]["random"])[0]]["curve"][b_i]["budget_added"])] = {"random_mean": float(np.nanmean(rs)), "boundary_mean": float(np.nanmean(bs)), "diff_boundary_minus_random_per_seed": [float(b - r) for b, r in zip(bs, rs)]}
    report["boundary_vs_random_test"] = comp
    report["wall_clock_s"] = time.time() - t_start
    (RESULTS / "acquisition.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    # E4: transfer grid at the largest budget per arm (first seed) plus no-update
    e4 = {"no_update": {n: base_eval[n]["transfer_grid"] for n in ("test", "stress")}}
    for arm in report["arms"]:
        for s in report["arms"][arm]:
            e4[f"{arm}-s{s}"] = {n: report["arms"][arm][s]["curve"][-1]["eval"][n]["transfer_grid"] for n in ("test", "stress")}
    (RESULTS4 / "transfer.json").write_text(json.dumps(e4, indent=1, default=float) + "\n")
    make_figures(report)
    print(f"[e3] done in {time.time() - t_start:.0f}s")
    return 0


def make_figures(report):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, bank in zip(axes, ("test", "stress")):
        for arm, seeds in report["arms"].items():
            xs = [p["charged_steps"] for p in next(iter(seeds.values()))["curve"]]
            ys = np.array([[p["eval"][bank]["at_matched"]["fsa"] for p in s["curve"]] for s in seeds.values()], float)
            ax.errorbar(xs, np.nanmean(ys, 0), yerr=np.nanstd(ys, 0), marker="o", label=arm, capsize=3)
        ax.axhline(report["no_update"][bank]["at_matched"]["fsa"], color="k", ls="--", label="no update")
        ax.set_xlabel("charged simulator steps (seed + acquired)")
        ax.set_ylabel("FSA at matched acceptance")
        ax.set_title(f"{bank} bank")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fsa_vs_charged_steps.png", dpi=130)


if __name__ == "__main__":
    raise SystemExit(main())
