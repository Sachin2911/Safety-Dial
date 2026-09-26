#!/usr/bin/env python3
"""E2: can extra experience repair the diagnosed error? (pushT.md)

1. Build the adaptation bank (about 512 branches from adaptation-split roots, contact-rich
   and ordinary) unless it exists; every simulated candidate is charged.
2. Cache replay clips (expert, replay split) and retention clips (expert, retention split).
3. Choose the recipe on the DEVELOPMENT bank only: loss, learning rate, steps, modules.
4. Evaluate the fixed recipe on the test and stress banks against the controls: no update,
   a fixed margin calibrated on development data, and a readout-only correction trained
   on the same new data. Report retention and ordinary motion.

Outputs the explicit new --results-dir/repair.json and figures; adapted weights to
<ns>/safetydial-pusht:adapted/<run_id>.

    uv run python experiments/scripts/pusht_e2_repair.py --banks-dir data/study/pusht/e1-clean-1 --study-dir data/study/pusht/e2-clean-1 --results-dir runs/e2-clean-1-results
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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

from helpers.branchBank import Bank, BankWriter, build_root, execute_proposals, expert_pairs, propose, exits_arena  # noqa: E402
from helpers.decomposition import RootLatentCache, decision_table, evaluate_model_on_bank, ordinary_motion, row_outcomes  # noqa: E402
from helpers.dialMetrics import auc_dial, clearance_error_stats, cluster_bootstrap, fsa, margin_for_acceptance  # noqa: E402
from helpers.hfStore import HFStore  # noqa: E402
from helpers.imagination import Imaginer, NominalPlanner  # noqa: E402
from helpers.poseProbes import load_probe, pose_to_target  # noqa: E402
from helpers.predictorAdapt import (  # noqa: E402
    AdaptConfig,
    ClipSet,
    adapt,
    branch_clips,
    expert_replay_clips,
    fit_readout_correction,
    predictor_side_state,
    rollout_loss,
    teacher_forced_loss,
)
from helpers.pushtAssets import ACTION_BLOCK, H5_PATH, load_model, load_scalers  # noqa: E402
from helpers.pushtSourceFamilies import SOURCE_FAMILY_PROTOCOL, geometric_source_family  # noqa: E402
from helpers.pushtLayouts import load_layouts  # noqa: E402
from helpers.pushtReplay import StepLedger
from helpers.acquisitionSafety import MeteredEnv, copy_ledger
from helpers.pushtContactReplay import make_env  # noqa: E402
from helpers.runManifest import validate_run_id, build_manifest, make_run_id, write_manifest, file_sha256  # noqa: E402
from helpers.splitIntegrity import validate_bank_splits, validate_decomposition_gate  # noqa: E402

ASSETS_RUN = REPO_ROOT / "runs" / "pusht-assets-20260926-1"
PROBES_RUN = REPO_ROOT / "runs" / "pusht-probes-20260926-1"
STUDY = REPO_ROOT / "data" / "study" / "pusht"
RESULTS = REPO_ROOT / "docs" / "mainPlan" / "results" / "e2"
KS = [0, 2, 4, 6]


def canonical_proposals(proposals, pusher_xy):
    """Check and execute exactly the float32 controls persisted by BankWriter."""
    accepted, rejected = [], 0
    for proposal in proposals:
        tape = np.asarray(proposal.tape, dtype=np.float32)
        if not np.isfinite(tape).all() or exits_arena(tape, pusher_xy):
            rejected += 1
        else:
            accepted.append(replace(proposal, tape=tape))
    return accepted, rejected


def metered_root_cache(imaginer, bank, ledger, category):
    cache = RootLatentCache(imaginer, bank)
    cache.env = MeteredEnv(cache.env, ledger, category)
    return cache


def metered_branch_clips(imaginer, bank, indices, ledger):
    env = MeteredEnv(make_env(), ledger, "adapt_training_history")
    try:
        return branch_clips(imaginer, bank, indices, env=env)
    finally:
        env.close()


def combined_simulator_ledger(collection, context, evaluation, goal_base, goal_adapted):
    """Disjoint actual work in this E2 run; historical source-bank costs stay upstream."""
    ledger = StepLedger()
    for prefix, source in (("collection_", collection), ("context_", context),
                           ("evaluation_", evaluation), ("goal_no_update_", goal_base),
                           ("goal_adapted_", goal_adapted)):
        copy_ledger(source, ledger, prefix)
    return ledger.to_dict()


def build_adapt_bank(n_roots, n_tapes, seed, splits, model, process, device, *, study_dir=STUDY) -> tuple[Path, StepLedger]:
    bank_dir = Path(study_dir) / "adapt"
    ledger = StepLedger()
    if (bank_dir / "roots.json").is_file():
        return bank_dir, ledger
    env = make_env()
    planner = NominalPlanner(model, process, device)
    rng = np.random.default_rng(seed)
    episodes = splits["roles"]["reserve"][:3000]
    pairs = expert_pairs(H5_PATH, episodes, rng, min(len(episodes), n_roots * 8))
    writer = BankWriter(bank_dir)
    contact_counts = {"typed_roots": 0, "roots_in_pusher_contact": 0, "typed_branches": 0, "pusher_contact_branches": 0}
    t0 = time.time()
    for i, pair in enumerate(pairs):
        if len(writer.roots) >= n_roots:
            break
        k = KS[i % len(KS)]
        try:
            root, ctx, _ = build_root(env, planner, pair, seed=seed + i, k=k, root_id=f"adapt-r{i:03d}", ledger=ledger)
        except ValueError as exc:
            if "censored or out-of-domain prefix" not in str(exc):
                raise
            ledger.add("discarded_invalid_root", 0, branches=1)
            continue
        if geometric_source_family(ctx.state, root.goal_state) != "familiar":
            ledger.add("discarded_nonfamiliar_source", 0, branches=1)
            continue
        root.meta["source_family"] = "familiar"
        root.meta["source_family_protocol"] = SOURCE_FAMILY_PROTOCOL["version"]
        props, rej = propose(rng, root, ctx, n_random=n_tapes - 4, sigmas=(0.05, 0.1, 0.2), n_stress=3)
        ledger.add("proposals_rejected_by_arena", 0, branches=rej)
        props, precision_rejections = canonical_proposals(props, ctx.state[:2])
        ledger.add("proposals_rejected_after_float32", 0, branches=precision_rejections)
        if not props:
            ledger.add("discarded_no_float32_proposals", 0, branches=1)
            continue
        br = execute_proposals(env, root, props, ledger=ledger)
        contact_counts["typed_roots"] += int(root.meta.get("contact_kind") == "pusher_block")
        contact_counts["roots_in_pusher_contact"] += int(root.meta.get("in_contact_last_block") is True)
        for branch in br:
            counts = getattr(branch.log, "pusher_block_contacts", None)
            contact_counts["typed_branches"] += int(counts is not None)
            contact_counts["pusher_contact_branches"] += int(counts is not None and (counts > 0).any())
        writer.add_root(root)
        writer.add_branches(br)
        if (i + 1) % 16 == 0:
            print(f"[e2] adapt bank {i + 1}/{n_roots} roots {time.time() - t0:.0f}s")
    writer.finish(ledger, {"bank": "adapt", "seed": seed, "source_family_protocol": SOURCE_FAMILY_PROTOCOL,
        "contact_kind": "pusher_block", "contact_counts": contact_counts,
        "contact_sampling": "nominal-depth schedule plus three tapes aimed at T geometry per root; actual body-specific contact share is reported"})
    if len(writer.roots) < n_roots:
        raise ValueError(f"Only {len(writer.roots)} valid adaptation roots from the fixed source pool")
    return bank_dir, ledger


def retention_metrics(model, imaginer_cls, process, probe, clips: ClipSet, device, ctx=3) -> dict:
    """Latent prediction error (teacher-forced and rollout) and horizon-5 pose error on held-out expert clips."""
    z = torch.from_numpy(clips.latents).to(device)
    a = torch.from_numpy(clips.actions).to(device)
    model.eval()
    with torch.no_grad():
        tf = float(teacher_forced_loss(model, z, a, ctx))
        ro = float(rollout_loss(model, z, a, ctx))
        # pose error at the last imagined step via the probe (clips: 3 history + 5 future frames)
        act_emb = model.action_encoder(a)
        emb_list = list(z[:, :ctx].unbind(1))
        for t in range(ctx, z.shape[1]):
            pred = model.predict(torch.stack(emb_list[t - ctx : t], 1), act_emb[:, t - ctx : t])[:, -1]
            emb_list.append(pred)
        z_last = emb_list[-1]
        pose_hat = probe.predict_pose(z_last)
        pose_true = probe.predict_pose(z[:, -1])  # readout of the real last frame (same probe, so readout error cancels)
    return {"latent_mse_tf": tf, "latent_mse_rollout": ro,
            "pose_h5_centre_px_vs_real_readout": float(np.median(np.linalg.norm(pose_hat[:, :2] - pose_true[:, :2], axis=1))), "n": int(len(z))}


def eval_rows_summary(rows, m_matched, m_zero=0.0) -> dict:
    u, censored = row_outcomes(rows)
    c = np.array([r["cmin_imagined"] for r in rows])
    ct = np.array([r["cmin_dense"] for r in rows])
    return {"at_m0": fsa(c, u, m_zero, censored=censored), "at_matched": fsa(c, u, m_matched, censored=censored), "auc_dial": auc_dial(c, u, censored=censored), "clearance_error": clearance_error_stats(c[~censored], ct[~censored]) if (~censored).any() else {"n": 0},
            "ordinary_motion": ordinary_motion(rows), "n": int(len(rows)), "n_unsafe": int(u.sum()), "n_censored": int(censored.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapt-roots", type=int, default=64)
    ap.add_argument("--adapt-tapes", type=int, default=8)
    ap.add_argument("--replay-clips", type=int, default=2000)
    ap.add_argument("--retention-clips", type=int, default=400)
    ap.add_argument("--retention-episodes", type=int, default=20)
    ap.add_argument("--retention-blocks", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--quick", action="store_true", help="smaller recipe grid")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--run-id", type=validate_run_id, help="Explicit stable run ID for queued workflows")
    ap.add_argument("--feasibility-report", type=Path, required=True, help="Passing development feasible-route witness artifact")
    ap.add_argument("--decomposition-report", type=Path, required=True, help="Passing development-only E1 report for these exact banks and probe")
    ap.add_argument("--banks-dir", type=Path, default=STUDY,
                    help="Existing source-disjoint E1 development/test/stress banks")
    ap.add_argument("--study-dir", type=Path, required=True,
                    help="Fresh adaptation bank and clip directory")
    ap.add_argument("--results-dir", type=Path, required=True,
                    help="Fresh results directory; historical E2 outputs are preserved")
    ap.add_argument("--assets-run", type=Path, default=ASSETS_RUN)
    ap.add_argument("--probes-run", type=Path, default=PROBES_RUN)
    args = ap.parse_args()
    t_start = time.time()
    study_dir, results = args.study_dir.resolve(), args.results_dir.resolve()
    bank_paths = {name: args.banks_dir / name for name in ("dev", "test", "stress")}
    split_report = validate_bank_splits(bank_paths)
    from helpers.pushtFeasibility import require_feasibility_report

    feasibility = require_feasibility_report(args.feasibility_report, bank_paths["dev"])
    decomposition_report = json.loads(args.decomposition_report.read_text())
    probe_name = validate_decomposition_gate(decomposition_report, args.banks_dir, args.probes_run)
    run_id = args.run_id or make_run_id("pusht", "adapt", "e2", n=args.n)
    run_dir = REPO_ROOT / "runs" / run_id
    for path in (study_dir, results, run_dir):
        if path.exists():
            ap.error(f"Use a fresh output path/run ID; preserving {path}")
    if study_dir == results:
        ap.error("Study and result directories must differ")
    results.mkdir(parents=True)
    study_dir.mkdir(parents=True, exist_ok=True)
    store = None if args.no_upload else HFStore()
    references = {} if store is None else {
        "assets": store.reference_run("pusht", "assets", args.assets_run),
        "probes": store.reference_run("pusht", "probes", args.probes_run),
        "evaluation_banks": {name: store.reference_run("pusht-banks", "banks", path)
                             for name, path in bank_paths.items()},
    }
    device = "cuda"
    model = load_model(device)
    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    process = load_scalers(args.assets_run / "scalers.npz")
    splits = json.loads((args.assets_run / "splits.json").read_text())
    probe, _ = load_probe(args.probes_run / f"{probe_name}.pt", device)
    imaginer = Imaginer(model, process, device)
    rng = np.random.default_rng(args.seed)

    # ---- data -------------------------------------------------------------------------
    adapt_dir, adapt_ledger = build_adapt_bank(args.adapt_roots, args.adapt_tapes, args.seed, splits, model, process, device, study_dir=study_dir)
    split_report = validate_bank_splits({**bank_paths, "adapt": adapt_dir})
    write_manifest(adapt_dir, build_manifest(run_id=f"{run_id}-adapt-bank", kind="adaptation-bank",
        seeds={"seed": args.seed}, data={"upstream": references, "split_integrity": split_report,
        "source_family_protocol": SOURCE_FAMILY_PROTOCOL}, costs=adapt_ledger.to_dict()))
    (adapt_dir / "config.yaml").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    (adapt_dir / "README.md").write_text("# Push-T adaptation bank\n\nComplete queried branches including invalid observations and censored outcomes. Only valid complete observations enter training.\n")
    if store is not None:
        store.upload_run("pusht-banks", "banks", adapt_dir, run_id=f"{run_id}-adapt-bank")
        references["adaptation_bank"] = store.reference_run("pusht-banks", "banks", adapt_dir)
    adapt_bank = Bank(adapt_dir)
    print(f"[e2] adapt bank: {len(adapt_bank)} branches, {len(adapt_bank.roots)} roots, charged {adapt_bank.ledger}")
    cache_dir = study_dir / "clips"
    cache_dir.mkdir(exist_ok=True)
    if (cache_dir / "replay.npz").is_file():
        replay = ClipSet.load(cache_dir / "replay.npz")
        retention = ClipSet.load(cache_dir / "retention.npz")
    else:
        replay = expert_replay_clips(model, process, H5_PATH, splits["roles"]["replay"], args.replay_clips, rng, device=device)
        retention = expert_replay_clips(model, process, H5_PATH, splits["roles"]["retention"], args.retention_clips, rng, device=device)
        replay.save(cache_dir / "replay.npz")
        retention.save(cache_dir / "retention.npz")
    training_indices = adapt_bank.valid_training_indices()
    context_ledger = StepLedger()
    acquired = metered_branch_clips(imaginer, adapt_bank, training_indices, context_ledger)
    acquired.save(cache_dir / "adapt_branches.npz")
    print(f"[e2] clips: replay {len(replay)}, retention {len(retention)}, acquired {len(acquired)}")
    write_manifest(cache_dir, build_manifest(run_id=f"{run_id}-clips", kind="cached-targets",
        seeds={"seed": args.seed}, data={"upstream": references,
        "training_branch_indices": training_indices.tolist(),
        "files_sha256": {p.name: file_sha256(p) for p in cache_dir.glob("*.npz")}},
        costs=context_ledger.to_dict()))
    (cache_dir / "config.yaml").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    (cache_dir / "README.md").write_text("# Cached frozen targets\n\nExact replay, retention and acquired latent clips used by this E2 run. Scalers and frozen encoder are pinned in the manifest.\n")
    if store is not None:
        store.upload_run("pusht", "assets", cache_dir, run_id=f"{run_id}-clips")
        references["cached_targets"] = store.reference_run("pusht", "assets", cache_dir)

    banks = {n: (Bank(args.banks_dir / n), load_layouts(args.banks_dir / n / "layouts.json")[0]) for n in ("dev", "test", "stress")}
    evaluation_ledger = StepLedger()
    caches = {n: metered_root_cache(imaginer, b, evaluation_ledger, n + "_history")
              for n, (b, _) in banks.items()}

    def evaluate(m, name, correction=None):
        im = Imaginer(m, process, device)
        caches[name].imaginer = im
        rows = evaluate_model_on_bank(name, banks[name][0], banks[name][1], im, probe, correction=correction, cache=caches[name])
        caches[name].imaginer = imaginer
        return rows

    # ---- baseline (no update) ---------------------------------------------------------
    base_rows = {n: evaluate(model, n) for n in banks}
    target_ar = float(np.mean([r["cmin_imagined"] >= 0 for r in base_rows["dev"]]))
    m_base = margin_for_acceptance(np.array([r["cmin_imagined"] for r in base_rows["dev"]]), target_ar)
    base_ret = retention_metrics(model, Imaginer, process, probe, retention, device)
    print(f"[e2] no-update dev: {json.dumps(fsa(np.array([r['cmin_imagined'] for r in base_rows['dev']]), row_outcomes(base_rows['dev'])[0], m_base, censored=row_outcomes(base_rows['dev'])[1]))} target AR {target_ar:.3f}")

    # ---- recipe selection on dev -------------------------------------------------------
    grid = []
    for modules in (["predictor_side"] if args.quick else ["predictor_side", "predictor_only"]):
        for loss in ["teacher_forced", "tf_plus_rollout"]:
            for lr in ([5e-5] if args.quick else [2e-5, 1e-4]):
                for steps in ([1000] if args.quick else [1000, 3000]):
                    grid.append(AdaptConfig(modules=modules, loss=loss, lr=lr, steps=steps, seed=args.seed))
    dev_results = []
    for cfg in grid:
        m_adapted, log = adapt(base_sd, model, acquired, replay, cfg, device=device, verbose=False)
        rows = evaluate(m_adapted, "dev")
        m_this = margin_for_acceptance(np.array([r["cmin_imagined"] for r in rows]), target_ar)
        s = eval_rows_summary(rows, m_this)
        ret = retention_metrics(m_adapted, Imaginer, process, probe, retention, device)
        rec = {"cfg": cfg.to_dict(), "dev": s, "retention": ret, "margin_matched": m_this, "train_time_s": log["wall_clock_s"], "final_loss": log["steps"][-1]["loss"]}
        dev_results.append(rec)
        print(f"[e2] recipe {cfg.modules}/{cfg.loss}/lr{cfg.lr}/{cfg.steps}: dev FSA@matched {s['at_matched']['fsa']:.3f} (AR {s['at_matched']['acceptance_rate']:.2f}) "
              f"retention tf {ret['latent_mse_tf']:.4f} (base {base_ret['latent_mse_tf']:.4f}) motion ratio {s['ordinary_motion'].get('disp_ratio', float('nan')):.2f} [{log['wall_clock_s']:.0f}s]")
    base_dev = eval_rows_summary(base_rows["dev"], m_base)
    ok = [r for r in dev_results if r["retention"]["latent_mse_tf"] <= 1.15 * base_ret["latent_mse_tf"] and not np.isnan(r["dev"]["at_matched"]["fsa"])]
    chosen = min(ok or dev_results, key=lambda r: r["dev"]["at_matched"]["fsa"])
    cfg = AdaptConfig(**chosen["cfg"])
    print(f"[e2] chosen recipe: {cfg.to_dict()} (dev FSA {chosen['dev']['at_matched']['fsa']:.3f} vs no-update {base_dev['at_matched']['fsa']:.3f})")

    # ---- final: fixed recipe, controls, test + stress ------------------------------------
    m_adapted, log = adapt(base_sd, model, acquired, replay, cfg, device=device, verbose=True)
    from helpers.pushtRetention import make_cases, evaluate_retention, compare_retention

    retention_cases = make_cases(splits, seed=args.seed + 10000, n=args.retention_episodes)
    case_text = json.dumps([root.to_dict() for root in retention_cases]) + "\n"
    case_sha256 = hashlib.sha256(case_text.encode()).hexdigest()
    goal_retention_base = evaluate_retention(model, process, retention_cases,
        n_blocks=args.retention_blocks, device=device)
    goal_retention_adapted = evaluate_retention(m_adapted, process, retention_cases,
        n_blocks=args.retention_blocks, device=device)
    goal_retention = {"no_update": goal_retention_base, "adapted": goal_retention_adapted,
        "comparison": compare_retention(goal_retention_base, goal_retention_adapted),
        "n_episodes": len(retention_cases), "n_blocks": args.retention_blocks,
        "case_file_sha256": case_sha256}
    # readout-only correction on the same new data (imagined latents of the adapt bank vs true endpoint poses)
    cache_a = metered_root_cache(imaginer, adapt_bank, context_ledger, "readout_correction_history")
    Zi, Yt = [], []
    for ri in range(len(adapt_bank.roots)):
        idx = np.intersect1d(adapt_bank.indices_for_root(ri), training_indices)
        if not len(idx):
            continue
        zh, hb, _ = cache_a.get(ri)
        tapes = np.stack([adapt_bank.h5["tape"][int(j)].astype(np.float64) for j in idx])
        Zi.append(imaginer.rollout(zh, hb, tapes).cpu().numpy())
        Yt.append(np.stack([pose_to_target(adapt_bank.h5["states"][int(j)][ACTION_BLOCK::ACTION_BLOCK]) for j in idx]))
    correction = fit_readout_correction(probe, np.concatenate(Zi), np.concatenate(Yt), device=device, seed=args.seed)
    arms = {"no_update": (model, None), "adapted": (m_adapted, None), "readout_correction": (model, correction)}
    report = {"run_id": run_id, "development_feasibility": feasibility, "goal_retention_cases_sha256": case_sha256, "goal_retention_cases": str(results / "goal_retention_cases.json"), "probe_name": probe_name, "decomposition_report": str(args.decomposition_report.resolve()), "assets_run": str(args.assets_run.resolve()), "probes_run": str(args.probes_run.resolve()), "clips_dir": str(cache_dir), "banks_dir": str(args.banks_dir.resolve()), "split_integrity": split_report, "goal_retention": goal_retention, "upstream": references, "target_acceptance_rate_dev": target_ar, "recipe_grid": dev_results, "chosen_recipe": cfg.to_dict(),
              "no_update_retention": base_ret, "adapted_retention": retention_metrics(m_adapted, Imaginer, process, probe, retention, device),
              "adapt_bank": {"n_branches": len(adapt_bank), "n_roots": len(adapt_bank.roots), "ledger": adapt_bank.ledger}, "train_log": log, "arms": {}}
    rows_by_arm = {}
    for arm, (mdl, corr) in arms.items():
        rows_by_arm[arm] = {n: (base_rows[n] if arm == "no_update" else evaluate(mdl, n, corr)) for n in banks}
        m_arm = margin_for_acceptance(np.array([r["cmin_imagined"] for r in rows_by_arm[arm]["dev"]]), target_ar)
        report["arms"][arm] = {"margin_matched_dev": m_arm, **{n: eval_rows_summary(rows_by_arm[arm][n], m_arm) for n in banks}}
        for n in ("test", "stress"):
            report["arms"][arm][n]["decision_table_matched"] = decision_table(rows_by_arm[arm][n], m_arm, sources=["imagined"])
    # fixed-margin control: the margin the UNADAPTED model needs on dev to reach the adapted model's dev FSA
    adapted_dev_fsa = report["arms"]["adapted"]["dev"]["at_matched"]["fsa"]
    cd = np.array([r["cmin_imagined"] for r in base_rows["dev"]])
    ud, ud_censored = row_outcomes(base_rows["dev"])
    m_fixed = next((m for m in np.linspace(-20, 80, 201) if (np.isfinite(fsa(cd, ud, m, censored=ud_censored)["fsa"]) and fsa(cd, ud, m, censored=ud_censored)["fsa"] <= adapted_dev_fsa)), 80.0)
    report["arms"]["fixed_margin"] = {"margin": float(m_fixed), **{n: eval_rows_summary(base_rows[n], m_fixed) for n in banks}}
    # paired differences on the same test tapes (root bootstrap)
    for n in ("test", "stress"):
        a = rows_by_arm["no_update"][n]
        b = rows_by_arm["adapted"][n]
        assert [r["branch"] for r in a] == [r["branch"] for r in b]
        ma, mb = report["arms"]["no_update"]["margin_matched_dev"], report["arms"]["adapted"]["margin_matched_dev"]
        report["arms"]["adapted"][n]["paired_fsa_vs_no_update"] = {
            "no_update": fsa(np.array([r["cmin_imagined"] for r in a]), row_outcomes(a)[0], ma, censored=row_outcomes(a)[1])["fsa"],
            "adapted": fsa(np.array([r["cmin_imagined"] for r in b]), row_outcomes(b)[0], mb, censored=row_outcomes(b)[1])["fsa"]}
        # bootstrap over roots of the paired difference at each arm's own matched margin
        root = np.array([r["root"] for r in a])
        ca, cb, u = np.array([r["cmin_imagined"] for r in a]), np.array([r["cmin_imagined"] for r in b]), row_outcomes(a)[0]
        censored = row_outcomes(a)[1]
        report["arms"]["adapted"][n]["paired_fsa_diff_ci"] = cluster_bootstrap(lambda ca, cb, u, censored: fsa(cb, u, mb, censored=censored)["fsa"] - fsa(ca, u, ma, censored=censored)["fsa"], root, n_boot=1000, ca=ca, cb=cb, u=u, censored=censored)
    # gate
    base_t, adap_t, ro_t = (report["arms"][k]["test"]["at_matched"]["fsa"] for k in ("no_update", "adapted", "readout_correction"))
    rel = (base_t - adap_t) / base_t if base_t else float("nan")
    clip_ret_ok = report["adapted_retention"]["latent_mse_tf"] <= 1.15 * base_ret["latent_mse_tf"]
    goal_ret_ok = bool(goal_retention["comparison"]["passes"])
    retention_complete = bool(goal_retention["comparison"]["complete"])
    ret_ok = clip_ret_ok and goal_ret_ok and retention_complete
    report["gate"] = {"test_fsa_no_update": base_t, "test_fsa_adapted": adap_t, "test_fsa_readout_correction": ro_t, "relative_reduction": rel,
                      "beats_readout_correction": bool(adap_t < ro_t), "retention_within_15pct": bool(clip_ret_ok), "goal_retention_passes": goal_ret_ok, "retention_complete": retention_complete, "passes": bool(rel >= 0.25 and adap_t < ro_t and ret_ok)}
    report["context_ledger"] = context_ledger.to_dict()
    report["evaluation_ledger"] = evaluation_ledger.to_dict()
    report["simulator_costs"] = combined_simulator_ledger(adapt_bank.ledger,
        context_ledger.to_dict(), evaluation_ledger.to_dict(),
        goal_retention_base["ledger"], goal_retention_adapted["ledger"])
    report["simulator_cost_accounting"] = (
        "All actual E2 collection, context-cache replay and goal-retention steps; "
        "pre-existing evaluation-bank generation costs remain in upstream manifests. "
        "Repeated uses of a filled latent cache add no simulator steps.")
    print(f"[e2] GATE: {json.dumps(report['gate'])}")

    # ---- save weights, upload, write ---------------------------------------------------
    run_dir.mkdir(parents=True, exist_ok=False)
    for destination in (run_dir, results):
        (destination / "goal_retention.json").write_text(json.dumps(goal_retention, default=float, indent=2) + "\n")
        (destination / "goal_retention_cases.json").write_text(case_text)
    torch.save(predictor_side_state(m_adapted, cfg.modules), run_dir / "weights.pt")
    torch.save(correction.state_dict(), run_dir / "readout_correction.pt")
    (run_dir / "config.yaml").write_text(json.dumps(cfg.to_dict(), indent=1) + "\n")
    manifest = build_manifest(run_id=run_id, kind="adapted", seeds={"seed": args.seed},
                              data={"development_feasibility": feasibility, "upstream": references, "decomposition_report_sha256": file_sha256(args.decomposition_report), "source_family_protocol": SOURCE_FAMILY_PROTOCOL, "assets_run": str(args.assets_run), "probes_run": str(args.probes_run),
                                    "assets_manifest_sha256": file_sha256(args.assets_run / "manifest.json"),
                                    "probe_manifest_sha256": file_sha256(args.probes_run / "manifest.json"),
                                    "bank_manifests": {name: json.loads((path / "manifest.json").read_text()) for name, path in bank_paths.items()},
                                    "split_integrity": split_report, "adapt_bank_branches": len(adapt_bank),
                                    "training_branches": len(training_indices), "replay_clips": len(replay)},
                              costs=report["simulator_costs"], metrics=report["gate"], started_at=t_start)
    write_manifest(run_dir, manifest)
    (run_dir / "README.md").write_text(f"# {run_id}\n\nPredictor-side adapted Push-T LeWM (E2). Only the trained modules ({cfg.modules}) are stored; load over the released weights. See manifest.json.\n")
    if not args.no_upload:
        report["hf_revision"] = store.upload_run("pusht", "adapted", run_dir, run_id=run_id)
        report["hf_repo"] = store.repo_id("pusht")
    report["wall_clock_s"] = time.time() - t_start
    report["manifest"] = manifest
    write_manifest(results, {**manifest, "hf_revision": report.get("hf_revision")})
    (results / "repair.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    np.savez_compressed(results / "repair_rows.npz", rows=json.dumps(rows_by_arm))
    from helpers.pushtRepairReport import write_repair_report

    write_repair_report(results / "repair.json", results / "repair_rows.npz",
                        args.banks_dir, results / "paired-report")
    print(f"[e2] done in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
