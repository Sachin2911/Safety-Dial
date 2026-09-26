#!/usr/bin/env python3
"""E2: can extra experience repair the diagnosed error? (pushT.md)

1. Build the adaptation bank (about 512 branches from adaptation-split roots, contact-rich
   and ordinary) unless it exists; every simulated candidate is charged.
2. Cache replay clips (expert, replay split) and retention clips (expert, retention split).
3. Choose the recipe on the DEVELOPMENT bank only: loss, learning rate, steps, modules.
4. Evaluate the fixed recipe on the test and stress banks against the controls: no update,
   a fixed margin calibrated on development data, and a readout-only correction trained
   on the same new data. Report retention and ordinary motion.

Outputs docs/mainPlan/results/e2/repair.json and figures; adapted weights to
<ns>/safetydial-pusht:adapted/<run_id>.

    uv run python experiments/scripts/pusht_e2_repair.py
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

from helpers.branchBank import Bank, BankWriter, build_root, execute_proposals, expert_pairs, propose  # noqa: E402
from helpers.decomposition import RootLatentCache, decision_table, evaluate_model_on_bank, ordinary_motion  # noqa: E402
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
from helpers.pushtLayouts import load_layouts  # noqa: E402
from helpers.pushtReplay import StepLedger, make_env  # noqa: E402
from helpers.runManifest import build_manifest, make_run_id, write_manifest  # noqa: E402

ASSETS_RUN = REPO_ROOT / "runs" / "pusht-assets-20260926-1"
PROBES_RUN = REPO_ROOT / "runs" / "pusht-probes-20260926-1"
STUDY = REPO_ROOT / "data" / "study" / "pusht"
RESULTS = REPO_ROOT / "docs" / "mainPlan" / "results" / "e2"
KS = [0, 2, 4, 6]


def build_adapt_bank(n_roots, n_tapes, seed, splits, model, process, device) -> tuple[Path, StepLedger]:
    bank_dir = STUDY / "adapt"
    ledger = StepLedger()
    if (bank_dir / "roots.json").is_file():
        return bank_dir, ledger
    env = make_env()
    planner = NominalPlanner(model, process, device)
    rng = np.random.default_rng(seed)
    pairs = expert_pairs(H5_PATH, splits["roles"]["reserve"][:3000], rng, n_roots * 2)
    writer = BankWriter(bank_dir)
    t0 = time.time()
    for i in range(n_roots):
        k = KS[i % len(KS)]
        root, ctx, _ = build_root(env, planner, pairs[i], seed=seed + i, k=k, root_id=f"adapt-r{i:03d}", ledger=ledger)
        props, rej = propose(rng, root, ctx, n_random=n_tapes - 4, sigmas=(0.05, 0.1, 0.2), n_stress=3)
        ledger.add("proposals_rejected_by_arena", 0, branches=rej)
        br = execute_proposals(env, root, props, ledger=ledger)
        writer.add_root(root)
        writer.add_branches(br)
        if (i + 1) % 16 == 0:
            print(f"[e2] adapt bank {i + 1}/{n_roots} roots {time.time() - t0:.0f}s")
    writer.finish(ledger, {"bank": "adapt", "seed": seed})
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
    u = np.array([r["cmin_dense"] < 0 for r in rows])
    c = np.array([r["cmin_imagined"] for r in rows])
    ct = np.array([r["cmin_dense"] for r in rows])
    return {"at_m0": fsa(c, u, m_zero), "at_matched": fsa(c, u, m_matched), "auc_dial": auc_dial(c, u), "clearance_error": clearance_error_stats(c, ct),
            "ordinary_motion": ordinary_motion(rows), "n": int(len(rows)), "n_unsafe": int(u.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapt-roots", type=int, default=64)
    ap.add_argument("--adapt-tapes", type=int, default=8)
    ap.add_argument("--replay-clips", type=int, default=2000)
    ap.add_argument("--retention-clips", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--quick", action="store_true", help="smaller recipe grid")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    t_start = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id("pusht", "adapt", "e2", n=args.n)
    device = "cuda"
    model = load_model(device)
    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    process = load_scalers(ASSETS_RUN / "scalers.npz")
    splits = json.loads((ASSETS_RUN / "splits.json").read_text())
    probe, _ = load_probe(PROBES_RUN / "block_pose_mlp.pt", device)
    imaginer = Imaginer(model, process, device)
    rng = np.random.default_rng(args.seed)

    # ---- data -------------------------------------------------------------------------
    adapt_dir, adapt_ledger = build_adapt_bank(args.adapt_roots, args.adapt_tapes, args.seed, splits, model, process, device)
    adapt_bank = Bank(adapt_dir)
    print(f"[e2] adapt bank: {len(adapt_bank)} branches, {len(adapt_bank.roots)} roots, charged {adapt_bank.ledger}")
    cache_dir = STUDY / "clips"
    cache_dir.mkdir(exist_ok=True)
    if (cache_dir / "replay.npz").is_file():
        replay = ClipSet.load(cache_dir / "replay.npz")
        retention = ClipSet.load(cache_dir / "retention.npz")
    else:
        replay = expert_replay_clips(model, process, H5_PATH, splits["roles"]["replay"], args.replay_clips, rng, device=device)
        retention = expert_replay_clips(model, process, H5_PATH, splits["roles"]["retention"], args.retention_clips, rng, device=device)
        replay.save(cache_dir / "replay.npz")
        retention.save(cache_dir / "retention.npz")
    acquired = branch_clips(imaginer, adapt_bank, range(len(adapt_bank)))
    acquired.save(STUDY / "clips" / "adapt_branches.npz")
    print(f"[e2] clips: replay {len(replay)}, retention {len(retention)}, acquired {len(acquired)}")

    banks = {n: (Bank(STUDY / n), load_layouts(STUDY / n / "layouts.json")[0]) for n in ("dev", "test", "stress")}
    caches = {n: RootLatentCache(imaginer, b) for n, (b, _) in banks.items()}

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
    print(f"[e2] no-update dev: {json.dumps(fsa(np.array([r['cmin_imagined'] for r in base_rows['dev']]), np.array([r['cmin_dense'] < 0 for r in base_rows['dev']]), m_base))} target AR {target_ar:.3f}")

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
    # readout-only correction on the same new data (imagined latents of the adapt bank vs true endpoint poses)
    cache_a = RootLatentCache(imaginer, adapt_bank)
    Zi, Yt = [], []
    for ri in range(len(adapt_bank.roots)):
        idx = adapt_bank.indices_for_root(ri)
        zh, hb, _ = cache_a.get(ri)
        tapes = np.stack([adapt_bank.h5["tape"][int(j)].astype(np.float64) for j in idx])
        Zi.append(imaginer.rollout(zh, hb, tapes).cpu().numpy())
        Yt.append(np.stack([pose_to_target(adapt_bank.h5["states"][int(j)][ACTION_BLOCK::ACTION_BLOCK]) for j in idx]))
    correction = fit_readout_correction(probe, np.concatenate(Zi), np.concatenate(Yt), device=device, seed=args.seed)
    arms = {"no_update": (model, None), "adapted": (m_adapted, None), "readout_correction": (model, correction)}
    report = {"run_id": run_id, "target_acceptance_rate_dev": target_ar, "recipe_grid": dev_results, "chosen_recipe": cfg.to_dict(),
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
    ud = np.array([r["cmin_dense"] < 0 for r in base_rows["dev"]])
    m_fixed = next((m for m in np.linspace(-20, 80, 201) if (fsa(cd, ud, m)["fsa"] <= adapted_dev_fsa or np.isnan(fsa(cd, ud, m)["fsa"]))), 80.0)
    report["arms"]["fixed_margin"] = {"margin": float(m_fixed), **{n: eval_rows_summary(base_rows[n], m_fixed) for n in banks}}
    # paired differences on the same test tapes (root bootstrap)
    for n in ("test", "stress"):
        a = rows_by_arm["no_update"][n]
        b = rows_by_arm["adapted"][n]
        assert [r["branch"] for r in a] == [r["branch"] for r in b]
        ma, mb = report["arms"]["no_update"]["margin_matched_dev"], report["arms"]["adapted"]["margin_matched_dev"]
        report["arms"]["adapted"][n]["paired_fsa_vs_no_update"] = {
            "no_update": fsa(np.array([r["cmin_imagined"] for r in a]), np.array([r["cmin_dense"] < 0 for r in a]), ma)["fsa"],
            "adapted": fsa(np.array([r["cmin_imagined"] for r in b]), np.array([r["cmin_dense"] < 0 for r in b]), mb)["fsa"]}
        # bootstrap over roots of the paired difference at each arm's own matched margin
        root = np.array([r["root"] for r in a])
        ca, cb, u = np.array([r["cmin_imagined"] for r in a]), np.array([r["cmin_imagined"] for r in b]), np.array([r["cmin_dense"] < 0 for r in a])
        report["arms"]["adapted"][n]["paired_fsa_diff_ci"] = cluster_bootstrap(lambda ca, cb, u: fsa(cb, u, mb)["fsa"] - fsa(ca, u, ma)["fsa"], root, n_boot=1000, ca=ca, cb=cb, u=u)
    # gate
    base_t, adap_t, ro_t = (report["arms"][k]["test"]["at_matched"]["fsa"] for k in ("no_update", "adapted", "readout_correction"))
    rel = (base_t - adap_t) / base_t if base_t else float("nan")
    ret_ok = report["adapted_retention"]["latent_mse_tf"] <= 1.15 * base_ret["latent_mse_tf"]
    report["gate"] = {"test_fsa_no_update": base_t, "test_fsa_adapted": adap_t, "test_fsa_readout_correction": ro_t, "relative_reduction": rel,
                      "beats_readout_correction": bool(adap_t < ro_t), "retention_within_15pct": bool(ret_ok), "passes": bool(rel >= 0.25 and adap_t < ro_t and ret_ok)}
    print(f"[e2] GATE: {json.dumps(report['gate'])}")

    # ---- save weights, upload, write ---------------------------------------------------
    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(predictor_side_state(m_adapted, cfg.modules), run_dir / "weights.pt")
    torch.save(correction.state_dict(), run_dir / "readout_correction.pt")
    (run_dir / "config.yaml").write_text(json.dumps(cfg.to_dict(), indent=1) + "\n")
    manifest = build_manifest(run_id=run_id, kind="adapted", seeds={"seed": args.seed},
                              data={"assets_run": ASSETS_RUN.name, "probes_run": PROBES_RUN.name, "adapt_bank_branches": len(adapt_bank), "replay_clips": len(replay)},
                              costs=adapt_bank.ledger, metrics=report["gate"], started_at=t_start)
    write_manifest(run_dir, manifest)
    (run_dir / "README.md").write_text(f"# {run_id}\n\nPredictor-side adapted Push-T LeWM (E2). Only the trained modules ({cfg.modules}) are stored; load over the released weights. See manifest.json.\n")
    if not args.no_upload:
        store = HFStore()
        report["hf_revision"] = store.upload_run("pusht", "adapted", run_dir, run_id=run_id)
        report["hf_repo"] = store.repo_id("pusht")
    report["wall_clock_s"] = time.time() - t_start
    report["manifest"] = manifest
    (RESULTS / "repair.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    np.savez_compressed(RESULTS / "repair_rows.npz", rows=json.dumps(rows_by_arm))
    print(f"[e2] done in {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
