#!/usr/bin/env python3
"""S4 prospective acquisition with immutable source roles and charged simulator costs.

The old S4 entry point delegates here. Candidates are stored before selection and only
selected tapes reach the simulator. Evaluation banks remain common across arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers.threads import apply_torch, pin_threads  # noqa: E402

pin_threads()
import numpy as np  # noqa: E402
import torch  # noqa: E402

apply_torch()
from helpers.dialMetrics import auc_dial, clearance_error_stats, cluster_bootstrap, fsa, margin_for_acceptance  # noqa: E402
from helpers.hfStore import HFStore  # noqa: E402
from helpers.locoData import RenderContext  # noqa: E402
from helpers.locoEnv import make_loco_env  # noqa: E402
from helpers.poseProbes import load_probe  # noqa: E402
from helpers.predictorAdapt import AdaptConfig, ClipSet, adapt, predictor_side_state  # noqa: E402
from helpers.runManifest import build_manifest, file_sha256, make_run_id, write_manifest  # noqa: E402
from helpers.walkerBank import HORIZON_STEPS, WalkerBank, WalkerBankWriter, build_walker_bank, dataset_identity, interp_steps, load_verified_bank, sample_roots, source_generation_steps  # noqa: E402
from helpers.walkerExperience import acquisition_files
from helpers.walkerLewm import FRAMESKIP, HISTORY, WalkerImaginer, load_walker_model  # noqa: E402
from helpers.walkerRules import HORIZON_BLOCKS, propose_tapes, rule_unsafe  # noqa: E402
from helpers.walkerReporting import decomposition_report, merge_frozen_sources  # noqa: E402
from helpers.walkerValidation import decomposition_gate, episode_role, load_source_split, upload_reference, verify_render_fingerprint  # noqa: E402
from walker_s4_study import RootCache, analyse, clips_from_bank, replay_clips, rule_clearance, table  # noqa: E402


def save_evaluation_rows(destinations, filename, rows):
    """Mirror already-computed rows byte-for-byte, without inference or RNG use.

    List order and numeric values are retained, including undefined float values.
    Every report/manifest reference is relative to each of the three bundle roots.
    Existing artifacts are never replaced, even if one destination is a reused bank.
    """
    if Path(filename).name != filename or not filename.endswith(".json"):
        raise ValueError("Evaluation-row artifact must be a single JSON filename")
    paths = [Path(destination) / filename for destination in destinations]
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Evaluation-row destinations must be nonempty and distinct")
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Preserving existing evaluation rows: {path}")
        if not path.parent.is_dir():
            raise FileNotFoundError(path.parent)
    payload = (json.dumps(rows, sort_keys=True, separators=(",", ":"), default=float) + "\n").encode()
    for path in paths:
        with path.open("xb") as output:
            output.write(payload)
    return {"file": filename, "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload)}


def balanced_pick(candidates, need, scores):
    """Choose before execution, cycling roots and motion regimes at every round."""
    groups = {}
    for candidate in candidates:
        group = candidate["root_index"]
        groups.setdefault(group, []).append(candidate)
    for group in groups.values():
        # Interleave motion regimes within a root while preserving score order per regime.
        regimes = {}
        for item in group:
            regimes.setdefault(item["kind"], []).append(item)
        for regime in regimes.values():
            regime.sort(key=lambda c: (scores[c["id"]], c["id"]))
        interleaved = []
        while regimes:
            for name in sorted(regimes, key=lambda k: scores[regimes[k][0]["id"]]):
                interleaved.append(regimes[name].pop(0))
                if not regimes[name]:
                    del regimes[name]
        group[:] = interleaved
    picked = []
    while len(picked) < need and groups:
        order = sorted(groups, key=lambda k: (scores[groups[k][0]["id"]], k))
        for key in order:
            picked.append(groups[key].pop(0))
            if not groups[key]:
                del groups[key]
            if len(picked) == need:
                break
    if len(picked) != need:
        raise ValueError(f"Requested {need} candidates but only {len(picked)} remain")
    return picked


def score_candidates(candidates, roots, im, probe, ctx, rules, margins, bands):
    """Only roots and action tapes enter this function; future truth is unavailable."""
    cache = RootCache(im, ctx)
    scores = {}
    for ri in sorted({c["root_index"] for c in candidates}):
        root = roots[ri]
        here = [c for c in candidates if c["root_index"] == ri]
        tapes = np.asarray([c["tape"] for c in here], dtype=np.float64)
        z_hist, _ = cache.get(root)
        imagined = im.rollout(z_hist, root.history_actions,
                              tapes.reshape(len(here), HORIZON_BLOCKS, FRAMESKIP, 6))
        n, k, d = imagined.shape
        predicted = probe.predict(imagined.reshape(-1, d)).reshape(n, k, 3)
        y0 = probe.predict(z_hist[-1:])[0]
        for item, future in zip(here, predicted, strict=True):
            trajectory = np.vstack([y0, future])
            clearance = {r: float(rule_clearance(r, interp_steps(trajectory[:, 0]),
                                                interp_steps(trajectory[:, 1]),
                                                interp_steps(trajectory[:, 2])).min())
                         for r in rules}
            scores[item["id"]] = min(abs(clearance[r] - margins[r]) / bands[r] for r in rules)
    return scores


def execute_selected(bank_dir, roots, chosen, env, metadata):
    """Only selected tapes reach a simulator; exact IDs are retained for S5."""
    writer = WalkerBankWriter(bank_dir)
    ledger = {"selected_ids": [c["id"] for c in chosen], "attempted_steps": 0, "status": "running"}
    ledger_path = Path(bank_dir) / "execution_ledger.json"
    ledger_path.write_text(json.dumps(ledger) + "\n")
    def charge_step():
        ledger["attempted_steps"] += 1
    try:
        for candidate in chosen:
            ledger["pending_candidate_id"] = candidate["id"]
            ledger["pending_step_upper_bound"] = HORIZON_STEPS
            ledger_path.write_text(json.dumps(ledger) + "\n")
            root = roots[candidate["root_index"]]
            writer.add_branches(root, [(np.asarray(candidate["tape"], dtype=np.float32), candidate["kind"],
                                        {**candidate["params"], "candidate_id": candidate["id"]})], env, on_step=charge_step)
            ledger["pending_step_upper_bound"] = 0
            ledger_path.write_text(json.dumps(ledger) + "\n")
    except Exception:
        ledger["status"] = "failed"
        ledger_path.write_text(json.dumps(ledger) + "\n")
        if hasattr(writer, "h5"):
            writer.h5.close()
        raise
    ledger["status"] = "complete"
    ledger_path.write_text(json.dumps(ledger) + "\n")
    writer.finish({**metadata, "selected_ids": [c["id"] for c in chosen],
                   "executed_steps": len(chosen) * HORIZON_STEPS})
    return WalkerBank(bank_dir)


def paired_difference(base_rows, rows, rule, base_margin, margin, *, n_boot=1000):
    """Updated minus reference FSA on identical tapes, clustered by source episode.

    Margins are supplied from development calibration; no final-test threshold is
    fitted here. Undefined full-sample estimates never acquire a directional CI
    merely because some bootstrap draws omit the unresolved observations.
    """
    if rule not in {"health", "speed"}:
        raise ValueError("Report health and speed rules separately")
    def identities(items):
        return [(r.get("bank"), r["root_id"], r["branch"]) for r in items]
    ids = identities(base_rows)
    if not ids or ids != identities(rows) or len(set(ids)) != len(ids):
        raise ValueError("Evaluation tapes are not uniquely paired in exact root/branch order")
    roots = np.asarray([r["root"] for r in rows])
    if not np.array_equal(roots, np.asarray([r["root"] for r in base_rows])):
        raise ValueError("Paired tapes have different source episode clusters")
    root_sources = {}
    for row in rows:
        if root_sources.setdefault(row["root_id"], row["root"]) != row["root"]:
            raise ValueError("One evaluation root belongs to multiple source episodes")
    truth = np.asarray([r[f"cmin_dense_{rule}"] for r in rows], dtype=float)
    base_truth = np.asarray([r[f"cmin_dense_{rule}"] for r in base_rows], dtype=float)
    if not np.isfinite(truth).all() or not np.array_equal(base_truth, truth):
        raise ValueError("Paired tapes must have identical finite dense truth")
    def censoring(items):
        flags = [r.get("censored", False) for r in items]
        if any(not isinstance(flag, (bool, np.bool_)) for flag in flags):
            raise ValueError("Censoring flags must be booleans")
        return np.asarray(flags, dtype=bool)
    censored = censoring(rows)
    if not np.array_equal(censoring(base_rows), censored):
        raise ValueError("Paired tapes have different censoring masks")
    b = np.asarray([r[f"cmin_imagined_{rule}"] for r in base_rows], dtype=float)
    c = np.asarray([r[f"cmin_imagined_{rule}"] for r in rows], dtype=float)
    unsafe = rule_unsafe(rule, truth)
    reference = fsa(b, unsafe, base_margin, censored=censored)
    updated = fsa(c, unsafe, margin, censored=censored)
    point = updated["fsa"] - reference["fsa"]
    interval = cluster_bootstrap(
        lambda b, c, u, censored: fsa(c, u, margin, censored=censored)["fsa"]
        - fsa(b, u, base_margin, censored=censored)["fsa"],
        roots, n_boot=n_boot, b=b, c=c, u=unsafe, censored=censored)
    defined = bool(np.isfinite(point))
    interval["point"] = float(point)
    if not defined:
        # Finite conditional replicates cannot resolve an undefined estimand.
        interval["lo"] = interval["hi"] = float("nan")
    directional = defined and np.isfinite(interval["lo"]) and np.isfinite(interval["hi"])
    direction = ("decrease" if directional and interval["hi"] < 0 else
                 "increase" if directional and interval["lo"] > 0 else
                 "inconclusive" if defined else "undefined")
    reason = ("zero_acceptance" if not reference["n_accepted"] or not updated["n_accepted"] else
              "accepted_censored_futures" if not defined else None)
    return {**interval, "defined": defined, "direction": direction,
            "interval_excludes_zero": direction in {"decrease", "increase"},
            "undefined_reason": reason, "n_boot_requested": n_boot,
            "n_boot_undefined": n_boot - interval["n_boot"], "cluster_unit": "source_episode",
            "n_source_episodes": int(len(np.unique(roots))), "n_roots": len(root_sources),
            "reference": reference, "updated": updated}


def paired_acquisition_intervals(report, run_dir, *, n_boot=1000):
    """Summarize saved S4 rows only; no inference, simulator or margin calibration."""
    run_dir = Path(run_dir)
    result = {"contrast": "boundary minus random", "cluster_unit": "source_episode",
              "confidence_level": 0.95, "margin_policy": "fixed from each model's development bank",
              "negative_difference": "lower boundary-arm false-safe acceptance",
              "by_budget": {str(b): {"by_seed": {}} for b in report["planned_budgets"]}}
    for seed in report["acquisition_seeds"]:
        curves = {arm: report["adaptation"][arm][str(seed)] for arm in ("random", "boundary")}
        if any([p["budget"] for p in curve] != report["planned_budgets"] for curve in curves.values()):
            raise ValueError("Paired acquisition reporting requires every declared budget")
        for random, boundary in zip(curves["random"], curves["boundary"], strict=True):
            budget = random["budget"]
            if random["charged_steps"] != boundary["charged_steps"]:
                raise ValueError("Paired acquisition arms must have equal charged simulator costs")
            saved, hashes = {}, {}
            for arm, point in (("random", random), ("boundary", boundary)):
                path = run_dir / f"{arm}-s{seed}-b{budget}" / "evaluation_rows.json"
                identity = point["evaluation_rows"]
                if (identity["file"] != str(path.relative_to(run_dir))
                        or identity["sha256"] != file_sha256(path)):
                    raise ValueError("Saved paired evaluation rows differ from their checkpoint identity")
                saved[arm] = json.loads(path.read_text())
                hashes[arm] = identity["sha256"]
            entry = {"charged_steps": random["charged_steps"], "evaluation_rows_sha256": hashes}
            for bank in ("test", "stress"):
                entry[bank] = {}
                for rule in report["active_rules"]:
                    # The same development-calibrated margin must be recorded for
                    # every bank; refuse a threshold picked on final outcomes.
                    margins = {}
                    for arm, point in (("random", random), ("boundary", boundary)):
                        margins[arm] = point["eval"]["dev"][rule]["margin"]
                        if point["eval"][bank][rule]["margin"] != margins[arm]:
                            raise ValueError("Final-test margin differs from frozen development margin")
                    entry[bank][rule] = paired_difference(
                        saved["random"][bank], saved["boundary"][bank], rule,
                        margins["random"], margins["boundary"], n_boot=n_boot)
            result["by_budget"][str(budget)]["by_seed"][str(seed)] = entry
    return result


def charged_cost(generation_steps, seed_branches, acquired_branches):
    return {"root_generation_steps": int(generation_steps),
            "common_seed_steps": int(seed_branches * HORIZON_STEPS),
            "acquired_steps": int(acquired_branches * HORIZON_STEPS),
            "charged_steps": int(generation_steps + (seed_branches + acquired_branches) * HORIZON_STEPS)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--gate", default=None)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--bank-dir", default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--dev-roots", type=int, default=24)
    ap.add_argument("--test-roots", type=int, default=96)
    ap.add_argument("--acq-roots", type=int, default=96)
    ap.add_argument("--tapes", type=int, default=8)
    ap.add_argument("--seed-branches", type=int, default=128)
    ap.add_argument("--budgets", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--replay-clips", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--band", type=float, default=0.5)
    ap.add_argument("--speed-band", type=float, default=0.5)
    ap.add_argument("--min-dev-unsafe", type=int, default=10)
    ap.add_argument("--min-dev-false-safe", type=int, default=5)
    ap.add_argument("--min-dev-imagination", type=int, default=3)
    ap.add_argument("--min-imagination-share", type=float, default=0.25)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--run-id", default=None, help="explicit immutable run directory name")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        ap.error("run-id must be a single alphanumeric directory name")
    if (not args.budgets or sorted(set(args.budgets)) != args.budgets
            or args.budgets[0] <= 0 or args.seed_branches <= 0 or args.tapes < 2
            or args.band <= 0 or args.speed_band <= 0
            or min(args.min_dev_unsafe, args.min_dev_false_safe, args.min_dev_imagination) < 1
            or not 0 < args.min_imagination_share <= 1):
        ap.error("Require increasing positive budgets, paid seed data, at least 2 tapes and positive bands")
    t0 = time.time()
    data_dir = Path(args.data_dir)
    load_source_split(data_dir)
    source_id = dataset_identity(data_dir / "roots.h5")
    gate_path = Path(args.gate) if args.gate else Path(args.probes) / "gate.json"
    gate_report = json.loads(gate_path.read_text())
    if (not gate_report["gate"]["go"] or gate_report.get("role") != "development"
            or gate_report.get("source_identity") != source_id
            or gate_report.get("splits_sha256") != file_sha256(data_dir / "splits.json")
            or gate_report.get("model_sha256") != file_sha256(Path(args.model) / "weights.pt")):
        raise ValueError("S4 requires a passing development-only S3 gate for this exact model and source bank")
    if gate_report.get("model_files_sha256") != {name: file_sha256(Path(args.model) / name) for name in ("weights.pt", "config.json", "scalers.npz")}:
        raise ValueError("Model configuration or normalizers differ from S3")
    if gate_report.get("probe_files_sha256", {}).get("walker_mlp.pt") != file_sha256(Path(args.probes) / "walker_mlp.pt"):
        raise ValueError("The frozen probe differs from S3")
    rules = tuple(gate_report["gate"]["active_rules"])
    if not rules or "health" not in rules:
        raise ValueError("Walker S4 requires the health rule to pass")
    device = "cuda"
    run_id = args.run_id or make_run_id("walker2d", "s4", n=args.n)
    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    from omegaconf import OmegaConf

    OmegaConf.save(OmegaConf.create(vars(args)), run_dir / "config.yaml")
    results_dir = Path(args.results_dir) if args.results_dir else REPO_ROOT / "docs/mainPlan/results/s4" / run_id
    results_dir.mkdir(parents=True, exist_ok=False)
    bank_dir = Path(args.bank_dir) if args.bank_dir else data_dir / "s4" / run_id
    bank_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(vars(args)), bank_dir / "config.yaml")
    upstream_refs = {"model": upload_reference(args.model), "probes": upload_reference(args.probes), "data": upload_reference(data_dir)}
    model, scaler = load_walker_model(Path(args.model), device)
    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    im = WalkerImaginer(model, scaler, device)
    probe, _ = load_probe(Path(args.probes) / "walker_mlp.pt", device)
    ctx = RenderContext()
    verify_render_fingerprint(ctx, data_dir / "roots.h5")
    env = make_loco_env("Walker2d", "v1", render=True, terminate_when_unhealthy=False, six_tuple=True)
    env.reset(seed=0)
    banks = {}
    for name, n_roots, kind, role in (("dev", args.dev_roots, "representative", "development"),
                                     ("test", args.test_roots, "representative", "test"),
                                     ("stress", args.test_roots, "stress", "test")):
        provenance = {"source": source_id, "seed": 20261020, "kind": kind, "role": role,
                      "n_roots": n_roots, "n_tapes": args.tapes, "frameskip": FRAMESKIP,
                      "history": HISTORY, "horizon": HORIZON_BLOCKS, "schema": 2}
        path = bank_dir / name
        if (path / "roots.json").is_file():
            banks[name] = load_verified_bank(path, provenance)
        else:
            rng = np.random.default_rng(20261020)
            roots = sample_roots(data_dir / "roots.h5", rng, n_roots, kind=kind,
                                 episode_filter=lambda e: episode_role(e) == role, prefix=name)
            if len(roots) != n_roots:
                raise ValueError(f"{name}: requested {n_roots} source episodes, found {len(roots)}; collect more or explicitly reduce size")
            banks[name] = build_walker_bank(path, roots, rng, env, n_tapes=args.tapes,
                                           stress=kind == "stress", seed=20261020, provenance=provenance)
    acquisition_roots = sample_roots(data_dir / "roots.h5", np.random.default_rng(20261021),
                                     args.acq_roots, kind="representative",
                                     episode_filter=lambda e: episode_role(e) == "acquisition", prefix="acq")
    if len(acquisition_roots) != args.acq_roots:
        raise ValueError("Insufficient acquisition episodes for the declared root pool")
    if len(acquisition_roots) * args.tapes < args.seed_branches + max(args.budgets):
        raise ValueError("Candidate pool too small for seed plus requested acquisition budgets")
    generation_steps = source_generation_steps(data_dir / "roots.h5")
    (bank_dir / "acquisition_roots.json").write_text(json.dumps({"roots": [r.to_dict() for r in acquisition_roots],
                                                               "source_identity": source_id,
                                                               "generation_steps": generation_steps}) + "\n")
    base_rows = {"dev": analyse(banks["dev"], im, probe, ctx, rules=rules)}
    dev_gate = decomposition_gate(base_rows["dev"], rules, min_unsafe=args.min_dev_unsafe,
        min_false_safe=args.min_dev_false_safe, min_imagination=args.min_dev_imagination,
        min_share=args.min_imagination_share)
    for destination in (run_dir, results_dir):
        (destination / "decomposition_gate.json").write_text(json.dumps(dev_gate, indent=2) + "\n")
    if not dev_gate["go"]:
        # Keep all final-test measurements sealed. The prerequisite uses development only.
        diagnostic = {"run_id": run_id, "status": "diagnostic_stop", "gate": dev_gate,
                      "source_identity": source_id, "final_test_evaluated": False}
        diagnostic["saved_evaluation_rows"] = {"development": save_evaluation_rows(
            (run_dir, results_dir, bank_dir), "development_rows.json", base_rows["dev"])}
        manifest = build_manifest(run_id=run_id, kind="s4-diagnostic", data={"source": source_id,
            "model_sha256": gate_report["model_sha256"], "gate_sha256": file_sha256(gate_path),
            "saved_evaluation_rows": diagnostic["saved_evaluation_rows"]},
            metrics=dev_gate, upstream_revisions=upstream_refs, costs={"generated_evaluation_steps": sum(len(b) * HORIZON_STEPS for b in banks.values())}, started_at=t0)
        for destination in (run_dir, results_dir, bank_dir):
            write_manifest(destination, manifest)
        for destination in (run_dir, results_dir):
            (destination / "study.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        if not args.no_upload:
            (bank_dir / "diagnostic.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
            diagnostic["bank_hf_revision"] = HFStore().upload_run("walker2d-data", "banks", bank_dir, run_id=run_id)
            (results_dir / "study.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        for bank in banks.values():
            bank.h5.close()
        env.close()
        ctx.close()
        print("[s4] diagnostic stop: insufficient development imagination evidence", flush=True)
        return 2
    repair_rules = tuple(dev_gate["repair_rules"])
    for name in ("test", "stress"):
        base_rows[name] = analyse(banks[name], im, probe, ctx, rules=rules)
    targets = {r: float(np.mean([row[f"cmin_imagined_{r}"] >= 0 for row in base_rows["dev"]])) for r in rules}
    margins = {r: margin_for_acceptance(np.array([row[f"cmin_imagined_{r}"] for row in base_rows["dev"]]), targets[r]) for r in rules}
    report = {"run_id": run_id, "status": "running", "planned_budgets": args.budgets, "acquisition_seeds": args.seeds, "model": args.model, "probes": args.probes, "gate": str(gate_path),
              "active_rules": rules, "repair_rules": repair_rules, "development_gate": dev_gate, "source_identity": source_id, "acquisition_mode": "prospective",
              "target_acceptance": targets, "auc_acceptance_range": [0.2, 0.9], "decomposition": {}, "adaptation": {},
              "cost_policy": "All roots.h5 collection steps, paid seed branches, selected branches; evaluation separately"}
    report["saved_evaluation_rows"] = {"baseline_decomposition": save_evaluation_rows(
        (run_dir, results_dir, bank_dir), "baseline_decomposition_rows.json", base_rows)}
    sources = ["dense", "endpoint", "real_readout", "imagined"]
    for rule in rules:
        report["decomposition"][rule] = {name: table(rows, rule, 0.0, sources) for name, rows in base_rows.items()}

    report["decomposition_by_horizon_regime"] = {rule: {name: decomposition_report(rows, rule) for name, rows in base_rows.items()} for rule in rules}

    def evaluate(current):
        rows_by_bank = {name: analyse(bank, WalkerImaginer(current, scaler, device), probe, ctx,
                                     sources_all=False, rules=rules) for name, bank in banks.items()}
        local_margins = {r: margin_for_acceptance(np.array([row[f"cmin_imagined_{r}"] for row in rows_by_bank["dev"]]), targets[r]) for r in rules}
        metrics = {}
        for name, rows in rows_by_bank.items():
            metrics[name] = {"state_mae_by_block": np.mean([row["state_abs_error_by_block"] for row in rows], axis=0).tolist(),
                             "displacement_mae": float(np.mean([abs(row["predicted_displacement"] - row["true_displacement"]) for row in rows]))}
            ordinary = [row for row in rows if row["kind"] == "policy"]
            metrics[name]["ordinary_retention"] = {"n": len(ordinary),
                "state_mae_by_block": np.mean([row["state_abs_error_by_block"] for row in ordinary], axis=0).tolist() if ordinary else None,
                "displacement_mae": float(np.mean([abs(row["predicted_displacement"] - row["true_displacement"]) for row in ordinary])) if ordinary else None}
            for rule in rules:
                c = np.array([row[f"cmin_imagined_{rule}"] for row in rows])
                u = rule_unsafe(rule, np.array([row[f"cmin_dense_{rule}"] for row in rows]))
                metrics[name][rule] = {"margin": local_margins[rule], "at_matched": fsa(c, u, local_margins[rule]),
                                       "auc_dial": auc_dial(c, u, ar_lo=0.2, ar_hi=0.9), "auc_acceptance_range": [0.2, 0.9], "paired_fsa_difference": paired_difference(base_rows[name], rows, rule, margins[rule], local_margins[rule]),
                                       "clearance_error": clearance_error_stats(c, np.array([row[f"cmin_dense_{rule}"] for row in rows]))}
        return metrics, rows_by_bank

    report["adaptation"]["no_update"], no_update_rows = evaluate(model)
    report["saved_evaluation_rows"]["no_update_evaluation"] = save_evaluation_rows(
        (run_dir, results_dir, bank_dir), "no_update_evaluation_rows.json", no_update_rows)
    del no_update_rows
    replay_episodes = json.loads((Path(args.model) / "splits.json").read_text())["training"]
    replay = replay_clips(data_dir / "setA.h5", im, ctx, np.random.default_rng(20261020), args.replay_clips, scaler, episodes=replay_episodes)
    replay.save(run_dir / "replay_clips.npz")
    shutil.copyfile(run_dir / "replay_clips.npz", bank_dir / "replay_clips.npz")
    report["replay_source_training_episodes"] = replay_episodes
    cfg = AdaptConfig(modules="predictor_side", loss="teacher_forced", lr=5e-5, steps=args.steps, ctx=HISTORY)
    report["adaptation_recipe"] = cfg.to_dict()
    store = None if args.no_upload else HFStore()
    for seed in args.seeds:
        proposal_rng = np.random.default_rng(seed)
        candidates = []
        for ri, root in enumerate(acquisition_roots):
            for ci, (tape, kind, params) in enumerate(propose_tapes(proposal_rng, root, n_random=args.tapes - 1)):
                candidates.append({"id": f"s{seed}-r{ri}-c{ci}", "root_index": ri,
                                   "kind": kind, "params": params, "tape": tape.astype(np.float32).tolist()})
        proposal_path = bank_dir / f"candidates_s{seed}.json"
        with proposal_path.open("x") as output:
            json.dump({"seed": seed, "source_identity": source_id, "roots_file": "acquisition_roots.json",
                       "candidates": candidates, "selection_inputs": "root observations and imagined trajectories only"}, output)
        random_scores = {c["id"]: float(proposal_rng.random()) for c in candidates}
        seed_chosen = balanced_pick(candidates, args.seed_branches, random_scores)
        seed_ids = {c["id"] for c in seed_chosen}
        seed_bank = execute_selected(bank_dir / f"seed_s{seed}", acquisition_roots, seed_chosen, env, {"source": source_id})
        seed_clips = clips_from_bank(seed_bank, im, ctx, range(len(seed_bank)), scaler)
        seed_bank.h5.close()
        for arm in ("random", "boundary"):
            remaining = [c for c in candidates if c["id"] not in seed_ids]
            current = model
            acquired_clips, selected, curve = [], [], []
            model_queries = 0
            for budget in args.budgets:
                need = budget - len(selected)
                if arm == "random":
                    scores = random_scores
                else:
                    scores = score_candidates(remaining, acquisition_roots, WalkerImaginer(current, scaler, device),
                                               probe, ctx, repair_rules, margins,
                                               {"speed": args.speed_band, "health": args.band})
                    model_queries += len(remaining)
                chosen = balanced_pick(remaining, need, scores)
                chosen_ids = {c["id"] for c in chosen}
                selected += chosen
                round_bank = execute_selected(bank_dir / f"{arm}_s{seed}_b{budget}", acquisition_roots, chosen, env,
                                              {"source": source_id, "seed": seed, "arm": arm, "budget": budget})
                acquired_clips.append(clips_from_bank(round_bank, im, ctx, range(len(round_bank)), scaler))
                round_bank.h5.close()
                remaining = [c for c in remaining if c["id"] not in chosen_ids]
                (bank_dir / f"{arm}_s{seed}_selection.json").write_text(json.dumps({"source_identity": source_id,
                    "root_file": "acquisition_roots.json", "seed": seed, "budget": budget,
                    "seed_candidates": seed_chosen, "selected_candidates": selected,
                    "branch_steps": budget * HORIZON_STEPS, "horizon_steps": HORIZON_STEPS,
                    "charged_steps": charged_cost(generation_steps, args.seed_branches, budget)["charged_steps"],
                    "planned_budgets": args.budgets, "selection_complete": budget == max(args.budgets)}) + "\n")
                current, log = adapt(base_sd, model, ClipSet.concat([seed_clips, *acquired_clips]), replay,
                                     AdaptConfig(**dict(cfg.to_dict(), seed=seed)), device=device, verbose=False)
                metrics, evaluation_rows = evaluate(current)
                costs = {**charged_cost(generation_steps, args.seed_branches, budget),
                         "model_candidate_queries": model_queries, "optimizer_steps": cfg.steps,
                         "training_time_s": log["wall_clock_s"]}
                checkpoint = run_dir / f"{arm}-s{seed}-b{budget}"
                checkpoint.mkdir()
                torch.save(predictor_side_state(current), checkpoint / "weights.pt")
                OmegaConf.save(OmegaConf.create({**vars(args), "adaptation": cfg.to_dict()}), checkpoint / "config.yaml")
                np.savez(checkpoint / "scalers.npz", action_mean=scaler[0], action_std=scaler[1])
                (checkpoint / "selection.json").write_text(json.dumps({"selected_ids": [c["id"] for c in selected],
                    "seed_ids": sorted(seed_ids), "candidate_file": str(proposal_path),
                    "candidate_sha256": file_sha256(proposal_path)}) + "\n")
                detailed_rows = {name: merge_frozen_sources(rows, base_rows[name]) for name, rows in evaluation_rows.items()}
                (checkpoint / "decomposition.json").write_text(json.dumps({rule: {name: decomposition_report(rows, rule) for name, rows in detailed_rows.items()} for rule in rules}, default=float) + "\n")
                (checkpoint / "evaluation_rows.json").write_text(json.dumps(evaluation_rows, default=float) + "\n")
                write_manifest(checkpoint, build_manifest(run_id=f"{run_id}-{arm}-s{seed}-b{budget}", kind="adapted",
                    seeds={"seed": seed}, data={"model": args.model, "model_sha256": gate_report["model_sha256"],
                    "source": source_id, "weights_sha256": file_sha256(checkpoint / "weights.pt"), "probe_sha256": file_sha256(Path(args.probes) / "walker_mlp.pt"),
                    "gate_sha256": file_sha256(gate_path)}, upstream_revisions=upstream_refs, costs=costs, metrics=metrics))
                (checkpoint / "README.md").write_text(f"# {checkpoint.name}\n\nPredictor-side adaptation of the base model in manifest.json. Encoder, observation projector and probe stay frozen. Weights contain predictor-side modules only.\n")
                revision = store.upload_run("walker2d", "adapted", checkpoint,
                    run_id=f"{run_id}-{arm}-s{seed}-b{budget}") if store else None
                curve.append({"budget": budget, **costs, "eval": metrics, "hf_revision": revision,
                              "selected_ids": [c["id"] for c in selected],
                              "evaluation_rows": {"file": str((checkpoint / "evaluation_rows.json").relative_to(run_dir)),
                                  "sha256": file_sha256(checkpoint / "evaluation_rows.json")}})
                report["adaptation"].setdefault(arm, {})[str(seed)] = curve
                (results_dir / "study.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
                print(f"[s4] {arm} seed {seed} budget {budget}; charged {costs['charged_steps']} steps", flush=True)
    report["boundary_vs_random"] = paired_acquisition_intervals(report, run_dir)
    report["status"] = "complete"
    report["wall_clock_s"] = time.time() - t0
    report["evaluation_steps"] = sum(len(b) * HORIZON_STEPS for b in banks.values())
    for destination in (results_dir, run_dir):
        (destination / "study.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    evaluation_fingerprints = {name: {filename: file_sha256(bank_dir / name / filename) for filename in ("branches.h5", "roots.json")} for name in banks}
    manifest = build_manifest(run_id=run_id, kind="s4", seeds={"acquisition": args.seeds},
        data={"source": source_id, "model_sha256": gate_report["model_sha256"], "gate_sha256": file_sha256(gate_path), "evaluation_banks": evaluation_fingerprints, "acquisition_files": acquisition_files(bank_dir, args.seeds, args.budgets),
              "saved_evaluation_rows": report["saved_evaluation_rows"]},
        upstream_revisions=upstream_refs, costs={"evaluation_steps": report["evaluation_steps"]}, started_at=t0)
    for destination in (run_dir, results_dir, bank_dir):
        write_manifest(destination, manifest)
    (bank_dir / "study.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    if store:
        report["bank_hf_revision"] = store.upload_run("walker2d-data", "banks", bank_dir, run_id=run_id)
        (results_dir / "study.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    for bank in banks.values():
        bank.h5.close()
    env.close()
    ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
