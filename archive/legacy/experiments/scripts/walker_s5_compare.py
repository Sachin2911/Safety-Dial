#!/usr/bin/env python3
"""Compare pretraining versus adaptation on the completed S4 evaluation banks.

Example:
  uv run python experiments/scripts/walker_s5_compare.py \
    --model-a runs/<A> --model-b runs/<B> --data-dir data/study/walker2d/<source> \
    --data-b data/study/walker2d/<matched-B> --s4-run runs/<S4> \
    --s4-bank data/study/walker2d/<source>/s4/<S4> --seed 0

Fresh exact-N A updates exclude the S4 common seed and root-history supervision.
Fresh A/B physical probes use exactly the same frame and episode lists, capacity,
optimizer and training seed. A's probe stays fixed across its predictor variants.
All margins use the original S4 target acceptance and development tapes only.
No simulator action is executed: test/stress banks are reused byte-for-byte.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from helpers.threads import apply_torch, pin_threads

pin_threads()
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from omegaconf import OmegaConf

apply_torch()
from helpers.dialMetrics import cluster_bootstrap, fsa, margin_for_acceptance
from helpers.hfStore import HFStore
from helpers.locoData import RenderContext
from helpers.poseProbes import ProbeSpec, fit_probe, save_probe
from helpers.predictorAdapt import AdaptConfig, ClipSet, predictor_side_state
from helpers.runManifest import build_manifest, make_run_id, write_manifest
from helpers.walkerBank import WalkerBank
from helpers.walkerComparison import completed_s4, load_predictor_variant, pinned_input, probe_sample_plan, verify_matched_training
from helpers.walkerExperience import acquisition_files, adapt_exact_experience, exact_branch_clips, verified_experience, verify_set_b_rows
from helpers.walkerLewm import WalkerImaginer, load_walker_model
from helpers.walkerProtocol import file_sha256
from helpers.walkerReporting import decomposition_report, imagined_summary, merge_frozen_sources
from helpers.walkerRules import rule_unsafe
from helpers.walkerValidation import episode_role, load_source_split, verify_render_fingerprint
from walker_s4_study import analyse


def paired_fsa(rows_a, rows_b, rule, margin_a, margin_b, n_boot):
    def identities(rows):
        return [(r["root_id"], r["branch"]) for r in rows]
    if identities(rows_a) != identities(rows_b):
        raise ValueError("Comparison rows do not refer to identical evaluation tapes")
    true_a = np.array([r[f"cmin_dense_{rule}"] for r in rows_a])
    true_b = np.array([r[f"cmin_dense_{rule}"] for r in rows_b])
    np.testing.assert_array_equal(true_a, true_b)
    a = np.array([r[f"cmin_imagined_{rule}"] for r in rows_a])
    b = np.array([r[f"cmin_imagined_{rule}"] for r in rows_b])
    u = rule_unsafe(rule, true_a)
    roots = np.array([r["root"] for r in rows_a])
    return cluster_bootstrap(lambda a, b, u: fsa(b, u, margin_b)["fsa"] - fsa(a, u, margin_a)["fsa"],
                             roots, n_boot=n_boot, a=a, b=b, u=u)


def fit_fresh_probes(models, scalers, source, plan, spec, ctx, run_dir, inputs, store, device):
    """Encode each shared frame for each representation; no final-test labels enter fitting."""
    encoders = {name: WalkerImaginer(model, scalers[name], device) for name, model in models.items()}
    latents = {name: [] for name in models}
    targets, roles = [], []
    with h5py.File(source, "r") as f:
        offsets, lengths = f["ep_offset"][:], f["ep_len"][:]
        for part in plan["parts"]:
            episode, indices = part["episode"], np.asarray(part["indices"])
            start, end = int(offsets[episode]), int(offsets[episode] + lengths[episode])
            # Contiguous reads, then shared frame indices within the source episode.
            qpos, qvel, velocity = f["qpos"][start:end], f["qvel"][start:end], f["x_velocity"][start:end]
            frames = ctx.render_many(qpos[indices], qvel[indices])
            for name, encoder in encoders.items():
                latents[name].append(encoder.encode(frames).cpu().numpy())
            targets.append(np.stack([qpos[indices, 1], qpos[indices, 2],
                np.where(indices > 0, velocity[np.maximum(indices - 1, 0)], 0.)], axis=1).astype(np.float32))
            roles.extend([part["role"]] * len(indices))
    targets = np.concatenate(targets)
    train = np.array(roles) == "training"
    validation = ~train
    probes, summaries, revisions = {}, {}, {}
    for name in models:
        z = np.concatenate(latents[name]).astype(np.float32)
        if z.shape[0] != plan["n_frames"] or len(targets) != plan["n_frames"]:
            raise ValueError("A/B probe sample identities differ")
        probe, stats = fit_probe(z[train], targets[train], z[validation], targets[validation],
                                  spec, device=device, verbose=False)
        for parameter in probe.parameters():
            parameter.requires_grad_(False)
        destination = run_dir / f"probe-{name}"
        destination.mkdir()
        save_probe(probe, stats, destination / "walker_mlp.pt")
        (destination / "sample_plan.json").write_text(json.dumps(plan) + "\n")
        OmegaConf.save(OmegaConf.create({"probe": asdict(spec), "model": name}), destination / "config.yaml")
        manifest = build_manifest(run_id=f"{run_dir.name}-probe-{name}", kind="probes",
            seeds={"probe": spec.seed}, data={"model": inputs[name], "probe_source_sha256": file_sha256(source),
                "sample_plan_sha256": file_sha256(destination / "sample_plan.json")},
            upstream_revisions={"model": inputs[name]["hf"]}, metrics=stats["val"])
        write_manifest(destination, manifest)
        (destination / "README.md").write_text(f"# Fresh {name} physical probe\n\nHeight, pitch and speed labels are privileged simulator supervision. A/B use the same recorded frame list, whole-episode split, architecture and optimizer.\n")
        revisions[name] = store.upload_run("walker2d", "probes", destination,
            run_id=f"{run_dir.name}-probe-{name}") if store else None
        probes[name], summaries[name] = probe.eval(), {"spec": asdict(spec), "validation": stats["val"]}
    return probes, summaries, revisions


def exact_updates(model, scaler, selections, locations, report, bank_dir, ctx, run_dir, inputs, store, device):
    """Fresh S5 checkpoints use exactly N additional rows, preserving original S4 runs."""
    cfg = AdaptConfig(**{**report["adaptation_recipe"], "seed": selections["random"]["seed"]})
    replay = ClipSet.load(bank_dir / "replay_clips.npz")
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    imaginer = WalkerImaginer(model, scaler, device)
    checkpoints, costs = {}, {}
    for arm in ("random", "boundary"):
        acquired = exact_branch_clips(selections[arm], locations[arm], imaginer, ctx, scaler)
        current, log = adapt_exact_experience(base_state, model, acquired, replay, cfg, device=device)
        checkpoint = run_dir / f"A-{arm}-exact-N"
        checkpoint.mkdir()
        torch.save(predictor_side_state(current), checkpoint / "weights.pt")
        OmegaConf.save(OmegaConf.create({"adaptation": cfg.to_dict(), "experience": acquired.meta}), checkpoint / "config.yaml")
        np.savez(checkpoint / "scalers.npz", action_mean=scaler[0], action_std=scaler[1])
        selection_record = {"selected_ids": acquired.meta["selected_ids"], "seed_ids": [],
            "branch_steps": selections[arm]["branch_steps"], "history_prefix_steps": 0,
            "source_selection_sha256": file_sha256(bank_dir / f"{arm}_s{cfg.seed}_selection.json")}
        (checkpoint / "selection.json").write_text(json.dumps(selection_record) + "\n")
        (checkpoint / "training_log.json").write_text(json.dumps(log) + "\n")
        costs[arm] = {"additional_simulator_steps": 0, "new_training_transitions": selections[arm]["branch_steps"],
            "seed_training_transitions": 0, "history_prefix_training_transitions": 0,
            "optimizer_steps": cfg.steps, "training_time_s": log["wall_clock_s"],
            "original_acquisition_charged_steps": selections[arm]["charged_steps"]}
        manifest = build_manifest(run_id=f"{run_dir.name}-A-{arm}-exact-N", kind="adapted",
            seeds={"seed": cfg.seed}, data={"model_sha256": inputs["A"]["sha256"]["weights.pt"],
                "weights_sha256": file_sha256(checkpoint / "weights.pt"),
                "source": selections[arm]["source_identity"], "experience": acquired.meta,
                "replay_sha256": file_sha256(bank_dir / "replay_clips.npz"),
                "selection": selection_record}, costs=costs[arm],
            upstream_revisions={"model": inputs["A"]["hf"], "experience_and_replay": inputs["banks"]["hf"]})
        write_manifest(checkpoint, manifest)
        (checkpoint / "README.md").write_text(f"# Exact-N {arm} adaptation\n\nFresh predictor-side update from LeWM-A using exactly {selections[arm]['branch_steps']} acquired pre-action rows. Common S4 seeds and root history are excluded. Original S4 replay, optimizer, seed, mixture and update count are fixed. Separate replay/acquired loss means allow unequal clip lengths without altering the replay.\n")
        if store:
            store.upload_run("walker2d", "adapted", checkpoint, run_id=f"{run_dir.name}-A-{arm}-exact-N")
        inputs[f"A_{arm}"] = pinned_input(checkpoint, ("weights.pt", "manifest.json", "selection.json", "config.yaml"), allow_local=store is None)
        checkpoints[arm] = checkpoint
        del current, acquired
        print(f"[s5-compare] built {arm} update with exactly {selections[arm]['branch_steps']} acquired rows", flush=True)
    return checkpoints, costs


def format_number(value):
    return f"{value:.4f}" if value is not None and np.isfinite(value) else "undefined"


def write_summary(path, report):
    lines = ["# Walker2d pretraining versus adaptation", "",
             "All variants use the identical S4 test tapes. Margins are calibrated on development tapes at the frozen S4 target acceptance. A/B use fresh probes with matched capacity and the same trajectory split; predictor variants keep their representation's probe fixed.", "",
             "| Variant | Bank | Rule | Unsafe accepted / accepted | FSA | Acceptance |",
             "|---|---|---|---:|---:|---:|"]
    for variant, banks in report["variants"].items():
        for name in ("test", "stress"):
            for rule, metrics in banks[name].items():
                score = metrics["at_matched"]
                lines.append(f"| {variant} | {name} | {rule} | {score['n_false_safe']} / {score['n_accepted']} | {format_number(score['fsa'])} | {format_number(score['acceptance_rate'])} |")
    lines += ["", "Paired differences below are the second variant minus the first. Intervals resample source episodes, preserving sibling tapes.", "",
              "| Contrast | Bank | Rule | FSA difference | 95% interval |",
              "|---|---|---|---:|---:|"]
    for contrast, banks in report["contrasts"].items():
        for name, rules in banks.items():
            for rule, ci in rules.items():
                lines.append(f"| {contrast} | {name} | {rule} | {format_number(ci['point'])} | [{format_number(ci['lo'])}, {format_number(ci['hi'])}] |")
    exposure = report.get("new_training_experience", {})
    if exposure:
        lines += ["", f"Fresh A adaptation and B pretraining use exactly {exposure['A_adaptation_steps']:,} acquired transitions each. A-random and B use identical recorded states and actions. Common S4 seeds and root-history prefixes are excluded from these new A updates. Original S4 checkpoints remain separate. Boundary uses its own equally charged selected N transitions."]
    lines += ["", f"Matched pretraining uses {report['training_control']['pretraining_steps_each']:,} data transitions and {report['training_control']['optimizer_steps_each']:,} optimizer updates per model. Set B replaces exactly {report['training_control']['additional_experience_steps']:,} ordinary transitions with the acquired random-arm transitions. This comparison executes zero additional simulator steps.", "",
              "This paired A/B run tests one pretraining seed and one matched acquisition seed. The report does not infer training-seed robustness or select a method using final-test outcomes.", ""]
    Path(path).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("model-a", "model-b", "data-dir", "data-b", "s4-run", "s4-bank"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--b-boundary", type=Path, default=None, help="optional existing B predictor-side checkpoint")
    ap.add_argument("--seed", type=int, default=0, help="S4 acquisition seed used to construct set B")
    ap.add_argument("--probe-seed", type=int, default=0)
    ap.add_argument("--probe-frames", type=int, default=30000)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--run-id", default=None, help="explicit immutable run directory name")
    ap.add_argument("--n", type=int, default=1)
    args = ap.parse_args()
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        ap.error("run-id must be a single alphanumeric directory name")
    if args.bootstrap < 1:
        ap.error("bootstrap count must be positive")
    started = time.time()
    source_split = load_source_split(args.data_dir)
    s4_report = json.loads((args.s4_run / "study.json").read_text())
    budget = completed_s4(s4_report, args.seed)
    s4_manifest = json.loads((args.s4_run / "manifest.json").read_text())
    bank_report = json.loads((args.s4_bank / "study.json").read_text())
    def report_core(report):
        return json.dumps({k: v for k, v in report.items() if k != "bank_hf_revision"}, sort_keys=True)
    if completed_s4(bank_report, args.seed) != budget or report_core(bank_report) != report_core(s4_report):
        raise ValueError("S4 report and archive do not identify the same complete study")
    roots_sha = file_sha256(args.data_dir / "roots.h5")
    if s4_report["source_identity"]["sha256"] != roots_sha:
        raise ValueError("Source roots differ from S4")
    selection_path = args.s4_bank / f"random_s{args.seed}_selection.json"
    selection = json.loads(selection_path.read_text())
    if not selection["selection_complete"] or selection["budget"] != budget:
        raise ValueError("S4 largest random budget is incomplete")
    if [c["id"] for c in selection["selected_candidates"]] != s4_report["adaptation"]["random"][str(args.seed)][-1]["selected_ids"]:
        raise ValueError("Set-B selection does not match the completed S4 report")
    training_control = verify_matched_training(args.model_a, args.model_b, args.data_b, budget, file_sha256(selection_path))
    allow_local = args.no_upload
    inputs = {"A": pinned_input(args.model_a, ("weights.pt", "config.json", "scalers.npz", "manifest.json"), allow_local=allow_local),
              "B": pinned_input(args.model_b, ("weights.pt", "config.json", "scalers.npz", "manifest.json"), allow_local=allow_local),
              "banks": pinned_input(args.s4_bank, ("study.json", "manifest.json"), allow_local=allow_local),
              "source": pinned_input(args.data_dir, ("probe.h5", "splits.json"), allow_local=allow_local),
              "setB": pinned_input(args.data_b, ("manifest.json", "splits.json"), allow_local=allow_local)}
    if inputs["A"]["sha256"]["weights.pt"] != s4_manifest["data"]["model_sha256"]:
        raise ValueError("LeWM-A is not the S4 base model")
    s4_checkpoints = {arm: args.s4_run / f"{arm}-s{args.seed}-b{budget}" for arm in ("random", "boundary")}
    for arm, checkpoint in s4_checkpoints.items():
        inputs[f"S4_{arm}"] = pinned_input(checkpoint, ("weights.pt", "manifest.json", "selection.json", "config.yaml"), allow_local=allow_local)
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        if manifest["data"].get("weights_sha256") != inputs[f"S4_{arm}"]["sha256"]["weights.pt"]:
            raise ValueError("Adapted checkpoint changed after the S4 run")
        original_config = OmegaConf.to_container(OmegaConf.load(checkpoint / "config.yaml"))["adaptation"]
        if original_config != s4_report["adaptation_recipe"]:
            raise ValueError("S4 adaptation recipe changed between report and checkpoint")
        selected = json.loads((checkpoint / "selection.json").read_text())["selected_ids"]
        if selected != s4_report["adaptation"][arm][str(args.seed)][-1]["selected_ids"]:
            raise ValueError("Adapted model does not use the completed S4 data budget")
    measured = acquisition_files(args.s4_bank, s4_report["acquisition_seeds"], s4_report["planned_budgets"])
    if measured != s4_manifest["data"].get("acquisition_files"):
        raise ValueError("S4 acquisition files or original replay changed")
    selections, locations = {}, {}
    for arm in ("random", "boundary"):
        selections[arm], locations[arm] = verified_experience(args.s4_bank, s4_report, arm, args.seed)
    row_identity = verify_set_b_rows(args.data_b, selections["random"], locations["random"])
    b_intervention = json.loads((args.data_b / "manifest.json").read_text())
    if b_intervention["metrics"]["original_acquisition_steps"] != selections["random"]["charged_steps"]:
        raise ValueError("Set B and A-random routes do not report the same acquisition charge")
    if args.b_boundary:
        inputs["B_boundary"] = pinned_input(args.b_boundary, ("weights.pt", "manifest.json"), allow_local=allow_local)
        extra_manifest = json.loads((args.b_boundary / "manifest.json").read_text())
        if extra_manifest["data"]["source"]["sha256"] != roots_sha:
            raise ValueError("Optional B adaptation uses different source roles")
    banks = {}
    for name, role in (("dev", "development"), ("test", "test"), ("stress", "test")):
        expected = s4_manifest["data"]["evaluation_banks"][name]
        if {f: file_sha256(args.s4_bank / name / f) for f in ("branches.h5", "roots.json")} != expected:
            raise ValueError("Final evaluation bank changed after S4")
        bank = WalkerBank(args.s4_bank / name)
        if any(episode_role(root.episode) != role or root.episode not in source_split["roots"][role] for root in bank.roots):
            raise ValueError("Evaluation source roles changed")
        banks[name] = bank
    run_id = args.run_id or make_run_id("walker2d", "s5-compare", f"s{args.seed}", n=args.n)
    run_dir = REPO_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    results = args.results_dir or REPO_ROOT / "docs/mainPlan/results/s5" / run_id
    results.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}), run_dir / "config.yaml")
    device = "cuda"
    model_a, scaler_a = load_walker_model(args.model_a, device)
    model_b, scaler_b = load_walker_model(args.model_b, device)
    with h5py.File(args.data_dir / "probe.h5", "r") as f:
        plan = probe_sample_plan(f["ep_len"][:], args.probe_frames, seed=args.probe_seed)
    spec = ProbeSpec(target="walker", kind="mlp", seed=args.probe_seed)
    ctx = RenderContext()
    verify_render_fingerprint(ctx, args.data_dir / "probe.h5")
    store = None if args.no_upload else HFStore()
    checkpoints, exact_update_costs = exact_updates(model_a, scaler_a, selections, locations, s4_report, args.s4_bank, ctx, run_dir, inputs, store, device)
    probes, probe_metrics, probe_revisions = fit_fresh_probes({"A": model_a, "B": model_b},
        {"A": scaler_a, "B": scaler_b}, args.data_dir / "probe.h5", plan, spec, ctx, run_dir, inputs, store, device)
    rules = tuple(s4_report["active_rules"])
    targets = s4_report["target_acceptance"]
    report = {"run_id": run_id, "status": "running", "s4_run_id": s4_report["run_id"],
        "acquisition_seed": args.seed, "largest_additional_budget": budget, "active_rules": rules,
        "target_acceptance": targets, "training_control": training_control, "inputs": inputs,
        "new_training_experience": {"A_adaptation_steps": budget * 100,
            "A_common_seed_steps": 0, "A_history_prefix_steps": 0, "A_additional_steps": budget * 100,
            "B_replaced_steps": training_control["additional_experience_steps"],
            "policy": "Fresh A updates and B use exactly N additional branch rows; common seed and history prefixes excluded",
            "random_B_row_identity": row_identity},
        "exact_N_adaptation_costs": exact_update_costs,
        "route_acquisition_charged_steps": {"A_unadapted": 0, "A_random": selections["random"]["charged_steps"],
            "A_boundary": selections["boundary"]["charged_steps"], "B_unadapted": selections["random"]["charged_steps"]},
        "probe_protocol": plan, "probe_metrics": probe_metrics, "probe_revisions": probe_revisions,
        "variants": {}, "contrasts": {}, "additional_simulator_steps": 0,
        "acquisition_costs": {arm: {k: v for k, v in s4_report["adaptation"][arm][str(args.seed)][-1].items()
            if k in ("charged_steps", "root_generation_steps", "common_seed_steps", "acquired_steps", "optimizer_steps", "training_time_s")}
            for arm in ("random", "boundary")}}
    if args.b_boundary:
        report["optional_B_adaptation_costs"] = extra_manifest.get("costs", {})
    all_rows, margins = {}, {}
    variants = [("A_unadapted", "A", None), ("A_random", "A", checkpoints["random"]),
                ("A_boundary", "A", checkpoints["boundary"]), ("B_unadapted", "B", None)]
    if args.b_boundary:
        variants.append(("B_boundary", "B", args.b_boundary))
    base_models, scalers = {"A": model_a, "B": model_b}, {"A": scaler_a, "B": scaler_b}
    for label, representation, checkpoint in variants:
        current = base_models[representation] if checkpoint is None else load_predictor_variant(
            base_models[representation], checkpoint, inputs[representation]["sha256"]["weights.pt"])
        imaginer = WalkerImaginer(current, scalers[representation], device)
        rows = {name: analyse(bank, imaginer, probes[representation], ctx,
                             sources_all=checkpoint is None, rules=rules) for name, bank in banks.items()}
        if checkpoint:
            rows = {name: merge_frozen_sources(items, all_rows[f"{representation}_unadapted"][name]) for name, items in rows.items()}
        local_margins = {rule: margin_for_acceptance(np.array([r[f"cmin_imagined_{rule}"] for r in rows["dev"]]), targets[rule]) for rule in rules}
        report["variants"][label] = {name: {rule: imagined_summary(items, rule, local_margins[rule], n_boot=args.bootstrap) for rule in rules} for name, items in rows.items()}
        (run_dir / f"{label}_rows.json").write_text(json.dumps(rows, default=float) + "\n")
        (run_dir / f"{label}_decomposition.json").write_text(json.dumps({rule: {name: decomposition_report(items, rule) for name, items in rows.items()} for rule in rules}, default=float) + "\n")
        all_rows[label], margins[label] = rows, local_margins
        if checkpoint:
            del current
        print(f"[s5-compare] evaluated {label} on fixed S4 banks", flush=True)
    contrasts = [("A_unadapted", "A_random"), ("A_unadapted", "A_boundary"),
                 ("A_random", "A_boundary"), ("A_random", "B_unadapted"), ("A_boundary", "B_unadapted")]
    if args.b_boundary:
        contrasts.append(("B_unadapted", "B_boundary"))
    for left, right in contrasts:
        report["contrasts"][f"{right} minus {left}"] = {name: {rule: paired_fsa(
            all_rows[left][name], all_rows[right][name], rule, margins[left][rule], margins[right][rule], args.bootstrap)
            for rule in rules} for name in ("test", "stress")}
    report["status"] = "complete"
    report["wall_clock_s"] = time.time() - started
    for destination in (run_dir, results):
        (destination / "comparison.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
        write_summary(destination / "README.md", report)
    manifest = build_manifest(run_id=run_id, kind="s5-comparison", seeds={"acquisition": args.seed, "probe": args.probe_seed},
        data={"inputs": inputs, "sample_plan": plan}, upstream_revisions={k: v["hf"] for k, v in inputs.items()},
        costs={"additional_simulator_steps": 0, **training_control}, metrics={"variants": report["variants"], "contrasts": report["contrasts"]}, started_at=started)
    for destination in (run_dir, results):
        write_manifest(destination, manifest)
    if store:
        report["report_hf_revision"] = store.upload_run("walker2d", "comparisons", run_dir, run_id=run_id)
        (results / "comparison.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    for bank in banks.values():
        bank.h5.close()
    ctx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
