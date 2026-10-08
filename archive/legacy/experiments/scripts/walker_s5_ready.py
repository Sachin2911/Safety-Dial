#!/usr/bin/env python3
"""Verify complete, pinned S4 and full LeWM-A before the separate S5 workflow."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from omegaconf import OmegaConf

from helpers.walkerComparison import completed_s4, pinned_input
from helpers.walkerExperience import acquisition_files, verified_experience
from helpers.walkerProtocol import file_sha256
from helpers.walkerValidation import load_source_split


def assess_completion(report, train_state, config, receipt, *, expected_steps, expected_budget, expected_seeds, seed):
    reasons = []
    budget = None
    try:
        budget = completed_s4(report, seed)
    except ValueError as exc:
        reasons.append(str(exc))
    if budget != expected_budget or sorted(report.get("acquisition_seeds", [])) != sorted(expected_seeds):
        reasons.append("S4 does not contain the declared largest budget and acquisition seeds")
    if any(value != expected_steps for value in (train_state.get("step"), config.get("total_steps"), receipt.get("step"))):
        reasons.append("LeWM-A state, recipe and final upload receipt do not match the declared full step budget")
    return {"go": not reasons, "reasons": reasons, "expected_optimizer_steps": expected_steps,
            "largest_additional_budget": expected_budget, "expected_acquisition_seeds": expected_seeds}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("model-a", "data-dir", "s4-run", "s4-bank", "out"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    ap.add_argument("--expected-steps", type=int, required=True)
    ap.add_argument("--expected-budget", type=int, required=True)
    ap.add_argument("--expected-seeds", nargs="+", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.out.exists() or min(args.expected_steps, args.expected_budget) < 1:
        ap.error("Require a fresh output and positive declared budgets")
    report = json.loads((args.s4_run / "study.json").read_text())
    state = json.loads((args.model_a / "train_state.json").read_text())
    config = OmegaConf.to_container(OmegaConf.load(args.model_a / "config.yaml"))
    receipt = json.loads((args.model_a / "upload_receipt.json").read_text())
    gate = assess_completion(report, state, config, receipt, expected_steps=args.expected_steps,
        expected_budget=args.expected_budget, expected_seeds=args.expected_seeds, seed=args.seed)
    result = {"gate": gate, "source": str(args.data_dir), "model_a": str(args.model_a), "s4_run": str(args.s4_run)}
    if gate["go"]:
        archive = json.loads((args.s4_bank / "study.json").read_text())
        def core(value):
            return {k: v for k, v in value.items() if k != "bank_hf_revision"}
        if core(archive) != core(report):
            raise ValueError("S4 final report and uploaded archive differ")
        manifest = json.loads((args.s4_run / "manifest.json").read_text())
        if json.loads((args.s4_bank / "manifest.json").read_text()) != manifest:
            raise ValueError("S4 final manifest and uploaded archive differ")
        if file_sha256(args.model_a / "weights.pt") != manifest["data"]["model_sha256"]:
            raise ValueError("S4 and S5 refer to different LeWM-A weights")
        if file_sha256(args.data_dir / "roots.h5") != report["source_identity"]["sha256"]:
            raise ValueError("S4 and S5 use different source roots")
        load_source_split(args.data_dir)
        measured = acquisition_files(args.s4_bank, report["acquisition_seeds"], report["planned_budgets"])
        if measured != manifest["data"].get("acquisition_files"):
            raise ValueError("S4 experience/replay archive changed")
        for arm in ("random", "boundary"):
            verified_experience(args.s4_bank, report, arm, args.seed)
        for name in ("dev", "test", "stress"):
            expected = manifest["data"]["evaluation_banks"][name]
            if {f: file_sha256(args.s4_bank / name / f) for f in ("branches.h5", "roots.json")} != expected:
                raise ValueError("S4 final evaluation bank changed")
        result["inputs"] = {
            "model_a": pinned_input(args.model_a, ("weights.pt", "train_state.json", "config.yaml", "scalers.npz", "manifest.json")),
            "source": pinned_input(args.data_dir, ("setA.h5", "roots.h5", "splits.json")),
            "s4_bank": pinned_input(args.s4_bank, ("manifest.json", "study.json"))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(gate))
    return 0 if gate["go"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
