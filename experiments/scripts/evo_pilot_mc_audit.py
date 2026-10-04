#!/usr/bin/env python3
"""Offline, retrospective Monte Carlo precision audit of the completed power pilot.

No model, physics, collection, fitting or policy selection is performed. Estimates
condition on the already frozen policies and inspected development episodes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.evoRanking import segment_metrics  # noqa: E402


def paired_mc_precision(high, low):
    """Conditional SE of a paired mean over independent seed/root/sample draws.

    Common random numbers pair high with low. Source and search variability are
    deliberately excluded: they belong to the study's crossed bootstrap.
    """
    high, low = np.asarray(high), np.asarray(low)
    if high.shape != low.shape or high.ndim != 3 or min(high.shape[:2]) < 1:
        raise ValueError("matching seed-by-root-by-sample arrays required")
    if high.shape[-1] < 2 or not all(np.isin(x, [0, 1]).all() for x in [high, low]):
        raise ValueError("at least two binary samples per root required")
    differences = high.astype(float) - low.astype(float)
    seeds, roots, samples = high.shape
    cells = seeds * roots
    variance = differences.var(axis=-1, ddof=1)
    se = float(np.sqrt(variance.sum() / (samples * cells**2)))
    halves = samples // 2
    return dict(
        search_seeds=seeds, fixed_development_episodes=roots, samples_per_root=samples,
        imagined_high_minus_low=float(differences.mean()),
        conditional_monte_carlo_se=se,
        worst_case_monte_carlo_se=float(1 / np.sqrt(cells * samples)),
        first_half_imagined_difference=float(differences[..., :halves].mean()),
        second_half_imagined_difference=float(differences[..., halves:].mean()),
        first_half_minus_full=float(differences[..., :halves].mean() - differences.mean()),
        maximum_rootwise_mc_se=float(np.sqrt(variance / samples).max()),
        zero_empirical_variance_does_not_prove_zero_population_variance=True,
    )


def audit(pilot):
    pilot = Path(pilot)
    hashes = {}

    def track(path):
        path = Path(path)
        hashes[str(path.relative_to(pilot))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    config = yaml.safe_load(track(pilot / "config.yaml").read_text())["power_pilot"]
    picks = json.loads(track(pilot / "frozen_selections.json").read_text())["picks"]
    roles = json.loads(track(pilot / "episode_roles.json").read_text())
    seeds = list(config["search_seeds"])
    indices = roles["indices"]["assessment"]
    episodes = roles["source_episodes"]["assessment"]
    if (len(episodes) != len(indices) or len(set(seeds)) != len(seeds)
            or len(set(episodes)) != len(episodes)):
        raise ValueError("unique search seeds and source episodes required")
    samples = config["samples"]["audit_k1"]
    with np.load(track(pilot / "paired_assessment.npz"), allow_pickle=False) as f:
        paired = {k: f[k].copy() for k in f.files}
    if not np.array_equal(paired["assessment_indices"], indices):
        raise ValueError("paired outcome root order changed")
    flags = {}
    for pick in picks:
        slot = pick["slot"]
        with np.load(track(pilot / f"audit_{slot:03d}_k1.npz"), allow_pickle=False) as f:
            ro = f["readout"]
            if ro.shape != (len(indices), samples, 10, 3) or not np.isfinite(ro).all():
                raise ValueError("invalid archived noisy audit")
            flags[slot] = segment_metrics(ro)["violated"]
        if not np.array_equal(flags[slot].mean(-1), paired["imagined_k1"][slot]):
            raise ValueError("sample means disagree with reported pilot probabilities")

    contrasts = {}
    for k in config["noise_levels"]:
        endpoints = []
        for pressure in ["high_pressure", "low_pressure"]:
            endpoint = config["analysis"][pressure]
            rows = []
            for seed in seeds:
                matches = [p for p in picks if p["seed"] == seed and p["noise_k"] == k
                           and p["population"] == endpoint["population"]
                           and p["generation"] == endpoint["generation"]]
                if len(matches) != 1:
                    raise ValueError("missing or duplicate declared pressure endpoint")
                rows.append(flags[matches[0]["slot"]])
            endpoints.append(np.stack(rows))
        contrasts[f"search_k{k:g}_audited_at_k1"] = paired_mc_precision(*endpoints)
    return dict(
        analysis="retrospective development-only audit; no change to frozen endpoints",
        source_run=pilot.name, source_sha256=hashes,
        selected_policy_archives_verified=len(flags), contrasts=contrasts,
        proposed_main_conditional_bound=dict(
            seeds=10, episodes=320, samples_per_root=32,
            paired_high_minus_low_mc_se_upper=float(1 / np.sqrt(10 * 320 * 32)),
            rationale="A paired Bernoulli difference lies in [-1,1], so its variance is at most 1.",
            assumptions="Independent audit draws across seed/root/sample cells; common draws within policy pairs; fixed selected policies.",
            counts_remain_provisional=True,
        ),
        limitations=[
            "This estimates audit sampling error, not search-seed or episode uncertainty.",
            "No ROSARL policies were evaluated; no ROSARL effect or statistical power is inferred.",
            "Per-root risk estimates can remain noisy even when the pooled contrast is precise.",
            "Fitness and policy-selection sampling adequacy are not tested here.",
            "Split halves reuse one stored 32-sample batch, not separately executed 16-sample kernels.",
        ],
        incremental_costs=dict(predictor_rows=0, real_steps=0, new_episodes=0, training_updates=0),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.pilot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result["contrasts"], indent=2))


if __name__ == "__main__":
    main()
