"""Three predeclared optimizers, one controller class, independent replicated runs."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np

from helpers.lewm17_execution import Executor, configure, ready, register, verify_freeze
from helpers.lewm17_protocol import CONFIG, OUT, job, write_json
from helpers.lewm17_selector import (
    MARGINS,
    anchors,
    es_gradient,
    reinforce_gradient,
    utility,
)


def rank(results):
    unsafe = any(r["margin_breach"] or r["any_violation"] for r in results)
    return (
        not unsafe,
        sum(r["requested_safe_completion"] for r in results),
        -sum(r["J"] for r in results),
        -sum(r["counters"]["predictor_steps"] for r in results),
    )


def optimize(env, replicate, method, args, settings):
    name = f"{env}-{method}-r{replicate}"
    folder = OUT / "optimizers" / name
    folder.mkdir(parents=True, exist_ok=False)
    development = json.loads((CONFIG / "development.json").read_text())[env]
    validation = json.loads((CONFIG / "validation.json").read_text())[env]
    rounds = settings["rounds"]
    executor = Executor(args.gpu, args.manifest, name, 24 + 24 * rounds + 24)
    base_seed = 17020000 + (1000 if env == "reacher" else 0) + replicate * 10000
    initial = np.clip(np.random.default_rng(base_seed).normal(0, 0.5, (6, 6)), -3, 3)
    rng = np.random.default_rng(base_seed + {"random": 1, "reinforce": 2, "evolution": 3}[method])
    center, previous, baseline = initial.copy(), initial.copy(), 0.0
    started = time.time()

    def evaluate(weights, points, stage, label, stochastic, candidate_index=0):
        results, episodes = [], []
        for point_index, (case, margin, planner_seed, mode_seed) in enumerate(points):
            ready()
            params = {
                "weights": weights.tolist(),
                "margin": margin,
                "stochastic": stochastic,
                "mode_seed": mode_seed + candidate_index * 100,
            }
            trial = job(
                env,
                stage,
                case,
                margin,
                planner_seed,
                f"{method}-r{replicate}-{label}-e{point_index}",
                params,
            )
            trial.update(
                optimizer=method,
                replicate=replicate,
                candidate=label,
                optimizer_run=name,
                diagnostic=False,
            )
            result, decisions = executor.run(trial)
            results.append(result)
            episodes.append((utility(result), decisions))
        return results, episodes

    anchor_points = [
        (
            development[j],
            MARGINS[env][j % 2],
            171000 + replicate * 100 + j,
            17200000 + replicate * 100 + j,
        )
        for j in range(3)
    ]
    anchor_results = []
    shared = anchors(env)
    for index, weights in enumerate(shared):
        results, _ = evaluate(weights, anchor_points, "A", f"anchor{index}", False)
        anchor_results.append(results)
    eligible = [i for i, rows in enumerate(anchor_results) if rank(rows)[0]]
    if not eligible:
        raise RuntimeError("All common anchors unsafe; no safe validation fallback")
    anchor_index = max(eligible, key=lambda i: rank(anchor_results[i]))
    anchor = shared[anchor_index]
    write_json(
        folder / "anchors.json",
        {
            "selected": anchor_index,
            "ranks": [list(rank(rows)) for rows in anchor_results],
            "weights": [w.tolist() for w in shared],
            "resources": executor.actual,
        },
    )
    history, incumbent, last_candidates, last_scores = [], initial.copy(), [], []
    for round_index in range(rounds):
        ready()
        points = [
            (
                development[(round_index * 3 + j) % len(development)],
                MARGINS[env][(round_index + j) % 2],
                173000 + replicate * 1000 + round_index * 10 + j,
                17400000 + replicate * 1000 + round_index * 10 + j,
            )
            for j in range(3)
        ]
        if method == "random":
            candidates = [incumbent.copy()] + [
                np.clip(rng.normal(0, 0.5, (6, 6)), -3, 3) for _ in range(7)
            ]
        elif method == "evolution":
            directions = rng.normal(size=(4, 6, 6))
            candidates = [
                np.clip(center + sign * 0.3 * d, -3, 3) for d in directions for sign in (1, -1)
            ]
        else:
            candidates = [center.copy() for _ in range(8)]
        write_json(
            folder / f"round{round_index:02d}-before.json",
            {
                "center": center.tolist(),
                "candidates": [w.tolist() for w in candidates],
                "points": [
                    {"case_id": c["id"], "margin": m, "planner_seed": s, "mode_seed": t}
                    for c, m, s, t in points
                ],
                "baseline": baseline,
                "rng_state": rng.bit_generator.state,
            },
        )
        candidate_rows, training_episodes = [], []
        for index, weights in enumerate(candidates):
            rows, episodes = evaluate(
                weights, points, "T", f"round{round_index}-c{index}", True, candidate_index=index
            )
            candidate_rows.append(rows)
            training_episodes.extend(episodes)
        scores = [float(np.mean([utility(r) for r in rows])) for rows in candidate_rows]
        previous = center.copy()
        entropy, gradient_norm = 0.0, 0.0
        if method == "random":
            best = max(range(8), key=lambda i: rank(candidate_rows[i]))
            incumbent = candidates[best].copy()
            center = incumbent.copy()
        else:
            if method == "reinforce":
                gradient, entropy = reinforce_gradient(training_episodes, baseline)
                baseline = 0.8 * baseline + 0.2 * float(np.mean([x[0] for x in training_episodes]))
            else:
                gradient = es_gradient(directions, scores)
                all_decisions = [d for _, decisions in training_episodes for d in decisions]
                entropy = (
                    float(
                        np.mean(
                            [
                                -sum(p * np.log(max(p, 1e-30)) for p in d["probabilities"])
                                for d in all_decisions
                            ]
                        )
                    )
                    if all_decisions
                    else 0.0
                )
            gradient_norm = float(np.linalg.norm(gradient))
            gradient /= max(1.0, gradient_norm)
            center = np.clip(center + 0.2 * gradient, -3, 3)
        modes = np.bincount(
            [d["mode"] for _, decisions in training_episodes for d in decisions], minlength=6
        ).tolist()
        row = {
            "round": round_index,
            "mean_utility": float(np.mean(scores)),
            "reward_variance": float(np.var([x[0] for x in training_episodes])),
            "entropy": entropy,
            "gradient_norm_before_clip": gradient_norm,
            "coefficient_change_l2": float(np.linalg.norm(center - previous)),
            "mode_counts": modes,
            "candidate_scores": scores,
            "candidate_ranks": [list(rank(rows)) for rows in candidate_rows],
            "actual_resources": dict(executor.actual),
            "episodes": executor.used,
            "wall_seconds": time.time() - started,
        }
        history.append(row)
        write_json(
            folder / f"round{round_index:02d}-after.json",
            {
                "weights": center.tolist(),
                "baseline": baseline,
                "diagnostics": row,
                "rng_state": rng.bit_generator.state,
            },
        )
        write_json(folder / "learning-history.json", history)
        last_candidates, last_scores = candidates, candidate_rows
    if method == "random":
        order = sorted(range(8), key=lambda i: rank(last_scores[i]), reverse=True)[:3]
        finalists = [anchor] + [last_candidates[i] for i in order]
        labels = ["anchor"] + [f"last-round-c{i}" for i in order]
    else:
        finalists = [anchor, center, previous, initial]
        labels = ["anchor", "final", "penultimate", "initial"]
    write_json(
        folder / "validation-freeze.json",
        {
            "frozen_epoch": time.time(),
            "labels": labels,
            "weights": [w.tolist() for w in finalists],
            "rule": "Disqualify unsafe, then safe completions, capped J, predictor steps, earlier index",
            "validation_panel_sha256": settings["validation_panel_sha256"],
        },
    )
    val_points = [
        (validation[j], margin, 175000 + replicate * 100 + j, 17600000 + replicate * 100 + j)
        for j in range(3)
        for margin in MARGINS[env]
    ]
    validation_rows = []
    for index, weights in enumerate(finalists):
        rows, _ = evaluate(weights, val_points, "V", f"finalist{index}", False)
        validation_rows.append(rows)
    eligible = [i for i, rows in enumerate(validation_rows) if rank(rows)[0]]
    if not eligible:
        raise RuntimeError("All validation candidates unsafe; deployment not admitted")
    selected = max(eligible, key=lambda i: rank(validation_rows[i]))
    assert executor.used == executor.allowance
    write_json(
        folder / "selected.json",
        {
            "env": env,
            "method": method,
            "replicate": replicate,
            "weights": finalists[selected].tolist(),
            "selected_label": labels[selected],
            "selected_index": selected,
            "validation_ranks": [list(rank(rows)) for rows in validation_rows],
            "resources": executor.actual,
            "episodes": executor.used,
            "wall_seconds": time.time() - started,
            "gpu": args.gpu,
            "status": "complete",
            "optimizer_seed": base_seed,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=2)
    args = parser.parse_args()
    register()
    ready()
    configure(args.gpu)
    import torch

    torch.set_num_threads(1)
    manifest = verify_freeze(args.manifest)
    settings = manifest["extra"]
    for env in ("tworoom", "reacher"):
        assigned_index = 0
        for replicate in range(5):
            gpu = replicate % 2 + (2 if env == "reacher" else 0)
            if args.gpu != gpu:
                continue
            methods = ["random", "reinforce", "evolution"]
            methods = methods[replicate % 3 :] + methods[: replicate % 3]
            for method in methods:
                lane = assigned_index % args.lanes
                assigned_index += 1
                if args.lane == lane:
                    optimize(env, replicate, method, args, settings)


if __name__ == "__main__":
    main()
