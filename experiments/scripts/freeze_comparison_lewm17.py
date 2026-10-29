"""Admit a balanced schedule using engineering timing, before any optimizer outcome."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np

from helpers.lewm17_budget import ceilings
from helpers.lewm17_protocol import CONFIG, ENVIRONMENTS, OUT, digest, freeze, job, write_json
from helpers.lewm17_selector import MARGINS, MODES, mode_params


def main():
    rows = [json.loads(p.read_text()) for p in (OUT / "attempts").glob("*/result.json")]
    engineering = [r for r in rows if r["job"]["stage"] in ("E", "E2")]
    if len(engineering) != 48 or any(r["status"] != "complete" for r in engineering):
        raise RuntimeError("Complete and audit engineering before workload admission")
    audit = json.loads((ROOT / "progress/receipts/17-engineering-final-audit.json").read_text())
    assert audit["passed"]
    guard = json.loads((OUT / "control/guard-status.json").read_text())
    available = guard["budget"]["deadline_epoch"] - 1800 - time.time()
    # Reserve another 20 minutes before guard closeout for analysis and final preservation.
    # Pack nine runs on the busier GPU across two lanes, at most five runs per lane.
    per_episode = {}
    for env in ENVIRONMENTS:
        timing = [
            r["seconds"]
            for r in engineering
            if r["job"]["env"] == env and r["job"]["condition"] != "native"
        ]
        per_episode[env] = float(np.mean(timing)) * 1.5
    forecasts = {}
    for name, rounds, final_cases in [("full", 8, 20), ("compact", 6, 12)]:
        run_episodes = 48 + 24 * rounds
        # Additional allowance for uneven evaluation assignment and process/model start.
        train_wall = 5 * run_episodes * max(per_episode.values())
        eval_per_env = (72 + 288) / 2 + final_cases * 2 * 2 * 17
        evaluation_wall = eval_per_env / 4 * max(per_episode.values()) * 1.4
        total = train_wall + evaluation_wall + 600 + 1200
        forecasts[name] = {
            "rounds": rounds,
            "final_cases": final_cases,
            "forecast_seconds_including_reporting": total,
            "forecast_training_seconds": train_wall,
            "forecast_context_tuning_final_seconds": evaluation_wall,
            "admissible": total < available,
        }
    selected = next((name for name in ("full", "compact") if forecasts[name]["admissible"]), None)
    if selected is None:
        write_json(
            ROOT / "progress/receipts/17-workload-not-admitted.json",
            {
                "forecast_revision": "Initial serial p90-per-episode forecast rejected full and compact. Before optimizer outcomes, improve lane packing (max5 rather than6 runs) and use measured all-mode arithmetic mean with50% contingency; retain initial rejection receipt. Evaluation adds40% imbalance and30minutes overhead/reporting.",
                "forecasts": forecasts,
                "available_seconds": available,
            },
        )
        raise RuntimeError("Neither predeclared workload fits; freeze a balanced smaller amendment")
    schedule = forecasts[selected]
    rounds, final_cases = schedule["rounds"], schedule["final_cases"]
    run_episodes = 48 + 24 * rounds
    settings = {
        "workload": selected,
        "rounds": rounds,
        "final_cases": final_cases,
        "optimizer_runs_per_method_environment": 5,
        "methods": ["random", "reinforce", "evolution"],
        "episodes_per_run": run_episodes,
        "per_run_resource_ceilings": {
            e: {k: v * run_episodes for k, v in ceilings(e).items()} for e in ENVIRONMENTS
        },
        "equal_allowances_not_equal_consumption": True,
        "efficiency_claim": "No resource-matched validation checkpoints; no search-efficiency claim",
        "forecast_revision": "Initial serial p90-per-episode forecast rejected full and compact. Before optimizer outcomes, improve lane packing (max5 rather than6 runs) and use measured all-mode arithmetic mean with50% contingency; retain initial rejection receipt. Evaluation adds40% imbalance and30minutes overhead/reporting.",
        "forecasts": forecasts,
        "available_seconds_at_admission": available,
        "timing_seconds_per_episode_mean_times_1_5": per_episode,
        "mode_libraries": MODES,
        "coefficient_shape": [6, 6],
        "bounds": [-3, 3],
        "initialization": "Independent optimizer replicates; paired methods share clipped N(0,0.5) initialization",
        "parameter_seed": "17020000 + (1000 for Reacher) + 10000*replicate",
        "optimizer_rng": "parameter seed + 1 random, +2 REINFORCE, +3 evolution",
        "feature_vector": [
            "bias=1",
            "relative latent cost improvement clipped [-1,1]",
            "prediction squared error/current latent cost floored at 1; clipped [0,2]/2",
            "previous feasible fraction clipped [0,1]",
            "replan/9 clipped [0,1]",
            "requested clearance/max tested clearance",
        ],
        "first_observation": "improvement=0, prediction_error=0, previous_feasible=1, replan=0",
        "training_decoder": "softmax temperature 1; separate mode RNG, not planner RNG",
        "deployment_decoder": "deterministic argmax, smallest mode index breaks ties",
        "anchors": "Six bias-only constant modes and two opposing margin routes; deterministic; 8*3 episodes charged per run",
        "utility": "requested_safe_completion - 0.1*J/50; J=50 on unsafe or unsuccessful episodes",
        "random": "One re-evaluated incumbent plus seven fresh clipped N(0,0.5) candidates each round; eight candidates on common three points; never compare stale scores",
        "evolution": "Four standard-normal antithetic directions; sigma=.3; raw mean utility pair differences/(2*sigma*4); gradient L2 clipped to 1; ascent step=.2; weights clipped [-3,3]",
        "reinforce": "Eight groups of three stochastic episodes; terminal reward times sum of log-softmax score gradients; batch mean over 24; baseline starts 0 and updates .8*old+.2*mean reward after gradient; L2 clip1; step .2; weights clipped [-3,3]; no entropy bonus",
        "validation": "24 deterministic episodes: 4 candidates x first3 validation cases x2 margins; random finalists best3 on last common batch plus anchor; other methods final/penultimate/initial plus anchor; all selected only by validation",
        "selection": "Disqualify any original or margin breach; then safe completions, lower failure-capped J, lower predictor work, earlier candidate index; anchor is first fallback; all-unsafe aborts",
        "fixed_tuning": "six modes x six development cases x two margins x two planner seeds per environment; same safety-first lexicographic selection separately per margin",
        "validation_panel_sha256": digest(CONFIG / "validation.json"),
        "final_panel_sha256": digest(CONFIG / "sealed/final.json"),
        "final_access": "Only finalization script opens sealed cases after every selected controller is frozen; log access",
        "practical_effect_target": 0.10,
        "primary": "Reacher at .015, evolution minus random, REINFORCE, newly tuned fixed",
        "uncertainty": "Paired crossed bootstrap resampling five optimizer-run indices and whole final case blocks independently; retain both planner seeds per block; 10000 draws seed170990; 95% descriptive intervals and 98.333333% primary contrast intervals for three comparisons",
        "claim_gate": "For all three primary comparators: mean completion gain >=.10 and simultaneous contrast interval lower>0; no increased breaches, with observed rates and uncertainty reported. No broad safety guarantee",
        "all_evidence": "Report every optimizer replicate, engagement, both safety labels, censoring, unfinished, J, all counters, rejected candidates, failed attempts and wall time",
        "placement": "Single host33061; replicate%2 GPU plus2 for Reacher; paired methods same GPU; runs packed alternately across two lanes; method execution order rotates by replicate",
    }
    write_json(CONFIG / "comparison.json", settings)
    development = json.loads((CONFIG / "development.json").read_text())
    jobs = []
    for env in ENVIRONMENTS:
        for case in development[env][:3]:
            for margin in MARGINS[env]:
                for seed in (170110, 170111, 170112):
                    for condition in ("native", "canonical"):
                        jobs.append(job(env, "C", case, margin, seed, condition))
        for case in development[env][:6]:
            for margin in MARGINS[env]:
                for seed in (170210, 170211):
                    for mode in range(6):
                        jobs.append(
                            job(env, "F", case, margin, seed, f"mode{mode}", mode_params(env, mode))
                        )
    assert len(jobs) == 360
    print(freeze("context-fixed", jobs, settings))
    print(freeze("optimizer-comparison", extra=settings))


if __name__ == "__main__":
    main()
