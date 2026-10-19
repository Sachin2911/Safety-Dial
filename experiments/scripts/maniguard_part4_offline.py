"""Execute the frozen Part 4 offline premise check without simulator or VLA calls."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from helpers.maniguard_part4_data import load_donors
from helpers.maniguard_part4_model import (
    calibrated,
    calibration_rule,
    discrimination,
    event_metrics,
    fit_predictors,
)
from helpers.maniguard_part4_protocol import CONFIG, ROOT, RUN, atomic, clock, digest, read


def evaluate(episodes, predictions, rules):
    result = {}
    for group in ("fit", "calibration", "development"):
        selected = [e for e in episodes if e["group"] == group]
        result[group] = {}
        for name, rule in rules.items():
            outcomes, labels, probabilities, warning_p, warning_y = [], [], [], [], []
            for episode in selected:
                raw = predictions[episode["episode_id"]]
                p = calibrated(raw[name], rule)
                outcomes.append(event_metrics(episode, p, raw["uncertainty"], rule))
                labels.append(episode["y"])
                probabilities.append(p)
                tau = episode["onset"]
                warning = np.zeros(len(p), dtype=bool) if tau is None else (
                    (episode["steps"] >= tau - 8) & (episode["steps"] <= tau - 4))
                keep = warning | (episode["y"] == 0)
                warning_p.append(p[keep])
                warning_y.append(episode["y"][keep])
            y, p = np.concatenate(labels), np.concatenate(probabilities)
            eligible = [o for o in outcomes if o["eligible_event"]]
            negative_windows = sum(o["negative_windows"] for o in outcomes)
            false_windows = sum(o["false_alarm_windows"] for o in outcomes)
            eligible_fa = [o for o in outcomes if o["episode_false_alarm"] is not None]
            wp, wy = np.concatenate(warning_p), np.concatenate(warning_y)
            result[group][name] = {
                **discrimination(y, p), "episodes": outcomes,
                "episode_count": len(selected), "task_count": len({e["task"] for e in selected}),
                "event_episodes": sum(e["onset"] is not None and e["onset"] > 0 for e in selected),
                "eligible_events": len(eligible), "missed_timely_warnings": sum(
                    not o["timely_warning"] for o in eligible),
                "confident_missed_events": sum(o["confident_miss"] for o in eligible),
                "event_tasks": len({o["task"] for o in eligible}),
                "negative_windows": negative_windows, "false_alarm_windows": false_windows,
                "negative_window_false_alarm": false_windows / negative_windows if negative_windows else None,
                "episode_false_alarm_count": sum(o["episode_false_alarm"] for o in eligible_fa),
                "episode_false_alarm_denominator": len(eligible_fa),
                "eligible_warning_windows": int(wy.sum()),
                "missed_warning_windows": int(((wp < rule["threshold"]) & (wy == 1)).sum()),
                "all_positive_windows": int(y.sum()), "total_windows": len(y),
                "threshold_sweep": [{"threshold": threshold,
                    "negative_window_false_alarm": float(np.mean(p[y == 0] >= threshold)),
                    "positive_window_miss": float(np.mean(p[y == 1] < threshold)) if y.any() else None}
                    for threshold in read(CONFIG / "offline-protocol.json")["operating_rule"]["threshold_sweep"]],
                "always_safe": {"event_misses": len(eligible), "false_alarm_windows": 0},
                "always_alarm": {"event_misses": 0, "false_alarm_windows": negative_windows},
            }
    return result


def premise_gate(metrics):
    criteria = read(CONFIG / "offline-protocol.json")["premise_gate"]
    fit, cal, dev = (metrics[g]["primary"] for g in ("fit", "calibration", "development"))
    support = {
        "fit_event_episodes": fit["eligible_events"] >= criteria["minimum_fit_event_episodes"],
        "fit_event_tasks": fit["event_tasks"] >= criteria["minimum_fit_event_tasks"],
        "calibration_event_episodes": cal["eligible_events"] >= criteria["minimum_calibration_event_episodes"],
        "development_event_episodes": dev["eligible_events"] >= criteria["minimum_development_event_episodes"],
        "development_event_tasks": dev["event_tasks"] >= criteria["minimum_development_event_tasks"],
    }
    accuracy = {
        "auroc": dev["auroc"] is not None and dev["auroc"] >= criteria["minimum_development_auroc"],
        "auprc": dev["auprc"] is not None and dev["auprc"] >= criteria["minimum_development_auprc"],
        "auprc_over_prevalence": dev["auprc"] is not None and dev["auprc"] >= (
            criteria["minimum_auprc_over_prevalence"] * dev["prevalence"]),
        "false_alarm_behavior": dev["negative_window_false_alarm"] is not None and (
            dev["negative_window_false_alarm"] <= criteria["maximum_development_negative_window_false_alarm"]),
    }
    misses = dev["confident_missed_events"] >= criteria["minimum_confident_missed_development_events"]
    passed = all(support.values()) and all(accuracy.values()) and misses
    status = "passed" if passed else (
        "failed_discrimination" if not all(accuracy.values()) else
        "inconclusive_event_support" if not all(support.values()) else "absent_confident_misses")
    return {"eligible": passed, "status": status, "support": support,
            "discrimination": accuracy, "confident_misses": misses,
            "failed_checks": [k for k, v in {**support, **accuracy, "confident_misses": misses}.items() if not v],
            "shared_initial_eligible": all(accuracy.values()) and not passed,
            "scope": "One bounded frozen-target premise check on exposed historical development tasks, not a fresh final audit"}


def run_offline(run=RUN):
    began = time.time()
    run = Path(run)
    if (run / "offline-audit.json").exists():
        raise FileExistsError("An offline premise check is not silently repeated")
    if clock()["physical_remaining_seconds"] < 600:
        raise RuntimeError("Insufficient budget before cleanup reserve")
    episodes = load_donors(run)
    directory = run / "models/baseline"
    predictor, ledger = fit_predictors(episodes, directory)
    predictions = {e["episode_id"]: predictor.predict(e) for e in episodes}
    calibration = [e for e in episodes if e["group"] == "calibration"]
    y = np.concatenate([e["y"] for e in calibration])
    weights = np.concatenate([np.full(len(e["y"]), 1 / len(e["y"])) for e in calibration])
    disagreement = np.concatenate([predictions[e["episode_id"]]["uncertainty"] for e in calibration])
    rules = {}
    for name in ("primary", "logistic", "visual"):
        raw = np.concatenate([predictions[e["episode_id"]][name] for e in calibration])
        rule = calibration_rule(raw, y, weights, disagreement)
        rules[name] = {k: v for k, v in rule.items() if k != "probability"}
    atomic(directory / "operating-rules.json", rules)
    source_paths = [CONFIG / "offline-protocol.json", CONFIG / "allocation.json", *sorted(
        (ROOT / "experiments/helpers").glob("maniguard_part4_*.py")), Path(__file__)]
    frozen = {"at": time.time(), "protocol_sha256": digest(CONFIG / "offline-protocol.json"),
              "sources": {str(p.relative_to(ROOT)): digest(p) for p in source_paths},
              "files": {p.name: digest(p) for p in directory.iterdir() if p.is_file()},
              "stage": "baseline_and_operating_rule_frozen_before_development_predictions_scored",
              "development_tasks_previously_exposed": True, "fresh_audit_accessed": False}
    atomic(directory / "baseline-freeze.json", frozen)
    metrics = evaluate(episodes, predictions, rules)
    gate = premise_gate(metrics)
    windows = {}
    for i, e in enumerate(episodes):
        raw = predictions[e["episode_id"]]
        windows[f"steps_{i}"] = e["steps"]
        windows[f"labels_{i}"] = e["y"]
        windows[f"members_{i}"] = raw["members"]
        windows[f"probability_{i}"] = calibrated(raw["primary"], rules["primary"])
        windows[f"uncertainty_{i}"] = raw["uncertainty"]
    np.savez_compressed(run / "offline-predictions.npz", **windows)
    audit = {"schema": "maniguard-part4-offline-audit-v1", "gate": gate,
             "metrics": metrics, "rules": rules, "clock": clock(),
             "seconds": time.time() - began, "training_seconds": ledger["seconds"],
             "donor_audit_sha256": digest(run / "donor-audit.json"),
             "baseline_freeze_sha256": digest(directory / "baseline-freeze.json"),
             "predictions_sha256": digest(run / "offline-predictions.npz"),
             "new_physical_controls": 0, "new_physics_steps": 0,
             "new_VLA_queries": 0, "new_encoder_calls": 0,
             "model_window_evaluations": {"primary_members": 5 * sum(len(e["steps"]) for e in episodes),
                                           "logistic_references": 2 * sum(len(e["steps"]) for e in episodes)},
             "fresh_audit_opened": False, "method_specific_episodes": 0}
    atomic(run / "offline-audit.json", audit)
    print(json.dumps({"gate": gate, "metrics": {g: {n: {k: m[k] for k in (
        "auroc", "auprc", "prevalence", "eligible_events", "missed_timely_warnings",
        "confident_missed_events", "negative_window_false_alarm")}
        for n, m in names.items()} for g, names in metrics.items()}, "seconds": audit["seconds"]}), flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    run_offline(parser.parse_args().run)
