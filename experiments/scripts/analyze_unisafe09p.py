"""Analyze the complete amended confirmation with physical-host stratification."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.dsrl06_infra import atomic_json, utc
from helpers.unisafe09_protocol import digest
from helpers.unisafe09p_statistics import blocked_binary, blocked_mean, blocked_rate, matched_groups
from scripts.analyze_unisafe09 import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text())
    original_path = ROOT / "progress/receipts/09-unisafe-C.json"
    freeze = json.loads(original_path.read_text())
    if not validation["passed"] or validation["freeze_sha256"] != digest(original_path):
        raise ValueError("A validated complete C scientific panel is required")
    rows = validation["episodes"]
    if len(rows) != 384 or len({r["job"]["id"] for r in rows}) != 384:
        raise ValueError("Confirmation requires all 384 distinct planned episodes")
    hosts = sorted({r["physical_host"] for r in rows})
    attempted = Counter(r["condition"] for r in rows)
    job_condition = {j["id"]: j["condition"] for j in freeze["jobs"]}
    for row in validation["invalid_attempts"]:
        attempted[job_condition[row["job_id"]]] += 1
    combined = summarize(rows, attempted)
    for cid, summary in combined.items():
        group = [r for r in rows if r["condition"] == cid]
        summary["C_host_interval95"] = blocked_rate(group, "safe_success")
        summary["V_host_interval95"] = blocked_rate(group, "violation")
        summary["host_interval_method"] = "Host-weighted exact binomial bounds; error .05/H"
    per_host = {}
    per_gpu = {}
    for host in hosts:
        subset = [r for r in rows if r["physical_host"] == host]
        counts = Counter(r["condition"] for r in subset)
        for invalid in validation["invalid_attempts"]:
            expected_label = "original" if host == "150513" else "parallel"
            if invalid["host"] == expected_label:
                counts[job_condition[invalid["job_id"]]] += 1
        per_host[host] = summarize(subset, counts)
        for lane in sorted({r["gpu_lane"] for r in subset}):
            group = [r for r in subset if r["gpu_lane"] == lane]
            per_gpu[f"{host}/gpu{lane}"] = summarize(group, Counter(r["condition"] for r in group))
    selection = freeze["selection"]
    contrasts = selection["primary_contrasts"]
    rng = np.random.default_rng(2026102609)
    draws = {}
    for host in hosts:
        n = len({r["seed"] for r in rows if r["physical_host"] == host})
        draws[host] = rng.integers(0, n, size=(10000, n))

    def compare(candidate, reference, alpha):
        groups = matched_groups(rows, candidate, reference)
        return {"candidate": candidate, "reference": reference,
                "C": blocked_binary(groups, "safe_success", alpha),
                "V": blocked_binary(groups, "violation", alpha),
                "J": blocked_mean(groups, draws, alpha=alpha)}

    alpha = .05/(3*len(contrasts)) if contrasts else .05
    simultaneous = [compare(a, b, alpha) for a, b in contrasts]
    speed, completion = selection["speed_endpoint"], selection["completion_endpoint"]
    supported = False
    for row in simultaneous:
        if row["candidate"] == completion and row["reference"] == speed:
            supported |= row["C"]["interval"][0] > 0 and row["J"]["interval"][0] > 0
        if row["candidate"] == speed and row["reference"] == completion:
            supported |= row["C"]["interval"][1] < 0 and row["J"]["interval"][1] < 0
    result = {
        "utc": utc(), "stage": "C", "conditions": combined, "conditions_by_host": per_host,
        "conditions_by_gpu_descriptive": per_gpu, "selection_estimates": False,
        "optimizer_updates": 0, "execution_protocol": validation["execution_protocol"],
        "estimand": "Observed physical-host mixture weighted by paired seed count",
        "validation_sha256": digest(args.validation),
        "primary_descriptive95": [compare(a, b, .05) for a, b in contrasts],
        "primary_simultaneous": simultaneous, "familywise_alpha": .05,
        "adjusted_endpoint_alpha": alpha, "supported_speed_completion_tradeoff": bool(supported),
        "risk_nonincrease_supported": [{"candidate": r["candidate"], "reference": r["reference"],
                                        "supported": r["V"]["interval"][1] <= 0}
                                       for r in simultaneous],
        "analysis_source_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in (
                Path(__file__).resolve(), ROOT / "experiments/helpers/unisafe09p_statistics.py"
            )
        },
    }
    atomic_json(args.output, result)
    print(json.dumps({"conditions": combined, "primary_simultaneous": simultaneous,
                      "supported_speed_completion_tradeoff": bool(supported)}))


if __name__ == "__main__":
    main()
