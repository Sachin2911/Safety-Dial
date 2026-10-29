"""Task 19 robustness checks for the confirmation result, written for the thesis.

Reads the counted outcomes in outputs/evo19/search/state.json and writes
outputs/evo19/analysis/sensitivity.json with:

- 95% Newcombe hybrid-score intervals for shield minus native, both endpoints (unpaired,
  matching the Fisher and Barnard tests), and the Bonferroni threshold for two endpoints;
- post hoc sensitivity: the shield was designed after some native confirmation attempts
  had finished (amendment 3, 23:51:27 UTC) and frozen at 01:07:13 UTC. The violation test
  is repeated without the pairs whose native attempt had finished by each of those times;
- the outcome transition table over the matched pairs (rows native, columns shield);
- the unpaired shield attempt, and the test with it added to the shield arm.

Descriptive checks only; the primary analysis is analyze_evo19.py.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "outputs/evo19/search/state.json"
OUT = ROOT / "outputs/evo19/analysis/sensitivity.json"
CUTS = {"amendment_3_shield_designed": "2026-10-26T23:51:27", "shield_frozen": "2026-10-27T01:07:13"}
ORDER = ("strict_safe", "violation", "stall_after_closure", "never_closed")


def category(o):
    if o["strict_safe"]:
        return "strict_safe"
    if o["raw_violation"]:
        return "violation"
    return "stall_after_closure" if o["closure_gate_opened"] else "never_closed"


def wilson(k, n, z=1.959964):
    centre = (k + z * z / 2) / (n + z * z)
    half = z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return centre - half, centre + half


def newcombe(k1, n1, k2, n2):
    """Difference k1/n1 - k2/n2 with the Newcombe hybrid-score 95% interval."""
    p1, p2 = k1 / n1, k2 / n2
    (l1, u1), (l2, u2) = wilson(k1, n1), wilson(k2, n2)
    d = p1 - p2
    return [round(100 * x, 1) for x in (d, d - np.hypot(p1 - l1, u2 - p2),
                                         d + np.hypot(u1 - p1, p2 - l2))]


def violations_test(pairs, native, shield):
    a = sum(native[p]["outcome"]["raw_violation"] for p in pairs)
    b = sum(shield[p]["outcome"]["raw_violation"] for p in pairs)
    n = len(pairs)
    p = fisher_exact([[b, n - b], [a, n - a]], alternative="less")[1]
    return {"pairs": n, "native": a, "shield": b, "fisher_one_sided_p": round(float(p), 4),
            "difference_newcombe95_points": newcombe(b, n, a, n)}


def main():
    jobs = json.loads(STATE.read_text())["jobs"].values()
    native = {(j["task"], j["seed"]): j for j in jobs
              if j["kind"] == "confirm_native" and j["status"] == "done"}
    shield = {(j["task"], j["seed"]): j for j in jobs
              if j["kind"] == "confirm_evolved" and j["status"] == "done"}
    pairs = sorted(set(native) & set(shield))
    assert len(pairs) == 107
    n = len(pairs)
    out = {"paired_task_seeds": n, "intervals": {}}
    for key in ("strict_safe", "raw_violation"):
        a = sum(native[p]["outcome"][key] for p in pairs)
        b = sum(shield[p]["outcome"][key] for p in pairs)
        out["intervals"][key] = {"native": a, "shield": b,
                                 "difference_newcombe95_points": newcombe(b, n, a, n)}
    assert (out["intervals"]["raw_violation"]["native"],
            out["intervals"]["raw_violation"]["shield"]) == (37, 21)
    out["bonferroni_two_endpoints_threshold"] = 0.025
    out["post_hoc_without_pairs_seen_before"] = {}
    for label, cut in CUTS.items():
        kept = [p for p in pairs if native[p]["finished_utc"][:19] > cut]
        out["post_hoc_without_pairs_seen_before"][label] = {
            "cut_utc": cut, "pairs_dropped": n - len(kept), **violations_test(kept, native, shield),
            "strict_safe_native": sum(native[p]["outcome"]["strict_safe"] for p in kept),
            "strict_safe_shield": sum(shield[p]["outcome"]["strict_safe"] for p in kept)}
    table = {r: {c: 0 for c in ORDER} for r in ORDER}
    for p in pairs:
        table[category(native[p]["outcome"])][category(shield[p]["outcome"])] += 1
    out["transitions_rows_native_columns_shield"] = table
    extra = sorted(set(shield) - set(native))
    out["unpaired_shield_attempts"] = [
        {"task": t, "seed": s, "category": category(shield[(t, s)]["outcome"])} for t, s in extra]
    v = sum(shield[p]["outcome"]["raw_violation"] for p in pairs + extra)
    out["shield_violations_including_unpaired"] = f"{v}/{len(pairs) + len(extra)}"
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
