"""Offline, post hoc Part 1 diagnostics. Never query a policy or advance physics."""

import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np

# A reporting threshold, not a changed controller or selection criterion.
TOLERANCE = 1e-6
TRACE_TYPES = ("proposal", "executed", "observed", "goal", "query", "chunk")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def index_trace(path):
    indexed = {kind: {} for kind in TRACE_TYPES}
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["type"] not in indexed:
                continue
            entries = indexed[row["type"]]
            if row["step"] in entries:
                raise ValueError(f"Duplicate {row['type']} step {row['step']}")
            entries[row["step"]] = row
    n = len(indexed["proposal"])
    for kind in ("proposal", "executed"):
        if set(indexed[kind]) != set(range(n)):
            raise ValueError(f"Noncontiguous {kind} evidence")
    for kind, expected in (("observed", range(n + 1)), ("goal", range(1, n + 1))):
        if set(indexed[kind]) != set(expected):
            raise ValueError(f"Missing or extra {kind} observations")
    if set(indexed["query"]) != set(indexed["chunk"]) or 0 not in indexed["query"]:
        raise ValueError("Missing query/chunk evidence")
    return indexed


def before_action_held(trace, step):
    # goal[k] is the observation after action k-1, available before action k.
    return None if step == 0 else trace["goal"][step]["detail"]["held"]


def first_violation(trace, native):
    if not native["ltl_violated"]:
        return None
    observed_step = native["ltl_violation_step"]
    action_step = observed_step - 1
    proposal = trace["proposal"][action_step]
    last_query = max(k for k in trace["query"] if k <= action_step)
    ap = trace["observed"][observed_step]["monitor"]["observation"]["ap"]
    # Upright propositions are positive, spilled/dropped propositions negative.
    failed = [k for k, v in ap.items() if (not v if k.endswith("upright") else v)]
    window = [trace["proposal"][k] for k in range(max(0, action_step - 19), action_step + 1)]
    return {
        "observed_step": observed_step,
        "action_step": action_step,
        "failed_propositions": failed,
        "last_query_step": last_query,
        "action_number_in_prefix": action_step - last_query + 1,
        "effective_horizon": proposal["event"]["effective_horizon"],
        "held_before_action": before_action_held(trace, action_step),
        "fingers_before_action_m": proposal["frame"]["fingers"],
        "grasp_heuristic": proposal["event"]["grasp_evidence"],
        "selected_target_error_cap_rad": proposal["event"]["step_limit"],
        "cap_active": proposal["event"]["cap_active"],
        "arm_change_rad": float(
            np.max(np.abs(np.array(proposal["adapted"][:7]) - proposal["original"][:7]))
        ),
        "cap_active_last_20_actions": sum(p["event"]["cap_active"] for p in window),
        "window_actions": len(window),
    }


def episode_diagnostics(trace, native):
    counts = Counter()
    angles = []
    first_change = first_cap = None
    for step, proposal in sorted(trace["proposal"].items()):
        event = proposal["event"]
        p, a, q = (
            np.array(x, dtype=float)
            for x in (
                proposal["original"][:7],
                proposal["adapted"][:7],
                proposal["frame"]["joints"],
            )
        )
        substantial = bool(np.max(np.abs(a - p)) > TOLERANCE)
        if substantial:
            counts["substantially_changed_actions"] += 1
            if first_change is None:
                first_change = step
            original_delta, adapted_delta = p - q, a - q
            norm = np.linalg.norm(original_delta) * np.linalg.norm(adapted_delta)
            if norm > 1e-12:
                angle = float(
                    np.degrees(
                        np.arccos(np.clip(np.dot(original_delta, adapted_delta) / norm, -1, 1))
                    )
                )
                angles.append(angle)
        if event["cap_active"] and first_cap is None:
            first_cap = step
        held = before_action_held(trace, step)
        if held is None:
            counts["initial_held_unknown_actions"] += 1
            continue
        predicted = event["grasp_evidence"]
        counts[f"held_{int(held)}_heuristic_{int(predicted)}"] += 1
        if held and proposal["original"][-1] < 0:
            counts["held_closing_actions"] += 1
            counts["held_closing_heuristic_actions"] += int(predicted)
            fingers = np.abs(proposal["frame"]["fingers"])
            outside = bool(np.any((fingers < 0.003) | (fingers > 0.035)))
            counts["held_closing_outside_finger_band"] += int(outside)
    keys = [f"held_{h}_heuristic_{p}" for h in (0, 1) for p in (0, 1)]
    keys += [
        "initial_held_unknown_actions",
        "substantially_changed_actions",
        "held_closing_actions",
        "held_closing_heuristic_actions",
        "held_closing_outside_finger_band",
    ]
    return {
        "counts": {k: counts[k] for k in keys},
        "first_substantial_change_step": first_change,
        "first_cap_active_step": first_cap,
        "direction_change": {
            "nonzero_changed_vectors": len(angles),
            "sum_angle_degrees": sum(angles),
            "max_angle_degrees": max(angles, default=None),
            "over_five_degrees": sum(a > 5 for a in angles),
        },
        "first_violation": first_violation(trace, native),
    }


def initial_pair(zero, other):
    a, b = (np.array(t["chunk"][0]["actions"]) for t in (zero, other))
    p0, p1 = (t["proposal"][0] for t in (zero, other))
    first = float(np.max(np.abs(np.array(p0["original"][:7]) - p1["original"][:7])))
    return {
        "state_identical": zero["query"][0]["state"] == other["query"][0]["state"],
        "image_hashes_identical": (
            zero["query"][0]["image_hashes"] == other["query"][0]["image_hashes"]
        ),
        "first_chunk_arm_max_difference_rad": float(np.max(np.abs(a[:, :7] - b[:, :7]))),
        "first_action_arm_max_difference_rad": first,
        "first_action_differs_over_tolerance": first > TOLERANCE,
        "either_first_action_adapted_over_tolerance": any(
            np.max(np.abs(np.array(p["adapted"]) - p["original"])) > TOLERANCE for p in (p0, p1)
        ),
    }


def build(run):
    run = Path(run).resolve()
    ledger_path = run / "audit-final/attempt-ledger.json"
    rows = [r for r in read(ledger_path)["attempts"] if r.get("purpose") in ("train", "validation")]
    if len(rows) != 48 or not all(r["valid"] and r["full_episode"] for r in rows):
        raise ValueError("This investigation requires the complete frozen 48-episode pilot")
    sources = [ledger_path, run / "protocol.json", run / "finalist.json"]
    episodes, initial = [], {}
    for row in rows:
        output = Path(row["output"])
        path, native_path = output / "trace.jsonl", output / "episode/results.jsonl"
        trace, native = index_trace(path), read(native_path)
        if len(trace["proposal"]) != row["steps"]:
            raise ValueError("Trace length differs from audited episode")
        episodes.append(
            {
                **{
                    k: row[k]
                    for k in (
                        "episode_id",
                        "purpose",
                        "scene",
                        "seed",
                        "dial",
                        "genome",
                        "steps",
                        "C_req",
                        "S_req",
                        "V",
                        "ever_grasped",
                    )
                },
                **episode_diagnostics(trace, native),
            }
        )
        if row["purpose"] == "validation":
            # Retain only the first input/output, keeping the analysis memory bounded.
            initial[row["episode_id"]] = {
                k: {0: trace[k][0]} for k in ("query", "chunk", "proposal")
            }
        sources.extend((path, native_path))
    pairs = []
    for _, group in itertools.groupby(
        sorted(
            (r for r in rows if r["purpose"] == "validation"), key=lambda r: (r["scene"], r["seed"])
        ),
        key=lambda r: (r["scene"], r["seed"]),
    ):
        group = {r["dial"]: r for r in group}
        for dial in (0.5, 1):
            zero, other = group[0], group[dial]
            pairs.append(
                {
                    "scene": zero["scene"],
                    "seed": zero["seed"],
                    "dial": dial,
                    **initial_pair(initial[zero["episode_id"]], initial[other["episode_id"]]),
                }
            )
    candidates, single_coordinate_pairs = [], []
    for generation in range(3):
        path = run / f"population-{generation}-result.json"
        result = read(path)
        sources.append(path)
        for slot, candidate in enumerate(result["population"]):
            candidates.append(
                {
                    "generation": generation,
                    "slot": slot,
                    **candidate,
                    "score": result["ranking"]["scores"][slot],
                    "safe_completions": sum(r["S_req"] for r in result["outcomes"][slot]),
                    "episodes": len(result["outcomes"][slot]),
                }
            )
        for i, j in itertools.combinations(range(4), 2):
            gi, gj = (result["population"][s]["genome"] for s in (i, j))
            changed = [k for k, (a, b) in enumerate(zip(gi, gj, strict=True)) if a != b]
            if len(changed) == 1:
                single_coordinate_pairs.append(
                    {
                        "generation": generation,
                        "slots": [i, j],
                        "genomes": [gi, gj],
                        "coordinate": changed[0],
                        "scores": [result["ranking"]["scores"][s] for s in (i, j)],
                    }
                )
    totals = []
    for dial in (0, 0.5, 1):
        selected = [r for r in episodes if r["purpose"] == "validation" and r["dial"] == dial]
        totals.append(
            {
                "dial": dial,
                "episodes": len(selected),
                "counts": {k: sum(r["counts"][k] for r in selected) for k in selected[0]["counts"]},
                "direction_change": {
                    k: (
                        max(r["direction_change"][k] or 0 for r in selected)
                        if k == "max_angle_degrees"
                        else sum(r["direction_change"][k] for r in selected)
                    )
                    for k in selected[0]["direction_change"]
                },
            }
        )
    return {
        "scope": "Exploratory CPU-only analysis of all 48 existing full episodes; no causal ablation.",
        "additional_policy_queries": 0,
        "additional_physics_steps": 0,
        "reporting_tolerance_rad": TOLERANCE,
        "alignment": "goal[k] follows action k-1; first violation at observed k uses action k-1.",
        "initial_held_rule": "Unknown at step zero; excluded from held-state denominators.",
        "sources_sha256": {str(p.relative_to(run)): sha(p) for p in sources},
        "diagnostic_code_sha256": {
            relative: sha(Path(__file__).resolve().parents[2] / relative)
            for relative in (
                "experiments/helpers/maniguard_part1_diagnostics.py",
                "experiments/scripts/maniguard_part1_diagnose.py",
                "experiments/tests/test_maniguard_part1_diagnostics.py",
            )
        },
        "comparison_totals": totals,
        "initial_pairs": pairs,
        "candidates": candidates,
        "within_generation_single_coordinate_pairs": single_coordinate_pairs,
        "episodes": episodes,
    }
