"""Prospectively declared nomination, selection and panel ordering for v3."""

import numpy as np

from maniguard_evolution_scoring import candidate_score, summarize


def nominate_elite(generations, initial):
    eligible = []
    for generation in generations:
        g = generation["generation"]
        for i, (score, parameters) in enumerate(zip(
                generation["scores"], generation["parameters"], strict=True)):
            if not np.array_equal(parameters, initial):
                eligible.append((float(score), -g, -i, parameters))
    if not eligible:
        raise ValueError("No evaluated non-anchor controller can be nominated")
    score, minus_g, minus_i, parameters = max(eligible, key=lambda row: row[:3])
    return {"parameters": list(parameters), "generation": -minus_g,
            "candidate_index": -minus_i, "training_fitness": score}


def task_major_cases(panel):
    return [(task, seed) for task in panel["tasks"] for seed in panel["policy_seeds"]]


def selection_schedule(panel):
    kinds = ["native", "elite", "mean"]
    return [(task, seed, kinds[i % 3:] + kinds[:i % 3])
            for i, (task, seed) in enumerate(task_major_cases(panel))]


def choose_selection(rows, objective="completion_progress_contact_v3"):
    kinds = ("native", "elite", "mean")
    if set(rows) != set(kinds) or not rows["native"]:
        raise ValueError("Selection requires native, elite and mean on a nonempty panel")
    n = len(rows["native"])
    if any(len(rows[kind]) != n or any(not row.get("valid") for row in rows[kind])
           for kind in kinds):
        raise ValueError("Every selection candidate requires the whole valid panel")
    summaries = {kind: summarize(rows[kind]) for kind in kinds}
    scores = {kind: candidate_score(rows[kind], objective) for kind in kinds}
    baseline = summaries["native"]["strict_safe_completion"]
    eligible = [kind for kind in ("elite", "mean")
                if summaries[kind]["strict_safe_completion"] > baseline]
    # max retains the first entry on an exact tie, so the evaluated elite wins ties.
    winner = max(eligible, key=lambda kind: scores[kind]) if eligible else "native"
    return {"winner": winner, "case_count_per_candidate": n,
            "summary": summaries, "scores": scores, "eligible_adapters": eligible,
            "native_fallback": winner == "native",
            "rule": "Strict completion must exceed native; maximize v3 fitness; elite wins ties"}
