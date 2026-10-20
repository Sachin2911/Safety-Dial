"""Factual Jar milestone audit; original scores remain selection's authority."""

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from helpers.maniguard_model_control import SequenceProgress
from helpers.maniguard_review_sensitivity import ROOT, TRAINING_STAGES


def episode_progress(path, receipt):
    """Join actual executed commands to before/after monitor and goal observations."""
    observed, goals, executed, scopes = {}, {}, {}, Counter()
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            step = row.get("step")
            if row["type"] == "observed":
                monitor = row["monitor"]
                if monitor.get("valid") is not True:
                    raise ValueError("Invalid factual monitor")
                observation = monitor["observation"]
                ap = observation["ap"]
                values = {"closed": ap["jar_closed"], "supported": ap["jar_on_support"],
                          "violated": observation["doomed"]}
                if any(type(value) is not bool for value in values.values()) or step in observed:
                    raise ValueError("Invalid or duplicate factual observation")
                observed[step] = values
            elif row["type"] == "goal":
                values = {key: row["detail"][key] for key in ("held", "intersects")}
                if any(type(value) is not bool for value in values.values()) or step in goals:
                    raise ValueError("Invalid or duplicate factual goal")
                goals[step] = values
            elif row["type"] == "executed":
                action = row["action"]
                if (step in executed or len(action) != 8
                        or any(type(x) not in (int, float) or not math.isfinite(x) for x in action)
                        or action[7] not in (-1, 1)):
                    raise ValueError("Invalid or duplicate executed command")
                executed[step] = action
                scopes[row.get("action_scope", "unannotated_in_trace")] += 1
    n = receipt["physical"]["steps"]
    if (list(observed) != list(range(n + 1)) or list(goals) != list(range(1, n + 1))
            or list(executed) != list(range(n))):
        raise ValueError("Incomplete or unordered factual episode")
    sequence, violation, first_doom = SequenceProgress(), False, None
    first_closed, first_safe_closed, first_held = None, None, None
    milestone_steps = {str(stage): None for stage in range(1, 5)}
    closed_supported_observations = close_commands = close_after_safe_closure = 0
    for step in range(n + 1):
        state = observed[step]
        violation |= state["violated"]
        if violation and first_doom is None:
            first_doom = step
        if state["closed"] and first_closed is None:
            first_closed = step
        safe_closed = state["closed"] and state["supported"] and not violation
        if safe_closed:
            closed_supported_observations += 1
            if first_safe_closed is None:
                first_safe_closed = step
        if step < n and executed[step][7] == -1:
            close_commands += 1
            close_after_safe_closure += int(safe_closed)
        if step:
            goal = goals[step]
            if goal["held"] and first_held is None:
                first_held = step
            before = sequence.stage
            sequence.observe(closed=state["closed"], held=goal["held"],
                             supported=state["supported"], in_goal=goal["intersects"],
                             violated=violation)
            for stage in range(before + 1, sequence.stage + 1):
                milestone_steps[str(stage)] = step
    outcome = receipt["outcome"]
    if (type(outcome["V"]) is not bool or violation != outcome["V"]
            or receipt["score"] != int(outcome["C_req"] and not violation) - 2 * int(violation)):
        raise ValueError("Factual monitor or utility disagrees with the receipt")
    return {
        "trace_sha256": digest.hexdigest(), "controls": n, "observations": n + 1,
        "first_closed_observation": first_closed,
        "first_closed_supported_before_violation": first_safe_closed,
        "closed_supported_observations_before_violation": closed_supported_observations,
        "first_held_observation": first_held, "first_doomed_observation": first_doom,
        "executed_close_commands": close_commands,
        "executed_close_commands_after_safe_supported_closure": close_after_safe_closure,
        "ordered_stage_before_any_violation": sequence.stage,
        "ordered_stage_terminal_secondary": 0 if violation else sequence.stage,
        "ordered_stage_first_observations": milestone_steps,
        "executed_action_scope_annotations": dict(scopes),
    }


def _summary(episodes):
    return {
        "episodes": len(episodes), "requested_completions": sum(e["C_req"] for e in episodes),
        "raw_completions": sum(e["benchmark_success"] for e in episodes),
        "safe_requested_completions": sum(e["S_req"] for e in episodes),
        "violating_episodes": sum(e["V"] for e in episodes),
        "safe_unfinished_episodes": sum(not e["V"] and not e["C_req"] for e in episodes),
        "safe_unfinished_stage_histogram": dict(sorted(Counter(
            str(e["ordered_stage_terminal_secondary"]) for e in episodes
            if not e["V"] and not e["C_req"]).items())),
        "episodes_with_safe_supported_closure": sum(
            e["first_closed_supported_before_violation"] is not None for e in episodes),
        "episodes_with_executed_close_commands": sum(e["executed_close_commands"] > 0
                                                    for e in episodes),
        "episodes_with_close_commands_after_safe_closure": sum(
            e["executed_close_commands_after_safe_supported_closure"] > 0 for e in episodes),
    }


def analyze(run):
    queue_bytes = (Path(run) / "queue.json").read_bytes()
    queue = json.loads(queue_bytes)
    episodes, excluded, panels = [], Counter(), defaultdict(list)
    for attempt in queue["attempts"]:
        if attempt["stage"] not in TRAINING_STAGES or attempt["status"] != "completed":
            excluded[attempt["stage"]] += 1
            continue  # Exclude checking before opening any output path.
        output = Path(attempt["output"])
        output = output if output.is_absolute() else ROOT / output
        job_bytes = (output / "job.json").read_bytes()
        job = json.loads(job_bytes)
        if job["family"] != "jar":
            continue
        receipt_bytes = (output / "verified-receipt.json").read_bytes()
        receipt = json.loads(receipt_bytes)
        if receipt["valid"] is not True or receipt["full_episode"] is not True:
            raise ValueError("Completed episode lacks full factual verification")
        if job["controller_sha256"] != receipt["controller_sha256"]:
            raise ValueError("Job and receipt controller identity differ")
        outcome = receipt["outcome"]
        item = {
            "attempt": attempt["id"], "stage": attempt["stage"], "scene": job["scene"],
            "seed": job["seed"], "controller_sha256": receipt["controller_sha256"],
            "job_sha256": hashlib.sha256(job_bytes).hexdigest(),
            "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "original_terminal_score": receipt["score"],
            **{key: outcome[key] for key in ("C_req", "S_req", "V", "benchmark_success")},
            **episode_progress(output / "trace.jsonl", receipt),
        }
        episodes.append(item)
        match = re.fullmatch(r"(.+)-case([01])-a\d+", attempt["id"])
        if match:
            panels[match[1]].append((int(match[2]), item))
    comparison = [e for e in episodes if e["stage"] != "development"]
    panel_results = []
    for identity, rows in sorted(panels.items()):
        if sorted(case for case, _ in rows) != [0, 1]:
            continue
        items = [item for _, item in sorted(rows)]
        panel_results.append({
            "candidate": identity, "attempts": [e["attempt"] for e in items],
            "original_scores": [e["original_terminal_score"] for e in items],
            "original_score_sum": sum(e["original_terminal_score"] for e in items),
            "terminal_ordered_stages": [e["ordered_stage_terminal_secondary"] for e in items],
            "sum_of_diagnostic_ordered_stages": sum(e["ordered_stage_terminal_secondary"]
                                                     for e in items),
        })
    return {
        "schema": "maniguard-review-ordered-progress-v1",
        "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sequence_source_sha256": hashlib.sha256(
            (ROOT / "experiments/helpers/maniguard_model_control.py").read_bytes()).hexdigest(),
        "jar_development_and_comparison": _summary(episodes),
        "comparison_only": _summary(comparison),
        "episodes": episodes, "complete_two_case_training_panels": panel_results,
        "excluded_attempt_counts_by_stage": dict(excluded),
        "new_physical_controls": 0, "new_vla_requests": 0, "new_encoder_images": 0,
        "new_learned_model_training_or_queries": 0,
        "scope": "Post hoc diagnostic on completed Jar development/training only. "
                 "Original terminal utility and selection remain unchanged. Stage labels are "
                 "privileged training supervision, not actor inputs or completion verdicts. "
                 "Repeated observations are not independent episodes. Executed records are "
                 "environment commands, not achieved motor positions. Existing full verification "
                 "is consumed rather than independently repeated. No independent checking is read.",
    }
