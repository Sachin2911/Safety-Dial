"""Attribute ordered progress around an actual switch, using full factual traces.

Call only with a compound receipt already accepted by the physical reader. This
adds diagnostic attribution, not physical validation or permission to expand.
"""

import hashlib
import json
from pathlib import Path

from helpers.maniguard_continuation_handoff import MODES
from helpers.maniguard_review_progress import episode_progress


def _require(value, message):
    if not value:
        raise ValueError(message)


def _trace_binding(trace, receipt, steps, switch):
    contracts, choices, digest = [], [], hashlib.sha256()
    with Path(trace).open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            row = json.loads(raw)
            if row["type"] == "continuation_contract":
                contracts.append(row["contract"])
            elif row["type"] == "continuation_choice":
                choices.append(row)
    _require(len(contracts) == 1, "Require one actual continuation contract")
    contract = contracts[0]
    mode = receipt["stream"]["mode"]
    identity = hashlib.sha256(
        json.dumps(contract, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    _require(identity == receipt["continuation_contract_sha256"]
             == receipt["stream"]["contract_sha256"]
             and contract["mode"] == mode
             and contract["closure_controller_sha256"] == receipt["closure_controller_sha256"],
             "Full trace and verified receipt must describe the same selector")
    _require([row["step"] for row in choices] == list(range(0, steps, 8)),
             "Require all actual branch choices from the verified full trace")
    for row in choices:
        step, choice = row["step"], row["choice"]
        native = mode == "native_only" or (switch is not None and step >= switch)
        _require(choice["contract_sha256"] == identity and choice["query_step"] == step
                 and choice["branch"] == ("native" if native else "closure")
                 and choice["state"]["switch_query_step"] == (
                     switch if switch is not None and step >= switch else None),
                 "Actual selected branches disagree with the verified switch")
    _require(receipt["stream"]["state"]["last_query"] == choices[-1]["step"],
             "Verified selector did not finish on the actual last query")
    return digest.hexdigest()


def attribute_progress(trace, receipt):
    """Reuse SequenceProgress's full-episode milestones, including boundary observations."""
    _require(receipt.get("valid") is True and receipt.get("full_episode") is True
             and receipt["physical"]["family"] == "jar"
             and receipt["outcome"].get("valid") is True,
             "Require an already verified full Jar compound receipt")
    stream, outcome = receipt["stream"], receipt["outcome"]
    steps, mode = receipt["physical"]["steps"], stream["mode"]
    switch = stream["state"]["switch_query_step"]
    _require(type(steps) is int and 1 <= steps <= 2000 and mode in MODES
             and stream["complete_native_rng_chain_verified"] is True
             and stream["actual_environment_inputs_verified"] == steps,
             "Require the verified complete compound stream")
    _require(all(type(outcome[key]) is bool for key in ("C_req", "S_req", "V"))
             and outcome["S_req"] == (outcome["C_req"] and not outcome["V"]),
             "Preserve complete requested and physical-violation outcomes")
    _require(switch is None or (type(switch) is int and 0 <= switch < steps and switch % 8 == 0),
             "An actual switch must precede an executed native prefix")
    _require(mode not in ("native_only", "closure_only") or switch is None,
             "Native-only and closure-only references have no actual handoff")
    post = 0 if switch is None else steps - switch
    _require(type(stream["actual_post_switch_controls"]) is int
             and stream["actual_post_switch_controls"] == post,
             "Post-switch accounting differs from actual full-episode controls")
    actual_trace = Path(receipt["physical"]["output"]).resolve() / "trace.jsonl"
    _require(Path(trace).resolve() == actual_trace,
             "Use the full trace from the verified compound episode")
    digest = _trace_binding(trace, receipt, steps, switch)
    progress = episode_progress(trace, receipt)
    _require(progress["trace_sha256"] == digest, "Full trace changed during attribution")
    first = progress["ordered_stage_first_observations"]
    at_switch = None if switch is None else max(
        [0] + [stage for stage in range(1, 5)
               if first[str(stage)] is not None and first[str(stage)] <= switch]
    )
    milestones = {}
    for stage in (2, 3, 4):
        step = first[str(stage)]
        delta = None if step is None or switch is None else step - switch
        relation = ("not_observed" if step is None else "no_switch" if switch is None
                    else "before_switch" if delta < 0 else "at_switch" if delta == 0
                    else "after_switch")
        milestones[str(stage)] = {"first_observation": step, "controls_from_switch": delta,
                                  "relation": relation}
    # A clock of zero produces native execution immediately, without a closure prefix.
    handoff = (mode in ("clock_handoff", "privileged_lid_handoff")
               and switch is not None and switch > 0)
    new = [stage for stage in (2, 3, 4)
           if handoff and milestones[str(stage)]["relation"] == "after_switch"]
    safe_new = bool(new) and not outcome["V"]
    prefix = (0 if mode == "native_only" else steps) if switch is None else switch
    return {
        "schema": "maniguard-continuation-progress-attribution-v1",
        "trace_sha256": digest, "continuation_contract_sha256": stream["contract_sha256"],
        "mode": mode, "whole_ordered_stage": progress["ordered_stage_before_any_violation"],
        "whole_terminal_secondary": progress["ordered_stage_terminal_secondary"],
        "stage_at_actual_switch": at_switch, "switch_query_step": switch,
        "first_ordered_milestones": milestones, "actual_post_switch_controls": post,
        "closure_prefix_controls": prefix,
        "actual_handoff_observed": handoff, "new_ordered_milestones_after_switch": new,
        "new_ordered_milestone_strictly_after_switch": bool(new),
        "requested_completion": outcome["C_req"], "safe_requested_completion": outcome["S_req"],
        "raw_completion": outcome["benchmark_success"], "physical_violation": outcome["V"],
        "nonprivileged_handoff_progress_eligible": safe_new and mode == "clock_handoff",
        "privileged_engineering_progress_eligible": safe_new and mode == "privileged_lid_handoff",
        "dispatch_authorized": False, "expansion_authorized": False,
        "scope": "Full factual ordered milestones, using the existing episode_progress definition. "
                 "The switch observation is pre-switch. New later milestones are descriptive, "
                 "not proof of causal benefit; compare the declared references. A whole-episode "
                 "violation removes progress eligibility. Physical verification is consumed, "
                 "not repeated. No prefix termination or automatic next-stage authority.",
    }
