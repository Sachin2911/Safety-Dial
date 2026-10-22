"""Analysis of factual gripper commands and measured query-time finger positions.

Consumes the separately verified four-start screen. This is diagnostic evidence,
with no actor change, simulation, model call or authority to admit more work.
"""

import hashlib
import json
import math
from pathlib import Path
from statistics import median

from helpers.maniguard_continuation_progress import attribute_progress
from helpers.maniguard_continuation_protocol import validate_plan
from helpers.maniguard_continuation_results import read_episode


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _commands(rows):
    close = [r["step"] for r in rows if r["action"][7] == -1]
    return {"controls": len(rows), "closing_inputs": len(close),
            "opening_inputs": len(rows) - len(close),
            "first_closing_control_index": close[0] if close else None}


def _positions(rows):
    joints = list(zip(*(r["frame"]["fingers"] for r in rows)))
    return {"query_samples": len(rows), "query_steps": [r["step"] for r in rows],
            "per_joint": [{"minimum": min(v), "median": median(v), "maximum": max(v)}
                          for v in joints] if rows else [],
            "first_positions": rows[0]["frame"]["fingers"] if rows else None,
            "last_positions": rows[-1]["frame"]["fingers"] if rows else None}


def _states(rows, *, after=None):
    selected = [r for r in rows if after is None or r.get("step", -1) > after]
    observed = [r for r in selected if r["type"] == "observed"]
    goals = [r for r in selected if r["type"] == "goal"]
    held = [r["step"] for r in goals if r["detail"]["held"]]
    closed = [r["step"] for r in observed if r["monitor"]["observation"]["ap"]["jar_closed"]]
    doomed = [r["step"] for r in observed if r["monitor"]["observation"]["doomed"]]
    return {"monitor_observations": len(observed), "goal_observations": len(goals),
            "held_observations": len(held), "first_held_observation": held[0] if held else None,
            "closed_observations": len(closed),
            "first_closed_observation": closed[0] if closed else None,
            "first_doomed_observation": doomed[0] if doomed else None}


def audit_episode(trace, receipt):
    """Read a full trace tied to an already accepted compound physical receipt."""
    progress = attribute_progress(trace, receipt)
    raw = Path(trace).read_bytes()
    if _sha(raw) != progress["trace_sha256"]:
        raise ValueError("The full factual trace changed during gripper analysis")
    rows = [json.loads(line) for line in raw.splitlines()]
    executed = [r for r in rows if r["type"] == "executed"]
    queries = [r for r in rows if r["type"] == "latent_features"]
    steps = receipt["physical"]["steps"]
    if [r["step"] for r in queries] != list(range(0, steps, 8)):
        raise ValueError("Require every actual query-time finger measurement in order")
    for row in queries:
        fingers = row["frame"]["fingers"]
        if (not isinstance(fingers, list) or len(fingers) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in fingers)):
            raise ValueError("Require two finite measured finger joint positions per query")
    switch, mode = progress["switch_query_step"], progress["mode"]
    prefix_controls = progress["closure_prefix_controls"]
    prefix = executed[:prefix_controls]
    post = [] if switch is None else [r for r in executed if r["step"] >= switch]
    native = executed if mode == "native_only" else []
    commands = {"whole_episode": _commands(executed), "closure_prefix": _commands(prefix),
                "native_reference": _commands(native), "native_after_switch": _commands(post)}
    if (len(prefix) + len(native) + len(post) != steps
            or len(post) != progress["actual_post_switch_controls"]):
        raise ValueError("The gripper command phases do not partition the actual controls")
    measured = {"whole_episode": _positions(queries)}
    if switch is not None:
        for label, predicate in (
            ("strictly_before_switch", lambda step: step < switch),
            ("at_switch_before_first_native_action", lambda step: step == switch),
            ("strictly_after_switch", lambda step: step > switch),
        ):
            measured[label] = _positions([r for r in queries if predicate(r["step"])])
    return {
        "schema": "maniguard-review-handoff-gripper-episode-v1",
        "mode": mode, "switch_query_step": switch, "trace_sha256": _sha(raw),
        "continuation_contract_sha256": progress["continuation_contract_sha256"],
        "commands": commands, "query_finger_joint_positions": measured,
        "raw_state_observations": {"whole_episode": _states(rows),
                                   "strictly_after_switch": None if switch is None
                                   else _states(rows, after=switch)},
        "ordered_progress_attribution": progress,
        "scope": "Executed +/-1 inputs are opening/closing commands, not achieved finger motion. "
                 "Positions are the two measured joint coordinates at native query instants, "
                 "not fingertip aperture or a continuous motion record. Command index t acts "
                 "before observation t+1. Query positions at the switch are measured before "
                 "the first native action and are excluded from strictly-after statistics. "
                 "Raw holding/closure counts include unsafe observations and earn no credit. "
                 "No physical grasp-link or causal benefit is inferred from these counts.",
        "dispatch_authorized": False, "expansion_authorized": False,
    }


def analyze_screen(run, protocol_path):
    """Refuse an incomplete screen before opening any episode result."""
    run, protocol_path = Path(run).resolve(), Path(protocol_path).resolve()
    protocol_raw = protocol_path.read_bytes()
    protocol = json.loads(protocol_raw)
    validate_plan(protocol)
    if protocol["status"] != "prospectively_frozen" or run != Path(protocol["run_directory"]):
        raise ValueError("Use the separately frozen screen's canonical run")
    state_raw = (run / "screen.json").read_bytes()
    state = json.loads(state_raw)
    attempts = state["attempts"]
    planned = {row["id"]: row for row in protocol["episodes"]}
    if (state.get("screen_complete") is not True or len(attempts) != 4
            or any(a["status"] != "completed" for a in attempts)
            or {a["protocol_episode_id"] for a in attempts} != set(planned)
            or state["protocol_sha256"] != _sha(protocol_raw)):
        raise ValueError("All four declared starts must finish and verify before this analysis")
    reports = []
    for attempt in attempts:
        planned_row = planned[attempt["protocol_episode_id"]]
        output = run / "episodes" / (planned_row["id"] + "-a0")
        if (Path(attempt["output"]).resolve() != output
                or attempt["condition"] != planned_row["condition"]
                or attempt["worker"] != planned_row["worker"]
                or (output / "protocol.json").read_bytes() != protocol_raw):
            raise ValueError("Screen ledger, output or copied protocol is misbound")
        receipt_raw = (output / "verified-receipt.json").read_bytes()
        receipt = json.loads(receipt_raw)
        if receipt != read_episode(output):
            raise ValueError("Cached receipt differs from fresh full physical verification")
        audit = audit_episode(output / "trace.jsonl", receipt)
        if (audit["mode"] != planned_row["condition"]
                or audit["ordered_progress_attribution"]
                != json.loads((output / "progress-attribution.json").read_text())):
            raise ValueError("Cached attribution differs from the full factual gripper audit")
        reports.append({"attempt": attempt["id"], "worker": attempt["worker"],
                        "verified_receipt_sha256": _sha(receipt_raw), **audit})
    if (run / "screen.json").read_bytes() != state_raw:
        raise ValueError("The screen ledger changed during read-only analysis")
    return {
        "schema": "maniguard-review-handoff-gripper-screen-v1", "run": str(run),
        "protocol_sha256": _sha(protocol_raw), "screen_ledger_sha256": _sha(state_raw),
        "analysis_source_sha256": _sha(Path(__file__).read_bytes()), "episodes": reports,
        "reused_controls": sum(r["commands"]["whole_episode"]["controls"] for r in reports),
        "new_physical_starts": 0, "new_policy_queries": 0, "new_encoder_calls": 0,
        "dispatch_authorized": False, "expansion_authorized": False,
        "scope": "One completed development screen on its selected exposed task and seed. "
                 "Actual full verification is repeated; no original checking output is opened. "
                 "Gripper analysis changes no controller, fitness, outcome or qualification.",
    }
