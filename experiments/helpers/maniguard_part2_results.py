"""Audit physical feedback-controller evidence; partial/failed attempts retain separate costs."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from helpers.maniguard_part2_controller import Config, FeedbackController, config_hash
from helpers.maniguard_phase2_outcomes import derive_outcome
from helpers.maniguard_phase3_protocol import sha
from helpers.maniguard_results import read_episode as read_physical

COSTS = (
    "physics_steps",
    "physics_seconds",
    "env_step_attempts",
    "env_steps_returned",
    "policy_queries",
    "policy_queries_completed",
    "policy_queries_failed",
    "policy_query_seconds",
    "policy_control_steps",
    "adapter_seconds",
    "initialization_seconds",
    "elapsed_seconds",
    "changed_steps",
    "passive_capture_seconds",
    "passive_frames",
)


def read_json(path):
    return json.loads(
        Path(path).read_text(),
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + x)),
    )


def exactly(a, b, description):
    if not np.array_equal(a, b):
        raise ValueError("Replay mismatch: " + description)


def replay(job, trace, steps):
    adapter = FeedbackController(Config(**job["controller"]))
    counts = {"proposal": 0, "executed": 0, "observed": 0, "goal": 0, "query": 0, "correction": 0}
    current_observation = -1
    current_step = 0
    chunk = None
    query_step = None
    used = 0
    proposal = clipped = None
    chunk_lengths = []
    activations = []
    for row in trace:
        kind, step = row.get("type"), row.get("step")
        if kind in counts:
            counts[kind] += 1
        if kind == "observed":
            if step != current_observation + 1 or step != current_step:
                raise ValueError("Missing or noncausal observation")
            state = row.get("monitor") or {}
            if state.get("valid") is not True or state.get("error") is not None:
                raise ValueError("Invalid physical monitor sample")
            current_observation = step
        elif kind == "query":
            if step != current_observation or step != current_step:
                raise ValueError("Query is not causal")
            if (query_step is not None and used != min(adapter.horizon, len(chunk))
                    and adapter.replan_reason(None) is None):
                raise ValueError("Unscheduled query")
            query_step, used = step, 0
            adapter.on_query(None)
        elif kind == "chunk":
            if step != query_step:
                raise ValueError("Chunk lacks its query")
            chunk = np.asarray(row["actions"])
            if chunk.ndim != 2 or chunk.shape[1] != 8 or not len(chunk):
                raise ValueError("Malformed policy chunk")
            chunk_lengths.append(len(chunk))
        elif kind == "proposal":
            if step != current_step or step != current_observation or row["chunk_index"] != used:
                raise ValueError("Proposal is not aligned with current observation/chunk")
            if chunk is None or used >= min(adapter.horizon, len(chunk)):
                raise ValueError("Chunk prefix was exceeded")
            event = row["event"]
            p = np.asarray(row["original"], dtype=event["proposal_dtype"])
            expected = np.asarray(chunk[used], dtype=p.dtype).copy()
            expected[-1] = np.sign(expected[-1]) if abs(expected[-1]) > 0.01 else -1.0
            exactly(p, expected, "native proposed command")
            frame = SimpleNamespace(
                joints=np.asarray(row["frame"]["joints"], dtype=event["joint_dtype"]),
                fingers=np.asarray(row["frame"]["fingers"], dtype=event["finger_dtype"]),
            )
            proposal, expected_event = adapter.act(p, frame)
            exactly(proposal, row["adapted"], "dial command")
            if event != expected_event:
                raise ValueError("Controller semantic event mismatch")
            activations.append(event)
        elif kind == "action_space":
            if step != current_step or proposal is None:
                raise ValueError("Action-space clipping lacks its proposal")
            clipped = np.clip(
                proposal,
                np.asarray(row["low"], dtype=row["dtype"]),
                np.asarray(row["high"], dtype=row["dtype"]),
            )
            exactly(clipped, row["clipped"], "downstream clipping")
        elif kind == "executed":
            if step != current_step or clipped is None:
                raise ValueError("Executed input lacks its causal clipped command")
            exactly(clipped, row["action"], "environment input")
            adapter.applied(clipped)
            current_step += 1
            used += 1
            proposal = clipped = None
        elif kind == "correction":
            if step != current_step - 1 or row["normalized"] != adapter.last_correction:
                raise ValueError("Correction differs from causal actual-input replay")
            if row["accumulated"] != adapter.correction or row["since_query"] != used:
                raise ValueError("Correction accumulator/reset mismatch")
        elif kind == "replan":
            if step != current_step or row["reason"] != adapter.replan_reason(None):
                raise ValueError("Noncausal or incorrect query trigger")
        elif kind == "goal":
            if step != current_step or step != current_observation:
                raise ValueError("Goal observation is not post-action")
    if current_step != steps or counts != {
        "proposal": steps,
        "executed": steps,
        "observed": steps + 1,
        "goal": steps,
        "query": len(chunk_lengths),
        "correction": steps,
    }:
        raise ValueError("Incomplete or duplicate action/monitor/goal/query trace")
    return {
        "min_chunk_length": min(chunk_lengths),
        "query_count": len(chunk_lengths),
        "changed_commands": sum(e["changed"] for e in activations),
        "grasp_heuristic_actions": sum(e["grasp_evidence"] for e in activations),
        "cap_active_actions": sum(e["cap_active"] for e in activations),
    }


def audit(output):
    output = Path(output)
    result = {
        "output": str(output),
        "valid": False,
        "engineering_valid": False,
        "full_episode": False,
        "errors": [],
        "costs": {},
        "U": None,
        "cost_scope": "unknown until physical ledger is available",
    }
    try:
        job = read_json(output / "job.json")
        result.update(
            {k: job[k] for k in ("episode_id", "purpose", "controller", "condition", "scene", "seed")}
        )
        result["full_episode"] = job["max_steps_override"] is None and job["action_horizon"] == 1100
        if (output / "ledger.json").exists():
            ledger = read_json(output / "ledger.json")
            result["costs"] = {key: ledger.get(key) for key in COSTS}
            result["cost_scope"] = "persisted counters; lower bounds until valid completion"
        else:
            raise ValueError("No physical ledger; interactions unknown, not zero")
        trace_path = output / "trace.jsonl"
        lower_bounds = {
            "policy_control_steps": 0,
            "policy_queries": 0,
            "policy_queries_completed": 0,
        }
        if trace_path.exists():
            for line in trace_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # Preserve complete records before a torn final record.
                key = {
                    "executed": "policy_control_steps",
                    "query": "policy_queries",
                    "chunk": "policy_queries_completed",
                }.get(row.get("type"))
                if key:
                    lower_bounds[key] += 1
            for key, lower in lower_bounds.items():
                result["costs"][key] = max(lower, result["costs"].get(key) or 0)
        result["pre_subscription_physics_ticks"] = ledger.get("pre_subscription_physics_ticks")
        physical = read_physical(output)
        native = read_json(output / "episode/results.jsonl")
        monitor = read_json(output / "episode" / (native["scene_name"] + "_ltl.json"))
        trace = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
        if config_hash(Config(**job["controller"])) != job["controller_sha256"]:
            raise ValueError("Controller hash mismatch")
        for relative, digest in job["code_hashes"].items():
            if sha(output / "study_source" / relative) != digest:
                raise ValueError("Changed source snapshot: " + relative)
        provenance = read_json(output / "runtime-provenance.json")
        if provenance["runtime_sha256"] != sha(output / "benchmark_instrumented.py"):
            raise ValueError("Generated evaluator hash mismatch")
        original_source = output / "upstream_benchmark.py"
        if not original_source.exists() and job["purpose"] == "engineering":
            from helpers.maniguard_phase2_families import ROOT

            original_source = ROOT / "third_party/ManiGuard/maniguard/eval/benchmark.py"
            result["engineering_provenance_note"] = "Original code verified by pinned hash"
        if provenance["original_sha256"] != sha(original_source):
            raise ValueError("Original evaluator source hash mismatch")
        snapshot_sources = read_json(output / "study_source/configs/maniguard/phase2/sources.json")
        if (
            job["policy"] != snapshot_sources["policies"]["clutter"]
            or provenance["revision"] != snapshot_sources["maniguard"]["revision"]
        ):
            raise ValueError("Policy/upstream pin differs from saved source manifest")
        for kind, key, value in (("job", "job", job), ("result", "result", native)):
            found = [r for r in trace if r["type"] == kind]
            if len(found) != 1 or found[0][key] != value:
                raise ValueError("Authoritative " + kind + " disagrees with trace")
        metadata = [r["metadata"] for r in trace if r["type"] == "policy_metadata"]
        if len(metadata) != 1 or any(
            metadata[0].get(k) != v
            for k, v in {
                "serve_config": job["policy"]["config"],
                "checkpoint_revision": job["policy"]["revision"],
                "episode_seed_reset": "study_episode_id",
            }.items()
        ):
            raise ValueError("Pinned policy metadata mismatch")
        receipt = metadata[0].get("checkpoint_provenance", {})
        checkpoint_receipt = job.get("checkpoint_receipt_sha256")
        if checkpoint_receipt is None and job["purpose"] == "engineering":
            from helpers.maniguard_phase2_families import ROOT

            checkpoint_receipt = sha(
                ROOT / "data/maniguard/policies/clutter/download-provenance.json"
            )
        if receipt.get("receipt_sha256") != checkpoint_receipt:
            raise ValueError("Checkpoint verification receipt mismatch")
        server_hashes = metadata[0].get("server_source_hashes", {})
        expected_server = {k: job["code_hashes"].get(k) for k in server_hashes}
        if job["purpose"] == "engineering":
            from helpers.maniguard_phase2_families import ROOT

            expected_server = {
                k: v if v is not None else sha(ROOT / k) for k, v in expected_server.items()
            }
        if not server_hashes or expected_server != server_hashes:
            raise ValueError("Policy server source hashes differ from episode snapshot")
        observed = [r["monitor"]["observation"] for r in trace if r["type"] == "observed"]
        if observed != monitor["log"]:
            raise ValueError("Observed safety trace differs from native physical monitor")
        outcome = derive_outcome(
            "clutter", native, observed, [r for r in trace if r["type"] == "goal"]
        )
        if not outcome["valid"]:
            raise ValueError("; ".join(outcome["invalid_reasons"]))
        stats = replay(job, trace, native["steps"])
        for key in COSTS:
            value = ledger.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Missing or invalid cost: " + key)
        if (
            ledger["policy_queries"] != stats["query_count"]
            or ledger["policy_queries_completed"] != stats["query_count"]
            or ledger["policy_queries_failed"] != 0
            or ledger["env_step_attempts"] != ledger["env_steps_returned"]
            or ledger["env_steps_returned"] < native["steps"]
            or ledger["changed_steps"] != stats["changed_commands"]
        ):
            raise ValueError("Trace and interaction ledger disagree")
        if not native["success"] and native["steps"] != job["action_horizon"]:
            raise ValueError("Unfinished episode lacks its full configured horizon")
        result.update(
            stats,
            **{k: outcome[k] for k in ("C_req", "S_req", "V", "U")},
            benchmark_success=native["success"],
            steps=native["steps"],
            ever_contacted=physical["ever_contacted"],
            ever_grasped=physical["ever_grasped"],
            valid=True,
            engineering_valid=job["purpose"] == "engineering",
            cost_scope="complete persisted counters, including initialization physics",
        )
    except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
        result["U"] = None
    return result
