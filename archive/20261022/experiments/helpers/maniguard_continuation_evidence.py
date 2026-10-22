"""Replay compound handoff evidence without a simulator or policy connection."""

import numpy as np

from helpers.maniguard_continuation_handoff import HandoffReference
from helpers.maniguard_latent import feature_vector, finite
from helpers.maniguard_latent_policy import array_hash


def encode_bounds(low, high):
    """JSON null denotes negative/positive infinity in lower/upper bounds."""
    low, high = np.asarray(low), np.asarray(high)
    if (low.shape != (8,) or high.shape != (8,) or np.isnan(low).any()
            or np.isnan(high).any() or np.isposinf(low).any() or np.isneginf(high).any()
            or np.any(low >= high)):
        raise ValueError("Invalid native outer action bounds")
    return {"low": [None if np.isneginf(x) else float(x) for x in low],
            "high": [None if np.isposinf(x) else float(x) for x in high]}


def decode_bounds(record):
    low = np.asarray([-np.inf if x is None else x for x in record["low"]], np.float64)
    high = np.asarray([np.inf if x is None else x for x in record["high"]], np.float64)
    if encode_bounds(low, high) != {key: record[key] for key in ("low", "high")}:
        raise ValueError("Native action bounds do not round-trip")
    return low, high


def choice_record(decision):
    """The selected component is distinct from the immutable closure component."""
    offset = decision["noise_offset"]
    if offset is None:
        offset = np.zeros((16, 32), dtype=np.float32)
    return {
        "branch": decision["branch"],
        "query_step": decision["query_step"],
        "controller_sha256": decision["controller_sha256"],
        "contract_sha256": decision["contract_sha256"],
        "code": decision["code"].tolist(),
        "offset_sha256": array_hash(offset),
        "privileged_actor_inputs": decision["privileged_actor_inputs"],
        "state": decision["state"],
    }


def check_return(result, choice, features, *, episode, seed, index, previous_rng):
    """Verify a received response against the one already admitted request."""
    receipt = result["latent_receipt"]
    expected = {
        "episode": episode,
        "seed": seed,
        "query_index": index,
        "weights_sha256": choice["controller_sha256"],
        "features_sha256": array_hash(features),
        "identity_native_path": choice["branch"] == "native",
        "audit": False,
        "status": "returned",
        "shape": [1, 16, 32],
        "offset_sha256": choice["offset_sha256"],
        "code": choice["code"],
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Received policy branch, identity, offset or request identity differs")
    if (type(receipt["audit"]) is not bool or type(receipt["identity_native_path"]) is not bool
            or type(receipt["seed"]) is not int or type(receipt["query_index"]) is not int):
        raise ValueError("Policy request identity and path flags require exact types")
    if previous_rng is not None and receipt["rng_before"] != previous_rng:
        raise ValueError("Native policy RNG chain broke across compound queries")
    for key in ("rng_before", "rng_after"):
        raw = np.asarray(receipt[key])
        if (raw.shape != (2,) or raw.dtype.kind not in "iu"
                or np.any(raw < 0) or np.any(raw > 2**32 - 1)):
            raise ValueError("Invalid policy RNG-key receipt")
    actions = np.asarray(result["actions"])
    if actions.shape != (16, 8) or not np.isfinite(actions).all():
        raise ValueError("Invalid compound policy action chunk")
    if array_hash(actions) != receipt["actions_sha256"]:
        raise ValueError("Received raw action bytes differ from the server receipt")
    return actions.astype(np.float32), receipt["rng_after"]


def replay_stream(rows, weights, *, mode, debounce, clock_switch_step, episode, seed, steps):
    """Reconstruct every choice from causal features and factual observations.

    This verifies a complete returned stream. The physical reader separately
    verifies native termination, goals, monitor validity, sources and accounting.
    Monitor labels are passed to the selector only for the privileged reference.
    """
    if type(steps) is not int or not 1 <= steps <= 2000:
        raise ValueError("Require an actual native Jar episode control count")
    selector = HandoffReference(
        weights, mode=mode, debounce=debounce, clock_switch_step=clock_switch_step
    )
    observed, executed, proposals, queries = [], [], [], []
    admissions, returns, choices, chunks = [], [], [], []
    previous_action = previous_eef = previous_rng = None
    features = choice = chunk = chunk_step = pending_action = None
    bounds = None
    admitted = returned = False
    contract_seen = False
    for row in rows:
        kind, step = row["type"], row.get("step")
        if kind == "continuation_contract":
            if contract_seen or row["contract"] != selector.contract:
                raise ValueError("Missing or different compound selector contract")
            contract_seen = True
            bounds = decode_bounds(row)
        elif kind == "observed":
            if step != len(observed) or step != len(executed):
                raise ValueError("Physical observation is missing or causally misaligned")
            monitor = row["monitor"]
            if monitor.get("valid") is not True:
                raise ValueError("Missing factual monitor state")
            if selector.privileged:
                value = monitor["observation"]
                selector.observe(
                    step,
                    closed=value["ap"]["jar_closed"],
                    supported=value["ap"]["jar_on_support"],
                    violated=value["doomed"],
                )
            observed.append(step)
        elif kind == "latent_features":
            if (not contract_seen or step != 8 * len(queries) or step != len(executed)
                    or len(observed) != step + 1):
                raise ValueError("Compound query is not at the next causal native boundary")
            if queries and not returned:
                raise ValueError("A policy query was replaced before its response")
            features = finite(row["features"], (44,), "causal actor features")
            frame = row["frame"]
            expected = feature_vector(
                frame["joints"], frame["fingers"], frame["eef"],
                features[20:28], features[28:36], features[36:44],
                previous_executed=previous_action, previous_eef=previous_eef,
            )
            np.testing.assert_array_equal(features, expected)
            previous_eef = frame["eef"]
            choice = choice_record(selector.select(step, features))
            np.testing.assert_array_equal(np.asarray(row["code"], np.float32), choice["code"])
            if row["selected_controller_sha256"] != choice["controller_sha256"]:
                raise ValueError("Feature receipt labels the wrong selected component")
            admitted = returned = False
            queries.append(step)
        elif kind == "continuation_choice":
            if choice is None or choices == queries or step != queries[-1] or row["choice"] != choice:
                raise ValueError("Recorded branch differs from factual selector replay")
            choices.append(step)
        elif kind == "continuation_request":
            if choice is None or choices != queries or admitted or step != queries[-1]:
                raise ValueError("Policy admission is duplicated, missing its choice or unordered")
            expected = {
                "index": len(returns), "episode": episode, "seed": seed,
                "controller_sha256": choice["controller_sha256"],
                "features_sha256": array_hash(features),
            }
            if row["request"] != expected:
                raise ValueError("Admitted policy request differs from selected component")
            admitted = True
            admissions.append(step)
        elif kind == "noise_receipt":
            if not admitted or returned or step != queries[-1]:
                raise ValueError("Policy return lacks its unique preceding admission")
            raw = np.asarray(row["raw_actions"], dtype=row["raw_action_dtype"])
            chunk, previous_rng = check_return(
                {"actions": raw, "latent_receipt": row["receipt"]}, choice, features,
                episode=episode, seed=seed, index=len(returns), previous_rng=previous_rng,
            )
            returned, chunk_step = True, step
            returns.append(step)
        elif kind == "chunk":
            if not returned or step != queries[-1] or chunks == queries:
                raise ValueError("Decoded chunk is missing its received policy return")
            np.testing.assert_array_equal(np.asarray(row["actions"], np.float32), chunk)
            chunks.append(step)
        elif kind == "proposal":
            if (chunk is None or chunks != queries or step != len(executed)
                    or step != len(proposals) or len(observed) != step + 1
                    or not 0 <= step - chunk_step < 8):
                raise ValueError("Proposed action is outside its received native prefix")
            expected = chunk[step - chunk_step].copy()
            expected[-1] = np.sign(expected[-1]) if abs(expected[-1]) > .01 else -1.
            np.testing.assert_array_equal(np.asarray(row["original"], np.float32), expected)
            np.testing.assert_array_equal(row["adapted"], row["original"])
            pending_action = expected
            proposals.append(step)
        elif kind == "executed":
            if (bounds is None or pending_action is None or step != len(executed)
                    or len(proposals) != step + 1 or len(observed) != step + 1):
                raise ValueError("Executed action lacks its causal proposal or native bounds")
            previous_action = finite(row["action"], (8,), "actual environment input")
            np.testing.assert_array_equal(previous_action, np.clip(pending_action, *bounds))
            pending_action = None
            executed.append(step)
    expected_queries = list(range(0, steps, 8))
    if (not contract_seen or observed != list(range(steps + 1))
            or executed != list(range(steps)) or proposals != executed
            or any(items != expected_queries for items in (queries, choices, admissions, returns, chunks))):
        raise ValueError("Incomplete compound physical/query stream")
    return {
        "contract_sha256": selector.contract_sha256,
        "mode": mode,
        "policy_requests_admitted": len(admissions),
        "policy_returns_verified": len(returns),
        "complete_native_rng_chain_verified": True,
        "actual_environment_inputs_verified": len(executed),
        "state": selector.state(),
    }
