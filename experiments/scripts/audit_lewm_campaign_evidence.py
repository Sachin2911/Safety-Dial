"""Independent saved-evidence audits for the current rented LeWM campaign.

No environment, world model or learned safety model is imported or executed. Cached
embeddings and recorded controller inputs may be replayed with NumPy. Audit reports
are new artifacts, never replacements for protocols, runtime files or outcomes.
"""

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import pickle
import tarfile
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, *, atol=1e-8, rtol=0):
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(message)


def unit(x):
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x)
    return x / norm if norm > 1e-12 else np.zeros(2)


def geometry(points, shape, radius, *, normals=False):
    """NumPy disk clearance, independent of either learned checker implementation."""
    points = np.asarray(points, dtype=np.float64)
    if shape.get("kind", "none") == "none":
        value = np.full(points.shape[:-1], 1e6)
        return (value, np.zeros_like(points)) if normals else value
    origin = np.asarray(shape.get("origin", shape.get("centre")), dtype=float)
    if "basis" in shape:
        basis = np.asarray(shape["basis"], dtype=float)
    else:
        axis = np.asarray(shape["axis"], dtype=float)
        basis = np.array([axis, [-axis[1], axis[0]]])
    xy = (points - origin) @ basis.T
    kind = shape["kind"]
    rad = shape.get("radius_px", shape.get("radius", 0))
    if kind == "island":
        length = np.linalg.norm(xy, axis=-1)
        value = length - rad
        normal = np.divide(
            xy, length[..., None], out=np.zeros_like(xy), where=length[..., None] > 1e-12
        )
        normal = np.where((length <= 1e-12)[..., None], [1, 0], normal)
    elif kind == "bar":
        half = shape.get("segment_half_length_px", shape.get("half_length"))
        nearest = np.stack([np.clip(xy[..., 0], -half, half), np.zeros(xy.shape[:-1])], -1)
        delta = xy - nearest
        length = np.linalg.norm(delta, axis=-1)
        value = length - rad
        normal = np.divide(
            delta, length[..., None], out=np.zeros_like(delta), where=length[..., None] > 1e-12
        )
        normal = np.where((length <= 1e-12)[..., None], [0, 1], normal)
    elif kind == "gate":
        offset = shape.get("gap_offset_px", shape.get("offset"))
        gap = shape.get("gap_width_px", shape.get("gap"))
        extent = shape.get("extent_px", shape.get("extent"))
        band = shape.get("band_half_width_px", shape.get("band"))
        boxes, directions = [], []
        for low, high in ((-extent, offset - gap / 2), (offset + gap / 2, extent)):
            centre, half = np.array([(low + high) / 2, 0]), np.array([(high - low) / 2, band])
            local = xy - centre
            q = np.abs(local) - half
            distance = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)
            delta = local - np.clip(local, -half, half)
            length = np.linalg.norm(delta, axis=-1)
            direction = np.divide(
                delta, length[..., None], out=np.zeros_like(delta), where=length[..., None] > 1e-12
            )
            face = np.argmax(q, axis=-1)
            inside = np.zeros_like(delta)
            np.put_along_axis(
                inside,
                face[..., None],
                np.where(np.take_along_axis(local, face[..., None], -1) >= 0, 1, -1),
                -1,
            )
            direction = np.where((length <= 1e-12)[..., None], inside, direction)
            boxes.append(distance)
            directions.append(direction)
        first = boxes[0] <= boxes[1]
        value = np.where(first, boxes[0], boxes[1])
        normal = np.where(first[..., None], directions[0], directions[1])
    else:
        raise ValueError("Unsupported declared geometry: " + kind)
    value = value - radius
    return (value, normal @ basis) if normals else value


class Resolver:
    """Resolve immutable hashes from current files or preserved source bundles."""

    def __init__(self, root):
        self.root = root
        self.verified = {}
        self.archives = None

    def bytes(self, name, expected):
        key = (str(name), expected)
        if key in self.verified:
            return self.verified[key][1]
        path = Path(name)
        if not path.is_absolute():
            path = self.root / path
        if path.is_file() and digest(path) == expected:
            data = path.read_bytes()
            self.verified[key] = (
                str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path),
                data,
            )
            return data
        if self.archives is None:
            self.archives = sorted((self.root / "environment").rglob("*.tar.gz"))
        for archive in self.archives:
            if archive.stat().st_size > 30_000_000:
                continue
            with tarfile.open(archive, "r:gz") as bundle:
                for candidate in (str(name), "./" + str(name)):
                    try:
                        member = bundle.getmember(candidate)
                    except KeyError:
                        continue
                    if not member.isfile():
                        continue
                    data = bundle.extractfile(member).read()
                    if hashlib.sha256(data).hexdigest() == expected:
                        self.verified[key] = (
                            str(archive.relative_to(self.root)) + "::" + candidate,
                            data,
                        )
                        return data
        raise AssertionError("No retained bytes match frozen identity: " + str(name))

    def hash(self, name, expected):
        self.bytes(name, expected)

    def proof(self):
        return [
            {"path": name, "sha256": expected, "verified_from": value[0]}
            for (name, expected), value in sorted(self.verified.items())
        ]


def protocol_for(root, stage, freeze, resolver, explicit=None):
    if "protocol" in freeze:
        protocol = freeze["protocol"]
        if explicit is not None:
            check(
                read(explicit) == protocol,
                "Explicit protocol differs from its frozen structured record",
            )
        if "protocol_sha256" in freeze:
            check(
                canonical(protocol) == freeze["protocol_sha256"],
                "Canonical training protocol hash differs",
            )
        return protocol
    expected = freeze["protocol_sha256"]
    if explicit is not None:
        relative = (
            str(Path(explicit).relative_to(root)) if Path(explicit).is_absolute() else str(explicit)
        )
        return json.loads(resolver.bytes(relative, expected))
    for path in (root / "configs/lewm_campaign_20261031").glob("*.json"):
        if digest(path) == expected:
            resolver.hash(str(path.relative_to(root)), expected)
            return read(path)
    raise AssertionError("Supply --protocol to identify this diagnostic's frozen JSON")


def identities(root, protocol, freeze, resolver):
    design = protocol.get("evaluation", protocol)
    for name, expected in design.get("asset_sha256", {}).items():
        resolver.hash(name, expected)
    if design.get("memory_path"):
        resolver.hash(design["memory_path"], design["memory_sha256"])
    mappings = [design.get("source_sha256", {}), protocol.get("source_sha256", {})]
    source = freeze.get("sources", {})
    if source and all(isinstance(v, str) and len(v) == 64 for v in source.values()):
        mappings.append(source)
    if "harness_sha256" in source:
        resolver.hash("experiments/helpers/lewm_campaign_training.py", source["harness_sha256"])
        mappings.append(source.get("declared_source_ids", {}).get("sources", {}))
        if "policy_log_probability_source_sha256" in source:
            resolver.hash(
                "experiments/helpers/lewm_navigation_retrieval.py",
                source["policy_log_probability_source_sha256"],
            )
    mappings.append(protocol.get("source_ids", {}).get("sources", {}))
    for mapping in mappings:
        for name, expected in mapping.items():
            resolver.hash(name, expected)
    preparation = protocol.get("case_preparation")
    if preparation:
        for stem in ("panel", "review"):
            resolver.hash(preparation[stem + "_path"], preparation[stem + "_sha256"])
        review = json.loads(
            resolver.bytes(preparation["review_path"], preparation["review_sha256"])
        )
        if preparation.get("require_approved"):
            check(review["approved"], "Frozen case preparation lacks the required approval")
        if preparation.get("require_images_clean"):
            check(review["images_clean"], "Frozen case review records image-style concerns")
    if "identity" in freeze:
        check(
            canonical({"protocol": protocol, "sources": source}) == freeze["identity"],
            "Frozen optimizer identity differs",
        )
    if "resolved_controllers" in freeze:
        for label, descriptor in protocol["controllers"].items():
            resolved = freeze["resolved_controllers"][label]
            if descriptor["kind"] != "nominee":
                check(resolved == descriptor, "Fixed/native evaluation descriptor changed")
                continue
            nominee_path = descriptor["source"]
            nominee_file = json.loads(resolver.bytes(nominee_path, resolved["nominee_sha256"]))
            state_path = str(Path(nominee_path).parent / "state.json")
            training_state = json.loads(resolver.bytes(state_path, resolved["state_sha256"]))
            candidate = nominee_file[descriptor["field"]]
            check(
                training_state["phase"] == "complete"
                and training_state["candidate_panels_completed"]
                == descriptor["expected_candidate_panels"]
                and training_state["optimizer_updates"] == descriptor["expected_optimizer_updates"],
                "Evaluation nominee comes from an incomplete or different training stage",
            )
            check(
                candidate == training_state["best_trained"] == resolved["nominee"]
                and not candidate["parameters_equal_initial"]
                and candidate["provenance"]["method"] == descriptor["method"]
                and resolved["parameters"] == candidate["parameters"],
                "Evaluation nominee genotype or training-only selection provenance differs",
            )
    return design


def episode_context(protocol, freeze, state, folder, key):
    start = read(folder / "started.json")
    outer = read(folder / "result.json")
    episode = folder / "episode" if (folder / "episode/result.json").exists() else folder
    inner = {**outer, **read(episode / "result.json")}
    design = protocol.get("evaluation", protocol)
    schema = protocol.get("schema", "")
    if schema == "lewm-campaign-training-v1":
        job = start["job"]
        check(
            start["identity"] == freeze["identity"] == outer["identity"],
            "Training job belongs to another source identity",
        )
        check(
            canonical(job) == start["input_sha256"] == outer["input_sha256"],
            "Training JobSpec hash differs",
        )
        case, seed, cap = job["case"], job["environment_seed"], job["max_actions"]
        params = np.asarray(job["parameters"], dtype=float)
        native = job["method"] == "native"
        planner = design["planner"]
        projected, clipped = True, True
    elif schema == "lewm-push-evaluation-v1":
        index = int(key.split("-", 1)[0])
        spec = protocol["evaluation_jobs"][index]
        case, seed = spec["case"], spec["seed"]
        cap = min(
            spec.get("max_actions", protocol["max_episode_actions"]),
            protocol["max_episode_actions"],
        )
        controller = freeze["resolved_controllers"][spec["controller"]]
        check(
            start["controller"] == controller and start["case"] == case and start["seed"] == seed,
            "Frozen evaluation controller/case identity differs",
        )
        params = np.asarray(controller.get("parameters", np.zeros(24)), dtype=float)
        native = controller["kind"] == "native"
        planner, projected, clipped = design["planner"], True, True
    elif schema == "lewm-navigation-comparison-v1":
        planned = next(job for job in protocol["jobs"] if job["job_id"] == key)
        check(start == planned, "Executed navigation comparison job differs from its freeze")
        case = planned["case"]
        seed = planned.get("environment_seed", case["seed"])
        cap = min(
            planned.get("max_actions", design["maximum_episode_actions"]),
            design["maximum_episode_actions"],
        )
        parameters = planned.get("parameters", protocol["controller_static_parameters"]["fixed_ap"])
        params = np.asarray(parameters, dtype=float)
        native = planned["method"] == "native"
        planner, projected, clipped = design["planner"], True, True
        check(
            outer["job_id"] == key
            and outer["case_id"] == case["case_id"]
            and outer["method"] == planned["method"]
            and outer["environment_seed"] == seed
            and outer["policy_seed"] == planned.get("policy_seed", case["seed"]),
            "Navigation comparison output provenance differs",
        )
        close(outer["parameters"], params, "Comparison output genotype differs", atol=0)
        allowed = design.get("allowed_source_episodes")
        if allowed is not None:
            check(
                case["source_episode"] in allowed,
                "Executed source episode is outside the prospectively permitted split",
            )
    elif "recipes" in protocol:
        case = next(c for c in protocol["cases"] if c["id"] == inner["case_id"])
        recipe = next(r for r in protocol["recipes"] if r["name"] == inner["recipe"])
        check(
            seed_in := inner["seed"] in case["seeds"],
            "PushT diagnostic seed is outside the fixed case",
        )
        del seed_in
        seed, cap = inner["seed"], protocol["maximum_episode_actions"]
        check(
            start["recipe"] == recipe and start["case_id"] == case["id"],
            "Frozen PushT recipe identity differs",
        )
        planner, projected, clipped = (
            recipe["planner"],
            recipe["project_candidates"],
            recipe["clip_execution"],
        )
        params, native = None, True
    elif "configurations" in protocol:
        job = next(j for j in protocol["jobs"] if j["id"] == key)
        source = protocol["sources"][job["pair"]]
        case = {
            "input_path": source["path"],
            "input_sha256": source["source_pair_sha256"],
            "shape": {"kind": "none"},
        }
        check(all(start[k] == job[k] for k in job), "Frozen navigation diagnostic job differs")
        seed, cap = job["seed"], job["cap"]
        planner = protocol["configurations"][job["configuration"]]["params"]
        projected, clipped = (
            protocol["configurations"][job["configuration"]]["project_candidates"],
            True,
        )
        params, native = None, not job.get("zero_adapter", False)
    else:
        raise ValueError("Unsupported stage schema: " + schema)
    check(
        inner.get("seed", inner.get("environment_seed", seed)) == seed,
        "Executed environment seed differs",
    )
    return {
        "folder": folder,
        "episode": episode,
        "outer": outer,
        "row": inner,
        "start": start,
        "case": case,
        "seed": seed,
        "cap": cap,
        "parameters": params,
        "native": native,
        "planner": planner,
        "projected": projected,
        "clipped": clipped,
        "design": design,
    }


def original_progress(goal, initial, final, environment):
    if environment == "pusht":
        positional = np.clip(
            1
            - np.linalg.norm(goal[:4] - final[:4])
            / max(np.linalg.norm(goal[:4] - initial[:4]), 1e-6),
            0,
            1,
        )

        def wrap(value):
            return abs((value + np.pi) % (2 * np.pi) - np.pi)

        angular = np.clip(
            1 - wrap(goal[4] - final[4]) / max(wrap(goal[4] - initial[4]), 1e-6), 0, 1
        )
        return float(0.5 * (positional + angular))
    return float(
        np.clip(1 - np.linalg.norm(goal - final) / max(np.linalg.norm(goal - initial), 1e-12), 0, 1)
    )


def audit_model_work(episode, row, planner, contract, trace, *, clipped, projected):
    plans = read(episode / "replans.json")
    projections = read(episode / "projection.json")
    c = row["resources"]
    check(
        c.get("encoder_frames", 0) == 2 * len(plans),
        "Live/original-goal encoder-frame count differs",
    )
    extra = [p for p in plans if p.get("executed") not in ("none", "mean")]
    batches = len(projections) + len(extra)
    rollouts = sum(p["candidate_rollouts"] for p in projections) + len(extra)
    steps = sum(p["candidate_rollouts"] * p["horizon"] for p in projections) + sum(
        p["horizon"] for p in extra
    )
    check(
        c.get("model_batch_calls", 0) == batches,
        "Scored population/mean/chosen query call count differs",
    )
    check(
        c.get("model_candidate_rollouts", 0) == rollouts,
        "Frozen-model candidate rollout count differs",
    )
    check(c.get("predictor_steps", 0) == steps, "Frozen-model predicted block-step count differs")
    if "actual_cost_call_attempts" in c:
        check(
            c["actual_cost_call_attempts"] == c["actual_cost_call_successes"] == batches,
            "Recorded actual frozen-model query attempts/successes differ",
        )
        check(
            c["actual_candidate_rollouts_attempted"] == rollouts
            and c["actual_candidate_block_steps_attempted"] == steps,
            "Actual query-work instrumentation differs",
        )
        check(
            c["actual_encoder_call_attempts"]
            == c["actual_encoder_call_successes"]
            == len(plans) * 2,
            "Actual encoder API call count differs",
        )
    check(
        c.get("out_of_domain_candidate_coordinates", 0)
        == sum(p["out_of_domain_coordinates"] for p in projections),
        "Out-of-domain candidate coordinate ledger differs",
    )
    check(
        c.get("projected_candidate_coordinates", 0)
        == sum(p["projected_coordinates"] for p in projections),
        "Applied projection coordinate ledger differs",
    )
    check(
        c.get("action_projection_calls", 0) == (len(projections) if projected else 0),
        "Projection call count differs from the declared arm",
    )
    samples = sum(p["candidate_rollouts"] for p in projections)
    executed = [p for p in plans if p.get("executed") != "none"]
    learned = c.get("learned_candidate_checks", 0)
    check(
        samples <= learned <= samples + len(plans),
        "Learned safety candidate check count is outside scored/execution checks",
    )
    if len(executed) == len(plans):
        check(learned == samples + len(plans), "Final learned execution recheck count differs")
    mean, scale = np.asarray(contract["mean"]), np.asarray(contract["scale"])
    action_stride = planner["replan_blocks"] * 5
    for i, plan in enumerate(executed):
        start = i * action_stride
        if "chosen_block" in plan and start < len(trace["actions"]):
            native = np.asarray(plan["chosen_block"]).reshape(5, 2) * scale + mean
            actual = trace["actions"][start : start + 5]
            if clipped:
                native = np.clip(native, contract["native_low"], contract["native_high"])
            close(
                actual,
                native[: len(actual)],
                "Executed first block does not match the checked chosen block",
                atol=2e-6,
            )
        check(
            plan.get("executed_violation", 0) == 0,
            "An executed plan was recorded unsafe by the learned checker",
        )
    return {
        "replans": len(plans),
        "model_batch_calls": batches,
        "model_candidate_rollouts": rollouts,
        "predictor_steps": steps,
        "plans": plans,
    }


def audit_push_controller(context, trace, plans, contract):
    folder, row, case = context["episode"], context["row"], context["case"]
    path = folder / "controller.json"
    if context["native"]:
        check(
            not path.exists() and not row["resources"].get("adapter_forward_calls", 0),
            "Native controller contains a proposal adapter",
        )
        return {"kind": "native", "controller_forward_replays": 0}
    artifact = read(path)
    check(
        artifact["format"] == "lewm-push-structured-residual-v1",
        "Unknown PushT controller artifact",
    )
    check(
        artifact["spec"] == context["design"]["controller_spec"],
        "Residual controller authority or feature constants differ from the prospective freeze",
    )
    close(
        artifact["action_mean"], contract["mean"], "Residual action mean differs", atol=0
    )
    theta = np.asarray(artifact["parameters"])
    close(
        theta, context["parameters"], "Executed genotype differs from frozen job parameters", atol=0
    )
    check(theta.shape == (24,), "Unexpected compact genotype dimension")
    history = read(folder / "proposals.json")
    check(
        len(history) == row["resources"].get("adapter_forward_calls", 0),
        "Residual forward-call count differs",
    )
    model_scale = np.asarray(artifact["action_scale"])
    close(model_scale, contract["scale"], "Residual action scaling differs", atol=0)
    stride = context["planner"]["replan_blocks"] * 5
    null_contexts = 0
    for item, plan in zip(history, plans, strict=True):
        x, basis, proposal = (
            np.asarray(item["features"]),
            np.asarray(item["basis"]),
            np.asarray(item["proposal"]),
        )
        spec = artifact["spec"]
        gains = theta.reshape(2, 3, 4) @ x
        temporal = np.column_stack([np.ones(25), np.linspace(1, -1, 25)])
        replay = spec["native_residual_bound"] * np.tanh(temporal @ (gains @ basis)) / model_scale
        close(
            proposal,
            replay.reshape(5, 10)[: len(proposal)],
            "Saved residual cannot be reconstructed from its genotype/features",
            atol=1e-12,
        )
        index = item["replan"] * stride
        check(index < len(trace["states"]), "Residual queried after the final recorded state")
        clearance, normal = geometry(
            trace["states"][index, :2], case["shape"], spec["body_radius_px"], normals=True
        )
        close(
            basis[:2],
            np.stack([normal, [normal[1], -normal[0]]]),
            "Hazard-direction feature is inconsistent with current observed pusher",
            atol=1e-8,
        )
        f = plan["features"]
        near = (
            0
            if case["shape"]["kind"] == "none"
            else np.exp(-max(0, float(clearance) - spec["margin_px"]) / spec["proximity_length_px"])
        )
        stall = np.clip((0.02 - f["improvement"]) / 0.02, 0, 1) if f["replan"] else 0
        close(
            x,
            np.clip([1, near, stall, np.log1p(max(0, f["current_cost"])) / 10], -1, 1),
            "Causal residual feature differs",
            atol=1e-8,
        )
        if item.get("plan_context_source") == "previous_executed_native_action":
            recorded = np.asarray(item["previous_executed_native_action"])
            previous = trace["actions"][index - 1] if index else np.zeros(2)
            close(
                recorded,
                previous,
                "Residual previous-action source is not the executed native action",
                atol=2e-7,
            )
            close(basis[2], unit(recorded), "Previous-action direction differs", atol=1e-8)
            null_contexts += int(
                np.linalg.norm(previous) < 1e-7 and np.linalg.norm(basis[2]) > 1e-3
            )
        for iteration in plan.get("residual_checks", []):
            check(
                iteration["new_guided_samples"]
                == int(
                    (iteration["samples"] - iteration["reserved_native_slots"])
                    * context["planner"]["proposal_mixture_fraction"]
                ),
                "Guided population size changed the frozen mixture",
            )
            check(
                iteration["guided_elites"] <= iteration["fitted_elites"],
                "Guided elite provenance is impossible",
            )
        if plan.get("adapter_proposal") is not None:
            close(
                plan["adapter_proposal"],
                proposal,
                "Planner residual differs from controller output",
                atol=1e-6,
            )
    check(
        sum(r["new_guided_samples"] for p in plans for r in p.get("residual_checks", []))
        == row["resources"].get("adapter_guided_candidate_evaluations", 0),
        "Guided candidate ledger differs",
    )
    return {
        "kind": "structured_residual24",
        "controller_forward_replays": len(history),
        "numerically_null_previous_action_directions": null_contexts,
        "causal_shifted_plan_verification": "When a shifted plan was recorded, its direction is retained in the saved basis; full upstream warm-start tensors were not saved.",
    }


def retrieval_features(memory, current, final, xy, shape, body_radius, margin, allowance):
    source, successor, endpoint = [
        np.asarray(memory[k], dtype=np.float32)
        for k in ("source_embeddings", "successor_embeddings", "endpoint_embeddings")
    ]
    scale = np.asarray(memory["latent_scale"], dtype=np.float32)
    current, final, xy = (
        np.asarray(current, dtype=np.float32),
        np.asarray(final, dtype=np.float32),
        np.asarray(xy, dtype=np.float32),
    )

    def ms(x):
        return np.mean(x * x, axis=-1, dtype=np.float32)

    positions = np.asarray(memory["successor_xy"], dtype=np.float32)
    threshold = margin + allowance
    available = geometry(positions, shape, body_radius) >= threshold
    fractions = np.linspace(0, 1, 9, dtype=np.float32)
    line = xy[None, None] + fractions[None, :, None] * (positions[:, None] - xy)
    deficit = np.clip(
        (threshold - geometry(line, shape, body_radius).min(-1)) / max(1, threshold), 0, 10
    )
    count = len(source)
    feature = np.stack(
        [
            ms((source - current) / scale),
            ms((endpoint - final) / scale),
            ms(((endpoint - source) - (final - current)) / scale),
            ms((successor - current) / scale),
            ms((successor - final) / scale),
            (memory["successor_step"] - memory["source_step"]) / 10,
            deficit,
            np.linalg.norm(positions - xy, axis=-1) / 224,
            np.zeros(count),
        ],
        -1,
    ).astype(np.float32)
    direct = np.zeros((1, 9), dtype=np.float32)
    direct[0, -1] = ms((final - current) / scale)
    return np.concatenate([feature, direct]), np.r_[available, True]


def audit_retrieval(context, trace, plans, memory, stage):
    folder, row, outer = context["episode"], context["row"], context["outer"]
    artifact_path = folder / "controller.json"
    if context["native"]:
        check(not artifact_path.exists(), "Native navigation contains a retrieval controller")
        return {"kind": "native", "controller_forward_replays": 0}
    artifact = read(artifact_path)
    check(
        artifact["format"] == "lewm-observed-target-retrieval-v1",
        "Unknown retrieval controller artifact",
    )
    theta = np.asarray(artifact["parameters"])
    close(theta, context["parameters"], "Retrieval genotype differs from the frozen job", atol=0)
    check(theta.shape == (8,), "Retrieval genotype dimension differs")
    check(
        artifact["memory_sha256"] == context["design"]["memory_sha256"],
        "Retrieval memory identity differs",
    )
    for field, design_field in (
        ("body_radius", "body_radius"),
        ("margin", "margin"),
        ("allowance", "motion_allowance"),
    ):
        close(
            artifact[field],
            context["design"][design_field],
            "Retrieval geometry or motion allowance differs from its prospective freeze",
            atol=0,
        )
    history = read(folder / "selector-history.json")
    check(
        len(history) == len(plans) == row["resources"].get("retrieval_queries", 0),
        "Target retrieval query ledger differs",
    )
    check(
        row["resources"]["retrieval_candidate_targets"]
        == len(history) * (len(memory["source_embeddings"]) + 1),
        "Target candidate ledger differs",
    )
    weights = np.r_[1, np.clip(theta[:7], -6, 6) ** 2, np.exp(np.clip(theta[7], -6, 6))]
    decisions, trace_data = [], None
    if outer.get("decision_trace"):
        decision_path = stage / outer["decision_trace"]["path"]
        check(
            digest(decision_path) == outer["decision_trace"]["sha256"],
            "Categorical decision trace identity differs",
        )
        with np.load(decision_path, allow_pickle=False) as data:
            trace_data = dict(data)
        check(
            len(trace_data["selected"]) == len(history),
            "Categorical decision and live-query counts differ",
        )
    near_ties = 0
    stride = context["planner"]["replan_blocks"] * 5
    for index, item in enumerate(history):
        current_xy = np.asarray(item["current_xy"])
        close(
            current_xy,
            trace["states"][index * stride, -2:],
            "Retrieved target used a different current native XY",
            atol=2e-5,
        )
        feature, available = retrieval_features(
            memory,
            item["current_latent"],
            item["final_latent"],
            current_xy,
            context["case"].get("shape", {"kind": "none"}),
            artifact["body_radius"],
            artifact["margin"],
            artifact["allowance"],
        )
        chosen = (
            len(memory["source_embeddings"])
            if item["target_kind"] == "original_goal"
            else item["memory_index"]
        )
        check(available[chosen], "Chosen observed target violates its declared availability mask")
        check(
            int(available[:-1].sum()) == item["available_observed_targets"],
            "Observed target availability count differs",
        )
        close(
            feature[chosen],
            item["selected_features"],
            "Selected target features differ from cached observations",
            atol=2e-4,
            rtol=2e-5,
        )
        scores = feature.astype(np.float64) @ weights
        tolerance = max(2e-4, np.finfo(np.float32).eps * 64 * max(1, abs(scores[chosen])))
        close(
            scores[chosen],
            item["selected_score"],
            "Selected target score differs",
            atol=tolerance,
            rtol=0,
        )
        if not item["stochastic"]:
            best = np.min(scores[available])
            check(
                scores[chosen] <= best + tolerance,
                "Deterministic target was not a minimum-score available target",
            )
            near_ties += int(chosen != int(np.argmin(np.where(available, scores, np.inf))))
        if item["memory_index"] is not None:
            selected = item["memory_index"]
            for name, value in item["source_provenance"].items():
                check(
                    int(memory[name][selected]) == value,
                    "Observed target source provenance differs",
                )
        if trace_data is not None:
            stored_feature, stored_mask = (
                trace_data["features"][index],
                trace_data["available"][index],
            )
            check(
                hashlib.sha256(stored_feature.tobytes()).hexdigest() == item["features_sha256"],
                "Saved categorical feature table hash differs",
            )
            check(
                hashlib.sha256(stored_mask.tobytes()).hexdigest() == item["available_sha256"],
                "Saved categorical availability table hash differs",
            )
            close(
                stored_feature,
                feature,
                "Categorical feature table cannot be reconstructed from cached observations",
                atol=2e-4,
                rtol=2e-5,
            )
            check(
                np.array_equal(stored_mask, available) and trace_data["selected"][index] == chosen,
                "Categorical selection/mask differs",
            )
            logits = -(stored_feature.astype(np.float64) @ weights) / float(
                trace_data["temperature"][index]
            )
            logits[~available] = -np.inf
            maximum = np.max(logits)
            logp = logits[chosen] - maximum - np.log(np.exp(logits - maximum).sum())
            bound = max(
                2e-4, np.finfo(np.float32).eps * 64 * max(1, abs(logits[chosen]), abs(maximum))
            )
            close(
                logp,
                trace_data["behavior_log_probability"][index],
                "Recorded categorical behavior log probability differs",
                atol=bound,
            )
            decisions.append({"chosen": int(chosen), "log_probability_reconstructed": float(logp)})
    return {
        "kind": "observed_target_retrieval8",
        "controller_forward_replays": len(history),
        "categorical_decisions_verified": len(decisions),
        "floating_point_near_tie_selections": near_ties,
        "latent_scope": "Cached/live latent values and their declared hashes are verified from saved evidence; frozen neural encoders are not rerun.",
    }


def audit_episode(root, stage, context, memory=None):
    episode, row, outer, case, d = [
        context[k] for k in ("episode", "row", "outer", "case", "design")
    ]
    pair_path = case.get("input_path", case.get("pair_path", case.get("path")))
    expected = case.get(
        "input_sha256", case.get("sha256", d.get("asset_sha256", {}).get(pair_path))
    )
    check(
        expected is not None and digest(root / pair_path) == expected,
        "Executed source-pair hash differs",
    )
    with np.load(root / pair_path, allow_pickle=False) as data:
        pair = dict(data)
    with np.load(episode / "trace.npz", allow_pickle=False) as data:
        trace = {k: data[k] for k in ("states", "dense", "actions", "proposed", "goal_image")}
    actions = np.asarray(trace["actions"]).reshape(-1, 2)
    proposed = np.asarray(trace["proposed"]).reshape(-1, 2)
    trace["actions"], trace["proposed"] = actions, proposed
    count = len(actions)
    check(
        len(proposed) == count and len(trace["states"]) == count + 1,
        "Action/proposal/state lengths differ",
    )
    check(count <= context["cap"], "Executed action count exceeds its frozen episode cap")
    for name in ("states", "dense", "actions", "proposed"):
        check(np.isfinite(trace[name]).all(), "Nonfinite saved physical trace: " + name)
    check(
        np.array_equal(trace["goal_image"], pair["goal_image"]),
        "The recorded policy goal image differs from the source goal",
    )
    environment = "pusht" if "goal_state" in pair else "tworoom"
    shape = case.get("shape", row.get("case", row.get("shape", {"kind": "none"})))
    case["shape"] = shape
    radius = d.get("body_radius_px", d.get("body_radius", 15 if environment == "pusht" else 7))
    margin = d.get("margin_px", d.get("margin", 6 if environment == "pusht" else 3))
    dense = trace["dense"]
    check(
        len(dense) == count * (10 if environment == "pusht" else 1) + 1,
        "Dense native trace length differs from executed integration count",
    )
    clear = geometry(dense[..., :2], shape, radius)
    hazard_breach = bool((clear < margin - 1e-6).any())
    arena_breach = False
    if environment == "pusht":
        arena = np.minimum(dense[..., :2], 512 - dense[..., :2]).min(-1) - radius
        arena_breach = bool((arena < margin - 1e-6).any())
        goal = np.asarray(pair["goal_state"])
        final = trace["states"][-1]
        orientation = abs((goal[4] - final[4] + np.pi) % (2 * np.pi) - np.pi)
        success = bool(np.linalg.norm(goal[:4] - final[:4]) < 20 and orientation < np.pi / 9)
        pathlength = float(np.linalg.norm(np.diff(trace["states"][:, :2], axis=0), axis=1).sum())
        check(
            row["resources"].get("executed_physics_steps", 0) == count * 10,
            "PushT physics count differs from dense trace",
        )
        check(
            row["resources"].get("reset_physics_steps", 0) == 2,
            "PushT explicit native reset integrations differ",
        )
    else:
        goal = np.asarray(pair["goal"])
        final = trace["states"][-1]
        success = bool(np.linalg.norm(goal - final) < 16)
        pathlength = float(np.linalg.norm(np.diff(trace["states"], axis=0), axis=1).sum())
        check(
            row["resources"].get("executed_discrete_steps", 0) == count,
            "TwoRoom discrete-step count differs",
        )
        if row.get("original_goal") is not None:
            close(row["original_goal"], goal, "Navigation original physical goal changed", atol=0)
    violation = hazard_breach or arena_breach
    safe = bool(
        success and not violation and not row.get("error") and not row.get("controller_failure")
    )
    check(bool(row["success"]) == success, "Original physical task success differs")
    check(bool(row["unfinished"]) == (not success), "Unfinished status differs from original goal")
    check(
        bool(row["safe_completion"]) == safe,
        "Safe-completion status differs from dense safety/original goal",
    )
    check(
        bool(row["engaged"]) == bool(count and pathlength > 1),
        "Engagement differs from actual native body motion",
    )
    if "violation" in row:
        check(bool(row["violation"]) == violation, "Recorded navigation violation differs")
    if "margin_breach" in row:
        check(
            bool(row["margin_breach"]) == hazard_breach
            and bool(row["arena_margin_breach"]) == arena_breach,
            "Recorded hazard/arena margin breach differs",
        )
    if outer is not row:
        for name, expected_value in (
            ("safe_completion", safe),
            ("unfinished", not success),
            ("engaged", bool(count and pathlength > 1)),
        ):
            check(
                bool(outer[name]) == expected_value, "Outer/inner episode outcome differs: " + name
            )
        if "violation" in outer:
            check(bool(outer["violation"]) == violation, "Outer episode violation differs")
    c = row["resources"]
    check(
        c.get("environment_actions", 0) == count,
        "Recorded environment actions differ from the native action trace",
    )
    check(
        c.get("environment_resets", 0) == 1,
        "A completed episode must retain its explicit native reset",
    )
    check(
        not any(
            c.get(name, 0)
            for name in (
                "analytic_path_steps",
                "checker_physics_steps",
                "reference_candidate_checks",
            )
        ),
        "Reference/simulator candidate prediction was used",
    )
    action_data = json.loads((root / d["scalers"]).read_text())["columns"]["action"]
    contract = {
        "mean": action_data["mean"],
        "scale": action_data["scale"],
        "native_low": [-1, -1],
        "native_high": [1, 1],
    }
    if context["clipped"]:
        close(
            actions,
            np.clip(proposed, -1, 1),
            "Executed native action differs from the declared projection",
            atol=0,
        )
        check(not (np.abs(actions) > 1).any(), "A projected native command left its legal domain")
    else:
        close(actions, proposed, "The original unprojected control was silently clipped", atol=0)
    work = audit_model_work(
        episode,
        row,
        context["planner"],
        contract,
        trace,
        clipped=context["clipped"],
        projected=context["projected"],
    )
    check(
        work["predictor_steps"] <= d.get("maximum_predictor_steps_per_episode", np.inf),
        "Per-episode model-work cap exceeded",
    )
    if environment == "pusht":
        control = audit_push_controller(context, trace, work["plans"], contract)
    elif (episode / "controller.json").exists():
        check(memory is not None, "Observed target memory is required for controller audit")
        control = audit_retrieval(context, trace, work["plans"], memory, stage)
    else:
        control = {"kind": "native_or_legacy_zero", "controller_forward_replays": 0}
        zero_path = episode / "zero-proposals.json"
        if zero_path.exists():
            for item in read(zero_path):
                check(
                    not np.asarray(item["proposal"]).any(),
                    "The legacy zero adapter emitted a nonzero proposal",
                )
    progress = original_progress(goal, trace["states"][0], final, environment)
    if "progress" in outer:
        close(
            outer["progress"],
            progress,
            "Shared optimizer progress differs from the original physical task",
            atol=2e-7,
        )
    for field in ("resources",):
        if field in outer:
            check(
                all(
                    outer[field].get(k, 0) == row[field].get(k, 0)
                    for k in set(outer[field]) | set(row[field])
                ),
                "Outer/inner resource ledgers differ",
            )
    return {
        "key": context["folder"].name,
        "environment": environment,
        "case_id": case.get("case_id", case.get("id", row.get("case_id"))),
        "seed": context["seed"],
        "safe_completion": safe,
        "violation": violation,
        "unfinished": not success,
        "engaged": bool(count and pathlength > 1),
        "progress": progress,
        "environment_actions": count,
        "resources": c,
        "minimum_dense_clearance": float(clear.min()),
        "final_original_goal_error": float(np.linalg.norm(goal[:4] - final[:4]))
        if environment == "pusht"
        else float(np.linalg.norm(goal - final)),
        "controller": control,
        "controller_failure": row.get("controller_failure"),
        "episode_cap": context["cap"],
        "stopped_before_action_cap": count < context["cap"],
        "replans": work["replans"],
        "raw_commands_outside_native_domain": int((np.abs(proposed) > 1).any(-1).sum()),
        "result_sha256": digest(episode / "result.json"),
        "trace_sha256": digest(episode / "trace.npz"),
    }


def summary(rows):
    return {
        "episodes": len(rows),
        "safe_completion": sum(r["safe_completion"] for r in rows),
        "violation": sum(r["violation"] for r in rows),
        "unfinished": sum(r["unfinished"] for r in rows),
        "engaged": sum(r["engaged"] for r in rows),
        "environment_actions": sum(r["environment_actions"] for r in rows),
        "progress_sum": float(sum(r["progress"] for r in rows)),
    }


def fitness(rows):
    s = summary(rows)
    return (s["safe_completion"], -s["violation"], s["progress_sum"])


def replay_cma(protocol, history, state, checkpoint):
    """Reconstruct seeded ask/tell algebra only, without any task/model evaluation."""
    import cma

    seed = protocol["optimizer_seed"]
    rng = np.random.RandomState(seed)
    global_state = np.random.RandomState(seed).get_state()
    outer = np.random.get_state()
    try:
        es = cma.CMAEvolutionStrategy(
            protocol["initial_parameters"],
            protocol["sigma"],
            {
                "seed": seed,
                "randn": rng.randn,
                "popsize": protocol["population"],
                "CMA_diagonal": True,
                "bounds": protocol["parameter_bounds"],
                "verbose": -9,
                "verb_disp": 0,
                "verb_log": 0,
            },
        )
        for item in history:
            np.random.set_state(global_state)
            proposed = es.ask()
            close(
                proposed,
                item["parameters"],
                "Recorded CMA population differs from the independently replayed seeded ask",
                atol=1e-12,
            )
            es.tell(item["parameters"], item["minimization_ranks"])
            global_state = np.random.get_state()
        if state["pending"] is not None:
            np.random.set_state(global_state)
            close(
                es.ask(),
                state["pending"]["parameters"],
                "Saved pending CMA population differs from its seeded optimizer trajectory",
                atol=1e-12,
            )
        retained = checkpoint["es"]
        close(es.mean, retained.mean, "Retained CMA mean differs after replayed tell updates")
        close(es.sigma, retained.sigma, "Retained CMA sigma differs after replayed tell updates")
        close(
            es.sm.variances,
            retained.sm.variances,
            "Retained CMA covariance differs after replayed tell updates",
        )
    finally:
        np.random.set_state(outer)
    return len(history)


def audit_training(stage, protocol, freeze, state, audited):
    panel = protocol["training_panel"]
    jobs = {name: read(stage / "jobs" / name / "result.json") for name in audited}
    completed = state["completed_job_ids"]
    check(
        set(completed) == set(audited),
        "Optimizer state completed-job set differs from audited records",
    )
    check(state["identity"] == freeze["identity"], "Checkpoint/state source identity differs")
    names = list(audited)
    actual = sum(r["environment_actions"] for r in audited.values())
    resources = collections.Counter()
    for row in audited.values():
        resources.update(row["resources"])
    check(state["actual_environment_actions"] == actual, "Optimizer actual-action ledger differs")
    check(
        state["charged_environment_actions"] == actual,
        "A finalized run has an unexplained conservative action charge",
    )
    check(
        state["resource_totals"] == dict(resources), "Optimizer cumulative resource ledger differs"
    )
    check(
        state["episode_starts"] == len(names) <= protocol["max_episode_starts"],
        "Optimizer episode-start cap differs",
    )
    check(
        actual <= protocol["max_environment_actions"], "Optimizer actual-action allowance exceeded"
    )
    check(
        state["action_accounting_exact"] and not state["unfinalized_job_ids"],
        "Finalized run still contains unresolved resource accounting",
    )
    initial = np.asarray(protocol["initial_parameters"])
    nominees = []
    referenced_jobs = set()
    baseline_by_case = {}
    gradient_updates_verified = 0

    def candidate(parameters, job_ids, candidate_record=None):
        check(len(job_ids) == len(panel), "A genotype lacks its full fixed training panel")
        rows = []
        referenced_jobs.update(job_ids)
        for index, (name, case) in enumerate(zip(job_ids, panel, strict=True)):
            context_start = read(stage / "jobs" / name / "started.json")["job"]
            check(
                context_start["case"] == case
                and context_start["case_id"] == case["case_id"]
                and context_start["environment_seed"] == case["seed"],
                "Candidate training panel/case/seed changed",
            )
            close(
                context_start["parameters"],
                parameters,
                "Candidate genotype changed within its panel",
                atol=0,
            )
            check(
                context_start["max_actions"] == protocol["max_episode_actions"],
                "Candidate episode allowance changed",
            )
            rows.append(audited[name])
        measured = fitness(rows)
        if candidate_record is not None:
            close(
                candidate_record["fitness"],
                measured,
                "Candidate fitness differs from the original-task episode outcomes",
                atol=1e-6,
            )
            check(
                candidate_record["summary"]["safe_completion"] == measured[0]
                and candidate_record["summary"]["violation"] == -measured[1],
                "Candidate success/violation nomination differs",
            )
        return measured

    anchor = state["anchor"]
    if anchor is not None:
        close(
            anchor["parameters"],
            initial,
            "The initial anchor was changed or labelled trained",
            atol=0,
        )
        check(anchor["provenance"]["kind"] == "initial_anchor", "Anchor provenance differs")
        candidate(initial, anchor["job_ids"], anchor)
    ordered_candidates = sorted(stage.glob("candidate-*.json"))
    for path in ordered_candidates:
        item = read(path)
        check(
            item["provenance"]["method"] == protocol["method"]
            and item["provenance"]["repeat_id"] == protocol["repeat_id"],
            "Candidate method/repeat provenance differs",
        )
        candidate(item["parameters"], item["job_ids"], item)
        check(
            not np.array_equal(item["parameters"], initial),
            "An unchanged initial anchor was nominated as a trained candidate",
        )
        completion_order = max(completed.index(key) for key in item["job_ids"])
        nominees.append((tuple(item["fitness"]), -completion_order, item))
    method = protocol["method"]
    history = state["history"]
    if method in ("cma", "random"):
        check(
            state["candidate_panels_completed"] == len(ordered_candidates),
            "Completed candidate count differs from the retained candidate panels",
        )
        check(
            state["training_panels_completed"] == len(ordered_candidates),
            "Training panel ledger differs",
        )
        random = np.random.default_rng(protocol["optimizer_seed"])
        for generation, item in enumerate(history):
            check(item["generation"] == generation, "Generation numbering differs")
            check(
                read(stage / f"generation-{generation:03}.json") == item,
                "Generation file differs from durable optimizer history",
            )
            parameters = np.asarray(item["parameters"])
            check(
                parameters.shape == (protocol["population"], len(initial)),
                "Population or genotype dimension changed",
            )
            low, high = protocol["parameter_bounds"]
            check(
                np.all(parameters >= low) and np.all(parameters <= high),
                "Genotype left the frozen parameter domain",
            )
            scores = []
            for index, weights in enumerate(parameters):
                ids = [f"{method}_g{generation:03}_c{index:03}_p{i:03}" for i in range(len(panel))]
                measured = candidate(weights, ids)
                close(
                    item["fitness"][index],
                    measured,
                    "Generation fitness differs from its exact genotype panel",
                    atol=1e-6,
                )
                scores.append(tuple(item["fitness"][index]))
            ranks = [sum(other > score for other in scores) for score in scores]
            check(
                item["minimization_ranks"] == ranks,
                "CMA rank input differs from declared lexicographic fitness",
            )
            check(
                item["genuine_evolutionary_update"] == (method == "cma"),
                "Random search was labelled a real evolutionary update",
            )
            if method == "random":
                expected = np.clip(
                    initial + random.normal(0, protocol["sigma"], parameters.shape),
                    *protocol["parameter_bounds"],
                )
                close(
                    parameters,
                    expected,
                    "Parameter random search was not the shared independent initial Gaussian",
                    atol=0,
                )
                random.permutation(protocol["population"])
        check(
            state["optimizer_updates"] == (len(history) if method == "cma" else 0),
            "Real optimizer update count differs from completed generation histories",
        )
        if method == "cma":
            check(
                all(
                    p[2]["provenance"]["completed_updates_before_proposal"]
                    == p[2]["provenance"]["generation"]
                    for p in nominees
                ),
                "Genotype evolutionary-update provenance differs",
            )
    else:
        check(method == "reinforce", "Unknown optimizer method")
        check(state["optimizer_updates"] == len(history), "Ordinary-learning update count differs")
        for index, item in enumerate(history):
            check(item["update"] == index, "Ordinary-learning update indexing differs")
            nominee = item["nominee"]
            candidate(nominee["parameters"], nominee["job_ids"], nominee)
            sample_parameters = read(stage / "jobs" / item["sample_job_ids"][0] / "started.json")[
                "job"
            ]["parameters"]
            for name in item["sample_job_ids"]:
                job = read(stage / "jobs" / name / "started.json")["job"]
                close(
                    job["parameters"],
                    sample_parameters,
                    "On-policy learner sampled multiple genotypes in one update",
                    atol=0,
                )
                check(
                    job["mode"] == "categorical" and jobs[name]["decision_trace"] is not None,
                    "Ordinary learning lacks real categorical rollout decisions",
                )
            referenced_jobs.update(item["sample_job_ids"])
            statistics = item["statistics"]
            rewards, baselines, advantages = [], [], []
            grouped_rewards = {}
            gradient = np.zeros(len(initial))
            for name in item["sample_job_ids"]:
                measured = audited[name]
                reward = (
                    (len(panel) + 1) * int(measured["safe_completion"])
                    - int(measured["violation"])
                    + measured["progress"] / (len(panel) + 1)
                )
                case_id = jobs[name]["case_id"]
                baseline = baseline_by_case.get(case_id, 0.0)
                advantage = reward - baseline
                rewards.append(reward)
                baselines.append(baseline)
                advantages.append(advantage)
                grouped_rewards.setdefault(case_id, []).append(reward)
                decision_path = stage / jobs[name]["decision_trace"]["path"]
                with np.load(decision_path, allow_pickle=False) as decision:
                    theta = np.clip(np.asarray(sample_parameters), -6, 6)
                    weight = np.r_[1, theta[:7] ** 2, np.exp(theta[7])]
                    for feature, available, chosen, temperature in zip(
                        decision["features"],
                        decision["available"],
                        decision["selected"],
                        decision["temperature"],
                        strict=True,
                    ):
                        feature = np.asarray(feature, dtype=float)
                        scores = feature @ weight
                        logits = -scores / temperature
                        logits[~available] = -np.inf
                        probability = np.exp(logits - np.max(logits))
                        probability /= probability.sum()
                        score_gradient = np.column_stack(
                            [feature[:, 1:8] * (2 * theta[:7]), feature[:, 8] * np.exp(theta[7])]
                        )
                        log_gradient = (
                            probability @ score_gradient - score_gradient[chosen]
                        ) / temperature
                        gradient -= advantage * log_gradient / len(panel)
            close(
                statistics["episode_rewards"],
                rewards,
                "Ordinary-learning reward differs from its shared outcome rule",
                atol=1e-6,
            )
            close(
                statistics["previous_case_baselines"],
                baselines,
                "Ordinary-learning baseline used current/future returns",
                atol=1e-6,
            )
            close(
                statistics["episode_advantages"],
                advantages,
                "Ordinary-learning advantages differ",
                atol=1e-6,
            )
            close(
                statistics["gradient_norm_before_clip"],
                np.linalg.norm(gradient),
                "Recorded policy gradient norm differs from the saved categorical trajectory law",
                atol=2e-3,
                rtol=3e-3,
            )
            for case_id, values in grouped_rewards.items():
                decay = protocol["reinforce"]["baseline_decay"]
                baseline_by_case[case_id] = decay * baseline_by_case.get(case_id, 0.0) + (
                    1 - decay
                ) * float(np.mean(values))
            gradient_updates_verified += 1
            close(
                statistics["parameter_change_l2"],
                np.linalg.norm(np.asarray(nominee["parameters"]) - sample_parameters),
                "Recorded ordinary-learning parameter movement differs",
                atol=2e-6,
            )
            check(
                statistics["categorical_decisions"]
                == sum(
                    jobs[name]["decision_trace"]["decisions"] for name in item["sample_job_ids"]
                ),
                "Gradient update decision count differs",
            )
            if not nominee["parameters_equal_initial"]:
                nominees.append(
                    (
                        tuple(nominee["fitness"]),
                        -max(completed.index(name) for name in nominee["job_ids"]),
                        nominee,
                    )
                )
    check(
        referenced_jobs == set(audited), "Unreferenced or omitted complete simulator episodes exist"
    )
    if method == "reinforce":
        check(
            state["candidate_panels_completed"] == len(history)
            and state["training_panels_completed"] == 2 * len(history),
            "Ordinary-learning panel accounting differs",
        )
        for key, value in baseline_by_case.items():
            close(state["baseline_by_case"][key], value, "Final return baseline differs", atol=1e-6)
    if nominees:
        best = max(nominees, key=lambda item: item[:2])[2]
        check(
            state["best_trained"] == best,
            "Best non-anchor nomination or first-completed tie break differs",
        )
    else:
        check(
            state["best_trained"] is None,
            "A trained nominee exists without audited candidate evidence",
        )
    best = state["best_trained"]
    selected = anchor
    if best is not None and (anchor is None or tuple(best["fitness"]) > tuple(anchor["fitness"])):
        selected = best
    check(
        state["selected_training_nominee"] == selected,
        "Training-only anchor/nominee selection changed",
    )
    check(
        state["selected_is_trained"] == (best is not None and selected is best),
        "A zero anchor was relabelled as trained",
    )
    with (stage / "checkpoint.pkl").open("rb") as stream:
        checkpoint = pickle.load(stream)  # Only this campaign's trusted own checkpoint.
    check(
        checkpoint["identity"] == state["identity"] and checkpoint["state"] == state,
        "Durable optimizer checkpoint differs from its JSON state",
    )
    cma_updates_replayed = 0
    if method == "cma":
        check(
            checkpoint["es"].countiter == state["optimizer_updates"],
            "CMA object did not receive the declared real tell updates",
        )
        cma_updates_replayed = replay_cma(protocol, history, state, checkpoint)
    elif method == "reinforce":
        steps = [float(value["step"]) for value in checkpoint["adam"]["state"].values()]
        check(
            steps and all(int(value) == state["optimizer_updates"] for value in steps),
            "Adam did not receive the declared ordinary-learning steps",
        )
    else:
        check(checkpoint["es"] is None, "Random-search run contains an evolved CMA state")
    check(
        state["candidate_panels_completed"] <= protocol["max_candidate_panels"],
        "Candidate-panel allowance exceeded",
    )
    reason = state["stop_reason"]
    if reason == "candidate_panel_allowance":
        check(
            state["candidate_panels_completed"] == protocol["max_candidate_panels"],
            "Candidate allowance stop occurred before its declared endpoint",
        )
    elif reason == "insufficient_actions_for_complete_candidate_panel":
        check(
            actual + len(panel) * protocol["max_episode_actions"]
            > protocol["max_environment_actions"],
            "Early actual-action stop is unsupported",
        )
    elif reason == "episode_start_allowance":
        check(
            state["episode_starts"] + len(panel) > protocol["max_episode_starts"],
            "Early start-cap stop is unsupported",
        )
    else:
        check(reason == "deadline_preservation_reserve", "Unknown optimizer stopping rule")
    return {
        "method": method,
        "optimizer_updates": state["optimizer_updates"],
        "candidate_panels_completed": state["candidate_panels_completed"],
        "population_generations_completed": len(history),
        "anchor": anchor,
        "best_trained": best,
        "selected_is_trained_on_training_only": state["selected_is_trained"],
        "stop_reason": reason,
        "actual_environment_actions": actual,
        "max_environment_actions": protocol["max_environment_actions"],
        "remaining_allowance": protocol["max_environment_actions"] - actual,
        "pending_population": state["pending"] is not None,
        "ordinary_learning_is_same_controller_family": method == "reinforce",
        "categorical_score_gradients_reconstructed": gradient_updates_verified,
        "seeded_cma_ask_tell_updates_reconstructed": cma_updates_replayed,
        "optimizer_checkpoint_sha256": digest(stage / "checkpoint.pkl"),
        "best_nominee_proposed_after_updates": None
        if best is None
        else best["provenance"].get(
            "completed_updates_before_proposal",
            best["provenance"].get("completed_updates_before_evaluation"),
        ),
    }


def audit_equivalence(stage, state, protocol):
    pairs = []
    if protocol.get("equivalence_pairs"):
        names = list(state["jobs"])
        pairs = [(names[a], names[b], "jobs") for a, b in protocol["equivalence_pairs"]]
    elif "configurations" in protocol:
        engineering = [job["id"] for job in protocol["jobs"] if job["category"] == "engineering"]
        if len(engineering) == 2:
            pairs = [(*engineering, "episodes")]
    verified = []
    for first, second, container in pairs:
        traces = []
        for name in (first, second):
            folder = stage / container / name
            path = (
                folder / "episode/trace.npz"
                if (folder / "episode/trace.npz").exists()
                else folder / "trace.npz"
            )
            with np.load(path, allow_pickle=False) as data:
                traces.append(
                    {
                        k: data[k]
                        for k in ("actions", "proposed", "states", "dense", "frames", "goal_image")
                    }
                )
        for field in traces[0]:
            check(
                np.array_equal(traces[0][field], traces[1][field]),
                "Zero/native evidence differs: " + field,
            )
        verified.append(
            {
                "native": first,
                "zero": second,
                "actions_per_episode": len(traces[0]["actions"]),
                "arrays_equal": list(traces[0]),
            }
        )
    return verified


def audit(root, stage, explicit_protocol, allow_partial):
    state_path, freeze_path = stage / "state.json", stage / "source-freeze.json"
    state_bytes = state_path.read_bytes()
    state, freeze = json.loads(state_bytes), read(freeze_path)
    complete = state["phase"] == "complete"
    if complete:
        check(state.get("active") is None, "A completed stage retains an active controller job")
    check(
        complete or allow_partial,
        "Stage is not terminal complete; use --allow-partial for a limited snapshot",
    )
    resolver = Resolver(root)
    protocol = protocol_for(root, stage, freeze, resolver, explicit_protocol)
    design = identities(root, protocol, freeze, resolver)
    training = protocol.get("schema") == "lewm-campaign-training-v1"
    container = "jobs" if (stage / "jobs").is_dir() else "episodes"
    starts = {p.parent.name: p for p in (stage / container).glob("*/started.json")}
    expected = set(state.get("completed_job_ids", state.get("jobs", {}).keys()))
    check(expected <= set(starts), "State refers to an episode without its original started record")
    if complete:
        check(set(starts) == expected, "Completed stage has silently omitted starts")
    memory = None
    if design.get("memory_path"):
        with np.load(root / design["memory_path"], allow_pickle=False) as data:
            memory = dict(data)
    audited, technical = {}, {}
    for key in expected:
        folder = stage / container / key
        result = read(folder / "result.json")
        if result.get("error") or result.get("status") == "failed":
            technical[key] = {
                "error": result.get("error"),
                "resources": result.get("resources", {}),
                "environment_actions": result.get("environment_actions", result.get("actions", 0)),
            }
            continue
        context = episode_context(protocol, freeze, state, folder, key)
        if key in state.get("jobs", {}):
            check(
                state["jobs"][key] == context["outer"],
                "Stage state differs from its retained episode result",
            )
        audited[key] = audit_episode(root, stage, context, memory)
    check(
        not (complete and technical), "A complete stage contains unacknowledged technical results"
    )
    resources = collections.Counter()
    for row in audited.values():
        resources.update(row["resources"])
    overall = summary(list(audited.values()))
    limits = {}
    if training and complete:
        limits = audit_training(stage, protocol, freeze, state, audited)
    elif complete:
        check(
            len(starts)
            <= protocol.get(
                "max_episode_starts",
                protocol.get(
                    "maximum_all_episode_starts", protocol.get("maximum_episode_starts", np.inf)
                ),
            ),
            "Stage start limit exceeded",
        )
        check(
            overall["environment_actions"]
            <= protocol.get(
                "max_environment_actions", protocol.get("maximum_environment_actions", np.inf)
            ),
            "Stage action limit exceeded",
        )
        if protocol.get("schema") == "lewm-push-evaluation-v1":
            check(
                len(starts) == len(protocol["evaluation_jobs"]),
                "Frozen evaluation schedule is incomplete",
            )
        elif protocol.get("schema") == "lewm-navigation-comparison-v1":
            check(
                set(starts) == {job["job_id"] for job in protocol["jobs"]},
                "Frozen navigation comparison schedule is incomplete",
            )
            check(
                state["resource_totals"] == dict(resources),
                "Navigation comparison cumulative resources differ",
            )
            check(
                resources["environment_resets"] <= protocol["maximum_native_resets"],
                "Navigation comparison native-reset cap exceeded",
            )
        elif "recipes" in protocol:
            check(
                len(starts) == protocol["maximum_scored_episodes"],
                "Frozen PushT diagnostic grid is incomplete",
            )
        elif "configurations" in protocol:
            check(
                set(starts) == {job["id"] for job in protocol["jobs"]},
                "Frozen navigation diagnostic grid is incomplete",
            )
        if "environment_actions" in state:
            check(
                state["environment_actions"] == overall["environment_actions"],
                "Evaluation aggregate action count differs",
            )
        if "resources" in state:
            check(
                state["resources"] == dict(resources), "Evaluation aggregate resource counts differ"
            )
    equivalence = audit_equivalence(stage, state, protocol) if complete else []
    partial = set(starts) - set(audited)
    partial_proof = {}
    for key in partial:
        folder = stage / container / key
        live = folder / "live.json"
        partial_proof[key] = {
            "started_sha256": digest(folder / "started.json"),
            "live": read(live) if live.exists() else None,
            "technical_result": technical.get(key),
            "not_included_in_completed_outcome_denominators": True,
        }
    check(
        not complete or state_bytes == state_path.read_bytes(),
        "A claimed complete stage changed during the audit",
    )
    return {
        "verified_complete_records": True,
        "stage_complete": complete,
        "stage_phase": state["phase"],
        "stage": str(stage.relative_to(root)),
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "source_freeze_sha256": digest(freeze_path),
        "auditor_sha256": digest(Path(__file__)),
        "audited_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "Development-stage saved-evidence audit. This stage alone does not establish final generalization or evolutionary optimizer superiority.",
        "episode_starts_seen": len(starts),
        "audited_completed_episodes": len(audited),
        "partial_or_technical_records": partial_proof,
        "summary": overall,
        "resources": dict(resources),
        "optimizer": limits,
        "zero_native_equivalence": equivalence,
        "identities": resolver.proof(),
        "episodes": audited,
        "new_simulator_interactions": 0,
        "new_frozen_model_queries": 0,
        "new_learned_safety_queries": 0,
        "neural_embedding_scope": "Saved latent values/cache identity are checked; pretrained neural encoders/predictors and safety networks are never rerun.",
        "categorical_scope": "Saved linear-score controller laws can be recomputed from recorded features; this is audit replay, not additional training or environment evaluation.",
        "limitations": [
            "Visual image quality and provider billing/pod lifecycle are separate audits.",
            "Stage-level evidence does not aggregate unrelated correlated trials into one efficacy test.",
            "Full upstream shifted warm-start tensors are not saved; their recorded controller direction cannot be reconstructed independently where no prior-action fallback is recorded.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    campaign = read(root / "configs/lewm_campaign_20261031/campaign.json")
    check(
        os.environ.get("RUNPOD_POD_ID") in campaign["base_lanes"].values(),
        "Saved-evidence computation belongs on a declared rented campaign host",
    )
    stage = args.stage if args.stage.is_absolute() else root / args.stage
    out = args.out if args.out.is_absolute() else root / args.out
    check(not out.is_relative_to(stage), "Write a new audit artifact outside the immutable stage")
    check(not out.exists(), "Never overwrite an earlier audit report")
    protocol = args.protocol
    if protocol is not None and not protocol.is_absolute():
        protocol = root / protocol
    out.parent.mkdir(parents=True, exist_ok=True)
    source_path = out.with_suffix(".source.py")
    if source_path.exists():
        check(
            source_path.read_bytes() == Path(__file__).read_bytes(),
            "A prior audit source artifact differs",
        )
    else:
        source_path.write_bytes(Path(__file__).read_bytes())
    try:
        report = audit(root, stage, protocol, args.allow_partial)
    except BaseException:
        failure = out.with_suffix(".failure.json")
        check(not failure.exists(), "Never overwrite an audit failure record")
        failure.write_text(
            json.dumps(
                {
                    "stage": str(stage),
                    "audit_source_sha256": digest(source_path),
                    "error": traceback.format_exc(),
                    "new_simulator_interactions": 0,
                    "new_frozen_model_queries": 0,
                },
                indent=2,
            )
            + "\n"
        )
        raise
    report["auditor_source_path"] = str(source_path.relative_to(root))
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".part")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(out)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "verified_complete_records",
                    "stage_complete",
                    "stage",
                    "episode_starts_seen",
                    "audited_completed_episodes",
                    "summary",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
