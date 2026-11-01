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
    for path in (root / "configs/lewm_campaign_20261031").rglob("*.json"):
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


def locate_config(root, resolver, expected):
    for path in sorted((root / "configs/lewm_campaign_20261031").rglob("*.json")):
        if digest(path) == expected:
            return json.loads(resolver.bytes(str(path.relative_to(root)), expected))
    raise AssertionError("No retained prospective config matches referenced SHA256: " + expected)


def audit_navigation_nominees(root, protocol, resolver):
    if not protocol.get("nominee_provenance"):
        return []
    reservation = locate_config(root, resolver, protocol["reservation"]["sha256"])
    evidence = protocol["nominee_provenance"]
    jobs = protocol["jobs"]
    proofs = []
    for method in ("cma", "random", "reinforce"):
        declaration = evidence[method]
        stage = (root / reservation["candidate_sources"][method]["output_namespace"]).resolve()
        check(
            stage.is_relative_to(root / "results"), "Producer stage lies outside campaign results"
        )
        state = json.loads(
            resolver.bytes(
                str((stage / "state.json").relative_to(root)), declaration["state_sha256"]
            )
        )
        source = json.loads(
            resolver.bytes(
                str((stage / "source-freeze.json").relative_to(root)),
                declaration["source_freeze_sha256"],
            )
        )
        nominees = json.loads(
            resolver.bytes(
                str((stage / "nominees.json").relative_to(root)), declaration["nominees_sha256"]
            )
        )
        producer_protocol = locate_config(root, resolver, declaration["protocol_sha256"])
        check(
            source["protocol"] == producer_protocol
            and source["identity"] == state["identity"] == declaration["identity"]
            and state["phase"] == "complete"
            and state["active"] is None,
            "Selection references another or incomplete optimizer stage",
        )
        check(
            nominees["trained"] == state["best_trained"] and nominees["anchor"] == state["anchor"],
            "Selection producer nominees/state differ",
        )
        selected = nominees["trained"] if nominees["trained"] is not None else nominees["anchor"]
        check(
            selected["provenance"] == declaration["producer"],
            "Selection producer proposal/update provenance changed",
        )
        expected_cases = {c["case_id"]: c for c in producer_protocol["training_panel"]}
        rows = []
        references = {item["job_id"]: item for item in declaration["panel_results"]}
        check(
            set(references) == set(selected["job_ids"]) and len(references) == len(expected_cases),
            "Selection producer panel is incomplete",
        )
        for name in selected["job_ids"]:
            folder = stage / "jobs" / name
            start = read(folder / "started.json")
            result = read(folder / "result.json")
            check(
                digest(folder / "started.json") == references[name]["started_sha256"]
                and digest(folder / "result.json") == references[name]["result_sha256"],
                "Selection producer panel bytes changed",
            )
            spec = start["job"]
            check(
                canonical(spec) == start["input_sha256"] == result["input_sha256"],
                "Selection producer JobSpec hash differs",
            )
            check(
                spec["case"] == expected_cases[spec["case_id"]]
                and spec["environment_seed"] == spec["case"]["seed"]
                and spec["max_actions"] == producer_protocol["max_episode_actions"],
                "Selection producer case/seed/cap differs",
            )
            check(
                spec["parameters"] == selected["parameters"] and result["status"] == "complete",
                "Selection producer genotype/episode status differs",
            )
            rows.append(result)
        check(
            list(fitness(rows)) == selected["fitness"] == declaration["training_fitness"],
            "Selection producer fitness differs",
        )
        chosen_vectors = [j["parameters"] for j in jobs if j["arm"] == method]
        check(
            chosen_vectors and all(vector == selected["parameters"] for vector in chosen_vectors),
            "Selected runtime genotype differs from producer",
        )
        equivalent = np.array_equal(
            np.asarray(selected["parameters"], np.float32),
            np.asarray(nominees["anchor"]["parameters"], np.float32),
        )
        check(
            declaration["is_trained"] == (nominees["trained"] is not None and not equivalent),
            "Selection gives trained credit to an initial-equivalent actor",
        )
        proofs.append(
            {
                "method": method,
                "producer_identity": state["identity"],
                "optimizer_updates": state["optimizer_updates"],
                "nominee_provenance": selected["provenance"],
                "full_training_panel_fitness": selected["fitness"],
                "effective_initial_equivalence": equivalent,
            }
        )
    fixed = evidence["tuned_fixed"]
    stage = root / reservation["candidate_sources"]["tuned_fixed"]["output"]
    state = json.loads(
        resolver.bytes(str((stage / "state.json").relative_to(root)), fixed["state_sha256"])
    )
    source = json.loads(
        resolver.bytes(
            str((stage / "source-freeze.json").relative_to(root)), fixed["source_freeze_sha256"]
        )
    )
    fixed_protocol = locate_config(root, resolver, fixed["protocol_sha256"])
    check(
        source["protocol_sha256"] == fixed["protocol_sha256"] and state["phase"] == "complete",
        "Fixed-grid producer freeze/status differs",
    )
    planned = fixed_protocol["jobs"]
    panel_size = len({job["case"]["case_id"] for job in planned})
    winner = None
    for index in range(0, len(planned), panel_size):
        panel = planned[index : index + panel_size]
        check(len(panel) == panel_size, "Fixed-grid producer has an incomplete panel")
        rows = []
        for job in panel:
            folder = stage / "jobs" / job["job_id"]
            row = read(folder / "result.json")
            check(
                read(folder / "started.json") == job
                and row == state["jobs"][job["job_id"]]
                and row["parameters"] == job["parameters"],
                "Fixed-grid producer retained bytes differ",
            )
            rows.append(row)
        candidate = (fitness(rows), panel[0]["parameters"], [job["job_id"] for job in panel])
        if winner is None or candidate[0] > winner[0]:
            winner = candidate
    check(
        winner[2] == fixed["selected_candidate_job_ids"]
        and list(winner[0]) == fixed["training_fitness"],
        "Fixed-grid strict-earliest winner differs",
    )
    check(
        all(job["parameters"] == winner[1] for job in jobs if job["arm"] == "tuned_fixed"),
        "Tuned-fixed runtime genotype differs",
    )
    proofs.append(
        {
            "method": "tuned_fixed",
            "complete_grid_panels": len(planned) // panel_size,
            "selected_job_ids": winner[2],
            "full_training_panel_fitness": list(winner[0]),
            "is_evolutionary_training": False,
        }
    )
    return proofs


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
    elif schema in ("lewm-navigation-comparison-v1", "lewm-navigation-comparison-v2"):
        planned = next(job for job in protocol["jobs"] if job["job_id"] == key)
        check(start == planned, "Executed navigation comparison job differs from its freeze")
        case = planned["case"]
        seed = planned.get("environment_seed", case["seed"])
        cap = min(
            planned.get("max_actions", design["maximum_episode_actions"]),
            design["maximum_episode_actions"],
        )
        parameters = planned.get("parameters")
        if parameters is None:
            static = protocol["controller_static_parameters"]
            parameters = static.get("fixed_ap", static.get("fixed_geometry_progress"))
        check(parameters is not None, "Comparison parameters lack a frozen static origin")
        params = np.asarray(parameters, dtype=float)
        native = planned["method"] in ("native", "native_projected", "decoded_native")
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


def audit_model_work(
    episode, row, planner, contract, trace, *, clipped, projected, extra_encoder_frames=0
):
    plans = read(episode / "replans.json")
    projections = read(episode / "projection.json")
    c = row["resources"]
    check(
        c.get("encoder_frames", 0) == 2 * len(plans) + extra_encoder_frames,
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
            == len(plans) * 2 + extra_encoder_frames,
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
    close(artifact["action_mean"], contract["mean"], "Residual action mean differs", atol=0)
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


def retrieval_v2_features(
    memory, current, final, xy, shape, radius, margin, allowance, config, prefix
):
    feature, available = retrieval_features(
        memory, current, final, xy, shape, radius, margin, allowance
    )
    xy = np.asarray(xy, dtype=np.float32)
    successors = np.asarray(memory["successor_xy"], dtype=np.float32)
    distance = np.linalg.norm(successors - xy, axis=-1)
    available[:-1] &= (distance >= config["min_target_distance"]) & (
        distance <= config["max_target_distance"]
    )
    feature[:-1, 7] = distance / config["max_target_distance"]

    def recent(targets):
        previous = prefix["recent_target_xy"]
        if not previous:
            return np.zeros(len(targets), dtype=np.float32)
        points = np.asarray(previous, dtype=np.float32)
        delta = np.linalg.norm(targets[:, None] - points[None], axis=-1)
        similarity = np.exp(-0.5 * (delta / config["recent_scale"]) ** 2)
        penalty = similarity.max(axis=-1)
        if prefix["previous_current_xy"] is not None:
            displacement = np.linalg.norm(
                xy - np.asarray(prefix["previous_current_xy"], np.float32)
            )
            stall = np.clip(1 - displacement / config["stall_distance"], 0, 1)
            penalty += stall * similarity[:, -1]
        return penalty

    repeated = recent(successors)
    estimated = None
    progress = np.zeros(len(successors), dtype=np.float32)
    direct_recent = direct_progress = np.float32(0)
    if "_readout_coefficient" in memory:
        estimated = (
            np.asarray(final, dtype=np.float64) @ memory["_readout_coefficient"]
            + memory["_readout_intercept"]
        ).astype(np.float32)
        goal_distance = np.linalg.norm(estimated - xy)
        if config["readout_progress"]:
            progress = np.clip(
                (np.linalg.norm(successors - estimated, axis=-1) - goal_distance)
                / config["max_target_distance"],
                -1,
                1,
            )
            direct_progress = np.clip(-goal_distance / config["max_target_distance"], -1, 1)
        direct_recent = recent(estimated[None])[0]
        feature[-1, 7] = goal_distance / config["max_target_distance"]
        if config["direct_goal_locality"]:
            available[-1] &= goal_distance <= config["max_target_distance"]
    feature = np.concatenate(
        [
            feature[:, :8],
            np.r_[repeated, direct_recent][:, None],
            np.r_[progress, direct_progress][:, None],
            feature[:, 8:],
        ],
        axis=-1,
    ).astype(np.float32)
    return feature, available, estimated


def audit_retrieval(context, trace, plans, memory, stage):
    folder, row, outer = context["episode"], context["row"], context["outer"]
    artifact_path = folder / "controller.json"
    if context["native"]:
        check(not artifact_path.exists(), "Native navigation contains a retrieval controller")
        return {"kind": "native", "controller_forward_replays": 0}
    artifact = read(artifact_path)
    version2 = artifact["format"] == "lewm-observed-target-retrieval-v2"
    check(
        version2 or artifact["format"] == "lewm-observed-target-retrieval-v1",
        "Unknown retrieval controller artifact",
    )
    theta = np.asarray(artifact["parameters"])
    close(theta, context["parameters"], "Retrieval genotype differs from the frozen job", atol=0)
    check(theta.shape == ((10,) if version2 else (8,)), "Retrieval genotype dimension differs")
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
    sparse_zero_query_counter = "retrieval_candidate_targets" not in row["resources"]
    if sparse_zero_query_counter:
        check(
            not history
            and not plans
            and row["resources"].get("retrieval_queries", 0) == 0
            and (row.get("controller_failure") or "").startswith("No admitted local")
            and len(trace["actions"]) == 0
            and row["unfinished"]
            and not row["safe_completion"]
            and not row["engaged"],
            "Absent target-candidate counter is admitted only for a recorded zero-action empty-support refusal",
        )
    check(
        row["resources"].get("retrieval_candidate_targets", 0)
        == len(history) * (len(memory["source_embeddings"]) + 1),
        "Target candidate ledger differs",
    )
    weights = (
        np.r_[1, np.clip(theta[:9], -6, 6), np.exp(np.clip(theta[9], -6, 6))]
        if version2
        else np.r_[1, np.clip(theta[:7], -6, 6) ** 2, np.exp(np.clip(theta[7], -6, 6))]
    )
    prefix = {"recent_target_xy": [], "previous_current_xy": None}
    if version2:
        check(
            artifact["controller_config"] == context["design"]["controller_config"],
            "V2 causal/locality constants changed",
        )
        check(
            artifact["goal_readout_sha256"] == context["design"]["goal_readout"]["sha256"],
            "V2 frozen readout identity changed",
        )
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
        arguments = (
            memory,
            item["current_latent"],
            item["final_latent"],
            current_xy,
            context["case"].get("shape", {"kind": "none"}),
            artifact["body_radius"],
            artifact["margin"],
            artifact["allowance"],
        )
        estimated = None
        if version2:
            check(
                item["causal_context"] == prefix,
                "V2 target context used a different or future history",
            )
            feature, available, estimated = retrieval_v2_features(
                *arguments, artifact["controller_config"], prefix
            )
            close(
                estimated,
                item["estimated_goal_xy"],
                "V2 goal readout cannot be replayed",
                atol=2e-5,
            )
            check(
                bool(available[-1]) == item["direct_goal_available"],
                "V2 direct-goal locality mask differs",
            )
        else:
            feature, available = retrieval_features(*arguments)
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
        if version2:
            point = (
                estimated
                if chosen == len(memory["source_embeddings"])
                else np.asarray(memory["successor_xy"][chosen], dtype=np.float32)
            )
            close(
                point,
                item["selected_target_xy_estimate"],
                "V2 saved target position differs",
                atol=2e-5,
            )
            prefix = {
                "recent_target_xy": [
                    *prefix["recent_target_xy"],
                    item["selected_target_xy_estimate"],
                ][-artifact["controller_config"]["recent_window"] :],
                "previous_current_xy": current_xy.astype(np.float32).tolist(),
            }
    saved_refusals = []
    if version2:
        counters = row["resources"]
        unrecorded_refusal = int(
            (row.get("controller_failure") or "").startswith("No admitted local")
        )
        decoded_calls = row["resources"].get("decoded_task_cost_call_attempts", 0)
        attempts = len(history) + unrecorded_refusal + 2 * decoded_calls
        check(
            counters.get("goal_readout_call_attempts", 0)
            == counters.get("goal_readout_call_successes", 0)
            == attempts
            and counters.get("goal_readout_call_failures", 0) == 0,
            "V2 readout attempt/success ledger differs from retained successful selector queries",
        )
        refusal_path = folder / "refusal-history.json"
        if refusal_path.exists():
            saved_refusals = read(refusal_path)
            check(len(saved_refusals) == unrecorded_refusal, "Final refusal sidecar count differs")
            for refusal in saved_refusals:
                check(
                    refusal["causal_context"] == prefix, "Saved refusal used a future causal prefix"
                )
                close(
                    refusal["current_xy"],
                    trace["states"][-1],
                    "Saved refusal current XY differs",
                    atol=2e-5,
                )
                _, mask, _ = retrieval_v2_features(
                    memory,
                    refusal["current_latent"],
                    refusal["final_latent"],
                    refusal["current_xy"],
                    context["case"].get("shape", {"kind": "none"}),
                    artifact["body_radius"],
                    artifact["margin"],
                    artifact["allowance"],
                    artifact["controller_config"],
                    prefix,
                )
                check(not mask.any(), "Empty-support refusal had an admitted target")
            unrecorded_refusal -= len(saved_refusals)
    return {
        "kind": "observed_target_retrieval10" if version2 else "observed_target_retrieval8",
        "controller_forward_replays": len(history),
        "sparse_zero_query_counter_validated": sparse_zero_query_counter,
        "refusal_queries_replayed": len(saved_refusals),
        "categorical_decisions_verified": len(decisions),
        "floating_point_near_tie_selections": near_ties,
        "unrecorded_terminal_support_refusal_queries": unrecorded_refusal if version2 else 0,
        "terminal_refusal_limitation": "Frozen v2 saved its controller artifact only initially; if a terminal support refusal occurred, its attempt/readout/encoder counters are verified but its unsaved current latent/feature table cannot be independently replayed."
        if version2 and unrecorded_refusal
        else None,
        "latent_scope": "Cached/live latent values and their declared hashes are verified from saved evidence; frozen neural encoders are not rerun.",
    }


def audit_journal(root, context):
    row = context["row"]
    declaration = row.get("progress_journal")
    if declaration is None:
        return None
    path = Path(declaration["path"])
    if not path.is_absolute():
        path = root / path
    check(digest(path) == declaration["sha256"], "Durable attempt-journal identity differs")
    check(
        declaration["finished"]
        and not declaration["failed"]
        and declaration["write_error"] is None,
        "Completed episode has an unfinished or failed persistence journal",
    )
    previous, previous_resources, previous_actions = None, {}, 0
    metadata, records, final = None, 0, None
    with path.open("rb") as stream:
        for index, encoded in enumerate(stream):
            check(encoded.endswith(b"\n"), "Completed journal has an unterminated byte tail")
            record = json.loads(encoded)
            recorded_hash = record.pop("record_digest")
            actual_hash = hashlib.sha256(
                json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            check(
                actual_hash == recorded_hash
                and record["seq"] == index
                and record["previous_digest"] == previous,
                "Durable journal hash chain or sequence differs",
            )
            actions, counters = record["environment_actions"], record["resources"]
            check(
                actions >= previous_actions and actions <= context["cap"],
                "Journal action count regressed or exceeds cap",
            )
            check(
                all(counters.get(key, -1) >= value for key, value in previous_resources.items())
                and counters["environment_actions"] == actions,
                "Durable journal dropped/regressed a cumulative attempt counter",
            )
            check(final is None, "Journal contains records after finalization")
            if index == 0:
                check(record["event"] == "start", "Journal has no immutable start record")
                metadata = record["metadata"]
                check(
                    metadata["job_id"] == context["folder"].name
                    and metadata["started_sha256"] == digest(context["folder"] / "started.json")
                    and metadata["case_id"] == context["case"]["case_id"]
                    and metadata["environment_seed"] == context["seed"]
                    and metadata["max_actions"] == context["cap"],
                    "Journal belongs to another job, case, seed or action cap",
                )
            if record["event"] == "final":
                final = record
            previous, previous_resources, previous_actions = recorded_hash, counters, actions
            records += 1
    check(
        final is not None and final["status"] == "complete",
        "Complete episode lacks final journal record",
    )
    result_resources = row["resources"]
    sparse_zero_action_counter = "environment_actions" not in result_resources
    if sparse_zero_action_counter:
        selector_path = context["episode"] / "selector-history.json"
        check(
            previous_actions == 0
            and previous_resources.get("environment_actions") == 0
            and (row.get("controller_failure") or "").startswith("No admitted local")
            and selector_path.exists()
            and read(selector_path) == []
            and result_resources.get("retrieval_queries", 0) == 0
            and row["unfinished"]
            and not row["safe_completion"]
            and not row["engaged"],
            "Absent action counter is admitted only for a recorded zero-action empty-support refusal",
        )
        result_resources = {**result_resources, "environment_actions": 0}
    check(
        previous_resources == result_resources,
        "Final durable attempt counters differ from the episode result",
    )
    check(
        records == declaration["records"] == declaration["fsync_calls"]
        and previous == declaration["last_record_digest"],
        "Journal record/fsync/digest summary differs",
    )
    pins = {
        "legacy_navigation_v1_sha256": "experiments/scripts/run_lewm_navigation_campaign.py",
        "shared_harness_sha256": "experiments/helpers/lewm_campaign_training.py",
        "controller_source_sha256": "experiments/helpers/lewm_navigation_retrieval_v2.py",
    }
    if "navigation_v3_sha256" in metadata["source_pins"]:
        pins.update(
            navigation_v3_sha256="experiments/scripts/run_lewm_navigation_campaign_v3.py",
            navigation_v2_parent_sha256="experiments/scripts/run_lewm_navigation_campaign_v2.py",
            decoded_task_cost_sha256="experiments/helpers/lewm_navigation_decoded_cost.py",
        )
    else:
        pins["navigation_v2_sha256"] = "experiments/scripts/run_lewm_navigation_campaign_v2.py"
    for key, source in pins.items():
        check(
            metadata["source_pins"][key] == digest(root / source),
            "Journal source pin differs: " + key,
        )
    check(
        metadata["writer_source_sha256"]
        == digest(root / "experiments/helpers/lewm_progress_journal.py"),
        "Journal writer source pin differs",
    )
    counters = row["resources"]
    check(
        counters.get("actual_safety_call_attempts", 0)
        == counters.get("actual_safety_call_successes", 0)
        and counters.get("actual_safety_call_failures", 0) == 0,
        "Safety-head attempt/success ledger differs",
    )
    check(
        counters.get("actual_safety_candidate_rollouts_attempted", 0)
        == counters.get("actual_safety_candidate_rollouts_successful", 0)
        == counters.get("learned_candidate_checks", 0)
        and counters.get("actual_safety_transition_rows_attempted", 0)
        == counters.get("actual_safety_transition_rows_successful", 0)
        == counters.get("learned_transition_rows", 0),
        "Safety-head attempted/successful row work differs from learned checks",
    )
    return {
        "sha256": declaration["sha256"],
        "hash_chained_records": records,
        "all_final_counters_reconstructed": True,
        "sparse_zero_action_counter_validated": sparse_zero_action_counter,
        "explicit_absent_zero_counter": "environment_actions"
        if sparse_zero_action_counter
        else None,
    }


def audit_final_sidecars(root, context):
    folder, outer = context["episode"], context["outer"]
    declaration = outer.get("extra", {}).get("final_sidecars", {})
    for name, record in declaration.items():
        check(
            name in {"controller-final.json", "refusal-history.json", "decoded-task-cost.json"},
            "Unknown final sidecar",
        )
        check(digest(folder / name) == record["sha256"], "Final sidecar hash differs")
        path = Path(record["path"])
        check(
            (path if path.is_absolute() else root / path) == folder / name,
            "Final sidecar path differs",
        )
    if (folder / "controller-final.json").exists():
        initial, final = read(folder / "controller.json"), read(folder / "controller-final.json")
        check(
            {k: v for k, v in initial.items() if k != "refusals"}
            == {k: v for k, v in final.items() if k != "refusals"},
            "Final controller parameters or static authority differ from its initial artifact",
        )
        check(
            final["refusals"] == read(folder / "refusal-history.json"),
            "Final refusal records differ",
        )
    return len(declaration)


def audit_decoded_cost(context, work, memory):
    folder, row = context["episode"], context["row"]
    backend = row.get("task_cost_backend", "original-latent-v1")
    path = folder / "decoded-task-cost.json"
    if backend == "original-latent-v1":
        check(not path.exists(), "Original task metric contains decoded scoring work")
        return None
    check(
        backend == "decoded-current-target-v1" and path.exists(),
        "Unknown or missing decoded metric evidence",
    )
    artifact = read(path)
    check(artifact["backend"] == backend, "Decoded metric backend changed")
    check(
        artifact["goal_readout_sha256"] == context["design"]["goal_readout"]["sha256"],
        "Decoded metric readout identity differs",
    )
    history, counters = artifact["history"], row["resources"]
    check(
        len(history)
        == counters["decoded_task_cost_call_attempts"]
        == counters["decoded_task_cost_call_successes"]
        == counters["actual_cost_call_attempts"],
        "Decoded scoring did not wrap exactly each original frozen cost call",
    )
    check(
        counters.get("decoded_task_cost_call_failures", 0) == 0, "Decoded scoring failure omitted"
    )
    endpoint_rows, target_rows = 0, 0
    for index, item in enumerate(history):
        predicted, target, cost = [
            np.asarray(item[k])
            for k in ("predicted_terminal_xy", "current_target_xy", "decoded_task_cost")
        ]
        check(
            item["call"] == index and predicted.shape[-1] == target.shape[-1] == 2,
            "Decoded row ordering/shape differs",
        )
        expected = ((predicted - target) ** 2).sum(-1).astype(np.float32)
        close(
            cost,
            expected,
            "Decoded XY distance does not reproduce its actual task scores",
            atol=1e-5,
            rtol=1e-6,
        )
        check(
            item["score_dtype"] == "torch.float32" and item["projection_dtype"] == "torch.float64",
            "Decoded scoring dtype contract differs",
        )
        check(
            item["candidate_rows"] == int(np.prod(predicted.shape[:-1])),
            "Decoded candidate row count differs",
        )
        endpoint_rows += item["candidate_rows"]
        target_rows += int(np.prod(target.shape[:-1]))
    for role, rows in (("endpoint", endpoint_rows), ("target", target_rows)):
        check(
            counters[f"decoded_task_{role}_rows_attempted"]
            == counters[f"decoded_task_{role}_rows_successful"]
            == rows,
            "Decoded projection attempted/successful rows differ",
        )
    check(
        counters["decoded_task_candidate_rows_attempted"]
        == counters["decoded_task_candidate_rows_successful"]
        == endpoint_rows
        == work["model_candidate_rollouts"],
        "Decoded scoring changed the underlying frozen imagination work",
    )
    check(
        counters["decoded_task_readout_call_attempts"]
        == counters["decoded_task_readout_call_successes"]
        == 2 * len(history)
        and counters["decoded_task_readout_rows_attempted"]
        == counters["decoded_task_readout_rows_successful"]
        == endpoint_rows + target_rows
        and counters.get("decoded_task_readout_call_failures", 0) == 0,
        "Decoded readout call/row ledger differs",
    )
    selector_path = folder / "selector-history.json"
    selectors = read(selector_path) if selector_path.exists() else []
    check(len(selectors) in (0, len(work["plans"])), "Decoded target query/replan count differs")
    cursor, target_replays = 0, 0
    for index, plan in enumerate(work["plans"]):
        calls = (
            context["planner"]["iterations"] + 1 + int(plan.get("executed") not in ("none", "mean"))
        )
        block = history[cursor : cursor + calls]
        check(len(block) == calls, "Decoded cost plan block is incomplete")
        for item in block[1:]:
            close(
                item["current_target_xy"],
                block[0]["current_target_xy"],
                "Planning target changed inside a CEM solve",
                atol=0,
            )
        if selectors:
            selected = selectors[index]
            embedding = (
                selected["final_latent"]
                if selected["memory_index"] is None
                else memory["successor_embeddings"][selected["memory_index"]]
            )
            expected_target = (
                np.asarray(embedding, dtype=np.float64) @ memory["_readout_coefficient"]
                + memory["_readout_intercept"]
            )
            close(
                np.asarray(block[0]["current_target_xy"]).reshape(-1, 2)[0],
                expected_target,
                "Decoded metric ignored the actual selected target",
                atol=1e-8,
            )
            target_replays += 1
        elif cursor:
            close(
                block[0]["current_target_xy"],
                history[0]["current_target_xy"],
                "Direct native target changed between replans",
                atol=0,
            )
        cursor += calls
    check(cursor == len(history), "Decoded scoring has unassigned model-cost calls")
    return {
        "cost_calls_reconstructed": len(history),
        "candidate_rows_reconstructed": endpoint_rows,
        "selected_target_readout_replays": target_replays,
        "new_frozen_model_queries": 0,
        "scope": "All recorded decoded XY/task scores and query/row counts are replayed. Selected memory/recorded goal targets are decoded independently where available; native RGB embeddings and raw predicted latent endpoints were not saved and neural encoders/predictors are not rerun.",
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
        extra_encoder_frames=2
        if (row.get("controller_failure") or "").startswith("No admitted local")
        else 0,
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
        "attempt_journal": audit_journal(root, context),
        "final_sidecars_verified": audit_final_sidecars(root, context),
        "decoded_cost": audit_decoded_cost(context, work, memory),
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
    effective_parameter_updates = 0

    def candidate(parameters, job_ids, candidate_record=None):
        check(len(job_ids) == len(panel), "A genotype lacks its full fixed training panel")
        rows = []
        low, high = protocol["parameter_bounds"]
        check(
            np.isfinite(parameters).all()
            and np.all(np.asarray(parameters) >= low)
            and np.all(np.asarray(parameters) <= high),
            "Executed persistent parameters violate the frozen bounds",
        )
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
        preceding_parameters = np.asarray(initial, dtype=np.float32).astype(float)
        for index, item in enumerate(history):
            check(item["update"] == index, "Ordinary-learning update indexing differs")
            check(
                read(stage / f"update-{index:03}.json") == item,
                "Ordinary-learning update file differs from its durable history",
            )
            nominee = item["nominee"]
            candidate(nominee["parameters"], nominee["job_ids"], nominee)
            sample_parameters = read(stage / "jobs" / item["sample_job_ids"][0] / "started.json")[
                "job"
            ]["parameters"]
            close(
                sample_parameters,
                preceding_parameters,
                "Ordinary-learning rollout was not sampled from the preceding actor parameters",
                atol=0,
            )
            candidate(sample_parameters, item["sample_job_ids"])
            check(
                nominee["provenance"]["method"] == "reinforce"
                and nominee["provenance"]["update"] == index
                and nominee["provenance"]["completed_updates_before_evaluation"] == index + 1,
                "Ordinary-learning nominee did not follow the declared gradient update",
            )
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
                    signed10 = len(theta) == 10
                    weight = (
                        np.r_[1, theta[:9], np.exp(theta[9])]
                        if signed10
                        else np.r_[1, theta[:7] ** 2, np.exp(theta[7])]
                    )
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
                        score_gradient = (
                            np.column_stack([feature[:, 1:10], feature[:, 10] * np.exp(theta[9])])
                            if signed10
                            else np.column_stack(
                                [
                                    feature[:, 1:8] * (2 * theta[:7]),
                                    feature[:, 8] * np.exp(theta[7]),
                                ]
                            )
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
            effective_parameter_updates += int(
                not np.array_equal(
                    np.asarray(nominee["parameters"], dtype=np.float32),
                    np.asarray(sample_parameters, dtype=np.float32),
                )
            )
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
            preceding_parameters = np.asarray(nominee["parameters"])
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
        close(
            checkpoint["parameter"],
            preceding_parameters,
            "Retained ordinary-learning actor differs from its final evaluated update",
            atol=0,
        )
        groups = checkpoint["adam"]["param_groups"]
        check(
            len(groups) == 1
            and groups[0]["lr"] == protocol["reinforce"]["learning_rate"]
            and tuple(groups[0]["betas"]) == (0.9, 0.999)
            and groups[0]["eps"] == 1e-8
            and groups[0]["weight_decay"] == 0,
            "Retained Adam hyperparameters differ from the frozen ordinary learner",
        )
        for value in checkpoint["adam"]["state"].values():
            first, second = np.asarray(value["exp_avg"]), np.asarray(value["exp_avg_sq"])
            check(
                first.shape == second.shape == initial.shape
                and np.isfinite(first).all()
                and np.isfinite(second).all()
                and (second >= 0).all(),
                "Retained Adam moments are malformed",
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
        "ordinary_updates_with_effective_float32_parameter_change": effective_parameter_updates,
        "best_nominee_effectively_equals_initial_actor": best is not None
        and np.array_equal(
            np.asarray(best["parameters"], dtype=np.float32),
            np.asarray(initial, dtype=np.float32),
        ),
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
    nominee_proofs = audit_navigation_nominees(root, protocol, resolver)
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
        if design.get("goal_readout"):
            declaration = design["goal_readout"]
            resolver.hash(declaration["path"], declaration["sha256"])
            with np.load(root / declaration["path"], allow_pickle=False) as data:
                memory["_readout_coefficient"] = data["raw_feature_coefficient"].copy()
                memory["_readout_intercept"] = data["raw_feature_intercept"].copy()
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
        elif protocol.get("schema") in (
            "lewm-navigation-comparison-v1",
            "lewm-navigation-comparison-v2",
        ):
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
        "navigation_nominee_provenance": nominee_proofs,
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


def execution_host_authority(root):
    """Admit the two original pods or the one exact prospectively relocated pod.

    This is analysis-host authority. It does not assert where the audited stage
    executed; the existing stage/source/protocol replay remains unchanged.
    """
    campaign_path = root / "configs/lewm_campaign_20261031/campaign.json"
    campaign = read(campaign_path)
    expected_base = {"pusht": "wq4ar2qx2ff6r4", "navigation": "vukovc3t4o1dui"}
    check(campaign["base_lanes"] == expected_base, "Original campaign host mapping changed")
    pod = os.environ.get("RUNPOD_POD_ID")
    if pod in expected_base.values():
        return {
            "pod_id": pod,
            "kind": "original_campaign_base_lane",
            "campaign": {
                "path": str(campaign_path.relative_to(root)),
                "sha256": digest(campaign_path),
            },
        }
    check(pod == "nj2fqkp2prgk8x", "Audit host is not an explicitly authorized campaign pod")

    identities = {
        "relocation": (
            "configs/lewm_campaign_20261031/navigation-secondary-r2-relocation-v1.json",
            "bdc7808f6ce172864a2d3e2fafdd00d1af6c5597714f20fca4741243ee924c70",
        ),
        "authoritative_secondary_manifest": (
            "configs/lewm_campaign_20261031/navigation-budget30-relocated-v1/manifest.json",
            "6cbf1c890c1451bd79e9702db3f7bbed69d783cb3df6496b3ad44c48d62fbf40",
        ),
        "original_secondary_manifest": (
            "configs/lewm_campaign_20261031/navigation-budget30-comparison-v1/manifest.json",
            "b6900ee408bd25bf8e73f2928750639ec81d75a81c50d813662c69d1e2a37225",
        ),
        "unstarted_proof": (
            "results/evidence-audits/navigation-secondary-r2-unstarted-v2.json",
            "0431d94920683616fb0ca988b132210e801e70ec0c339f28332f2a504a5f6d62",
        ),
    }
    references, values = {}, {}
    for label, (name, expected) in identities.items():
        path = root / name
        check(digest(path) == expected, "Recovery audit authority bytes differ: " + label)
        references[label] = {"path": name, "sha256": expected}
        values[label] = read(path)
    relocation = values["relocation"]
    manifest = values["authoritative_secondary_manifest"]
    proof = values["unstarted_proof"]
    check(
        relocation["schema"] == "lewm-navigation-secondary-relocation-v1"
        and relocation["study_family"] == "secondary30"
        and relocation["repeat_id"] == 2
        and relocation["new_host"] == {"host": "recovery", "pod_id": pod}
        and relocation["old_host"] == {"host": "pusht", "pod_id": expected_base["pusht"]},
        "Only the exact authorized secondary-repeat2 recovery host may qualify",
    )
    check(
        all(
            relocation[field] is True
            for field in (
                "whole_unstarted_repeat_block",
                "scientific_settings_unchanged",
                "all_cases_seeds_budgets_preserved",
                "all_local_methods_and_controls_moved",
            )
        )
        and relocation["final_pixels_read"] == 0
        and relocation["workers_started"] == 0,
        "Relocation authority must preserve the prospective complete scientific block",
    )
    check(
        relocation["unstarted_proof"] == references["unstarted_proof"]
        and relocation["original_secondary_manifest"]
        == references["original_secondary_manifest"]
        and manifest["metadata_relocation"] == references["relocation"]
        and manifest["original_secondary_manifest"]
        == references["original_secondary_manifest"],
        "Recovery host authority references differ",
    )
    repeat = [row for row in manifest["schedule"] if row["repeat_id"] == 2]
    check(
        len(repeat) == 1
        and repeat[0]["host"] == "recovery"
        and repeat[0]["pod_id"] == pod
        and repeat[0]["training_controls"]
        == "results/navigation-budget30-relocated-v1/r2/training-controls",
        "Authoritative secondary manifest does not assign the complete repeat to this pod",
    )
    check(
        proof["schema"] == "lewm-secondary-r2-unstarted-proof-v1"
        and proof["pod_id"] == expected_base["pusht"]
        and proof["old_secondary_r2_output_namespace"]
        == "results/navigation-budget30-comparison-v1/r2"
        and proof["old_secondary_r2_output_exists"] is False
        and proof["old_secondary_r2_output_children"] == []
        and proof["original_secondary_manifest"]["sha256"]
        == references["original_secondary_manifest"]["sha256"]
        and proof["new_nn_calls"] == proof["new_simulator_calls"] == 0,
        "Qualified old-host unstarted proof differs",
    )
    return {"pod_id": pod, "kind": "exact_prospective_secondary_r2_relocation", **references}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    host_authority = execution_host_authority(root)
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
    report["execution_host_authority"] = host_authority
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
