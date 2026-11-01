"""Rented-host bridge from a compact PushT controller to common-panel CMA/RS."""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import importlib

importlib.import_module("triton")
import argparse
import collections
from contextlib import ExitStack
import datetime as dt
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm_campaign_training import CampaignTraining, EpisodeOutcome, JobSpec
from helpers.lewm18_planner import PlanningCEM
from helpers.lewm_navigation_evolution import ActionContract
from helpers.lewm_proposal_pilot import write_json
from helpers.lewm_push_evolution import PushEvolutionAdapter, PushEvolutionSpec, PushPlanningCEM
from helpers.learned_motion import LearnedMotionChecker, load_motion
from helpers.lewm_screen import image_stats
from helpers.pusht_smoke import Scaler, arena_clearance, physical_clearance
from helpers.safety_model import pusht_constraints, scaler_digest
from scripts.run_lewm_push_diagnostic import run_episode


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identities():
    names = (
        "experiments/scripts/run_lewm_push_campaign.py",
        "experiments/scripts/run_lewm_push_diagnostic.py",
        "experiments/scripts/run_pusht_smoke.py",
        "experiments/helpers/lewm_campaign_training.py",
        "experiments/helpers/lewm_push_evolution.py",
        "experiments/helpers/lewm_navigation_evolution.py",
        "experiments/helpers/lewm18_planner.py",
        "experiments/helpers/lewm16_planner.py",
        "experiments/helpers/cem_refinements.py",
        "experiments/helpers/lewm_proposal_adapter.py",
        "experiments/helpers/lewm_residual.py",
        "experiments/helpers/learned_motion.py",
        "experiments/helpers/pusht_smoke.py",
        "experiments/helpers/safety_model.py",
        "experiments/helpers/safecem10_planner.py",
    )
    return {name: digest(ROOT / name) for name in names}


class ProgressCounter(collections.Counter):
    def __init__(self, report):
        super().__init__()
        self.report = report

    def publish(self):
        self.report(int(self.get("environment_actions", 0)), dict(self))

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if key in ("environment_resets", "environment_actions"):
            self.publish()


class AccountedModel(torch.nn.Module):
    def __init__(self, frozen, counters):
        super().__init__()
        self.frozen, self.counters = frozen, counters

    def encode(self, info):
        c = self.counters
        c["actual_encoder_call_attempts"] += 1
        c["actual_encoder_frames_attempted"] += int(
            info["pixels"].shape[0] * info["pixels"].shape[1]
        )
        c.publish()
        result = self.frozen.encode(info)
        c["actual_encoder_call_successes"] += 1
        return result

    def get_cost(self, info, candidates):
        c = self.counters
        b, n, h, _ = candidates.shape
        c["actual_cost_call_attempts"] += 1
        c["actual_candidate_rollouts_attempted"] += int(b * n)
        c["actual_candidate_block_steps_attempted"] += int(b * n * h)
        c.publish()
        result = self.frozen.get_cost(info, candidates)
        c["actual_cost_call_successes"] += 1
        return result


class PushContextAdapter(PushEvolutionAdapter):
    def artifact(self):
        return {
            **super().artifact(),
            "causal_plan_context": "Shifted native plan direction when available; otherwise previous actually executed native action supplied through the upstream policy action input. No CEM initialization changes.",
        }


class CausalPushPlanningCEM(PushPlanningCEM):
    """Use existing causal action feedback when full-plan execution empties the buffer."""

    def solve(self, info_dict, init_action=None):
        controller = self.proposal_adapter
        count, source, native_previous = 0, None, None
        if controller is not None:
            context = init_action
            source = "shifted_native_plan"
            if context is None or not context.shape[1]:
                previous = info_dict["action"]
                if previous.ndim != 3 or previous.shape[0] != 1 or previous.shape[-1] != 2:
                    raise ValueError("Expected the upstream normalized previous native action")
                context = previous[:, -1:].repeat(1, 1, 5)
                native_previous = (
                    previous[0, -1].detach().cpu().double().numpy() * controller.action_scale
                    + controller.action_mean
                ).tolist()
                source = "previous_executed_native_action"
            controller.set_plan_context(context)
            count = len(controller.history)
        try:
            # Avoid the old context wrapper resetting the declared fallback to None.
            # The native CEM receives its original init_action unchanged.
            return PlanningCEM.solve(self, info_dict, init_action)
        finally:
            if controller is not None and len(controller.history) > count:
                controller.history[-1].update(
                    plan_context_source=source,
                    previous_executed_native_action=native_previous,
                    goal_information="Frozen current/goal latent cost and causal shifted-plan or previous-executed-action direction; no exact final-goal XY",
                )


def load_push_runtime(design):
    if os.environ.get("RUNPOD_POD_ID") != design["execution_pod_id"]:
        raise RuntimeError("Only the declared rented PushT pod may load this stage")
    if not torch.cuda.is_available():
        raise RuntimeError("The declared rented GPU is required")
    for name, expected in design["asset_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("Frozen asset changed: " + name)
    current_sources = source_identities()
    if current_sources != design["source_sha256"]:
        raise RuntimeError("Frozen PushT stage source changed")
    from stable_worldmodel.policy import AutoCostModel

    torch.set_num_threads(4)
    checkpoint = ROOT / design["world_model"]
    model = AutoCostModel(str(checkpoint)[: -len("_object.ckpt")], cache_dir=checkpoint.parent)
    model = model.cuda().eval().requires_grad_(False)
    head, provenance = load_motion(ROOT / design["motion_model"], device="cuda")
    if not head.calibration or not head.calibration["ready"]:
        raise RuntimeError("The frozen learned motion checker is not ready")
    if head.spec.max_horizon < design["planner"]["horizon"] or head.spec.action_block != 5:
        raise RuntimeError("Frozen motion model does not cover the planner horizon")
    if head.spec.body_radius != design["body_radius_px"]:
        raise RuntimeError("Protected body differs from the prospective protocol")
    if head.spec.world_model_sha256 and head.spec.world_model_sha256 != digest(checkpoint):
        raise RuntimeError("Learned safety and world-model identities differ")
    columns = json.loads((ROOT / design["scalers"]).read_text())["columns"]
    process = {name: Scaler(value["mean"], value["scale"]) for name, value in columns.items()}
    action = process["action"]
    if (
        head.spec.action_scaler_sha256
        and scaler_digest(action.mean_, action.scale_) != head.spec.action_scaler_sha256
    ):
        raise RuntimeError("Learned safety and action-scaler identities differ")
    contract = ActionContract(tuple(action.mean_), tuple(action.scale_), (-1, -1), (1, 1))
    spec = PushEvolutionSpec(**design["controller_spec"])
    if spec.margin_px != design["margin_px"] or spec.body_radius_px != design["body_radius_px"]:
        raise RuntimeError(
            "Controller geometry features differ from the fixed safety specification"
        )
    return SimpleNamespace(
        model=model,
        head=head,
        provenance=provenance,
        process=process,
        contract=contract,
        spec=spec,
        sources=current_sources,
    )


def task_progress(row):
    goal = np.asarray(row["goal_state"])
    start, final = np.asarray(row["initial_state"]), np.asarray(row["final_state"])
    initial_position = np.linalg.norm(goal[:4] - start[:4])
    final_position = np.linalg.norm(goal[:4] - final[:4])

    def angle(a, b):
        return abs((a - b + np.pi) % (2 * np.pi) - np.pi)

    initial_orientation, final_orientation = angle(goal[4], start[4]), angle(goal[4], final[4])
    position = np.clip(1 - final_position / max(initial_position, 1e-6), 0, 1)
    orientation = np.clip(1 - final_orientation / max(initial_orientation, 1e-6), 0, 1)
    return float(0.5 * (position + orientation))


class PushEvaluator:
    def __init__(self, design, output, runtime):
        self.design, self.output, self.rt = design, Path(output), runtime
        p = design["planner"]
        if (
            not p["checked"]
            or p["objective"] != "learned"
            or p["proposal_mode"] != "residual_mixture"
        ):
            raise ValueError(
                "Frozen learned task cost, checker and declared proposal mixture required"
            )

    def __call__(self, job, report_progress):
        d, rt, case = self.design, self.rt, job.case
        if source_identities() != rt.sources:
            raise RuntimeError("Frozen evaluator source changed mid-stage")
        if dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=20) >= dt.datetime.fromisoformat(
            d["deadline_utc"]
        ):
            raise TimeoutError("Shared deadline preservation reserve reached")
        expected = case["input_sha256"]
        if digest(ROOT / case["input_path"]) != expected:
            raise RuntimeError("Frozen source pair identity differs")
        with np.load(ROOT / case["input_path"], allow_pickle=False) as data:
            pair = dict(data)
        shape, folder = case["shape"], Path(job.directory) / "episode"
        c = ProgressCounter(report_progress)
        c["native_environment_constructions"] += 1
        c.publish()
        controller = (
            None
            if job.method == "native"
            else PushContextAdapter(
                rt.spec, job.parameters, shape, rt.contract.mean, rt.contract.scale
            )
        )
        protocol = {
            "planner": d["planner"],
            "replan_blocks": d["planner"]["replan_blocks"],
            "seed": job.environment_seed,
            "margin_px": d["margin_px"],
            "max_episode_actions": min(job.max_actions, d["maximum_episode_actions"]),
            "max_runtime_minutes": 5,
            "verify_analytic_dynamics": False,
        }

        class AdmittedChecker(LearnedMotionChecker):
            def observe(self, observed):
                if self.observed is None:
                    point = torch.as_tensor(np.asarray(observed)[:2][None], dtype=torch.float64)
                    threshold = d["margin_px"] + float(rt.head.error_allowance)
                    if (
                        min(
                            float(physical_clearance(point, shape).min()),
                            float(arena_clearance(point).min()),
                        )
                        < threshold
                    ):
                        raise RuntimeError("Settled initial pusher violates fixed safety admission")
                return super().observe(observed)

        def checker_factory(hazard, margin, scaler, counters, enabled):
            if not enabled:
                raise RuntimeError("The learned checker remains enabled")
            return AdmittedChecker(rt.head, pusht_constraints(hazard), margin, counters)

        def forbidden(*args, **kwargs):
            raise RuntimeError("Analytic/reference prediction remains forbidden")

        started = time.perf_counter()
        try:
            with ExitStack() as guard:
                for target in (
                    "scripts.run_lewm_push_diagnostic.pd_response",
                    "helpers.safecem10_planner.pd_response",
                    "helpers.pusht_smoke.ShapeChecker.check",
                    "helpers.pusht_smoke.ShapeChecker.paths",
                ):
                    guard.enter_context(patch(target, side_effect=forbidden))
                row, _ = run_episode(
                    "native" if controller is None else "structured",
                    shape,
                    AccountedModel(rt.model, c),
                    rt.process,
                    pair,
                    protocol,
                    folder,
                    time.time(),
                    checker_factory=checker_factory,
                    action_contract=rt.contract,
                    project_candidates=True,
                    clip_execution=True,
                    proposal_adapter=controller,
                    planner_type=CausalPushPlanningCEM,
                    device="cuda",
                    resource_counters=c,
                )
            with np.load(folder / "trace.npz", allow_pickle=False) as data:
                flags = [
                    i for i, frame in enumerate(data["frames"]) if image_stats(frame)["flagged"]
                ]
            row.update(
                job_id=job.job_id,
                case_id=job.case_id,
                method=job.method,
                environment_seed=job.environment_seed,
                policy_seed=job.policy_seed,
                frame_flags=flags,
                case_sha256=expected,
                action_contract=rt.contract.record(),
            )
            if flags and not row["error"]:
                row["error"] = (
                    "Potential image corruption flag requires review; this flag is not a validated classifier"
                )
            if c.get("predictor_steps", 0) > d["maximum_predictor_steps_per_episode"]:
                row["error"] = "Frozen per-episode predictor ceiling exceeded"
            row["progress"] = task_progress(row)
            write_json(folder / "result.json", row)
            error = row["error"]
            outcome = EpisodeOutcome(
                safe_completion=bool(row["safe_completion"] and not error),
                violation=bool(row["margin_breach"] or row["arena_margin_breach"]),
                unfinished=bool(row["unfinished"]),
                engaged=bool(row["engaged"]),
                progress=row["progress"],
                environment_actions=int(c["environment_actions"]),
                resources=dict(c),
                error=error,
                extra={
                    "episode_result": str(folder / "result.json"),
                    "episode_result_sha256": digest(folder / "result.json"),
                    "controller": None if controller is None else str(folder / "controller.json"),
                    "controller_failure": row["controller_failure"],
                    "frame_flags": flags,
                    "original_task": "PushT joint-position-and-orientation completion, unchanged",
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
        except BaseException:
            error = traceback.format_exc()
            folder.mkdir(parents=True, exist_ok=True)
            write_json(
                folder / "evaluator-error.json",
                {
                    "error": error,
                    "resources": dict(c),
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
            outcome = EpisodeOutcome(
                False,
                False,
                True,
                False,
                0.0,
                int(c.get("environment_actions", 0)),
                resources=dict(c),
                error=error,
                extra={"failure_trace": str(folder / "evaluator-error.json")},
            )
        c.publish()
        return outcome


def resolve_controllers(protocol):
    resolved = {}
    for label, descriptor in protocol["controllers"].items():
        value = dict(descriptor)
        if descriptor["kind"] == "nominee":
            path = ROOT / descriptor["source"]
            state_path = path.parent / "state.json"
            state = json.loads(state_path.read_text())
            if (
                state["phase"] != "complete"
                or state["candidate_panels_completed"] != descriptor["expected_candidate_panels"]
            ):
                raise RuntimeError("The frozen optimizer run did not finish its candidate panels")
            if state["optimizer_updates"] != descriptor["expected_optimizer_updates"]:
                raise RuntimeError("The expected real optimizer updates did not finish")
            candidate = json.loads(path.read_text())[descriptor["field"]]
            if candidate is None or candidate["parameters_equal_initial"]:
                raise RuntimeError(
                    "A trained candidate cannot be silently replaced by the zero anchor"
                )
            if (
                candidate != state["best_trained"]
                or candidate["provenance"]["method"] != descriptor["method"]
            ):
                raise RuntimeError("Optimizer nominee identity/provenance differs")
            value.update(
                kind="structured",
                parameters=candidate["parameters"],
                nominee=candidate,
                nominee_sha256=digest(path),
                state_sha256=digest(state_path),
            )
        resolved[label] = value
    return resolved


def verify_equivalence(out, state, pairs):
    keys = list(state["jobs"])
    records = []
    for native_index, zero_index in pairs:
        native_key, zero_key = keys[native_index], keys[zero_index]
        traces = []
        for key in (native_key, zero_key):
            with np.load(out / "jobs" / key / "episode/trace.npz", allow_pickle=False) as data:
                traces.append(
                    {
                        name: data[name]
                        for name in (
                            "actions",
                            "proposed",
                            "states",
                            "dense",
                            "frames",
                            "goal_image",
                        )
                    }
                )
        for field in traces[0]:
            np.testing.assert_array_equal(traces[0][field], traces[1][field])
        for field in (
            "predictor_steps",
            "encoder_frames",
            "model_batch_calls",
            "model_candidate_rollouts",
            "actual_cost_call_attempts",
            "actual_candidate_block_steps_attempted",
        ):
            if state["jobs"][native_key]["resources"].get(field, 0) != state["jobs"][zero_key][
                "resources"
            ].get(field, 0):
                raise RuntimeError("Zero anchor changed native model work")
        records.append(
            {
                "native": native_key,
                "zero": zero_key,
                "actions_per_episode": len(traces[0]["actions"]),
                "verified": True,
            }
        )
    write_json(
        out / "equivalence.json",
        {
            "verified": True,
            "pairs": records,
            "new_simulator_interactions": 0,
            "new_frozen_model_queries": 0,
        },
    )


def run_evaluation(protocol, output, runtime):
    out = Path(output)
    if out.exists():
        raise FileExistsError("Never overwrite an evaluation stage")
    out.mkdir(parents=True)
    (out / "jobs").mkdir()
    controllers = resolve_controllers(protocol)
    write_json(
        out / "source-freeze.json",
        {
            "protocol": protocol,
            "sources": runtime.sources,
            "resolved_controllers": controllers,
            "pod_id": protocol["evaluation"]["execution_pod_id"],
        },
    )
    evaluator = PushEvaluator(protocol["evaluation"], out, runtime)
    state = {
        "phase": "running",
        "jobs": {},
        "active": None,
        "resources": {},
        "environment_actions": 0,
    }
    write_json(out / "state.json", state)
    try:
        for index, spec in enumerate(protocol["evaluation_jobs"]):
            if len(state["jobs"]) >= protocol["max_episode_starts"]:
                raise RuntimeError("Frozen evaluation episode-start limit reached")
            maximum = min(
                spec.get("max_actions", protocol["max_episode_actions"]),
                protocol["max_episode_actions"],
            )
            if state["environment_actions"] + maximum > protocol["max_environment_actions"]:
                raise RuntimeError("No full evaluation allowance remains")
            descriptor = controllers[spec["controller"]]
            case = spec["case"]
            parameters = descriptor.get("parameters", runtime.spec.initial_parameters().tolist())
            key = f"{index:03}-{spec['controller']}-{case['case_id']}-s{spec['seed']}"
            folder = out / "jobs" / key
            folder.mkdir()
            job = JobSpec(
                key,
                folder,
                case["case_id"],
                case,
                spec["seed"],
                spec["seed"],
                tuple(parameters),
                "deterministic",
                maximum,
                1.0,
                "native" if descriptor["kind"] == "native" else spec["controller"],
                0,
                "evaluation",
                -1,
                index,
            )
            write_json(
                folder / "started.json",
                {
                    "key": key,
                    "controller": descriptor,
                    "case": case,
                    "seed": spec["seed"],
                    "max_actions": maximum,
                },
            )
            state["active"] = key
            write_json(out / "state.json", state)

            def report(actions, resources):
                write_json(
                    folder / "live.json", {"environment_actions": actions, "resources": resources}
                )

            outcome = evaluator(job, report)
            row = vars(outcome).copy()
            row.update(
                key=key, case_id=case["case_id"], controller=spec["controller"], seed=spec["seed"]
            )
            write_json(folder / "result.json", row)
            state["jobs"][key] = row
            state["environment_actions"] += outcome.environment_actions
            for name, value in outcome.resources.items():
                state["resources"][name] = state["resources"].get(name, 0) + value
            state["active"] = None
            write_json(out / "state.json", state)
            if outcome.error:
                raise RuntimeError("Retained technical evaluation failure: " + key)
            print(
                json.dumps(
                    {
                        "key": key,
                        "safe_completion": outcome.safe_completion,
                        "actions": outcome.environment_actions,
                    }
                ),
                flush=True,
            )
        if protocol.get("equivalence_pairs"):
            verify_equivalence(out, state, protocol["equivalence_pairs"])
        state["phase"] = "complete"
        state["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    except BaseException:
        state["phase"] = "technical_failure"
        state["error"] = traceback.format_exc()
        raise
    finally:
        write_json(out / "state.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    runtime = load_push_runtime(protocol["evaluation"])

    def stop(signum, frame):
        raise InterruptedError("Shared campaign stop requested")

    signal.signal(signal.SIGTERM, stop)
    if args.mode == "evaluate":
        run_evaluation(protocol, args.out, runtime)
    else:
        if protocol["method"] not in ("cma", "random"):
            raise RuntimeError(
                "This deterministic24-gene pilot does not implement ordinary learning"
            )
        if len(protocol["initial_parameters"]) != runtime.spec.parameter_count:
            raise RuntimeError("Genotype dimension differs from the frozen controller")
        CampaignTraining(
            protocol,
            args.out,
            PushEvaluator(protocol["evaluation"], args.out, runtime),
            resume=args.resume,
        ).run()


if __name__ == "__main__":
    main()
