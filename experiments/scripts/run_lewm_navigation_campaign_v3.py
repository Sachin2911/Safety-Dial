"""NAV v3: explicit current-target decoded scoring and final controller evidence.

Copied from frozen NAV v2; the ten-gene actor, safety head, CEM and action path
are unchanged. Per-job prospective metadata chooses original or decoded task cost.
"""

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import argparse
import collections
from dataclasses import fields
import datetime as dt
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm
from stable_worldmodel.policy import AutoCostModel
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm16_planner import NoSafeAction
from helpers.lewm_navigation_evolution import ActionContract, ProjectedCostModel
from helpers.lewm_navigation_decoded_cost import (
    DECODED_BACKEND,
    ORIGINAL_BACKEND,
    DecodedCurrentTargetCost,
)
from helpers.lewm_navigation_retrieval import (
    RetrievalController,
    RetrievalMemory,
    RetrievalPlanningCEM,
)
from helpers.lewm_progress_journal import BACKEND, ProgressJournal, recover_progress_journals
from helpers.lewm_proposal_pilot import write_json
from helpers.lewm_screen import image_stats
from helpers.portable_env import NativeEnv
from helpers.portable_motion import MotionChecker, clearance, load_model, shape_constraints
from helpers.pusht_smoke import Scaler
from helpers.safecem10_planner import IMAGENET_MEAN, IMAGENET_STD, image_transform


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class ProgressCounter(collections.Counter):
    """Persist explicit simulator calls when charged, including unsuccessful starts."""

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
    """Record actual frozen-model query attempts as well as the native planner ledger."""

    def __init__(self, frozen, counters):
        super().__init__()
        self.frozen, self.counters = frozen, counters

    def encode(self, info):
        c = self.counters
        c["actual_encoder_call_attempts"] += 1
        pixels = info["pixels"]
        c["actual_encoder_frames_attempted"] += int(pixels.shape[0] * pixels.shape[1])
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


class AccountedSafetyHead(torch.nn.Module):
    """Count each actual safety-head forward before execution, without changing inputs."""

    def __init__(self, frozen, counters):
        super().__init__()
        self.frozen, self.counters = frozen, counters
        self.spec, self.calibration = frozen.spec, frozen.calibration

    def forward(self, initial, actions):
        c = self.counters
        candidates = int(actions.shape[0])
        rows = int(candidates * actions.shape[-2])
        c["actual_safety_call_attempts"] += 1
        c["actual_safety_candidate_rollouts_attempted"] += candidates
        c["actual_safety_transition_rows_attempted"] += rows
        c.publish()
        try:
            result = self.frozen(initial, actions)
            c["actual_safety_call_successes"] += 1
            c["actual_safety_candidate_rollouts_successful"] += candidates
            c["actual_safety_transition_rows_successful"] += rows
            return result
        except BaseException:
            c["actual_safety_call_failures"] += 1
            raise
        finally:
            c.publish()


class ProgressCostModel(ProjectedCostModel):
    def get_cost(self, info, candidates):
        try:
            return super().get_cost(info, candidates)
        finally:
            self.counters.publish()


class AccountedGoalReadout:
    """Preserve the frozen FP64 map while journaling each actual readout attempt."""

    def __init__(self, frozen, counters):
        self.frozen, self.counters = frozen, counters

    def __getattr__(self, name):
        return getattr(self.frozen, name)

    def __call__(self, embedding):
        c = self.counters
        c["goal_readout_queries"] += 1
        c["goal_readout_call_attempts"] += 1
        c.publish()
        try:
            result = self.frozen(embedding)
            c["goal_readout_call_successes"] += 1
            return result
        except BaseException:
            c["goal_readout_call_failures"] += 1
            raise
        finally:
            c.publish()


def controller_type_for_design(design):
    """Resolve an explicit frozen family; the v1 path exists for persistence parity."""
    backend = design.get("controller_backend")
    if backend == "observed-target-v1":
        return RetrievalController
    if backend == "observed-target-v2":
        from helpers.lewm_navigation_retrieval_v2 import RetrievalController as V2Controller

        return V2Controller
    raise ValueError("Declare an observed-target-v1 or observed-target-v2 controller backend")


def load_controller_readout(design, *, root=ROOT):
    declaration = design.get("goal_readout")
    if declaration is None:
        config = design.get("controller_config", {})
        if config.get("readout_progress", False) or config.get("direct_goal_locality", False):
            raise ValueError("Readout progress requires a prospectively pinned readout asset")
        return None
    if design["controller_backend"] != "observed-target-v2":
        raise ValueError("Persistence-parity v1 may not receive a new goal-readout feature")
    from helpers.lewm_navigation_retrieval_v2 import FrozenGoalReadout

    path = root / declaration["path"]
    if digest(path) != declaration["sha256"]:
        raise RuntimeError("Frozen visual-goal readout asset changed")
    readout = FrozenGoalReadout.from_npz(path, device="cuda")
    if readout.fingerprint != declaration["sha256"]:
        raise RuntimeError("Loaded goal-readout identity differs from its freeze")
    return readout


def make_navigation_controller(design, runtime, job, shape, counters):
    controller_type = controller_type_for_design(design)
    common = {
        "body_radius": design["body_radius"],
        "margin": design["margin"],
        "allowance": runtime.head.calibration["allowance"],
        "stochastic": job.mode == "categorical",
        "temperature": job.temperature,
        "generator": torch.Generator(device="cuda").manual_seed(job.policy_seed),
    }
    if design["controller_backend"] == "observed-target-v2":
        common.update(
            goal_readout=(
                AccountedGoalReadout(runtime.goal_readout, counters)
                if runtime.goal_readout is not None
                else None
            ),
            controller_config=design["controller_config"],
        )
    return controller_type(runtime.memory, job.parameters, shape, **common)


def load_navigation_runtime(design, *, root=ROOT):
    """Load fixed assets only after validating this new campaign's declared host."""
    if os.environ.get("RUNPOD_POD_ID") != design["execution_pod_id"]:
        raise RuntimeError("Only the declared rented navigation pod may load this runtime")
    controller_type_for_design(design)
    for name, expected in design["asset_sha256"].items():
        if digest(root / name) != expected:
            raise RuntimeError("Frozen runtime asset changed: " + name)
    memory_path = root / design["memory_path"]
    if digest(memory_path) != design["memory_sha256"]:
        raise RuntimeError("Observed target memory changed")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.is_autocast_enabled():
        raise RuntimeError("Live frozen encoding must use the bank's FP32 contract")
    preprocessing = {
        "function_source": inspect.getsource(image_transform),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "input": "uint8 RGB224",
        "layout": "B,T,C,H,W",
        "encoding": "FP32 without autocast or TF32; exact frozen AutoCostModel.encode last frame",
    }
    preprocessing_sha = hashlib.sha256(
        json.dumps(preprocessing, sort_keys=True).encode()
    ).hexdigest()
    metadata_path = root / design.get(
        "memory_metadata_path", str(memory_path.parent / "memory-metadata.json")
    )
    metadata = json.loads(metadata_path.read_text())
    if metadata["preprocessing_sha256"] != preprocessing_sha:
        raise RuntimeError("Live image preprocessing differs from frozen observed memory")
    checkpoint = root / design["world_model"]
    if metadata["frozen_world_model_sha256"] != digest(checkpoint):
        raise RuntimeError("Observed memory was encoded by a different frozen world model")
    model = AutoCostModel(str(checkpoint)[: -len("_object.ckpt")], cache_dir=checkpoint.parent)
    model = model.cuda().eval().requires_grad_(False)
    head = load_model(root / design["motion_model"])
    if not head.calibration["ready"]:
        raise RuntimeError("Frozen learned motion model is not released")
    if head.spec["environment"] != "tworoom" or head.spec["action_dim"] != 2:
        raise RuntimeError("Motion model is not the declared native TwoRoom model")
    if abs(head.spec["body_radius"] - design["body_radius"]) > 1e-9:
        raise RuntimeError("Protected body differs from the frozen head")
    action = json.loads((root / design["scalers"]).read_text())["columns"]["action"]
    normal = Scaler(action["mean"], action["scale"])
    contract = ActionContract(
        tuple(action["mean"]), tuple(action["scale"]), (-1.0, -1.0), (1.0, 1.0)
    )
    memory = RetrievalMemory.from_npz(memory_path, device="cuda")
    if memory.fingerprint != design["memory_sha256"]:
        raise RuntimeError("Loaded memory identity differs")
    precision = {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "autocast_enabled": bool(torch.is_autocast_enabled()),
        "world_model_parameter_dtype": str(next(model.parameters()).dtype),
        "preprocessing_sha256": preprocessing_sha,
        "memory_metadata_sha256": digest(metadata_path),
    }
    if next(model.parameters()).dtype != torch.float32:
        raise RuntimeError("Frozen world model parameters must match FP32 memory encoding")
    return SimpleNamespace(
        model=model,
        head=head,
        normal=normal,
        contract=contract,
        memory=memory,
        precision=precision,
        goal_readout=load_controller_readout(design, root=root),
    )


class NavigationEvaluator:
    """Bridge a frozen case and persistent genotype to the common optimizer harness."""

    def __init__(self, design, output, runtime, *, root=ROOT):
        self.design, self.output, self.rt, self.root = design, Path(output), runtime, root
        p = design["planner"]
        if not p["checked"] or p["objective"] != "learned":
            raise ValueError("Frozen learned task cost and safety checking must remain active")

    def __call__(self, job, report_progress):
        return self.evaluate(job, report_progress)

    def evaluate(self, job, report_progress):
        if self.design.get("progress_backend") != BACKEND:
            raise ValueError("The new persistence backend must be prospectively declared")
        pins = {
            "navigation_v3_sha256": digest(Path(__file__)),
            "navigation_v2_parent_sha256": digest(
                ROOT / "experiments/scripts/run_lewm_navigation_campaign_v2.py"
            ),
            "decoded_task_cost_sha256": digest(
                ROOT / "experiments/helpers/lewm_navigation_decoded_cost.py"
            ),
            "legacy_navigation_v1_sha256": digest(
                ROOT / "experiments/scripts/run_lewm_navigation_campaign.py"
            ),
            "shared_harness_sha256": digest(ROOT / "experiments/helpers/lewm_campaign_training.py"),
            "controller_source_sha256": digest(
                Path(inspect.getfile(controller_type_for_design(self.design)))
            ),
        }
        journal = ProgressJournal(
            Path(job.directory) / "progress-journal.jsonl",
            report_progress,
            max_actions=job.max_actions,
            metadata={
                "job_id": job.job_id,
                "started_sha256": digest(Path(job.directory) / "started.json"),
                "case_id": job.case_id,
                "case_sha256": job.case.get("input_sha256", job.case.get("sha256")),
                "environment_seed": job.environment_seed,
                "policy_seed": job.policy_seed,
                "source_pins": pins,
            },
        )
        outcome = None
        try:
            outcome = self._evaluate(job, journal)
        except BaseException:
            # Preserve the original failure even if a final snapshot also fails.
            try:
                journal.finish(status="error")
            except BaseException:
                pass
            raise
        else:
            journal.finish(status="error" if outcome.error else "complete")
        finally:
            journal.close()
        provenance = {
            **journal.summary(),
            "sha256": digest(journal.path),
            "source_pins": pins,
            "journal_is_authoritative_for_attempts": True,
        }
        outcome.extra["controller_backend"] = self.design["controller_backend"]
        outcome.extra["goal_readout"] = self.design.get("goal_readout")
        outcome.extra["progress_backend"] = BACKEND
        outcome.extra["progress_journal"] = provenance
        result_path = Path(outcome.extra["episode_result"])
        row = json.loads(result_path.read_text())
        row["controller_backend"] = self.design["controller_backend"]
        row["goal_readout"] = self.design.get("goal_readout")
        row["progress_backend"] = BACKEND
        row["progress_journal"] = provenance
        write_json(result_path, row)
        outcome.extra["episode_result_sha256"] = digest(result_path)
        return outcome

    def _evaluate(self, job, report_progress):
        from helpers.lewm_campaign_training import EpisodeOutcome

        d, rt, case = self.design, self.rt, job.case
        folder = Path(job.directory) / "episode"
        folder.mkdir(parents=True, exist_ok=False)
        c = ProgressCounter(report_progress)
        case_path = self.root / case.get("input_path", case.get("pair_path", case.get("path", "")))
        expected = case.get("input_sha256", case.get("sha256"))
        source_episode = case.get("source_episode")
        if source_episode is not None and int(source_episode) in set(
            rt.memory.metadata["source_episode"].tolist()
        ):
            raise RuntimeError("A live query source overlaps the observed memory")
        allowed_sources = d.get("allowed_source_episodes")
        if allowed_sources is not None and source_episode not in allowed_sources:
            raise RuntimeError("Query source is not authorized for this frozen stage")
        if not expected or digest(case_path) != expected:
            raise RuntimeError("Case identity differs from the prospective panel")
        with np.load(case_path, allow_pickle=False) as data:
            pair = dict(data)
        shape = case.get("shape", {"kind": "none"})
        constraints = shape_constraints(shape)
        controller = None
        if job.method not in ("native", "native_projected", "decoded_native"):
            controller = make_navigation_controller(d, rt, job, shape, c)
            write_json(folder / "controller.json", controller.artifact())
        task_cost_backend = case.get("task_cost_backend", d.get("task_cost_backend"))
        if task_cost_backend not in (ORIGINAL_BACKEND, DECODED_BACKEND):
            raise ValueError("Freeze an explicit original or decoded current-target task metric")
        if task_cost_backend == DECODED_BACKEND and rt.goal_readout is None:
            raise ValueError("Decoded scoring requires the prospectively pinned shared readout")
        maximum = min(job.max_actions, d["maximum_episode_actions"])
        if maximum < 1:
            raise RuntimeError("No authorized environment interaction remains")
        c["native_environment_constructions"] += 1
        c.publish()
        env = None
        states, actions, raw_commands, frames, dense, frame_checks = [], [], [], [], [], []
        success, failure, error = False, None, None
        cost = solver = decoded_cost = None
        initial_distance = final_distance = None
        planning = 0.0
        started = time.perf_counter()

        def capture():
            frame = env.render()
            try:
                stats = image_stats(frame)
            except Exception:
                np.save(folder / f"invalid-frame-{len(frames):04}.npy", frame, allow_pickle=False)
                raise
            frames.append(frame)
            frame_checks.append(stats)
            return frame

        try:
            torch.manual_seed(job.environment_seed)
            np.random.seed(job.environment_seed)
            env = NativeEnv("tworoom", c, folder / "interactions.jsonl")
            if not np.array_equal(
                env.env.action_space.low, rt.contract.native_low
            ) or not np.array_equal(env.env.action_space.high, rt.contract.native_high):
                raise RuntimeError("Native action domain differs from the frozen contract")
            observation = env.reset(
                job.environment_seed, pair["initial"], pair["goal"], pair["goal_image"]
            )
            c.publish()
            np.testing.assert_allclose(observation, pair["initial"], atol=1e-5, rtol=0)
            initial_distance = float(np.linalg.norm(observation - pair["goal"]))
            final_distance = initial_distance
            if env.success() or initial_distance < 16:
                raise RuntimeError("Already successful original-task start")
            admission = clearance(
                torch.as_tensor(np.stack([observation, pair["goal"]])),
                constraints,
                d["body_radius"],
            )
            if bool((admission < d["margin"] + rt.head.calibration["allowance"]).any()):
                raise RuntimeError("Frozen source or final goal fails geometric safety admission")
            image_stats(pair["goal_image"])
            checker = MotionChecker(AccountedSafetyHead(rt.head, c), constraints, d["margin"], c)
            checker.observe(observation)
            model = AccountedModel(rt.model, c)
            if task_cost_backend == DECODED_BACKEND:
                decoded_cost = DecodedCurrentTargetCost(
                    model, AccountedGoalReadout(rt.goal_readout, c), c
                )
                model = decoded_cost
            cost = ProgressCostModel(model, checker, d["planner"], c, rt.contract)
            solver = RetrievalPlanningCEM(
                cost, d["planner"], job.environment_seed, "cuda", controller
            )
            policy = swm.policy.WorldModelPolicy(
                solver,
                swm.PlanConfig(
                    horizon=5, receding_horizon=d["planner"]["replan_blocks"], action_block=5
                ),
                process={"action": rt.normal},
                transform={"pixels": image_transform(), "goal": image_transform()},
            )
            policy.set_env(
                SimpleNamespace(
                    num_envs=1,
                    single_action_space=env.env.action_space,
                    action_space=gym.spaces.Box(-1, 1, (1, 2), dtype=np.float64),
                )
            )
            states.append(observation.copy())
            dense.append(observation.copy())
            previous = np.zeros(2)
            for _ in range(maximum):
                if dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(d["deadline_utc"]):
                    raise TimeoutError("Global seven-hour deadline reached")
                frame = capture()
                before = time.perf_counter()
                try:
                    command = policy.get_action(
                        {
                            "pixels": frame[None, None],
                            "goal": pair["goal_image"][None, None],
                            "action": previous[None, None],
                        }
                    )[0]
                finally:
                    torch.cuda.synchronize()
                    planning += time.perf_counter() - before
                    c.publish()
                actual = rt.contract.execute(command)
                observation, path = env.step(actual)
                c.publish()
                checker.observe(observation)
                states.append(observation.copy())
                dense.extend(path)
                actions.append(actual.copy())
                raw_commands.append(np.asarray(command).copy())
                previous = actual
                final_distance = float(np.linalg.norm(observation - pair["goal"]))
                success = env.success()
                if success:
                    break
                if time.perf_counter() - started > d.get("max_episode_seconds", 300):
                    raise TimeoutError("Frozen episode wall-time limit")
                if c.get("predictor_steps", 0) > d["maximum_predictor_steps_per_episode"]:
                    raise RuntimeError("Frozen per-episode model-work limit exceeded")
        except NoSafeAction as exc:
            failure = str(exc)
        except Exception:
            error = traceback.format_exc()
        finally:
            if env is not None:
                try:
                    capture()
                except Exception:
                    if error is None:
                        error = traceback.format_exc()
                finally:
                    try:
                        env.close()
                    except Exception:
                        if error is None:
                            error = traceback.format_exc()
            try:
                persist_episode_sidecars(folder, controller, decoded_cost)
            except Exception:
                if error is None:
                    error = traceback.format_exc()
            c.publish()
        elapsed = time.perf_counter() - started
        path_array = np.asarray(dense, dtype=np.float32).reshape(-1, 2)
        clear = clearance(torch.as_tensor(path_array), constraints, d["body_radius"]).numpy()
        unsafe = bool((clear < d["margin"] - 1e-6).any())
        contact = bool((clear <= 0).any())
        state_array = np.asarray(states, dtype=np.float32).reshape(-1, 2)
        pathlength = float(np.linalg.norm(np.diff(state_array, axis=0), axis=1).sum())
        engaged = bool(len(actions) and pathlength > 1.0)
        safe = bool(success and not unsafe and not error and not failure)
        progress = (
            float(np.clip(1 - final_distance / initial_distance, 0, 1))
            if initial_distance and final_distance is not None
            else 0.0
        )
        queries = controller.history if controller is not None else []
        write_json(folder / "selector-history.json", queries)
        write_json(folder / "replans.json", [] if solver is None else solver.replans)
        write_json(folder / "cost-history.json", [] if cost is None else cost.history)
        write_json(folder / "projection.json", [] if cost is None else cost.projection_history)
        write_json(folder / "frame-checks.json", frame_checks)
        np.savez_compressed(
            folder / "trace.npz",
            states=state_array,
            dense=path_array,
            actions=np.asarray(actions, dtype=np.float64).reshape(-1, 2),
            proposed=np.asarray(raw_commands, dtype=np.float64).reshape(-1, 2),
            frames=np.stack(frames) if frames else np.empty((0, 224, 224, 3), dtype=np.uint8),
            goal_image=pair["goal_image"],
            original_goal=pair["goal"],
        )
        row = {
            "job_id": job.job_id,
            "method": job.method,
            "mode": job.mode,
            "case_id": job.case_id,
            "environment_seed": job.environment_seed,
            "policy_seed": job.policy_seed,
            "maximum_actions": maximum,
            "safe_completion": safe,
            "success": bool(success),
            "violation": unsafe,
            "hazard_violation": contact,
            "unfinished": not bool(success),
            "engaged": engaged,
            "progress": progress,
            "environment_actions": int(c.get("environment_actions", 0)),
            "recorded_successful_actions": len(actions),
            "resources": dict(c),
            "protected_body_path_length": pathlength,
            "initial_original_goal_distance": initial_distance,
            "final_original_goal_distance": final_distance,
            "original_goal": pair["goal"].tolist(),
            "shape": shape,
            "task_cost_backend": task_cost_backend,
            "minimum_clearance": float(clear.min()) if len(clear) else None,
            "planning_seconds": planning,
            "episode_seconds": elapsed,
            "error": error,
            "controller_failure": failure,
            "frame_flags": [i for i, value in enumerate(frame_checks) if value["flagged"]],
            "case_sha256": expected,
            "source_provenance": {
                k: case[k]
                for k in (
                    "source_episode",
                    "source_step",
                    "goal_step",
                    "source_global_row",
                    "goal_global_row",
                )
                if k in case
            },
            "memory_sha256": rt.memory.fingerprint,
            "candidate_projection": True,
            "precision": rt.precision,
            "action_contract": rt.contract.record(),
        }
        write_json(folder / "result.json", row)
        extras = {
            "episode_result": str(folder / "result.json"),
            "episode_result_sha256": digest(folder / "result.json"),
            "decision_reconstruction": str(folder / "selector-history.json"),
            "decision_reconstruction_sha256": digest(folder / "selector-history.json"),
            "frame_flags": row["frame_flags"],
            "controller_failure": failure,
            "initial_original_goal_distance": initial_distance,
            "final_original_goal_distance": final_distance,
            "task_cost_backend": task_cost_backend,
            "final_sidecars": {
                name: {"path": str(folder / name), "sha256": digest(folder / name)}
                for name in (
                    "controller-final.json",
                    "refusal-history.json",
                    "decoded-task-cost.json",
                )
                if (folder / name).exists()
            },
        }
        decisions = controller.training_decisions if controller is not None else []
        return EpisodeOutcome(
            safe_completion=safe,
            violation=unsafe,
            unfinished=not bool(success),
            engaged=engaged,
            progress=progress,
            environment_actions=int(c.get("environment_actions", 0)),
            resources=dict(c),
            decisions=decisions,
            error=error,
            extra=extras,
        )


def persist_episode_sidecars(folder, controller, decoded_cost):
    """Keep initial controller bytes intact and retain final refusal/projection proof."""
    if controller is not None:
        write_json(folder / "controller-final.json", controller.artifact())
        write_json(folder / "refusal-history.json", getattr(controller, "refusals", []))
    if decoded_cost is not None:
        write_json(folder / "decoded-task-cost.json", decoded_cost.artifact())


def verify_case_preparation(protocol):
    """Require a hash-bound native/dataset visual review before controller episodes."""
    preparation = protocol["case_preparation"]
    panel_path = ROOT / preparation["panel_path"]
    expected = preparation["panel_sha256"]
    if digest(panel_path) != expected:
        raise RuntimeError("Frozen case panel differs from the reviewed preparation")
    review_path = ROOT / preparation["review_path"]
    review = json.loads(review_path.read_text())
    if (
        not review.get("approved")
        or not review.get("images_clean")
        or review.get("panel_sha256") != expected
    ):
        raise RuntimeError(
            "Native/dataset input gallery review is required before controller episodes"
        )
    if not review.get("reviewed_utc") or not review.get("galleries"):
        raise RuntimeError("Visual review must identify its time and inspected gallery hashes")
    for name, sha in review["galleries"].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError("Inspected input gallery changed")
    return {
        "panel_sha256": expected,
        "review_sha256": digest(review_path),
        "review_path": str(review_path),
    }


def run_comparison(protocol, out, protocol_path, *, runtime=None):
    """Evaluate prospectively enumerated native/fixed/learned nominees without training."""
    from helpers.lewm_campaign_training import EpisodeOutcome, JobSpec

    design = protocol["evaluation"]
    if os.environ.get("RUNPOD_POD_ID") != design["execution_pod_id"]:
        raise RuntimeError("Only the declared rented pod may run the comparison")
    actual_sources = {name: digest(ROOT / name) for name in protocol["source_files"]}
    if actual_sources != protocol["source_sha256"]:
        raise RuntimeError("Comparison source differs from its prospective freeze")
    review = verify_case_preparation(protocol)
    if out.exists():
        raise FileExistsError("Use a new comparison namespace; no silent outcome replay")
    jobs = protocol["jobs"]
    if len(jobs) > protocol["maximum_episode_starts"]:
        raise ValueError("Comparison schedule exceeds its frozen episode cap")
    ids = [job["job_id"] for job in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate comparison job identity")
    out.mkdir(parents=True)
    (out / "jobs").mkdir()
    write_json(
        out / "source-freeze.json",
        {
            "protocol_sha256": digest(protocol_path),
            "sources": actual_sources,
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "pod_id": design["execution_pod_id"],
        },
    )
    state = {
        "phase": "running",
        "jobs": {},
        "active": None,
        "resource_totals": {},
        "case_preparation_review": review,
    }
    write_json(out / "state.json", state)
    runtime = load_navigation_runtime(design) if runtime is None else runtime
    evaluator = NavigationEvaluator(design, out, runtime)
    actions = 0
    for planned in jobs:
        cap = min(
            planned.get("max_actions", design["maximum_episode_actions"]),
            design["maximum_episode_actions"],
        )
        if actions + cap > protocol["maximum_environment_actions"]:
            raise RuntimeError(
                "Insufficient frozen allowance for another complete comparison episode"
            )
        if dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=15) >= dt.datetime.fromisoformat(
            design["deadline_utc"]
        ):
            raise RuntimeError("Global preservation reserve reached")
        folder = out / "jobs" / planned["job_id"]
        folder.mkdir()
        spec = JobSpec(
            job_id=planned["job_id"],
            directory=folder,
            case_id=planned["case"]["case_id"],
            case=planned["case"],
            environment_seed=planned.get("environment_seed", planned["case"]["seed"]),
            policy_seed=planned.get("policy_seed", planned["case"]["seed"]),
            parameters=tuple(
                planned.get("parameters", controller_type_for_design(design).fixed_ap_parameters())
            ),
            mode="deterministic",
            max_actions=cap,
            temperature=planned.get("temperature", 1.0),
            method=planned["method"],
            repeat_id=planned.get("repeat_id", 0),
            stage="comparison",
            generation=-1,
            candidate_index=-1,
        )
        write_json(folder / "started.json", planned)
        state["active"] = spec.job_id
        write_json(out / "state.json", state)

        def progress(count, resources=None):
            write_json(
                folder / "live.json", {"environment_actions": count, "resources": resources or {}}
            )

        outcome = evaluator(spec, progress)
        row = {
            f.name: getattr(outcome, f.name)
            for f in fields(EpisodeOutcome)
            if f.name != "decisions"
        }
        row.update(
            job_id=spec.job_id,
            case_id=spec.case_id,
            method=spec.method,
            parameters=list(spec.parameters),
            environment_seed=spec.environment_seed,
            policy_seed=spec.policy_seed,
        )
        write_json(folder / "result.json", row)
        state["jobs"][spec.job_id] = row
        state["active"] = None
        actions += row["environment_actions"]
        total = collections.Counter(state["resource_totals"])
        total.update(row["resources"])
        state["resource_totals"] = dict(total)
        write_json(out / "state.json", state)
        if row["error"] or row["extra"]["frame_flags"]:
            state["phase"] = "technical_or_image_review_required"
            write_json(out / "state.json", state)
            raise RuntimeError("Comparison retained a technical/input-quality failure")
    state["phase"] = "complete"
    state["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(out / "state.json", state)
    return state


def main():
    from helpers.lewm_campaign_training import CampaignTraining

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("schema") in ("lewm-navigation-comparison-v1", "lewm-navigation-comparison-v2"):
        if args.resume:
            raise RuntimeError("Comparison replay is not supported; inspect saved evidence")
        run_comparison(protocol, args.out, args.protocol)
        return
    verify_case_preparation(protocol)
    design = protocol["evaluation"]
    if os.environ.get("RUNPOD_POD_ID") != design["execution_pod_id"]:
        raise RuntimeError("Only the declared rented navigation pod may resume this runtime")
    if design.get("progress_backend") != BACKEND:
        raise ValueError("Freeze the new progress backend before any training")
    if args.resume:
        recovery = recover_progress_journals(args.out)
        write_json(args.out / "journal-recovery.json", recovery)
    runtime = load_navigation_runtime(design)
    evaluator = NavigationEvaluator(design, args.out, runtime)
    trainer = CampaignTraining(
        protocol,
        args.out,
        evaluator,
        policy_log_probability=controller_type_for_design(design).log_probability,
        resume=args.resume,
    )
    trainer.run()


if __name__ == "__main__":
    main()
