"""Separate compound runtime; the frozen constant-controller runtime is unchanged."""

import json
import os
import time
import zlib
from pathlib import Path

import numpy as np

from helpers.maniguard_continuation_evidence import check_return, choice_record, encode_bounds
from helpers.maniguard_continuation_handoff import HandoffReference
from helpers.maniguard_continuation_limits import counter_violations, validate_limits
from helpers.maniguard_latent import LatentController, feature_vector, outcome_score
from helpers.maniguard_latent_policy import array_hash
from helpers.maniguard_phase2_outcomes import derive_outcome

# Import reusable definitions without constructing either old global runtime.
_job_path = os.environ.pop("SAFETY_DIAL_MANIGUARD_JOB", None)
try:
    from helpers.maniguard_latent_runtime import LatentStudy
    from helpers.maniguard_runtime import Study
finally:
    if _job_path is not None:
        os.environ["SAFETY_DIAL_MANIGUARD_JOB"] = _job_path


class ContinuationStudy(LatentStudy):
    def __init__(self):
        job = json.loads(Path(os.environ["SAFETY_DIAL_MANIGUARD_JOB"]).read_text())
        if (job["study"] != "continuation-feasibility-v1" or job["family"] != "jar"
                or job["random_policy"] or job["max_steps_override"] is not None):
            raise ValueError("Compound continuation requires its own full Jar protocol")
        with np.load(job["controller"], allow_pickle=False) as data:
            self.reference = HandoffReference(data["weights"], **job["continuation"])
        if self.reference.contract_sha256 != job["continuation_contract_sha256"]:
            raise ValueError("Compound selector contract differs from the episode job")
        self.limits = job["runtime_limits"]
        validate_limits(self.limits)
        self.initialization_finished = False
        self.budget_failures = []
        self.choice = None
        self.previous_policy_rng = None
        super().__init__()
        self.counts.update(native_policy_requests_admitted=0, native_policy_responses_received=0,
                           native_policy_responses_verified=0, initialization_env_step_attempts=0)
        self.write("technical_budget_contract", limits=self.limits)
        self.ledger()

    def ledger(self):
        self.counts["initialization_finished"] = self.initialization_finished
        for reason in counter_violations(self.counts, self.limits):
            if reason not in self.budget_failures:
                self.budget_failures.append(reason)
        self.counts["budget_violation_reasons"] = list(self.budget_failures)
        super().ledger()

    def enforce_budget(self, reason=None):
        if reason is not None and reason not in self.budget_failures:
            self.budget_failures.append(reason)
        self.ledger()
        if self.budget_failures:
            raise RuntimeError("Continuation technical budget exceeded: " + ", ".join(self.budget_failures))

    def install_accounting(self, og):
        super().install_accounting(og)
        original_step = og.Environment.step
        study = self

        def bounded_step(env, *args, **kwargs):
            study.enforce_budget()
            if study.counts["env_step_attempts"] >= study.limits["total_environment_steps"]:
                study.enforce_budget("next_environment_step_would_exceed_limit")
            if not study.initialization_finished:
                if (study.counts["initialization_env_step_attempts"]
                        >= study.limits["initialization_environment_steps"]):
                    study.enforce_budget("next_initialization_step_would_exceed_limit")
                study.counts["initialization_env_step_attempts"] += 1
                study.ledger()
            return original_step(env, *args, **kwargs)

        og.Environment.step = bounded_step

    def observed(self, step, obs, monitor):
        # Use the same single passive snapshot as the original telemetry hook.
        # Do not update or advance the native monitor here.
        state = monitor.recording_state(step) if monitor is not None else None
        self.write("observed", step=step, state=obs["states"], monitor=state)
        if self.reference.privileged:
            if state is None or state.get("valid") is not True:
                raise ValueError("Privileged handoff requires this actual monitor snapshot")
            value = state["observation"]
            self.reference.observe(
                step, closed=value["ap"]["jar_closed"],
                supported=value["ap"]["jar_on_support"], violated=value["doomed"],
            )

    def start(self, scene, obs, env, robot, monitor):
        super().start(scene, obs, env, robot, monitor)
        self.enforce_budget()
        self.counts["initialization_physics_steps"] = self.counts["physics_steps"]
        self.initialization_finished = True
        self.counts["initialization_completed_epoch"] = time.time()
        self.ledger()
        self.expected_policy_seed = zlib.crc32(
            f"{self.job['seed']}:{scene['name']}".encode()
        ) & 0x7FFFFFFF
        self.write("continuation_contract", contract=self.reference.contract,
                   **encode_bounds(robot.action_space.low, robot.action_space.high))

    def before_query(self, step, obs, frame):
        # Encode once, then log the actual selected component, including native.
        self.enforce_budget()
        self.query_step = step
        Study.before_query(self, step, obs, frame)
        try:
            overview, wrist, context = self.encoder.encode_images(
                obs["overview_image"], obs["wrist_images"]
            )
            features = feature_vector(
                frame.joints, frame.fingers, frame.eef_position, overview, wrist,
                self.encoder.text_features, previous_executed=self.adapter.last_executed,
                previous_eef=self.previous_eef,
            )
            decision = self.reference.select(step, features)
            self.choice = choice_record(decision)
            component = self.reference.native if decision["branch"] == "native" else self.controller
            self.policy_payload = {"latent_weights": component.weights, "latent_features": features}
            self.write("latent_features", step=step, features=features,
                       frame={"joints": frame.joints, "fingers": frame.fingers,
                              "eef": frame.eef_position}, context=context,
                       previous_executed=self.adapter.last_executed,
                       previous_eef=self.previous_eef, code=decision["code"],
                       selected_controller_sha256=decision["controller_sha256"])
            self.write("continuation_choice", step=step, choice=self.choice)
            folder = self.output / "query_observations"
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder / f"{step:06d}.npz", state=obs["states"],
                                overview=obs["overview_image"], wrist=obs["wrist_images"],
                                prompt=np.asarray(obs["task_descriptions"]), features=features)
            self.previous_eef = np.asarray(frame.eef_position).copy()
        finally:
            self.counts.update(self.encoder.ledger())
            self.ledger()

    def connect_policy(self, cfg, upstream_connect):
        if self.job["random_policy"] or not cfg.use_openpi_client:
            raise ValueError("Compound continuation requires the pinned native policy service")
        from helpers.maniguard_client import StudyClient

        study = self

        class CompoundClient(StudyClient):
            def infer(self, observation):
                if study.choice is None:
                    raise ValueError("A compound request needs a preceding branch choice")
                features = np.asarray(observation["latent_features"], np.float32)
                weights = LatentController(observation["latent_weights"])
                if (observation["study_episode_id"] != study.job["episode_id"]
                        or observation["episode_seed"] != study.expected_policy_seed
                        or weights.sha256() != study.choice["controller_sha256"]
                        or not np.array_equal(features, study.policy_payload["latent_features"])):
                    raise ValueError("Actual policy payload differs from the selected compound request")
                index = study.counts["native_policy_requests_admitted"]
                if index >= 250 or index != study.counts["native_policy_responses_verified"]:
                    raise ValueError("Compound query budget or single-stream ownership violated")
                request = {"index": index, "episode": study.job["episode_id"],
                           "seed": study.expected_policy_seed,
                           "controller_sha256": weights.sha256(),
                           "features_sha256": array_hash(features)}
                study.counts["native_policy_requests_admitted"] += 1
                study.write("continuation_request", step=study.query_step, request=request)
                study.ledger()  # Admission is durable before inference can fail.
                try:
                    result = super().infer(observation)
                    study.counts["native_policy_responses_received"] += 1
                    study.write("noise_receipt", step=study.query_step,
                                receipt=result["latent_receipt"], raw_actions=result["actions"],
                                raw_action_dtype=str(np.asarray(result["actions"]).dtype))
                    _, study.previous_policy_rng = check_return(
                        result, study.choice, features, episode=study.job["episode_id"],
                        seed=study.expected_policy_seed, index=index,
                        previous_rng=study.previous_policy_rng,
                    )
                    study.counts["native_policy_responses_verified"] += 1
                    return result
                except Exception as exc:
                    study.write("continuation_request_failed", step=study.query_step,
                                index=index, error_type=type(exc).__name__)
                    raise
                finally:
                    study.ledger()

        return CompoundClient(host=cfg.host, port=cfg.port), "openpi"

    def finish(self, result):
        # The original LatentStudy.finish labels one fixed genotype, so derive
        # the unchanged physical outcome separately for this compound contract.
        Study.finish(self, result)
        rows = [json.loads(line) for line in (self.output / "trace.jsonl").read_text().splitlines()]
        observations = []
        for row in rows:
            if row["type"] == "observed":
                monitor = row.get("monitor") or {}
                observations.append(
                    {"step": row["step"], **monitor["observation"]}
                    if monitor.get("valid") is True
                    else {"step": row["step"], "invalid_monitor": True}
                )
        goals = [row for row in rows if row["type"] == "goal"]
        outcome = derive_outcome(self.job["family"], result, observations, goals)
        if outcome["valid"]:
            outcome["score"] = outcome_score(outcome["C_req"], outcome["V"])
        outcome.update(continuation_contract_sha256=self.reference.contract_sha256,
                       closure_controller_sha256=self.controller.sha256(),
                       continuation_state=self.reference.state())
        (self.output / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")


STUDY = ContinuationStudy() if _job_path is not None else None
