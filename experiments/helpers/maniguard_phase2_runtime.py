"""Phase-two observation-limited controller hooks with separate outcome telemetry."""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from helpers.maniguard_phase2_adapter import Phase2Adapter
from helpers.maniguard_phase2_features import FrozenContextEncoder, combine_features, robot_features
from helpers.maniguard_phase2_model import genome_hash, load_model
from helpers.maniguard_runtime import Study


class Phase2Study(Study):
    def __init__(self):
        job = json.loads(Path(os.environ["SAFETY_DIAL_MANIGUARD_JOB"]).read_text())
        genome, spec, _ = load_model(job["model"])
        if spec.input_dim != 64 or spec.hidden_dim != 8:
            raise ValueError("The frozen pilot requires the declared 692-parameter actor")
        self.genome_hash = genome_hash(genome)
        if self.genome_hash != job["actor_sha256"]:
            raise ValueError("Episode actor differs from the declared genome")
        controller = Phase2Adapter(
            genome, stochastic=job["stochastic"], seed=job["sampling_seed"],
            mode=job["mode"], fixed_choices=job.get("fixed_choices"),
        )
        config = SimpleNamespace(kind="phase2", privileged=False,
                                 guard_joint_limits=True, horizon=16)
        self.encoder = None
        self.last_decision = -4
        self.last_decision_eef = None
        self.chunk = None
        self.query_step = None
        self.decisions = []
        self.extra_state = []
        super().__init__(adapter=controller, config=config)
        self.counts.update(encoder_image_calls=0, encoder_images=0, encoder_text_calls=0,
                           encoder_seconds=0.0, encoder_load_seconds=0.0,
                           adapter_decisions=0, adapter_seconds=0.0, encoder_load_attempts=0)
        self.ledger()

    def start(self, scene, obs, env=None, robot=None, monitor=None):
        self.env, self.robot = env, robot
        self.latest_obs = obs
        super().start(scene, obs)
        started = time.perf_counter()
        self.counts["encoder_load_attempts"] += 1
        try:
            self.encoder = FrozenContextEncoder(device=self.job["encoder_device"])
        finally:
            self.counts["encoder_load_seconds"] = time.perf_counter() - started
            self.ledger()
        self.encoder.begin_episode(obs["task_descriptions"])
        self.counts.update(self.encoder.ledger())
        self.write("phase2_features", feature_dim=64, robot_features=40,
                   vision_features=16, text_features=8, encoder=self.encoder.spec,
                   runtime_provenance=self.encoder.runtime_provenance,
                   actor_sha256=self.genome_hash, information_access="observation_only")
        if monitor is not None:
            self.write("observed", step=0, state=obs["states"], monitor=monitor.recording_state(0))
        self._evaluation_state(0)
        self.ledger()

    def _evaluation_state(self, step):
        from helpers.maniguard_phase2_outcomes import evaluation_state
        state = evaluation_state(self.env, self.scene, self.job["family"], step)
        if state:
            self.extra_state.append(state)
            self.write("evaluation_only", **state)

    def before_query(self, step, obs, frame):
        if self.query_step is not None and step - self.query_step < 4:
            if (self.chunk is None or len(self.chunk) >= 4
                    or step - self.query_step < len(self.chunk)):
                raise ValueError("Query interval shorter than the declared minimum")
            self.write("short_chunk_exhaustion", step=step, previous_query=self.query_step,
                       available_actions=len(self.chunk))
        self.query_step = step
        self.latest_obs = obs
        super().before_query(step, obs, frame)

    def after_query(self, step, chunk):
        super().after_query(step, chunk)
        chunk = np.asarray(chunk)
        if chunk.ndim != 2 or chunk.shape[1] != 8 or not len(chunk):
            raise ValueError("Phase two requires nonempty eight-dimensional action chunks")
        self.chunk = chunk[:16].copy()

    def _decide(self, step, frame, proposal):
        if step == self.last_decision:
            return
        if step % 4:
            raise ValueError("A decision was requested outside the fixed four-action schedule")
        overview, wrist, context = self.encoder.encode_images(
            self.latest_obs["overview_image"], self.latest_obs["wrist_images"])
        proposal = np.asarray(proposal).copy()
        proposal[-1] = np.sign(proposal[-1]) if abs(proposal[-1]) > .01 else -1.0
        robot = robot_features(
            frame, proposal, last_executed=self.adapter.last_executed,
            last_proposal=self.adapter.last_proposal, last_eef=self.last_decision_eef,
            since_query=self.adapter.since_query, opening_steps=self.adapter.opening_steps,
            choices=self.adapter.choices,
        )
        features = combine_features(robot, overview, wrist, self.encoder.text_features)
        started = time.perf_counter()
        record = self.adapter.decide(features, step)
        self.counts["adapter_seconds"] += time.perf_counter() - started
        self.counts["adapter_decisions"] += 1
        self.counts.update(self.encoder.ledger())
        record = {**record, "observation_step": step, "context": context,
                  "actor_sha256": self.genome_hash}
        self.decisions.append(record)
        self.write("decision", **record)
        self.last_decision = step
        self.last_decision_eef = np.asarray(frame.eef_position).copy()

    def action(self, step, ci, original, frame):
        if step == 0:
            self._decide(step, frame, original)
        if self.last_decision != step - step % 4:
            raise ValueError("A control action lacks its causal four-action decision")
        return super().action(step, ci, original, frame)

    def should_replan(self, step, frame):
        if step >= self.job["action_horizon"]:
            return False
        if step % 4 == 0:
            index = self.adapter.since_query
            proposal = self.chunk[index] if index < len(self.chunk) else self.adapter.last_proposal
            self._decide(step, frame, proposal)
        return super().should_replan(step, frame)

    def observed(self, step, obs, monitor):
        self.latest_obs = obs
        super().observed(step, obs, monitor)
        self._evaluation_state(step)

    def close(self):
        if self.encoder is not None:
            self.counts.update(self.encoder.ledger())
        super().close()

    def finish(self, result):
        if self.encoder is not None:
            self.counts.update(self.encoder.ledger())
        self.write("phase2_actor", actor_sha256=self.genome_hash,
                   decisions=len(self.decisions), stochastic=self.job["stochastic"])
        super().finish(result)


STUDY = Phase2Study() if os.environ.get("SAFETY_DIAL_MANIGUARD_JOB") else None
