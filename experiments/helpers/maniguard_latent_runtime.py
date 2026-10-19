"""Causal semantic controller inputs, physical traces and passive outcome labels."""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from helpers.maniguard_adaptation import AdapterConfig
from helpers.maniguard_latent import LatentController, feature_vector, outcome_score
from helpers.maniguard_phase2_features import FrozenContextEncoder
from helpers.maniguard_phase2_outcomes import derive_outcome

# Import the reusable base class without starting its historical global runtime.
_job_path = os.environ.pop("SAFETY_DIAL_MANIGUARD_JOB", None)
try:
    from helpers.maniguard_runtime import Study
finally:
    if _job_path is not None:
        os.environ["SAFETY_DIAL_MANIGUARD_JOB"] = _job_path


class LatentStudy(Study):
    def __init__(self):
        job = json.loads(Path(os.environ["SAFETY_DIAL_MANIGUARD_JOB"]).read_text())
        with np.load(job["controller"], allow_pickle=False) as data:
            self.controller = LatentController(data["weights"])
        if self.controller.sha256() != job["controller_sha256"]:
            raise ValueError("Controller file differs from its immutable episode job")
        self.encoder = None
        self.previous_eef = None
        self.policy_payload = {}
        self.query_step = None
        super().__init__(config=AdapterConfig(kind="identity", horizon=8))
        self.counts.update(encoder_load_seconds=0., encoder_load_attempts=0,
                           encoder_image_calls=0, encoder_images=0,
                           encoder_text_calls=0, encoder_seconds=0.)
        self.ledger()

    def install_accounting(self, og):
        super().install_accounting(og)
        # The bundled Vulkan loader's device order was checked against GPU UUIDs.
        # Kit consumes this after benchmark parsing; physics keeps CUDA device 0.
        if any(arg.startswith("--/renderer/activeGpu=") for arg in sys.argv):
            raise ValueError("Unexpected preexisting renderer GPU override")
        sys.argv.append(f"--/renderer/activeGpu={self.job['renderer_vulkan_index']}")
        self.write("gpu_mapping", renderer_vulkan_index=self.job["renderer_vulkan_index"],
                   simulation_gpu_uuid="GPU-0bfb3ed5-af55-472e-3199-058727ff80a7",
                   policy_gpu_uuid="GPU-4f452774-6f8b-6fcc-4ce1-2d4d190e6d07")

    def connect_policy(self, cfg, upstream_connect):
        if self.job["random_policy"]:
            return upstream_connect(cfg)
        study = self
        from helpers.maniguard_client import StudyClient

        class ReceiptClient(StudyClient):
            def infer(self, observation):
                result = super().infer(observation)
                receipt = result.get("latent_receipt", {})
                if (receipt.get("weights_sha256") != study.controller.sha256()
                        or receipt.get("audit") is not False):
                    raise ValueError("Physical server did not use the frozen controller")
                study.write("noise_receipt", step=study.query_step, receipt=receipt,
                            raw_actions=result["actions"],
                            raw_action_dtype=str(np.asarray(result["actions"]).dtype))
                return result

        return ReceiptClient(host=cfg.host, port=cfg.port), "openpi"

    def validate_policy(self, policy, client_type):
        super().validate_policy(policy, client_type)
        if self.job["random_policy"]:
            return
        metadata = policy.get_server_metadata()
        if (metadata.get("native_noise_steering") is not True
                or metadata.get("audit_service") is not False
                or metadata.get("model_shape") != [16, 32]):
            raise ValueError("Physical study requires the verified native-noise service")

    def start(self, scene, obs, env, robot, monitor):
        self.env, self.robot = env, robot
        super().start(scene, obs)
        from PIL import Image

        references = robot.prim.GetMetadata("references")
        reference_paths = [] if references is None else [
            str(ref.assetPath) for ref in references.GetAddedOrExplicitItems()]
        layers = [str(spec.layer.identifier) for spec in robot.prim.GetPrimStack()]
        self.write("runtime_contract", robot_usd_path=str(robot.usd_path),
                   reference_asset_paths=reference_paths, robot_prim_layers=layers,
                   overview_shape=list(np.shape(obs["overview_image"])),
                   wrist_shape=list(np.shape(obs["wrist_images"])))
        if not any("franka_panda_longfinger" in path for path in reference_paths + layers):
            raise ValueError("The live robot USD prim does not prove long-finger asset loading")
        for key in ["overview_image", "wrist_images"]:
            value = np.asarray(obs[key])
            if value.shape != (640, 640, 3) or value.dtype != np.uint8 or value.std() < 1:
                raise ValueError("Native 640-pixel camera is unavailable or blank: " + key)
            Image.fromarray(value).save(self.output / (key + ".png"))
        # The monitor is evaluation telemetry only; it is absent from feature_vector.
        if monitor is None:
            raise ValueError("Physical outcomes require a live passive monitor")
        self.observed(0, obs, monitor)
        started = time.monotonic()
        self.counts["encoder_load_attempts"] += 1
        try:
            self.encoder = FrozenContextEncoder(device=self.job["encoder_device"])
        finally:
            self.counts["encoder_load_seconds"] = time.monotonic() - started
            self.ledger()
        self.encoder.begin_episode(obs["task_descriptions"])
        self.counts.update(self.encoder.ledger())
        self.write("feature_contract", encoder=self.encoder.spec,
                   runtime_provenance=self.encoder.runtime_provenance,
                   controller_sha256=self.controller.sha256(), features=44,
                   actor_information="robot/history, images, instruction only")
        self.ledger()

    def before_query(self, step, obs, frame):
        self.query_step = step
        super().before_query(step, obs, frame)
        try:
            overview, wrist, context = self.encoder.encode_images(
                obs["overview_image"], obs["wrist_images"])
            features = feature_vector(
                frame.joints, frame.fingers, frame.eef_position, overview, wrist,
                self.encoder.text_features, previous_executed=self.adapter.last_executed,
                previous_eef=self.previous_eef)
            self.policy_payload = {"latent_weights": self.controller.weights,
                                   "latent_features": features}
            self.write("latent_features", step=step, features=features,
                       frame={"joints": frame.joints, "fingers": frame.fingers,
                              "eef": frame.eef_position}, context=context,
                       previous_executed=self.adapter.last_executed,
                       previous_eef=self.previous_eef, code=self.controller.code(features))
            folder = self.output / "query_observations"
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder / f"{step:06d}.npz", state=obs["states"],
                                overview=obs["overview_image"], wrist=obs["wrist_images"],
                                prompt=np.asarray(obs["task_descriptions"]), features=features)
            self.previous_eef = np.asarray(frame.eef_position).copy()
        finally:
            self.counts.update(self.encoder.ledger())
            self.ledger()

    def close(self):
        if self.encoder is not None:
            self.counts.update(self.encoder.ledger())
        super().close()

    def finish(self, result):
        super().finish(result)
        rows = [json.loads(line) for line in (self.output / "trace.jsonl").read_text().splitlines()]
        observations = []
        for row in rows:
            if row["type"] == "observed":
                monitor = row.get("monitor") or {}
                if monitor.get("valid") is not True:
                    observations.append({"step": row["step"], "invalid_monitor": True})
                else:
                    observations.append({"step": row["step"], **monitor["observation"]})
        goals = [row for row in rows if row["type"] == "goal"]
        outcome = derive_outcome(self.job["family"], result, observations, goals)
        if outcome["valid"]:
            outcome["score"] = outcome_score(outcome["C_req"], outcome["V"])
        outcome["controller_sha256"] = self.controller.sha256()
        (self.output / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")


STUDY = LatentStudy() if _job_path is not None else None
