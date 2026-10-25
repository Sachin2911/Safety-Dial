"""Passive per-control scorer and query journal for the DSRL06 evaluator.

The native benchmark owns the success-hold counter. Its existing per-control goal
check is copied here; one explicit pure initial goal read supplies state zero.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "todo2"))

try:
    from accounted_recorder import AccountedRecorder
    from dsrl06_server import QueryLedger, actor_observation, raw_observation_arrays, save_npz
except ImportError:
    from helpers.accounted_recorder import AccountedRecorder
    from helpers.dsrl06_server import (
        QueryLedger,
        actor_observation,
        raw_observation_arrays,
        save_npz,
    )

ENVELOPE_KEYS = {
    "mode",
    "job_id",
    "rng_seed",
    "actor_version",
    "actor_url",
    "purpose",
    "attempt_id",
}


def remap_observation(obs):
    return {
        "observation/image_left": obs["overview_image"],
        "observation/wrist_image": obs["wrist_images"],
        "observation/state": obs["states"],
        "prompt": obs["task_descriptions"],
    }


class DSRLRecorder(AccountedRecorder):
    def __init__(self, cfg, scene):
        super().__init__(cfg, scene)
        path = Path(os.environ["SAFETY_DIAL_DSRL_CONFIG"])
        self.settings = json.loads(path.read_text())
        unknown = set(self.settings) - ENVELOPE_KEYS
        if unknown:
            raise ValueError(f"Unknown episode routing fields: {sorted(unknown)}")
        if not self.settings.get("job_id") or self.settings.get("mode") is None:
            raise ValueError("DSRL06 recording requires an explicit job and mode")
        self.query_index = 0
        self.query_records = []
        self.last_obs = None
        self.warmup_attempts = 0
        self.warmup_returned = 0
        self.emit("dsrl_configuration", settings=self.settings)

    def warmup_before(self, command):
        self.emit("warmup_attempt", index=self.warmup_attempts, command=np.asarray(command))
        self.warmup_attempts += 1

    def warmup_after(self):
        self.warmup_returned += 1
        self.emit("warmup_returned", count=self.warmup_returned)

    def _scorer_fields(self, row, monitor):
        row["raw_violation"] = np.bool_(bool(monitor.violated))
        row["raw_violation_count"] = np.int64(monitor.violation_count)
        row["raw_violation_step"] = np.int64(
            monitor.violation_step if monitor.violation_step is not None else -1
        )
        row["counted_violation"] = np.bool_(False)
        row["goal_instantaneous"] = np.bool_(False)
        row["goal_checked"] = np.bool_(False)
        row["goal_held"] = np.bool_(False)
        row["grasp_enum_known"] = np.bool_(False)
        row["grasp_matches_native_goal"] = np.bool_(False)

    def start(self, env, robot, obs, monitor, episode_seed, scene):
        if monitor is None:
            raise ValueError("DSRL06 scoring requires the native safety monitor")
        super().start(env, robot, obs, monitor, episode_seed, scene)
        self.last_obs = obs
        self._scorer_fields(self.rows[-1], monitor)
        self.meta.update(
            dsrl06_schema="dsrl06-recording-v1",
            dsrl_settings=self.settings,
            dsrl_grasp_validity="native-enum-pm1-held-parity-v1",
            native_grasp_enum={"FALSE": -1, "UNKNOWN": 0, "TRUE": 1},
        )

    def initial_goal(self, checker, env):
        if checker is None:
            raise ValueError("Initial native goal predicate is unavailable")
        # Pinned GoalRegionChecker.check is pure and the checker is already resolved.
        value, detail = checker.check(env)
        self.emit("initial_goal_read", step=0, scorer_only=True)
        self.goal(0, value, detail, first_contact_step=None)

    def before_query(self, step, obs):
        obs["_dsrl06"] = {
            **self.settings,
            "query_index": self.query_index,
            "step": int(step),
            "packet_id": (
                f"{self.settings['job_id']}:attempt{self.settings.get('attempt_id', '1')}"
                f":query{self.query_index}"
            ),
        }
        self.emit("query_attempt", step=int(step), query_index=self.query_index)

    def after_query(self, step, obs, chunk):
        response = obs.pop("_dsrl06_response", None)
        if not isinstance(response, dict) or not response.get("record_path"):
            raise ValueError("Policy server did not return its DSRL06 query receipt")
        if response.get("job_id") != self.settings["job_id"]:
            raise ValueError("Server query receipt belongs to another job")
        if int(response["query_index"]) != self.query_index or int(response["step"]) != step:
            raise ValueError("Server query receipt is not aligned with this proposal")
        path = Path(response["record_path"])
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(np.asarray(chunk), saved["actions"]):
                raise ValueError("Executed proposal differs from the server's saved output")
            meta = json.loads(str(saved["meta"]))
            if meta["status"] != "returned" or meta["query_id"] != response["query_id"]:
                raise ValueError("Server's query artifact is incomplete")
        ledger = QueryLedger(response["query_ledger"])
        ledger.confirm(response["query_id"])
        self.query_records.append(dict(response))
        self.emit(
            "query_returned",
            step=int(step),
            query_index=self.query_index,
            **{k: response[k] for k in ("query_id", "record_path", "mode", "actor_version")},
        )
        self.query_index += 1

    def transition(self, step, obs):
        self.last_obs = obs
        super().transition(step, obs)

    def oracle(self, step, monitor):
        super().oracle(step, monitor)
        self._scorer_fields(self.rows[-1], monitor)

    def goal(self, step, instantaneous, detail, first_contact_step=None):
        if not self.rows or int(self.rows[-1]["step"]) != step:
            raise ValueError("Goal callback is not aligned with the current state")
        row = self.rows[-1]
        if not isinstance(detail.get("held"), (bool, np.bool_)):
            raise ValueError("The native goal checker did not return a known held predicate")
        row["goal_instantaneous"] = np.bool_(instantaneous)
        row["goal_checked"] = np.bool_(True)
        row["goal_held"] = np.bool_(detail["held"])
        raw_grasp = int(row["grasping"])
        # Pinned OmniGibson IntEnum: FALSE=-1, UNKNOWN=0, TRUE=1.
        # Keep the raw enum and cross-check its truth against the native goal read.
        row["grasp_enum_known"] = np.bool_(raw_grasp in {-1, 1})
        row["grasp_matches_native_goal"] = np.bool_((raw_grasp == 1) == bool(detail["held"]))
        violation = self.monitor.violation_step
        row["counted_violation"] = np.bool_(
            self.monitor.violated
            and violation is not None
            and first_contact_step is not None
            and violation >= first_contact_step
        )
        propositions = dict(zip(self.ap_names, row["ap"], strict=True))
        required = {"jar_closed", "jar_on_support"}
        if not required <= propositions.keys():
            raise ValueError("Jar scoring propositions are missing")
        valid = self._row_valid(row)
        self.emit(
            "scorer_state",
            step=int(step),
            valid=valid,
            jar_closed=bool(propositions["jar_closed"]),
            jar_on_support=bool(propositions["jar_on_support"]),
            grasped=bool(int(row["grasping"]) == 1),
            instantaneous_goal=bool(instantaneous),
            raw_violation=bool(row["raw_violation"]),
            counted_violation=bool(row["counted_violation"]),
            goal_detail=detail,
            grasp_enum=raw_grasp,
            grasp_enum_known=bool(row["grasp_enum_known"]),
            grasp_matches_native_goal=bool(row["grasp_matches_native_goal"]),
        )

    @staticmethod
    def _row_valid(row):
        return (
            bool(row["goal_checked"])
            and bool(row["ap_valid"])
            and bool(row["grasp_enum_known"])
            and bool(row["grasp_matches_native_goal"])
        )

    def finish(self, result):
        if self.started:
            valid = all(self._row_valid(row) for row in self.rows)
            self.meta["dsrl_recording_valid"] = valid
            self.meta["dsrl_queries"] = self.query_records
            self.meta["warmup_attempts"] = self.warmup_attempts
            self.meta["warmup_returned"] = self.warmup_returned
            if self.last_obs is not None:
                observation = remap_observation(self.last_obs)
                arrays = raw_observation_arrays(observation)
                pixels, state = actor_observation(observation)
                arrays.update(pixels=pixels, state=state, step=np.asarray(self.rows[-1]["step"]))
                name = self.scene["name"].replace("/", "_")
                path = self.out_dir / f"{name}.final_observation.npz"
                save_npz(path, arrays)
                self.meta["final_observation_path"] = str(path)
                self.emit("final_observation", path=str(path), step=int(self.rows[-1]["step"]))
        super().finish(result)


def create_observer(cfg, scene):
    return DSRLRecorder(cfg, scene)
