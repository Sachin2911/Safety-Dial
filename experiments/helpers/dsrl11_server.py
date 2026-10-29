"""Native-preserving latent residual routing for the frozen Jar policy (task 11).

For each residual query the server takes the exact native Gaussian draw that
``Policy.infer`` would sample from its recorded key, adds ``0.25 * tile(u, 16)``
and supplies the sum through the explicit-noise path. A zero residual therefore
passes the native draw through unchanged. The original task-6 modes stay
available, unchanged, for the model-only identity probes.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

import numpy as np

from helpers.dsrl06_server import (
    ACTOR_TRANSFORM,
    RAW_KEYS,
    PolicyBridge,
    QueryLedger,
    actor_observation,
    json_value,
    raw_observation_arrays,
    save_npz,
    utc,
)
from helpers.dsrl11_protocol import (
    CONDITIONS,
    LATENT_SHAPE,
    RESIDUAL_SCALE,
    inject_residual,
    uniform_residual,
)

QUERY_LIMIT = 27_000
NOISE_ORIGIN = "native draw from the recorded policy key plus 0.25 tiled residual"
SOURCES = {"zero", "uniform", "actor", "explicit"}


class Dsrl11QueryLedger(QueryLedger):
    """The task-6 SQLite ledger with the fresh task-11 query ceiling."""

    def __init__(self, path, maximum=QUERY_LIMIT):
        if not 0 < int(maximum) <= QUERY_LIMIT:
            raise ValueError("Invalid global model-query limit")
        # The parent constructor enforces its own historical ceiling, so build the
        # identical schema here with the new cap rather than relaxing the old module.
        self.path = Path(path)
        self.maximum = int(maximum)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value TEXT)")
            db.execute(
                "INSERT OR IGNORE INTO settings VALUES ('maximum_queries', ?)",
                (str(self.maximum),),
            )
            existing = db.execute(
                "SELECT value FROM settings WHERE name='maximum_queries'"
            ).fetchone()[0]
            if int(existing) != self.maximum:
                raise ValueError("A resumed server cannot change the query cap")
            db.execute(
                "CREATE TABLE IF NOT EXISTS queries ("
                "id INTEGER PRIMARY KEY, reserved_utc TEXT NOT NULL, job_id TEXT NOT NULL, "
                "query_index INTEGER NOT NULL, purpose TEXT NOT NULL, mode TEXT NOT NULL, "
                "actor_version INTEGER NOT NULL, packet_id TEXT NOT NULL, "
                "attempt_id TEXT NOT NULL, "
                "status TEXT NOT NULL, model_returned_utc TEXT, client_confirmed_utc TEXT, "
                "record_path TEXT, error_type TEXT)"
            )


def _local_actor_url(url):
    parsed = urllib.parse.urlsplit(str(url))
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The actor endpoint must be a local HTTP service")
    if parsed.username or parsed.password:
        raise ValueError("The actor endpoint cannot carry credentials")
    return str(url).rstrip("/") + ("" if parsed.path.endswith("/act") else "/act")


class ResidualPolicyBridge(PolicyBridge):
    """Adds the task-11 residual route; other routes are the unchanged parent paths."""

    def __init__(self, policy, capture_root, query_ledger, *, actor_url=None, backend=None):
        super().__init__(policy, capture_root, query_ledger, actor_url=actor_url, backend=backend)
        self.residual_job = None
        self.residual_next_query = 0
        self.residual_rng_after = None

    @property
    def metadata(self):
        return {
            **super().metadata,
            "dsrl11_schema": "dsrl11-query-v1",
            "residual_scale": RESIDUAL_SCALE,
            "noise_origin": NOISE_ORIGIN,
        }

    def _infer(self, incoming):
        envelope = incoming.get("_dsrl06") or {}
        if envelope.get("mode") != "residual":
            return super()._infer(incoming)
        return self._residual_infer(incoming)

    @staticmethod
    def _residual_context(envelope):
        settings = dict(envelope)
        job = str(settings.get("job_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", job):
            raise ValueError("Invalid query job identifier")
        index = settings.get("query_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("Residual queries require a nonnegative integer query index")
        purpose = str(settings.get("purpose", "policy"))
        source = settings.get("residual_source")
        if purpose not in {"policy", "probe"} or source not in SOURCES:
            raise ValueError("Unknown residual purpose or source")
        if source == "explicit" and purpose != "probe":
            raise ValueError("An explicit residual vector is a model probe only")
        context = {
            "job_id": job,
            "query_index": index,
            "mode": "residual",
            "purpose": purpose,
            "actor_version": int(settings.get("actor_version", -1)),
            "packet_id": str(settings.get("packet_id", f"{job}:{index}")),
            "attempt_id": str(settings.get("attempt_id", "1")),
            "step": int(settings.get("step", index * 8)),
            "residual_source": source,
            "residual_scale": RESIDUAL_SCALE,
        }
        if purpose == "policy":
            seed = settings.get("epsilon_seed")
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
                raise ValueError("A policy residual query needs its frozen epsilon seed")
            context["epsilon_seed"] = seed
        elif "rng_before" not in settings:
            raise ValueError("A residual probe must replay a recorded policy key")
        if source == "uniform":
            seed = settings.get("uniform_seed")
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
                raise ValueError("A uniform residual needs its frozen stream seed")
            context["uniform_seed"] = seed
        if source == "actor":
            condition = settings.get("actor_condition")
            if condition not in CONDITIONS or context["actor_version"] < 0:
                raise ValueError("An actor residual needs a known condition and version")
            seed = settings.get("rng_seed")
            if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
                raise ValueError("An actor residual needs its frozen sampling seed")
            context.update(actor_condition=condition, rng_seed=seed)
            context["actor_url"] = _local_actor_url(settings.get("actor_url"))
        return settings, context

    def _actor_residual(self, obs, context):
        pixels, state = actor_observation(obs)
        request = {
            "condition": context["actor_condition"],
            "actor_version": context["actor_version"],
            "job_id": context["job_id"],
            "query_index": context["query_index"],
            "rng_seed": context["rng_seed"],
            "pixels_b64": base64.b64encode(pixels.tobytes()).decode("ascii"),
            "state": state.tolist(),
        }
        payload = urllib.request.Request(
            context["actor_url"],
            data=json.dumps(request).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(payload, timeout=120) as response:
            result = json.load(response)
        if (
            int(result["actor_version"]) != context["actor_version"]
            or result.get("condition") != context["actor_condition"]
        ):
            raise ValueError("Actor service answered for a different condition or version")
        value = np.asarray(result["residual"], dtype=np.float32)
        if value.shape != (32,) or not np.isfinite(value).all() or np.max(np.abs(value)) > 1:
            raise ValueError("Actor residual must be a finite 32-vector in [-1, 1]")
        return value

    def _residual(self, obs, settings, context):
        source = context["residual_source"]
        if source == "zero":
            return np.zeros(32, np.float32)
        if source == "uniform":
            return uniform_residual(context["uniform_seed"], context["query_index"])
        if source == "actor":
            return self._actor_residual(obs, context)
        value = np.asarray(settings.get("residual_u"), dtype=np.float32)
        if value.shape != (32,) or not np.isfinite(value).all() or np.max(np.abs(value)) > 1:
            raise ValueError("Explicit probe residual must be a finite 32-vector in [-1, 1]")
        return value

    def _residual_infer(self, incoming):
        obs = dict(incoming)
        settings, context = self._residual_context(obs.pop("_dsrl06", {}))
        obs.pop("_dsrl06_response", None)
        episode_seed = obs.pop("episode_seed", None)
        context["benchmark_episode_seed"] = int(episode_seed) if episode_seed is not None else None
        probe = context["purpose"] == "probe"
        job_key = (context["job_id"], context["attempt_id"])
        if not probe:
            # Continuity is checked before reserving: a misordered request is refused.
            if context["query_index"] == 0:
                pass
            elif job_key != self.residual_job or context["query_index"] != self.residual_next_query:
                raise ValueError("Residual query is out of order for this server's active job")
        identifier = self.ledger.reserve(context)
        directory = self.capture_root / context["job_id"]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"query_{identifier:08d}.npz"
        arrays = {}
        previous_rng, previous_seed = self.policy._rng, self.last_seed
        metadata = {
            **{k: v for k, v in context.items() if k != "actor_url"},
            "query_id": identifier,
            "reserved_utc": utc(),
            "status": "reserved",
            "actor_transform": ACTOR_TRANSFORM,
            "noise_origin": NOISE_ORIGIN,
        }
        try:
            if set(obs) != set(RAW_KEYS):
                raise ValueError("Unexpected fields at the frozen VLA input boundary")
            arrays = raw_observation_arrays(obs)
            pixels, state = actor_observation(obs)
            arrays.update(pixels=pixels, state=state)
            metadata["status"] = "input_saved"
            arrays["meta"] = np.asarray(json.dumps(metadata, default=json_value))
            save_npz(path, arrays)
            if probe:
                self.policy._rng = self.backend.unpack(settings["rng_before"])
                metadata["rng_origin"] = "recorded probe key"
            elif context["query_index"] == 0:
                self.policy._rng = self.backend.seed(context["epsilon_seed"])
                self.last_seed = None
                metadata["rng_origin"] = "seeded from the frozen epsilon_seed at query zero"
            else:
                if not np.array_equal(self.backend.pack(self.policy._rng), self.residual_rng_after):
                    raise ValueError("Native policy key continuity is broken within the job")
                metadata["rng_origin"] = "continued from the previous query of this job"
            rng_before = self.policy._rng
            arrays["rng_before"] = self.backend.pack(rng_before)
            expected_next, sample_key = self.backend.split(rng_before)
            arrays["sample_rng"] = self.backend.pack(sample_key)
            epsilon = self.backend.normal(sample_key, (1, *LATENT_SHAPE))[0]
            if epsilon.dtype != np.float32:
                raise ValueError("The native draw must already be float32")
            residual = self._residual(obs, settings, context)
            noise = inject_residual(epsilon, residual)
            arrays["epsilon"] = epsilon
            arrays["residual_u"] = residual
            arrays["noise"] = noise
            result = self.policy.infer(obs, noise=noise)
            self.ledger.model_returned(identifier)
            metadata["model_returned_utc"] = utc()
            arrays["actions_native"] = np.asarray(result["actions"])
            arrays["actions"] = np.asarray(result["actions"], dtype=np.float32)
            arrays["rng_after"] = self.backend.pack(self.policy._rng)
            if not np.array_equal(arrays["rng_after"], self.backend.pack(expected_next)):
                raise ValueError("Policy RNG advancement differs from pinned Policy.infer")
            if arrays["actions"].shape != (16, 8) or not np.isfinite(arrays["actions"]).all():
                raise ValueError("Frozen policy did not return a finite 16 by 8 action chunk")
            metadata["status"] = "returned"
            arrays["meta"] = np.asarray(json.dumps(metadata, default=json_value))
            save_npz(path, arrays)
            self.ledger.returned(identifier, path)
            if not probe:
                self.residual_job = job_key
                self.residual_next_query = context["query_index"] + 1
                self.residual_rng_after = arrays["rng_after"]
            return {
                **result,
                "_dsrl06": {
                    **{k: v for k, v in context.items() if k != "actor_url"},
                    "query_id": identifier,
                    "record_path": str(path),
                    "query_ledger": str(self.ledger.path),
                    "noise_origin": NOISE_ORIGIN,
                },
            }
        except BaseException as error:
            metadata.update(status="failed", error_type=type(error).__name__)
            if arrays:
                arrays["meta"] = np.asarray(json.dumps(metadata, default=json_value))
                save_npz(path, arrays)
            self.ledger.failed(identifier, type(error).__name__, path if arrays else None)
            if not probe:
                # A failed query ends this job's admissible stream on this server.
                self.residual_job = None
            raise
        finally:
            if probe:
                self.policy._rng, self.last_seed = previous_rng, previous_seed
