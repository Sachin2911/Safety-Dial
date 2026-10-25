"""Frozen-policy routing and lossless query accounting for the DSRL06 protocol.

Native inference calls Policy.infer without a noise argument. The original key split
and sampler remain untouched. Its Gaussian tensor is reconstructed from the recorded
sample key only after inference, without consuming or changing policy RNG state.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import os
from pathlib import Path
import re
import sqlite3
import threading
import urllib.parse
import urllib.request

import numpy as np

LATENT_SHAPE = (16, 32)
QUERY_LIMIT = 21_000
ACTOR_TRANSFORM = "Pillow RGB BILINEAR 256x256-to-64x64; no crop or flip"
RAW_KEYS = (
    "observation/image_left",
    "observation/wrist_image",
    "observation/state",
    "prompt",
)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def actor_observation(obs):
    """Return exactly the permitted actor inputs, with a frozen image transform."""
    from PIL import Image

    image = np.asarray(obs["observation/image_left"])
    state = np.asarray(obs["observation/state"], dtype=np.float32)
    if image.shape != (256, 256, 3) or image.dtype != np.uint8:
        raise ValueError("Actor requires the native 256x256 RGB uint8 image")
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Actor requires eight finite proprioceptive coordinates")
    pixels = np.asarray(Image.fromarray(image).resize((64, 64), Image.Resampling.BILINEAR))
    return np.ascontiguousarray(pixels), state.copy()


def raw_observation_arrays(obs):
    arrays = {
        "observation_image_left": np.array(obs[RAW_KEYS[0]], copy=True),
        "observation_wrist_image": np.array(obs[RAW_KEYS[1]], copy=True),
        "observation_state": np.array(obs[RAW_KEYS[2]], copy=True),
        "prompt": np.asarray(str(obs["prompt"])),
    }
    for key in ("observation_image_left", "observation_wrist_image"):
        if arrays[key].shape != (256, 256, 3) or arrays[key].dtype != np.uint8:
            raise ValueError("Query capture requires two native 256x256 RGB uint8 images")
    if arrays["observation_state"].shape != (8,):
        raise ValueError("Query capture requires the native eight-coordinate state")
    return arrays


def save_npz(path, arrays):
    """Atomically replace a query's initial input record with its completed record."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class QueryLedger:
    """SQLite reservations are global across the four single-policy processes.

    A reserved row counts even if inference fails or the client loses its response.
    model_returned_utc and client_confirmed_utc distinguish those two boundaries.
    Back up a live ledger with SQLite's backup API, not by copying its WAL blindly.
    """

    def __init__(self, path, maximum=QUERY_LIMIT):
        self.path = Path(path)
        self.maximum = int(maximum)
        if not 0 < self.maximum <= QUERY_LIMIT:
            raise ValueError("Invalid global model-query limit")
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

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def reserve(self, context):
        db = self.connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            count = db.execute("SELECT COALESCE(MAX(id), 0) FROM queries").fetchone()[0]
            if count >= self.maximum:
                raise RuntimeError("Global model-query allowance exhausted")
            identifier = count + 1
            db.execute(
                "INSERT INTO queries (id,reserved_utc,job_id,query_index,purpose,mode,"
                "actor_version,packet_id,attempt_id,status) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    utc(),
                    context["job_id"],
                    context["query_index"],
                    context["purpose"],
                    context["mode"],
                    context["actor_version"],
                    context["packet_id"],
                    str(context.get("attempt_id", "1")),
                    "reserved",
                ),
            )
            db.commit()
            return identifier
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def model_returned(self, identifier):
        with self.connect() as db:
            db.execute(
                "UPDATE queries SET model_returned_utc=?,status='model_returned' WHERE id=?",
                (utc(), identifier),
            )

    def returned(self, identifier, path):
        with self.connect() as db:
            db.execute(
                "UPDATE queries SET status='returned',record_path=? WHERE id=?",
                (str(path), identifier),
            )

    def failed(self, identifier, error_type, path=None):
        with self.connect() as db:
            db.execute(
                "UPDATE queries SET status='failed',error_type=?,record_path=? WHERE id=?",
                (str(error_type), str(path) if path else None, identifier),
            )

    def confirm(self, identifier):
        with self.connect() as db:
            row = db.execute(
                "SELECT status,model_returned_utc FROM queries WHERE id=?", (int(identifier),)
            ).fetchone()
            if row is None or row[0] != "returned" or row[1] is None:
                raise ValueError("Cannot confirm an unreturned model query")
            db.execute(
                "UPDATE queries SET client_confirmed_utc=? WHERE id=?", (utc(), int(identifier))
            )


class JaxBackend:
    """Import JAX only inside the intended OpenPI interpreter."""

    def __init__(self):
        import jax

        self.jax = jax

    def seed(self, value):
        return self.jax.random.key(int(value))

    def pack(self, key):
        return np.asarray(self.jax.random.key_data(key), dtype=np.uint32)

    def unpack(self, data):
        return self.jax.random.wrap_key_data(np.asarray(data, dtype=np.uint32))

    def split(self, key):
        return self.jax.random.split(key)

    def normal(self, key, shape):
        return np.asarray(self.jax.random.normal(key, shape))


class PolicyBridge:
    """One frozen policy instance, with a serial request boundary and private actor RNG."""

    def __init__(self, policy, capture_root, query_ledger, *, actor_url=None, backend=None):
        self.policy = policy
        if getattr(policy, "_is_pytorch_model", False):
            raise ValueError("The frozen DSRL06 protocol requires the pinned JAX policy")
        model = getattr(policy, "_model", None)
        if model is not None and (int(model.action_horizon), int(model.action_dim)) != LATENT_SHAPE:
            raise ValueError("Loaded model does not have the frozen 16 by 32 latent shape")
        if "noise" in getattr(policy, "_sample_kwargs", {}):
            raise ValueError("The native sampler must not contain a preset noise override")
        self.capture_root = Path(capture_root)
        self.capture_root.mkdir(parents=True, exist_ok=True)
        self.ledger = query_ledger
        self.actor_url = actor_url
        self.backend = backend or JaxBackend()
        self.last_seed = None
        self.unscoped_queries = 0
        self.lock = threading.RLock()

    @property
    def metadata(self):
        return {
            **getattr(self.policy, "metadata", {}),
            "dsrl06_schema": "dsrl06-query-v1",
            "latent_shape": list(LATENT_SHAPE),
            "actor_transform": ACTOR_TRANSFORM,
            "query_ledger": str(self.ledger.path),
        }

    def _context(self, envelope):
        settings = dict(envelope)
        mode = str(settings.get("mode", "native"))
        aliases = {"bounded_random": "random", "dsrl": "actor", "untrained": "actor"}
        mode = aliases.get(mode, mode)
        if mode not in {
            "native",
            "native_replay",
            "zero",
            "random",
            "gaussian",
            "actor",
            "explicit",
        }:
            raise ValueError("Unknown DSRL06 mode")
        job = str(settings.get("job_id", "startup"))
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,120}", job):
            raise ValueError("Invalid query job identifier")
        index = int(settings.get("query_index", self.unscoped_queries))
        if index < 0:
            raise ValueError("Negative query index")
        context = {
            "job_id": job,
            "query_index": index,
            "mode": mode,
            "purpose": str(settings.get("purpose", "startup" if not envelope else "policy")),
            "actor_version": int(settings.get("actor_version", -1)),
            "packet_id": str(settings.get("packet_id", f"{job}:{index}")),
            "attempt_id": str(settings.get("attempt_id", "1")),
            "step": int(settings.get("step", index * 8)),
            "rng_seed": int(settings.get("rng_seed", 0)),
        }
        self.unscoped_queries += 1
        return settings, context

    def _actor_noise(self, obs, settings, context):
        url = settings.get("actor_url") or self.actor_url
        parsed = urllib.parse.urlsplit(str(url))
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("The learner endpoint must be a local HTTP service")
        if parsed.username or parsed.password or context["actor_version"] < 0:
            raise ValueError("Invalid actor endpoint or unpublished actor version")
        pixels, state = actor_observation(obs)
        request = {
            "job_id": context["job_id"],
            "query_index": context["query_index"],
            "actor_version": context["actor_version"],
            "rng_seed": context["rng_seed"],
            "pixels_b64": base64.b64encode(pixels.tobytes()).decode("ascii"),
            "state": state.tolist(),
        }
        address = str(url).rstrip("/") + ("" if parsed.path.endswith("/act") else "/act")
        payload = urllib.request.Request(
            address, data=json.dumps(request).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(payload, timeout=120) as response:
            result = json.load(response)
        if int(result["actor_version"]) != context["actor_version"]:
            raise ValueError("Learner returned a different actor version")
        return np.asarray(result["noise"], dtype=np.float32)

    def _noise(self, obs, settings, context):
        mode = context["mode"]
        if mode == "actor":
            value = self._actor_noise(obs, settings, context)
        elif mode == "explicit":
            value = np.asarray(settings["noise"], dtype=np.float32)
            if value.shape == (1, *LATENT_SHAPE):
                value = value[0]
        else:
            seed = np.random.SeedSequence(
                [context["rng_seed"] & 0xFFFFFFFF, context["query_index"]]
            )
            rng = np.random.Generator(np.random.PCG64(seed))
            if mode == "zero":
                value = np.zeros(32, np.float32)
            elif mode == "random":
                value = rng.uniform(-1.0, 1.0, 32).astype(np.float32)
            elif mode == "gaussian":
                value = rng.standard_normal(32).astype(np.float32)
            else:
                raise ValueError("Native mode must not enter the explicit-noise path")
        if value.shape == (32,):
            value = np.repeat(value[None, :], 16, axis=0)
        if value.shape != LATENT_SHAPE or not np.isfinite(value).all():
            raise ValueError("Noise must be a finite 32-vector or 16 by 32 tensor")
        unbounded_replay = mode == "explicit" and bool(settings.get("native_draw_replay"))
        if unbounded_replay and context["purpose"] != "probe":
            raise ValueError("Unbounded native-noise replay is only a model probe")
        if mode != "gaussian" and not unbounded_replay and np.max(np.abs(value)) > 1.000001:
            raise ValueError("Bounded actor/probe noise exceeds magnitude 1")
        return np.asarray(value, dtype=np.float32)

    def infer(self, obs):
        with self.lock:
            return self._infer(obs)

    def _infer(self, incoming):
        obs = dict(incoming)
        settings, context = self._context(obs.pop("_dsrl06", {}))
        obs.pop("_dsrl06_response", None)
        episode_seed = obs.pop("episode_seed", None)
        context["episode_seed"] = int(episode_seed) if episode_seed is not None else None
        identifier = self.ledger.reserve(context)
        directory = self.capture_root / context["job_id"]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"query_{identifier:08d}.npz"
        arrays = {}
        previous_rng, previous_seed = self.policy._rng, self.last_seed
        preserve = context["purpose"] == "probe"
        metadata = {
            **context,
            "query_id": identifier,
            "reserved_utc": utc(),
            "status": "reserved",
            "actor_transform": ACTOR_TRANSFORM,
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
            if episode_seed is not None and int(episode_seed) != self.last_seed:
                self.policy._rng = self.backend.seed(int(episode_seed))
                self.last_seed = int(episode_seed)
            if context["mode"] == "native_replay":
                if context["purpose"] != "probe":
                    raise ValueError("Original native replay is only a model probe")
                self.policy._rng = self.backend.unpack(settings["rng_before"])
            rng_before = self.policy._rng
            arrays["rng_before"] = self.backend.pack(rng_before)
            if context["mode"] in {"native", "native_replay"}:
                # No noise keyword, no sampler replacement and no additional model RNG advance.
                result = self.policy.infer(obs)
            else:
                arrays["noise"] = self._noise(obs, settings, context)
                result = self.policy.infer(obs, noise=arrays["noise"])
            self.ledger.model_returned(identifier)
            metadata["model_returned_utc"] = utc()
            arrays["actions_native"] = np.asarray(result["actions"])
            # The frozen benchmark explicitly converts the decoded chunk to float32.
            arrays["actions"] = np.asarray(result["actions"], dtype=np.float32)
            arrays["rng_after"] = self.backend.pack(self.policy._rng)
            expected_next, sample_key = self.backend.split(rng_before)
            if not np.array_equal(arrays["rng_after"], self.backend.pack(expected_next)):
                raise ValueError("Policy RNG advancement differs from pinned Policy.infer")
            arrays["sample_rng"] = self.backend.pack(sample_key)
            if context["mode"] in {"native", "native_replay"}:
                arrays["noise"] = self.backend.normal(sample_key, (1, *LATENT_SHAPE))[0]
                metadata["noise_origin"] = "same-key Gaussian reconstruction after native inference"
            else:
                metadata["noise_origin"] = "explicit noise supplied before inference"
            if arrays["actions"].shape != (16, 8) or not np.isfinite(arrays["actions"]).all():
                raise ValueError("Frozen policy did not return a finite 16 by 8 action chunk")
            if arrays["noise"].shape != LATENT_SHAPE or not np.isfinite(arrays["noise"]).all():
                raise ValueError("Native latent reconstruction has an unexpected shape")
            metadata["status"] = "returned"
            arrays["meta"] = np.asarray(json.dumps(metadata, default=json_value))
            save_npz(path, arrays)
            self.ledger.returned(identifier, path)
            return {
                **result,
                "_dsrl06": {
                    **context,
                    "query_id": identifier,
                    "record_path": str(path),
                    "query_ledger": str(self.ledger.path),
                    "noise_origin": metadata["noise_origin"],
                },
            }
        except BaseException as error:
            metadata.update(status="failed", error_type=type(error).__name__)
            if arrays:
                arrays["meta"] = np.asarray(json.dumps(metadata, default=json_value))
                save_npz(path, arrays)
            self.ledger.failed(identifier, type(error).__name__, path if arrays else None)
            raise
        finally:
            if preserve:
                self.policy._rng, self.last_seed = previous_rng, previous_seed
