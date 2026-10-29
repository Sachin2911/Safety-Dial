"""Exercise the native-preserving residual route, its key continuity and query ledger."""

import ast
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.dsrl06_server import QueryLedger
from helpers.dsrl11_protocol import inject_residual, uniform_residual
from helpers.dsrl11_server import NOISE_ORIGIN, Dsrl11QueryLedger, ResidualPolicyBridge


class Backend:
    def seed(self, value):
        return int(value)

    def pack(self, key):
        return np.array([int(key), 0], np.uint32)

    def unpack(self, value):
        return int(value[0])

    def split(self, key):
        return int(key) + 1, int(key) + 100

    def normal(self, key, shape):
        return np.random.default_rng(int(key)).standard_normal(shape).astype(np.float32)


class FakePolicy:
    """No neural network; the decoded chunk is the first eight latent columns."""

    def __init__(self):
        self._rng = 0
        self._model = SimpleNamespace(action_horizon=16, action_dim=32)
        self._sample_kwargs = {}
        self.metadata = {"frozen": True}
        self.calls = []
        self.backend = Backend()

    def infer(self, obs, **kwargs):
        self.calls.append((set(obs), dict(kwargs)))
        self._rng, key = self.backend.split(self._rng)
        noise = kwargs.get("noise")
        if noise is None:
            noise = self.backend.normal(key, (16, 32))
        return {"actions": np.asarray(noise[:, :8], dtype=np.float64)}


def observation(seed=17):
    return {
        "observation/image_left": np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3),
        "observation/wrist_image": np.full((256, 256, 3), 22, np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
        "prompt": "Close and carry the jar.",
        "episode_seed": seed,
    }


def packet(source="zero", index=0, job="B00", **extra):
    values = {
        "mode": "residual",
        "residual_source": source,
        "query_index": index,
        "job_id": job,
        "epsilon_seed": 4242,
        **extra,
    }
    if source == "uniform":
        values.setdefault("uniform_seed", 77)
    return values


def bridge(tmp_path, policy=None, maximum=27_000):
    p = policy or FakePolicy()
    ledger = Dsrl11QueryLedger(tmp_path / "queries.sqlite", maximum)
    return ResidualPolicyBridge(p, tmp_path / "queries", ledger, backend=Backend()), p, ledger


def load_query(result):
    with np.load(result["_dsrl06"]["record_path"], allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def statuses(ledger):
    with ledger.connect() as db:
        return [row[0] for row in db.execute("SELECT status FROM queries ORDER BY id")]


def test_zero_residual_reproduces_native_decoding_and_key_advance(tmp_path):
    wrapped, policy, _ = bridge(tmp_path)
    reference = FakePolicy()
    reference._rng = 4242  # The frozen epsilon seed replaces the benchmark episode seed.
    original = observation()
    for index in range(3):
        expected = reference.infer({k: v for k, v in original.items() if k != "episode_seed"})
        actual = wrapped.infer({**original, "_dsrl06": packet(index=index)})
        assert np.array_equal(actual["actions"], expected["actions"])
        assert policy._rng == reference._rng
        saved = load_query(actual)
        assert np.array_equal(saved["noise"], saved["epsilon"])
        assert not np.any(saved["residual_u"]) and saved["noise"].dtype == np.float32
        assert np.array_equal(policy.calls[-1][1]["noise"], saved["epsilon"])
        meta = json.loads(str(saved["meta"]))
        assert meta["residual_source"] == "zero" and meta["noise_origin"] == NOISE_ORIGIN
        assert meta["benchmark_episode_seed"] == 17 and meta["status"] == "returned"
        assert "episode_seed" not in policy.calls[-1][0] and "_dsrl06" not in policy.calls[-1][0]
    origins = [
        json.loads(str(np.load(p)["meta"]))["rng_origin"]
        for p in sorted((tmp_path / "queries/B00").glob("*.npz"))
    ]
    assert origins[0].startswith("seeded") and all(o.startswith("continued") for o in origins[1:])


def test_uniform_residual_is_added_to_the_same_native_draw(tmp_path):
    zero, _, _ = bridge(tmp_path / "zero")
    uniform, _, _ = bridge(tmp_path / "uniform")
    for index in range(2):
        a = load_query(zero.infer({**observation(), "_dsrl06": packet(index=index)}))
        b = load_query(uniform.infer({**observation(), "_dsrl06": packet("uniform", index=index)}))
        assert np.array_equal(a["epsilon"], b["epsilon"])
        assert np.array_equal(b["residual_u"], uniform_residual(77, index))
        assert np.array_equal(b["noise"], inject_residual(b["epsilon"], b["residual_u"]))
        assert np.array_equal(b["actions"], b["noise"][:, :8].astype(np.float32))


def test_out_of_order_and_cross_job_queries_are_refused_before_reservation(tmp_path):
    wrapped, policy, ledger = bridge(tmp_path)
    wrapped.infer({**observation(), "_dsrl06": packet(index=0)})
    with pytest.raises(ValueError, match="out of order"):
        wrapped.infer({**observation(), "_dsrl06": packet(index=2)})
    with pytest.raises(ValueError, match="out of order"):
        wrapped.infer({**observation(), "_dsrl06": packet(index=1, job="B01")})
    with pytest.raises(ValueError, match="out of order"):
        wrapped.infer({**observation(), "_dsrl06": packet(index=1, attempt_id="2")})
    assert statuses(ledger) == ["returned"] and len(policy.calls) == 1
    # A new job (or replacement attempt) restarts at query zero with its own seed.
    wrapped.infer({**observation(), "_dsrl06": packet(index=0, job="B01", epsilon_seed=9)})
    assert policy._rng == 10


def test_key_tampering_between_queries_is_detected_and_ends_the_stream(tmp_path):
    wrapped, policy, ledger = bridge(tmp_path)
    wrapped.infer({**observation(), "_dsrl06": packet(index=0)})
    policy._rng += 5
    with pytest.raises(ValueError, match="continuity"):
        wrapped.infer({**observation(), "_dsrl06": packet(index=1)})
    assert statuses(ledger) == ["returned", "failed"]
    with pytest.raises(ValueError, match="out of order"):
        wrapped.infer({**observation(), "_dsrl06": packet(index=2)})


def test_rng_advancement_must_match_the_pinned_infer_contract(tmp_path):
    class Skipping(FakePolicy):
        def infer(self, obs, **kwargs):
            result = super().infer(obs, **kwargs)
            self._rng += 1  # An extra split would change every later native draw.
            return result

    wrapped, _, ledger = bridge(tmp_path, Skipping())
    with pytest.raises(ValueError, match="advancement"):
        wrapped.infer({**observation(), "_dsrl06": packet()})
    assert statuses(ledger) == ["failed"]


class Reply:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.value).encode()


def actor_packet(index=0, **extra):
    defaults = {
        "job": "T_r1_prior_3",
        "actor_condition": "r1_prior",
        "actor_version": 0,
        "rng_seed": 123,
        "actor_url": "http://127.0.0.1:9011",
    }
    return packet("actor", index=index, **{**defaults, **extra})


def test_actor_residual_boundary_and_answer_checks(tmp_path, monkeypatch):
    import helpers.dsrl11_server as module

    wrapped, _, ledger = bridge(tmp_path)
    calls = []
    answers = [
        {"residual": [0.5] * 32, "actor_version": 0, "condition": "r1_prior"},
        {"residual": [0.5] * 32, "actor_version": 0, "condition": "r0_prior"},
    ]

    def request(value, timeout):
        calls.append((value.full_url, json.loads(value.data)))
        return Reply(answers[len(calls) - 1])

    monkeypatch.setattr(module.urllib.request, "urlopen", request)
    result = wrapped.infer({**observation(), "_dsrl06": actor_packet()})
    url, payload = calls[0]
    assert url == "http://127.0.0.1:9011/act"
    assert set(payload) == {
        "condition",
        "actor_version",
        "job_id",
        "query_index",
        "rng_seed",
        "pixels_b64",
        "state",
    }
    assert payload["condition"] == "r1_prior" and len(payload["state"]) == 8
    saved = load_query(result)
    assert saved["pixels"].shape == (64, 64, 3)
    assert np.array_equal(saved["noise"], saved["epsilon"] + np.float32(0.125))
    assert "actor_url" not in json.loads(str(saved["meta"]))
    with pytest.raises(ValueError, match="different condition"):
        wrapped.infer({**observation(), "_dsrl06": actor_packet(index=1)})
    assert statuses(ledger) == ["returned", "failed"]


@pytest.mark.parametrize(
    "answer",
    [
        {"residual": [1.5] * 32, "actor_version": 0, "condition": "r1_prior"},
        {"residual": [0.1] * 31, "actor_version": 0, "condition": "r1_prior"},
        {"residual": [0.1] * 32, "actor_version": 1, "condition": "r1_prior"},
    ],
)
def test_actor_answers_outside_the_contract_fail_closed(tmp_path, monkeypatch, answer):
    import helpers.dsrl11_server as module

    wrapped, policy, ledger = bridge(tmp_path)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda value, timeout: Reply(answer))
    with pytest.raises(ValueError):
        wrapped.infer({**observation(), "_dsrl06": actor_packet()})
    assert statuses(ledger) == ["failed"] and not policy.calls


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"epsilon_seed": None}, "epsilon seed"),
        ({"epsilon_seed": True}, "epsilon seed"),
        ({"query_index": -1}, "query index"),
        ({"query_index": "0"}, "query index"),
        ({"residual_source": "gaussian"}, "source"),
        ({"residual_source": "explicit", "residual_u": [0.0] * 32}, "probe only"),
        ({"job_id": "../escape"}, "job identifier"),
    ],
)
def test_policy_contexts_are_validated_before_any_reservation(tmp_path, changes, message):
    wrapped, policy, ledger = bridge(tmp_path)
    request = {**packet(), **changes}
    with pytest.raises(ValueError, match=message):
        wrapped.infer({**observation(), "_dsrl06": request})
    assert statuses(ledger) == [] and not policy.calls


@pytest.mark.parametrize(
    "url", ["http://10.0.0.2:9011", "https://127.0.0.1:9011", "http://user:pw@127.0.0.1:9011"]
)
def test_actor_endpoint_must_be_local_without_credentials(tmp_path, url):
    wrapped, _, ledger = bridge(tmp_path)
    with pytest.raises(ValueError, match="local HTTP|credentials"):
        wrapped.infer({**observation(), "_dsrl06": actor_packet(actor_url=url)})
    assert statuses(ledger) == []


def test_unexpected_vla_inputs_are_recorded_as_failed_queries(tmp_path):
    wrapped, policy, ledger = bridge(tmp_path)
    with pytest.raises(ValueError, match="input boundary"):
        wrapped.infer({**observation(), "privileged": 1, "_dsrl06": packet()})
    assert statuses(ledger) == ["failed"] and not policy.calls


def test_probe_replays_a_recorded_key_and_restores_the_running_stream(tmp_path):
    wrapped, policy, _ = bridge(tmp_path)
    first = load_query(wrapped.infer({**observation(), "_dsrl06": packet(index=0)}))
    running = policy._rng
    probe = {
        "mode": "residual",
        "purpose": "probe",
        "job_id": "A2-decoder",
        "query_index": 0,
        "residual_source": "explicit",
        "residual_u": np.full(32, -1.0, np.float32),
        "rng_before": first["rng_before"],
    }
    replay = load_query(wrapped.infer({**observation(), "_dsrl06": probe}))
    assert np.array_equal(replay["epsilon"], first["epsilon"])
    assert np.array_equal(replay["noise"], first["epsilon"] - np.float32(0.25))
    assert policy._rng == running
    # The policy job continues as if the probe never happened.
    wrapped.infer({**observation(), "_dsrl06": packet(index=1)})
    del probe["rng_before"]
    with pytest.raises(ValueError, match="recorded policy key"):
        wrapped.infer({**observation(), "_dsrl06": probe})


def test_legacy_native_route_is_the_unchanged_parent_path(tmp_path):
    wrapped, policy, _ = bridge(tmp_path)
    result = wrapped.infer(
        {**observation(), "_dsrl06": {"mode": "native", "job_id": "startup", "query_index": 0}}
    )
    assert policy.calls[-1][1] == {}
    assert "reconstruction" in json.loads(str(load_query(result)["meta"]))["noise_origin"]


def test_query_ledger_carries_the_fresh_ceiling(tmp_path):
    ledger = Dsrl11QueryLedger(tmp_path / "q.sqlite")
    assert ledger.maximum == 27_000
    with pytest.raises(ValueError, match="cannot change"):
        Dsrl11QueryLedger(tmp_path / "q.sqlite", 100)
    with pytest.raises(ValueError):
        QueryLedger(tmp_path / "q.sqlite")  # The task-6 ceiling cannot reopen it.
    with pytest.raises(ValueError, match="limit"):
        Dsrl11QueryLedger(tmp_path / "other.sqlite", 27_001)
    small = Dsrl11QueryLedger(tmp_path / "small.sqlite", 1)
    wrapped = ResidualPolicyBridge(FakePolicy(), tmp_path / "queries", small, backend=Backend())
    wrapped.infer({**observation(), "_dsrl06": packet()})
    with pytest.raises(RuntimeError, match="exhausted"):
        wrapped.infer({**observation(), "_dsrl06": packet(index=1)})


def test_real_pinned_policy_infer_zero_residual_equals_native(tmp_path, monkeypatch):
    """The actual pinned Policy.infer, with a synthetic sampler and the real JAX keys."""
    path = Path("/tmp/dsrl06-server-src/policy.py")
    if not path.exists():
        pytest.skip("Download pinned policy.py as documented in the server build receipt")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "a2ac5236e23b11d0f5a840d4a8fef148d89dba921d5452ec5dfcd5a6ffa46f20"
    )
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    from helpers.dsrl06_server import JaxBackend

    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "Policy"
    )
    infer = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "infer")
    infer.decorator_list = []
    ns = {
        "jax": jax,
        "jnp": jnp,
        "np": np,
        "time": time,
        "_model": SimpleNamespace(Observation=SimpleNamespace(from_dict=lambda x: x)),
    }
    module = ast.fix_missing_locations(ast.Module(body=[infer], type_ignores=[]))
    exec(compile(module, str(path), "exec"), ns)

    class Synthetic:
        infer = ns["infer"]
        metadata = {}
        _is_pytorch_model = False
        _sample_kwargs = {}
        _model = SimpleNamespace(action_horizon=16, action_dim=32)

        def __init__(self, seed):
            self._rng = jax.random.key(seed)
            self._input_transform = lambda obs: {"state": obs["observation/state"]}
            self._output_transform = lambda obs: obs
            self._sample_actions = self.sample

        def sample(self, key, obs, noise=None):
            if noise is None:
                noise = jax.random.normal(key, (1, 16, 32))
            return noise[:, :, :8] * 3.0 + obs["state"][:, None, :]

    reference, actual = Synthetic(4242), Synthetic(0)
    ledger = Dsrl11QueryLedger(tmp_path / "q.sqlite")
    wrapped = ResidualPolicyBridge(actual, tmp_path / "queries", ledger, backend=JaxBackend())
    native = {k: v for k, v in observation().items() if k != "episode_seed"}
    for index in range(3):
        expected = reference.infer(native)
        result = wrapped.infer({**observation(), "_dsrl06": packet(index=index)})
        assert np.array_equal(expected["actions"], result["actions"])
        assert np.array_equal(jax.random.key_data(reference._rng), jax.random.key_data(actual._rng))
    nonzero = wrapped.infer({**observation(), "_dsrl06": packet("uniform", index=0, job="E4")})
    saved = load_query(nonzero)
    assert not np.array_equal(saved["noise"], saved["epsilon"])
    np.testing.assert_allclose(
        nonzero["actions"], saved["noise"][:, :8] * 3.0 + np.arange(8), rtol=0, atol=1e-5
    )
