"""Exercise native routing, persistent query counts and passive evaluator boundaries."""

import ast
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from helpers.dsrl06_recording import DSRLRecorder
from helpers.dsrl06_server import PolicyBridge, QueryLedger, actor_observation

ROOT = Path(__file__).resolve().parents[2]


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
    """No neural network; the key and kwargs expose any routing-side changes."""

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


def observation():
    return {
        "observation/image_left": np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3),
        "observation/wrist_image": np.full((256, 256, 3), 22, np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
        "prompt": "Close and carry the jar.",
        "episode_seed": 17,
    }


def packet(mode="native", index=0, job="E0", **extra):
    return {
        "mode": mode,
        "query_index": index,
        "job_id": job,
        "rng_seed": 91,
        "actor_version": 0,
        **extra,
    }


def bridge(tmp_path, maximum=21000, policy=None):
    p = policy or FakePolicy()
    ledger = QueryLedger(tmp_path / "queries.sqlite", maximum)
    return PolicyBridge(p, tmp_path / "queries", ledger, backend=Backend()), p, ledger


def load_query(result):
    with np.load(result["_dsrl06"]["record_path"], allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def test_native_preserves_original_infer_kwargs_rng_and_reconstructed_draw(tmp_path):
    wrapped, policy, ledger = bridge(tmp_path)
    reference = FakePolicy()
    reference._rng = 17
    original = observation()
    for i in range(3):
        request = {**original, "_dsrl06": packet(index=i)}
        expected = reference.infer({k: original[k] for k in original if k != "episode_seed"})
        actual = wrapped.infer(request)
        assert np.array_equal(actual["actions"], expected["actions"])
        assert policy._rng == reference._rng
        assert policy.calls[-1][1] == {}  # No explicit noise, including noise=None.
        assert "_dsrl06" not in policy.calls[-1][0] and "episode_seed" not in policy.calls[-1][0]
        saved = load_query(actual)
        assert saved["noise"].shape == (16, 32)
        assert np.array_equal(saved["actions"], saved["noise"][:, :8])
        assert saved["actions_native"].dtype == np.float64
        assert saved["actions"].dtype == np.float32
        assert "reconstruction" in json.loads(str(saved["meta"]))["noise_origin"]
    assert "episode_seed" in original
    with ledger.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM queries").fetchone()[0] == 3


def test_rollout_packet_identity_distinguishes_replacement_attempts():
    identifiers = []
    for attempt in ("1", "2"):
        recorder = DSRLRecorder.__new__(DSRLRecorder)
        recorder.settings = {"job_id": "E0", "mode": "native", "attempt_id": attempt}
        recorder.query_index = 0
        recorder.emit = lambda *args, **kwargs: None
        observation = {}
        recorder.before_query(0, observation)
        identifiers.append(observation["_dsrl06"]["packet_id"])
    assert identifiers == ["E0:attempt1:query0", "E0:attempt2:query0"]


def test_new_episode_reseeds_and_probe_replay_restores_running_rng(tmp_path):
    wrapped, policy, _ = bridge(tmp_path)
    first = wrapped.infer({**observation(), "_dsrl06": packet()})
    saved = load_query(first)
    key_after = policy._rng
    replay = wrapped.infer(
        {
            **observation(),
            "_dsrl06": packet(
                "native_replay",
                job="probe",
                purpose="probe",
                rng_before=saved["rng_before"],
            ),
        }
    )
    assert policy._rng == key_after and wrapped.last_seed == 17
    assert np.array_equal(first["actions"], replay["actions"])
    changed = {**observation(), "episode_seed": 27, "_dsrl06": packet(job="E1")}
    wrapped.infer(changed)
    assert policy._rng == 28


@pytest.mark.parametrize("mode", ["zero", "random", "gaussian"])
def test_tiled_modes_are_episode_deterministic_and_do_not_use_model_rng(tmp_path, mode):
    one, p, _ = bridge(tmp_path / "a")
    two, q, _ = bridge(tmp_path / "b")
    a = one.infer({**observation(), "_dsrl06": packet(mode, index=3)})
    b = two.infer({**observation(), "episode_seed": 99, "_dsrl06": packet(mode, index=3)})
    x, y = load_query(a)["noise"], load_query(b)["noise"]
    assert np.array_equal(x, y)
    assert np.array_equal(x, np.repeat(x[:1], 16, axis=0))
    if mode != "gaussian":
        assert np.max(abs(x)) <= 1
    assert p.calls[-1][1]["noise"].shape == (16, 32)
    assert q._rng == 100


def test_actor_boundary_contains_only_bilinear_image_and_eight_state_values(tmp_path, monkeypatch):
    import helpers.dsrl06_server as module

    wrapped, _, _ = bridge(tmp_path)
    calls = []

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"noise": [0.2] * 32, "actor_version": 8}).encode()

    def request(value, timeout):
        calls.append((value.full_url, json.loads(value.data), timeout))
        return Reply()

    monkeypatch.setattr(module.urllib.request, "urlopen", request)
    result = wrapped.infer(
        {
            **observation(),
            "_dsrl06": packet(
                "actor",
                actor_version=8,
                actor_url="http://127.0.0.1:9009",
            ),
        }
    )
    payload = calls[0][1]
    assert set(payload) == {
        "job_id",
        "query_index",
        "actor_version",
        "rng_seed",
        "pixels_b64",
        "state",
    }
    assert calls[0][0] == "http://127.0.0.1:9009/act"
    assert len(payload["state"]) == 8
    assert load_query(result)["pixels"].shape == (64, 64, 3)
    assert np.allclose(load_query(result)["noise"], 0.2)


def test_failed_query_and_lost_response_are_counted_before_model_call(tmp_path):
    wrapped, policy, ledger = bridge(tmp_path, maximum=2)
    with pytest.raises(ValueError, match="exceeds"):
        wrapped.infer({**observation(), "_dsrl06": packet("explicit", noise=np.ones((16, 32)) * 2)})
    assert not policy.calls
    result = wrapped.infer({**observation(), "_dsrl06": packet(index=1)})
    with ledger.connect() as db:
        rows = db.execute(
            "SELECT status,model_returned_utc,client_confirmed_utc FROM queries ORDER BY id"
        ).fetchall()
    assert rows[0] == ("failed", None, None)
    assert rows[1][0] == "returned" and rows[1][1] and rows[1][2] is None
    ledger.confirm(result["_dsrl06"]["query_id"])
    with pytest.raises(RuntimeError, match="exhausted"):
        wrapped.infer({**observation(), "_dsrl06": packet(index=2)})
    with pytest.raises(ValueError, match="cannot change"):
        QueryLedger(ledger.path, maximum=3)


def test_global_sqlite_reservations_are_atomic_across_servers(tmp_path):
    ledger = QueryLedger(tmp_path / "global.sqlite", maximum=30)

    def reserve(i):
        return ledger.reserve(
            {
                "job_id": f"job{i}",
                "query_index": 0,
                "purpose": "test",
                "mode": "native",
                "actor_version": 0,
                "packet_id": str(i),
            }
        )

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        ids = list(pool.map(reserve, range(30)))
    assert sorted(ids) == list(range(1, 31))
    with pytest.raises(RuntimeError, match="exhausted"):
        reserve(31)


def test_real_pinned_policy_infer_jax_path_preserves_native_output(tmp_path, monkeypatch):
    """Execute the actual pinned infer function with a synthetic, non-neural sampler."""
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
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[infer], type_ignores=[])), str(path), "exec"
        ),
        ns,
    )

    class Synthetic:
        infer = ns["infer"]
        metadata = {}
        _is_pytorch_model = False
        _sample_kwargs = {}
        _model = SimpleNamespace(action_horizon=16, action_dim=32)

        def __init__(self):
            self._rng = jax.random.key(17)
            self._input_transform = lambda obs: {"state": obs["observation/state"]}
            self._output_transform = lambda obs: obs
            self._sample_actions = self.sample

        def sample(self, key, obs, noise=None):
            if noise is None:
                noise = jax.random.normal(key, (1, 16, 32))
            return noise[:, :, :8] + obs["state"][:, None, :]

    reference, actual = Synthetic(), Synthetic()
    wrapped = PolicyBridge(
        actual, tmp_path / "queries", QueryLedger(tmp_path / "q.sqlite"), backend=JaxBackend()
    )
    for i in range(3):
        expected = reference.infer({k: v for k, v in observation().items() if k != "episode_seed"})
        result = wrapped.infer({**observation(), "_dsrl06": packet(index=i)})
        assert np.array_equal(expected["actions"], result["actions"])
        assert np.array_equal(jax.random.key_data(reference._rng), jax.random.key_data(actual._rng))


def test_patch_is_source_guarded_reversible_and_keeps_native_hold_counter():
    spec = importlib.util.spec_from_file_location(
        "patch06", ROOT / "experiments/scripts/patch_dsrl06.py"
    )
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    source_path = ROOT / "outputs/adapter05/upstream/maniguard/eval/benchmark.py"
    if not source_path.exists():
        pytest.skip("Pinned evaluator source is checked again by the runtime patch command")
    original = source_path.read_text()
    changed = patch.patched(original)
    assert patch.restore(changed) == original
    assert changed.count("consec_success += 1") == original.count("consec_success += 1") == 1
    assert changed.index("recorder.initial_goal(goal_checker, env)") < changed.index(
        "consec_success = 0"
    )
    with pytest.raises(ValueError, match="frozen"):
        patch.patched(original + "\n# changed")


def recorder(tmp_path, monkeypatch):
    config = tmp_path / "route.json"
    config.write_text(json.dumps({"mode": "native", "job_id": "E0", "actor_version": 0}))
    monkeypatch.setenv("SAFETY_DIAL_DSRL_CONFIG", str(config))
    cfg = SimpleNamespace(recording_output_dir=str(tmp_path / "recording"))
    return DSRLRecorder(cfg, {"name": "task_0001/base"})


def test_warmup_failure_journal_survives_before_initialization(tmp_path, monkeypatch):
    rec = recorder(tmp_path, monkeypatch)
    rec.warmup_before(np.zeros(8))
    rec.warmup_after()
    rec.warmup_before(np.zeros(8))
    rec.finish({"status": "initialization_failed"})
    events = [
        json.loads(s)
        for s in (tmp_path / "recording/task_0001_base.journal.jsonl").read_text().splitlines()
    ]
    assert sum(e["event"] == "warmup_attempt" for e in events) == 2
    assert sum(e["event"] == "warmup_returned" for e in events) == 1
    assert events[-1]["event"] == "finish"


def test_initial_goal_read_does_not_touch_hold_and_raw_gate_is_exact(tmp_path, monkeypatch):
    rec = recorder(tmp_path, monkeypatch)
    monitor = SimpleNamespace(violated=False, violation_count=0, violation_step=None)
    rec.monitor = monitor
    rec.ap_names = ["jar_closed", "jar_on_support"]
    rec.rows = [{"step": 0, "ap": np.array([True, True]), "ap_valid": True, "grasping": 1}]
    rec._scorer_fields(rec.rows[0], monitor)
    env = SimpleNamespace(native_hold_counter=0)
    checker = SimpleNamespace(check=lambda _: (True, {"held": True}))
    rec.initial_goal(checker, env)
    assert env.native_hold_counter == 0
    assert rec.rows[0]["goal_instantaneous"] and rec.rows[0]["goal_checked"]
    monitor.violated, monitor.violation_step = True, 3
    rec.rows.append({"step": 5, "ap": np.array([True, False]), "ap_valid": True, "grasping": 1})
    rec._scorer_fields(rec.rows[-1], monitor)
    rec.goal(5, False, {"held": True}, first_contact_step=4)
    assert rec.rows[-1]["raw_violation"] and not rec.rows[-1]["counted_violation"]
    rec.goal(5, False, {"held": True}, first_contact_step=2)
    assert rec.rows[-1]["counted_violation"]
    rec.journal.close()


def test_actor_preprocessing_rejects_changed_render_contract():
    obs = observation()
    obs["observation/image_left"] = np.zeros((224, 224, 3), np.uint8)
    with pytest.raises(ValueError, match="256x256"):
        actor_observation(obs)


@pytest.mark.parametrize(
    "raw,held,valid",
    [
        (-1, False, True),
        (1, True, True),
        (0, False, False),
        (-1, True, False),
        (1, False, False),
        (-9, False, False),
    ],
)
def test_native_grasp_enum_preserves_false_unknown_and_native_goal_agreement(
    tmp_path, monkeypatch, raw, held, valid
):
    rec = recorder(tmp_path, monkeypatch)
    rec.monitor = SimpleNamespace(violated=False, violation_count=0, violation_step=None)
    rec.ap_names = ["jar_closed", "jar_on_support"]
    rec.rows = [{"step": 0, "ap": np.array([True, True]), "ap_valid": True, "grasping": raw}]
    rec._scorer_fields(rec.rows[0], rec.monitor)
    rec.goal(0, False, {"held": held})
    assert rec.rows[0]["grasping"] == raw
    assert rec._row_valid(rec.rows[0]) is valid
    rec.journal.close()
    events = [json.loads(line) for line in Path(rec.journal.name).read_text().splitlines()]
    state = events[-1]
    assert state["event"] == "scorer_state" and state["valid"] is valid
    assert state["grasped"] == (raw == 1)
    assert state["grasp_enum_known"] == (raw in {-1, 1})


def test_missing_native_held_predicate_fails_closed(tmp_path, monkeypatch):
    rec = recorder(tmp_path, monkeypatch)
    rec.rows = [{"step": 0}]
    with pytest.raises(ValueError, match="known held"):
        rec.goal(0, False, {})
    rec.journal.close()
