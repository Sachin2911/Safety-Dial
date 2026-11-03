"""Frozen AP evaluator and existing-wrapper comparison; rented-host execution only."""

import argparse
import collections
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/pusht-ap-reproduction-20261102"
REF = ROOT / "assets/ap-reference/code/ap"
DERIVED = OUT / "derived"
REVISION = "984327424ca7a2e52c75a8279197750d9d0b7c8e"
sys.path[:0] = [str(REF), str(ROOT / "experiments"), str(ROOT / "experiments/scripts")]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def asset():
    return dict(
        checkpoint=ROOT / "assets/pusht/weights.pt",
        dataset=ROOT / "data/pusht_expert_train.h5",
        latent_cache=OUT / "latents.npy", state_key="state",
    )


def guard():
    start = json.loads((OUT / "setup-launch.json").read_text())["started"]
    if time.time() > start + 14400:
        raise TimeoutError("Frozen four-hour task limit reached; rental stays running")
    if shutil.disk_usage(ROOT).free < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB free; rental stays running")


def verify_sources():
    frozen = json.loads((OUT / "freeze.json").read_text())
    for name, sha in frozen["sources"].items():
        assert digest(ROOT / name) == sha, name
    return frozen


def prepare():
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    import torch
    from anchored_planning import runtime

    assert not (OUT / "freeze.json").exists()
    guard()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REF, text=True
    ).strip()
    assert revision == REVISION
    assert not subprocess.check_output(["git", "diff", "HEAD", "--"], cwd=REF)
    pins = {}
    for name in ["native-common.txt", "native.txt"]:
        for line in (REF / "requirements" / name).read_text().splitlines():
            if "==" in line and not line.startswith("#"):
                package, version = line.strip().split("==")
                actual = importlib.metadata.version(package)
                assert actual.split("+")[0] == version, (package, actual, version)
                pins[package] = actual
    assert torch.version.cuda == "12.6"
    runtime.check_simulator_versions("pusht")
    protocol = json.loads((REF / "protocols/main.json").read_text())
    queries = protocol["studies"]["pusht"]["confirm"]
    train = protocol["bank_train_episodes"]["pusht"]
    assert len(queries) == 128 and len(train) == len(set(train)) == 16688
    assert not set(train) & {q["episode"] for q in queries}
    protocol["arms"] = ["gaussian_observed"]
    protocol["conditions"] = ["clean"]
    save(DERIVED / "protocols/main.json", protocol)
    shutil.copyfile(REF / "protocols/scalers.json", DERIVED / "protocols/scalers.json")
    original = json.loads((REF / "protocols/main.json").read_text())
    assert {k: v for k, v in protocol.items() if k not in ["arms", "conditions"]} == {
        k: v for k, v in original.items() if k not in ["arms", "conditions"]
    }
    assets = asset()
    assert digest(assets["checkpoint"]) == (
        "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607"
    )
    with h5py.File(assets["dataset"], "r") as data:
        rows = len(data["pixels"])
        assert rows == 2336736
        lengths = np.asarray(data["ep_len"], np.int64)
        offsets = np.asarray(data["ep_offset"], np.int64)
        assert int(lengths.sum()) == rows and len(lengths) == 18685
        np.testing.assert_array_equal(offsets[1:], np.cumsum(lengths)[:-1])
        for q in queries:
            assert q["global_goal"] == q["global_start"] + 140
            assert offsets[q["episode"]] + q["local_start"] == q["global_start"]
            assert q["global_goal"] < offsets[q["episode"]] + lengths[q["episode"]]
    source_paths = [
        ROOT / "PushTAPReproduction.md", Path(__file__),
        ROOT / "experiments/scripts/setup_pusht_ap_reproduction.py",
        ROOT / "experiments/scripts/audit_pusht_ap_reproduction.py",
        ROOT / "experiments/scripts/launch_pusht_ap_reproduction.py",
        ROOT / "experiments/scripts/manip_pipeline.py",
        ROOT / "experiments/helpers/manip.py",
        ROOT / "experiments/helpers/qualification.py",
        *sorted((REF / "anchored_planning").glob("*.py")),
        REF / "protocols/main.json", REF / "protocols/scalers.json",
        DERIVED / "protocols/main.json", DERIVED / "protocols/scalers.json",
    ]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    for name in sources:
        dest = OUT / "sources" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dest)
    save(OUT / "freeze.json", dict(
        utc=time.time(), sources=sources, reference_revision=revision,
        dependencies=pins, python=sys.version, gpu=torch.cuda.get_device_name(),
        hardware_deviation="RTX 4090 instead of H20",
        python_deviation="Python 3.11.9 instead of 3.11.14; same minor version",
        data_sha256=digest(assets["dataset"]), checkpoint_sha256=digest(assets["checkpoint"]),
        rows=rows, queries=queries, memory_episodes=train, shards=8,
        reset_cap=262, action_cap=72260, rental_shutdown="manual",
    ))
    assert not assets["latent_cache"].exists()
    partial = assets["latent_cache"].with_suffix(".partial.npy")
    assert not partial.exists()
    array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=(rows, 192))
    array[:] = np.nan
    array.flush()
    save(OUT / "prepared.json", dict(completed=time.time(), rows=rows))


def cache(shard):
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    import torch
    from anchored_planning.model import encode_retrieval, load_model

    frozen = verify_sources()
    assets = asset()
    model = load_model(assets["checkpoint"], torch.device("cuda:0"))
    rows = frozen["rows"]
    batches = (rows + 127) // 128
    start = (batches * shard // 8) * 128
    stop = min((batches * (shard + 1) // 8) * 128, rows)
    result = np.load(assets["latent_cache"].with_suffix(".partial.npy"), mmap_mode="r+")
    started = time.time()
    with h5py.File(assets["dataset"], "r") as data, torch.inference_mode():
        for i in range(start, stop, 128):
            guard()
            end = min(i + 128, stop)
            value = encode_retrieval(
                model, np.asarray(data["pixels"][i:end]), torch.device("cuda:0")
            )
            assert np.isfinite(value).all()
            result[i:end] = value.astype(np.float16)
            if (i - start) // 128 % 100 == 0:
                save(OUT / f"cache-progress-{shard}.json", dict(start=start, stop=stop, done=end))
    result.flush()
    save(OUT / f"cache-complete-{shard}.json", dict(
        start=start, stop=stop, encoder_frames=stop - start,
        seconds=time.time() - started, gpu=shard,
    ))


def cache_finish():
    import numpy as np

    records = [json.loads((OUT / f"cache-complete-{i}.json").read_text()) for i in range(8)]
    assert records[0]["start"] == 0
    assert all(a["stop"] == b["start"] for a, b in zip(records, records[1:]))
    path = asset()["latent_cache"]
    partial = path.with_suffix(".partial.npy")
    data = np.load(partial, mmap_mode="r")
    assert records[-1]["stop"] == len(data)
    assert np.isfinite(data).all()
    del data
    assert not path.exists()
    partial.rename(path)
    save(OUT / "cache-complete.json", dict(
        shards=records, sha256=digest(path), encoder_frames=2336736,
        batch_size=128, dtype="float16", retrieval_precision="bfloat16",
    ))


class Recorder:
    """Transparent observer; preserves step inputs/outputs and environment methods."""

    def __init__(self, env, path, ledger, goal):
        self.env, self.path, self.ledger, self.goal = env, path, ledger, goal
        self.states, self.actions, self.frames, self.frame_steps = [], [], [], []
        self.terminated, self.truncated = [], []
        self.closed = False

    def __getattr__(self, name):
        return getattr(self.env, name)

    def start(self):
        import numpy as np

        self.states.append(np.asarray(self.env._get_obs()).copy())
        self.frames.append(np.asarray(self.env.render()).copy())
        self.frame_steps.append(0)

    def step(self, action):
        import numpy as np

        guard()
        assert len(self.actions) < 280
        assert np.isfinite(action).all()
        assert np.all(action >= self.action_space.low) and np.all(action <= self.action_space.high)
        self.ledger.write(json.dumps(dict(event="action_attempt", action=action.tolist())) + "\n")
        self.ledger.flush()
        result = self.env.step(action)
        self.actions.append(action.copy())
        self.states.append(np.asarray(self.env._get_obs()).copy())
        self.terminated.append(bool(result[2]))
        self.truncated.append(bool(result[3]))
        if len(self.actions) % 5 == 0:
            self.frames.append(np.asarray(self.env.render()).copy())
            self.frame_steps.append(len(self.actions))
        self.ledger.write(json.dumps(dict(event="action_complete", step=len(self.actions))) + "\n")
        self.ledger.flush()
        return result

    def close(self):
        import numpy as np

        if self.closed:
            return
        self.closed = True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.path, states=np.asarray(self.states), actions=np.asarray(self.actions).reshape(-1, 2),
            frames=np.asarray(self.frames), frame_steps=np.asarray(self.frame_steps),
            goal=np.asarray(self.goal), terminated=np.asarray(self.terminated),
            truncated=np.asarray(self.truncated),
        )
        self.env.close()


def engineering():
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    from anchored_planning import environments as E
    from helpers import manip as M

    verify_sources()
    row = json.loads((OUT / "freeze.json").read_text())["queries"][0]
    with h5py.File(asset()["dataset"], "r") as data:
        states = {i: np.asarray(data["state"][i], np.float32)
                  for i in [row["global_start"], row["global_goal"]]}
        arrays = M.case_arrays("pusht", data, row["global_start"], row["global_goal"])
    context = dict(task="pusht", arrays=dict(states=states))
    traces = []
    with (OUT / "engineering-ledger.jsonl").open("x") as ledger:
        for index in range(4):
            ledger.write(json.dumps(dict(event="reset_attempt", index=index)) + "\n")
            ledger.flush()
            if index < 2:
                env, _, _, goal = E._make_live(context, row)
            else:
                env = M.make_env("pusht")
                M.reset_case(env, "pusht", arrays, row["environment_seed"])
                goal = arrays["goal_state"]
            if index % 2:
                env = Recorder(env, OUT / f"engineering-{index}.npz", ledger, goal)
                env.start()
            observations = [np.asarray(env._get_obs()).copy()]
            images = [env.render().copy()]
            try:
                for _ in range(5):
                    if index % 2 == 0:
                        ledger.write(json.dumps(dict(event="action_attempt")) + "\n")
                        ledger.flush()
                    env.step(np.array([0.1, -0.05], np.float32))
                    if index % 2 == 0:
                        ledger.write(json.dumps(dict(event="action_complete")) + "\n")
                        ledger.flush()
                    observations.append(np.asarray(env._get_obs()).copy())
                    images.append(env.render().copy())
            finally:
                env.close()
            traces.append((np.asarray(observations), np.asarray(images)))
    for obs, frames in traces[1:]:
        np.testing.assert_array_equal(obs, traces[0][0])
        np.testing.assert_array_equal(frames, traces[0][1])
    save(OUT / "engineering.json", dict(
        passed=True, resets=4, actions=20, transparent_observer_bit_exact=True,
        author_wrapper_reset_and_fixed_action_parity=True,
    ))


def evaluation(arm, shard):
    import h5py
    import hdf5plugin  # noqa: F401
    import numpy as np
    import torch
    from anchored_planning import environments as E, model, runtime
    from anchored_planning.gaussian_cem import query_seed
    from anchored_planning.observation_bank import ObservationBank

    frozen = verify_sources()
    assert json.loads((OUT / "engineering.json").read_text())["passed"]
    assert (OUT / "cache-complete.json").exists()
    folder = OUT / "run-v1" / arm
    folder.mkdir(parents=True, exist_ok=True)
    ledger = (folder / f"ledger-{shard}.jsonl").open("x")
    counters = collections.Counter()
    load = model.load_model

    def counted_load(*args, **kwargs):
        instance = load(*args, **kwargs)

        def encoder(_module, inputs):
            counters["encoder_frames"] += int(inputs[0].shape[0])

        def predictor(_module, inputs):
            counters["predictor_calls"] += 1
            counters["predictor_candidate_blocks"] += int(inputs[0].shape[0])

        instance.official.encoder.register_forward_pre_hook(encoder)
        instance.official.predictor.register_forward_pre_hook(predictor)
        return instance

    model.load_model = counted_load
    nearest = ObservationBank.nearest_topk

    def counted_nearest(self, *args, **kwargs):
        counters["retrieval_queries"] += 1
        return nearest(self, *args, **kwargs)

    ObservationBank.nearest_topk = counted_nearest
    started = time.time()
    status = "failed"
    try:
        if arm == "authors":
            make = E._make_live

            def observed_make(context, row):
                guard()
                ledger.write(json.dumps(dict(event="reset_attempt", query=row["record_id"])) + "\n")
                ledger.flush()
                env, obs, info, goal = make(context, row)
                observed = Recorder(env, folder / "physical" / (row["record_id"] + ".npz"),
                                    ledger, goal)
                observed.start()
                return observed, obs, info, goal

            E._make_live = observed_make
            args = SimpleNamespace(
                task="pusht", study="main", root=DERIVED, output=folder,
                shards=8, shard_index=shard, limit=None, device="cuda:0",
            )
            runtime.evaluate(args, asset())
        else:
            import manip_pipeline as P
            from helpers import manip as M

            _, native = M.native_model("pusht")
            make, reset = M.make_env, M.reset_case
            current_row = None

            def observed_env(task):
                return Recorder(
                    make(task), folder / "physical" / (current_row["record_id"] + ".npz"),
                    ledger, current_row["goal_state"],
                )

            def seeded_reset(env, task, arrays, _seed):
                ledger.write(json.dumps(dict(
                    event="reset_attempt", query=current_row["record_id"]
                )) + "\n")
                ledger.flush()
                result = reset(env, task, arrays, current_row["environment_seed"])
                env.start()
                return result

            M.make_env, M.reset_case = observed_env, seeded_reset
            totals = collections.Counter()
            deadline = time.monotonic() + max(
                0, json.loads((OUT / "setup-launch.json").read_text())["started"] + 14400 - time.time()
            )
            with h5py.File(asset()["dataset"], "r") as data, torch.inference_mode():
                bank = ObservationBank(
                    dict(lengths=np.asarray(data["ep_len"], np.int64),
                         offsets=np.asarray(data["ep_offset"], np.int64)),
                    np.load(asset()["latent_cache"], mmap_mode="r"),
                    np.asarray(frozen["memory_episodes"], np.int64),
                    5, np.ones(3, np.float32), torch.device("cuda:0"),
                )
                for ordinal, row in enumerate(frozen["queries"]):
                    if ordinal % 8 != shard:
                        continue
                    guard()
                    current_row = dict(row, goal_state=np.asarray(data["state"][row["global_goal"]]))
                    case = dict(id=row["record_id"], start_row=row["global_start"],
                                goal_row=row["global_goal"], horizon=140)
                    P.run_episode("pusht", folder, case, "ap", query_seed("pusht", ordinal),
                                  data, native, bank, totals, deadline)
            save(folder / f"wrapper-resources-{shard}.json", dict(totals))
        status = "complete"
    finally:
        ledger.close()
        save(folder / f"resources-{shard}.json", dict(
            status=status, counts=dict(counters), elapsed_seconds=time.time() - started,
            gpu=shard, arm=arm,
        ))


def main():
    assert socket.gethostname() == "c45db82630e6", "Remote experiment execution only"
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "cache", "cache-finish", "engineering", "eval"])
    parser.add_argument("--shard", type=int, default=0, choices=range(8))
    parser.add_argument("--arm", choices=["authors", "wrapper"])
    args = parser.parse_args()
    from anchored_planning.runtime import configure_runtime

    configure_runtime()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "cache":
        cache(args.shard)
    elif args.stage == "cache-finish":
        cache_finish()
    elif args.stage == "engineering":
        engineering()
    else:
        assert args.arm is not None
        evaluation(args.arm, args.shard)


if __name__ == "__main__":
    main()
