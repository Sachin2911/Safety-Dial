"""Independent saved-evidence audit; no simulator instances or model inference."""

import collections
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/replication-20261101/run-v1"
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.pilot_common import digest, save, score  # noqa: E402
from helpers.replication_budget import allocation, eligible, episode_status  # noqa: E402


def distance(a, b, c):
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    delta = b - a
    denominator = np.dot(delta, delta)
    fraction = np.clip(np.dot(c - a, delta) / denominator, 0, 1) if denominator else 0
    return np.linalg.norm(a + fraction * delta - c)


def main():
    import torch

    from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter

    torch.set_num_threads(4)
    actor = RecurrentActionAdapter(AdapterSpec(396, 2)).eval()
    replay_calls = 0
    summary = json.loads((OUT / "summary.json").read_text())
    freeze = json.loads((OUT / "freeze.json").read_text())
    for name, sha in freeze["sources"].items():
        assert digest(OUT / "sources" / name) == sha
        assert digest(ROOT / name) == sha
    memory_panel = json.loads((OUT / "memory-panel.json").read_text())
    allowed = set(memory_panel["memory_episodes"])
    assert not allowed.intersection(memory_panel["excluded_episodes"])
    # Map original frame rows to episodes without re-running any retrieval/model.
    import h5py

    with h5py.File(ROOT / "data/tworoom.h5", "r") as data:
        offsets = data["ep_offset"][:]
        lengths = data["ep_len"][:]
    resources = collections.Counter()
    rows = {}
    for path in sorted((OUT / "episodes").glob("*/result.json")):
        row = json.loads(path.read_text())
        folder = path.parent
        spec = json.loads((folder / "spec.json").read_text())
        execution = json.loads((folder / "execution.json").read_text())
        assert execution["status"] == row["status"]
        assert row["status"] in ["complete", "budget_exhausted"]
        assert row["resources"] == execution["resources"]
        resources.update(row["resources"])
        with np.load(folder / "trace.npz") as arrays:
            trace = {key: arrays[key] for key in arrays.files}
            n = len(trace["actions"])
            assert 0 < n == row["resources"]["environment_actions"] <= spec["action_limit"] <= 200
            assert row["action_limit"] == spec["action_limit"]
            assert row["status"] == episode_status(
                n, bool(trace["terminated"][-1] or trace["truncated"][-1]), spec["action_limit"]
            )
            native_success = np.linalg.norm(trace["states"][1:] - trace["goal"], axis=1) < 16
            np.testing.assert_array_equal(native_success, trace["terminated"])
            assert not np.any(trace["truncated"])
            assert row["repeat"] == spec["repeat"] and row["candidate"] == spec["candidate"]
            assert len(trace["states"]) == len(trace["frames"]) == n + 1
            np.testing.assert_array_equal(trace["states"][0], spec["case"]["initial"])
            assert np.isfinite(trace["actions"]).all() and np.max(np.abs(trace["actions"])) <= 1
            np.testing.assert_array_equal(
                trace["actions"], np.clip(trace["proposed_actions"], -1, 1)
            )
            hits = [
                distance(a, b, spec["case"]["hazard_centre"]) <= 18
                for a, b in zip(trace["states"][:-1], trace["states"][1:])
            ]
            np.testing.assert_array_equal(hits, trace["hit"])
            assert row["violation"] == any(hits)
            assert row["success"] == bool(trace["terminated"][-1])
            assert not np.any(trace["terminated"][:-1])
            assert not np.any(trace["truncated"][:-1])
            assert row["safe_completion"] == (row["success"] and not row["violation"])
            assert row["unfinished"] == (not row["success"])
            length = np.linalg.norm(np.diff(trace["states"], axis=0), axis=1).sum()
            assert row["engaged"] == bool(length > 1)
            assert np.isclose(row["path_length"], length)
            witness = row["arm"] in ["safe", "risky"]
            adapter = row["resources"].get("adapter_calls", 0) > 0
            if adapter:
                assert trace["inputs"].shape == (n, 396)
                assert trace["hidden"].shape == (n, 32)
                assert row["resources"]["adapter_calls"] == n
                np.testing.assert_array_equal(
                    trace["proposed_actions"],
                    trace["gains"] * trace["base_actions"] + trace["residuals"],
                )
                assert np.max(np.abs(trace["residuals"])) <= 1
                assert np.min(trace["gains"]) >= 0 and np.max(trace["gains"]) <= 2
                np.testing.assert_allclose(
                    trace["inputs"][:, 384:386], trace["states"][:-1] / 112 - 1, atol=1e-7
                )
                np.testing.assert_array_equal(trace["inputs"][:, 389:391], trace["base_actions"])
                np.testing.assert_array_equal(trace["inputs"][0, 391:395], np.zeros(4))
                np.testing.assert_array_equal(trace["inputs"][1:, 391:393], trace["actions"][:-1])
                np.testing.assert_allclose(
                    trace["inputs"][1:, 393:395],
                    np.diff(trace["states"], axis=0)[:-1] / 5,
                    atol=1e-7,
                )
                np.testing.assert_allclose(
                    trace["inputs"][:, 386:388],
                    (
                        np.asarray(spec["case"]["hazard_centre"], dtype=np.float32)
                        - trace["states"][:-1]
                    )
                    / 224,
                    atol=1e-7,
                )
                np.testing.assert_allclose(trace["inputs"][:, 388], 18 / 224, atol=1e-7)
                np.testing.assert_allclose(trace["inputs"][:, 395], np.arange(n) / 200, atol=1e-7)
                np.testing.assert_array_equal(
                    trace["inputs"][:, 192:384],
                    np.repeat(trace["inputs"][0:1, 192:384], n, axis=0),
                )
                if row["arm"] == "neutral":
                    with np.load(OUT / "initialization" / (row["repeat"] + ".npz")) as checkpoint:
                        parameters = checkpoint["neutral"].copy()
                elif row["arm"] == "untrained":
                    with np.load(
                        OUT / "initialization" / (row["repeat"] + "-untrained.npz")
                    ) as checkpoint:
                        parameters = checkpoint["parameters"].copy()
                else:
                    if row["stage"] == "training":
                        candidate_id = row["candidate"]
                    else:
                        selection = json.loads((OUT / "selection.json").read_text())
                        candidate_id = selection["nominees"][f"{row['repeat']}-{row['arm']}"]["id"]
                        assert row["candidate"] == candidate_id
                    with np.load(OUT / "candidates" / (candidate_id + ".npz")) as checkpoint:
                        parameters = checkpoint["parameters"].copy()
                actor.load_vector(parameters)
                state = actor.initial_state()
                with torch.inference_mode():
                    for step in range(n):
                        action, state, detail = actor(
                            torch.from_numpy(trace["inputs"][step])[None],
                            state,
                            torch.from_numpy(trace["base_actions"][step])[None],
                        )
                        np.testing.assert_array_equal(action[0].numpy(), trace["actions"][step])
                        np.testing.assert_array_equal(state[0].numpy(), trace["hidden"][step])
                        np.testing.assert_array_equal(
                            detail["proposed"][0].numpy(), trace["proposed_actions"][step]
                        )
                        replay_calls += 1
            elif not witness:
                np.testing.assert_array_equal(trace["actions"], trace["base_actions"])
        decisions = json.loads((folder / "planning.json").read_text())
        assert len(decisions) == row["resources"].get("replans", 0)
        if not witness:
            assert len(decisions) == (n + 4) // 5
            assert [d["step"] for d in decisions] == list(range(0, n, 5))
            assert row["resources"]["model_batch_calls"] == 31 * len(decisions)
            assert row["resources"]["candidate_model_steps"] == 9002 * len(decisions)
            for decision in decisions:
                source = decision["source"]
                episode = int(np.searchsorted(offsets, source, side="right") - 1)
                assert episode in allowed
                assert source + max(5, 100 - decision["step"]) < offsets[episode] + lengths[episode]
                assert decision["target_index"] == source + 5
        rows[row["id"]] = row
    assert len(rows) == len(summary["results"])
    for row in summary["results"]:
        assert row == rows[row["id"]]
    assert dict(resources) == summary["resources"]
    assert summary["status"] == "complete"
    assert resources["environment_actions"] <= 321608
    assert len(list((OUT / "episodes").glob("*/spec.json"))) == len(rows)
    assert len(list((OUT / "episodes").glob("*/execution.json"))) == len(rows)
    for pair in json.loads((OUT / "parity.json").read_text())["pairs"]:
        with np.load(OUT / "episodes" / pair[0] / "trace.npz") as a:
            with np.load(OUT / "episodes" / pair[1] / "trace.npz") as b:
                for key in ["actions", "states", "frames", "hit", "rewards"]:
                    np.testing.assert_array_equal(a[key], b[key])
    audit_searches(summary, rows)
    save(
        OUT / "audit.json",
        dict(
            passed=True,
            episodes=len(rows),
            resources=dict(resources),
            status=summary["status"],
            source_bindings=len(freeze["sources"]),
            audit_adapter_replay_calls=replay_calls,
        ),
    )
    print(json.dumps(dict(passed=True, episodes=len(rows), status=summary["status"])))


def audit_searches(summary, rows):
    config = json.loads((OUT / "sources/configs/adapter-replication/run.json").read_text())
    panel = json.loads((OUT / "sources/configs/adapter-replication/panel.json").read_text())
    selected = json.loads((OUT / "selection.json").read_text())
    candidates = {c["id"]: c for c in summary["candidates"]}
    training = [r for r in rows.values() if r["stage"] == "training"]
    assert sum(r["resources"]["environment_actions"] for r in training) == 240000
    assert all(r["candidate"] in candidates for r in training)
    for candidate in candidates.values():
        episodes = [r for r in training if r["candidate"] == candidate["id"]]
        assert candidate["episodes"] == len(episodes)
        assert candidate["eligible"] == eligible(episodes)
        assert candidate["actions"] == sum(r["resources"]["environment_actions"] for r in episodes)
        assert candidate["actions"] <= candidate["quota"]
        assert [
            r["case"]
            for r in sorted(
                episodes, key=lambda r: [c["id"] for c in panel["training"]].index(r["case"])
            )
        ] == [c["id"] for c in panel["training"][: len(episodes)]]
        if candidate["eligible"]:
            for key, value in score(episodes).items():
                assert candidate[key] == value
        else:
            assert candidate["score"] is None
        rid = candidate["repeat"]
        with np.load(OUT / "initialization" / (rid + ".npz")) as initial:
            with np.load(OUT / "candidates" / (candidate["id"] + ".npz")) as checkpoint:
                expected = initial["neutral"] + initial["scales"] * checkpoint["genotype"].astype(
                    np.float32
                )
                np.testing.assert_array_equal(expected, checkpoint["parameters"])
    for repeat in config["repeats"]:
        rid = repeat["id"]
        for optimizer in ["cma", "random"]:
            key = f"{rid}-{optimizer}"
            entries = [
                c
                for c in summary["candidates"]
                if c["repeat"] == rid and c["optimizer"] == optimizer
            ]
            assert (
                max((c for c in entries if c["eligible"]), key=lambda c: c["score"])
                == (selected["nominees"][key])
            )
            folder = OUT / "searches" / key
            complete = json.loads((folder / "complete.json").read_text())
            assert complete == summary["searches"][key]
            assert sum(c["actions"] for c in entries) == complete["actions"] == 40000
            spent, seen = 0, []
            for path in sorted(folder.glob("dispatch-*.json")):
                dispatch = json.loads(path.read_text())
                assert dispatch["spent_before"] == spent
                first = int(dispatch["ids"][0].rsplit("-", 1)[1])
                assert dispatch["quotas"] == allocation(40000 - spent, 8 - first % 8)
                assert dispatch["actuals"] == [candidates[i]["actions"] for i in dispatch["ids"]]
                assert dispatch["refunded"] == sum(dispatch["quotas"]) - sum(dispatch["actuals"])
                spent += sum(dispatch["actuals"])
                assert spent == dispatch["spent_after"] <= 40000
                seen.extend(dispatch["ids"])
            assert spent == 40000
            assert seen == [c["id"] for c in entries]
            updates = 0
            for path in sorted(folder.glob("generation-*.json")):
                generation = json.loads(path.read_text())
                g = generation["generation"]
                asked = [f"{key}-{g * 8 + i:05d}" for i in range(8)]
                assert generation["asked"] == asked
                for cid in asked:
                    assert (OUT / "candidates" / (cid + ".npz")).exists()
                evaluated = [candidates[cid] for cid in asked if cid in candidates]
                assert generation["evaluated"] == len(evaluated)
                full = len(evaluated) == 8 and all(c["eligible"] for c in evaluated)
                assert generation["complete"] == full
                assert generation["scores"] == [c["score"] for c in evaluated if c["eligible"]]
                state_path = folder / f"state-{g:04d}.pkl"
                if optimizer == "cma" and full:
                    updates += 1
                    with state_path.open("rb") as stream:
                        state = pickle.load(stream)
                    assert state.countiter == updates and state.countevals == updates * 8
                    assert type(state.adapt_sigma).__name__ == "CMAAdaptSigmaCSA"
                    assert state.opts["CMA_diagonal"] is True and state.opts["CMA_mirrors"] == 0
                    assert np.isfinite(state.mean).all() and np.isfinite(state.sigma)
                else:
                    assert not state_path.exists()
                assert generation["updates"] == updates
            assert updates == complete["updates"]
    evaluation = [r for r in rows.values() if r["stage"] == "evaluation"]
    assert len(evaluation) == 360
    assert selected["locked_at"] <= min(
        (OUT / "episodes" / r["id"] / "spec.json").stat().st_mtime for r in evaluation
    )
    groups = [(None, "native")] + [
        (r["id"], arm) for r in config["repeats"] for arm in ["untrained", "cma", "random"]
    ]
    expected = {(c["id"], s) for c in panel["evaluation"] for s in panel["evaluation_seeds"]}
    for rid, arm in groups:
        group = [r for r in evaluation if r["repeat"] == rid and r["arm"] == arm]
        assert len(group) == 36 and {(r["case"], r["seed"]) for r in group} == expected
        assert all(r["status"] == "complete" for r in group)
        assert summary["evaluation"][f"{rid or 'shared'}-{arm}"] == score(group)
    witnesses = [r for r in rows.values() if r["stage"] == "witness"]
    assert len(witnesses) == 24
    assert all(r["success"] and r["violation"] == (r["arm"] == "risky") for r in witnesses)
    assert len([r for r in rows.values() if r["stage"] == "engineering"]) == 25
    import replicate_adapter as runner

    runner.OUT = OUT
    for probe in json.loads((OUT / "parity.json").read_text())["prefixes"]:
        runner.prefix_check(probe["full"], probe["short"], probe["limit"])


if __name__ == "__main__":
    main()
