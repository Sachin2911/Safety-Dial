"""Independent saved-evidence audit; no simulator instances or model inference."""

import collections
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/adapter-pilot-20261101/run-v1"
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.pilot_common import digest, save, score  # noqa: E402


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
        assert execution["status"] == "complete"
        assert row["resources"] == execution["resources"]
        resources.update(row["resources"])
        with np.load(folder / "trace.npz") as trace:
            n = len(trace["actions"])
            assert n == row["resources"]["environment_actions"] <= 200
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
                    with np.load(OUT / "initialization.npz") as checkpoint:
                        parameters = checkpoint["neutral"].copy()
                elif row["arm"] == "untrained":
                    with np.load(OUT / "untrained.npz") as checkpoint:
                        parameters = checkpoint["parameters"].copy()
                else:
                    if row["stage"] == "training":
                        candidate_id = row["id"].split("_", 1)[0]
                    else:
                        selection = json.loads((OUT / "selection.json").read_text())
                        candidate_id = selection["nominees"][row["arm"]]["id"]
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
    assert resources["environment_resets"] <= 592
    assert resources["environment_actions"] <= 118400
    for pair in json.loads((OUT / "parity.json").read_text())["pairs"]:
        with np.load(OUT / "episodes" / pair[0] / "trace.npz") as a:
            with np.load(OUT / "episodes" / pair[1] / "trace.npz") as b:
                for key in ["actions", "states", "frames", "hit", "rewards"]:
                    np.testing.assert_array_equal(a[key], b[key])
    for candidate in summary["candidates"]:
        episodes = [r for r in rows.values() if r["id"].startswith(candidate["id"] + "_")]
        assert len(episodes) == 8
        for key, value in score(episodes).items():
            assert candidate[key] == value
        with np.load(OUT / "initialization.npz") as initial:
            with np.load(OUT / "candidates" / (candidate["id"] + ".npz")) as checkpoint:
                expected = initial["neutral"] + initial["scales"] * checkpoint["genotype"].astype(
                    np.float32
                )
                np.testing.assert_array_equal(checkpoint["parameters"], expected)
    if summary["status"] == "complete":
        assert len(rows) == 591
        selected = json.loads((OUT / "selection.json").read_text())["nominees"]
        for optimizer, candidate in selected.items():
            candidates = [r for r in summary["candidates"] if r["optimizer"] == optimizer]
            assert len(candidates) == 32
            assert max(candidates, key=lambda r: r["score"]) == candidate
        previous_mean = np.zeros(len(expected))
        for generation in range(4):
            with (OUT / f"cma-state-{generation:02d}.pkl").open("rb") as stream:
                state = pickle.load(stream)
            assert state.countiter == generation + 1
            assert state.countevals == 8 * (generation + 1)
            assert np.isfinite(state.mean).all() and np.isfinite(state.sigma)
            assert not np.array_equal(state.mean, previous_mean)
            previous_mean = state.mean.copy()
        assert (OUT / "selection.json").stat().st_mtime < min(
            (OUT / "episodes" / r["id"] / "spec.json").stat().st_mtime
            for r in rows.values()
            if r["stage"] == "evaluation"
        )
        for arm, metrics in summary["evaluation"].items():
            assert metrics == score(
                [r for r in rows.values() if r["stage"] == "evaluation" and r["arm"] == arm]
            )
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


if __name__ == "__main__":
    main()
