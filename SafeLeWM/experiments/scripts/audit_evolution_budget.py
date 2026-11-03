"""Independent saved-evidence audit; no simulator instances or model inference."""

import collections
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/evolution-budget-20261102/run-v1"
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.pilot_common import digest, save  # noqa: E402
from helpers.replication_budget import eligible, episode_status  # noqa: E402


def distance(a, b, c):
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    delta = b - a
    denominator = np.dot(delta, delta)
    fraction = np.clip(np.dot(c - a, delta) / denominator, 0, 1) if denominator else 0
    return np.linalg.norm(a + fraction * delta - c)


def main():
    import torch
    import socket

    assert socket.gethostname() == "8cf3a706cdd1"

    from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter

    torch.set_num_threads(4)
    actor = RecurrentActionAdapter(AdapterSpec(396, 2)).eval()
    replay_calls = 0
    config = json.loads((OUT / "sources/configs/evolution-budget/run.json").read_text())
    panel = json.loads((OUT / "sources/configs/evolution-budget/panel.json").read_text())
    mappings = build_mappings(config)
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
                stochastic = row["stage"] == "ppo-collection" or (
                    row["stage"] == "engineering" and row["arm"] == "ppo"
                )
                if stochastic:
                    from helpers.ppo_adapter import CoordinatePolicy

                    rid = row["repeat"]
                    policy = CoordinatePolicy(mappings[rid], 1)
                    if rid == "engineering":
                        checkpoint_path = OUT / "initialization/engineering-policy.pt"
                    else:
                        iteration = int(row["candidate"].rsplit("-", 1)[1])
                        checkpoint_path = (
                            OUT / "searches" / row["search"] / f"policy-{iteration:05d}.pt"
                        )
                    policy.load_state_dict(
                        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                    )
                    generator = torch.Generator().manual_seed(spec["sampling_seed"])
                    assert trace["raw_logits"].shape == (n, 4)
                    assert trace["values"].shape == trace["log_probability"].shape == (n,)
                    parameters = policy.export()
                elif row["arm"] == "neutral":
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
                        selection = json.loads(
                            (OUT / f"selection-{row['repeat']}.json").read_text()
                        )
                        candidate_id = selection["nominees"][row["search"]]["id"]
                        assert row["candidate"] == candidate_id
                    with np.load(OUT / "candidates" / (candidate_id + ".npz")) as checkpoint:
                        parameters = checkpoint["parameters"].copy()
                actor.load_vector(parameters)
                state = actor.initial_state()
                with torch.inference_mode():
                    for step in range(n):
                        inputs = torch.from_numpy(trace["inputs"][step])[None]
                        base = torch.from_numpy(trace["base_actions"][step])[None]
                        if stochastic:
                            action, state, detail = policy.sample(inputs, state, base, generator)
                            for field in ["raw_logits", "log_probability", "values"]:
                                np.testing.assert_array_equal(
                                    detail[field][0].numpy(), trace[field][step]
                                )
                        else:
                            action, state, detail = actor(inputs, state, base)
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
    assert resources["environment_actions"] <= summary["host"]["action_cap"]
    assert len(list((OUT / "episodes").glob("*/spec.json"))) == len(rows)
    assert len(list((OUT / "episodes").glob("*/execution.json"))) == len(rows)
    for case in panel["training"]:
        pair = [
            f"parity-{arm}_{case['id']}_{panel['training_seed']}" for arm in ["native", "neutral"]
        ]
        with (
            np.load(OUT / "episodes" / pair[0] / "trace.npz") as a,
            np.load(OUT / "episodes" / pair[1] / "trace.npz") as b,
        ):
            for field in ["actions", "states", "frames", "hit", "rewards"]:
                np.testing.assert_array_equal(a[field], b[field])
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


def build_mappings(config):
    from helpers.mechanism import SearchMap
    from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter
    mappings = {}
    for repeat in config["repeats"]:
        actor = RecurrentActionAdapter(AdapterSpec(396, 2, initialization_seed=repeat["initialization_seed"]))
        with np.load(OUT / "initialization" / (repeat["id"]+".npz")) as init:
            np.testing.assert_array_equal(init["neutral"], actor.parameters_vector())
            np.testing.assert_array_equal(init["scales"], actor.coordinate_scales())
            mapping = SearchMap(init["neutral"], init["scales"], "head")
        with np.load(OUT / "initialization" / (repeat["id"]+"-untrained.npz")) as init:
            z = np.random.default_rng(repeat["untrained_seed"]).normal(0, .5, 132)
            np.testing.assert_array_equal(z, init["genotype"])
            np.testing.assert_array_equal(mapping.decode(z), init["parameters"])
        mappings[repeat["id"]] = mapping
    return mappings


def audit_searches(summary, rows):
    from helpers.mechanism import objective
    config = json.loads((OUT / "sources/configs/evolution-budget/run.json").read_text())
    panel = json.loads((OUT / "sources/configs/evolution-budget/panel.json").read_text())
    mappings = build_mappings(config)
    training = [r for r in rows.values() if r["stage"] == "training"]
    candidates = {c["id"]: c for c in summary["candidates"]}
    assert len(candidates) == len(summary["candidates"])
    for cid, candidate in candidates.items():
        episodes = [r for r in training if r["candidate"] == cid]
        assert len(episodes) == candidate["episodes"]
        assert eligible(episodes) == candidate["eligible"]
        assert sum(r["resources"]["environment_actions"] for r in episodes) == candidate["actions"]
        assert candidate["actions"] <= candidate["quota"]
        assert [r["case"] for r in sorted(episodes, key=lambda r: next(
            i for i,c in enumerate(panel["training"]) if c["id"]==r["case"]))] == [
            c["id"] for c in panel["training"][:len(episodes)]]
        if candidate["eligible"]:
            for key, value in objective(episodes, candidate["arm"]).items():
                assert candidate[key] == value
        else:
            assert candidate["score"] is None
    groups = {}
    for repeat in config["repeats"]:
        rid = repeat["id"]
        lock = OUT / ("selection-"+rid+".json")
        selected = json.loads(lock.read_text())
        development = [r for r in rows.values() if r["stage"] == "development" and r["repeat"] == rid]
        assert len(development) == 144
        assert lock.stat().st_mtime <= min((OUT / "episodes" / r["id"] / "spec.json").stat().st_mtime
                                          for r in development)
        assert selected["locked_at"] < min((OUT / "episodes" / r["id"] / "result.json").stat().st_mtime
                                           for r in development)
        for arm in ["cma8", "cma4", "random"]:
            key = f"{rid}-head-{arm}"
            folder = OUT / "searches" / key
            entries = [c for c in summary["candidates"] if c["search"] == key]
            assert max((c for c in entries if c["eligible"]),key=lambda c:c["score"]) == selected["nominees"][key]
            result = json.loads((folder / "complete.json").read_text())
            population = 4 if arm == "cma4" else 8
            assert result == summary["searches"][key] and result["population"] == population
            assert sum(c["actions"] for c in entries) == result["actions"] == 40000
            spent, seen = 0, []
            for path in sorted(folder.glob("dispatch-*.json")):
                dispatch = json.loads(path.read_text())
                assert spent == dispatch["spent_before"]
                for cid, actual, quota in zip(dispatch["ids"],dispatch["actuals"],dispatch["quotas"],strict=True):
                    assert candidates[cid]["actions"] == actual <= quota
                spent += sum(dispatch["actuals"])
                assert spent == dispatch["spent_after"] <= 40000
                seen.extend(dispatch["ids"])
            assert spent == 40000 and seen == [c["id"] for c in entries]
            updates = 0
            for path in sorted(folder.glob("generation-*.json")):
                generation = json.loads(path.read_text())
                g = generation["generation"]
                asked = [f"{key}-{g*population+i:05d}" for i in range(population)]
                assert asked == generation["asked"]
                for cid in asked:
                    with np.load(OUT / "candidates" / (cid+".npz")) as arrays:
                        np.testing.assert_array_equal(mappings[rid].decode(arrays["genotype"]), arrays["parameters"])
                evaluated = [candidates[cid] for cid in asked if cid in candidates]
                full = len(evaluated) == population and all(c["eligible"] for c in evaluated)
                assert full == generation["complete"]
                assert generation["scores"] == [c["score"] for c in evaluated if c["eligible"]]
                state_path = folder / f"state-{g:04d}.pkl"
                if arm != "random" and full:
                    updates += 1
                    with state_path.open("rb") as stream:
                        state = pickle.load(stream)
                    assert state.countiter == updates and state.countevals == updates*population
                    assert type(state.adapt_sigma).__name__ == "CMAAdaptSigmaCSA"
                else:
                    assert not state_path.exists()
                assert generation["updates"] == updates
            assert updates == result["updates"]
        for arm in ["untrained", "cma8", "cma4", "random"]:
            subset = [r for r in development if r["arm"] == arm]
            assert len(subset) == 36 and {(r["case"],r["seed"]) for r in subset} == {
                (c["id"],seed) for c in panel["evaluation"] for seed in panel["evaluation_seeds"]}
            groups[f"{rid}-{arm}"] = dict(n=36, **{metric:sum(r[metric] for r in subset)
                for metric in ["safe_completion","success","violation","unfinished","engaged"]})
    native = [r for r in rows.values() if r["stage"] == "development" and r["arm"] == "native"]
    assert len(native) == 36
    groups["shared-native"] = dict(n=36, **{metric:sum(r[metric] for r in native)
        for metric in ["safe_completion","success","violation","unfinished","engaged"]})
    assert len([r for r in rows.values() if r["stage"] == "development"]) == 324
    assert len([r for r in rows.values() if r["stage"] == "engineering"]) == 16
    gate = all(groups[f"{rid}-cma4"]["safe_completion"] >= groups[f"{rid}-{control}"]["safe_completion"]+2
               for rid in ["E0","E1"] for control in ["cma8","random"])
    gate = gate and all(groups[f"{rid}-cma4"]["violation"] <= groups[f"{rid}-cma8"]["violation"]
                        for rid in ["E0","E1"])
    save(OUT / "analysis.json", dict(groups=groups, development_gate_passed=gate,
        final_test=False, resources=summary["resources"], seconds=summary["seconds"]))


if __name__ == "__main__":
    main()
