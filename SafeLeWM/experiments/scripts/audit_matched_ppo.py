"""Independent saved-evidence audit; no simulator instances or model inference."""

import collections
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/ppo-20261101/run-v1"
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
    config = json.loads((OUT / "sources/configs/matched-ppo/run.json").read_text())
    panel = json.loads((OUT / "sources/configs/matched-ppo/panel.json").read_text())
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
    from helpers.mechanism import SearchMap, projection_matrix
    from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter

    summary = json.loads((OUT / "summary.json").read_text())
    winner = summary["selected_representation"]
    mappings = {}
    for repeat in config["repeats"]:
        rid = repeat["id"]
        if rid not in summary["host"]["repeats"]:
            continue
        actor = RecurrentActionAdapter(
            AdapterSpec(396, 2, initialization_seed=repeat["initialization_seed"])
        )
        with np.load(OUT / "initialization" / (rid + ".npz")) as initial:
            np.testing.assert_array_equal(initial["neutral"], actor.parameters_vector())
            np.testing.assert_array_equal(initial["scales"], actor.coordinate_scales())
            projection = np.load(OUT / "initialization" / (rid + "-projection.npy"))
            np.testing.assert_array_equal(projection, projection_matrix(repeat["projection_seed"]))
            mappings[rid] = SearchMap(initial["neutral"], initial["scales"], winner, projection)
        with np.load(OUT / "initialization" / (rid + "-untrained.npz")) as initial:
            z = np.random.default_rng(repeat["untrained_seed"]).normal(
                0, 0.5, mappings[rid].dimension
            )
            np.testing.assert_array_equal(z, initial["genotype"])
            np.testing.assert_array_equal(mappings[rid].decode(z), initial["parameters"])
    with np.load(OUT / "initialization/engineering.npz") as initial:
        mappings["engineering"] = SearchMap(initial["neutral"], initial["scales"], "head")
    return mappings


def audit_searches(summary, rows):
    import torch
    from helpers.ppo_adapter import CoordinatePolicy

    config = json.loads((OUT / "sources/configs/matched-ppo/run.json").read_text())
    panel = json.loads((OUT / "sources/configs/matched-ppo/panel.json").read_text())
    maps = build_mappings(config)
    candidates = {c["id"]: c for c in summary["candidates"]}
    assert len(candidates) == len(summary["candidates"])
    training = [r for r in rows.values() if r["stage"] in ["training", "ppo-collection"]]
    assert sum(r["resources"]["environment_actions"] for r in training) == 120000 * len(
        summary["host"]["repeats"]
    )
    selection = json.loads((OUT / "representation-selection.json").read_text())
    assert selection["winner"] == summary["selected_representation"]
    assert (
        digest(OUT / "representation-selection.json") == summary["representation_selection_sha256"]
    )
    for candidate in candidates.values():
        episodes = [
            r for r in training if r["stage"] == "training" and r["candidate"] == candidate["id"]
        ]
        assert (
            len(episodes) == candidate["episodes"] and eligible(episodes) == candidate["eligible"]
        )
        assert (
            sum(r["resources"]["environment_actions"] for r in episodes)
            == candidate["actions"]
            <= candidate["quota"]
        )
        assert {r["case"] for r in episodes} == {
            c["id"] for c in panel["training"][: len(episodes)]
        }
        if candidate["eligible"]:
            for field, value in score(episodes).items():
                assert candidate[field] == value
        else:
            assert candidate["score"] is None
    for rid in summary["host"]["repeats"]:
        selected = json.loads((OUT / f"selection-{rid}.json").read_text())
        mapping = maps[rid]
        for arm in ["main", "random", "ppo"]:
            key = f"{rid}-{mapping.representation}-{arm}"
            folder = OUT / "searches" / key
            entries = [c for c in summary["candidates"] if c["search"] == key]
            assert (
                max((c for c in entries if c["eligible"]), key=lambda c: c["score"])
                == selected["nominees"][key]
            )
            complete = json.loads((folder / "complete.json").read_text())
            assert complete == summary["searches"][key]
            search_rows = [r for r in training if r["search"] == key]
            assert (
                sum(r["resources"]["environment_actions"] for r in search_rows)
                == complete["actions"]
                == 40000
            )
            if arm == "ppo":
                spent, updates = 0, 0
                seen = []
                for path in sorted(folder.glob("iteration-*.json")):
                    if path.name.endswith("-intent.json"):
                        continue
                    iteration = json.loads(path.read_text())
                    i = iteration["iteration"]
                    intent = json.loads((folder / f"iteration-{i:05d}-intent.json").read_text())
                    assert intent["spent_before"] == spent
                    collection = [rows[eid] for eid in iteration["collection_ids"]]
                    seen.extend(iteration["collection_ids"])
                    assert (
                        sum(r["resources"]["environment_actions"] for r in collection)
                        == iteration["collection_actions"]
                        == intent["collection_quota"]
                        == min(3200, 40000 - spent)
                    )
                    assert iteration["eligible_collection_ids"] == [
                        r["id"] for r in collection if r["status"] == "complete"
                    ]
                    spent += iteration["collection_actions"]
                    cid = f"{key}-{i:05d}"
                    checkpoint = torch.load(
                        folder / f"updated-{i:05d}.pt", weights_only=True, map_location="cpu"
                    )
                    policy = CoordinatePolicy(mapping, 1)
                    policy.load_state_dict(checkpoint["policy"])
                    with np.load(OUT / "candidates" / (cid + ".npz")) as arrays:
                        np.testing.assert_array_equal(policy.export(), arrays["parameters"])
                        np.testing.assert_array_equal(
                            policy.coordinates.detach().numpy(), arrays["genotype"]
                        )
                    if cid in candidates:
                        assert candidates[cid]["quota"] == min(1600, 40000 - spent)
                        assert candidates[cid]["actions"] == iteration["selection_actions"]
                    else:
                        assert iteration["selection_actions"] == 0 and spent == 40000
                    spent += iteration["selection_actions"]
                    assert spent == iteration["spent"] <= 40000
                    updates += int(iteration["update_performed"])
                    assert iteration["update_performed"] == bool(
                        iteration["eligible_collection_ids"]
                    )
                    assert iteration["updates"] == updates
                    assert all(np.isfinite(t.detach().numpy()).all() for t in policy.parameters())
                assert set(seen) == {r["id"] for r in search_rows if r["stage"] == "ppo-collection"}
                assert spent == 40000 and complete["updates"] == updates
            else:
                spent, seen = 0, []
                for path in sorted(folder.glob("dispatch-*.json")):
                    dispatch = json.loads(path.read_text())
                    assert dispatch["spent_before"] == spent
                    first = int(dispatch["ids"][0].rsplit("-", 1)[1])
                    assert dispatch["quotas"] == allocation(40000 - spent, 8 - first % 8)
                    assert dispatch["actuals"] == [
                        candidates[i]["actions"] for i in dispatch["ids"]
                    ]
                    assert dispatch["refunded"] == sum(dispatch["quotas"]) - sum(
                        dispatch["actuals"]
                    )
                    spent += sum(dispatch["actuals"])
                    assert spent == dispatch["spent_after"] <= 40000
                    seen.extend(dispatch["ids"])
                assert spent == 40000 and seen == [c["id"] for c in entries]
                updates = 0
                for path in sorted(folder.glob("generation-*.json")):
                    generation = json.loads(path.read_text())
                    g = generation["generation"]
                    asked = [f"{key}-{g * 8 + i:05d}" for i in range(8)]
                    assert generation["asked"] == asked
                    for cid in asked:
                        with np.load(OUT / "candidates" / (cid + ".npz")) as arrays:
                            np.testing.assert_array_equal(
                                mapping.decode(arrays["genotype"]), arrays["parameters"]
                            )
                    evaluated = [candidates[cid] for cid in asked if cid in candidates]
                    full = len(evaluated) == 8 and all(c["eligible"] for c in evaluated)
                    assert generation["complete"] == full
                    assert generation["scores"] == [c["score"] for c in evaluated if c["eligible"]]
                    state_path = folder / f"state-{g:04d}.pkl"
                    if arm == "main" and full:
                        updates += 1
                        with state_path.open("rb") as stream:
                            state = pickle.load(stream)
                        assert state.countiter == updates and state.countevals == updates * 8
                        assert type(state.adapt_sigma).__name__ == "CMAAdaptSigmaCSA"
                    else:
                        assert not state_path.exists()
                    assert generation["updates"] == updates
                assert updates == complete["updates"]
        evaluation = [r for r in rows.values() if r["stage"] == "evaluation" and r["repeat"] == rid]
        assert selected["locked_at"] <= min(
            (OUT / "episodes" / r["id"] / "spec.json").stat().st_mtime for r in evaluation
        )
        for arm in ["untrained", "main", "random", "ppo"]:
            group = [r for r in evaluation if r["arm"] == arm]
            assert len(group) == 72 and {(r["case"], r["seed"]) for r in group} == {
                (c["id"], s) for c in panel["evaluation"] for s in panel["evaluation_seeds"]
            }
    native = [r for r in rows.values() if r["stage"] == "evaluation" and r["arm"] == "native"]
    assert len(native) == 72
    assert len([r for r in rows.values() if r["stage"] == "engineering"]) == 20
    assert json.loads((OUT / "engineering.json").read_text())["passed"]
    engineering_actions = sum(
        r["resources"]["environment_actions"] for r in rows.values() if r["stage"] == "engineering"
    )
    assert engineering_actions <= 10000
    groups = {}
    for rid, arm in [(None, "native")] + [
        (r, a) for r in summary["host"]["repeats"] for a in ["untrained", "main", "random", "ppo"]
    ]:
        group = [
            r
            for r in rows.values()
            if r["stage"] == "evaluation" and r["repeat"] == rid and r["arm"] == arm
        ]
        groups[f"{rid or 'shared'}-{arm}"] = {
            "n": len(group),
            **{
                k: sum(r[k] for r in group)
                for k in ["safe_completion", "success", "violation", "unfinished", "engaged"]
            },
        }
    save(
        OUT / "analysis.json",
        dict(
            host=summary["host"],
            representation=summary["selected_representation"],
            groups=groups,
            resources=summary["resources"],
            seconds=summary["seconds"],
        ),
    )


if __name__ == "__main__":
    main()
