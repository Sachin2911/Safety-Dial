"""Independent saved-evidence audit; no simulator instances or model inference."""

import collections
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/safety-dial-library-20261102/run-v1"
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.pilot_common import digest, save  # noqa: E402
from helpers.replication_budget import episode_status  # noqa: E402


def distance(a, b, c):
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    delta = b - a
    denominator = np.dot(delta, delta)
    fraction = np.clip(np.dot(c - a, delta) / denominator, 0, 1) if denominator else 0
    return np.linalg.norm(a + fraction * delta - c)


def main():
    import torch
    import socket

    assert socket.gethostname() == "075cb160c9b3"

    from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter

    torch.set_num_threads(4)
    actor = RecurrentActionAdapter(AdapterSpec(396, 2)).eval()
    replay_calls = 0
    config = json.loads((OUT / "sources/configs/safety-dial-library/run.json").read_text())
    panel = json.loads((OUT / "sources/configs/safety-dial-library/panel.json").read_text())
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
                    parameters = np.load(OUT / "initialization/engineering.npz")["neutral"].copy()
                else:
                    pid = spec["library_policy"]
                    parameters = np.load(OUT.parent / "weights" / (pid + ".npz"))[
                        "parameters"
                    ].copy()
                actor.load_vector(parameters)
                state = actor.initial_state()
                with torch.inference_mode():
                    for step in range(n):
                        inputs = torch.from_numpy(trace["inputs"][step])[None]
                        base = torch.from_numpy(trace["base_actions"][step])[None]
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
    audit_library(summary, rows, config, panel)
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


def audit_library(summary, rows, config, panel):
    import random

    sys.path.insert(0, str(ROOT / "experiments/scripts"))
    from prepare_dial_library import build_library

    base = OUT.parent
    inputs = base / "selection-inputs"
    analysis = json.loads((inputs / "analysis.json").read_text())
    summaries = [json.loads((inputs / f"summary-{i}.json").read_text()) for i in [0, 1]]
    policies, curves = build_library(analysis, summaries)
    library = json.loads((base / "library.json").read_text())
    assert policies == library["policies"] and curves == library["curves"]
    schedule = list(policies)
    random.Random(config["schedule_seed"]).shuffle(schedule)
    assert schedule == library["schedule"]
    assert len(policies) == 21
    for pid, weight in library["weights"].items():
        assert digest(base / "weights" / (pid + ".npz")) == weight["sha256"]
    specs = list((OUT / "episodes").glob("*/spec.json"))
    assert (OUT / "freeze.json").stat().st_mtime <= min(p.stat().st_mtime for p in specs)
    assert (base / "library.json").stat().st_mtime <= min(p.stat().st_mtime for p in specs)
    assert len(rows) == 772 and summary["resources"]["environment_resets"] == 772
    assert summary["resources"]["environment_actions"] <= config["action_cap"]
    engineering = [r for r in rows.values() if r["stage"] == "engineering"]
    development = [r for r in rows.values() if r["stage"] == "development"]
    assert len(engineering) == 16 and len(development) == 756
    expected = {(c["id"], seed) for c in panel["evaluation"] for seed in panel["evaluation_seeds"]}
    metrics = ["safe_completion", "success", "violation", "unfinished", "engaged"]
    groups = {}
    observed_order = []
    for row in summary["results"]:
        if row["stage"] == "development":
            spec = json.loads((OUT / "episodes" / row["id"] / "spec.json").read_text())
            pid = spec["library_policy"]
            if not observed_order or observed_order[-1] != pid:
                observed_order.append(pid)
    assert observed_order == schedule
    for pid, policy in policies.items():
        subset = []
        for row in development:
            spec = json.loads((OUT / "episodes" / row["id"] / "spec.json").read_text())
            if spec["library_policy"] != pid:
                continue
            for key in ["arm", "repeat", "candidate", "search"]:
                assert row[key] == policy[key]
            assert spec["case"] == next(c for c in panel["evaluation"] if c["id"] == row["case"])
            subset.append(row)
        assert len(subset) == 36
        assert {(r["case"], r["seed"]) for r in subset} == expected
        groups[pid] = dict(n=36, **{m: sum(r[m] for r in subset) for m in metrics})
    results = {}
    for key, selections in curves.items():
        points = []
        for selection in selections:
            candidate = selection["candidate"]
            if candidate is None:
                points.append(dict(epsilon=selection["epsilon"], available=False))
            else:
                values = groups[candidate["id"]]
                points.append(
                    dict(
                        epsilon=selection["epsilon"],
                        available=True,
                        candidate=candidate["id"],
                        **values,
                        empirical_target_exceeded=values["violation"] / 36 > selection["epsilon"],
                    )
                )
        available = [p for p in points if p["available"]]
        risks = [p["violation"] for p in available]
        safes = [p["safe_completion"] for p in available]
        results[key] = dict(
            points=points,
            nondecreasing_observed_risk=all(a <= b for a, b in zip(risks, risks[1:])),
            violation_span=max(risks) - min(risks) if risks else None,
            safe_completion_span=max(safes) - min(safes) if safes else None,
        )
    save(
        OUT / "analysis.json",
        dict(
            groups=groups,
            curves=results,
            final_test=False,
            calibrated_risk_claim=False,
            resources=summary["resources"],
            prior_acquisition_actions=360000,
            seconds=summary["seconds"],
        ),
    )


if __name__ == "__main__":
    main()
