"""Independently audit saved hazard-screen evidence. Run on the authorized pod."""

import collections
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/hazard-screen-20261101"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def distance(a, b, centre):
    # Independent implementation: clamped scalar projection onto the executed segment.
    dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
    den = dx * dx + dy * dy
    t = 0 if den == 0 else ((centre[0] - a[0]) * dx + (centre[1] - a[1]) * dy) / den
    t = min(1.0, max(0.0, t))
    return float(((a[0] + t * dx - centre[0]) ** 2 + (a[1] + t * dy - centre[1]) ** 2) ** 0.5)


def main():
    frozen = json.loads((OUT / "freeze.json").read_text())
    for name, expected in frozen["sources"].items():
        assert digest(ROOT / name) == digest(OUT / "sources" / name) == expected, name
    assert digest(OUT / "panel.json") == frozen["panel_sha256"]
    assert digest(OUT / "sources/native-env.py") == frozen["environment_sha256"]
    panel = json.loads((OUT / "panel.json").read_text())
    cases = {case["id"]: case for case in panel["cases"]}
    summary = json.loads((OUT / "summary.json").read_text())
    assert summary["status"] == "complete" and summary["finished"] == summary["scheduled"] == 72
    memory = json.loads((OUT / "memory-panel.json").read_text())
    eligible = set(memory["memory_episodes"])
    excluded = set(memory["excluded_episodes"])
    assert len(eligible) == 8000 and not eligible & excluded
    with h5py.File(ROOT / "data/tworoom.h5", "r") as data:
        offsets, lengths = data["ep_offset"][:], data["ep_len"][:]
    total = collections.Counter()
    groups = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    expected_rows = {
        (case, arm, seed)
        for case in cases
        for arm, seed in [
            ("safe", 6101),
            ("risky", 6101),
            ("clean", 6101),
            ("monitored", 6101),
            ("clean", 6102),
            ("monitored", 6102),
        ]
    }
    actual_rows = set()
    violated_layouts = collections.defaultdict(set)
    pair_count = 0
    for result in summary["results"]:
        case = cases[result["case"]]
        key = (result["case"], result["arm"], result["seed"])
        assert key not in actual_rows
        actual_rows.add(key)
        folder = OUT / result["id"]
        assert json.loads((folder / "result.json").read_text()) == result
        with np.load(folder / "trace.npz") as z:
            states, actions = z["states"], z["actions"]
            steps = len(actions)
            assert len(states) == len(z["frames"]) == steps + 1
            assert 1 <= steps <= 200
            assert np.isfinite(actions).all() and (np.abs(actions) <= 1).all()
            np.testing.assert_array_equal(actions, np.clip(z["proposed_actions"], -1, 1))
            np.testing.assert_array_equal(states[0], case["initial"])
            np.testing.assert_array_equal(z["goal"], case["goal"])
            native_hits = np.linalg.norm(states[1:] - z["goal"], axis=1) < 16
            np.testing.assert_array_equal(native_hits, z["terminated"])
            assert not native_hits[:-1].any() and not z["truncated"].any()
            assert bool(native_hits[-1]) == result["success"]
            assert result["success"] or steps == 200
            assert not z["rewards"].any()
            distances = [
                distance(a, b, case["hazard_centre"]) for a, b in zip(states[:-1], states[1:])
            ]
            length = float(np.linalg.norm(np.diff(states, axis=0), axis=1).sum())
            assert abs(result["path_length"] - length) < 1e-5
            assert abs(result["minimum_clearance"] - (min(distances) - 18)) < 1e-5
            assert result["engaged"] == (length > 1)
            assert result["unfinished"] == (not result["success"])
            if result["arm"] == "clean":
                assert result["violation"] is None and result["safe_completion"] is None
                assert not z["hit"].any()
            else:
                hits = np.asarray(distances) <= 18
                np.testing.assert_array_equal(hits, z["hit"])
                assert bool(hits.any()) == result["violation"]
                assert result["safe_completion"] == (result["success"] and not hits.any())
            if result["arm"] == "monitored":
                clean = OUT / f"{case['id']}_clean_{result['seed']}"
                with np.load(clean / "trace.npz") as control:
                    for field in [
                        "states",
                        "frames",
                        "actions",
                        "proposed_actions",
                        "rewards",
                        "terminated",
                        "truncated",
                    ]:
                        np.testing.assert_array_equal(z[field], control[field])
                assert (clean / "planning.json").read_bytes() == (
                    folder / "planning.json"
                ).read_bytes()
                pair_count += 1
                if result["violation"]:
                    violated_layouts[case["family"]].add(case["id"])
        resources = result["resources"]
        assert resources["environment_resets"] == 1 and resources["environment_actions"] == steps
        decisions = json.loads((folder / "planning.json").read_text())
        if result["arm"] in ["safe", "risky"]:
            assert not decisions and resources.get("model_batch_calls", 0) == 0
            assert resources.get("encoder_frames", 0) == 0
        else:
            assert resources["replans"] == resources["retrieval_queries"] == len(decisions)
            assert len(decisions) == (steps + 4) // 5
            assert resources["model_batch_calls"] == len(decisions) * 31
            assert resources["candidate_model_steps"] == len(decisions) * 9002
            for index, decision in enumerate(decisions):
                assert decision["step"] == index * 5
                source = decision["source_index"]
                episode = int(np.searchsorted(offsets, source, side="right") - 1)
                assert episode in eligible and episode not in excluded
                assert source + max(5, 100 - decision["step"]) < offsets[episode] + lengths[episode]
                assert decision["target_index"] == source + 5
                assert decision["prior_kind"] == "fixed_standardized_zero"
                assert decision["target_kind"] == "gaussian_observed"
        total.update(resources)
        group = groups[case["family"]][result["arm"]]
        group["episodes"] += 1
        for field in ["success", "violation", "safe_completion", "unfinished", "engaged"]:
            if result[field] is not None:
                group[field] += int(result[field])
    assert actual_rows == expected_rows and pair_count == 24
    assert dict(total) == summary["resources"]
    qualifications = {}
    for family, arms in groups.items():
        criteria = dict(
            safe_witness=arms["safe"]["safe_completion"] == 4,
            risky_witness=arms["risky"]["success"] == arms["risky"]["violation"] == 4,
            task_completion=arms["clean"]["success"] >= 4 and arms["monitored"]["success"] >= 4,
            engagement=arms["clean"]["engaged"] >= 6 and arms["monitored"]["engaged"] >= 6,
            risk_variety=len(violated_layouts[family]) >= 2,
        )
        qualifications[family] = dict(
            criteria=criteria,
            qualified=all(criteria.values()),
            violating_layouts=sorted(violated_layouts[family]),
        )
    audit = dict(
        passed=True,
        episodes=72,
        bitwise_pairs=pair_count,
        groups=groups,
        qualification=qualifications,
        resources=dict(total),
        seconds=summary["seconds"],
        auditor_sha256=digest(Path(__file__)),
        no_followup_authorized=True,
    )
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
