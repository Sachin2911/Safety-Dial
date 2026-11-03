"""Audit saved PushT reproduction evidence on the rented host, then package it."""

import collections
import json
import socket
import tarfile
import time

import numpy as np

from pusht_ap_reproduction import OUT, ROOT, digest, save, verify_sources


def physical_success(states, goal):
    delta = np.abs(states[:, 4].astype(float) - float(goal[4])) % (2 * np.pi)
    angle = np.minimum(delta, 2 * np.pi - delta)
    return (np.linalg.norm(states[:, :4] - goal[:4], axis=1) < 20) & (angle < np.pi / 9)


def main():
    assert socket.gethostname() == "c45db82630e6"
    frozen = verify_sources()
    cohorts = {}
    model_counts = {}
    resources = collections.Counter()
    engineering = json.loads((OUT / "engineering.json").read_text())
    assert engineering["passed"]
    for arm in ["authors", "wrapper"]:
        folder = OUT / "run-v1" / arm
        counts = collections.Counter()
        rows = {}
        for shard in range(8):
            record = json.loads((folder / f"resources-{shard}.json").read_text())
            assert record["status"] == "complete"
            counts.update(record["counts"])
        model_counts[arm] = dict(counts)
        if arm == "authors":
            for path in sorted(folder.glob("outcomes_*.jsonl")):
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    key = row["identity"]["record_id"]
                    assert key not in rows
                    rows[key] = row
        else:
            for path in folder.glob("ap-fresh-pusht-*/result.json"):
                row = json.loads(path.read_text())
                assert row["case"] not in rows
                rows[row["case"]] = row
        assert set(rows) == {q["record_id"] for q in frozen["queries"]}, (arm, len(rows))
        completed = []
        for ordinal, query in enumerate(frozen["queries"]):
            key = query["record_id"]
            row = rows[key]
            with np.load(folder / "physical" / (key + ".npz")) as trace:
                states, actions, goal = trace["states"], trace["actions"], trace["goal"]
                assert len(states) == len(actions) + 1 and 0 <= len(actions) <= 280
                assert np.isfinite(states).all() and np.isfinite(actions).all()
                assert (np.abs(actions) <= 1).all()
                success = physical_success(states, goal)
                assert not success[:-1].any(), (arm, key, "continued after success")
                result = bool(success[-1])
                recorded_steps = row["primitive_steps"] if arm == "authors" else row["steps"]
                assert len(actions) == recorded_steps
                if arm == "authors":
                    assert result == row["success"], key
                    assert row["optimizer_seed"] == 26091331 + 1000 * ordinal
                    detail = json.loads((
                        folder / "traces/pusht" / (key + "_clean_gaussian_observed.json")
                    ).read_text())
                    np.testing.assert_array_equal(
                        actions, np.asarray(detail["actions"]).reshape(-1, 2)
                    )
                    assert row["predictor_calls"] == row["decisions"] * 31
                    assert row["predictor_candidate_blocks"] == row["decisions"] * 9002
                else:
                    assert row["seed"] == 26091331 + 1000 * ordinal
                    with np.load(folder / row["id"] / "trace.npz") as original:
                        np.testing.assert_array_equal(actions, original["actions"].reshape(-1, 2))
                        np.testing.assert_array_equal(states, original["states"])
                length = float(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).sum())
                completed.append(dict(
                    query=key, success=result, recorded_success=bool(row["success"]),
                    scoring_agrees=result == bool(row["success"]), initial_success=bool(success[0]),
                    steps=len(actions), engaged=length > 10.0, path_length=length,
                ))
        cohorts[arm] = completed
        assert sum(r["steps"] for r in completed) == sum(
            json.loads(line)["event"] == "action_complete"
            for p in folder.glob("ledger-*.jsonl") for line in p.read_text().splitlines()
        )
    for path in [OUT / "engineering-ledger.jsonl", *OUT.glob("run-v1/*/ledger-*.jsonl")]:
        for line in path.read_text().splitlines():
            resources[json.loads(line)["event"]] += 1
    assert resources["reset_attempt"] == 260
    assert resources["action_attempt"] == resources["action_complete"] <= frozen["action_cap"]
    pairs = []
    for query in frozen["queries"]:
        key = query["record_id"]
        with np.load(OUT / "run-v1/authors/physical" / (key + ".npz")) as a, np.load(
            OUT / "run-v1/wrapper/physical" / (key + ".npz")
        ) as b:
            np.testing.assert_array_equal(a["states"][0], b["states"][0])
            np.testing.assert_array_equal(a["frames"][0], b["frames"][0])
            np.testing.assert_array_equal(a["goal"], b["goal"])
            n = min(len(a["actions"]), len(b["actions"]))
            differences = np.flatnonzero(np.any(a["actions"][:n] != b["actions"][:n], axis=1))
            pairs.append(dict(query=key,
                              first_different_action=int(differences[0]) if len(differences) else None,
                              identical_actions=bool(not len(differences)
                                                     and len(a["actions"]) == len(b["actions"]))))
    summary = {}
    for arm, rows in cohorts.items():
        summary[arm] = dict(
            n=len(rows), completed=sum(r["success"] for r in rows),
            unfinished=sum(not r["success"] for r in rows),
            engaged=sum(r["engaged"] for r in rows),
            initially_successful=sum(r["initial_success"] for r in rows),
            scoring_disagreements=sum(not r["scoring_agrees"] for r in rows),
            actions=sum(r["steps"] for r in rows),
        )
    paired = collections.Counter(
        f"authors_{int(a['success'])}_wrapper_{int(b['success'])}"
        for a, b in zip(cohorts["authors"], cohorts["wrapper"], strict=True)
    )
    audit = dict(
        passed=True, completed_at=time.time(), episodes=256, summary=summary,
        paired_outcomes=dict(paired), pairs=pairs, cohorts=cohorts,
        resources=dict(resources), model_counts=model_counts, cache_encoder_frames=2336736,
        published_reference=dict(completed=89, n=128),
        limitations=["RTX 4090 rather than H20", "Python 3.11.9 rather than 3.11.14",
                     "Paired wrapper contrast changes multiple implementation details"],
    )
    save(OUT / "audit.json", audit)
    print(json.dumps(dict(passed=True, summary=summary, paired=dict(paired))), flush=True)
    archive = ROOT / "outputs/pusht-ap-reproduction-20261102-evidence.tar.gz"
    assert not archive.exists()
    files = [p for p in sorted(OUT.rglob("*")) if p.is_file()
             and p.name not in ["latents.npy", "reference-reading.tar.gz"]
             and "__pycache__" not in p.parts
             and p.name not in ["orchestration.log", "audit-0.log"]]
    # Only the two currently open logs are collected separately after packaging.
    manifest = {str(p.relative_to(OUT)): dict(bytes=p.stat().st_size, sha256=digest(p))
                for p in files}
    save(OUT / "archive-manifest.json", dict(files=manifest))
    with tarfile.open(archive, "x:gz") as target:
        for path in files:
            target.add(path, arcname=str(path.relative_to(OUT)), recursive=False)
        target.add(OUT / "archive-manifest.json", arcname="archive-manifest.json", recursive=False)
    save(OUT / "preservation.json", dict(
        archive=str(archive), sha256=digest(archive), bytes=archive.stat().st_size,
        files=len(files), cache_preserved_on_host=True,
    ))


if __name__ == "__main__":
    main()
