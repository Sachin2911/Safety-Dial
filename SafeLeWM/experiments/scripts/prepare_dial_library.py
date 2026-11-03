"""Freeze a previously selected library without consulting any final outcomes."""

import hashlib
import json
from pathlib import Path
import shutil
import socket
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/safety-dial-library-20261102"
DIAGNOSTIC = ROOT / "outputs/safety-dial-20261102"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_library(analysis, summaries):
    policies, curves = {}, {}
    for summary in summaries:
        assert summary["status"] == "complete"
        for key, search in summary["searches"].items():
            pool = [c for c in summary["candidates"] if c["search"] == key and c["eligible"]]
            assert pool and all(c["n"] == c["episodes"] == 8 for c in pool)
            selections = []
            for epsilon in [0, 0.25, 0.5, 0.75, 1]:
                eligible = [c for c in pool if c["violations"] / 8 <= epsilon]
                best = (
                    max(eligible, key=lambda c: (c["complete"], c["safe"], -c["violations"]))
                    if eligible
                    else None
                )
                chosen = (
                    None
                    if best is None
                    else {k: best[k] for k in ["id", "complete", "safe", "violations", "n"]}
                )
                selections.append(dict(epsilon=epsilon, candidate=chosen))
                if best is not None:
                    cid = best["id"]
                    policies[cid] = dict(
                        id=cid,
                        candidate=cid,
                        repeat=cid.split("-")[0],
                        arm=search["optimizer"],
                        search=key,
                        original_path=f"candidates/{cid}.npz",
                    )
            assert selections == analysis["groups"][key]["selections"]
            curves[key] = selections
    assert set(curves) == set(analysis["groups"])
    for rid in ["R0", "R1", "R2"]:
        pid = rid + "-untrained"
        policies[pid] = dict(
            id=pid,
            candidate=None,
            repeat=rid,
            arm="untrained",
            search=None,
            original_path=f"initialization/{pid}.npz",
        )
    policies["shared-native"] = dict(
        id="shared-native",
        candidate=None,
        repeat=None,
        arm="native",
        search=None,
        original_path=None,
    )
    return policies, curves


def main():
    assert socket.gethostname() == "075cb160c9b3"
    BASE.mkdir(parents=True, exist_ok=False)
    (BASE / "preparation.json").write_text(json.dumps(dict(started=time.time())) + "\n")
    config = json.loads((ROOT / "configs/safety-dial-library/run.json").read_text())
    analysis = json.loads((DIAGNOSTIC / "analysis.json").read_text())
    freeze = json.loads((DIAGNOSTIC / "freeze.json").read_text())
    inputs = BASE / "selection-inputs"
    inputs.mkdir()
    for name, sha in freeze["sources"].items():
        assert digest(DIAGNOSTIC / "sources" / name) == sha
    paths = [
        "outputs/ppo-20261101/run-v1/summary.json",
        "outputs/ppo-20261101/l4-summary-for-analysis.json",
    ]
    summaries = []
    for index, name in enumerate(paths):
        source = DIAGNOSTIC / "sources" / name
        shutil.copyfile(source, inputs / f"summary-{index}.json")
        summaries.append(json.loads(source.read_text()))
    for name in ["analysis.json", "freeze.json"]:
        shutil.copyfile(DIAGNOSTIC / name, inputs / name)
    policies, curves = build_library(analysis, summaries)
    assert len(policies) == config["evaluation_policies"]
    assert len([p for p in policies.values() if p["candidate"]]) == config["selected_actors"]
    weights = BASE / "weights"
    weights.mkdir()
    foreign = ROOT / "outputs/safety-dial-r2-transfer-20261102"
    foreign_manifest = json.loads((foreign / "manifest.json").read_text())
    shutil.copyfile(foreign / "manifest.json", inputs / "r2-transfer-manifest.json")
    original_manifest = json.loads((ROOT / "outputs/ppo-4090-20261101-manifest.json").read_text())
    records = {}
    for pid, policy in policies.items():
        rel = policy["original_path"]
        if rel is None:
            continue
        source = (
            (foreign / rel)
            if policy["repeat"] == "R2"
            else ROOT / "outputs/ppo-20261101/run-v1" / rel
        )
        sha = digest(source)
        if policy["repeat"] == "R2":
            assert foreign_manifest["files"][rel]["sha256"] == sha
        else:
            original_key = "outputs/ppo-20261101/run-v1/" + rel
            assert original_manifest["original_files"][original_key]["sha256"] == sha
        target = weights / (pid + ".npz")
        shutil.copyfile(source, target)
        records[pid] = dict(
            sha256=sha, source=str(source.relative_to(ROOT)), bytes=target.stat().st_size
        )
    import random

    schedule = list(policies)
    random.Random(config["schedule_seed"]).shuffle(schedule)
    library = dict(
        policies=policies,
        curves=curves,
        weights=records,
        schedule=schedule,
        locked_at=time.time(),
        prior_acquisition_actions=360000,
        final_outcomes_used=False,
    )
    (BASE / "library.json").write_text(json.dumps(library, indent=2) + "\n")
    print(json.dumps(dict(policies=len(policies), selected=config["selected_actors"])), flush=True)


if __name__ == "__main__":
    main()
