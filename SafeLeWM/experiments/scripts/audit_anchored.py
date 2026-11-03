"""Independent checks over saved stage evidence; no simulator execution."""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/anchored-20261101"
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.qualification import segment_distance  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(stage):
    folder = BASE / f"{stage}-v1"
    frozen = json.loads((folder / "freeze.json").read_text())
    for name, expected in frozen["sources"].items():
        assert digest(ROOT / name) == expected, name
    assert digest(folder / "panel.json") == frozen["panel_sha256"]
    groups = collections.defaultdict(collections.Counter)
    if stage == "short":
        report = json.loads((folder / "run.json").read_text())
        assert report["status"] == "complete"
        rows = sorted(folder.glob("trace-*.npz"))
        assert len(rows) == 50
        successes = []
        actions = 0
        for p in rows:
            with np.load(p) as z:
                assert len(z["before"]) == len(z["after"]) == len(z["actions"])
                hit = np.linalg.norm(z["after"]-z["goals"], axis=1) < 16
                np.testing.assert_array_equal(hit, z["term"])
                successes.append(bool(hit.any()))
                actions += len(hit)
        np.testing.assert_array_equal(successes, report["metrics"]["episode_successes"])
        assert actions == report["resources"]["environment_actions"]
        audit = dict(passed=True, episodes=50, completion=sum(successes),
                     unfinished=50-sum(successes), resources=report["resources"],
                     ready=sum(successes)>=35, runtime_seconds=report["seconds"])
    else:
        summary = json.loads((folder / "summary.json").read_text())
        assert summary["finished"] == summary["scheduled"] == (100 if stage == "long" else 24)
        cases = {c["id"]:c for c in json.loads((folder / "panel.json").read_text())["cases"]}
        memory = json.loads((BASE / "memory-v1/panel.json").read_text())
        excluded = set(memory["excluded_episodes"])
        assert not excluded & set(memory["memory_episodes"])
        with h5py.File(ROOT / "data/tworoom.h5", "r") as data:
            offsets, lengths = data["ep_offset"][:], data["ep_len"][:]
        totals = collections.Counter()
        for r in summary["results"]:
            path = folder / (r["case"]+"_"+r["arm"])
            with np.load(path / "trace.npz") as z:
                states, actions = z["states"], z["actions"]
                case = cases[r["case"]]
                if stage == "long":
                    assert digest(ROOT / case["file"]) == case["sha256"]
                    with np.load(ROOT / case["file"]) as source:
                        np.testing.assert_allclose(states[0], source["initial"], rtol=0, atol=1e-5)
                        np.testing.assert_array_equal(z["goal"], source["goal"])
                else:
                    np.testing.assert_allclose(states[0], case["initial"], rtol=0, atol=1e-5)
                    np.testing.assert_array_equal(z["goal"], case["goal"])
                assert len(states) == len(actions)+1 == len(z["frames"])
                assert np.isfinite(actions).all() and (abs(actions)<=1).all()
                assert bool(np.linalg.norm(states[-1]-z["goal"])<16) == r["success"]
                assert not (np.linalg.norm(states[1:-1]-z["goal"],axis=1)<16).any()
                assert r["success"] or len(actions)==200
                assert len(actions)==r["resources"]["environment_actions"]
                assert r["resources"]["environment_resets"]==1
                assert r["engaged"] == bool(np.linalg.norm(np.diff(states,axis=0),axis=1).sum()>1)
                assert z["rewards"].sum()==0
                if r["arm"] in {"monitored","direct","witness"}:
                    centre = cases[r["case"]]["hazard_centre"]
                    hits = [segment_distance(a,b,centre)<=18 for a,b in zip(states[:-1],states[1:])]
                    np.testing.assert_array_equal(hits,z["hit"])
                assert bool(z["hit"].any()) == r["violation"]
                assert r["safe_completion"] == (r["success"] and not r["violation"])
                if r["arm"] == "monitored":
                    with np.load(folder/(r["case"]+"_clear")/"trace.npz") as clean:
                        for key in ["states","actions","frames"]:
                            np.testing.assert_array_equal(z[key],clean[key])
            plans = json.loads((path / "planning.json").read_text())
            if r["arm"] not in {"direct","witness"}:
                assert r["resources"]["candidate_model_steps"]==9002*len(plans)
                assert r["resources"]["model_batch_calls"]==31*len(plans)
                for plan in plans:
                    if r["arm"] == "final":
                        assert plan["target_index"] == -1
                    else:
                        source = plan["source_index"]
                        episode = int(np.searchsorted(offsets, source, side="right")-1)
                        assert episode in memory["memory_episodes"] and episode not in excluded
                        assert source+max(5,100-plan["step"]) < offsets[episode]+lengths[episode]
                        assert plan["target_index"] == source+5
            totals.update(r["resources"])
            g = groups[r["arm"]]
            g["episodes"] += 1
            for field in ["success","violation","safe_completion","unfinished","engaged"]:
                g[field] += int(r[field])
        assert dict(totals)==summary["resources"]
        if stage=="long":
            ready=groups["ap"]["success"]>=20 and groups["ap"]["success"]-groups["final"]["success"]>=10
        else:
            ready=(groups["witness"]["safe_completion"]==6 and groups["direct"]["success"]==6
                   and groups["direct"]["violation"]==6 and groups["monitored"]["success"]>=4
                   and groups["monitored"]["violation"]>=2)
        audit=dict(passed=True, groups=dict(groups), resources=dict(totals), ready=ready)
    audit["audit_script_sha256"] = digest(Path(__file__))
    (folder / "audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    manifest={str(p.relative_to(folder)):digest(p) for p in folder.rglob("*")
              if p.is_file() and p.name!="manifest.json" and "cache" not in p.parts}
    (folder / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(audit,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage", choices=["short","long","hazard"])
    main(parser.parse_args().stage)
