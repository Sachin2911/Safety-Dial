"""Implement the frozen two-contrast PPO analysis on its existing rented host."""

import collections
import hashlib
import json
from pathlib import Path
import socket

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/ppo-20261101"


def main():
    assert socket.gethostname() == "075cb160c9b3"
    summaries = {
        "4090": json.loads((OUT / "run-v1/summary.json").read_text()),
        "l4": json.loads((OUT / "l4-summary-for-analysis.json").read_text()),
    }
    audits = {
        "4090": json.loads((OUT / "run-v1/audit.json").read_text()),
        "l4": json.loads((OUT / "l4-audit-for-analysis.json").read_text()),
    }
    assert all(a["passed"] for a in audits.values())
    rows = [r for s in summaries.values() for r in s["results"] if r["stage"] == "evaluation"]
    assert len(rows) == 1008
    metrics = ["safe_completion", "success", "violation", "unfinished", "engaged"]
    groups = {}
    for host, summary in summaries.items():
        hostrows = [r for r in summary["results"] if r["stage"] == "evaluation"]
        for repeat, arm in [(None, "native")] + [
            (rid, a) for rid in summary["host"]["repeats"]
            for a in ["untrained", "main", "random", "ppo"]
        ]:
            subset = [r for r in hostrows if r["repeat"] == repeat and r["arm"] == arm]
            assert len(subset) == 72
            groups[f"{host}-{repeat or 'shared'}-{arm}"] = {
                "n": 72, **{metric: sum(r[metric] for r in subset) for metric in metrics}
            }
    pooled = {}
    for arm in ["native", "untrained", "main", "random", "ppo"]:
        subset = [r for r in rows if r["arm"] == arm]
        pooled[arm] = dict(n=len(subset), **{
            metric: sum(r[metric] for r in subset) for metric in metrics
        })
    cases = sorted({r["case"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    assert len(cases) == 24 and len(seeds) == 3
    values = {}
    for arm in ["main", "random", "ppo"]:
        mapping = {(r["repeat"], r["case"], r["seed"]): int(r["safe_completion"])
                   for r in rows if r["arm"] == arm}
        assert len(mapping) == 216
        values[arm] = np.asarray([
            [[mapping[rid, case, seed] for seed in seeds] for case in cases]
            for rid in ["R0", "R1", "R2"]
        ], float).mean(axis=2)
    rng = np.random.default_rng(12501)
    samples = {arm: [] for arm in ["ppo", "random"]}
    for _ in range(2000):
        ri = rng.integers(0, 3, 3)
        ci = rng.integers(0, 24, 24)
        for arm in samples:
            diff = values["main"] - values[arm]
            samples[arm].append(float(diff[np.ix_(ri, ci)].mean()))
    contrasts = {"cma_minus_" + arm: dict(
        difference_pp=100 * float((values["main"] - values[arm]).mean()),
        interval95_pp=(100 * np.quantile(samples[arm], [0.025, 0.975])).tolist(),
        per_repeat_pp=(100 * (values["main"] - values[arm]).mean(axis=1)).tolist(),
    ) for arm in samples}
    training = {}
    for host, summary in summaries.items():
        for name, result in summary["searches"].items():
            assert result["actions"] == 40000
            related = [r for r in summary["results"] if r.get("search") == name
                       and r["stage"] != "evaluation"]
            stage_counts = collections.Counter()
            for row in related:
                stage_counts[row["stage"]] += row["resources"]["environment_actions"]
            training[name] = dict(host=host, actions=result["actions"],
                                  updates=result["updates"], actions_by_stage=dict(stage_counts))
    result = dict(
        protocol="MatchedPPO.md", analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        groups=groups, pooled=pooled, contrasts=contrasts, training=training,
        bootstrap=dict(draws=2000, seed=12501, scheme="crossed repeat/case; planner seeds together"),
        host_resources={host: s["resources"] for host, s in summaries.items()},
        host_seconds={host: s["seconds"] for host, s in summaries.items()},
        limitations=["Three repeats", "Two hardware blocks", "One fixed PPO setting",
                     "Shared final panel with mechanism study; not independent replication"],
    )
    destination = OUT / "combined-analysis.json"
    assert not destination.exists()
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(pooled=pooled, contrasts=contrasts)), flush=True)


if __name__ == "__main__":
    main()
