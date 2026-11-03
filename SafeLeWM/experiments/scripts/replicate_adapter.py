"""Three paired repeats with deterministic, exact simulator-action budgets."""

import collections
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import socket
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/replication-20261101/run-v1"
sys.path.insert(0, str(ROOT / "experiments"))
import adapter_pilot_v2 as qualified  # noqa: E402
from helpers.pilot_common import digest, save, score  # noqa: E402
from helpers.replication_budget import allocation, eligible  # noqa: E402


class Pool(qualified.Pool):
    def __init__(self, deadline):
        from replication_worker import worker

        context = mp.get_context("spawn")
        self.output = context.Queue()
        self.inputs = [context.Queue() for _ in range(8)]
        self.processes = [
            context.Process(
                target=worker, args=(i, self.inputs[i], self.output, str(OUT), deadline)
            )
            for i in range(8)
        ]
        self.deadline = deadline
        try:
            for process in self.processes:
                process.start()
            for _ in self.processes:
                assert self.receive()["kind"] == "ready"
        except BaseException:
            self.close()
            raise


def freeze():
    # Reuse all qualified source/asset checks, extending the record before any reset.
    qualified.OUT = OUT
    qualified.freeze()
    frozen = json.loads((OUT / "freeze.json").read_text())
    sources = [
        ROOT / "AdapterReplication.md",
        Path(__file__),
        ROOT / "experiments/scripts/replication_worker.py",
        ROOT / "experiments/scripts/audit_replication.py",
        ROOT / "experiments/scripts/analyze_replication.py",
        ROOT / "experiments/helpers/replication_budget.py",
        ROOT / "experiments/tests/test_replication.py",
        *sorted((ROOT / "configs/adapter-replication").glob("*")),
    ]
    for path in sources:
        name = str(path.relative_to(ROOT))
        frozen["sources"][name] = digest(path)
        destination = OUT / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    frozen["study"] = "three-repeat exact-action replication"
    save(OUT / "freeze.json", frozen)


def spec(stage, case, arm, seed, prefix, repeat=None, candidate=None):
    return dict(
        id=f"{prefix}_{case['id']}_{seed}",
        stage=stage,
        case=case,
        arm=arm,
        seed=seed,
        repeat=repeat,
        candidate=candidate,
    )


def prefix_check(full, short, limit):
    with np.load(OUT / "episodes" / full / "trace.npz") as a:
        with np.load(OUT / "episodes" / short / "trace.npz") as b:
            for key in [
                "states",
                "frames",
                "actions",
                "base_actions",
                "hit",
                "rewards",
                "terminated",
                "truncated",
            ]:
                length = limit + int(key in ["states", "frames"])
                np.testing.assert_array_equal(a[key][:length], b[key])
    row = json.loads((OUT / "episodes" / short / "result.json").read_text())
    assert row["status"] == "budget_exhausted"
    assert row["resources"]["environment_actions"] == limit


def search(pool, repeat, optimizer, neutral, scales, config, panel, summary, record):
    from helpers.pilot_optimizer import make_optimizer, tell_ordered
    from helpers.recurrent_adapter import standardized_parameters

    rid = repeat["id"]
    key = f"{rid}-{optimizer}"
    es = make_optimizer(len(neutral), config["sigma"], repeat["cma_seed"], generations=5000)
    rng = np.random.default_rng(repeat["random_seed"])
    spent, generation, updates = 0, 0, 0
    budget = config["training_actions_per_search"]
    candidates, dispatches = [], []
    folder = OUT / "searches" / key
    folder.mkdir(parents=True, exist_ok=False)
    while spent < budget:
        assert generation < 5000
        genotypes = (
            es.ask() if optimizer == "cma" else rng.normal(0, config["sigma"], (8, len(neutral)))
        )
        jobs = []
        for i, genotype in enumerate(genotypes):
            cid = f"{key}-{generation * 8 + i:05d}"
            parameters = standardized_parameters(neutral, scales, genotype)
            np.savez(OUT / "candidates" / (cid + ".npz"), genotype=genotype, parameters=parameters)
            jobs.append(
                dict(
                    id=cid,
                    device=i,
                    parameters=parameters,
                    episodes=[
                        spec("training", case, optimizer, panel["training_seed"], cid, rid, cid)
                        for case in panel["training"]
                    ],
                )
            )
        scores, cursor = [], 0
        while cursor < 8 and spent < budget:
            quotas = allocation(budget - spent, 8 - cursor)
            batch = jobs[cursor : cursor + len(quotas)]
            for job, quota in zip(batch, quotas):
                job["action_quota"] = quota
            dispatch = dict(
                index=len(dispatches),
                spent_before=spent,
                ids=[j["id"] for j in batch],
                quotas=quotas,
            )
            # Save reservations before work, including a dispatch interrupted by failure.
            save(folder / f"dispatch-{len(dispatches):04d}.json", dispatch)
            values = pool.batch(batch)
            actuals = []
            for job in batch:
                rows = values[job["id"]]
                actual = sum(r["resources"]["environment_actions"] for r in rows)
                assert 0 < actual <= job["action_quota"]
                actuals.append(actual)
                spent += actual
                assert spent <= budget
                complete = eligible(rows)
                entry = dict(
                    id=job["id"],
                    repeat=rid,
                    optimizer=optimizer,
                    generation=generation,
                    eligible=complete,
                    actions=actual,
                    quota=job["action_quota"],
                    score=None,
                    episodes=len(rows),
                )
                if complete:
                    entry.update(score(rows))
                    scores.append(entry["score"])
                candidates.append(entry)
                summary["candidates"].append(entry)
                record(rows)
            dispatch.update(actuals=actuals, spent_after=spent, refunded=sum(quotas) - sum(actuals))
            save(folder / f"dispatch-{len(dispatches):04d}.json", dispatch)
            dispatches.append(dispatch)
            cursor += len(batch)
        complete_generation = len(scores) == 8
        if optimizer == "cma" and complete_generation:
            tell_ordered(es, genotypes, scores)
            updates += 1
            with (folder / f"state-{generation:04d}.pkl").open("wb") as stream:
                pickle.dump(es, stream)
        save(
            folder / f"generation-{generation:04d}.json",
            dict(
                generation=generation,
                asked=[j["id"] for j in jobs],
                evaluated=cursor,
                complete=complete_generation,
                scores=scores,
                spent=spent,
                updates=updates,
            ),
        )
        save(OUT / "summary.json", summary)
        print(
            json.dumps(
                dict(search=key, generation=generation, spent=spent, budget=budget, scores=scores)
            ),
            flush=True,
        )
        generation += 1
    assert spent == budget
    best = max((c for c in candidates if c["eligible"]), key=lambda c: c["score"])
    result = dict(
        repeat=rid,
        optimizer=optimizer,
        actions=spent,
        generations=generation,
        updates=updates,
        eligible=sum(c["eligible"] for c in candidates),
        partial=sum(not c["eligible"] for c in candidates),
        nominee=best,
    )
    save(folder / "complete.json", result)
    summary["searches"][key] = result
    return best


def run():
    from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter
    from helpers.recurrent_adapter import standardized_parameters

    config = json.loads((ROOT / "configs/adapter-replication/run.json").read_text())
    panel = json.loads((ROOT / "configs/adapter-replication/panel.json").read_text())
    assert socket.gethostname() == config["runtime_hostname"]
    assert importlib.metadata.version("cma") == "4.4.4"
    assert (OUT.parent / "software-checks-passed").exists()
    freeze()
    started = time.monotonic()
    deadline = started + config["run_seconds"]
    summary = dict(
        status="running", phase="engineering", results=[], candidates=[], searches={}, resources={}
    )
    counts = collections.Counter()
    pool = None
    initializations, untrained = {}, {}
    (OUT / "initialization").mkdir()
    (OUT / "candidates").mkdir()
    for repeat in config["repeats"]:
        rid = repeat["id"]
        actor = RecurrentActionAdapter(
            AdapterSpec(396, 2, initialization_seed=repeat["initialization_seed"])
        )
        neutral, scales = actor.parameters_vector(), actor.coordinate_scales()
        initializations[rid] = neutral, scales
        np.savez(OUT / "initialization" / (rid + ".npz"), neutral=neutral, scales=scales)
        save(OUT / "initialization" / (rid + "-spec.json"), actor.specification())
        z = np.random.default_rng(repeat["untrained_seed"]).normal(0, config["sigma"], len(neutral))
        untrained[rid] = standardized_parameters(neutral, scales, z)
        np.savez(
            OUT / "initialization" / (rid + "-untrained.npz"), parameters=untrained[rid], genotype=z
        )

    def record(rows):
        summary["results"].extend(rows)
        for row in rows:
            counts.update(row["resources"])
        summary["resources"] = dict(counts)
        summary["seconds"] = time.monotonic() - started
        assert counts["environment_actions"] <= config["max_total_actions"]
        save(OUT / "summary.json", summary)

    def batches(jobs):
        for offset in range(0, len(jobs), 8):
            batch = jobs[offset : offset + 8]
            for i, job in enumerate(batch):
                job["device"] = i
            values = pool.batch(batch)
            for job in batch:
                record(values[job["id"]])

    try:
        pool = Pool(deadline)
        neutral = initializations["r0"][0]
        for arm, parameters in [("native", None), ("neutral", neutral)]:
            batches(
                [
                    dict(
                        id=f"parity-{arm}-{i}",
                        parameters=parameters,
                        episodes=[
                            spec(
                                "engineering",
                                case,
                                arm,
                                panel["training_seed"],
                                f"parity-{arm}",
                                "r0",
                            )
                        ],
                    )
                    for i, case in enumerate(panel["training"])
                ]
            )
        pairs = []
        for case in panel["training"]:
            suffix = f"{case['id']}_{panel['training_seed']}"
            pair = [f"parity-native_{suffix}", f"parity-neutral_{suffix}"]
            qualified.parity(*pair)
            pairs.append(pair)
        first = panel["training"][0]
        full_id = f"parity-native_{first['id']}_{panel['training_seed']}"
        jobs = [
            dict(
                id=f"cross-{i}",
                device=i,
                parameters=None,
                episodes=[
                    spec("engineering", first, "native", panel["training_seed"], f"cross-{i}")
                ],
            )
            for i in range(1, 8)
        ]
        values = pool.batch(jobs)
        for job in jobs:
            record(values[job["id"]])
            pair = [full_id, job["episodes"][0]["id"]]
            qualified.parity(*pair)
            pairs.append(pair)
        probes = []
        for limit in [1, 7]:
            probe = spec("engineering", first, "native", panel["training_seed"], f"prefix-{limit}")
            job = dict(
                id=f"prefix-{limit}",
                device=0,
                parameters=None,
                action_quota=limit,
                episodes=[probe],
            )
            record(pool.batch([job])[job["id"]])
            prefix_check(full_id, probe["id"], limit)
            probes.append(dict(full=full_id, short=probe["id"], limit=limit))
        save(OUT / "parity.json", dict(passed=True, pairs=pairs, prefixes=probes, episodes=25))
        summary["phase"] = "witnesses"
        batches(
            [
                dict(
                    id=f"witness-{i}-{arm}",
                    parameters=None,
                    episodes=[spec("witness", case, arm, panel["witness_seed"], "witness-" + arm)],
                )
                for i, case in enumerate(panel["evaluation"])
                for arm in ["safe", "risky"]
            ]
        )
        witnesses = [r for r in summary["results"] if r["stage"] == "witness"]
        assert len(witnesses) == 24
        assert all(r["success"] and r["violation"] == (r["arm"] == "risky") for r in witnesses)
        save(OUT / "witnesses.json", dict(passed=True, episodes=24))
        nominees = {}
        for repeat in config["repeats"]:
            rid = repeat["id"]
            neutral, scales = initializations[rid]
            for optimizer in repeat["optimizer_order"]:
                summary["phase"] = f"{rid}-{optimizer}"
                nominees[f"{rid}-{optimizer}"] = search(
                    pool, repeat, optimizer, neutral, scales, config, panel, summary, record
                )
        assert len(nominees) == 6
        save(
            OUT / "selection.json",
            dict(
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                locked_at=time.time(),
                nominees=nominees,
            ),
        )
        summary["phase"] = "evaluation"
        arms = [("native", None, None, None)]
        for repeat in config["repeats"]:
            rid = repeat["id"]
            arms.append(("untrained", rid, untrained[rid], None))
            for optimizer in ["cma", "random"]:
                cid = nominees[f"{rid}-{optimizer}"]["id"]
                with np.load(OUT / "candidates" / (cid + ".npz")) as checkpoint:
                    arms.append((optimizer, rid, checkpoint["parameters"].copy(), cid))
        for arm, rid, parameters, cid in arms:
            prefix = f"eval-{rid or 'shared'}-{arm}"
            jobs = [
                dict(
                    id=f"{prefix}-{i}",
                    parameters=parameters,
                    episodes=[spec("evaluation", case, arm, seed, prefix, rid, cid)],
                )
                for i, (case, seed) in enumerate(
                    (c, s) for c in panel["evaluation"] for s in panel["evaluation_seeds"]
                )
            ]
            batches(jobs)
        evaluation = [r for r in summary["results"] if r["stage"] == "evaluation"]
        assert len(evaluation) == 360
        assert sum(s["actions"] for s in summary["searches"].values()) == 240000
        summary["evaluation"] = {
            f"{rid or 'shared'}-{arm}": score(
                [r for r in evaluation if r["repeat"] == rid and r["arm"] == arm]
            )
            for arm, rid, _, _ in arms
        }
        summary["status"] = "complete"
    except BaseException:
        summary["status"] = "failed"
        summary["failure"] = traceback.format_exc()
        raise
    finally:
        if pool is not None:
            pool.close()
        summary["resources"] = qualified.resource_summary()
        summary["seconds"] = time.monotonic() - started
        save(OUT / "summary.json", summary)


if __name__ == "__main__":
    run()
