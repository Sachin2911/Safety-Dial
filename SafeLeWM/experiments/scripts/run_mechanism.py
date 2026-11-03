"""Prospectively frozen representation selection followed by matched mechanism tests."""

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
OUT = ROOT / "outputs/mechanism-20261101/run-v1"
sys.path.insert(0, str(ROOT / "experiments"))
import adapter_pilot_v2 as qualified  # noqa: E402
import replicate_adapter as previous  # noqa: E402
from helpers.pilot_common import digest, save  # noqa: E402
from helpers.replication_budget import allocation, eligible  # noqa: E402
from helpers.mechanism import SearchMap, choose_representation, objective, projection_matrix  # noqa: E402


class Pool(qualified.Pool):
    def __init__(self, deadline):
        from mechanism_worker import worker

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
    previous.OUT = OUT
    previous.freeze()
    frozen = json.loads((OUT / "freeze.json").read_text())
    sources = [
        ROOT / "AdapterMechanism.md",
        Path(__file__),
        ROOT / "experiments/scripts/mechanism_worker.py",
        ROOT / "experiments/helpers/mechanism.py",
        ROOT / "experiments/scripts/audit_mechanism.py",
        ROOT / "experiments/scripts/analyze_mechanism.py",
        ROOT / "experiments/scripts/launch_mechanism.py",
        ROOT / "experiments/scripts/preserve_mechanism.py",
        ROOT / "experiments/tests/test_mechanism.py",
        *sorted((ROOT / "configs/adapter-mechanism").glob("*")),
    ]
    for path in sources:
        name = str(path.relative_to(ROOT))
        frozen["sources"][name] = digest(path)
        target = OUT / "sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    frozen["study"] = "representation selection and mechanism ablations"
    save(OUT / "freeze.json", frozen)


def spec(
    stage, case, arm, seed, prefix, repeat=None, candidate=None, representation=None, search=None
):
    return dict(
        id=f"{prefix}_{case['id']}_{seed}",
        stage=stage,
        case=case,
        arm=arm,
        seed=seed,
        repeat=repeat,
        candidate=candidate,
        representation=representation,
        search=search,
    )


def prefix_check(full, short, limit):
    previous.OUT = OUT
    return previous.prefix_check(full, short, limit)


def dispatch(pool, jobs, record):
    # Retain paired device assignment even in the sparse counterfactual panel.
    pending = list(jobs)
    while pending:
        batch, waiting, used = [], [], set()
        for job in pending:
            if job["device"] not in used:
                batch.append(job)
                used.add(job["device"])
            else:
                waiting.append(job)
        pending = waiting
        values = pool.batch(batch)
        for job in batch:
            record(values[job["id"]])


def search(pool, repeat, arm, mapping, config, panel, summary, record):
    from helpers.pilot_optimizer import make_optimizer, tell_ordered

    rid = repeat["id"]
    representation = mapping.representation
    optimizer = "random" if arm == "random" else "cma"
    key = f"{rid}-{representation}-{arm}"
    es = make_optimizer(mapping.dimension, config["sigma"], repeat["cma_seed"], generations=5000)
    rng = np.random.default_rng(repeat.get("random_seed", repeat["cma_seed"] + 1))
    spent, generation, updates = 0, 0, 0
    budget = config["training_actions_per_search"]
    candidates, dispatches = [], []
    folder = OUT / "searches" / key
    folder.mkdir(parents=True, exist_ok=False)
    while spent < budget:
        assert generation < 5000
        genotypes = (
            es.ask()
            if optimizer == "cma"
            else rng.normal(0, config["sigma"], (8, mapping.dimension))
        )
        jobs = []
        for i, genotype in enumerate(genotypes):
            cid = f"{key}-{generation * 8 + i:05d}"
            parameters = mapping.decode(genotype)
            np.savez(OUT / "candidates" / (cid + ".npz"), genotype=genotype, parameters=parameters)
            jobs.append(
                dict(
                    id=cid,
                    device=i,
                    parameters=parameters,
                    episodes=[
                        spec(
                            "training",
                            case,
                            arm,
                            panel["training_seed"],
                            cid,
                            rid,
                            cid,
                            representation,
                            key,
                        )
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
                    arm=arm,
                    representation=representation,
                    search=key,
                    generation=generation,
                    eligible=complete,
                    actions=actual,
                    quota=job["action_quota"],
                    score=None,
                    episodes=len(rows),
                )
                if complete:
                    entry.update(objective(rows, arm))
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
        arm=arm,
        representation=representation,
        dimension=mapping.dimension,
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

    config = json.loads((ROOT / "configs/adapter-mechanism/run.json").read_text())
    panel = json.loads((ROOT / "configs/adapter-mechanism/panel.json").read_text())
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
    initializations = {}
    (OUT / "initialization").mkdir()
    (OUT / "candidates").mkdir()
    for repeat in config["stage_a"] + config["stage_b"]:
        rid = repeat["id"]
        actor = RecurrentActionAdapter(
            AdapterSpec(396, 2, initialization_seed=repeat["initialization_seed"])
        )
        neutral, scales = actor.parameters_vector(), actor.coordinate_scales()
        projection = projection_matrix(repeat["projection_seed"])
        initializations[rid] = neutral, scales, projection
        np.savez(OUT / "initialization" / (rid + ".npz"), neutral=neutral, scales=scales)
        np.save(OUT / "initialization" / (rid + "-projection.npy"), projection)
        save(OUT / "initialization" / (rid + "-spec.json"), actor.specification())

    def record(rows):
        summary["results"].extend(rows)
        for row in rows:
            counts.update(row["resources"])
        summary["resources"] = dict(counts)
        summary["seconds"] = time.monotonic() - started
        assert counts["environment_actions"] <= config["max_total_actions"]
        save(OUT / "summary.json", summary)

    def evaluate(stage, cases, seeds, arm, rid, parameters, cid, representation, search_id):
        prefix = f"{stage}-{rid or 'shared'}-{representation or 'native'}-{arm}"
        jobs = []
        for ci, case in enumerate(cases):
            base_index = case.get("base_index", ci)
            for si, seed in enumerate(seeds):
                episode = spec(stage, case, arm, seed, prefix, rid, cid, representation, search_id)
                jobs.append(
                    dict(
                        id=episode["id"],
                        device=(base_index * len(seeds) + si) % 8,
                        parameters=parameters,
                        episodes=[episode],
                    )
                )
        dispatch(pool, jobs, record)

    def parameters(cid):
        with np.load(OUT / "candidates" / (cid + ".npz")) as checkpoint:
            return checkpoint["parameters"].copy()

    try:
        pool = Pool(deadline)
        neutral = initializations["A0"][0]
        for arm, values in [("native", None), ("neutral", neutral)]:
            jobs = []
            for i, case in enumerate(panel["training"]):
                episode = spec(
                    "engineering",
                    case,
                    arm,
                    panel["training_seed"],
                    "parity-" + arm,
                    "A0",
                    representation="full",
                )
                jobs.append(dict(id=episode["id"], device=i, parameters=values, episodes=[episode]))
            dispatch(pool, jobs, record)
        pairs = []
        reference_pairs = []
        for case in panel["training"]:
            suffix = f"{case['id']}_{panel['training_seed']}"
            pair = [f"parity-native_{suffix}", f"parity-neutral_{suffix}"]
            qualified.parity(*pair)
            pairs.append(pair)
            reference = (
                ROOT / "outputs/replication-20261101/run-v1/episodes" / pair[0] / "trace.npz"
            )
            with np.load(reference) as a:
                with np.load(OUT / "episodes" / pair[0] / "trace.npz") as b:
                    for key in ["actions", "states", "frames", "hit", "rewards", "terminated"]:
                        np.testing.assert_array_equal(a[key], b[key])
            reference_pairs.append(
                dict(
                    current=pair[0],
                    reference=str(reference.relative_to(ROOT)),
                    sha256=digest(reference),
                )
            )
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
        dispatch(pool, jobs, record)
        for job in jobs:
            pair = [full_id, job["episodes"][0]["id"]]
            qualified.parity(*pair)
            pairs.append(pair)
        probes = []
        for limit in [1, 7]:
            probe = spec("engineering", first, "native", panel["training_seed"], f"prefix-{limit}")
            job = dict(
                id=probe["id"], device=0, parameters=None, action_quota=limit, episodes=[probe]
            )
            dispatch(pool, [job], record)
            prefix_check(full_id, probe["id"], limit)
            probes.append(dict(full=full_id, short=probe["id"], limit=limit))
        save(
            OUT / "parity.json",
            dict(
                passed=True, pairs=pairs, prefixes=probes, references=reference_pairs, episodes=25
            ),
        )
        summary["phase"] = "witnesses"
        jobs = []
        for i, case in enumerate(panel["evaluation"] + panel["diagnostic"]):
            for j, arm in enumerate(["safe", "risky"]):
                episode = spec("witness", case, arm, panel["witness_seed"], "witness-" + arm)
                jobs.append(
                    dict(
                        id=episode["id"],
                        device=(2 * i + j) % 8,
                        parameters=None,
                        episodes=[episode],
                    )
                )
        dispatch(pool, jobs, record)
        witnesses = [r for r in summary["results"] if r["stage"] == "witness"]
        assert len(witnesses) == 64
        assert all(r["success"] and r["violation"] == (r["arm"] == "risky") for r in witnesses)
        save(OUT / "witnesses.json", dict(passed=True, episodes=64))
        nominees_a = {}
        for repeat in config["stage_a"]:
            rid = repeat["id"]
            neutral, scales, projection = initializations[rid]
            for representation in repeat["order"]:
                key = f"{rid}-{representation}-main"
                summary["phase"] = "training-" + key
                mapping = SearchMap(neutral, scales, representation, projection)
                nominees_a[key] = search(
                    pool, repeat, "main", mapping, config, panel, summary, record
                )
        save(OUT / "selection-a.json", dict(locked_at=time.time(), nominees=nominees_a))
        summary["phase"] = "development"
        for repeat in config["stage_a"]:
            rid = repeat["id"]
            for representation in config["representations"]:
                key = f"{rid}-{representation}-main"
                cid = nominees_a[key]["id"]
                evaluate(
                    "development",
                    panel["development"],
                    panel["development_seeds"],
                    "main",
                    rid,
                    parameters(cid),
                    cid,
                    representation,
                    key,
                )
        development = [r for r in summary["results"] if r["stage"] == "development"]
        assert len(development) == 324
        winner, metrics = choose_representation(development)
        save(
            OUT / "representation-selection.json",
            dict(
                locked_at=time.time(),
                winner=winner,
                metrics=metrics,
                episodes=324,
                tie_preference=config["representations"],
            ),
        )
        summary["selected_representation"] = winner
        print(json.dumps(dict(representation_selected=winner, metrics=metrics)), flush=True)
        nominees_b = {}
        untrained = {}
        for repeat in config["stage_b"]:
            rid = repeat["id"]
            neutral, scales, projection = initializations[rid]
            mapping = SearchMap(neutral, scales, winner, projection)
            z = np.random.default_rng(repeat["untrained_seed"]).normal(
                0, config["sigma"], mapping.dimension
            )
            untrained[rid] = mapping.decode(z)
            np.savez(
                OUT / "initialization" / (rid + "-untrained.npz"),
                genotype=z,
                parameters=untrained[rid],
            )
            for arm in repeat["order"]:
                key = f"{rid}-{winner}-{arm}"
                summary["phase"] = "training-" + key
                nominees_b[key] = search(pool, repeat, arm, mapping, config, panel, summary, record)
        assert len(nominees_b) == 15
        save(
            OUT / "selection-b.json",
            dict(locked_at=time.time(), representation=winner, nominees=nominees_b),
        )
        summary["phase"] = "evaluation"
        evaluate(
            "evaluation",
            panel["evaluation"],
            panel["evaluation_seeds"],
            "native",
            None,
            None,
            None,
            None,
            None,
        )
        for repeat in config["stage_b"]:
            rid = repeat["id"]
            evaluate(
                "evaluation",
                panel["evaluation"],
                panel["evaluation_seeds"],
                "untrained",
                rid,
                untrained[rid],
                None,
                winner,
                None,
            )
            for arm in config["arms"]:
                key = f"{rid}-{winner}-{arm}"
                cid = nominees_b[key]["id"]
                evaluate(
                    "evaluation",
                    panel["evaluation"],
                    panel["evaluation_seeds"],
                    arm,
                    rid,
                    parameters(cid),
                    cid,
                    winner,
                    key,
                )
        summary["phase"] = "diagnostic"
        for repeat in config["stage_b"]:
            rid = repeat["id"]
            for arm in ["main", "blind"]:
                key = f"{rid}-{winner}-{arm}"
                cid = nominees_b[key]["id"]
                evaluate(
                    "diagnostic",
                    panel["diagnostic"],
                    panel["evaluation_seeds"],
                    arm,
                    rid,
                    parameters(cid),
                    cid,
                    winner,
                    key,
                )
        assert len([r for r in summary["results"] if r["stage"] == "evaluation"]) == 1368
        assert len([r for r in summary["results"] if r["stage"] == "diagnostic"]) == 144
        assert len(summary["searches"]) == 24
        assert sum(s["actions"] for s in summary["searches"].values()) == 960000
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
