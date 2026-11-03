"""Two paired development repeats of CMA4, CMA8 and random search on the L4."""
import collections
import json
import os
from pathlib import Path
import pickle
import socket
import subprocess
import sys
import time
import traceback
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/evolution-budget-20261102"
OUT = BASE / "run-v1"
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "experiments/scripts")]
import adapter_pilot_v2 as qualified  # noqa: E402
import run_matched_ppo as prior  # noqa: E402
from run_mechanism import spec  # noqa: E402
from helpers.pilot_common import digest, save  # noqa: E402
from helpers.replication_budget import allocation, eligible  # noqa: E402
from helpers.mechanism import SearchMap, objective  # noqa: E402

def search(pool, repeat, arm, mapping, config, panel, summary, record):
    from helpers.pilot_optimizer import make_optimizer, tell_ordered

    population = 4 if arm == "cma4" else 8
    rid = repeat["id"]
    representation = mapping.representation
    optimizer = "random" if arm == "random" else "cma"
    key = f"{rid}-{representation}-{arm}"
    es = make_optimizer(mapping.dimension, config["sigma"], repeat["cma_seed"], population=population, generations=5000)
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
            else rng.normal(0, config["sigma"], (population, mapping.dimension))
        )
        jobs = []
        for i, genotype in enumerate(genotypes):
            cid = f"{key}-{generation * population + i:05d}"
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
        while cursor < population and spent < budget:
            quotas = allocation(budget - spent, population - cursor, workers=population)
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
        complete_generation = len(scores) == population
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
        population=population,
        eligible=sum(c["eligible"] for c in candidates),
        partial=sum(not c["eligible"] for c in candidates),
        nominee=best,
    )
    save(folder / "complete.json", result)
    summary["searches"][key] = result
    return best


def execute():
    import torch
    from helpers.pilot_optimizer import make_optimizer, tell_ordered
    from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter

    config = json.loads((ROOT / "configs/evolution-budget/run.json").read_text())
    panel = json.loads((ROOT / "configs/evolution-budget/panel.json").read_text())
    assert socket.gethostname() == config["host"]
    assert torch.__version__ == "2.4.1+cu124"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    deadline = started + config["run_seconds"]
    qualified.OUT = prior.OUT = OUT
    qualified.freeze()
    frozen = json.loads((OUT / "freeze.json").read_text())
    files = [ROOT / "EvolutionBudgetStudy.md", Path(__file__),
             ROOT / "experiments/scripts/audit_evolution_budget.py",
             ROOT / "experiments/scripts/run_matched_ppo.py",
             ROOT / "experiments/scripts/run_mechanism.py",
             ROOT / "experiments/helpers/ppo_adapter.py",
             ROOT / "experiments/scripts/ppo_worker.py",
             ROOT / "experiments/helpers/mechanism.py",
             ROOT / "experiments/helpers/replication_budget.py",
             *sorted((ROOT / "configs/evolution-budget").glob("*.json"))]
    for path in files:
        rel = str(path.relative_to(ROOT))
        frozen["sources"][rel] = digest(path)
        target = OUT / "sources" / rel
        target.parent.mkdir(exist_ok=True, parents=True)
        target.write_bytes(path.read_bytes())
    frozen.update(instance_id=os.environ.get("RUNPOD_POD_ID"), machine_id=None,
                  host=socket.gethostname(), study="population-size development comparison",
                  action_cap=config["action_cap"], deadline=time.time()+config["run_seconds"])
    save(OUT / "freeze.json", frozen)
    for population in [4, 8]:
        es = make_optimizer(132, .5, 17900+population, population=population, generations=10)
        vectors = es.ask()
        assert len(vectors) == population
        tell_ordered(es, vectors, [-float(np.square(v).sum()) for v in vectors])
        assert es.countiter == 1 and es.countevals == population
        assert len(es.ask()) == population
        try:
            tell_ordered(es, vectors[:-1], [1.]*(population-1))
        except ValueError:
            pass
        else:
            raise AssertionError("Incomplete population was accepted")
    save(OUT / "optimizer-checks.json", dict(passed=True, populations=[4, 8]))
    summary = dict(status="running", phase="engineering", results=[], candidates=[], searches={},
                   host=dict(name=config["host"], action_cap=config["action_cap"]), resources={})
    counts = collections.Counter()
    for name in ["candidates", "initialization"]:
        (OUT / name).mkdir()

    def record(rows):
        summary["results"].extend(rows)
        for row in rows:
            counts.update(row["resources"])
        summary.update(resources=dict(counts), seconds=time.monotonic()-started)
        assert counts["environment_actions"] <= config["action_cap"]
        save(OUT / "summary.json", summary)

    def evaluate(arm, rid, parameters, cid=None, search_key=None):
        for case in panel["evaluation"]:
            for seed in panel["evaluation_seeds"]:
                job = spec("development", case, arm, seed, f"dev-{rid or 'shared'}-{arm}",
                           rid, cid, "head", search_key)
                record([pool.episode(job, parameters)])

    pool = None
    try:
        pool = prior.SerialPool(deadline)
        actor = RecurrentActionAdapter(AdapterSpec(396, 2, initialization_seed=17101)).eval()
        neutral = actor.parameters_vector()
        np.savez(OUT / "initialization/engineering.npz", neutral=neutral)
        for case in panel["training"]:
            names = []
            for arm, parameters in [("native", None), ("neutral", neutral)]:
                job = spec("engineering", case, arm, panel["training_seed"], "parity-"+arm, "engineering")
                names.append(job["id"])
                record([pool.episode(job, parameters)])
            qualified.parity(*names)
        save(OUT / "engineering.json", dict(passed=True, episodes=16))
        native_done = False
        for repeat in config["repeats"]:
            rid = repeat["id"]
            actor = RecurrentActionAdapter(AdapterSpec(396, 2, initialization_seed=repeat["initialization_seed"]))
            neutral, scales = actor.parameters_vector(), actor.coordinate_scales()
            np.savez(OUT / "initialization" / (rid+".npz"), neutral=neutral, scales=scales)
            mapping = SearchMap(neutral, scales, "head")
            z = np.random.default_rng(repeat["untrained_seed"]).normal(0, .5, mapping.dimension)
            untrained = mapping.decode(z)
            np.savez(OUT / "initialization" / (rid+"-untrained.npz"), genotype=z, parameters=untrained)
            nominees = {}
            for arm in repeat["order"]:
                key = f"{rid}-head-{arm}"
                summary["phase"] = "training-"+key
                nominees[key] = search(pool, repeat, arm, mapping, config, panel, summary, record)
            save(OUT / ("selection-"+rid+".json"), dict(locked_at=time.time(), nominees=nominees))
            summary["phase"] = "development-"+rid
            if not native_done:
                evaluate("native", None, None)
                native_done = True
            evaluate("untrained", rid, untrained)
            for arm in ["cma8", "cma4", "random"]:
                key = f"{rid}-head-{arm}"
                cid = nominees[key]["id"]
                parameters = np.load(OUT / "candidates" / (cid+".npz"))["parameters"].copy()
                evaluate(arm, rid, parameters, cid, key)
        assert sum(v["actions"] for v in summary["searches"].values()) == 240000
        summary["status"] = "complete"
    except BaseException:
        summary.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if pool is not None:
            pool.close()
        summary.update(resources=qualified.resource_summary(), seconds=time.monotonic()-started)
        save(OUT / "summary.json", summary)


def main():
    assert socket.gethostname() == "8cf3a706cdd1"
    if len(sys.argv) == 1:
        BASE.mkdir(exist_ok=False, parents=True)
        with (BASE / "run.log").open("xb") as log:
            child = subprocess.Popen([sys.executable, __file__, "execute"], cwd=ROOT,
                stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True,
                env=dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1"))
        save(BASE / "launch.json", dict(pid=child.pid, started=time.time(), rental_shutdown="manual"))
        print(json.dumps(dict(pid=child.pid)), flush=True)
    else:
        assert sys.argv[1:] == ["execute"]
        execute()
        subprocess.run([sys.executable, str(ROOT / "experiments/scripts/audit_evolution_budget.py")],
                       cwd=ROOT, check=True)
        save(BASE / "pipeline-complete.json", dict(completed=time.time()))


if __name__ == "__main__":
    main()
