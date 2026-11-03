"""Bounded eight-worker recurrent-adapter pilot, on the authorized Vast host only."""

import argparse
import collections
import importlib.metadata
import inspect
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import queue
import socket
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/adapter-pilot-20261101/run-v1"
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.pilot_common import digest, save, score  # noqa: E402


def freeze():
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    import anchored_pipeline as ap

    OUT.mkdir(parents=True, exist_ok=False)
    sources = [
        ROOT / "AdapterPilot.md",
        *sorted((ROOT / "configs/adapter-pilot").glob("*")),
        ROOT / "configs/tworoom-scalers.json",
        Path(__file__),
        ROOT / "experiments/scripts/adapter_worker.py",
        ROOT / "experiments/scripts/audit_adapter_pilot.py",
        ROOT / "experiments/scripts/plot_adapter_pilot.py",
        ROOT / "experiments/scripts/anchored_pipeline.py",
        ROOT / "experiments/helpers/recurrent_adapter.py",
        ROOT / "experiments/helpers/pilot_common.py",
        ROOT / "experiments/helpers/qualification.py",
        ROOT / "experiments/tests/test_recurrent_adapter.py",
        *sorted((ap.REF / "anchored_planning").glob("*.py")),
        ap.REF / "protocols/scalers.json",
    ]
    records = {}
    for path in sources:
        name = str(path.relative_to(ROOT))
        records[name] = digest(path)
        destination = OUT / "sources" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    memory = ap.BASE / "memory-v1"
    old = json.loads((memory / "freeze.json").read_text())["sources"]
    for name in records:
        if name in old:
            assert records[name] == old[name], name
    assert digest(ROOT / "assets/weights.pt") == (
        "566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd"
    )
    assert digest(memory / "latents.npy") == (
        "e9ccf6a75ce56a5463fa75b6e63402971a366e91d4ea911cdc664b14c5923ab0"
    )
    for name in ["panel.json", "complete.json", "freeze.json"]:
        (OUT / ("memory-" + name)).write_bytes((memory / name).read_bytes())
    native_source = Path(inspect.getfile(TwoRoomEnv))
    assert digest(native_source) == (
        "5e1d392de5b02472062dbe872aded67fd465fcc8f7eaa1c02a753b2fc31c61f0"
    ), "Native environment source differs from the qualified baseline"
    (OUT / "sources/native-env.py").write_bytes(native_source.read_bytes())
    save(
        OUT / "freeze.json",
        dict(
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            sources=records,
            instance_id=58676387,
            machine_id=19781,
            environment_sha256=digest(native_source),
            checkpoint_sha256=digest(ROOT / "assets/weights.pt"),
            memory_sha256=digest(memory / "latents.npy"),
            cma_version=importlib.metadata.version("cma"),
            gpu_inventory=subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ),
        ),
    )


class Pool:
    def __init__(self, deadline):
        from adapter_worker import worker

        context = mp.get_context("spawn")
        self.output = context.Queue()
        self.inputs = [context.Queue() for _ in range(8)]
        self.processes = [
            context.Process(
                target=worker,
                args=(i, self.inputs[i], self.output, str(OUT), deadline),
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

    def receive(self):
        while time.monotonic() < self.deadline:
            try:
                result = self.output.get(timeout=5)
            except queue.Empty:
                if any(p.exitcode is not None for p in self.processes):
                    raise RuntimeError("Worker exited unexpectedly; inspect all partial evidence")
                continue
            if result["kind"] == "error":
                raise RuntimeError(result["error"])
            return result
        raise TimeoutError("Frozen job deadline reached")

    def batch(self, jobs):
        assert len(jobs) == len({j["device"] for j in jobs})
        for job in jobs:
            self.inputs[job["device"]].put(job)
        result = {}
        for _ in jobs:
            item = self.receive()
            assert item["kind"] == "result"
            result[item["job"]] = item["results"]
        return result

    def close(self):
        for incoming in self.inputs:
            incoming.put(None)
        for process in self.processes:
            if process.pid is None:
                continue
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


def episode_spec(stage, case, arm, seed, prefix):
    return dict(id=f"{prefix}_{case['id']}_{seed}", stage=stage, case=case, arm=arm, seed=seed)


def parity(left, right):
    with np.load(OUT / "episodes" / left / "trace.npz") as a:
        with np.load(OUT / "episodes" / right / "trace.npz") as b:
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
                np.testing.assert_array_equal(a[key], b[key], err_msg=f"{left}:{right}:{key}")
    for key in [
        "environment_actions",
        "model_batch_calls",
        "candidate_model_steps",
        "encoder_frames",
        "replans",
        "retrieval_queries",
    ]:
        a = json.loads((OUT / "episodes" / left / "result.json").read_text())
        b = json.loads((OUT / "episodes" / right / "result.json").read_text())
        assert a["resources"].get(key, 0) == b["resources"].get(key, 0)


def resource_summary():
    counts = collections.Counter()
    executions = list((OUT / "episodes").glob("*/execution.json"))
    for path in executions:
        counts.update(json.loads(path.read_text())["resources"])
    # Preserve attempted work from a killed process that has no execution footer.
    for path in (OUT / "episodes").glob("*/live.json"):
        if not path.with_name("execution.json").exists():
            counts.update(json.loads(path.read_text())["resources"])
    return dict(counts)


def run():
    import cma
    from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter
    from helpers.recurrent_adapter import standardized_parameters

    config = json.loads((ROOT / "configs/adapter-pilot/run.json").read_text())
    assert socket.gethostname() == config["runtime_hostname"], "Fresh authorized Vast instance only"
    assert importlib.metadata.version("cma") == "4.4.4"
    assert (OUT.parent / "software-checks-passed").exists(), "Run the rented-host tests first"
    panel = json.loads((ROOT / "configs/adapter-pilot/panel.json").read_text())
    freeze()
    started = time.monotonic()
    deadline = started + config["run_seconds"]
    summary = dict(status="running", phase="initializing", results=[], candidates=[], resources={})
    save(OUT / "summary.json", summary)
    actor = RecurrentActionAdapter(AdapterSpec(396, 2))
    neutral, scales = actor.parameters_vector(), actor.coordinate_scales()
    np.savez(OUT / "initialization.npz", neutral=neutral, scales=scales)
    save(OUT / "adapter-spec.json", actor.specification())
    untrained_z = np.random.default_rng(config["untrained_seed"]).normal(
        0, config["sigma"], len(neutral)
    )
    untrained = standardized_parameters(neutral, scales, untrained_z)
    np.savez(OUT / "untrained.npz", parameters=untrained, genotype=untrained_z)
    pool = None

    def record(results):
        summary["results"].extend(results)
        summary["resources"] = resource_summary()
        summary["seconds"] = time.monotonic() - started
        assert summary["resources"].get("environment_resets", 0) <= config["max_episodes"]
        assert summary["resources"].get("environment_actions", 0) <= 118400
        save(OUT / "summary.json", summary)

    try:
        pool = Pool(deadline)
        summary["phase"] = "engineering"
        # Native and neutral comparisons run on the same device for each layout.
        for arm, parameters in [("native", None), ("neutral", neutral)]:
            jobs = [
                dict(
                    id=f"parity-{arm}-{i}",
                    device=i,
                    parameters=parameters,
                    episodes=[
                        episode_spec(
                            "engineering", case, arm, panel["training_seed"], f"parity-{arm}"
                        ),
                    ],
                )
                for i, case in enumerate(panel["training"])
            ]
            result = pool.batch(jobs)
            for job in jobs:
                record(result[job["id"]])
        pairs = []
        for case in panel["training"]:
            suffix = f"{case['id']}_{panel['training_seed']}"
            pair = [f"parity-native_{suffix}", f"parity-neutral_{suffix}"]
            parity(*pair)
            pairs.append(pair)
        first = panel["training"][0]
        jobs = [
            dict(
                id=f"cross-{i}",
                device=i,
                parameters=None,
                episodes=[
                    episode_spec(
                        "engineering", first, "native", panel["training_seed"], f"cross-{i}"
                    ),
                ],
            )
            for i in range(1, 8)
        ]
        result = pool.batch(jobs)
        for job in jobs:
            record(result[job["id"]])
            pair = [
                f"parity-native_{first['id']}_{panel['training_seed']}",
                job["episodes"][0]["id"],
            ]
            parity(*pair)
            pairs.append(pair)
        save(OUT / "parity.json", dict(passed=True, pairs=pairs, episodes=23))
        jobs = []
        for i, case in enumerate(panel["evaluation"]):
            for j, arm in enumerate(["safe", "risky"]):
                jobs.append(
                    dict(
                        id=f"witness-{i}-{arm}",
                        device=2 * i + j,
                        parameters=None,
                        episodes=[
                            episode_spec(
                                "witness", case, arm, panel["witness_seed"], "witness-" + arm
                            )
                        ],
                    )
                )
        result = pool.batch(jobs)
        for job in jobs:
            record(result[job["id"]])
        for values in result.values():
            assert values[0]["success"]
            assert values[0]["violation"] == (values[0]["arm"] == "risky")
        save(OUT / "witnesses.json", dict(passed=True, episodes=8))

        es = cma.CMAEvolutionStrategy(
            np.zeros(len(neutral)),
            config["sigma"],
            dict(
                CMA_diagonal=True,
                popsize=8,
                seed=config["cma_seed"],
                verbose=-9,
                verb_log=0,
                verb_disp=0,
                maxiter=4,
            ),
        )
        random_rng = np.random.default_rng(config["random_seed"])
        tie_rng = np.random.default_rng(config["tie_seed"])
        nominees = {}
        for optimizer in ["cma", "random"]:
            summary["phase"] = optimizer
            optimizer_rows = []
            for generation in range(4):
                genotypes = (
                    es.ask()
                    if optimizer == "cma"
                    else random_rng.normal(0, config["sigma"], (8, len(neutral)))
                )
                jobs = []
                for i, genotype in enumerate(genotypes):
                    candidate = f"{optimizer}-{generation * 8 + i:03d}"
                    parameters = standardized_parameters(neutral, scales, genotype)
                    path = OUT / "candidates" / (candidate + ".npz")
                    path.parent.mkdir(exist_ok=True)
                    np.savez(path, genotype=genotype, parameters=parameters)
                    jobs.append(
                        dict(
                            id=candidate,
                            device=i,
                            parameters=parameters,
                            episodes=[
                                episode_spec(
                                    "training", case, optimizer, panel["training_seed"], candidate
                                )
                                for case in panel["training"]
                            ],
                        )
                    )
                results = pool.batch(jobs)
                scores = []
                for job in jobs:
                    rows = results[job["id"]]
                    record(rows)
                    entry = dict(
                        id=job["id"], optimizer=optimizer, generation=generation, **score(rows)
                    )
                    scores.append(entry["score"])
                    optimizer_rows.append(entry)
                    summary["candidates"].append(entry)
                order = tie_rng.permutation(8)
                if optimizer == "cma":
                    es.tell([genotypes[i] for i in order], [-scores[i] for i in order])
                    with (OUT / f"cma-state-{generation:02d}.pkl").open("wb") as stream:
                        pickle.dump(es, stream)
                save(
                    OUT / f"{optimizer}-generation-{generation:02d}.json",
                    dict(
                        generation=generation,
                        scores=scores,
                        presentation_order=order.tolist(),
                        all_tied=len(set(scores)) == 1,
                        updates=int(es.countiter) if optimizer == "cma" else 0,
                    ),
                )
                save(OUT / "summary.json", summary)
                print(
                    json.dumps(
                        dict(
                            optimizer=optimizer,
                            generation=generation,
                            scores=scores,
                            episodes=len(summary["results"]),
                        )
                    ),
                    flush=True,
                )
            best = max(optimizer_rows, key=lambda row: row["score"])
            nominees[optimizer] = best
        # This file must precede every native or learned final-evaluation episode.
        save(
            OUT / "selection.json",
            dict(
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                nominees=nominees,
                rule="maximum training score; earliest candidate on exact ties",
            ),
        )
        parameters_by_arm = dict(native=None, untrained=untrained)
        for name, row in nominees.items():
            with np.load(OUT / "candidates" / (row["id"] + ".npz")) as saved:
                parameters_by_arm[name] = saved["parameters"].copy()
        summary["phase"] = "evaluation"
        evaluation = [
            (case, seed) for case in panel["evaluation"] for seed in panel["evaluation_seeds"]
        ]
        for arm, parameters in parameters_by_arm.items():
            for offset in range(0, len(evaluation), 8):
                jobs = [
                    dict(
                        id=f"eval-{arm}-{offset + i}",
                        device=(offset + i) % 8,
                        parameters=parameters,
                        episodes=[
                            episode_spec("evaluation", case, arm, seed, "eval-" + arm),
                        ],
                    )
                    for i, (case, seed) in enumerate(evaluation[offset : offset + 8])
                ]
                result = pool.batch(jobs)
                for job in jobs:
                    record(result[job["id"]])
        summary["status"] = "complete"
        summary["evaluation"] = {
            arm: score(
                [r for r in summary["results"] if r["stage"] == "evaluation" and r["arm"] == arm]
            )
            for arm in parameters_by_arm
        }
    except BaseException:
        summary["status"] = "failed"
        summary["failure"] = traceback.format_exc()
        raise
    finally:
        if pool is not None:
            pool.close()
        summary["resources"] = resource_summary()
        summary["seconds"] = time.monotonic() - started
        save(OUT / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.parse_args()
    run()
