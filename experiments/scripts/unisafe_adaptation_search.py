"""Run recorded evolutionary, random and policy-gradient adaptation comparisons."""

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers.unisafe_search import AdapterSearch
from helpers.unisafe_source import ROOT, save_implementation


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/unisafe/adaptation-study.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["evolution", "random", "policy_gradient"])
    parser.add_argument("--search-seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--population", type=int, default=4)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    save_implementation(args.output)
    write(args.output / "protocol.json", protocol)
    write(args.output / "settings.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    all_runs, queries = [], Counter()
    started = time.perf_counter()
    for search_seed in args.search_seeds:
        for method in args.methods:
            optimizer = AdapterSearch(
                method, protocol["initial_search_space"], args.population, search_seed
            )
            directory = args.output / f"{method}-{search_seed}"
            directory.mkdir()
            order_rng = np.random.default_rng(search_seed + 917)
            for generation in range(args.generations):
                genes = optimizer.ask()
                episodes = []
                for seed in protocol["training_seeds"]:
                    for candidate in order_rng.permutation(len(genes)):
                        episodes.append({
                            "name": method, "candidate": int(candidate), "seed": seed,
                            "adapter": optimizer.decode(genes[candidate]),
                        })
                plan = {
                    "purpose": "Training candidates; every evaluation including elite reevaluation counts",
                    "method": method, "search_seed": search_seed, "generation": generation,
                    "record_cameras": False, "episodes": episodes,
                    "adapter_kind": protocol.get("adapter_kind", "release_delay"),
                }
                name = f"generation-{generation:02d}"
                path = directory / f"{name}.json"
                write(path, plan)
                output = directory / name
                print("Starting", method, search_seed, name, flush=True)
                with (directory / f"{name}.log").open("w") as log:
                    result = subprocess.run([
                        sys.executable, "-u", str(ROOT / "experiments/scripts/unisafe_adaptation_worker.py"),
                        "--plan", str(path.resolve()), "--output", str(output.resolve()),
                    ], check=False, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    write(directory / "error.json", {
                        "returncode": result.returncode, "generation": generation,
                        "partial_artifacts": str(output), "rerun_automatically": False,
                    })
                    raise RuntimeError(f"Worker failed: {output}; inspect and count partial artifacts")
                measured = json.loads((output / "summary.json").read_text())
                fitness = [float(np.mean([
                    e["utility"] for e in measured["episodes"] if e["job"]["candidate"] == i
                ])) for i in range(len(genes))]
                optimizer.tell(genes, fitness)
                write(directory / f"optimizer-{generation:02d}.json", optimizer.state())
                queries.update(measured["query_counts"])
                all_runs.append({
                    "method": method, "search_seed": search_seed, "generation": generation,
                    "genes": genes, "fitness": fitness,
                    "outcomes": dict(Counter(e["outcome"] for e in measured["episodes"])),
                    "episodes": len(measured["episodes"]),
                    "environment_steps": measured["environment_steps"],
                    "physics_steps": measured["physics_steps"],
                    "worker_elapsed_seconds": measured["elapsed_seconds"],
                    "directory": str(output),
                })
                write(args.output / "progress.json", all_runs)
                print("Finished", method, search_seed, name, "fitness", fitness, flush=True)
            write(directory / "finalists.json", optimizer.state())
    write(args.output / "summary.json", {
        "status": "complete", "runs": all_runs,
        "episodes": sum(r["episodes"] for r in all_runs),
        "environment_steps": sum(r["environment_steps"] for r in all_runs),
        "physics_steps": sum(r["physics_steps"] for r in all_runs),
        "query_counts": dict(queries), "pretrained_weight_updates": 0,
        "optimizer_updates": len(all_runs), "elapsed_seconds": time.perf_counter() - started,
        "worker_elapsed_seconds_overlapping": sum(r["worker_elapsed_seconds"] for r in all_runs),
        "selection_scope": "Training only; finalists require separate validation before test",
    })


if __name__ == "__main__":
    main()
