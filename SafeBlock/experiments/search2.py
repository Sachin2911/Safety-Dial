"""Improved search on the frozen Block Plucking system (protocol: SafeBlock/Search2.md).

    nohup /venv/main/bin/python SafeBlock/experiments/search2.py --phase all \
        --config SafeBlock/configs/search2-20261102-r0.json --output outputs/search2-20261102-r0 \
        > outputs/search2-20261102-r0/driver.log 2>&1 &

Ideas 1 to 4 of the 2 November discussion: eight common starts per candidate with a
lexicographic fitness key and continuous tie-breakers; the nominee chosen on a held-out
selection panel; elite re-evaluation by forced injection into CMA-ES; frozen-system
scores of each candidate's imagined block as adapter features (the selection2 adapter).
Phases in order: smoke, cma, selectA, finalA, random, selectB, finalB. Resumable.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from pilot_search import Lanes, find_attempt, make_optimizer, rows, status  # noqa: E402
from probe_common import ROOT, atomic_json, project_path, utc  # noqa: E402

PHASES = ["smoke", "cma", "selectA", "finalA", "random", "selectB", "finalB"]
FEATURES = [
    "retrieval_distance", "imagined_cost", "mean_arm", "closing_fraction", "progress",
    "imagined_min_q", "imagined_failure_probability", "imagined_disagreement",
]


def write_genotype(path, params, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, {"params": [float(v) for v in params], "features": FEATURES, "source": source, "utc": utc()})
    return path


def draw(seed, scale):
    import adapters

    return adapters.generic_genotype("selection2", seed, scale)


def measures(summary):
    """Per-episode measures behind the fitness key; a missing episode is the worst case."""
    if summary is None:
        return {"sustained": 0, "success": 0, "violated": 1, "max_top_xy": 1.0, "progress": 0.0, "completion_time": None, "missing": 1, "controls": None}
    success = int(bool(summary.get("safe_success")))
    return {
        "sustained": int(bool(summary.get("sustained_safe_success"))),
        "success": success,
        "violated": int(bool(summary.get("violation")) or bool(summary.get("post_success_violation"))),
        "max_top_xy": float(summary.get("max_top_base_xy_distance_m") or 0.0),
        "progress": min(float(summary.get("max_middle_displacement_m") or 0.0) / 0.07, 1.0),
        "completion_time": summary.get("completion_time_s") if success else None,
        "missing": 0,
        "controls": summary.get("controls"),
    }


def fitness_key(ms):
    """Lexicographic: sustained, success, fewer violations, smaller top-block excursion,
    more progress, faster completion. Higher is better."""
    n = len(ms)
    times = [m["completion_time"] for m in ms if m["completion_time"] is not None]
    return (
        sum(m["sustained"] for m in ms),
        sum(m["success"] for m in ms),
        -sum(m["violated"] for m in ms),
        -sum(m["max_top_xy"] for m in ms) / n,
        sum(m["progress"] for m in ms) / n,
        -(sum(times) / len(times) if times else 300.0),
    )


def ranks_of(keys):
    order = sorted(range(len(keys)), key=lambda i: keys[i], reverse=True)
    ranks = [0] * len(keys)
    for position, i in enumerate(order):
        ranks[i] = position
    return ranks


def evaluate(lanes, phase, prefix, genotype_files, starts, adapter="selection2"):
    """Run every (genotype, start) pair; return {name: [measures per start]} and raw results."""
    jobs = []
    for name, path in genotype_files.items():
        for s in starts:
            job = {"id": f"{prefix}-{name}-{s}", "phase": phase, "seed": s, "adapter": adapter if path else "none", "genotype_json": str(path) if path else None}
            jobs.append(job)
    results = lanes.run(phase, jobs)
    per = {name: [measures(results.get(f"{prefix}-{name}-{s}")) for s in starts] for name in genotype_files}
    return per, results


# ---------------------------------------------------------------- phases


def phase_smoke(cfg, lanes, root):
    """Zero-weight selection2 must reproduce the bare planner bitwise on this host.

    The bare reference is run on the same host (fresh process), so the check is
    host-local; agreement with the Stage 1 references from the pilot host is recorded
    separately as cross-host information.
    """
    zero = write_genotype(root / "genotypes" / "zero.json", [0.0] * 8, {"kind": "zero"})
    starts = cfg["smoke"]["starts"]
    per, results = evaluate(lanes, "smoke", "smoke-zero", {"bare": None, "z": zero}, starts)
    report = {"utc": utc(), "pairs": [], "passed": True, "cross_host": []}
    for s in starts:
        stage1 = find_attempt(project_path(cfg["smoke"]["reference_root"]) / f"S1C-{s}-bare")
        bare = find_attempt(root / "smoke" / f"smoke-zero-bare-{s}")
        if stage1 is not None and bare is not None:
            ra, rb = rows(stage1), rows(bare)
            n = min(len(ra), len(rb))
            report["cross_host"].append({"seed": s, "same_action": sum(ra[i]["executed_normalized_action"] == rb[i]["executed_normalized_action"] for i in range(n)), "same_camera": sum(ra[i]["camera_sha256"] == rb[i]["camera_sha256"] for i in range(n)), "controls": [len(ra), len(rb)]})
    for s in starts:
        ours = find_attempt(root / "smoke" / f"smoke-zero-z-{s}")
        ref = find_attempt(root / "smoke" / f"smoke-zero-bare-{s}")
        if ours is None or ref is None:
            report["pairs"].append({"seed": s, "error": "missing episode"})
            report["passed"] = False
            continue
        ra, rb = rows(ref), rows(ours)
        n = min(len(ra), len(rb))
        same_action = sum(ra[i]["executed_normalized_action"] == rb[i]["executed_normalized_action"] for i in range(n))
        same_camera = sum(ra[i]["camera_sha256"] == rb[i]["camera_sha256"] for i in range(n))
        same_shadow = sum(ra[i]["uncertainty"] == rb[i]["uncertainty"] and ra[i]["next_min_q"] == rb[i]["next_min_q"] for i in range(n))
        sr, so = results[f"smoke-zero-bare-{s}"], results[f"smoke-zero-z-{s}"]
        keys = ("safe_success", "sustained_safe_success", "violation", "post_success_violation")
        same_outcome = all(bool(sr.get(k)) == bool(so.get(k)) for k in keys) and sr.get("steps") == so.get("steps")
        changed = int(so.get("adapter_choice_changes") or 0)
        ok = same_outcome and len(ra) == len(rb) and same_action == n and same_camera == n and same_shadow == n and changed == 0
        report["pairs"].append({"seed": s, "controls_reference": len(ra), "controls_ours": len(rb), "same_action": same_action, "same_camera": same_camera, "same_shadow_signals": same_shadow, "same_outcome": same_outcome, "adapter_changes": changed, "passed": ok})
        report["passed"] = report["passed"] and ok
    atomic_json(root / "smoke" / "parity.json", report)
    return report["passed"]


def phase_cma(cfg, lanes, root):
    c = cfg["cma"]
    cdir = root / "cma"
    cdir.mkdir(parents=True, exist_ok=True)
    state_path, trace_path = cdir / "state.pkl", cdir / "trace.json"
    if state_path.exists() and trace_path.exists():
        es = pickle.loads(state_path.read_bytes())
        trace = json.loads(trace_path.read_text())
    else:
        es = make_optimizer(c)
        trace = {"settings": c, "generations": []}
    n_starts = cfg["starts_per_candidate"]
    elite = np.asarray(trace["generations"][-1]["elite"], dtype=float) if trace["generations"] else None
    for g in range(len(trace["generations"]) + 1, c["generations"] + 1):
        status(root, phase="cma", generation=g, episodes_done=lanes.done)
        asked_path = cdir / f"gen-{g:02d}-asked.json"
        if elite is not None:
            es.inject([elite], force=True)
        candidates = [np.asarray(x, dtype=float) for x in es.ask()]
        if asked_path.exists():
            candidates = [np.asarray(x, dtype=float) for x in json.loads(asked_path.read_text())["candidates"]]
        else:
            atomic_json(asked_path, {"generation": g, "candidates": [x.tolist() for x in candidates], "sigma": float(es.sigma), "mean": es.mean.tolist(), "injected_elite": None if elite is None else elite.tolist(), "utc": utc()})
        starts = cfg["training_starts"][(g - 1) * n_starts : g * n_starts]
        files = {f"c{i}": write_genotype(root / "genotypes" / f"cma-g{g:02d}-c{i}.json", x, {"kind": "cma", "generation": g, "candidate": i}) for i, x in enumerate(candidates)}
        per, _ = evaluate(lanes, "cma", f"cma-g{g:02d}", files, starts)
        keys = [fitness_key(per[f"c{i}"]) for i in range(len(candidates))]
        ranks = ranks_of(keys)
        es.tell(candidates, [float(r) for r in ranks])
        best = ranks.index(0)
        elite = candidates[best]
        injected_index = next((i for i, x in enumerate(candidates) if elite is not None and np.allclose(x, json.loads(asked_path.read_text()).get("injected_elite") or [np.nan] * len(x))), None)
        record = {
            "generation": g, "starts": starts, "candidates": [x.tolist() for x in candidates], "keys": [list(k) for k in keys], "ranks": ranks,
            "measures": {name: per[name] for name in per}, "elite_index": best, "elite": elite.tolist(), "elite_key": list(keys[best]),
            "injected_elite_index": injected_index, "mean_after": es.mean.tolist(), "sigma_after": float(es.sigma), "utc": utc(),
        }
        try:
            record["coordinate_std_after"] = np.asarray(es.stds).tolist()
        except Exception:  # noqa: BLE001
            pass
        trace["generations"].append(record)
        state_path.write_bytes(pickle.dumps(es))
        atomic_json(trace_path, trace)
    write_genotype(root / "genotypes" / "cma-final-mean.json", es.mean.tolist(), {"kind": "cma_final_mean", "generations": c["generations"]})
    write_genotype(root / "genotypes" / "cma-elite.json", elite.tolist(), {"kind": "cma_elite", "rule": "best of the last generation by the fitness key, after re-evaluation chains", "key": list(trace["generations"][-1]["elite_key"])})
    status(root, phase="cma", generation="done", episodes_done=lanes.done)


def phase_select(cfg, lanes, root, name, candidates, nominee_name):
    """Evaluate nominee candidates on the selection panel; the best key becomes the nominee."""
    status(root, phase=name, episodes_done=lanes.done)
    files = {k: root / "genotypes" / v for k, v in candidates.items()}
    for k, path in files.items():
        if not path.exists():
            raise SystemExit(f"{path.name} is missing: run its search phase first")
    per, _ = evaluate(lanes, "select", name, files, cfg["selection_panel"])
    keys = {k: fitness_key(per[k]) for k in files}
    winner = max(keys, key=lambda k: keys[k])
    params = json.loads(files[winner].read_text())["params"]
    write_genotype(root / "genotypes" / nominee_name, params, {"kind": nominee_name.replace(".json", ""), "chosen": winner, "candidates": {k: list(v) for k, v in keys.items()}, "panel": cfg["selection_panel"]})
    atomic_json(root / "select" / f"{name}.json", {"utc": utc(), "winner": winner, "keys": {k: list(v) for k, v in keys.items()}, "measures": per})
    status(root, phase=name, episodes_done=lanes.done, winner=winner)


def phase_final(cfg, lanes, root, name, arms):
    status(root, phase=name, episodes_done=lanes.done)
    files = {}
    for arm in arms:
        if arm == "native":
            files[arm] = None
        elif arm == "untrained":
            path = root / "genotypes" / "untrained.json"
            if not path.exists():
                write_genotype(path, draw(cfg["untrained"]["seed"], cfg["untrained"]["scale"]), {"kind": "untrained", "generator_seed": cfg["untrained"]["seed"], "scale": cfg["untrained"]["scale"]})
            files[arm] = path
        else:
            files[arm] = root / "genotypes" / {"cma": "cma-nominee.json", "random": "random-nominee.json"}[arm]
            if not files[arm].exists():
                raise SystemExit(f"{files[arm].name} is missing")
    per, _ = evaluate(lanes, "final", "final", files, cfg["final_panel"])
    receipt = {"utc": utc(), "arms": {arm: {k: sum(m[k] for m in per[arm]) for k in ("sustained", "success", "violated", "missing")} for arm in arms}}
    atomic_json(root / "final" / f"{name}.json", receipt)
    status(root, phase=name, episodes_done=lanes.done, result=receipt["arms"])


def phase_random(cfg, lanes, root):
    r = cfg["random_search"]
    n_starts = cfg["starts_per_candidate"]
    status(root, phase="random", episodes_done=lanes.done)
    table = []
    draws = list(range(r["first_seed"], r["first_seed"] + r["draws"]))
    block_size = cfg["cma"]["population"]
    for b in range(0, len(draws), block_size):
        block = draws[b : b + block_size]
        g = b // block_size + 1
        starts = cfg["training_starts"][(g - 1) * n_starts : g * n_starts]
        files = {f"d{seed}": write_genotype(root / "genotypes" / f"random-{seed}.json", draw(seed, r["scale"]), {"kind": "random", "generator_seed": seed, "scale": r["scale"], "block": g}) for seed in block}
        per, _ = evaluate(lanes, "random", f"random-b{g:02d}", files, starts)
        for seed in block:
            ms = per[f"d{seed}"]
            table.append({"generator_seed": seed, "block": g, "starts": starts, "key": list(fitness_key(ms)), "measures": ms, "params": json.loads(files[f"d{seed}"].read_text())["params"]})
    table.sort(key=lambda row: (tuple(row["key"]), -row["generator_seed"]), reverse=True)
    top = table[:2]
    for i, row in enumerate(top):
        write_genotype(root / "genotypes" / f"random-top{i + 1}.json", row["params"], {"kind": "random_top", "rank": i + 1, "generator_seed": row["generator_seed"], "key": row["key"]})
    atomic_json(root / "random" / "search.json", {"utc": utc(), "draws": table, "top": [row["generator_seed"] for row in top]})
    status(root, phase="random", episodes_done=lanes.done, top=[row["generator_seed"] for row in top])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all"] + PHASES, default="all")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", type=int, help="override execution.lanes for this host (recorded)")
    args = parser.parse_args()
    cfg = json.loads((args.config if args.config.is_absolute() else ROOT / args.config).read_text())
    if args.lanes:
        cfg["execution"]["lanes"] = args.lanes
    root = project_path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / f"driver-start-{int(time.time())}.json", {"utc": utc(), "phase": args.phase, "config": str(args.config), "python": sys.executable, "lanes": cfg["execution"]["lanes"]})
    lanes = Lanes(cfg, root)
    for phase in (PHASES if args.phase == "all" else [args.phase]):
        tick = time.perf_counter()
        if phase == "smoke":
            if not phase_smoke(cfg, lanes, root):
                status(root, phase="smoke", failed=True, episodes_done=lanes.done)
                print("PARITY SMOKE FAILED; stopping", flush=True)
                sys.exit(2)
        elif phase == "cma":
            phase_cma(cfg, lanes, root)
        elif phase == "selectA":
            phase_select(cfg, lanes, root, "selectA", {"final_mean": "cma-final-mean.json", "elite": "cma-elite.json"}, "cma-nominee.json")
        elif phase == "finalA":
            phase_final(cfg, lanes, root, "finalA", ["native", "untrained", "cma"])
        elif phase == "random":
            phase_random(cfg, lanes, root)
        elif phase == "selectB":
            phase_select(cfg, lanes, root, "selectB", {"top1": "random-top1.json", "top2": "random-top2.json"}, "random-nominee.json")
        elif phase == "finalB":
            phase_final(cfg, lanes, root, "finalB", ["random"])
        atomic_json(root / f"phase-{phase}.json", {"utc": utc(), "phase": phase, "seconds": time.perf_counter() - tick, "episodes_done_total": lanes.done})
        print(json.dumps({"phase": phase, "done": True, "seconds": round(time.perf_counter() - tick), "episodes": lanes.done}), flush=True)
    status(root, phase="complete" if args.phase == "all" else args.phase, episodes_done=lanes.done)


if __name__ == "__main__":
    main()
