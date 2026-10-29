"""Prospective stage freezes, development selection, and paired final comparison."""

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from panels_lewm18 import CONFIG, OUT, ROOT, freeze, params
from helpers.dsrl06_infra import atomic_json, utc


def read(path):
    return json.loads(Path(path).read_text())


def results(manifest):
    rows = []
    for job in read(manifest)["jobs"]:
        path = OUT / "attempts" / job["id"] / "result.json"
        if not path.exists():
            raise RuntimeError(f"Missing result: {job['id']}")
        row = read(path)
        if row["error"] or row["censored"]:
            raise RuntimeError(f"Invalid execution requires a new frozen repair: {job['id']}")
        rows.append(row)
    return rows


def run_stage(manifest, lanes=2):
    started = time.perf_counter()
    processes, logs = [], []
    name = read(manifest)["name"]
    for lane in range(lanes):
        log = (OUT / "logs" / f"{name}-{lane}.log").open("a")
        logs.append(log)
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "experiments/scripts/worker_lewm18.py"),
                    str(manifest),
                    "--lane",
                    str(lane),
                    "--lanes",
                    str(lanes),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        )
    codes = [p.wait() for p in processes]
    for log in logs:
        log.close()
    atomic_json(
        OUT / f"{name}-execution.json",
        {
            "utc": utc(),
            "wall_seconds": time.perf_counter() - started,
            "lanes": lanes,
            "return_codes": codes,
        },
    )
    if any(codes):
        raise RuntimeError(f"Worker failure in {name}")
    return results(manifest)


def job(identifier, stage, kind, case, seed, method, margin, settings, **extra):
    return {
        "id": identifier,
        "stage": stage,
        "env": kind,
        "case": case,
        "seed": seed,
        "method": method,
        "margin": margin,
        "params": settings,
        **extra,
    }


def fixed_jobs(kind, protocol, panel):
    jobs = []
    for mi, margin in enumerate(protocol["dials"][kind]):
        for ci, case in enumerate(panel["development"]):
            for seed in protocol["development"]["planner_seeds"]:
                modes = list(enumerate(protocol["development"]["fixed_modes"]))
                np.random.default_rng(seed + ci + 17 * mi).shuffle(modes)
                for index, mode in modes:
                    jobs.append(
                        job(
                            f"D-{kind}-m{mi}-c{ci}-s{seed}-k{index}",
                            "D",
                            kind,
                            case,
                            seed,
                            "fixed_candidate",
                            margin,
                            params(mode),
                            candidate=index,
                        )
                    )
    return jobs


def select(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["job"]["candidate"]].append(row)
    ranking = []
    for index, group in groups.items():
        ranking.append(
            {
                "candidate": index,
                "n": len(group),
                "safe": sum(r["requested_safe_completion"] for r in group),
                "J_safe_sum": sum(r["J_safe"] for r in group),
                "predictor_steps": sum(r["resources"].get("predictor_steps", 0) for r in group),
                "params": group[0]["job"]["params"],
            }
        )
    if len({r["n"] for r in ranking}) != 1:
        raise RuntimeError("Unbalanced selection panel")
    ranking.sort(key=lambda r: (-r["safe"], r["J_safe_sum"], r["predictor_steps"], r["candidate"]))
    return {"selected": ranking[0], "ranking": ranking}


def admit(protocol, kinds=("cube", "pusht")):
    suffix = f"-{kinds[0]}" if len(kinds) == 1 else ""
    path = OUT / f"schedule-admission{suffix}.json"
    if path.exists():
        return read(path)
    if any(list((OUT / "attempts").glob(f"D-{kind}-*/result.json")) for kind in kinds):
        raise RuntimeError("Schedule must be selected before development outcomes")
    rental = read(ROOT / "progress/rental-lewm18-20261026.json")
    deadline = dt.datetime.fromisoformat(rental["deadline_utc"]).timestamp()
    measured = {}
    for kind in kinds:
        rows = results(OUT / f"engineering-{kind}-freeze.json")
        # Scale early completions to the full horizon. Include failed-plan wall time.
        measured[kind] = max(
            (r["episode_seconds"] - r["native_parity_seconds"]) * 50 / max(r["actions"], 5)
            for r in rows
        )
    candidates = []
    for tier in protocol["final"]["tiers"]:
        counts = {}
        for kind in kinds:
            counts[kind] = 216 + 72 + 12 + tier["cases_per_env"] * len(tier["seeds"]) * 10
        projected = 1.5 * (
            sum(counts[k] * measured[k] for k in measured) / 2 + 8 * sum(measured.values())
        )
        candidates.append(
            {
                "tier": tier,
                "projected_seconds": projected,
                "remaining_seconds": deadline - time.time() - 2400,
                "jobs_per_environment": counts,
            }
        )
    selected = next(
        (r for r in candidates if r["projected_seconds"] <= r["remaining_seconds"]), None
    )
    if selected is None:
        atomic_json(
            OUT / f"schedule-not-admitted{suffix}.json",
            {
                "utc": utc(),
                "measurements": measured,
                "candidates": candidates,
                "reason": "Prospective workload amendment required before development; no efficacy gate.",
            },
        )
        raise RuntimeError("Neither frozen tier fits remaining measured time")
    admission = {
        "utc": utc(),
        "selected": selected,
        "measurements": measured,
        "candidates": candidates,
        "criterion": "Timing only, no development outcomes",
    }
    atomic_json(path, admission)
    return admission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("admit", "run"))
    parser.add_argument("--env", choices=("cube", "pusht"))
    args = parser.parse_args()
    protocol = read(CONFIG / "protocol.json")
    kinds = (args.env,) if args.env else ("cube", "pusht")
    suffix = f"-{args.env}" if args.env else ""
    admission = admit(protocol, kinds)
    if args.action == "admit":
        print(json.dumps(admission), flush=True)
        return
    tier = admission["selected"]["tier"]
    panels = {kind: read(OUT / f"{kind}-panels.json") for kind in kinds}
    stages = {}
    fixed = {}
    d_jobs = sum([fixed_jobs(kind, protocol, panels[kind]) for kind in panels], [])
    manifest = freeze(f"development-fixed{suffix}", d_jobs, {"admission": admission})
    stages["fixed"] = str(manifest)
    rows = run_stage(manifest)
    for kind in panels:
        fixed[kind] = {}
        for margin in protocol["dials"][kind]:
            fixed[kind][str(margin)] = select(
                [r for r in rows if r["job"]["env"] == kind and r["job"]["margin"] == margin]
            )
    atomic_json(OUT / f"fixed-selection{suffix}.json", {"utc": utc(), "environments": fixed})

    f_jobs = []
    for kind, panel in panels.items():
        for mi, margin in enumerate(protocol["dials"][kind]):
            base = fixed[kind][str(margin)]["selected"]["params"]
            for ci, case in enumerate(panel["development"]):
                for seed in protocol["development"]["feedback_seed"]:
                    presets = list(enumerate(protocol["development"]["feedback_presets"]))
                    np.random.default_rng(seed + ci + mi).shuffle(presets)
                    for index, preset in presets:
                        settings = dict(
                            base,
                            **preset,
                            adaptive=True,
                            repair_horizon=min(base["horizon"], 2),
                            stall_horizon=3,
                            exploration_multiplier=2,
                            stall_retain=0.5,
                        )
                        f_jobs.append(
                            job(
                                f"F-{kind}-m{mi}-c{ci}-s{seed}-k{index}",
                                "F",
                                kind,
                                case,
                                seed,
                                "feedback_candidate",
                                margin,
                                settings,
                                candidate=index,
                            )
                        )
    manifest = freeze(f"development-feedback{suffix}", f_jobs, {"fixed_selection": fixed})
    stages["feedback"] = str(manifest)
    rows = run_stage(manifest)
    feedback = {}
    for kind in panels:
        feedback[kind] = {}
        for margin in protocol["dials"][kind]:
            feedback[kind][str(margin)] = select(
                [r for r in rows if r["job"]["env"] == kind and r["job"]["margin"] == margin]
            )
    selection = {"utc": utc(), "fixed": fixed, "feedback": feedback, "tier": tier}
    atomic_json(OUT / f"final-selection{suffix}.json", selection)
    final_jobs = []
    for kind, panel in panels.items():
        for ci, case in enumerate(panel["final"][: tier["cases_per_env"]]):
            for seed in tier["seeds"]:
                block = [
                    job(
                        f"H-{kind}-c{ci}-s{seed}-native",
                        "H",
                        kind,
                        case,
                        seed,
                        "native",
                        0,
                        params(checked=False),
                        save_frames=ci < 2 and seed == tier["seeds"][0],
                    )
                ]
                for mi, margin in enumerate(protocol["dials"][kind]):
                    for method, settings in (
                        ("canonical", params()),
                        ("fixed", fixed[kind][str(margin)]["selected"]["params"]),
                        ("feedback", feedback[kind][str(margin)]["selected"]["params"]),
                    ):
                        block.append(
                            job(
                                f"H-{kind}-c{ci}-s{seed}-m{mi}-{method}",
                                "H",
                                kind,
                                case,
                                seed,
                                method,
                                margin,
                                settings,
                                save_frames=ci < 2 and seed == tier["seeds"][0] and mi == 2,
                            )
                        )
                np.random.default_rng(seed + ci).shuffle(block)
                final_jobs.extend(block)
    manifest = freeze(f"final-comparison{suffix}", final_jobs, selection)
    stages["final"] = str(manifest)
    run_stage(manifest)
    context = []
    for kind, panel in panels.items():
        for i, case in enumerate(panel["benchmark_context"]):
            context.append(
                job(
                    f"C-{kind}-{i}",
                    "C",
                    kind,
                    case,
                    180300 + i,
                    "published_context",
                    0,
                    params(checked=False),
                    executed_blocks=5,
                    save_frames=i == 0,
                )
            )
    manifest = freeze(
        f"benchmark-context{suffix}",
        context,
        {
            "scope": "Unfiltered +25 dataset pairs, 25 actions/replan; 12 per task, "
            "not the full OGBench leaderboard. Report initially solved separately."
        },
    )
    stages["context"] = str(manifest)
    run_stage(manifest)
    # A small serial pass measures latency without competition from another worker.
    timing_jobs = []
    for kind, panel in panels.items():
        margin = protocol["dials"][kind][-1]
        for ci, case in enumerate(panel["final"][:2]):
            block = []
            for method, settings in (
                ("native", params(checked=False)),
                ("canonical", params()),
                ("fixed", fixed[kind][str(margin)]["selected"]["params"]),
                ("feedback", feedback[kind][str(margin)]["selected"]["params"]),
            ):
                block.append(
                    job(
                        f"T-{kind}-c{ci}-{method}",
                        "T",
                        kind,
                        case,
                        180400,
                        method,
                        margin,
                        settings,
                    )
                )
            np.random.default_rng(180400 + ci).shuffle(block)
            timing_jobs.extend(block)
    manifest = freeze(
        f"serial-timing{suffix}",
        timing_jobs,
        {
            "purpose": "Descriptive uncontended latency, separate from final efficacy counts; "
            "first two reserved cases, one new seed, highest margin, no outcome selection."
        },
    )
    stages["serial_timing"] = str(manifest)
    run_stage(manifest, lanes=1)
    atomic_json(OUT / f"campaign-complete{suffix}.json", {"utc": utc(), "stages": stages})
    print("Campaign complete", flush=True)


if __name__ == "__main__":
    main()
