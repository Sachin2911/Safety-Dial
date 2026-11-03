"""Check completed Block Search 2 records on their rented host, without rollouts.

This verifies recorded predicates and scoring, not simulator replay or reconstruction
of every model prediction. It writes a new paper-integration audit outside the run.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import socket
import time


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close_vector(actual, expected):
    assert len(actual) == len(expected)
    assert all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
               for a, b in zip(actual, expected)), (actual, expected)


def fitness(records):
    outcomes = [r["result"] for r in records]
    summaries = [r["summary"] for r in records]
    times = [s["completion_time_s"] for s in summaries if s["safe_success"]]
    return [
        sum(r["sustained"] for r in outcomes),
        sum(r["benchmark_success"] for r in outcomes),
        -sum(r["violation"] for r in outcomes),
        -sum(s["max_top_base_xy_distance_m"] for s in summaries) / len(summaries),
        sum(min(s["max_middle_displacement_m"] / 0.07, 1.0) for s in summaries)
        / len(summaries),
        -sum(times) / len(times) if times else -300.0,
    ]


def check_search(run, cfg, records):
    import torch

    trace = read(run / "cma/trace.json")["generations"]
    assert len(trace) == cfg["cma"]["generations"] == 8
    previous_elite = None
    for generation, g in enumerate(trace, 1):
        seeds = cfg["training_starts"][(generation - 1) * 8:generation * 8]
        assert g["starts"] == seeds and g["generation"] == generation
        asked = read(run / f"cma/gen-{generation:02d}-asked.json")
        assert asked["candidates"] == g["candidates"]
        keys = []
        for index, params in enumerate(g["candidates"]):
            episodes = [records[f"cma-g{generation:02d}-c{index}-{s}"] for s in seeds]
            for seed, record in zip(seeds, episodes):
                assert record["summary"]["job"]["seed"] == seed
                close_vector(record["summary"]["adapter"]["params"], params)
            keys.append(fitness(episodes))
            close_vector(g["keys"][index], keys[-1])
        order = sorted(range(8), key=lambda i: keys[i], reverse=True)
        ranks = [order.index(i) for i in range(8)]
        assert g["ranks"] == ranks
        assert g["elite_index"] == order[0]
        close_vector(g["elite"], g["candidates"][order[0]])
        close_vector(g["elite_key"], keys[order[0]])
        if previous_elite is not None:
            close_vector(asked["injected_elite"], previous_elite)
            injected = g["injected_elite_index"]
            assert injected is not None, "Recorded injected elite is absent"
            close_vector(g["candidates"][injected], previous_elite)
        previous_elite = g["elite"]
    close_vector(read(run / "genotypes/cma-elite.json")["params"], previous_elite)
    close_vector(read(run / "genotypes/cma-final-mean.json")["params"], trace[-1]["mean_after"])

    random = read(run / "random/search.json")
    assert len(random["draws"]) == 64
    expected_seeds = set(range(cfg["random_search"]["first_seed"],
                               cfg["random_search"]["first_seed"] + 64))
    assert {d["generator_seed"] for d in random["draws"]} == expected_seeds
    for draw in random["draws"]:
        seed = draw["generator_seed"]
        block = (seed - cfg["random_search"]["first_seed"]) // 8 + 1
        starts = cfg["training_starts"][(block - 1) * 8:block * 8]
        assert draw["starts"] == starts and draw["block"] == block
        generator = torch.Generator(device="cpu").manual_seed(seed)
        params = (torch.randn(8, generator=generator) * cfg["random_search"]["scale"]).tolist()
        assert draw["params"] == params
        episodes = [records[f"random-b{block:02d}-d{seed}-{s}"] for s in starts]
        for start, record in zip(starts, episodes):
            assert record["summary"]["job"]["seed"] == start
            assert record["summary"]["adapter"]["params"] == params
        close_vector(draw["key"], fitness(episodes))
    ordered = sorted(random["draws"],
                     key=lambda d: (tuple(d["key"]), -d["generator_seed"]), reverse=True)
    assert random["draws"] == ordered
    assert random["top"] == [d["generator_seed"] for d in ordered[:2]]
    for index, draw in enumerate(ordered[:2], 1):
        assert read(run / f"genotypes/random-top{index}.json")["params"] == draw["params"]

    for phase, candidates, nominee in [
        ("selectA", {"final_mean": "cma-final-mean", "elite": "cma-elite"}, "cma"),
        ("selectB", {"top1": "random-top1", "top2": "random-top2"}, "random"),
    ]:
        selection = read(run / "select" / (phase + ".json"))
        keys = {}
        for name, filename in candidates.items():
            episodes = [records[f"{phase}-{name}-{s}"] for s in cfg["selection_panel"]]
            params = read(run / "genotypes" / (filename + ".json"))["params"]
            for seed, record in zip(cfg["selection_panel"], episodes):
                assert record["summary"]["job"]["seed"] == seed
                close_vector(record["summary"]["adapter"]["params"], params)
            keys[name] = fitness(episodes)
            close_vector(selection["keys"][name], keys[name])
        winner = max(keys, key=keys.get)
        assert selection["winner"] == winner
        selected = read(run / "genotypes" / (nominee + "-nominee.json"))
        assert selected["source"]["chosen"] == winner
        assert selected["params"] == read(run / "genotypes" / (candidates[winner] + ".json"))[
            "params"
        ]


def check_episode(summary, rows):
    """Recompute the benchmark and 40-control continuation from saved predicates."""
    assert summary["status"] == "complete" and summary["continuation_enabled"]
    assert len(rows) == summary["controls"] and 0 < len(rows) <= 340
    benchmark = None
    extension = 0
    lost = post_violation = False
    finished = False
    for index, row in enumerate(rows):
        assert not finished and row["step"] == index
        assert math.isclose(row["simulation_time_s"], (index + 1) * 0.05)
        terms, after = row["termination_terms"], row["after"]
        assert bool(terms["success"]) == bool(after["physical_success"])
        failure = bool(
            after["physical_failure"] or terms["cube_2_dropping"] or terms["cube_3_dropping"]
        )
        boundary = bool(terms["ee_position"])
        violation = failure or boundary
        action = row["executed_normalized_action"]
        assert all(math.isfinite(x) and abs(x) <= 1.000001 for x in action)
        if benchmark is None:
            outcome = (
                "physical_failure" if failure else "boundary_exit" if boundary else
                "success" if terms["success"] else "timeout" if index + 1 >= 300 else None
            )
            if outcome is not None:
                benchmark = dict(
                    outcome=outcome,
                    safe_success=outcome == "success",
                    violation=violation,
                    physical_failure=failure,
                    boundary_exit=boundary,
                    timeout=outcome == "timeout",
                    steps=index + 1,
                )
                finished = outcome != "success"
        else:
            extension += 1
            lost |= not bool(terms["success"])
            post_violation |= violation
            finished = violation or extension >= 40
    assert finished and benchmark is not None
    for key, value in benchmark.items():
        assert summary[key] == value, (key, summary[key], value)
    sustained = benchmark["safe_success"] and extension == 40 and not lost and not post_violation
    assert summary["continuation_steps"] == extension
    assert summary["post_success_predicate_loss"] == lost
    assert summary["post_success_violation"] == post_violation
    assert summary["sustained_safe_success"] == sustained
    assert summary["final_state"] == rows[-1]["after"]
    assert summary["interventions"] == sum(bool(r["intervened"]) for r in rows)
    assert summary["max_top_base_xy_distance_m"] == max(
        r["after"]["top_base_xy_distance_m"] for r in rows
    )
    middle_start = rows[0]["before"]["cube_positions"][1]
    max_middle = max(math.dist(r["after"]["cube_positions"][1], middle_start) for r in rows)
    assert math.isclose(summary["max_middle_displacement_m"], max_middle, abs_tol=1e-10)
    if benchmark["safe_success"]:
        assert math.isclose(summary["completion_time_s"], benchmark["steps"] * 0.05)
    else:
        assert summary["completion_time_s"] is None
    assert summary["rollout_physics_steps"] == (
        summary["physics_steps"] - summary["physics_after_reset"]
    )
    return dict(
        sustained=int(sustained),
        benchmark_success=int(benchmark["safe_success"]),
        violation=int(benchmark["violation"] or post_violation),
        unfinished=int(not benchmark["safe_success"]),
        controls=len(rows),
        interventions=summary["interventions"],
        eef_path_length_m=summary["eef_path_length_m"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--repeat", type=int, choices=[0, 1, 2], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    assert str(repo) == "/workspace/Safety-Dial", "Run only on the existing rented hosts"
    out = args.output.resolve()
    assert not out.exists(), "Preserve previous audit attempts"
    out.mkdir(parents=True)
    run = repo / f"outputs/search2-20261102-r{args.repeat}"
    config_path = repo / f"SafeBlock/configs/search2-20261102-r{args.repeat}.json"
    cfg = read(config_path)
    assert read(run / "status.json")["phase"] == "complete", "Study is not terminal"
    groups = [set(cfg[k]) for k in ("training_starts", "selection_panel", "final_panel")]
    assert [len(x) for x in groups] == [64, 16, 16]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    bindings = {}
    for name, expected in cfg["code_sha256"].items():
        path = repo / "SafeBlock/experiments" / name
        assert digest(path) == expected, (name, "current code differs from configured source")
        bindings[str(path.relative_to(repo))] = expected
    final = {arm: {} for arm in ("native", "untrained", "random", "cma")}
    phase_counts = Counter()
    resources = Counter()
    failures = []
    manifest = {}
    records = {}
    for phase in ("smoke", "cma", "select", "final", "random"):
        for job in sorted((run / phase).iterdir()):
            if not job.is_dir():
                continue
            attempts = sorted(job.glob("attempt-*"))
            complete = [a for a in attempts if (a / "summary.json").exists()]
            assert len(complete) <= 1, (job.name, "multiple completed attempts")
            for attempt in attempts:
                resources["attempts"] += 1
                if attempt not in complete:
                    accounting = read(attempt / "accounting.json") if (
                        attempt / "accounting.json"
                    ).exists() else {}
                    failures.append(dict(
                        path=str(attempt.relative_to(repo)), accounting=accounting,
                        reset_evidence_present=(attempt / "initial_state.npz").exists(),
                    ))
                    resources["failed_attempt_recorded_controls"] += accounting.get(
                        "completed_controls", 0
                    )
            assert complete, (job.name, "missing completed episode; requires review")
            attempt = complete[0]
            summary = read(attempt / "summary.json")
            rows = [json.loads(line) for line in (attempt / "steps.jsonl").read_text().splitlines()]
            result = check_episode(summary, rows)
            assert summary["job"]["task"] == "hard"
            assert summary["job"]["condition"] == "ap_rank"
            assert summary["job"]["seed"] == summary["job"]["policy_seed"]
            for source, expected in summary["implementation_hashes"].items():
                assert digest(attempt / "implementation" / source) == expected, source
                if source in bindings:
                    assert expected == bindings[source], (source, "episode source differs")
            assert summary["freeze_sha256"] == cfg["system"]["freeze_sha256"]
            accounting = read(attempt / "accounting.json")
            assert accounting["completed_controls"] == len(rows)
            phase_counts[phase] += 1
            resources["completed_controls"] += result["controls"]
            resources["completed_rollout_physics_steps"] += summary["rollout_physics_steps"]
            records[job.name] = dict(result=result, summary=summary, attempt=attempt)
            for name in ("summary.json", "steps.jsonl", "accounting.json", "worker-start.json"):
                path = attempt / name
                manifest[str(path.relative_to(repo))] = dict(
                    sha256=digest(path), bytes=path.stat().st_size
                )
            if phase == "final":
                _, arm, seed = job.name.split("-")
                seed = int(seed)
                assert seed in groups[2] and seed not in final[arm]
                assert seed == summary["job"]["seed"]
                if arm == "native":
                    assert summary["job"]["adapter"] == "none"
                else:
                    name = "untrained" if arm == "untrained" else arm + "-nominee"
                    genotype = run / "genotypes" / (name + ".json")
                    assert summary["adapter"]["params"] == read(genotype)["params"]
                    assert genotype.stat().st_mtime <= (attempt / "worker-start.json").stat().st_mtime
                final[arm][str(seed)] = result
    expected_counts = dict(smoke=4, cma=512, select=64, final=64, random=512)
    assert dict(phase_counts) == expected_counts, dict(phase_counts)
    for arm, outcomes in final.items():
        assert set(map(int, outcomes)) == groups[2], arm
    check_search(run, cfg, records)
    for seed in cfg["smoke"]["starts"]:
        a = records[f"smoke-zero-bare-{seed}"]
        b = records[f"smoke-zero-z-{seed}"]
        assert a["result"] == b["result"]
        ar = [json.loads(x) for x in (a["attempt"] / "steps.jsonl").read_text().splitlines()]
        br = [json.loads(x) for x in (b["attempt"] / "steps.jsonl").read_text().splitlines()]
        for x, y in zip(ar, br):
            for key in ("executed_normalized_action", "camera_sha256", "uncertainty", "next_min_q"):
                assert x[key] == y[key], (seed, key)
    result = dict(
        passed=True, checked_at=time.time(), host=socket.gethostname(), repeat=args.repeat,
        scope="Saved predicate/outcome, panel, source-copy, search ranking and nominee audit",
        config_sha256=digest(config_path), auditor_sha256=digest(Path(__file__)),
        phase_counts=dict(phase_counts), resources=dict(resources), failed_attempts=failures,
        final=final, source_bindings=bindings,
        limitations=[
            "No simulator or world-model replay; geometry predicates are those recorded by the worker",
            "CMA population/rank records checked; optimizer numerical updates not independently replayed",
            "Camera hashes matched between smoke arms; complete camera-byte preservation not checked",
            "Discarded five-lane launches and the withdrawn fourth repeat are outside this run directory",
            "The host-local smoke uses four episodes, versus two in the original budget table",
        ],
    )
    (out / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("passed", "repeat", "phase_counts", "resources")}))


if __name__ == "__main__":
    main()
