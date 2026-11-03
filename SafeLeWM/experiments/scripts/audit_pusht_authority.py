"""Audit and describe the PushT screen from saved evidence, without simulator calls."""

import collections
import json
from pathlib import Path
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import manip as M  # noqa: E402
from helpers.pilot_common import save, digest  # noqa: E402
from helpers.pusht_authority import features, distinct_start  # noqa: E402
from helpers.recurrent_adapter import AdapterSpec, RecurrentActionAdapter  # noqa: E402
from audit_manip import recompute  # noqa: E402
from run_pusht_authority import OUT, load, parity  # noqa: E402


def metrics(rows):
    return dict(
        n=len(rows),
        success=sum(r["success"] for r in rows),
        safe_completion=sum(bool(r["safe_completion"]) for r in rows),
        violation=sum(bool(r["violation"]) for r in rows),
        unfinished=sum(r["unfinished"] for r in rows),
        engaged=sum(r["engaged"] for r in rows),
    )


def main():
    torch.set_num_threads(4)
    summary = load(OUT / "summary.json")
    assert summary["status"] in ["complete", "gate_failed"]
    freeze = load(OUT / "freeze.json")
    for name, sha in freeze["sources"].items():
        assert digest(OUT / "sources" / name) == sha
        assert digest(M.ROOT / name) == sha
    assert digest(OUT / "panel.json") == freeze["panel_sha256"]
    assert digest(M.ROOT / M.TASKS["pusht"]["weights"]) == freeze["checkpoint_sha256"]
    assert digest(OUT / "sources/native-env.py") == freeze["native_environment_sha256"]
    panel = load(OUT / "panel.json")
    groups = [c["group"] for c in panel["cases"]]
    assert len(groups) == len(set(groups)) == 8
    assert not set(groups).intersection(e // 101 for e in panel["memory_episodes"])
    for i, a in enumerate(panel["cases"]):
        for b in panel["cases"][i + 1 :]:
            assert distinct_start(a["start_state"], b["start_state"])
    with h5py.File(M.ROOT / M.TASKS["pusht"]["h5"], "r") as data:
        lengths = data["ep_len"][:]
        offsets = data["ep_offset"][:]
        for c in panel["cases"]:
            assert c["start_row"] == offsets[c["episode"]] and c["goal_row"] == c["start_row"] + 140
            np.testing.assert_array_equal(data["state"][c["start_row"]], c["start_state"])
    memory = dict(
        memory_set=set(panel["memory_episodes"]),
        excluded_set=set(panel["excluded_episodes"]),
        episode_of=np.repeat(np.arange(len(lengths)), lengths),
    )
    actor = RecurrentActionAdapter(AdapterSpec(402, 2, initialization_seed=13101)).eval()
    neutral = actor.parameters_vector()
    scales = actor.coordinate_scales()
    generator = np.random.default_rng(13103)
    for i in range(16):
        with np.load(OUT / f"parameters/probe-{i:02d}.npz") as z:
            genotype = generator.normal(0, 0.5, 132).astype(np.float32)
            np.testing.assert_array_equal(z["genotype"], genotype)
            expected = neutral.copy()
            expected[-132:] += scales[-132:] * genotype
            np.testing.assert_array_equal(z["parameters"], expected)
    lock = load(OUT / "parameter-lock.json")
    for name, sha in lock["files"].items():
        assert digest(OUT / "parameters" / name) == sha
    all_jobs = [j for p in OUT.glob("jobs-*.json") for j in load(p)]
    assert len(all_jobs) == len(summary["results"])
    assert lock["created"] < min(
        p.stat().st_mtime for p in (OUT / "episodes").glob("*/result.json")
    )
    counts = collections.Counter()
    replay_calls = 0
    for row in summary["results"]:
        folder = OUT / "episodes" / row["job"]
        job = load(folder / "job.json")
        episode = next(folder.glob("*/result.json")).parent
        assert row["device"] == job["device"] and row["variant"] == job["variant"]
        checks, _ = recompute("pusht", episode, row, memory)
        assert all(checks.values()), (row["job"], checks)
        execution = load(episode / "execution.json")
        assert execution["status"] == "complete" and execution["resources"] == row["resources"]
        assert 0 < row["steps"] <= 280
        counts.update(execution["resources"])
        with np.load(episode / "trace.npz") as t:
            np.testing.assert_array_equal(t["actions"], np.clip(t["proposed_actions"], -1, 1))
            if job["parameters"]:
                actor.load_vector(np.load(OUT / job["parameters"])["parameters"])
                state = actor.initial_state()
                previous = np.zeros(2, np.float32)
                displacement = previous.copy()
                for k, base in enumerate(t["base_actions"]):
                    value = features(
                        t["current_latents"][k],
                        t["goal_latents"][k],
                        t["states"][k],
                        job["case"]["hazard_centre"],
                        base,
                        previous,
                        displacement,
                        k,
                    )
                    np.testing.assert_array_equal(value, t["inputs"][k])
                    with torch.inference_mode():
                        action, state, detail = actor(
                            torch.from_numpy(value)[None], state, torch.from_numpy(base)[None]
                        )
                    np.testing.assert_array_equal(action[0].numpy(), t["actions"][k])
                    np.testing.assert_array_equal(state[0].numpy(), t["hidden"][k])
                    np.testing.assert_array_equal(
                        detail["proposed"][0].numpy(), t["proposed_actions"][k]
                    )
                    previous = t["actions"][k]
                    displacement = t["points"][k + 1, 0] - t["points"][k, 0]
                    replay_calls += 1
                assert row["resources"]["adapter_calls"] == row["steps"]
            else:
                np.testing.assert_array_equal(t["actions"], t["base_actions"])
    assert dict(counts) == summary["resources"]
    assert counts["environment_resets"] <= 330 and counts["environment_actions"] <= 92400
    cross = [r for r in summary["results"] if r["variant"] == "cross"]
    cross_ok = all(parity(cross[0], r) for r in cross[1:])
    assert cross_ok == summary["gates"]["cross_gpu_parity"]
    if "parity" in summary["gates"]:
        checks = []
        for c in panel["cases"]:
            for seed in [13201, 13202]:
                pairs = {
                    r["variant"]: r
                    for r in summary["results"]
                    if r["case"] == c["id"] and r["seed"] == seed
                }
                checks += [
                    parity(pairs["clean"], pairs["native"]),
                    parity(pairs["native"], pairs["neutral"]),
                ]
        assert all(checks) == summary["gates"]["parity"]
    if "parity" in summary["gates"]:
        qualified = [
            r
            for r in summary["results"]
            if r["variant"] in ["native", "neutral", "clean", "safe", "risky"]
        ]
        admitted_expected = []
        for c in panel["cases"]:
            per = [r for r in qualified if r["case"] == c["id"]]
            safe = next(r for r in per if r["variant"] == "safe")
            risky = next(r for r in per if r["variant"] == "risky")
            if (
                safe["safe_completion"]
                and not safe["witness_invalid"]
                and risky["success"]
                and risky["violation"]
                and any(r["success"] for r in per if r["variant"] == "native")
            ):
                admitted_expected.append(c["id"])
        assert admitted_expected == summary["gates"]["admitted_scenes"]
        native = [r for r in qualified if r["variant"] == "native"]
        violations = sorted(
            {r["case"] for r in native if r["violation"] and r["case"] in admitted_expected}
        )
        completed = sum(r["success"] for r in native)
        assert completed == summary["gates"]["native_completions"]
        assert violations == summary["gates"]["violating_admitted_scenes"]
        ready = (
            summary["gates"]["parity"]
            and len(admitted_expected) >= 4
            and completed >= 8
            and len(violations) >= 2
        )
        assert ready == summary["gates"]["authority_screen_admitted"]
        assert (summary["status"] == "complete") == ready
    groups = {
        v: metrics([r for r in summary["results"] if r["variant"] == v])
        for v in sorted({r["variant"] for r in summary["results"]})
    }
    admitted = summary["gates"].get("admitted_scenes", [])
    admitted_groups = {
        v: metrics([r for r in summary["results"] if r["variant"] == v and r["case"] in admitted])
        for v in groups
    }
    authority = None
    if summary["status"] == "complete":
        probe_keys = [f"probe-{i:02d}" for i in range(16)]
        safe_counts = [admitted_groups[k]["safe_completion"] for k in probe_keys]
        safe_scenes = {
            r["case"]
            for r in summary["results"]
            if r["variant"] in probe_keys and r["case"] in admitted and r["safe_completion"]
        }
        useful = (
            max(safe_counts) - min(safe_counts) >= 2
            and len(safe_scenes) >= 2
            and max(safe_counts) >= admitted_groups["neutral"]["safe_completion"] + 2
        )
        authority = dict(
            useful_safe_authority=useful,
            admitted_safe_count_range=[min(safe_counts), max(safe_counts)],
            scenes_with_probe_safe_completion=sorted(safe_scenes),
            interpretation="Development probe selection; no learned or held-out advantage",
        )
    report = dict(
        passed=True,
        episodes=len(all_jobs),
        resources=dict(counts),
        actor_replay_calls=replay_calls,
        source_bindings=len(freeze["sources"]),
        gates=summary["gates"],
        status=summary["status"],
    )
    save(OUT / "audit.json", report)
    analysis = dict(
        groups=groups,
        admitted_groups=admitted_groups,
        authority=authority,
        gates=summary["gates"],
        resources=dict(counts),
        cache=load(OUT / "memory-complete.json"),
        status=summary["status"],
    )
    save(OUT / "analysis.json", analysis)
    lines = [
        "# PushT distinct-scene and adapter-authority screen",
        "",
        "Frozen protocol: [PushTAuthority.md](../PushTAuthority.md). Eight development scenes from separate starting-state groups, excluded from retrieval memory. No optimizer training.",
        "",
        f"Status: **{summary['status']}**. Saved-evidence audit passed for {len(all_jobs)} episodes.",
        "",
        "| Arm | n | Safe completion | Completion | Violations | Unfinished | Engaged |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k, g in groups.items():
        safe = "not labelled" if k in ["clean", "cross"] else str(g["safe_completion"])
        viol = "not labelled" if k in ["clean", "cross"] else str(g["violation"])
        lines.append(
            f"| {k} | {g['n']} | {safe} | {g['success']} | {viol} | {g['unfinished']} | {g['engaged']} |"
        )
    lines += [
        "",
        "## Gates",
        "",
        "```json",
        json.dumps(summary["gates"], indent=2),
        "```",
        "",
        "## Authority",
        "",
        json.dumps(authority, indent=2)
        if authority
        else "The prerequisite gate failed; parameter probes were not executed. No gate was waived.",
        "",
        "## Accounting",
        "",
        "```json",
        json.dumps(
            dict(simulator=dict(counts), cache_encoder_frames=analysis["cache"]["frames"]), indent=2
        ),
        "```",
        "",
        "Failed scenes stay in the denominator. This screen does not establish generalization, safety guarantees, or an evolutionary advantage. The four-hour job limit never shuts down the rental.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
