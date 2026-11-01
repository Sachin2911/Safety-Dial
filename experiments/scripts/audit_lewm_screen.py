"""Audit the frozen screen using saved evidence, without loading models or simulators."""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from helpers.lewm_proposal_pilot import write_json
from helpers.lewm_screen import (
    CubeProposalAdapter,
    CubeProposalSpec,
    ScreenPushTAdapter,
    distance,
    geometry,
    image_stats,
    no_hazard_gate,
    summarize,
)
from helpers.lewm_residual import influence_summary
from helpers.lewm_proposal_adapter import ProposalSpec
from helpers.pusht_smoke import physical_clearance
from helpers.portable_motion import clearance, shape_constraints


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    d = json.loads(args.protocol.read_text())
    out = args.out
    s = json.loads((out / "state.json").read_text())
    freeze = json.loads((out / "source-freeze.json").read_text())
    assert s["phase"] == "complete_with_declared_gates"
    assert sha(args.protocol) == freeze["protocol_sha256"]
    for names in (freeze["sources"], d["asset_sha256"]):
        for name, expected in names.items():
            assert sha(ROOT / name) == expected, name
    jobs = s["jobs"]
    starts = {p.parent.name: p for p in (out / "episodes").glob("*/started.json")}
    assert set(jobs) <= set(starts)
    assert len(starts) <= 100
    assert sum(k.startswith("engineering_") for k in starts) <= 4
    totals = collections.Counter()
    groups = {}
    environment_resources = {name: collections.Counter() for name in ("pusht", "cube")}
    command_ranges = {}
    audited = []
    controller_parameters = {}
    influence_totals = collections.Counter()
    for key, row in jobs.items():
        folder = out / "episodes" / key
        assert json.loads((folder / "result.json").read_text()) == row
        start = json.loads(starts[key].read_text())
        env = row["environment"]
        cfg = d["environments"][env]
        assert all(start[k] == row[k] for k in ("role", "kind", "seed"))
        assert start["pair"] == row["pair_index"]
        engineering = key.startswith("engineering_")
        assert start["cap"] == (6 if engineering else 150)
        panel = json.loads((out / "panels" / (env + ".json")).read_text())
        case = panel[row["pair_index"]]
        assert row["shape"] == case["shapes"][row["kind"]]
        source = d["sources"][env][row["pair_index"]]
        with np.load(ROOT / source["path"], allow_pickle=False) as data:
            pair = dict(data)
        regenerated = geometry(
            env,
            row["pair_index"],
            np.asarray(case["settled_initial"]),
            pair["goal_state"][:2] if env == "pusht" else pair["reference_route"][-1],
            None if env == "pusht" else pair["reference_route"],
            source["source_metadata"]["shape"],
        )
        assert regenerated == case["shapes"]
        base = folder / "episode" if env == "pusht" else folder
        with np.load(base / "trace.npz", allow_pickle=False) as data:
            t = dict(data)
        n = len(t["actions"])
        assert n == row["actions"] <= start["cap"]
        assert len(t["states"]) == n + 1
        factor = 10 if env == "pusht" else 25
        assert len(t["dense"]) == factor * n + 1
        points = t["dense"][:, :2]
        numpy_clear = distance(points, row["shape"], cfg["body_radius"])
        tensor = torch.tensor(points, dtype=torch.float64)
        independent = (
            physical_clearance(tensor, row["shape"]).numpy()
            if env == "pusht"
            else clearance(tensor, shape_constraints(row["shape"]), cfg["body_radius"]).numpy()
        )
        if row["kind"] != "none":
            np.testing.assert_allclose(numpy_clear, independent, rtol=0, atol=1e-9)
        breach = bool((independent < cfg["margin"] - 1e-6).any())
        assert breach == row["margin_breach"]
        if env == "pusht":
            arena = np.minimum(points, 512 - points).min(-1) - 15
            assert bool((arena < cfg["margin"] - 1e-6).any()) == row["arena_margin_breach"]
            final, goal = t["states"][-1], pair["goal_state"]
            angle = (goal[4] - final[4] + np.pi) % (2 * np.pi) - np.pi
            success = np.linalg.norm(goal[:4] - final[:4]) < 20 and abs(angle) < np.pi / 9
            assert row["resources"]["executed_physics_steps"] == n * 10
            assert np.isfinite(t["actions"]).all()
            assert (t["actions"] >= -1).all() and (t["actions"] <= 1).all()
        else:
            success = np.linalg.norm(t["cube_xyz"][-1] - pair["goal"]) <= 0.04
            assert row["resources"]["native_executed_physics_steps"] == n * 25
            assert row["resources"]["executed_physics_steps"] == n * 25
            np.testing.assert_array_equal(t["actions"], np.clip(t["proposed"], -1, 1))
            np.testing.assert_allclose(t["states"][0], case["settled_initial"], atol=1e-8, rtol=0)
            assert t["frame_fresh"][-1]
            for i, action_index in enumerate(t["frame_actions"][:-1]):
                assert bool(t["frame_fresh"][i]) == (action_index % 25 == 0)
                if not t["frame_fresh"][i]:
                    np.testing.assert_array_equal(t["frames"][i], t["frames"][i - 1])
        assert bool(success) == row["success"]
        assert row["unfinished"] == (not bool(success))
        assert row["safe_completion"] == bool(
            success
            and not breach
            and not row["arena_margin_breach"]
            and not row["error"]
            and not row["controller_failure"]
        )
        flags = [i for i, frame in enumerate(t["frames"]) if image_stats(frame)["flagged"]]
        assert flags == row["frame_flags"]
        resources = row["resources"]
        assert resources["environment_actions"] == n
        assert resources["environment_resets"] == 1
        assert resources.get("predictor_steps", 0) <= cfg["maximum_predictor_steps"]
        assert not any(
            resources.get(k, 0)
            for k in ("analytic_path_steps", "checker_physics_steps", "reference_candidate_checks")
        )
        totals.update(resources)
        environment_resources[env].update(resources)
        command_ranges[key] = {
            "minimum": float(t["actions"].min()) if n else None,
            "maximum": float(t["actions"].max()) if n else None,
        }
        plans = json.loads((base / "replans.json").read_text())
        assert resources["encoder_frames"] == 2 * len(plans)
        for plan in plans:
            assert plan["horizon"] == cfg["planner"]["horizon"]
            if plan.get("executed") != "none":
                assert plan["executed_violation"] == 0
                assert plan["samples"] == (250 if env == "pusht" else 300)
        if row["role"] == "native":
            assert not (folder / "controller.json").exists()
            assert not resources.get("adapter_forward_calls", 0)
        else:
            artifact = json.loads((folder / "controller.json").read_text())
            cls, speccls = (
                (ScreenPushTAdapter, ProposalSpec)
                if env == "pusht"
                else (CubeProposalAdapter, CubeProposalSpec)
            )
            spec = speccls(**artifact["spec"])
            adapter = cls(spec, artifact["parameters"], row["shape"])
            expected = (
                np.asarray(json.loads((ROOT / cfg["controller"]).read_text())["parameters"])
                if env == "pusht"
                else spec.initial_parameters(d["cube_adapter_seed"])
            )
            if row["role"] == "zero":
                expected[spec.hidden_size * 30 :] = 0
            np.testing.assert_array_equal(expected, artifact["parameters"])
            history = json.loads((folder / "proposals.json").read_text())
            assert len(history) == resources["adapter_forward_calls"]
            for item in history:
                np.testing.assert_allclose(
                    adapter.forward(item["features"], len(item["proposal"])),
                    item["proposal"],
                    atol=1e-12,
                    rtol=0,
                )
            for plan in plans:
                if plan.get("executed") == "none":
                    continue
                item = next(p for p in history if p["replan"] == plan["features"]["replan"])
                np.testing.assert_allclose(
                    item["proposal"], plan["adapter_proposal"], atol=1e-6, rtol=0
                )
                assert len(plan["residual_checks"]) == cfg["planner"]["iterations"]
                for check in plan["residual_checks"]:
                    assert check["new_guided_samples"] == int(
                        (check["samples"] - check["reserved_native_slots"]) * 0.5
                    )
                    assert check["guided_elites"] <= check["fitted_elites"]
            influence = influence_summary(plans)
            assert influence["new_guided_samples"] == resources.get(
                "adapter_guided_candidate_evaluations", 0
            )
            influence_totals.update(influence)
            controller_parameters.setdefault(
                env + "/" + row["role"], sha(folder / "controller.json")
            )
        if not engineering:
            groups.setdefault(env, {}).setdefault(row["role"], {}).setdefault(
                row["kind"], []
            ).append(row)
        audited.append(key)
    # Include starts which failed before a completed result, never silently discard them.
    failed = {}
    for key in set(starts) - set(jobs):
        folder = starts[key].parent
        evidence = [str(p.relative_to(out)) for p in folder.rglob("*.json")]
        assert evidence
        failed[key] = evidence
        result = folder / "result.json"
        if result.exists():
            totals.update(json.loads(result.read_text()).get("resources", {}))
        else:
            error = folder / "technical-error.json"
            assert error.exists()
            totals.update(json.loads(error.read_text()).get("resources", {}))
    preparation = {}
    for env in ("pusht", "cube"):
        prep = out / "preparation"
        resource_file = prep / (env + "-resources.json")
        rejected = prep / (env + "-rejected.json")
        constructor = prep / (env + "-controls-constructor.json")
        values = json.loads(resource_file.read_text()) if resource_file.exists() else {}
        startup = values.get("constructor", {})
        if constructor.exists():
            recorded = json.loads(constructor.read_text())
            assert not startup or startup == recorded
            startup = recorded
        interactions = values.get("preparation", {})
        if rejected.exists():
            interactions = json.loads(rejected.read_text())["resources"]
        main_constructor = prep / (env + "-main-constructor.json")
        main = json.loads(main_constructor.read_text()) if main_constructor.exists() else {}
        for counters in (startup, interactions, main):
            totals.update(counters)
            environment_resources[env].update(counters)
        preparation[env] = {
            "constructor": startup,
            "preparation": interactions,
            "main_constructor": main,
        }
    assert sum(v["preparation"].get("environment_resets", 0) for v in preparation.values()) <= 6
    review = json.loads((out / "control-review.json").read_text())
    decisions = {}
    for env, status in s["environment_states"].items():
        data = groups.get(env, {})
        controls = data.get("native", {}).get("none", [])
        if len(controls) == 6:
            assert no_hazard_gate(controls) == status["performance_gate"]
        complete = status["phase"] == "complete"
        if complete:
            assert no_hazard_gate(controls) and review[env]["images_clean"]
            expected = set()
            for role in ("native", "untrained"):
                for i in range(3):
                    for kind in ("none", "island", "bar", "gate"):
                        for rep in range(2):
                            seed = d["planner_seed_base"][env] + 10 * i + rep
                            expected.add(f"{env}_{role}_p{i:03}_{kind}_s{seed}")
            assert expected == {
                k for k, r in jobs.items() if r["environment"] == env and r["category"] == "scored"
            }
        equivalence = out / "engineering" / (env + "-equivalence.json")
        if controls:
            assert json.loads(equivalence.read_text())["verified"]
            seed = d["engineering_seed_base"][env]
            traces = []
            for role in ("native", "zero"):
                f = out / "episodes" / f"engineering_{env}_{role}_p000_none_s{seed}"
                if env == "pusht":
                    f = f / "episode"
                with np.load(f / "trace.npz", allow_pickle=False) as t:
                    traces.append({k: t[k] for k in ("actions", "states", "dense", "frames")})
            for k in traces[0]:
                np.testing.assert_array_equal(traces[0][k], traces[1][k])
        summaries = {
            role: {kind: summarize(rows) for kind, rows in cases.items()}
            for role, cases in data.items()
        }
        gate = False
        paired = []
        if complete:
            native = [r for k, rows in data["native"].items() if k != "none" for r in rows]
            adapter = [r for k, rows in data["untrained"].items() if k != "none" for r in rows]
            a, b = summarize(native), summarize(adapter)
            ac, bc = summarize(controls), summarize(data["untrained"]["none"])
            gate = bool(
                1 <= a["safe_completion"] <= 17
                and b["safe_completion"] > a["safe_completion"]
                and b["violation"] <= a["violation"]
                and b["engaged"] >= a["engaged"]
                and bc["safe_completion"] >= ac["safe_completion"]
                and bc["engaged"] >= ac["engaged"]
                and not bc["violation"]
                and not any(
                    r["frame_flags"]
                    for r in native + adapter + controls + data["untrained"]["none"]
                )
            )
            for row in native + controls:
                match = jobs[row["key"].replace("_native_", "_untrained_")]
                paired.append(
                    {
                        "pair": row["pair_index"],
                        "kind": row["kind"],
                        "seed": row["seed"],
                        "native_safe": row["safe_completion"],
                        "adapter_safe": match["safe_completion"],
                        "native_actions": row["actions"],
                        "adapter_actions": match["actions"],
                    }
                )
        decisions[env] = {
            "phase": status["phase"],
            "summary": summaries,
            "future_training_gate": gate,
            "paired": paired,
        }
    report = {
        "verified": True,
        "phase": s["phase"],
        "pod_id": d["execution_pod_id"],
        "protocol_sha256": freeze["protocol_sha256"],
        "audited_completed_episodes": len(audited),
        "all_episode_starts": len(starts),
        "uncompleted_starts": failed,
        "scored_episodes": sum(r["category"] == "scored" for r in jobs.values()),
        "engineering_episodes": sum(r["category"] == "engineering" for r in jobs.values()),
        "preparation": preparation,
        "resource_totals": dict(totals),
        "resource_by_environment": {
            name: dict(value) for name, value in environment_resources.items()
        },
        "native_physics_steps_including_resets": sum(
            environment_resources["pusht"].get(k, 0)
            for k in ("executed_physics_steps", "reset_physics_steps")
        )
        + sum(
            environment_resources["cube"].get(k, 0)
            for k in ("native_executed_physics_steps", "native_reset_physics_steps")
        ),
        "executed_action_ranges": command_ranges,
        "native_action_contract": "Both native interfaces accept [-1,1]. PushT uses relative=True: native PD target = current pusher position + 100 * command. Cube logs raw policy output and clipped executed command separately. All executed commands verified in range.",
        "environments": decisions,
        "adapter_influence": dict(influence_totals),
        "optimizer_updates": 0,
        "new_simulator_interactions": 0,
        "new_frozen_model_queries": 0,
        "physics_accounting": "native_* counters count all Cube mj_step calls. executed_physics_steps/reset_physics_steps are overlapping DenseCube diagnostics and must not be added to native_*.",
        "evolutionary_advantage_established": False,
    }
    write_json(out / "audit.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("preparation", "resource_totals")})
    )


if __name__ == "__main__":
    main()
