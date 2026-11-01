"""Frozen, bounded proposal-weight development pilot on its declared rented GPU."""

import argparse
import collections
from contextlib import ExitStack
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from helpers.lewm_proposal_adapter import ProposalAdapter, ProposalSpec
from helpers.lewm_proposal_pilot import ProposalPilot, panel_fitness, summarize, write_json


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def clearances(dense, shape):
    """Independent NumPy monitor, including the complete 15 px pusher disk."""
    xy = (dense[:, :2] - shape["origin"]) @ np.asarray(shape["basis"]).T
    if shape["kind"] == "island":
        distance = np.linalg.norm(xy, axis=-1) - shape["radius_px"]
    elif shape["kind"] == "bar":
        nearest = np.c_[
            np.clip(xy[:, 0], -shape["segment_half_length_px"], shape["segment_half_length_px"]),
            np.zeros(len(xy)),
        ]
        distance = np.linalg.norm(xy - nearest, axis=-1) - shape["radius_px"]
    else:
        offset, half, extent = shape["gap_offset_px"], shape["gap_width_px"] / 2, shape["extent_px"]
        values = []
        for lo, hi in ((-extent, offset - half), (offset + half, extent)):
            q = np.abs(xy - [(lo + hi) / 2, 0]) - [(hi - lo) / 2, shape["band_half_width_px"]]
            values.append(np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0))
        distance = np.minimum(*values)
    arena = np.minimum(dense[:, :2], 512 - dense[:, :2]).min(-1) - 15
    return distance - 15, arena


def task_error(goal, state):
    angular = abs(goal[4] - state[4]) % (2 * np.pi)
    angular = min(angular, 2 * np.pi - angular)
    return np.linalg.norm(goal[:4] - state[:4]) / 20 + angular / (np.pi / 9)


def progress(row):
    if row["margin_breach"] or row["arena_margin_breach"] or row.get("error"):
        return 0.0
    if row["safe_completion"]:
        return 1.0
    goal = np.asarray(row["goal_state"])
    initial = task_error(goal, np.asarray(row["initial_state"]))
    return float(np.clip(1 - task_error(goal, np.asarray(row["final_state"])) / initial, 0, 1))


def identities(design, freeze):
    for name, expected in freeze["sources"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("Frozen source changed: " + name)
    for name, expected in design["asset_sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError("Frozen asset changed: " + name)


def train(args, design):
    import torch
    from stable_worldmodel.policy import AutoCostModel
    from helpers.learned_motion import LearnedMotionChecker, load_motion
    from helpers.lewm18_planner import PlanningCEM
    from helpers.pusht_smoke import Scaler
    from helpers.safety_model import pusht_constraints, scaler_digest
    from scripts.run_pusht_smoke import run_episode

    root = args.out
    freeze_path = root / "source-freeze.json"
    if args.resume:
        freeze = json.loads(freeze_path.read_text())
        if freeze["protocol_sha256"] != digest(args.protocol):
            raise RuntimeError("Pilot protocol changed")
    else:
        root.mkdir(parents=True, exist_ok=False)
        freeze = {
            "protocol_sha256": digest(args.protocol),
            "pod_id": os.environ["RUNPOD_POD_ID"],
            "sources": {
                str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "experiments").rglob("*.py")
            },
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        write_json(freeze_path, freeze)
    identities(design, freeze)
    torch.set_num_threads(4)
    checkpoint = Path(design["episode"]["world_model_checkpoint"])
    model = AutoCostModel(str(checkpoint)[: -len("_object.ckpt")], cache_dir=checkpoint.parent)
    model = model.to("cuda").eval().requires_grad_(False)
    head, _ = load_motion(Path(design["episode"]["safety_checkpoint"]), device="cuda")
    spec = head.spec
    if (
        spec.environment,
        spec.units,
        spec.body_radius,
        spec.action_block,
        spec.action_dim,
        spec.protected_body,
        spec.coordinate_frame,
    ) != ("pusht", "px", 15, 5, 2, "pusher_disk", "arena_xy"):
        raise RuntimeError("Safety model contract differs")
    if (
        not head.calibration
        or not head.calibration["ready"]
        or digest(checkpoint) != spec.world_model_sha256
    ):
        raise RuntimeError("Frozen safety model is not calibrated for this world model")
    inputs = Path(design["episode"]["inputs"])
    columns = json.loads((inputs / "scalers.json").read_text())["columns"]
    action = columns["action"]
    if scaler_digest(action["mean"], action["scale"]) != spec.action_scaler_sha256:
        raise RuntimeError("Action scaling differs from safety training")
    process = {key: Scaler(value["mean"], value["scale"]) for key, value in columns.items()}
    with np.load(inputs / "pair.npz", allow_pickle=False) as data:
        pair = dict(data)
    adapter_spec = ProposalSpec(**design["adapter"]["spec"])

    def evaluate(key, parameters, kind, seed):
        folder = root / "episodes" / key
        if (folder / "result.json").exists():
            return json.loads((folder / "result.json").read_text())
        if folder.exists():
            raise RuntimeError(
                "Incomplete saved job requires explicit diagnosis, never replay: " + key
            )
        deadline = dt.datetime.fromisoformat(design["deadline_utc"])
        if dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=10) >= deadline:
            raise RuntimeError("Preservation reserve reached; preserve incomplete pilot")
        if (
            len(list((root / "episodes").glob("*/started.json")))
            >= design["maximum_episode_starts"]
        ):
            raise RuntimeError("Episode-start ceiling reached")
        folder.mkdir(parents=True)
        case = design["cases"][kind]
        protocol = {**design["episode"], "seed": seed, "case": case}
        write_json(
            folder / "started.json",
            {
                "key": key,
                "seed": seed,
                "case": kind,
                "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        write_json(folder / "protocol.json", protocol)
        adapter = None if parameters is None else ProposalAdapter(adapter_spec, parameters, case)
        if adapter is not None:
            write_json(folder / "controller.json", adapter.artifact())
        counters = collections.defaultdict(int)

        class AdmittedChecker(LearnedMotionChecker):
            def observe(self, observed):
                if self.observed is None:
                    position = np.asarray(observed).reshape(1, -1)[:, :2]
                    clearance, arena = clearances(position, case)
                    if min(clearance.min(), arena.min()) < protocol["margin_px"]:
                        raise RuntimeError(
                            "Settled reset is outside the frozen safe admission region"
                        )
                return super().observe(observed)

        def checker(shape, margin, scaler, resources, enabled):
            if not enabled:
                raise RuntimeError("Safety checker must remain enabled")
            return AdmittedChecker(head, pusht_constraints(shape), margin, resources)

        def solver(cost, env, params, planner_seed, device):
            return PlanningCEM(cost, env, params, planner_seed, device, proposal_adapter=adapter)

        def forbidden(*unused, **kwargs):
            raise RuntimeError("Analytic/reference dynamics are forbidden")

        try:
            with ExitStack() as guard:
                for target in (
                    "scripts.run_pusht_smoke.pd_response",
                    "helpers.safecem10_planner.pd_response",
                    "helpers.pusht_smoke.ShapeChecker.check",
                    "helpers.pusht_smoke.ShapeChecker.paths",
                ):
                    guard.enter_context(patch(target, side_effect=forbidden))
                row, _ = run_episode(
                    "proposal" if adapter else "native",
                    case,
                    model,
                    process,
                    pair,
                    protocol,
                    folder / "episode",
                    time.time(),
                    checker_factory=checker,
                    planner_type=solver,
                    device="cuda",
                    resource_counters=counters,
                )
            row["safe_progress"] = progress(row)
        except Exception:
            row = {
                "error": traceback.format_exc(),
                "resources": dict(counters),
                "actions": counters["environment_actions"],
            }
        finally:
            if adapter is not None:
                write_json(folder / "proposals.json", adapter.history)
        row.update(key=key, case_kind=kind, seed=seed, adapter=adapter is not None)
        write_json(folder / "result.json", row)
        if row["resources"].get("predictor_steps", 0) > protocol["maximum_predictor_steps"]:
            raise RuntimeError("Per-episode predictor ceiling exceeded")
        print(
            json.dumps(
                {
                    k: row.get(k)
                    for k in ("key", "safe_completion", "actions", "episode_seconds", "error")
                }
            ),
            flush=True,
        )
        return row

    pilot = ProposalPilot(
        design,
        adapter_spec.initial_parameters(design["adapter"]["seed"]),
        root,
        evaluate,
        resume=args.resume,
    )
    pilot.run()
    audit(args, design)


def audit(args, design):
    root = args.out
    state = json.loads((root / "state.json").read_text())
    freeze = json.loads((root / "source-freeze.json").read_text())
    if digest(args.protocol) != freeze["protocol_sha256"]:
        raise RuntimeError("Protocol changed")
    identities(design, freeze)
    if state["phase"] != "complete" or len(state["jobs"]) != design["maximum_valid_episodes"]:
        raise RuntimeError("Pilot is incomplete")
    if digest(root / "selection.json") != state["selection_sha256"]:
        raise RuntimeError("Selection receipt changed")
    total = collections.Counter()
    checks = []
    for key, row in state["jobs"].items():
        folder = root / "episodes" / key
        if row != json.loads((folder / "result.json").read_text()) or row.get("error"):
            raise RuntimeError("Invalid or changed saved job: " + key)
        with np.load(folder / "episode/trace.npz", allow_pickle=False) as data:
            states, actions, dense = data["states"], data["actions"], data["dense"]
        clearance, arena = clearances(dense, design["cases"][row["case_kind"]])
        unsafe = bool((clearance < 6 - 1e-6).any() or (arena < 6 - 1e-6).any())
        goal = np.asarray(row["goal_state"])
        angle = abs(goal[4] - states[-1, 4]) % (2 * np.pi)
        angle = min(angle, 2 * np.pi - angle)
        success = bool(np.linalg.norm(goal[:4] - states[-1, :4]) < 20 and angle < np.pi / 9)
        safe = success and not unsafe and not row["controller_failure"]
        if (
            unsafe != bool(row["margin_breach"] or row["arena_margin_breach"])
            or success != row["success"]
            or safe != row["safe_completion"]
        ):
            raise RuntimeError("Independent outcome audit mismatch: " + key)
        counts = row["resources"]
        if (
            len(actions) != row["actions"]
            or len(actions) > 150
            or len(dense) != 10 * len(actions) + 1
        ):
            raise RuntimeError("Saved interaction accounting mismatch")
        if (
            counts.get("executed_physics_steps", 0) != 10 * len(actions)
            or counts["predictor_steps"] > design["episode"]["maximum_predictor_steps"]
        ):
            raise RuntimeError("Resource accounting mismatch")
        if progress(row) != row["safe_progress"]:
            raise RuntimeError("Progress feedback mismatch")
        plans = json.loads((folder / "episode/replans.json").read_text())
        direct = 0
        if row["adapter"]:
            adapter = ProposalAdapter.load(
                folder / "controller.json", design["cases"][row["case_kind"]]
            )
            proposals = json.loads((folder / "proposals.json").read_text())
            for proposal in proposals:
                if not np.array_equal(
                    adapter.forward(proposal["features"], len(proposal["proposal"])),
                    np.asarray(proposal["proposal"]),
                ):
                    raise RuntimeError("Proposal weight/feature replay mismatch")
            if counts.get("adapter_forward_calls", 0) != len(proposals):
                raise RuntimeError("Adapter-call counter mismatch")
            for plan in plans:
                if plan.get("executed") == "none":
                    continue
                saved = next(p for p in proposals if p["replan"] == plan["features"]["replan"])
                if len(plan["adapter_checks"]) != design["episode"]["planner"][
                    "iterations"
                ] or not np.allclose(
                    saved["proposal"], plan["adapter_proposal"], atol=1e-6, rtol=0
                ):
                    raise RuntimeError("Adapter injection audit failed")
                direct += int(plan["adapter_selected_directly"])
        if any(p.get("executed") != "none" and p["executed_violation"] > 0 for p in plans):
            raise RuntimeError("Unchecked execution recorded")
        total.update(counts)
        checks.append(
            {
                "key": key,
                "verified": True,
                "minimum_clearance_px": float(clearance.min()),
                "minimum_arena_clearance_px": float(arena.min()),
                "adapter_direct_selections": direct,
            }
        )
    if (
        total["environment_actions"] > design["maximum_total_environment_actions"]
        or total["predictor_steps"] > design["maximum_total_predictor_steps"]
    ):
        raise RuntimeError("Total resource ceiling exceeded")
    groups = {
        kind: [r for k, r in state["jobs"].items() if k.startswith(kind + "_")]
        for kind in (
            "initial",
            "selection_native",
            "selection_initial",
            "selection_elite",
            "validation_native",
            "validation_selected",
        )
    }
    value = {
        "verified": True,
        "episodes": len(checks),
        "optimizer_updates": design["search"]["generations"],
        "selection": state["selection"],
        "summaries": {k: summarize(v) for k, v in groups.items()},
        "resources": dict(total),
        "checks": checks,
        "limitations": design["limitations"],
        "selection_fitness_recomputed": {
            k: panel_fitness(groups["selection_" + k]) for k in ("native", "initial", "elite")
        },
    }
    write_json(root / "audit.json", value)
    print(
        json.dumps({"verified": True, "summaries": value["summaries"], "resources": dict(total)}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "audit"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    design = json.loads(args.protocol.read_text())
    if os.environ.get("RUNPOD_POD_ID") != design["execution_pod_id"]:
        raise RuntimeError("Only the declared rented pod may execute this entry point")
    (train if args.mode == "train" else audit)(args, design)


if __name__ == "__main__":
    main()
