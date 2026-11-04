"""Run the prospective frozen-checkpoint evaluation on rented compute only."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import h5py
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
import stable_worldmodel as swm
import torch
from torchvision.transforms import v2 as transforms
import yaml

from compute_common import atomic_json
from model import CountedCost, load_original
from run_io import publish_artifact, report_progress


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if torch.is_tensor(value):
        return serial(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        # DMControl's unused info['success'] is NaN; termination remains the label.
        return {"nonfinite": "NaN" if math.isnan(value) else
                "Infinity" if value > 0 else "-Infinity"}
    return value


def publish(name, value):
    publish_artifact(name, data=json.dumps(serial(value), indent=2, allow_nan=False) + "\n")


class Ledger:
    """Persist attempted actions before execution, including failed calls."""

    def __init__(self, root, phase):
        self.path = root / "interaction-ledger.json"
        self.phase = phase
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {
            "engineering": {"episodes": 0, "actions_attempted": 0},
            "scientific": {"episodes": 0, "actions_attempted": 0}, "panels": []}

    def reserve(self, key, count):
        if key in self.data["panels"]:
            raise ValueError("This scientific panel has already been attempted: " + key)
        limit = 20 if self.phase == "engineering" else 750
        if self.data[self.phase]["episodes"] + count > limit:
            raise ValueError("Prospective episode budget exhausted")
        self.data["panels"].append(key)
        self.data[self.phase]["episodes"] += count
        atomic_json(self.path, self.data)

    def action(self):
        limit = 1000 if self.phase == "engineering" else 37500
        row = self.data[self.phase]
        if row["actions_attempted"] >= limit:
            raise ValueError("Prospective action budget exhausted")
        row["actions_attempted"] += 1
        atomic_json(self.path, self.data)


def panel(dataset, seed, count):
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episodes = np.asarray(dataset.get_col_data(col)).reshape(-1)
    steps = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    unique, inverse = np.unique(episodes, return_inverse=True)
    maxima = np.full(len(unique), -1, dtype=steps.dtype)
    np.maximum.at(maxima, inverse, steps)
    valid = np.flatnonzero(steps <= maxima[inverse] - 25)
    selected = np.sort(valid[np.random.default_rng(seed).choice(len(valid) - 1,
                                                              size=count, replace=False)])
    result = [{"case": i, "row": int(row), "episode": int(episodes[row]),
               "start_step": int(steps[row]), "goal_step": int(steps[row]) + 25}
              for i, row in enumerate(selected)]
    # Verify endpoint indexing using the installed reader before any simulator work.
    for case in result:
        ep, start = case["episode"], case["start_step"]
        offset = int(dataset.offsets[ep])
        assert offset + start == case["row"]
        assert int(steps[offset + case["goal_step"]]) == case["goal_step"]
        assert int(episodes[offset + case["goal_step"]]) == ep
    return result


def fit_scalers(dataset, names):
    processors, records = {}, {}
    for name in names:
        if name == "pixels":
            continue
        values = dataset.get_col_data(name)
        values = values[~np.isnan(values).any(axis=1)]
        scaler = StandardScaler().fit(values)
        processors[name] = scaler
        records[name] = {"mean": scaler.mean_, "scale": scaler.scale_,
                         "variance": scaler.var_, "n_samples": scaler.n_samples_seen_}
        if name != "action":
            processors["goal_" + name] = scaler
    return processors, records


class Recorder:
    def __init__(self, world, ledger, prefix, count):
        self.world, self.ledger, self.prefix = world, ledger, prefix
        self.counts = Counter()
        self.rows = []
        self.chunk = 0
        self.phase = "reset"
        self.case_steps = [0] * count
        self.successes = [False] * count
        self.first_success = [None] * count
        self.count = count
        self.world_steps = 0
        self.restore_image_errors = []
        for i, wrapped in enumerate(world.envs.unwrapped.envs):
            base = wrapped.unwrapped
            self.instrument(base, i)
        original_step = world.step

        def world_step():
            if self.world_steps == 0:
                for index, wrapped in enumerate(world.envs.unwrapped.envs[:2]):
                    rendered = np.asarray(wrapped.unwrapped.render())
                    expected = np.asarray(world.infos["pixels"][index, -1])
                    assert rendered.shape == expected.shape == (224, 224, 3)
                    error = float(np.abs(rendered.astype(float) - expected).mean() / 255)
                    self.restore_image_errors.append(error)
                    path = Path(os.environ["LEWM_RUN_DIR"]) / ".control" / "restored.png"
                    Image.fromarray(rendered).save(path)
                    publish_artifact(f"artifacts/{prefix}/restored-{index}.png", source=path)
            self.phase = "evaluation"
            original_step()
            self.world_steps += 1
            if self.world_steps % 5 == 0:
                self.flush()
            report_progress(self.world_steps * self.count, 50 * self.count,
                            unit="panel environment actions")

        world.step = world_step

    def instrument(self, base, index):
        original_step, original_reset = base.step, base.reset

        def step(action, *args, **kwargs):
            phase = self.phase
            self.counts[phase + "_step_attempts"] += 1
            if phase == "evaluation":
                self.ledger.action()
            if not np.isfinite(action).all():
                raise ValueError("Nonfinite environment action")
            result = original_step(action, *args, **kwargs)
            self.counts[phase + "_step_returns"] += 1
            if phase == "evaluation":
                _, reward, terminated, truncated, info = result
                self.case_steps[index] += 1
                if terminated and self.first_success[index] is None:
                    self.first_success[index] = self.case_steps[index]
                self.successes[index] |= bool(terminated)
                fields = {k: info[k] for k in ("proprio", "state", "qpos", "qvel",
                          "success", "distance_to_target", "coverage", "block_pose") if k in info}
                self.rows.append({"case": index, "step": self.case_steps[index],
                                  "action": action, "reward": reward,
                                  "terminated": bool(terminated), "truncated": bool(truncated),
                                  "info": fields})
            return result

        def reset(*args, **kwargs):
            self.counts["resets_attempted"] += 1
            result = original_reset(*args, **kwargs)
            self.counts["resets_returned"] += 1
            return result

        base.step, base.reset = step, reset
        # PushT creates a new Pymunk space at reset, then advances it while
        # restoring states. Count those advances separately from control actions.
        if hasattr(base, "space") and hasattr(base, "_setup"):
            original_setup = base._setup

            def setup(*args, **kwargs):
                result = original_setup(*args, **kwargs)
                original_physics_step = base.space.step

                def physics_step(*step_args, **step_kwargs):
                    key = self.phase + "_physics_frames"
                    self.counts[key + "_attempted"] += 1
                    value = original_physics_step(*step_args, **step_kwargs)
                    self.counts[key + "_returned"] += 1
                    return value

                base.space.step = physics_step
                return result

            base._setup = setup
        # OGBench exposes these low-level physics advances, including reset settling.
        if hasattr(base, "do_simulation"):
            original_simulate = base.do_simulation

            def simulate(ctrl, n_frames, *args, **kwargs):
                self.counts[self.phase + "_physics_frames_attempted"] += int(n_frames)
                return original_simulate(ctrl, n_frames, *args, **kwargs)

            base.do_simulation = simulate

    def flush(self):
        if self.rows:
            name = f"metrics/{self.prefix}/steps-{self.chunk:03d}.jsonl"
            publish_artifact(name, data="".join(json.dumps(serial(row), allow_nan=False) + "\n"
                                                for row in self.rows))
            self.chunk += 1
            self.rows.clear()


def runtime_record():
    distribution = importlib.metadata.distribution("stable-worldmodel")
    hashes = {}
    for item in distribution.files:
        if str(item).startswith("stable_worldmodel/") and str(item).endswith(".py"):
            hashes[str(item)] = hashlib.sha256(distribution.locate_file(item).read_bytes()).hexdigest()
    return {"stable_worldmodel_version": distribution.version, "source_sha256": hashes,
            "interpreter": sys.executable,
            "simulator_packages": {name: importlib.metadata.version(name)
                                   for name in ("mujoco", "dm-control", "ogbench")},
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda, "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32}


def run_panel(args, config, root, dataset, task_cfg, cases, processors, ledger, seed, iterations):
    count = len(cases)
    prefix = f"{args.task}-{args.phase}-seed{seed}-cem{iterations}"
    key = prefix if args.phase == "scientific" else prefix + "-" + Path(os.environ["LEWM_RUN_DIR"]).name
    publish(f"artifacts/{prefix}/cases.json", {"cases": cases, "seed": seed,
            "actual_goal_offset": 25, "evaluator_chunk_length": 26,
            "evaluation_budget": 50, "cem_iterations": iterations})
    original, model_record = load_original(root, args.task, config["tasks"][args.task])
    model = CountedCost(original)
    publish(f"artifacts/{prefix}/model.json", model_record)
    transform = transforms.Compose([transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=224)])
    solver = swm.solver.CEMSolver(model=model, batch_size=1, num_samples=300, var_scale=1,
                                  n_steps=iterations, topk=30, device="cuda", seed=seed)
    policy = swm.policy.WorldModelPolicy(solver=solver, config=swm.PlanConfig(
        horizon=5, receding_horizon=5, action_block=5), process=processors,
        transform={"pixels": transform, "goal": transform})
    world_config = dict(task_cfg["world"], num_envs=count, max_episode_steps=100)
    publish(f"artifacts/{prefix}/configuration.json", {"world": world_config,
            "original_task_config": task_cfg, "iterations": iterations})
    if args.resume_unstarted_run:
        prior = root.parents[1] / "runs" / args.resume_unstarted_run
        assert prior.parent == root.parents[1] / "runs"
        assert json.loads((prior / "result.json").read_text())["exit_code"] != 0
        assert json.loads((prior / "config.json").read_text()) == config
        previous_cases = json.loads((prior / "artifacts" / prefix / "cases.json").read_text())
        assert previous_cases["cases"] == cases
        assert previous_cases["cem_iterations"] == iterations
        assert not list((prior / "metrics" / prefix).glob("*"))
        assert not (prior / "artifacts" / prefix / "initial.png").exists()
        assert [p for p in ledger.data["panels"] if "-scientific-" in p][-1] == key
        # Only an unexecuted reservation can resume. All scientific actions
        # must already be accounted for by other saved summaries.
        recorded = 0
        for summary in (root.parents[1] / "runs").glob("*/metrics/*/summary.json"):
            row = json.loads(summary.read_text())
            if row["phase"] == "scientific":
                recorded += row.get("interaction_counts", {}).get("evaluation_step_attempts", 0)
        assert recorded == ledger.data["scientific"]["actions_attempted"]
        resumptions = ledger.data.setdefault("reservation_resumptions", {})
        assert key not in resumptions
        record = {"prior_run": args.resume_unstarted_run,
                  "continuation_run": Path(os.environ["LEWM_RUN_DIR"]).name,
                  "prior_actions": 0, "prior_model_queries": 0,
                  "basis": "Stalled during World construction before initial-image publication; "
                           "all scientific actions reconcile to other completed summaries"}
        resumptions[key] = record
        atomic_json(ledger.path, ledger.data)
        publish("artifacts/reservation-continuation.json", record)
    else:
        ledger.reserve(key, count)
    world, recorder = None, None
    start = time.time()
    result = {"task": args.task, "phase": args.phase, "seed": seed,
              "iterations": iterations, "episodes": count, "status": "incomplete"}
    try:
        world = swm.World(**world_config, image_shape=(224, 224), verbose=0)
        recorder = Recorder(world, ledger, prefix, count)
        # Turn the evaluator's permissive missing-callable warning into a hard gate.
        for wrapped in world.envs.unwrapped.envs:
            for call in task_cfg["eval"]["callables"]:
                assert hasattr(wrapped.unwrapped, call["method"]), call["method"]
                for arg in call["args"].values():
                    if arg.get("in_dataset", True):
                        name = arg["value"].removeprefix("goal_")
                        assert name in dataset.column_names, name
        # Check the actual loaded goal, not just requested metadata.
        first = cases[0]
        chunk = dataset.load_chunk([first["episode"]], [first["start_step"]],
                                   [first["goal_step"] + 1])[0]
        assert int(chunk["step_idx"][-1]) == first["goal_step"]
        for label, frame in (("initial", chunk["pixels"][0]), ("goal", chunk["pixels"][-1])):
            image_path = Path(os.environ["LEWM_RUN_DIR"]) / ".control" / f"{prefix}-{label}.png"
            Image.fromarray(frame.permute(1, 2, 0).numpy()).save(image_path)
            publish_artifact(f"artifacts/{prefix}/{label}.png", source=image_path)
        world.set_policy(policy)
        with torch.inference_mode():
            metrics = world.evaluate_from_dataset(dataset=dataset,
                episodes_idx=[c["episode"] for c in cases],
                start_steps=[c["start_step"] for c in cases], goal_offset_steps=26,
                eval_budget=50, callables=task_cfg["eval"]["callables"], save_video=False)
        successes = np.asarray(metrics["episode_successes"], dtype=bool)
        assert np.array_equal(successes, recorder.successes)
        assert recorder.case_steps == [50] * count
        assert all(not p.requires_grad for p in original.parameters())
        rows = [dict(case, success=bool(successes[i]), actions=recorder.case_steps[i],
                     first_success_step=recorder.first_success[i]) for i, case in enumerate(cases)]
        publish(f"metrics/{prefix}/episodes.json", rows)
        result.update(status="completed", successes=int(successes.sum()),
                      success_rate=float(successes.mean()), native_metrics=metrics)
        print(json.dumps(serial(result)), flush=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result.update(elapsed_seconds=time.time() - start, model_queries=model.counts,
                      ledger=ledger.data)
        if recorder:
            result["interaction_counts"] = dict(recorder.counts)
            result["case_steps"] = recorder.case_steps
            result["restored_image_mean_absolute_errors"] = recorder.restore_image_errors
            try:
                recorder.flush()
            except Exception as error:
                result["trace_flush_error"] = repr(error)
                result["status"] = "failed"
        try:
            publish(f"metrics/{prefix}/summary.json", result)
        finally:
            if world:
                world.close()
        del model, original
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["tworoom", "reacher", "pusht", "cube"])
    parser.add_argument("--phase", choices=["engineering", "scientific"], required=True)
    parser.add_argument("--panel-seed", type=int, choices=[42, 43, 44])
    parser.add_argument("--cem-iterations", type=int, choices=[10, 30])
    parser.add_argument("--resume-unstarted-run")
    args = parser.parse_args()
    if args.panel_seed is not None or args.cem_iterations is not None:
        assert args.phase == "scientific"
        assert args.panel_seed is not None and args.cem_iterations is not None
        allowed = {(seed, 30 if args.task == "pusht" else 10) for seed in (42, 43, 44)}
        if args.task != "pusht":
            allowed.add((42, 30))
        assert (args.panel_seed, args.cem_iterations) in allowed
    if args.resume_unstarted_run:
        assert args.phase == "scientific" and args.panel_seed is not None
    config = json.loads(Path(os.environ["LEWM_CONFIG"]).read_text())
    root = Path(config["asset_root"])
    if not str(root).startswith("/workspace/LeWM_Experiment1/"):
        raise ValueError("Use the rented workspace")
    torch.set_num_threads(2)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    runtime = runtime_record()
    gate_path = root / f"engineering-{args.task}.json"
    if args.phase == "scientific":
        gate = json.loads(gate_path.read_text())
        assert gate["status"] == "passed" and gate["evaluation_source_sha256"] == source_hash
        assert gate["model_source_sha256"] == hashlib.sha256(
            Path(__file__).with_name("model.py").read_bytes()).hexdigest()
        assert gate["simulator_packages"] == runtime["simulator_packages"]
    task_cfg = yaml.safe_load((root / "official-source/config/eval" / (args.task + ".yaml")).read_text())
    spec = config["tasks"][args.task]
    dataset = swm.data.HDF5Dataset(spec["dataset_name"], cache_dir=root / "datasets",
                                   keys_to_cache=task_cfg["dataset"]["keys_to_cache"])
    # Verify the pinned download receipt and HDF5 layout before executing anything.
    receipt = json.loads((root / "datasets" / (spec["dataset_name"] + ".receipt.json")).read_text())
    assert receipt["archive_sha256"] == spec["archive_sha256"]
    with h5py.File(dataset.h5_path, "r") as handle:
        publish("artifacts/dataset-layout.json", {k: {"shape": v.shape, "dtype": str(v.dtype)}
                                                   for k, v in handle.items()})
    publish("artifacts/dataset-provenance.json", receipt)
    publish("artifacts/runtime.json", runtime)
    processors, scalers = fit_scalers(dataset, task_cfg["dataset"]["keys_to_cache"])
    publish("artifacts/scalers.json", scalers)
    ledger = Ledger(root, args.phase)
    count = 2 if args.phase == "engineering" else 50
    seeds = [0] if args.phase == "engineering" else [42, 43, 44]
    # Freeze every panel for this task before seeing any of its outcomes.
    panels = {seed: panel(dataset, seed, count) for seed in seeds}
    publish("artifacts/all-panels.json", panels)
    iterations = 30 if args.task == "pusht" else 10
    for seed in seeds:
        if args.panel_seed is not None and (seed, iterations) != (
                args.panel_seed, args.cem_iterations):
            continue
        run_panel(args, config, root, dataset, task_cfg, panels[seed], processors,
                  ledger, seed, iterations)
    if args.phase == "scientific" and args.task != "pusht" and (
            args.panel_seed is None or (args.panel_seed, args.cem_iterations) == (42, 30)):
        run_panel(args, config, root, dataset, task_cfg, panels[42], processors,
                  ledger, 42, 30)
    if dataset.h5_file:
        dataset.h5_file.close()
    if args.phase == "engineering":
        gate = {"status": "passed", "task": args.task, "run_id": Path(os.environ["LEWM_RUN_DIR"]).name,
                "simulator_packages": runtime["simulator_packages"],
                "evaluation_source_sha256": source_hash,
                "model_source_sha256": hashlib.sha256(
                    Path(__file__).with_name("model.py").read_bytes()).hexdigest()}
        atomic_json(gate_path, gate)
        publish("artifacts/engineering-gate.json", gate)


if __name__ == "__main__":
    main()
