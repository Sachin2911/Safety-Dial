"""Training-only four-action motion screening, with causal predicted robot states."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from helpers.maniguard_followup_budget import atomic
from helpers.maniguard_part2_controller import Config, FeedbackController
from helpers.maniguard_part3_data import load_group
from helpers.maniguard_part3_model import Predictor, linear, state_for, trust_scores
from helpers.maniguard_phase3_protocol import sha


def context_bank(features, directory):
    features, directory = Path(features), Path(directory)
    artifacts = Predictor(directory).artifacts
    corpus = json.loads((features / "corpus.json").read_text())
    sources = {e["episode_id"]: e for e in corpus["episodes"]}
    eligible = {}
    for episode in load_group(features, "fit"):
        output = Path(sources[episode["episode_id"]]["output"])
        trace = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
        chunks = {r["step"]: r for r in trace if r["type"] == "chunk"}
        bounds = {r["step"]: r for r in trace if r["type"] == "action_space"}
        state = state_for(episode, artifacts)
        for step in np.flatnonzero(episode["query"]):
            if step < 1 or step + 4 >= len(state):
                continue
            chunk = np.asarray(chunks[int(step)]["actions"], np.float32)[:4].copy()
            if len(chunk) != 4:
                continue
            chunk[:, -1] = np.where(np.abs(chunk[:, -1]) > .01, np.sign(chunk[:, -1]), -1)
            row = {"episode_id": episode["episode_id"], "scene": episode["scene"], "step": int(step),
                   "x": state[step], "previous": state[step - 1],
                   "prior_action": episode["action"][step - 1],
                   "previous_fingers": episode["robot"][step - 1, -2:], "proposals": chunk,
                   "low": np.asarray(bounds[int(step)]["low"], np.float32),
                   "high": np.asarray(bounds[int(step)]["high"], np.float32)}
            eligible.setdefault(episode["scene"], []).append(row)
    selected = []
    for scene in sorted(eligible):
        rows = eligible[scene]
        if len(rows) < 16:
            raise ValueError("Insufficient fit-only query contexts")
        selected += [rows[int(i)] for i in np.linspace(0, len(rows) - 1, 16, dtype=int)]
    if len(selected) != 32:
        raise ValueError("The prospective bank contains 32 contexts across two fit scenes")
    np.savez_compressed(directory / "context-bank.npz",
                        **{key: np.asarray([r[key] for r in selected]) for key in selected[0]})
    atomic(directory / "context-bank.json", {
        "selection": "16 equally spaced eligible query starts per fitting scene, sorted corpus order",
        "contexts": [{k: r[k] for k in ("episode_id", "scene", "step")} for r in selected],
        "sha256": sha(directory / "context-bank.npz"), "new_VLA_queries": 0,
    })


def screen(pool, directory, device="cpu"):
    import torch

    directory = Path(directory)
    predictor = Predictor(directory, device)
    a = predictor.artifacts
    with np.load(directory / "context-bank.npz") as data:
        bank = {key: data[key].copy() for key in data.files}
    calibration = json.loads((directory / "calibration.json").read_text())
    records = []
    started = time.monotonic()
    for proposal in pool:
        cfg = Config.from_genome(proposal["genome"])
        # Each ensemble member propagates its own predicted robot state, history,
        # controller and action. No future VLA chunk or observation is supplied.
        current = np.repeat(bank["x"][None], 5, axis=0).copy()
        previous = np.repeat(bank["previous"][None], 5, axis=0).copy()
        prior_action = np.repeat(bank["prior_action"][None], 5, axis=0).copy()
        adapters = [[FeedbackController(cfg) for _ in range(len(bank["x"]))] for _ in range(5)]
        for member in adapters:
            for i, controller in enumerate(member):
                controller.last_executed = bank["prior_action"][i].copy()
                controller.previous_fingers = bank["previous_fingers"][i].copy()
                controller.on_query(None)
        trajectory = []
        for step in range(4):
            next_states = []
            for m, model in enumerate(predictor.models):
                robot = (current[m] * a["state_scale"] + a["state_mean"])[:, 32:]
                commands = []
                for i, controller in enumerate(adapters[m]):
                    frame = SimpleNamespace(joints=robot[i, :7].astype(np.float32),
                                            fingers=robot[i, 7:].astype(np.float32))
                    command, _ = controller.act(bank["proposals"][i, step], frame)
                    command = np.clip(command, bank["low"][i], bank["high"][i])
                    controller.applied(command)
                    commands.append(command)
                command = np.asarray(commands)
                design = np.column_stack([
                    current[m], previous[m],
                    (prior_action[m] - a["action_mean"]) / a["action_scale"],
                    (command - a["action_mean"]) / a["action_scale"],
                ]).astype(np.float32)
                with torch.inference_mode():
                    residual = model(torch.as_tensor(design, device=device)).cpu().numpy()
                next_states.append(current[m] + residual)
                prior_action[m] = command
                predictor.member_calls += 1
                predictor.member_examples += len(command)
            previous, current = current, np.stack(next_states)
            trajectory.append(current.copy())
        prediction = np.stack(trajectory, axis=2)
        planned_correction = np.mean([[c.correction for c in member] for member in adapters], axis=0)
        disagreement = np.var(prediction[:, :, 3, :32], axis=0).mean(axis=1)
        score_m, score_c = trust_scores(disagreement, planned_correction, calibration)
        supported = ((planned_correction >= calibration["correction_support"][0]) &
                     (planned_correction <= calibration["correction_support"][1]) &
                     (np.linalg.norm(bank["x"], axis=1) <= calibration["state_norm_limit"]))
        accepted_m = supported & (score_m <= calibration["M_threshold"])
        accepted_c = supported & (score_c <= calibration["C_threshold"])
        heads = linear(prediction.reshape(-1, 41), a["readout_pca"]).reshape(5, -1, 4, 2)
        crossings = float(np.mean(np.any(heads[:, :, :, 0] < 0, axis=2)))
        progress = float(np.mean(heads[:, :, 3, 1] - linear(bank["x"], a["readout_pca"])[None, :, 1]))
        deviation = float(planned_correction.mean())
        records.append({**proposal, "ranking": [crossings, -progress, deviation],
                        "M_trusted": bool(supported.all() and accepted_m.mean() >= .5),
                        "C_trusted": bool(supported.all() and accepted_c.mean() >= .5),
                        "M_context_acceptance": float(accepted_m.mean()),
                        "C_context_acceptance": float(accepted_c.mean()),
                        "supported_contexts": int(supported.sum()),
                        "contexts": len(supported), "planned_correction": planned_correction.tolist(),
                        "disagreement": disagreement.tolist(),
                        "correction_scope": "counterfactual planned, never factual residual data"})
    return {"records": records, "seconds": time.monotonic() - started,
            "member_calls": predictor.member_calls, "member_examples": predictor.member_examples,
            "VLA_queries": 0, "encoder_calls": 0, "known_prefix_actions": 4}
