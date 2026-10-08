"""Exact acquired experience and predictor updates for the Walker S5 comparison."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from helpers.predictorAdapt import ClipSet, freeze_for_adaptation, keep_frozen_eval, teacher_forced_loss
from helpers.walkerProtocol import file_sha256


def acquisition_files(bank_dir, seeds, budgets):
    """The files that define prospective selections and their measured transitions."""
    bank_dir = Path(bank_dir)
    names = ["acquisition_roots.json", "replay_clips.npz"]
    for seed in seeds:
        names.append(f"candidates_s{seed}.json")
        for arm in ("random", "boundary"):
            names.append(f"{arm}_s{seed}_selection.json")
            for budget in budgets:
                names.extend(f"{arm}_s{seed}_b{budget}/{name}" for name in
                             ("branches.h5", "roots.json", "execution_ledger.json"))
    return {name: file_sha256(bank_dir / name) for name in names}


def verified_experience(bank_dir, report, arm, seed):
    """Validate exact selection, roots, tapes, ordering and charged execution records."""
    bank_dir = Path(bank_dir)
    point = report["adaptation"][arm][str(seed)][-1]
    selection = json.loads((bank_dir / f"{arm}_s{seed}_selection.json").read_text())
    chosen = selection["selected_candidates"]
    ids = [c["id"] for c in chosen]
    if not selection["selection_complete"] or ids != point["selected_ids"] or len(set(ids)) != point["budget"]:
        raise ValueError("S5 experience must be the complete largest S4 selection")
    if set(ids) & {c["id"] for c in selection["seed_candidates"]}:
        raise ValueError("S5 additional experience contains common seed branches")
    if selection["horizon_steps"] != 100 or selection["branch_steps"] != 100 * len(ids):
        raise ValueError("S5 requires exactly 100 measured transitions per branch")
    if selection["charged_steps"] != point["charged_steps"] or point["acquired_steps"] != selection["branch_steps"]:
        raise ValueError("S5 selection and S4 charged costs differ")
    roots = json.loads((bank_dir / "acquisition_roots.json").read_text())["roots"]
    proposed = json.loads((bank_dir / f"candidates_s{seed}.json").read_text())
    if proposed["source_identity"] != selection["source_identity"] or selection["source_identity"] != report["source_identity"]:
        raise ValueError("S5 candidate root sources differ")
    by_id = {c["id"]: c for c in proposed["candidates"]}
    if any(by_id.get(c["id"]) != c for c in chosen):
        raise ValueError("S5 selected candidates differ from pre-execution proposals")
    locations, ordered = {}, []
    for budget in report["planned_budgets"]:
        path = bank_dir / f"{arm}_s{seed}_b{budget}"
        meta = json.loads((path / "roots.json").read_text())
        ledger = json.loads((path / "execution_ledger.json").read_text())
        round_ids = meta["meta"]["selected_ids"]
        if (ledger["status"] != "complete" or ledger["selected_ids"] != round_ids
                or ledger["attempted_steps"] != len(round_ids) * 100
                or meta["meta"]["executed_steps"] != len(round_ids) * 100):
            raise ValueError("S5 branch execution is incomplete or mischarged")
        with h5py.File(path / "branches.h5", "r") as f:
            if len(f["tape"]) != len(round_ids):
                raise ValueError("S5 branch row order differs from selected IDs")
            for row, candidate_id in enumerate(round_ids):
                if candidate_id in locations or candidate_id not in ids:
                    raise ValueError("S5 branch ID is duplicated or outside the selection")
                candidate = by_id[candidate_id]
                root = roots[candidate["root_index"]]
                stored = meta["roots"][int(f["root_index"][row])]
                for key in ("root_id", "episode", "step", "qpos", "qvel", "history_qpos", "history_qvel", "history_actions"):
                    np.testing.assert_array_equal(stored[key], root[key], err_msg=f"Root mismatch: {key}")
                if int(root["episode"]) % 4 not in (2, 3):
                    raise ValueError("S5 training experience is outside acquisition roles")
                if f["tape"][row].shape != (100, 6) or f["qpos"][row].shape[0] != 101:
                    raise ValueError("S5 branch is incomplete")
                np.testing.assert_array_equal(f["tape"][row], np.asarray(candidate["tape"], dtype=np.float32))
                np.testing.assert_array_equal(f["qpos"][row][0], root["qpos"])
                np.testing.assert_array_equal(f["qvel"][row][0], root["qvel"])
                locations[candidate_id] = (path, row)
                ordered.append(candidate_id)
    if ordered != ids:
        raise ValueError("S5 cumulative branch order differs from selection")
    return selection, locations


def verify_set_b_rows(data_b, selection, locations):
    """The A-random update and B pretraining must consume identical recorded rows."""
    data_b = Path(data_b)
    origins = json.loads((data_b / "origins.json").read_text())
    branch_origins = [x for x in origins if "candidate_id" in x]
    if [x["candidate_id"] for x in branch_origins] != [c["id"] for c in selection["selected_candidates"]]:
        raise ValueError("Set B acquired episode order differs from A-random experience")
    with h5py.File(data_b / "setB.h5", "r") as b:
        for origin in branch_origins:
            if origin["role"] != "training":
                raise ValueError("Set B acquired episode is not in training")
            ep = origin["episode"]
            if int(b["ep_len"][ep]) != 100:
                raise ValueError("Set B did not retain exactly 100 rows per branch")
            start = int(b["ep_offset"][ep])
            path, row = locations[origin["candidate_id"]]
            with h5py.File(path / "branches.h5", "r") as f:
                for name in ("qpos", "qvel", "x_velocity", "action"):
                    value = f["tape" if name == "action" else name][row]
                    if name in ("qpos", "qvel"):
                        value = value[:-1]
                    np.testing.assert_array_equal(b[name][start:start + 100], value,
                        err_msg=f"Set B and A-random experience differ: {name}")
    return {"n_branches": len(branch_origins), "n_identical_transitions": 100 * len(branch_origins),
            "fields": ["qpos", "qvel", "action", "x_velocity"], "root_and_tape_identity_verified": True}


def exact_branch_clips(selection, locations, imaginer, ctx, scaler, *, clip_stride=1):
    """Only the N pre-action rows stored in B; no seed or root history is supervised."""
    if clip_stride < 1:
        raise ValueError("clip_stride must be positive")
    latents, actions = [], []
    mean, std = scaler
    for candidate in selection["selected_candidates"]:
        path, row = locations[candidate["id"]]
        with h5py.File(path / "branches.h5", "r") as f:
            qpos, qvel, tape = f["qpos"][row][:-1], f["qvel"][row][:-1], f["tape"][row]
        # The last recorded row is 99. No unrecorded terminal state at step 100 enters.
        z = imaginer.encode(ctx.render_many(qpos, qvel)).cpu().numpy()
        for start in range(0, len(qpos) - 30, clip_stride):
            latents.append(z[start + np.arange(4) * 10])
            blocks = tape[start:start + 30].reshape(3, 10, 6)
            actions.append(((blocks - mean) / std).reshape(3, 60))
    return ClipSet(np.asarray(latents, np.float32), np.asarray(actions, np.float32),
        {"source": "exact_additional_branch_rows", "n_transitions": selection["branch_steps"],
         "history_prefix_steps": 0, "seed_steps": 0, "clip_stride": clip_stride,
         "selected_ids": [c["id"] for c in selection["selected_candidates"]]})


def adapt_exact_experience(base_state, template, acquired, replay, cfg, *, device="cuda"):
    """S4's optimizer and replay mixture with separate means for unequal clip lengths.

    Each minibatch loss is the same sample-weighted mixture as S4. Separating the two
    means keeps its original replay clips intact while acquired clips use only B rows.
    """
    if cfg.modules != "predictor_side" or cfg.loss != "teacher_forced" or cfg.ctx != 3:
        raise ValueError("Exact S5 adapter requires the declared S4 teacher-forced predictor-side recipe")
    if len(acquired) == 0 or len(replay) == 0 or not 0 < cfg.replay_frac < 1 or cfg.steps < 1:
        raise ValueError("Exact S5 adaptation requires acquired data and original replay")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = copy.deepcopy(template).to(device)
    model.load_state_dict(base_state, strict=True)
    parameters = freeze_for_adaptation(model, cfg.modules)
    opt = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    rng = np.random.default_rng(cfg.seed)
    zr, ar = torch.from_numpy(replay.latents).to(device), torch.from_numpy(replay.actions).to(device)
    za, aa = torch.from_numpy(acquired.latents).to(device), torch.from_numpy(acquired.actions).to(device)
    nr = int(round(cfg.batch_size * cfg.replay_frac))
    na = cfg.batch_size - nr
    if min(nr, na) < 1:
        raise ValueError("Both experience sources must appear in every minibatch")
    log = {"cfg": cfg.to_dict(), "steps": [], "n_acquired": len(acquired), "n_replay": len(replay)}
    started = time.time()
    for step in range(cfg.steps):
        keep_frozen_eval(model, cfg.modules)
        ir = torch.as_tensor(rng.integers(len(zr), size=nr), device=device)
        ia = torch.as_tensor(rng.integers(len(za), size=na), device=device)
        loss_replay = teacher_forced_loss(model, zr[ir], ar[ir], cfg.ctx)
        loss_acquired = teacher_forced_loss(model, za[ia], aa[ia], cfg.ctx)
        loss = (nr * loss_replay + na * loss_acquired) / cfg.batch_size
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite exact-experience adaptation loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip, error_if_nonfinite=True)
        opt.step()
        scheduler.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == cfg.steps:
            log["steps"].append({"step": step + 1, "loss": float(loss.detach()),
                "loss_acquired": float(loss_acquired.detach()), "loss_replay": float(loss_replay.detach())})
    log["wall_clock_s"] = time.time() - started
    return model.eval(), log
