#!/usr/bin/env python3
"""Verify hazard section helpers + end-to-end sweep (subset of seeds for CI)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "le-wm"))

import torch
import stable_worldmodel as swm
from sklearn import preprocessing

from utils import get_img_preprocessor
from helpers.linProbeHelpers import wm_transform, make_pusht_state, real_violation_stats
from helpers.hazardSweep import (
    calibrate_box,
    pick_best_box,
    detour_feasible,
    estimate_px_per_step,
    measure_cost_scales,
    suggest_lambda_grid,
    sweep_lambda,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("SKIP: CUDA required for full hazard verification")
        return 0

    h5 = REPO_ROOT / "data" / "processed" / "pusht_expert_train.h5"
    if not h5.is_file():
        print(f"SKIP: missing dataset {h5}")
        return 0

    import os
    os.environ["STABLEWM_HOME"] = str(REPO_ROOT / "data" / "stablewm")

    # --- minimal unit checks (no GPU) ---
    cands = [
        {"usable": False, "baseline_viol": 0.0, "box": (0, 10, 0, 10)},
        {"usable": True, "baseline_viol": 0.25, "box": (1, 11, 1, 11)},
    ]
    assert pick_best_box(cands) == (1, 11, 1, 11)
    lams = suggest_lambda_grid(200.0, 50.0)
    assert lams[0] == 0.0
    assert len(lams) == 6
    print("unit checks OK")

    # --- baseline rollout (cell 31 equivalent) ---
    start_state = make_pusht_state((70, 70), (200, 200), block_theta=np.pi / 4)
    goal_state = make_pusht_state((180, 180), (250, 250), block_theta=np.pi / 4)

    pixel_prep = get_img_preprocessor("pixels", "pixels", img_size=224)
    goal_prep = get_img_preprocessor("goal", "goal", img_size=224)
    transform = {
        "pixels": wm_transform(pixel_prep, "pixels"),
        "goal": wm_transform(goal_prep, "goal"),
    }

    dataset = swm.data.HDF5Dataset(
        path=str(h5), keys_to_cache=["action", "proprio", "state"],
    )
    process = {}
    for col in ["action", "proprio", "state"]:
        proc = preprocessing.StandardScaler()
        data = dataset.get_col_data(col)
        data = data[~np.isnan(data).any(axis=1)]
        proc.fit(data)
        process[col] = proc
        if col != "action":
            process[f"goal_{col}"] = proc

    from stable_worldmodel.solver import CEMSolver

    base_model = swm.policy.AutoCostModel("pusht/lewm").to("cuda").eval()
    solver = CEMSolver(
        model=base_model, batch_size=1, num_samples=300, var_scale=1.0,
        n_steps=30, topk=30, device="cuda", seed=0,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(horizon=5, receding_horizon=1, action_block=5),
        process=process, transform=transform,
    )
    from helpers.linProbeHelpers import make_on_step

    EVAL_BUDGET = 50
    world = swm.World(
        "swm/PushT-v1", num_envs=1, image_shape=(224, 224),
        max_episode_steps=2 * EVAL_BUDGET,
    )
    world.set_policy(policy)
    world.reset(seed=0, options={"state": start_state, "goal_state": goal_state})
    frames, states = [], []
    world._run(max_steps=EVAL_BUDGET, mode="wait", on_step=make_on_step(frames, states))
    baseline_states = np.stack(states)

    # --- box calibration ---
    candidates = calibrate_box(baseline_states, target=(0.15, 0.50), verbose=False)
    hazard_box = pick_best_box(candidates)
    baseline_viol = real_violation_stats(baseline_states, hazard_box)["frac_violating"]
    print(f"baseline_viol={baseline_viol:.2f}  box={tuple(round(b) for b in hazard_box)}")
    assert 0.15 <= baseline_viol <= 0.50, f"baseline viol out of range: {baseline_viol}"

    PROBE_MARGIN = 34.4
    feas = detour_feasible(
        baseline_states[0, :2], goal_state[:2], hazard_box,
        PROBE_MARGIN, estimate_px_per_step(baseline_states), EVAL_BUDGET,
    )
    print(f"detour_feasible={feas['feasible']}")

    # --- load MLP probe (quick train on subset) ---
    import h5py

    class MLPProbe(torch.nn.Module):
        def __init__(self, in_dim=192, hidden=(256, 128), out_dim=2):
            super().__init__()
            layers, d = [], in_dim
            for h in hidden:
                layers += [torch.nn.Linear(d, h), torch.nn.ReLU()]
                d = h
            layers.append(torch.nn.Linear(d, out_dim))
            self.net = torch.nn.Sequential(*layers)

        def forward(self, z):
            return self.net(z)

    prep = get_img_preprocessor("pixels", "pixels", img_size=224)
    with h5py.File(h5) as f:
        pix = f["pixels"][:2000]
        coords = f["state"][:2000, :2]

    pusher_mlp = MLPProbe().to("cuda")
    xs, ys = [], []
    with torch.inference_mode():
        for i in range(0, 2000, 64):
            batch = torch.from_numpy(pix[i:i + 64]).permute(0, 3, 1, 2).unsqueeze(1).to("cuda")
            sample = {"pixels": batch}
            prep(sample)
            z = base_model.encode({"pixels": sample["pixels"]})["emb"][:, 0].cpu().numpy()
            xs.append(z)
            ys.append(coords[i:i + 64])
    Z = np.concatenate(xs)
    Y = np.concatenate(ys)

    Z_t = torch.from_numpy(Z).float().to("cuda")
    Y_t = torch.from_numpy(Y).float().to("cuda")
    opt = torch.optim.Adam(pusher_mlp.parameters(), lr=1e-3)
    pusher_mlp.train()
    for _ in range(30):
        pred = pusher_mlp(Z_t)
        loss = torch.nn.functional.mse_loss(pred, Y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
    pusher_mlp.eval()

    scales = measure_cost_scales(
        base_model, pusher_mlp, hazard_box,
        swm=swm, process=process, transform=transform,
        start_state=start_state, goal_state=goal_state,
        margin=PROBE_MARGIN, eval_budget=EVAL_BUDGET,
    )
    LAMS = suggest_lambda_grid(
        scales["goal_median"],
        scales["hazard_median"],
        hazard_max=scales["hazard_max"],
    )
    print(f"LAMS={[round(lam, 4) for lam in LAMS]}")

    # Quick sweep: 2 lambdas x 1 seed
    test_lams = [LAMS[0], LAMS[len(LAMS) // 2]]
    rows = sweep_lambda(
        test_lams, seeds=(0,),
        base_model=base_model, probe=pusher_mlp, hazard_box=hazard_box,
        margin=PROBE_MARGIN, swm=swm, process=process, transform=transform,
        start_state=start_state, goal_state=goal_state,
        eval_budget=EVAL_BUDGET, verbose=True,
    )
    viols = [r["frac_violating"] for r in rows]
    print(f"sweep viols={viols}")
    print("hazard verification OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
