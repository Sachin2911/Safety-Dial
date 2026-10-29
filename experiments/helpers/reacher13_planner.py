"""SafeCEM on Reacher (task 13): the task-10 constraint-priority cost model with an exact
check that rolls the arm's own MuJoCo model forward.

Reacher is a two-link arm driven by joint torques, and the simulator state is just the
joint positions and velocities. The fingertip after every env step of a candidate plan is
therefore known exactly from the current true state: `ArmModel` steps a copy of the same
compiled model with mujoco.rollout (each env action held for the wrapper's two physics
steps, with MuJoCo's own control clamping) and forward kinematics gives the fingertip. On
the workstation this matched env.step to 0.0 over 25 steps. No probe and no world-model
imagination enter the check.

Execution modes (solver wrappers live in reacher13_runtime):
    mean       the upstream elite mean, which the check never evaluates
    best       the best checked candidate of the solve (iCEM style)
    mean_safe  the elite mean if it passes the same exact check, else the best checked
               candidate; the mean hedges world-model error and the check verifies it
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from mujoco import rollout as mj_rollout

from helpers.reacher13_geometry import fingertip
from helpers.safecem10_planner import DialCostModel, substep_violation

EXECUTE_MODES = ("mean", "best", "mean_safe")


def raw_actions(action_candidates, action_mean, action_scale):
    """(B, S, T, action_block * 2) normalized -> (B, S, T * action_block, 2) env actions."""
    B, S, T, D = action_candidates.shape
    a = action_candidates.reshape(B, S, T * (D // 2), 2)
    dev, dtype = a.device, a.dtype
    mean = torch.as_tensor(np.asarray(action_mean), device=dev, dtype=dtype).view(1, 1, 1, 2)
    scale = torch.as_tensor(np.asarray(action_scale), device=dev, dtype=dtype).view(1, 1, 1, 2)
    return a * scale + mean


class ArmModel:
    """A private copy of the Reacher physics for batched exact rollouts."""

    def __init__(self, threads=4):
        from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper

        self._env = ReacherDMControlWrapper(task="qpos_match")  # owns the compiled model
        self._env.reset(seed=0)
        physics = self._env.env.physics
        self.model = physics.model.ptr
        self.repeat = int(self._env.action_repeat)
        hand = self.model.body("hand").id
        finger = self.model.body("finger").id
        self.links = (float(self.model.body_pos[hand][0]), float(self.model.body_pos[finger][0]))
        self.nstate = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        if self.nstate != 1 + self.model.nq + self.model.nv:
            raise ValueError("Unexpected Reacher state layout")
        self.datas = [mujoco.MjData(self.model) for _ in range(max(1, int(threads)))]

    def paths(self, state, controls, physics_steps=False):
        """Fingertip after each env step, (S, M, 2), for controls (S, M, 2) from state (4,);
        with physics_steps, after every physics step instead, (S, M * repeat, 2)."""
        controls = np.ascontiguousarray(controls, dtype=np.float64)
        S, M, _ = controls.shape
        init = np.zeros((S, self.nstate))
        init[:, 1:5] = np.asarray(state, dtype=np.float64)[:4]
        held = np.repeat(controls, self.repeat, axis=1)
        out, _ = mj_rollout.rollout(self.model, self.datas, init, held)
        qpos = out[:, :, 1:3] if physics_steps else out[:, self.repeat - 1 :: self.repeat, 1:3]
        return fingertip(qpos.reshape(-1, 2), self.links).reshape(S, -1, 2)


def replay_error(states, start_state, actions, arm, terminated=False):
    """Gap between saved fingertip positions and the exact model replaying the actions.

    The wrapper stops repeating an action as soon as the task terminates, so the final env
    step of a successful episode may be a single physics step; it is compared with both."""
    states = np.asarray(states, dtype=np.float64)
    acts = np.asarray(actions, dtype=np.float64).reshape(1, -1, 2)
    fine = arm.paths(start_state, acts, physics_steps=True)[0]
    path = fine[arm.repeat - 1 :: arm.repeat]
    n = min(len(path), len(states))
    err = np.linalg.norm(path[:n] - states[:n, 4:6], axis=1)
    if terminated and n:
        early = np.linalg.norm(fine[(n - 1) * arm.repeat] - states[n - 1, 4:6])
        err[n - 1] = min(err[n - 1], early)
    return {
        "median": float(np.median(err)) if n else None,
        "max": float(err.max()) if n else None,
        "steps": int(n),
    }


class ReacherCostModel(DialCostModel):
    """World-model goal cost plus a clearance constraint checked on the exact fingertip path."""

    def __init__(
        self,
        base_model,
        boxes,
        clearances,
        *,
        arm,
        action_mean,
        action_scale,
        mode="safe",
        lam=0.0,
        tol=0.0,
        track_best=False,
    ):
        super().__init__(
            base_model,
            None,
            boxes,
            clearances,
            mode=mode,
            lam=lam,
            tol=tol,
            arena_constraint=False,
            action_mean=action_mean,
            action_scale=action_scale,
            check="exact",
            track_best=track_best,
            controller={"substeps": 1},
        )
        self.arm = arm
        self.arm_state = None  # (q1, q2, v1, v2) now; pusher_xy holds the fingertip

    def controller_path(self, action_candidates):
        """Fingertip after every env step of every candidate, (B, S, T * block, 2)."""
        if self.arm_state is None:
            raise RuntimeError("The exact check needs the true arm state")
        raw = raw_actions(action_candidates, self.action_mean, self.action_scale)
        B, S, M, _ = raw.shape
        controls = raw.detach().to(torch.float64).cpu().numpy().reshape(B * S, M, 2)
        xy = self.arm.paths(self.arm_state, controls).reshape(B, S, M, 2)
        return torch.as_tensor(xy, device=action_candidates.device, dtype=action_candidates.dtype)

    def exact_violation(self, action_candidates):
        """Summed penetration of the scheduled inflated boxes, (B, S); as in get_cost."""
        horizon = int(action_candidates.shape[2])
        clearances = self.clearances + [self.clearances[-1]] * max(
            0, horizon - len(self.clearances)
        )
        path = self.controller_path(action_candidates)
        per_block = action_candidates.shape[3] // 2
        return substep_violation(path, self.boxes, clearances, per_block)

    def record_choice(self, kind, violation=None):
        """Annotate the solve's last iteration record with the executed plan's provenance."""
        if not self.history:
            return
        entry = {"executed": kind}
        if violation is not None:
            entry["executed_viol"] = float(violation)
        if kind == "best" and self._best and self._best[0] is not None:
            best = self._best[0]
            entry.update(
                {
                    "executed_feasible": best["feasible"],
                    "executed_viol": best["violation"],
                    "executed_goal": best["goal"],
                }
            )
        self.history[-1].update(entry)


def summarize_execution(history, n_steps):
    """How often the elite mean was executed versus the best checked candidate."""
    finals = [h for h in history[n_steps - 1 :: n_steps] if "executed" in h]
    if not finals:
        return {}
    kinds = [h["executed"] for h in finals]
    return {
        "replans_with_choice": len(finals),
        "mean_executed_frac": float(np.mean([k == "mean" for k in kinds])),
        "best_executed_frac": float(np.mean([k == "best" for k in kinds])),
        "executed_viol_mean": float(np.mean([h.get("executed_viol", 0.0) for h in finals])),
    }
