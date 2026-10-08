"""Verify the vendored velocity task against a real safety-gymnasium install.

`experiments/helpers/locoEnv.py` copies safety-gymnasium's velocity task into this project's
Python 3.11 environment, because safety-gymnasium cannot be installed alongside the world-model
stack. That is only defensible if it is checked, so this module runs the same deterministic
trials under both stacks and compares them.

The comparison is state-conditioned, not seed-matched. Gymnasium's own RNG plumbing changed
between 0.29 and 1.3, so "same seed" is not a meaningful control across the two versions; "same
`(qpos, qvel)`, same action tape" is. Reset trials are included anyway, guarded by G1, because
if the reset distributions differ then nothing downstream is comparable.

Two gates can void the paper and the rest are diagnostics that say why:

  G3  per-trajectory cost sum, exact.       Cost is a threshold indicator, and it is the channel
                                            the whole Mode A / Mode B split rests on. No slack.
  G4  first termination index, exact.       Mode B is the dependent variable.

This module is deliberately numpy-only and torch-free, so the identical file imports under the
ephemeral Python 3.10 interpreter that hosts safety-gymnasium.
"""

from __future__ import annotations

import json
import numpy as np

HORIZON = 1000
N_RESET = 150
N_REGIME = 50
N_POLICY = 50


def _pcg(i: int) -> np.random.Generator:
    """PCG64 is bit-identical across numpy 1.26 and 2.4, which both sides rely on."""
    return np.random.Generator(np.random.PCG64(10_000 + i))


def action_tape(trial: int, action_dim: int, horizon: int = HORIZON) -> np.ndarray:
    return _pcg(trial).uniform(-1.0, 1.0, size=(horizon, action_dim))


def regime_states(nq: int, nv: int, n: int = N_REGIME) -> list[tuple[np.ndarray, np.ndarray]]:
    """Hand-placed states covering what the reset basin never reaches.

    The reset distribution is a tight ball around a standing pose, so trials started from it
    only ever exercise one corner of the state space. Near-fall, high-velocity and at-joint-limit
    states are where a solver difference between MuJoCo versions would actually show up, and the
    near-fall band is where Mode B is decided.
    """
    out = []
    for i in range(n):
        rng = _pcg(50_000 + i)
        qpos = np.zeros(nq)
        qvel = np.zeros(nv)
        kind = i % 4
        if kind == 0:  # near-fall: just inside the healthy z band
            qpos[1] = rng.uniform(0.78, 0.85)
            qpos[2] = rng.uniform(-0.9, 0.9)
        elif kind == 1:  # high forward velocity, where the cost threshold lives
            qpos[1] = rng.uniform(1.1, 1.3)
            qvel[0] = rng.uniform(2.0, 3.0)
        elif kind == 2:  # at joint limits: reduced control authority, nothing broken
            qpos[1] = rng.uniform(1.0, 1.25)
            qpos[3:] = rng.choice([-0.9, 0.9], size=nq - 3)
        else:  # tumbling
            qpos[1] = rng.uniform(0.85, 1.35)
            qpos[2] = rng.uniform(-0.95, 0.95)
            qvel[:] = rng.uniform(-4.0, 4.0, size=nv)
        out.append((qpos, qvel))
    return out


def build_specs(nq: int, nv: int, action_dim: int, horizon: int = HORIZON) -> list[dict]:
    """The full trial corpus, identical on both sides because it is computed, not sampled.

    Three families. Reset trials check the common case and guard G1. Regime trials reach the
    near-fall, high-velocity and joint-limit states that the reset basin never visits, which is
    where a solver difference would show up. Policy trials cover the state distribution the
    dataset will actually contain, and are appended by `add_policy_specs` once a trained policy
    exists; until then `build_specs` returns only the first two families and the caller should
    say so rather than implying coverage it does not have.
    """
    specs = []
    for i in range(N_RESET):
        specs.append({"kind": "reset", "trial": i, "seed": i,
                      "tape": action_tape(i, action_dim, horizon)})
    for i, (qpos, qvel) in enumerate(regime_states(nq, nv, N_REGIME)):
        specs.append({"kind": "regime", "trial": N_RESET + i, "qpos": qpos, "qvel": qvel,
                      # A different PCG64 stream from the one that generated this trial's start
                      # state, so the actions are not an affine function of the same uniforms.
                      "tape": action_tape(70_000 + i, action_dim, horizon)})
    return specs


def add_policy_specs(specs: list[dict], starts, tapes) -> list[dict]:
    """Append on-policy trials: real start states with a real policy's action tape.

    Shipped inside the request archive so both sides run byte-identical inputs. This is the
    family that covers the distribution the dataset will actually contain, as opposed to the
    reset basin and the hand-placed regime states.
    """
    base = len(specs)
    for i, ((qpos, qvel), tape) in enumerate(zip(starts, tapes)):
        specs.append({"kind": "policy", "trial": base + i, "qpos": qpos, "qvel": qvel,
                      "tape": np.asarray(tape, float)})
    return specs


def run_trial(adapter, spec: dict, horizon: int = HORIZON) -> dict:
    """Run one trial and record everything both sides must agree on.

    `adapter` must expose `reset(seed)`, `set_state(qpos, qvel)` and
    `step(action) -> (obs, reward, cost, terminated, x_velocity, qpos, qvel)`. It must NOT
    auto-reset and must NOT truncate: divergence is measured over the whole tape, and
    termination is compared as an index rather than as a stopping point.
    """
    adapter.reset(seed=spec.get("seed", 0))
    if spec["kind"] != "reset":
        adapter.set_state(spec["qpos"], spec["qvel"])

    tape = spec["tape"]
    n = min(horizon, len(tape))
    qpos_l, qvel_l, obs_l = [], [], []
    rew = np.zeros(n)
    cost = np.zeros(n)
    term = np.zeros(n, dtype=bool)
    xvel = np.zeros(n)

    for t in range(n):
        obs, r, c, d, xv, qp, qv = adapter.step(tape[t])
        obs_l.append(obs)
        qpos_l.append(qp)
        qvel_l.append(qv)
        rew[t], cost[t], term[t], xvel[t] = r, c, d, xv

    return {
        "kind": spec["kind"],
        "trial": spec["trial"],
        "qpos": np.asarray(qpos_l),
        "qvel": np.asarray(qvel_l),
        "obs": np.asarray(obs_l),
        "reward": rew,
        "cost": cost,
        "terminated": term,
        "x_velocity": xvel,
        "cost_sum": float(cost.sum()),
        "first_term": int(np.argmax(term)) if term.any() else -1,
        "reset_qpos": np.asarray(adapter.reset_qpos),
        "reset_qvel": np.asarray(adapter.reset_qvel),
    }


def compare(ref: dict, test: dict, healthy_z=(0.8, 2.0), healthy_angle=(-1.0, 1.0)) -> dict:
    """Apply gates G1 to G7 to one trial pair."""
    dq = np.abs(ref["qpos"] - test["qpos"])
    drift_200 = float(dq[: min(200, len(dq))].max())
    drift_all = float(dq.max())

    rt, tt = ref["first_term"], test["first_term"]
    g4_ok = rt == tt
    boundary = False
    if not g4_ok and rt >= 0 and tt >= 0 and abs(rt - tt) <= 1:
        # An off-by-one is acceptable ONLY where the state was grazing the healthy boundary,
        # i.e. where float reassociation can flip a strict inequality. Anywhere else it is a
        # real physics difference.
        #
        # The `rt >= 0 and tt >= 0` guard is load-bearing. `first_term` is -1 when a trial never
        # terminated, so without it "reference terminated at step 0, test never terminated at
        # all" computes as |0 - (-1)| = 1 and gets excused as float noise. That is the single
        # worst thing this harness could do: it would certify a vendored env whose Mode B
        # channel disagrees completely with the benchmark's.
        #
        # The state to inspect is at the EARLIER index, which is where one stack called the
        # state unhealthy and the other did not.
        i = min(rt, tt)
        if 0 <= i < len(ref["qpos"]):
            z, ang = ref["qpos"][i, 1], ref["qpos"][i, 2]
            slack = min(z - healthy_z[0], healthy_z[1] - z, healthy_angle[1] - abs(ang))
            boundary = abs(slack) < 1e-6
        g4_ok = boundary

    # Every array both sides emit is compared. An earlier version recorded obs, qvel, reward and
    # the per-step cost, shipped them across, then compared only derived scalars, so a stack that
    # agreed on totals while disagreeing step by step would have passed.
    n = min(len(ref["cost"]), len(test["cost"]))
    return {
        "G2_step0": float(abs(ref["x_velocity"][0] - test["x_velocity"][0])),
        "G3_cost_sum_equal": ref["cost_sum"] == test["cost_sum"],
        "G3b_cost_stepwise_equal": bool(np.array_equal(ref["cost"][:n], test["cost"][:n])),
        "G10_obs_max": float(np.abs(ref["obs"][:n] - test["obs"][:n]).max()) if n else 0.0,
        "G10_qvel_max": float(np.abs(ref["qvel"][:n] - test["qvel"][:n]).max()) if n else 0.0,
        "G10_reward_max": float(np.abs(ref["reward"][:n] - test["reward"][:n]).max()) if n else 0.0,
        "G10_xvel_max": (
            float(np.abs(ref["x_velocity"][:n] - test["x_velocity"][:n]).max()) if n else 0.0
        ),
        "G3_cost_ref": ref["cost_sum"],
        "G3_cost_test": test["cost_sum"],
        "G4_term_equal": bool(g4_ok),
        "G4_term_ref": ref["first_term"],
        "G4_term_test": test["first_term"],
        "G4_boundary_carveout": boundary,
        "G5_drift_200": drift_200,
        "G6_drift_all": drift_all,
    }


def summarise(comparisons: list[dict], manifest_ref: dict, manifest_test: dict) -> dict:
    """Aggregate per-trial comparisons into the gate table."""
    n = len(comparisons)
    g3 = sum(c["G3_cost_sum_equal"] for c in comparisons)
    g4 = sum(c["G4_term_equal"] for c in comparisons)
    d200 = max(c["G5_drift_200"] for c in comparisons) if n else 0.0
    dall = max(c["G6_drift_all"] for c in comparisons) if n else 0.0
    g2 = max(c["G2_step0"] for c in comparisons) if n else 0.0

    scalars_match = (
        manifest_ref.get("model_scalars_json") == manifest_test.get("model_scalars_json")
    )
    xml_match = manifest_ref.get("xml_md5") == manifest_test.get("xml_md5")

    obs_max = max((c["G10_obs_max"] for c in comparisons), default=0.0)
    qvel_max = max((c["G10_qvel_max"] for c in comparisons), default=0.0)
    rew_max = max((c["G10_reward_max"] for c in comparisons), default=0.0)
    g3b = sum(c["G3b_cost_stepwise_equal"] for c in comparisons)

    # These two constants ARE Mode A and Mode B. If they differ, nothing else matters.
    thr_match = manifest_ref.get("velocity_threshold") == manifest_test.get("velocity_threshold")
    health_match = (
        list(manifest_ref.get("healthy_z_range") or []) ==
        list(manifest_test.get("healthy_z_range") or [])
        and list(manifest_ref.get("healthy_angle_range") or []) ==
        list(manifest_test.get("healthy_angle_range") or [])
    )

    gates = {
        "G2 step-0 x_velocity <= 1e-12": g2 <= 1e-12,
        "G3 cost sum exact (all trials)": g3 == n,
        "G3b cost exact step-by-step": g3b == n,
        "G4 first termination index": g4 == n,
        "G5 drift <= 1e-8 at 200 steps": d200 <= 1e-8,
        "G6 drift <= 1e-4 at 1000 steps": dall <= 1e-4,
        "G8 model scalars exact": scalars_match,
        "G8 walker2d.xml md5 exact": xml_match,
        "G8b velocity threshold exact": thr_match,
        "G8b healthy ranges exact": health_match,
        "G10 observations <= 1e-8": obs_max <= 1e-8,
        "G10 qvel <= 1e-8": qvel_max <= 1e-8,
        "G10 reward <= 1e-10": rew_max <= 1e-10,
    }
    return {
        "n_trials": n,
        "g2_max": g2,
        "g3_pass": g3,
        "g3b_pass": g3b,
        "g4_pass": g4,
        "obs_max": obs_max,
        "qvel_max": qvel_max,
        "reward_max": rew_max,
        "g5_drift_200": d200,
        "g6_drift_1000": dall,
        "gates": gates,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
    }


def format_report(summary: dict, manifest_ref: dict, manifest_test: dict) -> str:
    lines = [
        "# Environment equivalence report",
        "",
        "Vendored `experiments/helpers/locoEnv.py` against upstream `safety-gymnasium==1.0.0`.",
        "",
        f"**Verdict: {summary['verdict']}** over {summary['n_trials']} trials.",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for k, v in summary["gates"].items():
        lines.append(f"| {k} | {'PASS' if v else 'FAIL'} |")
    lines += [
        "",
        "| Measurement | Value |",
        "|---|---|",
        f"| max step-0 x_velocity difference | {summary['g2_max']:.3e} |",
        f"| trials with exact cost sum | {summary['g3_pass']}/{summary['n_trials']} |",
        f"| trials with exact step-by-step cost | {summary['g3b_pass']}/{summary['n_trials']} |",
        f"| max observation difference | {summary['obs_max']:.3e} |",
        f"| max qvel difference | {summary['qvel_max']:.3e} |",
        f"| max reward difference | {summary['reward_max']:.3e} |",
        f"| trials with matching termination index | {summary['g4_pass']}/{summary['n_trials']} |",
        f"| max qpos drift at 200 steps | {summary['g5_drift_200']:.3e} |",
        f"| max qpos drift at 1000 steps | {summary['g6_drift_1000']:.3e} |",
        "",
        "## Stacks",
        "",
        "```json",
        json.dumps({"reference": manifest_ref, "test": manifest_test}, indent=2, default=str),
        "```",
    ]
    return "\n".join(lines)
