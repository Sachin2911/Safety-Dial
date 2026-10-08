# Safety-Gymnasium locomotion: measured facts

Everything here was verified against `safety-gymnasium==1.0.0` and against the vendored
reimplementation in `experiments/helpers/locoEnv.py`. Dates: 20 September 2026.

## The suite

24 registered `Safety*Velocity` ids (six robots x v0/v1 x plain/`Gymnasium` flavour). All have
`max_episode_steps=1000`.

| Robot | obs | action | dt | frame_skip | nq / nv |
|---|---|---|---|---|---|
| Hopper | 11 | 3 in [-1,1] | 0.008 | 4 | 6 / 6 |
| Walker2d | 17 | 6 in [-1,1] | 0.008 | 4 | 9 / 9 |
| HalfCheetah | 17 | 6 in [-1,1] | 0.05 | 5 | 9 / 9 |
| Ant | 27 | 8 in [-1,1] | 0.05 | 5 | 15 / 14 |
| Swimmer | 8 | 2 in [-1,1] | 0.04 | 4 | 5 / 5 |
| Humanoid | 376 | 17 in [-0.4,0.4] | 0.015 | 5 | 24 / 23 |

`step()` returns a **six-tuple**: `(obs, reward, cost, terminated, truncated, info)`. The
`*VelocityGymnasium-*` ids wrap that back into the standard five-tuple with `info["cost"]`.
`locoEnv.CostInInfo` does the same thing, and is what the registered `safetydial/*` ids use, so
stock gymnasium wrappers and stable-baselines3 work unmodified.

## Mode A: the cost channel

`cost = float(x_velocity > threshold)`, strictly binary, no shaped variant.

| Robot | v0 | **v1** | measure |
|---|---|---|---|
| Hopper | 0.37315 | **0.7402** | signed x velocity |
| Walker2d | 1.7075 | **2.3415** | signed x velocity |
| HalfCheetah | 2.8795 | **3.2096** | signed x velocity |
| Ant | 2.5745 | **2.6222** | 2-D speed of torso COM |
| Swimmer | 0.04845 | **0.2282** | v0 2-D speed, v1 signed x |
| Humanoid | 2.3475 | **1.4149** | 2-D speed of mass centre |

Walker2d, Hopper, HalfCheetah and Swimmer-v1 compare **signed** x velocity, so moving backwards
arbitrarily fast costs nothing. That asymmetry matters: a Walker2d that stumbles and folds
backwards is a Mode B failure with no Mode A signal at all.

## Mode B: the termination channel

`terminated` is `not is_healthy`, forwarded from the underlying gymnasium MuJoCo env.

| Robot | healthy iff |
|---|---|
| Walker2d | `0.8 < z < 2.0` and `-1.0 < pitch < 1.0` |
| Hopper | `z > 0.7`, `\|pitch\| < 0.2`, all other state entries in (-100, 100) |
| Ant | `0.2 <= z <= 1.0` |
| Humanoid | `1.0 < z < 2.0` |
| **HalfCheetah** | **never terminates** (hardcoded `False`) |
| **Swimmer** | **never terminates** (no `is_healthy` at all) |

HalfCheetah and Swimmer carry no Mode B and are useless for this project. Walker2d and Hopper
are the candidates; Ant and Humanoid are possible but were not probed.

Walker2d is not only a z threshold. A random-action termination was observed tripping on the
**angle** (`pitch = -1.018` with `z = 1.117` still comfortably healthy), so "fell over" here
means "torso below 0.8 **or** tilted past 57 degrees".

## Observation layout (Walker2d and Hopper)

`obs = concat(qpos[1:], clip(qvel, -10, 10))`. Only `qpos[0]`, the forward position, is excluded.
So **torso height is `obs[0]` and pitch is `obs[1]`**: the two quantities `terminated` thresholds
are both directly observable. Note the velocities are **clipped to +/-10 in the observation but
not in `qvel`**; raw values above 17 were observed.

Raw state: `env.unwrapped.data.qpos` / `.qvel`, or `state_vector()`. `info` already carries
`x_position` and `x_velocity` every step, which is cheaper than reaching into `data`.

## Measured behaviour

3000 random-action steps per robot:

| Robot | terminations | cost steps | first 5 episode lengths |
|---|---|---|---|
| Walker2d | 139 | **0 / 3000** | 15, 18, 20, 12, 34 |
| Hopper | 138 | 89 / 3000 | 19, 13, 10, 23, 17 |
| Ant | 20 | 15 / 3000 | 135, 41, 79, 151, 54 |
| Humanoid | 125 | 0 / 3000 | 40, 44, 38, 21, 23 |
| HalfCheetah | 0 | 0 / 3000 | 1000, 1000, 1000 |
| Swimmer | 0 | 761 / 3000 | 1000, 1000, 1000 |

Walker2d is the cleanest separation: it falls constantly and never triggers cost. Hopper's
threshold of 0.7402 is so low that its within-hop velocity oscillation crosses it routinely.

## Published baselines (OmniSafe, 1e7 steps, 5 seeds, cost_limit 25)

| Algorithm | Walker2d reward / cost | Hopper reward / cost |
|---|---|---|
| PPO | 6239.52 / 902.68 | 2337.11 / 550.02 |
| PPOLag | 2982.27 / **13.49** | 961.92 / 13.96 |
| TRPOLag | 3207.10 / 14.98 | 1391.79 / 11.22 |
| CPO | 2074.76 / 21.90 | 1713.71 / 13.40 |

**Read these carefully, because they set the Gate 0 problem.** Walker2d reward is
`x_velocity + 1.0 per healthy step - ctrl_cost`. PPOLag at 2982 with cost 13.49 backs out to
roughly 994 steps at mean velocity ~2.0, i.e. **episodes run essentially to truncation and the
fall rate is near zero**. Unconstrained PPO at 6239 runs ~1000 steps at ~5.2 m/s with about 90%
of steps violating. So both trained policies fail a naive "does it fall while cost stays rare"
gate, from opposite directions.

That is a property of converged experts, not of the benchmark, and it is why Gate 0 gates the
**data regime** rather than the environment. See the plan: a PPO-Lagrangian checkpoint mixture
(D4RL medium-replay style) is what supplies falls honestly, because the Lagrange multiplier
holds velocity down at every competence level while balance competence rises independently.

## Rendering

Headless `rgb_array` works with `MUJOCO_GL=egl`, which must be set **before** `import mujoco`.
Default resolution is 480x480; `width=` / `height=` are passed through `make()` and must be set
there, because MuJoCo sizes its offscreen framebuffer once when the GL context is created and
mutating them afterwards silently does nothing.

Measured on the RTX 5090 box: physics only 6535 steps/s; `step` + `render` at 224x224 2070/s;
`set_state` + `render` (replay, no physics) **3101 frames/s**. Multiprocess physics saturates at
about 24 workers and **76k steps/s**; 48 workers is slower, at 68k.

A 224x224x3 frame is 150,528 bytes raw, 20,262 as PNG, 12,767 as JPEG95. Storing state instead
costs 264 bytes/step, which is the basis of the lazy-render design in `locoCollect.py`.

## Why the package is vendored rather than installed

`safety-gymnasium==1.0.0` declares `Requires-Python: >=3.8` and its classifiers **do list Python
3.11**, so the widely repeated "it needs 3.10" is not quite the issue. The real blockers:

1. Hard `==` pins: `gymnasium==0.28.1`, `gymnasium-robotics==1.2.2`, `mujoco==2.3.3`,
   `pygame==2.1.0`. These are unsolvable against `stable-worldmodel`'s gymnasium 1.3 / mujoco
   3.11, so `uv lock` fails outright rather than degrading.
2. A Python 3.11 import crash: `safety_gymnasium/assets/geoms/*.py` declare dataclass fields with
   mutable `np.ndarray` defaults, which 3.11 rejects. Every one of those files belongs to the
   **navigation** suite, which this project does not use.

The velocity task itself is a 27-line file subclassing a stock gymnasium MuJoCo env. It is
copied verbatim into `experiments/helpers/locoEnv.py` (Apache-2.0, header retained), and checked
against the real package by `experiments/scripts/verify_env_equivalence.py`.

Pilot equivalence measurements, vendored (gymnasium 1.3 / mujoco 3.11) against upstream
(gymnasium 0.29.1 / mujoco 2.3.7), same action tape:

| Quantity | Result |
|---|---|
| `qpos` after `reset(seed=0)` | bit-identical |
| cost sum over 300 steps | 5.0 vs 5.0, exact |
| first termination index | 19 vs 19, exact |
| `walker2d.xml` md5 | `7fb5491cd0097896df56653c3d0f18a0`, identical |
| model scalars (`nq`, `nv`, `dt`, solver, iterations, integrator, timestep) | identical |
| max `qpos` drift at step 299 | 4.6e-12 |
