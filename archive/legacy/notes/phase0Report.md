# Phase 0 report: does Safety-Gymnasium locomotion support the SafetyDial premise?

Sachin Mohan (2699183), 20 September 2026. Covers the first implementation phase against
[the historical irreversibility proposal](https://github.com/Sachin2911/Safety-Dial/blob/a7799a22f76b7af3284a2e94b0ca67f6c65d4f95/docs/revisedProp/RevisedProposal.pdf).
The [current research direction](../docs/researchDirection.md) supersedes its requirements.

## Verdict

**No. Not on Walker2d, which is the robot the proposal names, and not on Hopper either once its
apparent boundary is checked.**

The revised proposal rests on a contrast: the velocity cost is *extrinsic and recoverable*, while
falling is *intrinsic and irreversible*. The second half is false for Walker2d. Given a 4 second
horizon and a competent planner it recovers from lying on the ground in every case tested, and
recoverability never drops below 0.88 at any offset either side of the failure flag.

Hopper looks different at first: recoverability falls from 1.000 at the flag to 0.450 at +40
steps. But re-labelling the same states under weaker recovery predicates dissolves it. Relaxing
the pitch tolerance from 0.10 to 0.15 rad, still well inside the benchmark's own 0.2 rad healthy
band, takes the irrecoverable fraction to zero, as does shortening the dwell requirement from 25
steps to 10:

| recovery predicate (Hopper, 40 states, H=500) | irrecoverable |
|---|---|
| as shipped: z>=0.95, \|pitch\|<=0.10, v<=5, dwell 25 | 0.050 |
| looser velocity, v<=10 | 0.100 |
| **looser pitch, \|pitch\|<=0.15** | **0.000** |
| **shorter dwell, 10 steps** | **0.000** |
| **the benchmark's own healthy band** | **0.000** |

The horizon behaves the same way: 45% irrecoverable at +150 with H=250, 5% with H=500. Hopper's
predicate was measuring whether a hopping robot can hold near-static balance, not whether it can
recover. This is the same class of error as the joint-velocity cap corrected for Walker2d, caught
in one place and left standing in another.

This is the outcome Phase 0 was built to detect. It cost one day rather than surfacing in Stage 2
in November.

Two things survive intact and are worth more than the environment: the **recoverability oracle**,
which is what produced the finding and is a contribution in its own right, and the **entire data
and verification pipeline**, which is environment-agnostic.

## 1. What was built

2,964 lines, all ruff-clean, no `src/` package (per the project rule).

| Module | Role |
|---|---|
| `experiments/helpers/locoEnv.py` | Safety-Gymnasium velocity task vendored into the 3.11 venv; threshold table; env registration; stack fingerprint |
| `experiments/helpers/locoCollect.py` | State-only HDF5 writer and rollout loop |
| `experiments/helpers/locoPolicies.py` | Uniform `act(obs)` adapters: random, scripted-forward, mixed, trained-actor |
| `experiments/helpers/locoData.py` | Lazy-render dataset subclassing `swm.data.Dataset`; worker-safe EGL; render fingerprint |
| `experiments/helpers/locoMetrics.py` | Gate 0 statistics, disjointness lift, matched pairs, AUROC, trivial baselines |
| `experiments/helpers/oracle.py` | The recoverability oracle. Five search tiers. Evaluation only |
| `experiments/helpers/equivCheck.py` | Vendored-env equivalence gates. numpy-only so it loads under both interpreters |
| `experiments/scripts/` | `run_triage.py`, `verify_replay.py`, `verify_env_equivalence.py`, `emit_reference_traj.py` |

## 2. Infrastructure, verified

**The vendored environment matches upstream.** `safety-gymnasium==1.0.0` cannot be installed
alongside `stable-worldmodel` (hard `==` pins on gymnasium 0.28, mujoco 2.3), so the velocity task
is copied verbatim rather than depended on. `walker2d.xml` md5 is `7fb5491cd0097896df56653c3d0f18a0`
on both stacks, with identical model scalars. The ephemeral-uv path for the reference side works
and resolves to safety-gymnasium's exact pins.

**Replay is exact, so pixels need not be stored.** Restoring `(qpos, qvel)` and replaying the
recorded actions reproduces `qpos` to 7.4e-15; renders from the same state are bit-identical.
Walker2d has `na == 0`, so `(qpos, qvel)` is the complete state.

| | bytes/step | 3.25M steps |
|---|---|---|
| state schema | 260 | **0.84 GB** |
| 224x224x3 pixels | 150,528 | 489 GB |

Against roughly 60 GB free, the pixel route does not fit and the state route is free. Frames are
regenerated lazily at 3,101 per second, and the dataset delivers 607 samples/s single-process
(2,427 renders/s), which is 4x to 7x more than the GPU consumes.

**Measured throughput.** Physics 6,500 to 8,200 steps/s per process; multiprocess saturates near
24 workers (48 is slower). Oracle cost per hard state: 8.1 s at tiers T1+T2, 39.1 s with CEM.

## 3. The recoverability oracle

`terminated` is not usable as ground truth for irreversibility. It is a threshold on torso height
and pitch, and both decode from a raw 32x32 grayscale frame at R^2 = 0.99, so scoring a detector
against it is close to circular.

So recoverability is measured directly: from a saved MuJoCo state, with termination disabled,
search for an action sequence returning the robot to a settled upright pose. Any success proves
recoverability, so the search is one-sided and more search can only shrink the irrecoverable set.

Recovery predicate (Walker2d): `1.00 <= z <= 1.80`, `|pitch| <= 0.50`, `max|qvel[0:3]| <= 5.0`,
held 25 consecutive steps. Every constant sits strictly inside the environment's own health set
(`0.8 < z < 2.0`, `|pitch| < 1.0`), which is what stops the oracle restating `terminated`.

Five tiers, short-circuiting: **T0** replays the actions that actually followed the state (a
certificate, not a search); **Tpi** a trained controller; **T1** constant torques; **T2** low-pass
random; **T3** CEM over 10 control knots on the true simulator.

**Validation.** V1 (benign upright states must be recoverable) reaches **1.000** on 40 states,
against 0.92 for the original design. T0 makes it 59x cheaper: 4 ms per state against 235 ms.

## 4. Findings

### 4.1 The benchmark declares failure long before the robot falls

At the step `terminated` fires, Walker2d is typically still standing:

| torso z | pitch (rad) |
|---|---|
| 1.083 | -0.953 |
| 1.135 | -0.973 |
| 1.269 | -0.955 |

Following one episode past the cutoff with termination disabled: z is 1.087 at the flag, 0.999 at
+5 steps, 0.794 at +20, and 0.258 at +60. **The robot does not reach the ground until roughly 60
steps (0.5 s) after the benchmark has already stopped the episode.**

### 4.2 Falling is not irreversible

This is the decisive result. Settled-fallen states, 150 steps past the flag, torso on the floor,
30 per cell, full CEM:

| Robot | mean torso z | irrecoverable at H=250 (2 s) | at H=500 (4 s) |
|---|---|---|---|
| Walker2d | 0.131 | **0.000** | **0.000** |
| Hopper | 0.156 | 0.333 | 0.167 |

And sweeping either side of the flag, 40 states per point, full CEM:

| offset | -15 | -8 | -2 | **0** | +10 | +20 | **+40** | +80 | +150 |
|---|---|---|---|---|---|---|---|---|---|
| Walker2d recoverable | 0.975 | 1.000 | 0.975 | 0.975 | 1.000 | 1.000 | 1.000 | 1.000 | 0.925 |
| Hopper recoverable | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.950 | **0.450** | 0.750 | 0.550 |

Walker2d never moves. Hopper falls sharply at +40, and the curve is not monotone beyond that; at
n=40 those points sit within each other's confidence intervals. **Hopper's drop does not survive
a sensitivity check on the recovery predicate: see the table in the Verdict above.** Relaxing the
pitch tolerance or the dwell requirement, or doubling the horizon, each takes its irrecoverable
fraction to zero.

A replayed Walker2d recovery, from z = 0.252, pitch = -2.90 (upside down on the ground), every
action inside [-1, 1]: t=0 z=0.252, t=40 z=0.420, t=100 z=0.540, **t=140 z=1.471, pitch -0.09**.
It rolls, pushes off, and stands. Real physics, not a simulator exploit: MuJoCo's Walker2d is a
2D planar model with gear-100 actuators and no self-collision.

### 4.3 The full triage

1,288 states labelled with full CEM across offsets either side of the flag (129 minutes):
recoverability sits at 0.93 to 1.00 at **every** offset, including +150. Overall 97.4%
recoverable. The curve is non-monotonic (0.86 at 30 steps before the flag against 0.98 at 150
steps after), which is what prompted the check rather than acceptance.

Matched pairs: only 21 could be built, from 33 irrecoverable states against 1,255 recoverable.
The strongest trivial baseline scores 0.571 on them, but with n=21 that number carries no weight.

### 4.4 Mode A and Mode B are not disjoint

**This corrects a number reported earlier in the session.** An initial run reported disjointness
lift 0.288, i.e. cost being *anti-correlated* with imminent failure, which would have been ideal
for the transfer experiment. That was a bug: control windows were drawn uniformly over the whole
episode, which under termination-disabled collection is ~57% synthetic post-failure padding, and a
fallen walker slides, tripping the velocity threshold 29x more often than an upright one. The
denominator was measuring cost caused *by* the fall.

Corrected, with controls confined to the pre-failure phase and length-matched:

| | value |
|---|---|
| cost rate, pre-failure phase | 0.00049 |
| P(cost in 125 steps before failure) | 0.035 (7 of 200 episodes) |
| P(cost in a matched control window) | 0.000 to 0.001 |
| lift | 35 to infinity across control seeds |

Read carefully: cost is **very rare** before a failure in absolute terms, but when it does occur
it is concentrated immediately before one. Both rates are tiny, so the ratio is unstable, and the
honest statement is that the gate fails rather than that the lift is any particular number. Note
this is under a crude scripted-forward policy that drives maximum torque, accelerates past the
threshold, then trips; a velocity-constrained policy would likely differ. The Gate 0 bound was
`lift <= 2.0`.

## 5. Corrections made during Phase 0

Design changes forced by measurement, not by preference:

1. **The velocity cap applies to the root DOFs only.** Walker2d joint velocities routinely exceed
   10 rad/s in normal gait; an all-DOF cap of 5.0 turned "recovered" into "recovered and then
   stood perfectly still". Measured: 91% of benign states pass an all-DOF cap against 98.9% for
   root-only, and that 91% was exactly the 92% V1 figure the earlier pilot reported.
2. **Added the T0 certificate tier.** Pure search systematically under-detects recoverability for
   *dynamic* states, because a robot in mid-gait needs an actual balancing controller to stay
   upright and zero torque simply drops it.
3. **Collection runs with termination disabled.** `rollout_episode` records state before each
   action, so `terminated[t]` means "action t made it unhealthy" while `qpos[t]` is the last
   *healthy* state, and the episode then ends. **Every row in such a dataset is healthy: it
   contains no failures at all.** This produced a false alarm during validation, where states at
   `terminated == 1` came back 96% recoverable and read as the premise collapsing; they were
   healthy standing robots. Episodes now continue past the cutoff and a `healthy` column carries
   the Mode B signal.

## 6. Adversarial audit

A 5-dimension audit with 3 independent skeptics per finding: 134 agents, 8.8M tokens, 43 findings
raised, 85 verifier votes to refute against 43 to keep. Roughly two thirds of raised findings were
killed, which is the ratio that makes the survivors worth acting on.

Confirmed and fixed:

| Severity | Defect |
|---|---|
| critical | `disjointness_lift` drew controls from the post-failure padding, inverting Gate 0 (section 4.4) |
| high | The recovery predicate had **no upper z bound**, so a ballistic arc could count as recovery: 25 steps is only 0.2 s of flight |
| high | Oracle labels were not reproducible. `set_state` leaves `qacc_warmstart` dirty, so the same rollout gave `-1.4618128503645678` or `...664` depending on what ran before. Now bit-identical |
| high | The G4 equivalence gate was inverted: `first_term` is -1 when no termination occurs, so "reference terminated at step 0, test never terminated" computed as an off-by-one and was **excused as float noise** |
| high | The equivalence harness emitted obs, qvel, reward and per-step cost, shipped them, then compared only totals |
| high | HalfCheetah was registered but crashed on first step; its upstream `step()` is genuinely a different body. Removed rather than left broken |
| high | T2's low-pass filter had no gain compensation: at beta=0.9 it commanded rms torque 0.132 out of 1.0, so the search was 4x weaker than intended |
| medium | `_random_sequences` silently returned fewer sequences than requested, shrinking the budget labels were produced under |
| medium | A truncated rollout returned `margins.max()`, an optimistic and different quantity from the windowed minimum it claimed to be |
| medium | CEM rollouts were excluded from `n_rollouts`, so the budget-saturation check read a near-constant |
| medium | `gate0_table` counted post-failure rows in the Mode A rate |
| medium | The writer did not truncate on a partial write, leaving a ragged file |

Still open, recorded rather than fixed: the lead-time statistic is a minimum over a sparse offset
grid and is censored at 60 steps, so it understates; and the hopeless early exit is the one place
the oracle's one-sidedness is an assumption rather than a guarantee.

## 7. Where this leaves the project

The oracle and the pipeline are sound and environment-agnostic. The environment is not.

Four options, in no particular order:

*(An earlier draft recommended option 1 on Hopper. The sensitivity result above withdraws that:
once the predicate is relaxed anywhere Hopper's irrecoverable set empties, so keeping the suite
would mean defending a threshold rather than measuring a property. Options 2 and 3 are the live
ones and they compose.)*

1. **Controller-relative viability.** Keep Safety-Gymnasium, and define irrecoverable as "outside
   the viability kernel of a realistic controller class" rather than "no omniscient planner can
   recover". Viability kernels in control theory are always relative to admissible controls, and a
   safety filter protects a *specific* policy, so this is arguably the right object. The `Tpi`
   tier already implements it. Cost: the failure set stops being purely a property of the dynamics.
2. **OGBench-Cube.** The August proposal's environment. A cube off the table is genuinely
   irrecoverable. Cost: loses the published safe-RL baselines.
3. **Make the negative result the paper.** "Standard safe-RL locomotion benchmarks have no
   irreversible failures, and their termination flags are hand-set constants a competent
   controller recovers from." Surprising, checkable, and mostly already done. Cost: a benchmark
   critique rather than the Safety Dial thesis.
4. **Add genuine irreversibility to the environment.** Cost: no longer the literal benchmark.

## 8. Reproducing

```bash
uv run python experiments/scripts/verify_replay.py            # storage and replay invariants
uv run python experiments/scripts/run_triage.py               # the full Phase 0 triage
uv run --isolated --no-project --python 3.10 \
  --with "safety-gymnasium==1.0.0" --with "numpy<2" \
  python experiments/scripts/emit_reference_traj.py           # equivalence, reference side
uv run python experiments/scripts/verify_env_equivalence.py   # equivalence, test side
```

Supporting records: `notes/safetyGymLocomotion.md` (measured environment facts),
`notes/terminationIsNotIrreversibility.md` (the finding in detail).

**Not yet run:** the equivalence check end to end (both halves are written and the reference
environment resolves, but the paired run has not been executed).

**Unrelated work in the tree:** `experiments/helpers/safeCEM.py`, `probes.py`,
`experiments/scripts/safe_dial_pusht.py`, `plot_safe_dial.py` and `notes/safeCEM.md` are Push-T
work, not part of Phase 0, and were left untouched.
