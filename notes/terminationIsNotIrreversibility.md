# Termination is not irreversibility, and neither is falling

Measured 20 September 2026 on `SafetyWalker2dVelocity-v1`, during Phase 0. This is the finding
the workshop paper is built around, and it is stronger than the proposal anticipated.

## What the benchmark calls failure

Safety-Gymnasium inherits `terminated` from gymnasium's Walker2d, which is `not is_healthy`:

```
healthy  iff  0.8 < qpos[1] (torso z) < 2.0   AND   -1.0 < qpos[2] (pitch) < 1.0
```

Both constants are hand-set in the gymnasium source. Nothing about them is derived from the
dynamics.

## What the robot is actually doing at that moment

Eight sampled states at the step the flag fires, under a mixed random / scripted-forward policy:

| torso z | pitch (rad) | max root \|qvel\| |
|---|---|---|
| 1.083 | -0.953 | 11.2 |
| 1.135 | -0.973 | 10.0 |
| 0.800 | -0.846 | 2.6 |
| 0.813 | -0.829 | 2.6 |
| 1.269 | -0.955 | 8.4 |
| 1.132 | -0.943 | 14.7 |

The robot is **standing**, at roughly its full height, leaning past about 55 degrees. It has not
hit the ground and in most cases is nowhere near it.

Following one episode past the cutoff, with termination disabled:

| offset from the flag | torso z | pitch | healthy |
|---|---|---|---|
| -2 | 1.119 | -0.880 | yes |
| **0** | **1.087** | **-1.105** | **no** |
| +5 | 0.999 | -1.610 | no |
| +20 | 0.794 | -1.735 | no |
| +60 | 0.258 | -3.384 | no |
| +150 | 0.250 | -2.922 | no |

The robot does not reach the ground until roughly 60 steps (0.5 s) after the benchmark has
already declared failure.

## The oracle's verdict

From states at the flag, with termination disabled and a CEM search over 10 control knots on the
true simulator, bounded to the same action range any policy has:

| search budget | fraction irrecoverable |
|---|---|
| T0 certificate + constant torques + 256 low-pass random sequences | 0.96 |
| the above plus CEM | **0.04** |

A competent controller recovers from 96% of the states the benchmark calls failure. Cheap search
does not, which is why a random-shooting oracle would have reported the opposite and why the CEM
tier is not optional.

## The decisive follow-up: falling is not irreversible either

The 96% figure above was first read as "the flag fires early". The stronger test is whether
states well past the flag, with the robot actually on the ground, are irrecoverable. They are
not.

Settled-fallen states (150 steps past the flag, torso on the floor), 30 per cell, full CEM:

| Robot | mean torso z | mean \|pitch\| | irrecoverable at H=250 (2 s) | at H=500 (4 s) |
|---|---|---|---|---|
| Walker2d | 0.131 | 2.50 | 0.033 | **0.000** |
| Hopper | 0.156 | 2.17 | 0.233 | 0.100 |

A replayed Walker2d recovery, from z = 0.252 and pitch = -2.90 (upside down on the ground),
with every action inside [-1, 1]:

| t | torso z | pitch |
|---|---|---|
| 0 | 0.252 | -2.90 |
| 40 | 0.420 | -3.76 |
| 100 | 0.540 | -1.89 |
| 140 | **1.471** | **-0.09** |

It rolls, pushes off, and stands. This is real physics, not a simulator exploit: MuJoCo's
Walker2d is a 2D planar model with gear-100 actuators and no self-collision, so it can right
itself.

**Conclusion: Walker2d has no irreversible failures; Hopper does, but not where the benchmark
says.** Walker2d recovers from everything given time, so any "irreversibility" result on it would
be an artefact of the horizon chosen. Hopper retains roughly half its well-fallen states as
irrecoverable, and its recoverability drops from 1.000 at the flag to 0.450 at +40 steps, which is
a real boundary displaced about 40 steps past the termination flag.

This kills the environment for the proposal's premise as written. It does not kill the oracle,
which is what produced the finding, and it does not kill the project: see the options recorded in
the plan file.

## Why this matters for the project

The proposal's premise was that falling is an intrinsic irreversible failure while the velocity
cost is an extrinsic hand-set one. Half of that is wrong in an interesting way: **the fall flag is
also a hand-set constant**, and it fires far earlier than physical irreversibility. The label-free
criterion is therefore not merely an alternative to the cost channel, it is measurably more
accurate than the benchmark's own failure flag.

That converts what looked like a weakness (see the proposal's "irreversibility is not harm" risk)
into the headline: *termination is not irreversibility, and here is by how much.*

## Two engineering consequences

**1. The failure state was never being recorded.** `rollout_episode` stores the state *before*
each action, so `terminated[t]` means "action t made it unhealthy" while `qpos[t]` is the last
**healthy** state. Under the benchmark's own termination the episode then ends, so every recorded
row in such a dataset is healthy and the dataset contains no failures at all.

This caused a false alarm during oracle validation: scoring states at `terminated == 1` gave 96%
recoverable, which read as the project's premise collapsing. They were healthy standing robots.

**2. Collection now runs with termination disabled.** `locoCollect.rollout_episode` takes
`stop_after_unhealthy`, and the env is built with `terminate_when_unhealthy=False`, so episodes
continue past the cutoff and the genuinely irrecoverable states exist in the data. A `healthy`
column carries the Mode B signal; `terminated` is all-zero by construction and the benchmark's
flag is exactly recoverable from `healthy`. `steps_to_failure` is signed: positive before the
first unhealthy step, zero at it, negative after.

Use `healthy`, never `terminated`, to identify failure states.
