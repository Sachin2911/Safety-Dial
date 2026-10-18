# ManiGuard Part 2: correction-triggered policy feedback

Execution began on 18 October 2026. The 16-episode comparison is in progress. Results
below will be populated only after independent physical trace auditing. The protocol
and source are frozen; no controller setting is being adjusted using these outcomes.

The question is whether command correction identifies a useful time to refresh a
frozen VLA chunk. Four matched training scene/seed blocks compare identity at interval
eight (A), the Part 1 finalist's .04/.02 rad caps at interval eight (B), those same caps
at interval four (C), and those caps with correction-triggered refresh between four and
eight (D). These conditions change feedback timing without changing the common
restraint, finger heuristic or smoothing. They preserve the native 1,100-action horizon.

D sums the L2 difference between proposed and final environment-input arm targets,
divided by `sqrt(7) * .04 rad`. Differences at most `1e-6 rad` are zeroed per coordinate.
The sum resets at every query. After four actual actions, D refreshes once the sum
reaches .5, or after eight actions regardless. Goal termination takes precedence.
History uses inputs after action-space clipping. This signal is command departure,
not measured movement or estimated physical risk. Monitor labels and object states
never enter the execution controller.

The cases are `task_0039/base` with base seeds 4000/4001 and `task_0010/base` with
4010/4011. Condition order cycles ABCD, BCDA, CDAB, DABC across the four blocks. The
native evaluator derives scene-specific sampling seeds, while each episode receives
an isolated policy stream. Initial images, states and first chunks are retained because
matching scene/seed settings does not guarantee identical observations or trajectories.
The main-study test partition remains unopened.

Three 32-action physical checks pass for A/C/D, totaling 96 controls. One earlier
engineering attempt fails before simulator startup because of an import-side-effect
bug. Its missing native counters and two-second process record are preserved. The
repair and one pre-app failure allowance were recorded before the comparison freeze;
no full-episode retry is permitted. Two earlier coordinator/tooling failures also
remain in the run ledger. Supplemental CPU checks include exact identity, actual-input
correction history, query scheduling, source integrity and independent process-tree
cleanup. Full outcome denominators exclude engineering.

Every existing observation is saved losslessly with both native cameras and the
unchanged robot state. Evaluation-only float32 uprightness margins reproduce the
pinned native local +Z convention and each object's tilt threshold. Target-to-goal
distance is also saved. These targets are outside the live controller and native
completion contract. Passive capture adds no render or physics call and applies to all
conditions. Engineering capture costs about 25 ms per observation, below the .15 s
prospective limit. This recording makes Part 3 possible but Part 2 does not depend on
training a world model.

The frozen records are [protocol](../../../configs/maniguard/followup/part2-protocol.json)
and [schedule](../../../configs/maniguard/followup/part2-jobs-plan.json). The source and
frozen protocol milestone is Git commit `d6ec9f2`. Raw evidence is in
`runs/maniguard/phase3-parts2-3/20261018-56582094/part2`. The shared
[GECCO paper](maniguard-phase3-part1-gecco/named.pdf) will incorporate the audited results,
with galleries for every matched block and both cameras.

Reproduce auditing without simulation:

```bash
uv run --no-project --python data/maniguard/envs/cpu/bin/python python \
  scripts/maniguard/followup_evidence.py all
```

The allocation still starts at 18 October 16:00:37 UTC. The user extended the original
23:00:37/00:00:37 physical/hard limits to 19 October 03:00:37/04:00:37 UTC, preserving
all prior setup, implementation, failed and idle time. The independent guard enforces
the physical target for policy, simulator and model workers. An extension changes the
rental limit, not the sixteen-episode allocation.
