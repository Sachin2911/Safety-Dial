# Task 5 run and accounting

24 October 2026. Instance 57544241 on physical machine 54514. Engineering is
complete; the frozen scored pilot is running. Final scored totals and terminal
provider state will be added after the queue finishes and evidence is preserved.

The first rental, 57539828, failed during container provisioning and was destroyed
without any simulator start or model query. Its elapsed rental cost is retained
in the failed-provisioning receipt. The replacement retains the original absolute
admission and shutdown deadlines.

## Engineering checks

The engineering queue began at 18:18:08.479916 UTC and finished at
18:35:37.506136 UTC, elapsed 17 minutes 29.026 seconds. Native task 1 seed 390 ran
on lane 0; adaptive task 10 seed 391 ran on lane 1. Both completed at the full
2,000-control horizon on their first attempt and passed every recording and
controller replay check. Their subprocess wall times were 1,027.716 and 1,046.550
seconds, including cold renderer startup, rollout and shutdown. The sum is
2,074.266 seconds across two lanes, not simulated time.

Engineering consumed 4,000 returned policy controls, 20 returned warm-up controls
and 500 returned policy queries. It additionally consumed two returned model-only
reseeding queries and six startup query attempts, of which four returned to the
probe client. The first request to each server lost its response during initial
compilation; both attempts remain charged. Total engineering query attempts are
508, with 506 confirmed client returns. No simulator interactions were retried.

Native task 1 ended unfinished with an uprightness violation. Adaptive task 10
ended unfinished without contact or violation. The latter's frequent correction
saturation is a behavioral concern, not a failure of the factual integration
checks. These different pool tasks are excluded from the efficacy comparison.

## Scored admission and execution

The [launch decision](05-adapter-launch-decision.md) was recorded after reviewing
all 12 task 4 episodes and before any scored task 5 start. Both engineering
validation hashes are in the runtime `ADMIT_PILOT.json`, admitted at
18:38:31.056352 UTC. Private task 4 evidence remains on the workstation and in the
authorized private dataset; only the admission flag and decision hash went to the
rental.

The pilot queue began at 18:39:49.119797 UTC. It uses plan v4, SHA-256
`631880a5f83e745fc11a392f7f06f1054e6ce7b9b4154f6d9ab358f07e5ae9cb`.
There are 36 preselected jobs: six development tasks, two fresh seeds and three
arms, balanced across the two GPU lanes. Each lane validates an episode before
admitting its successor. No controller setting or task selection has changed.
Source archives preserve the actual executed v4 code separately from the labeled
workstation-only analysis changes.

The policy servers were restarted between engineering and pilot stages. Pilot
startup added six query attempts with four confirmed returns, retained separately
from the episode queries. Every scored episode also has an explicitly charged
reseeding probe. The append-only attempt ledgers retain reservations, starts,
exits, validation results and any partial attempts across queue restarts.

## Accounting units and ceilings

The allocation allows 38 planned starts and at most two infrastructure replacements,
with a 40-start ceiling. Each initialized episode has ten explicit warm-up hold
controls. The maximum is 80,000 policy controls plus 400 warm-up controls. Commands,
returned steps, attempted queries and returned queries are separate journal events.
Policy queries must equal ceil(N/8) in each valid episode. Executed actions are
20 Hz, with eight executed actions from each 16-action proposal, a 2,000-control
horizon and ten-step goal hold under the pinned legacy 256-camera configuration.

Explicit simulator controls are `env.step` calls, not internal physics substeps.
Scene construction and reset may advance internal physics; their substep total is
not instrumented and is not mislabeled as zero. Every process start remains charged.
Subprocess wall time, recorder callback overhead, controls and model queries are
reported separately. Neither violations nor inactivity terminate a valid trial.

Admissions close at 21:26:49.018166 UTC. The verified watchdog stops this rental
by the original allocation deadline, 22:56:49.018166 UTC, retaining disk if needed.
Destruction requires a complete local copy and a verified private upload. No
optimizer search or replacement intervention belongs to this allocation.
