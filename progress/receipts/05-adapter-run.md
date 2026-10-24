# Task 5 run and accounting

24 October 2026. Instance 57544241 on physical machine 54514. Both engineering
episodes and all 36 scored episodes completed on their first attempt. There were
38 starts, no invalid or partial simulator attempts, no infrastructure replacements
and no outcome-driven repeats. The complete frozen task and seed panel passed
validation. Final preservation and provider teardown are recorded separately in
`05-adapter-preserve.md`.

The first rental, 57539828, failed during container provisioning and was destroyed
without any simulator start or model query. Its provider-reported storage charge
was $0.096 at the first post-destruction query; the original conservative cost
estimate remains labeled separately in the failed-provisioning receipt. The replacement retains the original absolute
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

## Final scored execution and accounting

The pilot queue finished at 20:09:03.725388 UTC, 89 minutes 14.606 seconds after
its start. The engineering queue took 17 minutes 29.026 seconds separately.
Summed evaluator process time was 10,461.980 seconds for the pilot and 2,074.266
for engineering, across two concurrent lanes. Recorder callback overhead summed
to 253.949 and 10.496 seconds respectively, already included in those process
times. These are wall-clock measurements, not simulator time.

| Quantity | Engineering | Scored pilot | Total |
| --- | ---: | ---: | ---: |
| Simulator starts, all valid | 2 | 36 | 38 |
| Policy controls attempted and returned | 4,000 | 69,409 | 73,409 |
| Explicit warm-up controls attempted and returned | 20 | 360 | 380 |
| Policy query attempts and returns | 500 | 8,677 | 9,177 |
| Model-only reseeding attempts and returns | 2 | 36 | 38 |
| Startup query attempts | 6 | 6 | 12 |
| Startup query client returns | 4 | 4 | 8 |
| All model-query attempts | 508 | 8,719 | 9,227 |
| All confirmed model-query returns | 506 | 8,717 | 9,223 |

No simulator command was retried. Four startup model responses were not confirmed
at their probe clients during initial compilation; those requests remain charged.
All episode policy and reseeding requests returned. Policy proposals were
16 by 8, with at most eight executed actions per query. The three early goal
terminations explain the differences in actual controls; all arms shared the
same maximum episode and query budget.

| Scored arm | Episodes | Policy controls | Policy queries | Warm-up controls |
| --- | ---: | ---: | ---: | ---: |
| Native | 12 | 22,343 | 2,793 | 120 |
| Fixed | 12 | 23,066 | 2,884 | 120 |
| Adaptive | 12 | 24,000 | 3,000 | 120 |

All eighteen frozen source hashes matched both executed source archives. All
38 recordings passed index, proposal, state, monitor, gripper, command and factual
adapter-replay validation. The final offline check additionally verified every
numeric NPZ field is finite and all 76 videos declare 256 by 256 frames and
exactly N+1 frames. Analysis decoded those frames and all four plots or frame
sheets for each episode were inspected. The preserved `completion-audit.json`
and `additional-recording-checks.json` provide the machine-readable checks.

The launcher and policy servers had exited by 20:09:47 UTC, with no GPU compute
processes left. All 625 files in the final remote output manifest, 744,023,524
bytes, matched their workstation copies by SHA-256 at 20:10:31 UTC. Additional
preparation logs copied earlier are retained too. The provider subsequently
reported the instance stopped with the container exited. Destruction follows
verified private preservation, as recorded in the final preservation receipt.

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

The frozen admission deadline was 21:26:49.018166 UTC and watchdog deadline
22:56:49.018166 UTC. All jobs finished before either deadline.
Destruction requires a complete local copy and a verified private upload. No
optimizer search or replacement intervention belongs to this allocation.
