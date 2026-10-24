# Task 5: bounded execution adapter

Authorized by Sachin on 24 October 2026: implement the discussed adapter, rent a
separate GPU instance, validate it and run the initial comparison. Task 4's native
recording run belongs to another session and remains unchanged. This protocol is
frozen before task 5 simulator starts. It authorizes an implementation pilot, not
an optimizer comparison. Its source hashes and exact jobs are in
`receipts/05-adapter-plan.json`.

## Mechanism and implementation

The question is whether bounded per-step joint-reference feedback can improve
execution and safe requested completion of the frozen Jar policy. Task 4 informs
whether the scored pilot is appropriate. Write a launch decision citing its
evidence before admitting scored jobs. If tracking is not implicated, retain the
implementation and engineering evidence and report the reason not to spend the
scored allocation. Do not silently substitute another intervention or use task 4
outcomes to retune this preset.

The correction form follows APEX appendix C, equation 33: reference plus terms
in position error, measured velocity and reference acceleration. The paper gives
online adaptation principles and hyperparameters but not executable author code
verified for its discrete implementation. This is therefore an **APEX-inspired
implementation**, not an exact reproduction or an inherited stability guarantee.
Source: https://arxiv.org/html/2606.16504v1 .

Our specified controller runs every 0.05 seconds. It reads only the native seven
arm targets and the corresponding measured joint positions and velocities. It
uses no jar pose, contact flag, goal state, monitor proposition or future state.
The gripper retains its native conversion and clipping. Low-level controller
settings, policy weights, observations, render profile and query schedule stay at
the pinned replication contract. Corrections enter before native action clipping.

Let e = reference - q. Backward finite differences give reference velocity and
acceleration, clipped per joint to 2 rad/s and 20 rad/s^2. Missing first-step
derivatives are zero. Let each joint's feature row be phi = [e, dq, ref_acc], and
let s = dq - ref_vel - 30e. The corrected arm target is reference + clip(phi c,
-0.10, 0.10), in radians. Three coefficients are shared across the seven joints.
Current coefficients produce the current command. For the next command update:

    c_next = project(c - dt * gamma * mean_joints(phi * s)
                     / (1 + mean_joints(phi**2)))

Projection bounds are [0, 0.6], [-0.05, 0.05], [-0.01, 0.01]; gamma is
[30, 0.1, 0.1]. This normalized projected discrete update, shared coefficients,
derivative limits and signed velocity/acceleration bounds are our explicit design
choices. Parameters and history reset each episode. No safety claim follows from
the numerical correction bound.

Three arms: native exact pass-through; fixed coefficients [0.2, 0, 0]; adaptive
coefficients initialized at zero. All have the same observation and query budgets.
Native is tested on identical command/state inputs rather than assuming identical
simulator launches from repeated seeds.

## Frozen episodes and controls

Two engineering jobs on pool tasks: native task 1 seed 390 on lane 0, adaptive task
10 seed 391 on lane 1. These validate live action boundaries and accounting, and
never enter the scored comparison. Both must validate before scored admission.

The scored panel is tasks 0, 3, 6, 9, 15, 17 at seeds 400 and 401 in each of the
three arms: 36 planned episodes. Jobs rotate arm order and GPU lane, giving each
arm six episodes per GPU. No held-out tasks. One policy server and one evaluator
per GPU. Explicitly re-seed model sampling before every episode through a
different-seed model-only request, charged separately. No outcome-driven repeats.

Use ManiGuard 6bf4c61, the pinned Jar checkpoint and legacy 256 configuration:
20 Hz, execute 8 of 16, horizon 2,000, goal hold 10, both camera videos. Run to
native success or horizon. Neither a violation nor inactivity stops a valid trial.

Budget: 38 planned starts (2 engineering + 36 scored), at most two infrastructure
replacement starts, at most one per job. Total ceiling 40 starts, 80,000 policy
controls plus at most 400 explicit warm-up controls. Record warm-up attempts and
returns, policy commands and returns, queries and elapsed time separately. Scene
construction/reset internal physics is not mislabeled as zero; simulator controls
and internal physics are distinct accounting units. Any failed or partial start
remains preserved and charged. Each evaluator has a 25-minute timeout.
An incomplete attempt halts admission for diagnosis. Any infrastructure replacement
requires a recorded repair, uses the existing start ledger and retains the same
total ceiling. A restart does not reset the budget.

Freeze a new rental at no more than $0.75/hour including selected disk, verified
580.x driver, two RTX 3090 GPUs and at least 120 GB disk. Record bandwidth prices.
No new simulator jobs after 4.5 rental hours; stop by six hours. A watchdog stops
the instance, retaining its disk if preservation is incomplete. Destroy only after
full local copy and verified private upload. Preparation failure must not leave an
idle rental running.

## Verification and analysis

CPU checks: exact native bytes and dtype, causal timing, reset, seven-joint mapping,
gripper preservation, finite inputs, correction/projection bounds, feedback sign
on a declared toy plant, and source patch reversibility. Toy-plant evidence is
only an implementation check, not Jar effectiveness evidence.

Every valid episode must have N executed actions, N+1 state rows and ceil(N/8)
16-by-8 proposals. Validate all step and chunk indices, finite arrays, unchanged
recorded scoring, monitor-sidecar equality and command-to-journal equality. Replay
the controller on its factual recorded inputs to check every corrected target.
Retain proposed, corrected, clipped and realized actions separately. Any recording
or integration defect halts admissions until documented repair; it does not consume
unlimited replacement starts.

Report all four benchmark classes, safe requested completion (closure no later
than first departure from support), raw and counted violations, engagement and
unfinished episodes with denominators. Show invalid attempts separately. Compare
native-reference tracking at lags 0 through 8, actual motion, correction size,
saturation and adaptive coefficients. Inspect all episode plots and event frames,
including safe stalls. Improved tracking alone is not an improved task outcome.

Twelve episodes per arm are descriptive. Report paired differences and uncertainty
without a confirmatory significance claim or treating matching seeds as exact
counterfactuals. Do not pool task 4, replication or task 3 results across hosts.
Report nominal Wilson intervals with their independent-episode assumption, and
paired differences using task-level bootstrap resampling (10,000 draws, seed
91405). Six task clusters cannot support a precise generalization claim. Tracking
uses the native absolute target before correction and action clipping; report the
actual controller clipping fraction separately.

## Deliverables

Reusable code under `experiments/helpers`, commands under `experiments/scripts`,
tests under `experiments/tests`. Receipt files `05-adapter-build.md`,
`05-adapter-host.md`, `05-adapter-launch-decision.md`, `05-adapter-run.md`,
`05-adapter-results.md` and `05-adapter-preserve.md`. Preserve every recording,
video, sidecar, result, attempt log, frozen source/config and SHA-256 manifest in
the private dataset under fresh `jar/adapter05_<arm>/`, `recordings/adapter05/` and
`receipts/adapter05/` prefixes. Download and verify every uploaded file at a pinned
revision. Add a completed-task Odyssey entry and recompile. Record final provider
state and charges. Use the findings to size or decline a later search; do not
launch CMA-ES, random search or an ordinary-learning comparison in this rental.
