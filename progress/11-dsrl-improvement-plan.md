# Task 11: make DSRL informative and preserve native behavior

Written 25 October 2026 for handoff. Status: proposed experiment, ready for local
implementation and review. This document does not authorize a rental. The proposed
new allocation is **$15 total**, separate from the completed tasks 6 and 7. Record
Sachin's approval of that allocation before creating an instance. An unused balance
or an old launch instruction is not an allocation for this phase.

## Objective and evidence

Keep the working frozen-policy DSRL pipeline and establish whether better value
estimation, a native-preserving steering interface and useful training experience
can produce a positive development result. This is an ordinary-learning comparator
and a mechanism pilot. A later evolutionary comparison requires a useful replicated
lever, a compact controller definition and matched information and interactions.

Read [first principles](../firstPrinciples.md), the
[task 7 result](receipts/07-dsrl-results.md),
[analysis](receipts/07-dsrl-analysis.json),
[critic diagnostic](receipts/07-dsrl-critic-diagnostic.json), and
[executed source freeze](receipts/07-dsrl-stage-b-v2.json).

Task 7 repaired numerical integration and completed 6 engineering, 24 training and
48 evaluation episodes. Native achieved 3/12 strict safe completions, versus 0/12
DSRL; both had 4/12 violations. Native grasped in 4/12 and DSRL in 0/12. Fixed-zero
and random steering also grasped less often than native. Training produced no
completion reward; 21/24 episodes had entirely zero reward and only 114/6,000
transitions had nonzero reward. The pipeline and final actor export were verified.

The logged critic mean rose from 0.307 to 37.186. The pinned caller disables entropy
in the critic backup, although the actor retains its entropy objective. Our terminal
reward and shaping bound the intended discounted return to [-0.25, 1]. The values
are therefore miscalibrated. Bootstrap overestimation is a hypothesis to test, not
an established explanation for the failed actor.

These observations motivate three distinct hypotheses:

1. Critic targets become unreliable under the long horizon and repeated updates.
2. Replacing full native noise with a repeated vector loses useful native behavior.
3. The learner lacks both successful experience and feedback before discrete milestones.

Repeated noise is part of the original DSRL pi0 design, not an accidental shortcut.
The new residual interface below is our adaptation and must be labelled as such.

## Literature and scope

- [DSRL, Wagenmaker et al.](https://arxiv.org/abs/2506.15799): learn in the frozen
  policy's noise space. Its pi0 implementation is the engineering starting point.
- [Residual RL, Johannink et al.](https://arxiv.org/abs/1812.03201): combine a useful
  base controller with a learned correction. Applying this idea to initial noise is
  a proposed adaptation, not a reproduced result from that paper.
- [RLPD, Ball et al.](https://proceedings.mlr.press/v202/ball23a.html): combine prior
  experience and new online data. We borrow the prior-data idea and explicitly
  describe our sampling rule; this is not a full reproduction of RLPD.
- [Potential shaping, Ng et al.](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf):
  supply progress feedback using the correctly discounted potential difference.

Do not add action-space corrections, gripper overrides, a new visual encoder,
checkpoint finetuning, a learned reward model, hindsight relabelling, curriculum
resets or an evolutionary search to this pilot. Those add separate hypotheses.
The Jar benchmark and strict outcome scorer remain the evaluation authority.

## Ownership, inputs and fresh paths

Task 7 cleanup belongs to its current agent. Its instance is 57632188, already
stopped. Do not restart, destroy or adopt it from this plan. Before any new rental,
read `receipts/07-dsrl-completion/README.md` and require completed preservation and
provider absence. If closure is still underway, local implementation can proceed.

The authoritative local task 7 snapshot is `outputs/dsrl07/remote-final`, not the
older continuously copied `outputs/dsrl07/remote`, which contains stale temporary
files. On another workstation, obtain exact revisions from
`receipts/07-dsrl-completion/preservation.json` and download the pinned private
dataset/model evidence. Do not use an intermediate upload revision or `main` as an
experimental input. Verify manifests and hashes before using replay or checkpoints.

Create new files under `configs/dsrl11/`, `experiments/helpers/dsrl11_*.py`,
`experiments/scripts/*dsrl11*`, `experiments/tests/test_dsrl11*.py`,
`outputs/dsrl11/`, and `progress/receipts/11-dsrl-*`. Keep tasks 6 and 7 immutable.
Reuse tested components by import or a small explicit adapter; do not change old
default behavior to accommodate this experiment. Other tasks 8, 9 and 10 have their
own files, agents and rentals. Preserve their work.

Retain the task 7 source pins unless a documented engineering defect requires a
new freeze: ManiGuard `6bf4c61a1f168cc46e02eb22c41bcb51815e9726`, BEHAVIOR
`88454bd04f75dc57c00ab1f1a00bcde1ff505950`, OpenPI
`215abfb217dbac7d5f1273282331b9b1866c0479`, and DSRL
`7f48937d4553e95244cd81c79236a3256df80597`. Pin the Jar checkpoint and asset hashes
from the existing receipts. Keep deterministic learner XLA operations enabled
before JAX initialization, and use the isolated intended interpreters. Never sync
the root legacy environment into `/venv/main`.

## Stage A: implement and audit before simulator learning

Finish the following on the workstation where possible. GPU numerical checks may
use the proposed rental after approval. Freeze their inputs, scripts, candidate
configurations, seeds and decision rules before executing them.

### A1. Reproduce and diagnose the critic issue

Use only the saved task 7 tiled-noise replay for this diagnostic. Its action labels
are valid in the old tiled-noise MDP. It does not become residual replay. Verify
reward signs/scales, durations, masks, action labels, target-network updates,
episode boundaries and the absence of entropy in the target. Recompute the return
identity and check raw state dictionaries rather than relying only on plotted loss.

Split complete episodes before fitting: calibration episodes T00, T04, T08, T12,
T16 and T20; the other 18 are fitting episodes. This is a development diagnostic,
not held-out benchmark evaluation. Include all selected episodes, including failures.

Run a fixed grid of mean versus minimum reduction over the same ten critics, and
update-to-data settings 5 versus 20, with diagnostic seeds 11061 and 11062. For each
seed, freeze one permutation of fitting transitions. Reveal an initial 500 without
updates, then five successive groups of 50. Each newly revealed transition grants
exactly 5 or 20 updates, sampled from the currently revealed pool. The eight runs
therefore use 1,250 or 5,000 updates each, 25,000 updates total, with the same
750-transition exposure per run. Report both exposure and gradient count. No new
VLA or simulator calls.
Use source-supported reduction options; record changes from task 7 explicitly.

At initialization and after each group of 50 new fitting transitions, measure
finite values, critic spread,
Bellman targets, fraction and magnitude outside [-0.25, 1], and Q for both recorded
actions and actor-sampled actions on calibration observations. Measure distances
from the return range with a 0.05 numerical allowance. Monte Carlo errors on failed
trajectories are descriptive: their zero observed returns do not prove every
alternative action has zero value. Report terminal and nonterminal samples separately.

Admission requires finite states and at most 1% of calibration predictions outside
[-0.30, 1.05] at each of the final three measurement checkpoints, for both diagnostic
seeds and both action sets. Rank admitted settings by worst-seed range-exceedance
fraction, then exceedance magnitude; ties prefer UTD 5, then minimum reduction.
Freeze the resulting single learner configuration before fresh collection. Do not
clip Q or targets into the permitted interval to pass this test. If no setting
passes, diagnose and document the cause within the engineering allowance, then
freeze a repair and repeat only the affected checks. Do not launch learning with
the known unexplained inflation.

This short grid may not reproduce drift that appeared over 90,000 updates. It
diagnoses estimation under saved data, and does not establish that the selected
setting will remain calibrated under the new residual MDP. Live checks below are
also required. Total diagnostic updates are capped at 50,000 including
repairs; every attempt counts.

### A2. Native-preserving latent residual

For each policy query, draw full native noise `epsilon` with shape 16 by 32 from a
dedicated, recorded native Gaussian stream. The actor outputs `u` in [-1, 1]^32.
The decoder receives `epsilon + 0.25 * tile(u, 16)`. Zero `u` must reproduce native
decoding on the same observation and Gaussian draw. Keep the coefficient 0.25 fixed
for this pilot; no magnitude selection from evaluation results. Do not clip the
full resulting noise, which would break zero-residual equivalence.

The actor/critic observe the same 64 by 64 overview RGB and native eight-dimensional
robot state as task 7. They do not receive privileged geometry, monitor history,
success flags, the potential, or the sampled epsilon. Epsilon is independent
environment randomness, drawn afresh per query. Store its exact value or a verified
reconstruction recipe, `u`, injected noise, decoded proposals and executed controls.
The original action normalization and gripper conversion remain unchanged.

Use the eight saved diagnostic-pool observations for same-process native/zero
residual checks at the original 1e-5 absolute/relative tolerance, requiring identical
converted gripper commands. Check nonzero finite decoder influence using four fixed
seeded residual vectors and their negatives. Count model-only calls and startup
queries separately. Cross-process differences remain descriptive. Gripper changes
are reported, not required: native grasp preservation and eventual completion are
the questions. Cap all engineering/model-only VLA calls at 1,200.

### A3. Freeze both reward definitions and geometry

Keep `gamma_control = 0.9998`, transition discount `gamma_control ** actual_controls`,
strict-safe completion base reward 1 only at successful termination, and shaping
scale 0.25. Timeout remains terminal with mask zero and terminal potential zero,
an explicit finite-horizon adaptation from the author's timeout bootstrapping.

R0 is the exact task 7 milestone potential. R1 supplies continuous pre-grasp feedback.
For nonterminal, violation-free, correctly ordered histories define:

`Phi1 = 0.10*reach + 0.20*closed + 0.25*closed*grasp +`
`       0.20*closed*grasp*lift + 0.25*closed*grasp*lift*goal`

Here `closed` is closure latched once observed under the native predicate, and
`grasp` is the current benchmark grasp predicate, not a latched grasp. Define
`reach = 1 - tanh(distance(finger_midpoint, jar_grasp_target)/0.10m)`,
`lift = clip((jar_height - initial_jar_height)/0.10m, 0, 1)`, and
`goal = 1 - tanh(distance(jar_goal_reference, goal_reference)/0.20m)`.
All components lie in [0, 1]. Set the entire potential to zero after a raw violation
or closure-order failure, and at every terminal state. The monitor labels; it does
not veto actions or terminate episodes early on a violation.

Before any live outcomes, identify the exact body/link names, frame conventions,
grasp target and goal references from the pinned benchmark and asset definitions.
Record fixed local offsets and units in the stage-A freeze. Use a benchmark grasp
affordance if one exists; otherwise choose a fixed asset-defined grasp reference
and explain it. Do not infer target coordinates by fitting evaluation trajectories.
If a reference cannot be defined honestly, resolve that engineering issue before
running R1. Do not silently substitute a guessed signal. The frozen geometry is
privileged training reward information, shared across algorithms using this reward.

For either potential, compute `r = base + 0.25*(discount*Phi_after - Phi_before)`.
Verify the full discounted sum equals discounted base return minus 0.25 times the
initial potential, including partial chunks, success, timeout, violations and
reopening/regrasp events. Both rewards preserve the same terminal objective; larger
shaped returns alone cannot demonstrate efficacy. Dense feedback can still be
unhelpful, and the objective remains indifferent between unsuccessful stalling and
unsuccessful violations in exact total return. No explicit violation penalty is
added in this factorial pilot. A later safety-cost variant needs a separate freeze.

### A4. Meaningful implementation checks

Test exact zero-residual injection and RNG isolation, recording/reconstruction of
epsilon, no leakage of privileged reward state, reward bounds and telescoping,
timeout handling, legitimate zero-residual replay labels, replay-mixture counts,
condition-isolated optimizer/RNG state, budget reservations before dispatch,
actor-version attribution, and source parity at the selected configuration.
Retain the task 7 full-state parity criteria. Do not broaden tolerances or hide
nonfinite predictions with clipping. Run relevant existing DSRL regressions and
Ruff. Record exact commands, pass counts and any legitimate skips.

Stage-A outputs: source/input hash freeze, critic audit, selected configuration,
geometry specification, input-boundary audit and local test receipt. These are
concrete launch-review artifacts. No simulator learning is admitted yet.

## Stage B: fresh live engineering and native prior bank

After rental approval, use a new host for the whole comparison and contemporary
native controls. Retain 20 Hz, 256 by 256 collection-time cameras, 16 proposed/eight
executed controls, 2,000 policy-control horizon and the ten-control goal hold.

Run eight engineering episodes on diagnostic-pool tasks 1, 4, 10 and 11: one native
and one bounded-random residual per task, roots 109000 through 109007. Validate
recording, geometry, both rewards, action provenance and all accounting. Profile
128 learner updates and check source parity on a live batch, using a separate
engineering optimizer state that never becomes a training initialization. Check
CPU/GPU actor agreement at the existing 1e-5 criterion and gripper-command identity.
Engineering episodes never enter training replay.

Use development task **3** for the focused learning pilot. This choice is made now:
task 7 native grasped in 2/2 task-3 evaluations and completed safely in 1/2. Those
two episodes are development evidence, not a reliable base-rate estimate. The task
provides a plausible starting point for learning grasp and continuation. Do not
switch tasks after observing this phase's outcomes.

Collect exactly twelve fresh native task-3 prior episodes, roots 110000 through
110011. Keep every valid episode, including failures and violations. Require at
least two strict safe completions to admit the success-bearing replay experiment.
If the fixed bank misses that requirement, report it and preserve the bank; do not
keep sampling until success or switch to a friendlier task. A revised collection
or curriculum would need a new decision and freeze.

Native episodes can be labelled residual action `u=0` only after A2 proves the
exact decoder/RNG construction matches the new environment's zero-residual path.
Log the native draw. Do not insert old full-noise task 7 trajectories into a
32-dimensional absolute-noise replay buffer by inventing latent labels. This
phase uses the fresh bank, not an unverified relabelling of historical success.
Record both R0 and R1 transition rewards from the same complete scorer traces.

Freeze the full stage-C job list, bank hashes, selected learner configuration and
measured time/cost projection before the first online training episode. If the full
remaining schedule cannot fit the authorized allowance, do not begin a partial
four-condition comparison or quietly remove conditions.

## Stage C: one focused four-condition learning comparison

Use the same residual interface and selected critic/UTD setting throughout. The
conditions are R0 without prior replay, R1 without prior replay, R0 with prior
replay, and R1 with prior replay. The predeclared primary condition is **R1 with
prior replay**. The other three are ablations, not candidates for choosing the best
evaluation result.

Use one learner initialization seed, 11060, identically initialized but with
separate mutable states, replay buffers and durable RNG streams per condition.
There are nine online task-3 episodes per condition, roots 111000 through 111008.
The first three use uniform residual `u` in [-1, 1]^32 with the fixed 0.25 scale.
They are online warm-up and count in the nine. Then collect two batches of three
episodes with one actor snapshot per batch. Insert in frozen order, perform the
selected UTD updates per newly admitted transition, then publish. No retroactive
warm-up update credit, extra offline pretraining or reward-dependent update count.

For prior conditions, each batch of 256 updates samples 128 transitions uniformly
from the entire twelve-episode bank and 128 from that condition's online replay.
No successful-only sampling and no failed-episode deletion. Without-prior conditions
sample all 256 from their online replay. Each receives the same post-warm-up
gradient count for the same number of transitions. Capacity at least 6,000 retains
the full bank and online data without eviction. Store condition-specific rewards;
do not share mutable replay or optimizer state between reward definitions.

Interleave conditions by collection round using a frozen seeded permutation so
host time and lane assignment are balanced. Use three simulator/VLA lanes and one
learner GPU. Record condition, episode, actor version and RNG stream on every query
and transition. Publish only after the complete three-episode batch. CPU actor
sampling remains stochastic and follows the author's actor module.

Monitor critic calibration every 500 updates on a fixed bank/online diagnostic
sample selected without evaluation outcomes. Nonfinite states stop updates. If the
same 1% range gate fails at three successive measurements, preserve all states and
pause the scientific comparison for diagnosis. Do not clip or change one condition's
learner while the others retain the old settings. A changed common configuration
requires a new comparison freeze and an explicitly accounted restart; no old work
or spending is erased. Do not spend remaining funds on repetitions that cannot
complete the planned comparison.

Save all publication checkpoints; evaluate only the final episode-nine actor for
each condition. Verify exported parameters and batch statistics against the actual
trained state. Do not select a checkpoint from training return or evaluation scores.

## Stage D: evaluation and interpretation

Before training, freeze forty final evaluations: native plus the four trained
conditions, eight fresh task-3 roots 112000 through 112007 per arm. Use an explicit
`dsrl11` namespace for episode, policy, actor and NumPy RNG derivations. Rotate arms
across all four evaluation lanes and pipeline the frozen schedule. Record the real
simulator-reset seeding behavior; identical episode roots do not prove identical
trajectories. Native uses full native Gaussian noise. No prior-bank or online
training episode is reused as an evaluation.

For every arm report strict safe requested completion, benchmark success, raw and
counted violations, unfinished episodes, contact/engagement, grasp, closure, lift
and requested order with denominators. Include time to milestones, policy controls,
VLA queries, actor requests and latency. Report all four learned conditions even
when none improve. Record both reward returns and critic diagnostics separately
from task outcomes.

The primary condition's practical continuation screen requires at least two more
strict safe completions than both native and R0 without prior, no more raw or
counted violation episodes than native, and engagement in at least 7/8 episodes
and no more than one below native. This is a development threshold, not a
significance test or a risk guarantee. If native is at a ceiling that makes the
threshold impossible, report a ceiling-limited screen; do not relax it afterwards.

Report the R1-minus-R0 contrasts at each replay setting and prior-minus-no-prior
contrasts at each reward. A positive combined condition alone does not establish
which component caused the gain. The residual interface has an exact no-op check,
but this experiment does not train a contemporary absolute-noise comparator, so
it cannot causally attribute performance changes to that interface. Cross-host
task 7 differences remain descriptive.

The prior bank costs twelve real episodes. Report physical shared collection once
and charge its full twelve episodes to each condition's attributable data budget,
including the deliberate non-use by no-prior ablations. This is a data-use ablation,
not proof of sample-efficiency superiority over a strong unseeded learner given
twelve extra online episodes. A later optimizer comparison must match usable
information and total interaction opportunities, including prior collection.

One task, one learner seed and eight evaluations per arm cannot establish
generalization or a stable optimizer advantage. A pass justifies a separately
budgeted replication with multiple learner seeds and the broader development panel.
Held-out tasks stay untouched. No automatic expansion or evolutionary launch.

## Compute, projections and caps

Recommend four RTX 4090s, driver 580.x, at least 48 effective CPU cores, 250 GB RAM
and 200 GB disk. During training, three GPUs serve frozen VLAs/simulators and one
runs the learner; after exporting actors, use all four for evaluation. Verify the
offer's RAM, driver, bandwidth and all-inclusive hourly rate before creating it.
Task 7's historical rate was $1.850370370/hour, including disk; this is evidence for
planning, not a current offer or a promised price. Maximum acceptable rate is
$2.20/hour including disk, subject to the complete-schedule projection fitting.

The proposed fresh $15 cap reserves $3 for transfer, stopped disk, billing margin
and preservation, leaving $12 active. Initiate shutdown before $11.90 active, with
independent local and remote guards and a five-minute acknowledgement margin.
Use the tighter of the money deadline and eight aggregate active rental hours.
The old tasks 6/7 ledger remains historical and is not charged against or silently
funded from this proposed new allocation. Report cumulative project spend separately.

At the historical rate, the active allowance is about 6.4 hours; at $2.20/hour it
is about 5.4 hours. Expect roughly 45-55 minutes for preparation, 20-35 for offline
diagnostics and compilation, 25-40 for live engineering, 25-35 for the bank, 90-120
for online collection, 8-35 for online updates, 65-90 for evaluation, and 20 for the
final local copy. These are planning ranges from task 7, not admission evidence.
They total roughly 5-7 hours, so the full schedule is conditional on a favorable
offer and measured engineering throughput. Use measured conservative projections,
including every remaining phase and cold start, before stage C. A worst-case quote
can legitimately fail admission under $15. Present that result rather than assuming
the budget will fit or increasing it automatically.

Hard ceilings, including discarded and failed attempts:

- 98 simulator starts: 8 engineering, 12 bank, 36 online, 40 evaluation, 2 infrastructure replacements.
- 196,000 policy controls and 980 explicit warm-up controls.
- 27,000 VLA-query attempts, including at most 1,200 model-only engineering/startup queries.
- 200,000 learner-update attempts total; at most 50,000 offline diagnostic,
  2,000 additional engineering, and 120,000 online development updates.
- Exactly one learner seed for the four-condition screen. No extra reward arm,
  coefficient sweep, tuning seed or continuation episode is implied by spare caps.

All planned episodes have 2,000-control maximums and eight-control chunks; count
actual partial chunks. Failed starts, engineering, diagnostics and uploaded bytes
remain charged. Simulator internal reset physics remains separately unknown when
not exposed. The first episode per lane has a 40-minute cold-start limit; cached
episodes have 25 minutes from launch. Reserve 20 minutes for a complete local copy,
and admit no new cached episode within 45 minutes of the effective stop deadline.
Do not start a cold lane within 60 minutes. Persist reservations before dispatch;
an observation timeout does not authorize restarting a still-live job.

If approval or the measured allowance cannot cover the specified full comparison,
finish local work and present the concrete launch review. Any smaller pilot or
larger budget needs an explicit new decision before its outcomes, not a silent
reinterpretation of this schedule.

## Preservation, deliverables and acceptance

Before rental, verify private write access and freeze fresh prefixes in the existing
private dataset `Sachioster/maniguard-jar-replication-20261022`:
`receipts/dsrl11/`, `recordings/dsrl11/` and `jar/dsrl11_*`. Checkpoints belong in a
new private `Sachioster/safety-dial-dsrl11-20261025` repository pinned by revision.
Keep credentials local or in the authorized account environment; never embed them
in configs, manifests, shell arguments or receipts. Record the exact model repo
name in the launch freeze if execution occurs on another date.

Continuously copy evidence locally. After workers are quiescent, freeze the query
ledger with SQLite's backup API, inventory every recording/source/checkpoint and
verify a complete immutable local copy. Stop compute promptly. Upload and verify
every file at pinned private revisions locally, then destroy only the new stopped
instance through the preservation guard. Preserve final provider absence and
billing in a separately pinned closure appendix. Add the finished experiment to
`theOdessy.tex`, compile `theOdessy.pdf` and inspect the new page. Do not leave an
otherwise finished rental running while local verification proceeds.

Deliver:

1. A workstation implementation and a stage-A freeze with tests, critic diagnosis,
   geometry and exact residual/replay/reward semantics.
2. A launch-readiness receipt naming the approval, offer, financial projection,
   input revisions and complete job lists. This is required before any rental/run
   authorization is treated as actionable.
3. Live-engineering and prior-bank receipts, then the pre-learning stage-C freeze.
4. Durable complete training states, replay, RNG and actor-version records for all
   conditions, and the forty frozen evaluations if admitted.
5. A report separating integration/calibration, learning-signal changes, grasp,
   strict safe completion, violations, inactivity, uncertainty and limitations.
6. Full interaction/update/cost accounting, pinned private preservation, rental
   destruction proof and a short finished-task Odyssey entry.

Completion means the authorized phase has been executed and honestly reported,
with every admitted job resolved and resources closed. A positive completion gain
is not guaranteed. If a named prerequisite blocks learning, preserve the concrete
diagnosis and obtain a decision about a materially changed experiment rather than
calling an unexecuted learning comparison complete.

The [handoff prompt](goal-prompt-11-dsrl.md) starts with local implementation and a
launch review. It deliberately does not inherit task 7's rental authorization.
