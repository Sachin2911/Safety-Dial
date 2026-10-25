# Task 6 DSRL steering of the frozen Jar policy

Final planning version, revised after independent review, 25 October 2026.
Sachin selected four RTX 4090 GPUs,
potential-shaped safe completion, and a **hard US$15 total cap**. This version
replaces the earlier $45 proposal. The independent review and subsequent source
checks are recorded in the review receipt. Sachin subsequently authorized
implementation, Vast rental and SSH execution within this plan and its $15 cap.
Stage A and B still require their respective freezes before eligible work begins.

The allocation is six engineering episodes, 24 training episodes for one learner,
and 48 evaluations, twelve each for native, fixed-zero, bounded-random and learned
noise. Two infrastructure replacement starts are reserved. Sparse-versus-shaped
reward training is deferred; the sparse reward is still logged for every episode.

## Question and contribution

Can an observation-conditioned noise policy improve safe requested completion of
the frozen ManiGuard Jar policy? Start with the authors' DSRL-SAC implementation,
adapt its interface to our pi0.5 checkpoint, and test it against contemporary native,
fixed and random-noise controls. This small development pilot tests integration,
learning behavior and an initial completion signal. It can nominate a useful lever
for replication and provide an ordinary-learning baseline; it cannot establish
an evolutionary advantage or isolate the effect of reward shaping.

Task 4 found command choice and continuation to be the broader explanation for
failure. Task 5 found safe requested completion of 1/12 native, 1/12 fixed feedback
and 0/12 adaptive feedback. Those observations motivate a new mechanism; they do
not establish that noise steering will work. See the [tracking results](receipts/04-tracking-results.md)
and [adapter results](receipts/05-adapter-results.md).

DSRL learns a policy over a frozen diffusion or flow policy's input noise. Its
published pi0 experiments use SAC and repeat a single noise vector across an action
chunk. Our pi0.5 integration, safety reward and parallel collection are explicit
adaptations, not an exact reproduction. [DSRL paper, appendix C.5](https://arxiv.org/html/2506.15799v1#A3.SS5)

Prior evidence on this checkpoint is unfavorable for the tested noise interventions.
The archived additive random-basis pilot established small arm-target changes but
no executed-prefix gripper flips in its directional probes. A separate gripper-coordinate
offset diagnostic found no command changes at amplitude 0.5 and changes in 3/4 Jar
observations at amplitude 2.0. It did not establish authority for this pilot's
absolute, tiled 32-dimensional noise bounded at magnitude 1.0. The old render/profile,
sample panel and intervention differ; these are historical warning signs, not a
resume instruction or proof that this interface cannot help.
[Progress audit](../archive/20261022/reports/Recent%20ManiGuard%20progress%20audit.md),
[pilot and supplemental probe](../archive/20261022/docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md).

A positive pilot would justify replicated development experiments and then an
optimizer comparison with matched interactions and information. Applying DSRL alone
is not the proposed evolutionary contribution. That later study must include tuned
fixed settings, parameter random search and ordinary learning on the same interface.

## Source and benchmark contract

Use the official [dsrl_pi0 source at 7f48937](https://github.com/nakamotoo/dsrl_pi0/tree/7f48937d4553e95244cd81c79236a3256df80597).
Reuse its learner rather than writing a new SAC implementation. Record its license,
resolved dependencies, any patches and the trainable actor and critic parameter counts.
Keep the simulator, OpenPI server and learner in separate intended environments.
Do not replace the known ManiGuard environment with the author's LIBERO installation.

Preserve the established ManiGuard 6bf4c61, BEHAVIOR 88454bd, OpenPI 215abfb and
Jar checkpoint revision 1d84eda070313a202a595e449fcf41a1a1e8a546. Verify full
revisions and assets against the existing installation receipts before execution.
Policy weights and their normalization stay frozen. Use 20 Hz control, 256 by 256
collection-time cameras, 16 proposed actions with the first 8 executed, a 2,000-control
horizon, and goal hold for 10 controls. Retain native preprocessing and action clipping.
See the [paper configuration](receipts/01-jar-eval.yaml) and [task 5 environment record](receipts/05-adapter-host.md).

All scored comparisons run on one new physical host with their own native controls.
Do not pool historical outcomes or split comparator arms across hosts. Repeated seeds
match the schedule, not exact simulator trajectories. Do not change cameras, solver,
control rate, horizon or recording frequency to improve throughput.

## Controller and information

At each native policy query, the actor reads a 64 by 64 downsample of the overview
camera and the same eight-dimensional robot state supplied to the base policy.
The base policy still receives both native camera images and its original prompt.
This first actor uses one camera, following the author configuration. It does not
receive task identifiers, simulator object poses, contact flags, monitor propositions,
reward history or future observations. The critic receives the same observation
and the chosen noise vector. The scorer may use privileged simulator state solely
to compute rewards and outcome labels.

The [workstation shape audit](receipts/06-dsrl-shape-audit.json) has read the exact
registered ManiGuard serving configuration and the pinned checkpoint's small metadata
files without importing JAX or loading weights. It confirms action_horizon = 16,
action_dim = 32 and pi05 = true. Include this receipt and its source hashes in the
stage A freeze before renting; do not postpone this check to the rental.

The actor samples a bounded 32-dimensional vector in [-1, 1] at training time.
Repeat that vector across the 16 latent time positions and supply it through the
explicit-noise sampler hook. Retain a runtime shape assertion as a backstop; any
different shape stops integration for review.
The decoded physical action has eight coordinates. Do not confuse that dimension
with the latent width or discard latent coordinates without an experiment.

There is one base-policy decode per query, no candidate selection, action residual,
retry controller or monitor veto. Separate observation preprocessing, noise generation,
decoded proposals, native clipping and execution in the recording. Native mode must
retain its original full Gaussian-noise path and random-number generator behavior.
Zero explicit noise is a separate control, not an identity setting.

## Learning configuration

Use the author's LIBERO preset as the starting configuration: batch 256, actor
learning rate 0.0001, critic and temperature rates 0.0003, hidden widths 128/128/128,
four 32-channel convolution layers, bottleneck 50, target smoothing 0.005,
10 critics with mean reduction, action magnitude 1 and automatic entropy target.
Retain the remaining encoder and augmentation settings from the resolved source.
Freeze the full configuration, including inherited defaults, before learning.
[Author launcher](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/launch_train_sim.py)

Use 20 gradient updates per newly admitted latent transition after warm-up.
The author's loop counts gradient updates separately from simulator controls;
`max_steps` is not an interaction budget. Discount across the actual number of
executed controls: gamma = 0.9998 per control and Gamma_t = gamma^m_t for a transition
of m_t controls, normally 8. A partial final chunk uses its actual m_t. The source
uses 0.999 on a 400-control task; 0.9998 over Jar's 2,000 controls retains approximately
the same end-of-horizon discount, about 0.67. This horizon adjustment is our explicit
choice, fixed before outcomes. The reward therefore favors earlier safe completion
as well as completion probability; report the undiscounted completion rate separately.
[Author environment setup](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_sim.py),
[training loop](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_utils_sim.py),
[critic target](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/jaxrl2/agents/pixel_sac/critic_updater.py)

Collect the first six training episodes, one per development task, with a fresh
32-dimensional standard Gaussian vector tiled across each chunk. This follows the
author's initial collection distribution and is distinct from native full noise.
If these produce no more than 500 transitions, collect another complete six-task
cycle with that same distribution, up to 12 warm-up episodes in total. If the
threshold is still not met, stop and report insufficient data. Warm-up is included
in the 24 training episodes. Do not add native historical rollouts or engineering
episodes to replay. Replay retains all valid training transitions without eviction;
capacity is at least 6,000. No retroactive gradient-update credit for warm-up.

After warm-up, collect three episodes concurrently with one actor snapshot, then
wait for all three to finish. Insert transitions in frozen job order, perform
20 updates per newly collected transition, and publish the next actor snapshot.
The three workers share one learner and replay buffer. Record actor version on
each transition. This changes the author's one-episode collection cadence and must
be reported. Do not add asynchronous workers, change update ratios, or vary collection
batch size midway through the pilot. Keep independent episode sampling streams.

Use one learner initialization, seed 6060. Save checkpoints after training episodes
12 and 24. Evaluate only the episode-24 checkpoint, sampling the frozen
actor with declared per-episode RNG streams as in the author's evaluation loop.
Do not switch to deterministic actions after seeing results or select the checkpoint
with the best observed return. The pilot tests this training run, not reproducibility
across learner seeds.

Before evaluation, export that published actor and release the GPU learner.
Serve only the frozen actor on CPU while four VLA/evaluator lanes use the four
GPUs. The actor-only service retains the author's module, weights, image transform
and stochastic per-query sampling rule. Workstation parity and latency checks
must pass before renting, and the evaluation mode is recorded in the stage A
freeze. No critic, optimizer or replay is needed for evaluation inference.

## Safe completion and potential shaping

The author loop assigns -1 at every unsuccessful query transition and 0 on success.
This per-transition time-to-go cost has no intermediate progress milestones. The
terminal 0/1 base reward plus shaping below replaces that step cost; it is not a
reproduction of the author's reward. Keep this adaptation explicit in the methods.

The base reward b_t is zero until the final transition. At that transition it is
1 if the goal is held for 10 controls, closure precedes or coincides with first
departure from support, and no raw monitor violation occurred. It is 0 for
every other outcome, including unsafe completion and safe unfinished episodes.
The earlier flat -1 unsafe penalty is removed. Monitor every control, including
controls between policy queries, and retain irreversible violation and sequence
failure flags. Counted violation is raw violation filtered by the benchmark's
contact gate, so raw is the stricter predicate. Preserve and report both original
labels. Do not infer safety or requested order from the last frame alone.

Give the learner the following reward at each query transition:

    r_t = b_t + 0.25 * (Gamma_t * Phi(x_next) - Phi(x_t))
    Gamma_t = 0.9998 ** m_t

x is the scorer's simulator state augmented with episode history, not an extra
actor or critic input. Set closed_once when the raw lid predicate is first true
and keep it true for the rest of the episode, including if closure flaps later.
Reset it at each episode; retain the unmodified current predicate in the record.
This matches the first-observed-closure rule and avoids injecting closure flicker
into the 0.20 milestone. Closure and first departure observed at the same control
are valid, regardless of update order in the recorder. Phi is a bounded potential
over valid task progress. Apply
these cases in priority order:

| Scorer state | Phi |
| --- | ---: |
| Any raw violation or carry-before-closure has occurred | 0 |
| closed_once and the native instantaneous goal predicate is satisfied | 1.00 |
| closed_once, jar currently grasped, and jar off its support | 0.75 |
| closed_once and jar currently grasped | 0.50 |
| closed_once | 0.20 |
| Otherwise | 0 |

Use the existing Jar closure, support and grasp predicates and the benchmark's
instantaneous goal test before its ten-control hold. Audit their exact source
locations and query-time alignment during implementation. No new threshold, pose
distance, contact bonus, motion bonus or learned reward model is introduced.
If those predicates cannot be exposed causally without changing native scoring or
physics, stop integration for review rather than substitute a hand-drawn goal region.

Set Phi(x_next) to zero on every valid terminal transition, including success and
timeout, overriding the table. Set the replay bootstrap mask to zero there too.
This timeout-terminal choice deliberately departs from the author loop, which
bootstraps on failed/time-limited episodes and terminates the target only on success.
Keep the finite-episode objective and zero terminal potential here; do not silently
switch to the continuing-task bootstrap. The actor/critic receive no privileged
clock or safety history, so remaining time and the scorer's history are partially
observed. This is a limitation of the chosen approximation, not a policy-invariance
or convergence guarantee for that learner.

Use actual m_t for shortened final chunks. Nonterminal transitions are eight controls,
so their discount is 0.9998^8, about 0.998401; shorter chunks occur only at a terminal.
For a complete trajectory, the discounted
shaping return then telescopes to -0.25 * Phi(x_0). Thus adding back that initial
constant recovers the discounted base return for that same episode. Verify this
identity numerically for every training episode and a representative set of synthetic
traces. The 0.25 scale and the potential values are declared engineering choices,
not literature-tuned constants. Keep them fixed throughout the pilot.

This is potential-based shaping in the sense of
[Ng, Harada and Russell](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).
The return identity depends on consistent discounting and terminal handling; it
does not guarantee convergence or faster learning with a visual SAC learner.
Progress toward closing, grasping and carrying provides intermediate feedback,
while repeated closure or holding a closed jar cannot earn an additional positive
episode return. Reward shaping still uses privileged training signals, so any later
optimizer comparison must have access to the same reward information and count its cost.

Run every valid episode until native goal termination or the full horizon, including
after a violation. Do not stop inactivity or censor unsafe goal completions. Failed
infrastructure attempts are not terminal training data. Log a separate episode safety
cost, 1 if any raw violation or requested-order failure occurred, otherwise 0.
Do not add a cost critic, Lagrange multiplier or action veto in this first pilot.
Advancement still requires completion gains, engagement and no observed rise in
violations. The base objective assigns zero to both a tip and a safe stall; it
does not itself penalize a rise in violations during exploration. The outcome screen
is a separate acceptance criterion. A null result at this small budget does not
refute DSRL or shaping.

## Stage A Integration and profiling

Finish workstation implementation before renting: an explicit-noise server interface,
latent transition recorder, reward and termination adapter, learner bridge, deterministic
job scheduler, resource ledger and resumable preservation. Put reusable code in
`experiments/helpers/`, commands in `experiments/scripts/`, tests in `experiments/tests/`
and new Hydra configurations in `configs/<group>/`. Do not modify archived code.

Meaningful checks must cover native equivalence on identical saved inputs and RNG
state; injected-noise shape and decoder use; image/state normalization; reset isolation;
latent-action attribution to all eight executed controls; partial chunks; terminal
masks; sequence order; raw violations before contact; potential priority and terminal
correction; the discounted return identity; failed-attempt accounting;
checkpoint/replay resume; and deterministic actor publication.
Compare the reused learner's update on a fixed batch with its pinned author code.
Synthetic transition checks establish plumbing, not robotics effectiveness.

Engineering has six planned simulator starts. All tasks below are in the diagnostic
pool. The last two jobs measure startup after the cache has been populated.

| Job | Task | Policy seed | GPU | Mode |
| --- | ---: | ---: | ---: | --- |
| E0 | 1 | 60000 | 0 | Native |
| E1 | 4 | 60001 | 1 | Native |
| E2 | 10 | 60002 | 2 | Native |
| E3 | 11 | 60003 | 3 | Native |
| E4 | 1 | 60004 | 0 | Bounded random noise, after the authority gate |
| E5 | 4 | 60005 | 1 | Untrained noise actor, after the authority gate |

Start E3 first, then admit E0 to E2 after E3's first policy response. Once E3 ends,
release its evaluator and policy server and use GPU 3 for the engineering learner
checks while E0 to E2 run. All startup and waiting time remains charged. Record
the actual overlap and headroom; inability to validate the topology within the
engineering allowance stops learning admission rather than adding episodes.

E0 to E3 provide native pool observations for the authority gate below. E4 and E5
run only if it passes; they verify live explicit-noise attribution and cached startup.
Verify rendering on every GPU and the live recorder. Profile three rollout workers
coexisting with the learner on the fourth GPU. Do not overlap the learner's initial
compilation with the four lanes' initial renderer/policy compilation. Use recorded
engineering observations for fixed-batch updates, throughput and checkpoint checks.
Charge every reference, warm-up, parity and profiling update to an engineering-update
ledger, with a maximum of 2,000 update calls. This is part of the 120,000 global update
cap, not an extra allowance. Discard engineering weights and replay before stage B.
These checks add no simulator episodes.

## Stage A Decoder authority gate

Use at most 128 saved-observation VLA queries, included in the global query cap.
The probe is frozen before E0 and tests the exact magnitude-1 tiled noise domain
that the actor will use. It does not test amplitude 2 or restart an archived audit.

Take two snapshots from each native episode E0 to E3. The first is its initial query.
The second is the first later query after observed closure while no raw/order failure
has occurred. If no such query exists, use the latest nonviolating query at or before
control 800, strictly later than the initial query. If an episode supplies no distinct
eligible second query, mark the panel insufficient and stop engineering without
substituting another episode. Snapshot selection never uses a candidate decoder output.
Freeze the eight query identifiers and lossless original camera/state/prompt hashes,
recorded native latent draws, and native decoded outputs before issuing probe queries.

For each snapshot make exactly sixteen decoder queries: two replays of its recorded
native latent draw, and fourteen tiled vectors. The latter are zero, +e7, -e7,
all ones, and both signs of five Walsh vectors r_j, with
r_j[k] = +1 when bit j of k is zero and -1 otherwise, j = 0 to 4 and k = 0 to 31.
Here e7 is the unit vector at zero-based latent coordinate 7; it is a nominated
direction, not an assumption of uncoupled gripper control. Each tiled vector is in
[-1, 1]^32 and is repeated across all sixteen latent positions. Hash and freeze the
explicit arrays on the workstation before renting. The generated arrays and the
gate specification are recorded in [06-dsrl-decoder-probe.json](receipts/06-dsrl-decoder-probe.json).

All sixteen native replays must match their factual native decoded chunks within
absolute and relative tolerance 1e-5 and have exactly matching converted prefix
gripper commands. For authority, inspect only the first eight
decoded actions after native output transforms. Apply the pinned evaluator's gripper
conversion: open if g > 0.01, otherwise close, followed by native action clipping.
An observation is sensitive only if two of its fourteen bounded candidates give
opposite converted commands at the same prefix position, with g_open >= 0.02 and
g_close <= 0.00. Those margins avoid passing on a tiny threshold crossing.

The minimum is **two sensitive snapshots out of eight, from at least two distinct
pool tasks**, plus all identity checks. Record per-snapshot prefix flips, raw gripper
ranges and margins, candidate/observation hashes and arm-target RMS changes. Report
changes against zero and the factual native output separately. This minimum is a
declared engineering criterion, not a calibrated effect-size or significance threshold.

If authority is absent, below the minimum, or invalid within 128 queries, stop after
engineering and preserve that result. Do not admit E4/E5, development learning or
evaluation, enlarge the bound, add directions or replace snapshots. Passing shows
decoder influence, not improved grasping or safe completion. Requiring gripper
authority is a chosen prerequisite for this grasp/continuation pilot, not a claim
that every useful arm-steering method must flip the gripper or that the probe exhausts
the latent space.

## Stage A Admission to learning

Measure peak GPU and host memory, reset and cold compilation time, controls and
queries per second, transition transfer time, gradient-update throughput, preservation
size and bandwidth. Finish engineering within 90 minutes of creation of the host
that actually runs E0 to E5. Earlier failed-provisioning hosts still consume the
six-hour aggregate rental limit and all money allowances; they do not consume this
working host's engineering clock or reset any global counter.
The stage A receipt must project the complete scheduled stage B within the remaining
money and time, including learning updates, process starts and preservation. More GPUs do not
make a serial collection loop parallel. If the three-worker design cannot be validated
or budgeted, stop after engineering and revise the plan before development learning.

## Stage B Development pilot

Keep [panels.json](panels.json) unchanged. Training and evaluation use development
tasks 0, 3, 6, 9, 15 and 17. No new held-out episodes. Those held-out tasks already
appear in baseline and diagnostic evidence; eventual claims must acknowledge that.

Training uses each development task at seeds 61000 through 61003: 24 episodes,
including warm-up, arranged in eight three-episode rounds. Cycle through the six tasks
in their listed order before advancing the seed. Record simulator reset seed,
policy RNG seed and actor RNG stream separately; derive them deterministically
from the frozen job identity rather than relying on a shared server RNG.

Evaluation uses seeds 62000 and 62001 on all six development tasks, twelve episodes
per arm and 48 in total:

| Arm | Noise supplied to the frozen policy |
| --- | --- |
| Native | Original full-dimensional Gaussian sampling, with no explicit-noise override |
| Fixed zero | One zero vector tiled across the 16 latent time positions |
| Random bounded noise | Fresh independent Uniform[-1, 1] coordinates, tiled across time |
| DSRL | A sample from the final frozen actor at each query, tiled in the same way |

The random arm tests untrained use of the same bounded latent space; it is not
parameter random search. Fixed zero is a diagnostic fixed setting, not a tuned
best fixed baseline. Both stronger baselines remain necessary before an evolutionary
claim. A second learner with sparse reward is deferred beyond the $15 allocation.
Logging the sparse reward on shaped-training trajectories is not a reward ablation.
No tuning budget is hidden in these controls.

After training, release the learner allocation and evaluate on all four GPUs.
Each task/seed block contains all four arms. For block b and arm a, use lane
(a + b) modulo 4, so each arm receives three episodes on each GPU. Freeze the complete
job list and RNG derivation before any stage B episode. Evaluation data never enter
replay, and no reward-driven repeats or checkpoint substitutions are allowed.

## Accounting and compute

Target one verified Vast host with 4 RTX 4090 GPUs, at least 48 allocated CPU
threads, at least 256 GB host RAM, 200 GB usable disk allocation and driver 580.x.
Verify actual devices and render-library ownership through SSH. Do not infer adequate
CPU performance from thread count alone. Retain the legacy render profile and the
separate interpreter setup. Read the archived environment notes before reusing helpers.

The 24 October offer snapshot was approximately $1.90 to $2.14/hour with 200 GB
storage. A suitable example was offer 52375914, machine 151081, four 4090s, about
256 GB RAM, 80 allocated threads and driver 580.173.02 at $2.135556/hour. This is
a price reference, not a reservation. Recheck the offer, disk and bandwidth charges
at launch. The rate ceiling is $2.20/hour including selected disk. Reject an offer
whose expected downloads, uploads and storage cannot fit the separate $2 reserve.

| Allocation | Planned starts | Maximum policy controls |
| --- | ---: | ---: |
| Engineering on pool tasks | 6 | 12,000 |
| Learning on development tasks, including warm-up | 24 | 48,000 |
| Four-arm development evaluation | 48 | 96,000 |
| Infrastructure replacement reserve across both stages | 2 | 4,000 |
| Total ceiling | 80 | 160,000 |

Allow at most one replacement per failed job, from the two-start global reserve.
An invalid attempt halts admissions until a recorded repair. Preserve and charge
partial attempts even when no result row exists. Do not replace a valid poor outcome.
Also cap explicit simulator warm-up at 800 controls, VLA model-query attempts at
21,000 including reseeding/startup/probes, and learner updates at 120,000 overall.
Record engineering updates separately (at most 2,000); post-warm-up training can
use at most 90,000 under this episode allocation. The 120,000 ceiling does not permit
extra training, engineering checks or a changed update ratio. Count actor
calls, critic updates, model requests and elapsed time separately. Track scene/reset
internal physics separately from policy and warm-up controls; do not label it zero
when the simulator does not expose a reliable counter.

The **$15 is a hard cap across all allocations**, not a target or a per-stage budget.
Allocate at most $13 to active compute plus selected disk and $2 to transfer and
other storage. All setup, compilation, failed provisioning and preservation time
counts. Maintain a conservative ledger when provider billing is delayed; a restart
or replacement rental does not reset money, time or interaction counters.

Rent for at most six aggregate hours, or until the $13 active-rental allowance is
exhausted, whichever comes first. For a constant rate r with C dollars already
charged to that allowance, remaining rental hours cannot exceed (13 - C) / r.
At the reference rate $2.135556, six hours is approximately $12.81, leaving the
$2 reserve. At the $2.20 ceiling the active allowance lasts about 5 hours 55 minutes.
Total projected charges must also remain below $15 when transfer/storage costs differ
from the reserve estimate. No automatic top-up or additional reward-training arm.

Before starting a phase, reserve enough time and money for its remaining work and
preservation. Reserve at least 20 minutes for the final local preservation step.
No new warm simulator starts within 45 minutes of the computed deadline, and no
first/cold attempt on a GPU within 60 minutes of it.
The first simulator attempt per GPU on each new host has a 40-minute wall timeout,
including scene construction, renderer compilation, first query and rollout. Every
later attempt on that GPU has 25 minutes. Do not hide startup time by starting the
clock at the first query, or retrospectively relabel a timeout as a valid result.
The 90-minute engineering limit and money/time watchdogs override episode limits.
Set the money watchdog using $12.90 of the $13 active allowance, keeping $0.10
for billing rounding and shutdown delay. At
$2.20/hour this schedules shutdown at about 5 hours 52 minutes, before the nominal
money limit. The provider watchdog stops compute at the earlier time or money
deadline. Copy records locally throughout execution, close and verify the
final local copy before stopping, and perform private upload verification locally
where possible. Destroy only after preservation is verified. Charge any disk retained
after stopping to the same reserve and record its cleanup deadline. If a verified
copy cannot be secured within the remaining allowance, stop admissions immediately,
preserve what is available and report the incomplete state; do not keep paid compute
running. Unexpected charges reduce the work that can be admitted, not the cap.

The implementation begins independent provider stop requests five minutes before
that calculated deadline to allow retries and provider acknowledgement. This
reduces available runtime; it does not increase any allowance. The earlier stop
time controls admission and the 20-minute preservation reserve.

The previous two-3090 pilot ran 36 episodes in 89 minutes. At that per-worker speed,
24 training episodes on three workers imply about 40 minutes of collection, and
48 evaluations on four workers about 60 minutes. These are rollout-only projections,
not measurements of the new host. Stage A must add update time, barriers, cached and
cold starts and data handling to establish feasibility. If the fixed allocation
does not fit, finish with engineering evidence and propose a separately reviewed
experiment. Do not silently shorten the scored panel or change the learner budget.

## Analysis and decision

Primary outcome, named `strict_safe_requested`: requested completion with no raw
monitor violation. Also report the existing `safe_requested` definition unchanged,
all four benchmark classes, raw and counted violations, unfinished episodes and engagement,
each with its denominator. Engagement remains monitor first contact, supplemented
by closure, grasp and lift counts so motion alone cannot stand in for task progress.
Reconcile every label against sidecars and distinguish unscored safety from safety.

Report training return and success by fixed six-task cycles, policy saturation,
latent distributions, Q and entropy diagnostics, memory, runtime and all interactions.
Report base reward, shaping reward, Phi, episode cost and the discounted-return
reconciliation separately. High shaped reward alone is not a completion gain.
For a complete episode, corrected discounted shaped return equals discounted base
return. More closure alone cannot increase that quantity. Intermediate rewards,
undiscounted per-step averages, potential occupancy or critic estimates may change
without a completion gain; label those quantities accurately.
Inspect every scored episode's event frames and recording validity, including safe
stalls. Save both camera videos and the observations needed to reproduce learner
inputs. Do not retrospectively retune from these diagnostics.

For this exploratory screen, advance to a separately planned replication only if
the final DSRL arm has at least two more primary successes than native out of twelve,
more primary successes than each fixed/random control, gains over native on at least
two tasks, no more raw or counted violations than native, and engagement at least
10/12 and no more than one episode below native. These are practical screening
thresholds, not a significance test or a claim that risk has not increased.

Otherwise record whether the result is invalid integration, incomplete budget,
valid but no observed gain, or inconclusive learning. Missing the screen does not
authorize more training, another seed, a reward change or a larger noise bound.
If training cannot reach episode 24, do not evaluate an earlier checkpoint as the
scheduled final result. Retain its training record as incomplete.

Show per-task counts and task/seed-block differences. Give nominal Wilson 95%
intervals with their independent-episode assumption. If using task-cluster bootstrap
intervals, use 10,000 resamples and seed 90606, report the six-cluster limitation,
and explicitly flag degenerate intervals. Zero violations in twelve episodes implies
only a rough 95% upper bound near 3/12, not zero risk. One learner seed and development
tasks cannot establish generalization or an optimizer advantage.

## Freeze and preservation

Before stage A, record the approved scope, exact engineering jobs, source revisions,
asset hashes, patches, tests, rental caps and ledger format. Before stage B, freeze
the resolved learner configuration, objective, full job list, RNG mapping, concurrency,
recording schema, analysis code and resource projection. No development outcomes
may inform that second freeze. This final planning document is not the machine-readable
run manifest; the latter is created and hashed during implementation.

Use new `outputs/dsrl06/` local paths, `progress/receipts/06-dsrl-*` receipts and fresh
private dataset prefixes such as `jar/dsrl06_<arm>/`, `recordings/dsrl06/` and
`receipts/dsrl06/`. Store checkpoints in a private Hugging Face model repository pinned
by revision. Preserve actor/critic/optimizer states, RNG state, replay, ledger and
configuration so a restart cannot reset budgets. Keep credentials out of artifacts.
Verify local and uploaded files by SHA-256, pin dataset revisions, and retain final
provider state and charges. Add the experiment's Odyssey entry only after it finishes.

The review record is [06-dsrl-plan-review.md](receipts/06-dsrl-plan-review.md).
Give the independent reviewer [06-review-prompt.md](06-review-prompt.md).
