# Main study: evolutionary adapters around frozen UNISafe block plucking

Working design, 1 November 2026. This is the Block Plucking companion to
[FirstExperiment.md](../SafeLeWM/FirstExperiment.md). It inherits that document's
shared question, adapter family, objective, optimizer, comparison arms, accounting
and reporting rules, and states only what is specific to this task. It authorises
no rental, installation or experiment.

> Can evolutionary search learn compact adapters that recover safe task completion
> around frozen world-model systems, reducing reliance on manually tuned behavioural rules?

## Why this is the main study

UNISafe block plucking supplies three things TwoRoom cannot:

1. **An intrinsic safety failure.** The task's own physical predicates define the
   violation: middle-block drop, top-block drop, top-block lateral displacement, and
   end-effector boundary exit. No virtual hazard is needed and nothing is shrunk.
2. **A needle that moves.** Native execution violates in about 6/64 reserved starts
   and the release-rule adapters in 4 to 9 of 64 (H11, T09). Feasibility-first
   selection and a risk limit therefore act on real events, not on a flat zero.
3. **Two hand-designed references.** The released UNISafe filter is a published,
   hand-tuned safety mechanism (uncertainty threshold 4.3, reachability threshold
   0.1). The task 9 release-delay rule with translation scaling is a hand-designed
   behavioural fix. Matching either without hand design is the contribution.

## The frozen system, as it exists

Everything below was established in October and is reused unchanged.

- **Task.** `Isaac-Takeoff-Franka-IK-Rel-v0`: a Franka arm extracts the middle block
  from a three-block stack. 300 controls at 20 Hz (15 s), physics 0.01 s with
  decimation 5. Action: 7 coordinates, 6 relative-IK arm deltas and 1 gripper,
  normalised to [-1, 1], scaled by 0.15 and clipped, then IK scale 0.5. A negative
  gripper coordinate closes, non-negative opens.
- **Success.** The released compound predicate including middle/base XY separation
  above 0.07 m. Task 9 added **sustained success**: the predicate holds for 40 further
  controls (2 s) with no violation. Scoring code: `experiments/helpers/unisafe09_outcomes.py`.
- **Frozen agent.** Dreamer world model (39.9 M parameters), task actor, five-member
  uncertainty ensemble (150.9 M) and reachability safety actor with two critics.
  Inputs: two 128 by 128 RGB cameras plus end-effector position and quaternion.
  Checkpoint mirror `Sachioster/unisafe-pretrained` revision 39bd3c84 (private).
- **Filter.** Replaces the arm command with the backup action when the uncertainty
  or reachability test fails; preserves the gripper proposal. It can run live or in
  shadow mode, where its decisions are recorded but not applied.
- **Stack.** Isaac Sim 6.2.0.0, Isaac Lab 3.0 at revision 6bfde876, Python 3.12,
  PyTorch 2.11 (CUDA 13). Setup, download, worker, accounting and preservation
  scripts exist under `experiments/scripts/*unisafe09*`. Last validated 25 October on
  one RTX 3090 (Vast), 586 valid episodes.
- **Throughput, measured on that host.** 0.104 s per control; 20.8 s median simulator
  start-up; 44 s per episode with one fresh process per episode. About 80 episodes
  per lane-hour.

What the record says about levers and failures:

| Controller (reserved starts, filter on) | Completion | Violations |
|---|---|---|
| Native actor | 2/64 (H11), 3/64 (T09) | 6/64 |
| Release-delay rule, 40 steps | 55/64 | about 6/64 |
| Evolved release rule, 15 steps | 54/64 | 8/64 |
| Random search over the same rule | 56/64 | about 6/64 |
| Delay 15 with translation 0.75 (T09) | 54/64 | 4/64 |

Native fails by opening the gripper early, about 2.15 s in, so the block settles short
of the extraction distance and the episode times out. Sensing the grasp mattered:
the history-only rule lost 29.7 completion points. Translation scaling 0.75 kept the
success predicate through the continuation in 44/64 starts against 22/64 (post hoc).
No speed-versus-completion dial was found over those knobs. Evolution tied with a
fixed rule, random search and policy gradient, so no evolutionary advantage exists yet,
and violations never fell.

## What this study changes

- **Generic adapter, no expert rule.** The release-delay family encoded the fix by
  hand. Here the adapter is the shared compact network from FirstExperiment.md with
  generic initialisation, no hold counters, no hand-chosen thresholds and no expert
  fallback. The rule family becomes a labelled behavioural reference.
- **Safety is in the objective.** Feasibility-first ranking on the episode-violation
  rate replaces the task-only utility used in H11. This is the first UNISafe study in
  which the learner is asked to reduce violations, not only to complete.
- **Monitor-only main condition.** The filter runs in shadow mode. The adapter acts
  on the actor's proposal and the executed command is what the adapter returns,
  clipped to the legal box. The physical predicates label outcomes and veto nothing.
  Shadow decisions are recorded for the secondary condition's interpretation.
- **Filter-on secondary condition.** The same frozen adapters are rerun with the
  filter live on the same cases. This is the shielded-deployment comparison and is
  reported separately, with intervention counts.

## Interface: bounded action residual

Target selection has no analogue in a 20 Hz velocity-command task, so Candidate B
from FirstExperiment.md is the interface:

$$
a_t=\operatorname{clip}_{[-1,1]^7}\bigl(a_t^{\mathrm{base}}+B\odot\tanh(u_t)\bigr),\qquad
B=(0.5,0.5,0.5,0.5,0.5,0.5,\;2.0).
$$

The arm bound 0.5 lets the adapter slow or redirect a full-speed proposal by half.
The gripper bound 2.0 lets it turn any release into a hold, because the environment
reads only the sign of that coordinate. These bounds are declared so that the known
lever is representable; if qualification shows the adapter cannot reproduce it, the
interface fails and a gain form is the declared fallback.

**Decision cadence.** One adapter decision per 5 controls (4 Hz), held constant in
between, giving 60 decisions per episode. Reward and cost accumulate over the held
block and the monitor checks every control. This matches the macro-step semantics in
FirstExperiment.md and the earlier finding that about 300 decisions per episode gave
ordinary learning nothing to work with.

**Inputs**, identical for every learner and for the untrained reference:

| Group | Dimensions |
|---|---|
| End-effector position and quaternion | 7 |
| Finger joint positions | 2 |
| The actor's current proposal | 7 |
| Last 3 executed commands | 21 |
| Last 3 finger-position readings | 6 |
| Episode time fraction | 1 |
| Total | 44 |

Hidden layer 8 tanh units, 7 tanh outputs: 423 parameters. Qualification may choose
hidden size 8 or 16 and history 3 or 5 under the declared allowance; nothing else.
No images, object state or outcome flags reach the adapter. Finger positions are the
same additional signal the H11 rules used; every arm receives it and the comparison
is labelled accordingly.

**Zero-residual parity.** With u set to zero the executed trajectory must reproduce
native execution bitwise under matched seeds. This is a qualification gate.

## Objective, optimizer and comparisons

As in FirstExperiment.md: task return is the sustained-safe-success indicator, risk is
the episode-violation indicator (any physical failure or boundary exit, including the
continuation window), and candidates are ranked feasibility-first at a fixed limit
epsilon. The primary endpoint is sustained safe completion on reserved starts.

Proposed main-comparison settings, to be frozen after qualification:

| Setting | Value |
|---|---|
| Episodes per candidate | 8 (rate increments of 12.5 points) |
| Risk limit epsilon | 0.125, at most one violation in eight |
| Population | 16 |
| Generations | 6 |
| Optimizer repeats | 3 |
| Arms | CMA-ES, parameter random search, PPO-Lagrangian |
| References | native actor; untrained adapter (3 draws); release rule d15_g075; native with filter on |

Common random numbers: all candidates in a generation share that generation's block
of 8 training starts; the policy seed equals the start seed, as in task 9. PPO uses a
Gaussian residual, a first-violation cost with discount one, and the same cadence.
Its sample budget is matched in episodes, which is small for RL, and the write-up
will say so.

## Panels and seeds

Seed ranges already consumed and excluded: 0 to 23, 100 to 107, 1000 to 1063,
2000 to 2031, 9,100,000 to 9,299,999.

| Panel | Starts | Use |
|---|---|---|
| Training | 3,100,000 to 3,100,047 | 6 blocks of 8, one block per generation |
| Selection | 3,200,000 to 3,200,015 | nominee re-evaluation on a common panel |
| Final | 3,300,000 to 3,300,063 | reserved; opened once, after all selections are locked |

Overlap of any start with the released checkpoint's training data is unknown, as
before, and is declared rather than claimed absent.

## Budget, and why throughput is the binding constraint

At 44 s per episode:

| Stage | Episodes | Lane-hours |
|---|---|---|
| Training, 3 arms by 3 repeats, 768 each | 6,912 | 84 |
| Selection panel, 9 nominees plus 4 references | 208 | 3 |
| Final, 13 controllers by 64 starts, filter off and on | 1,664 | 20 |
| Total | 8,784 | about 107 |

That is roughly $35 to $70 of GPU time at $0.33 to $0.65 per lane-hour, or about
27 wall-hours on four lanes. Two levers could cut it substantially and are measured
in qualification before anything is frozen:

1. **In-process resets.** Start-up is 21 of the 44 s. Task 9 chose a fresh process
   per episode for determinism; parity of in-process resets must be shown first.
2. **Vectorised environments.** Isaac Lab runs several environments per process and
   the Dreamer agent batches. If 4 to 8 environments pass outcome parity, a whole
   candidate's panel runs at once.

If neither lever qualifies, the frozen comparison shrinks to 2 repeats and the PPO arm
is deferred to a second allocation. The honours and GECCO timelines favour a smaller
comparison that finishes over a larger one that does not.

## Staged order

1. **Qualification** ([Qualification.md](Qualification.md)): rebuild the platform on a
   fresh host, reproduce the task 9 native outcome, prove zero-residual parity, measure
   adapter authority with generic genotypes, measure the two throughput levers, and
   anchor the references on the selection panel.
2. **Freeze** the actor size, cadence, epsilon, panels, budgets, selection rule,
   analysis and stopping rules.
3. **Main comparison**, filter off, then the filter-on rerun of the frozen controllers.
4. **Focused ablations** on the selected arm: trained versus untrained; feasibility
   versus task-only ranking; without finger positions.
5. **Dial study**, separately frozen: a small library at requested limits such as
   0, 0.125 and 0.25, evaluated for actual violations, completion and engagement.

## Decisions for Sachin

1. Primary endpoint: sustained safe success (recommended) or the benchmark predicate.
2. Main condition with the filter in shadow mode (recommended) or live.
3. Adapter cadence of 5 controls (recommended) or every control.
4. Keep PPO-Lagrangian in the first freeze, or defer it until CMA-ES versus random
   search has run.
5. Host form: single 24 GB lanes as in task 9, or one multi-GPU host with the serial
   warm-up and launch lock that task 11 showed are needed.
6. Authorise the qualification rental in [Qualification.md](Qualification.md).
