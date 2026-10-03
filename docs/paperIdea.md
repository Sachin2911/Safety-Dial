# Paper idea: evolution cheats imagination

**Status: current direction, chosen 3 October 2026, to be confirmed with the supervisor.**
It replaces the [LeWM experience study](researchDirection.md), completed through S4, as
the focus of new work, and answers the [allocated topic](allocatedTopic.md). Target venue:
[GECCO 2027](gecco2027.md). Background research: the
[deep-research report](research/reports/Evolutionary%20safe%20RL%20with%20world%20models.md)
([PDF](research/reports/Evolutionary%20safe%20RL%20with%20world%20models.pdf)).

## In one sentence

If you train a robot to be safe inside its own imagination, it learns to cheat the
imagination, and we show how to stop it.

## The story

1. **Practise in your head.** Trying thousands of robot brains in the real world is slow,
   and they fall over. Trying them inside the world model is cheap, and nobody gets hurt.
2. **But the imagination is sometimes wrong about danger.** Evolution keeps whatever scores
   best, and what scores best is often a brain that found a spot where the model wrongly
   says "safe". It is like a student who passes by finding a bug in the auto-grader
   instead of learning the material.
3. **The fix: make the imagination shaky where it is unsure.** Add noise there, and the
   cheating spots stop being reliable, because sometimes you fall. ROSARL's penalty makes
   falling cost enough that the robot stays away, with no hand-tuned number.
4. **The check: catch every cheat.** For each brain evolution picks, replay the exact same
   moment in the real simulator, and put imagined safety next to real safety.

**What we would show:** the harder evolution searches, the more it cheats, and noise plus
ROSARL bends that curve back down.

Feynman: "The first principle is that you must not fool yourself, and you are the easiest
person to fool." Evolution inside a world model is a machine for fooling itself. This paper
measures it and fixes it.

## Why we expect cheating

- Selecting the best of many candidates under an imperfect evaluator favours the candidates
  the evaluator is most wrong about. More candidates and generations mean more cheating.
- Ha and Schmidhuber's controller evolved in a learned "dream" scored 2,086 there and 193
  in the real game, until they added noise to the dream.
- Our own Walker LeWM: on the S4 representative test with no update, 225 of 504 futures it
  accepted as safe under the health rule were actually unsafe (about 45%).

## How noise and ROSARL fit together

- ROSARL's penalty is one number for the whole task (in the RLC 2026 version, the gap
  between the lowest and highest estimated values). The local caution comes from the
  noise: where the model is unsure, noisy rollouts reach unsafe states more often, so those
  regions look risky.
- Each candidate is scored on many start states (and, with noise, several noisy rollouts
  of each), so its fitness is roughly (1 − p) × (return when safe) + p × (penalty), where
  p is its violation rate. The penalty's size decides how much return is traded for lower
  risk, and ROSARL is the rule for choosing that size without hand-tuning. (With a single
  rollout per candidate, any large enough penalty would reduce to Deb's feasibility rule;
  averaging is what makes the size matter.)
- Noise raises p exactly where the model is unsure, so the same penalty now pushes
  evolution away from behaviour the model cannot vouch for.

## Closest existing work

- **MOReL (2020):** sends uncertain regions to a "HALT" failure state with a hand-picked
  penalty. ROSARL could replace that hand-picked number.
- **CAP (AAAI 2022):** inflates predicted cost by model uncertainty, with a coefficient
  tuned from real feedback.
- **Dyna-SAuR (April 2026 preprint):** a safety filter that avoids high-uncertainty
  regions, tested on Walker. Gradient-based; no evolution and no ROSARL.
- **GuSS (preprint):** MAP-Elites as a cost-first planner in a learned model; lists model
  uncertainty as future work.
- **Safe CEM is not new:** its feasibility-first ranking is Wen and Topcu's constrained
  cross-entropy method (NeurIPS 2018) and SafeDreamer's planner (ICLR 2024). Use it as a
  credited baseline, not a contribution.

No paper combining noise injection, ROSARL's penalty and evolution was found. That is a
search result, not proof; the checks in the report still apply.

## The experiment

The step-by-step implementation (code, tests, gates, dates and run sizes) is in the
[implementation plan](evoPlan/README.md).

### Setup

Walker only, using what the [LeWM experience study](mainPlan/README.md) already built. No
new model training.

- **Environment:** `SafetyWalker2dVelocity-v1`, with the health rule (torso height and
  pitch) as the safety constraint.
- **World model:** the Walker LeWM trained from scratch (214,780 updates), with its
  physical probes.
- **Policy:** a linear map from the LeWM latent to Walker's six motor commands, about
  1,200 numbers.
- **Evolution:** CMA-ES.
- **Scoring in imagination:** each policy runs in short 0.8 s segments (10 model blocks)
  starting from real saved moments. The probes read forward speed (reward) and torso
  height and pitch (safety).
- **Ground truth:** every policy evolution selects is replayed from the same MuJoCo
  snapshots in the real simulator.

### Steps

1. **Build the harness,** on the local machine, before renting anything. `WalkerImaginer`
   only replays fixed action tapes, so it needs a closed-loop rollout where the policy
   reacts to each imagined latent. Also needed: real-simulator replay of a latent policy
   (render, encode, act), and noise calibrated from the gap between predicted and real
   encoded latents.
2. **Transfer check (go/no-go).** Score about 100 policies in imagination and in reality
   and compare the rankings. If they roughly agree, continue. If not, the centre of the
   work becomes evolution in the real simulator against PPO-Lagrangian.
3. **The cheating curve.** No noise. Increase population size and generations, and track
   imagined against real violation rates of the selected policies.
4. **The fix grid.** 5 noise levels × 3 ranking rules (ROSARL's penalty, Deb's
   feasibility rule, a fixed penalty) × 10 seeds. Measure real violation rate and real
   return.

### What success looks like

- Step 3 shows a clear cheating curve: the gap between imagined and real violations grows
  with search pressure.
- In step 4, noise plus ROSARL shrinks that gap more than the other two rules, without
  wrecking return.

A missing curve or a failed fix is still a result, but a weaker paper.

### What we measure

- Imagined and real violation rates of selected policies, paired on identical snapshots.
- Real return.
- Rank agreement between imagined and real scores, by generation.
- Real simulator steps (including the data the LeWM was trained on), imagined steps and
  wall-clock time, reported separately.
- Violation rates with intervals. Zero observed violations in n episodes only bounds the
  rate at about 3/n.

### Cost and timing

- **Compute:** roughly half a day to a day on one RTX 5090, about $10 to $25, rented only
  when a stage is ready to run. This is an estimate; a one-hour benchmark on the rented
  machine comes first.
- **17 October 2026:** harness and transfer check done (gate).
- **8 November 2026:** core results frozen; thesis due late November.
- **December to January:** `SafetyHopperVelocity` as a second environment, a LeWM
  ensemble only if simple noise is not enough, then the GECCO paper (deadline expected
  late January 2027, unconfirmed).

## Known catches

- Noise covers random model error. The model's errors are partly systematic bias, which
  noise does not fix.
- The Walker LeWM was trained mostly on competent walking, so it has seen few falls, and
  evolution will push into exactly those states.
- Too much noise means over-caution. In the RLC 2026 version, ROSARL's PointGoal1 return
  fell from 10.17 to 1.87 while cost fell from 0.62 (TRPO-Lagrangian) to 0.08.

## To raise with the supervisor

- Does this fit ROSARL's intent, and which version should be cited? The evidence points
  to RLC 2026 ("An Unreasonably Simple Approach to Safe RL"), not RLC 2024.
- When the supervisor said ROSARL handles noise well, was that noisy dynamics or noisy
  evaluations?
- Agreement to make this the main direction of the thesis and the GECCO paper.
