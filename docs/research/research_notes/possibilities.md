# Possibilities after todo 2

Brainstorm of 23 October 2026, after the todo 2 description
([Where the leverage is](Where%20the%20leverage%20is.md)). Everything we could build is
listed first; the ranking that matters is the last section: cheapest thing that, if it
works, is a GECCO contribution, and still leaves something to say if it does not.

## What todo 2 fixed as facts

- The policy approaches the same way every time; same-seed repeats are identical until
  the fingers meet the lid. Outcomes are decided after contact, in the lid push and the
  regrasp descent. 18 of 26 tasks are knife-edge, 8 are systematic.
- Of 104 episodes, 36 end idle with most of the horizon left (18 at the open lid, 18 with
  the lid closed and the robot parked), 19 tip the jar after closing it, 9 before, 3 lift
  it open, 5 fail in the carry, 24 succeed safely, 7 succeed with a violation.
- An episode is a draw from a per-task rate. Seeing a ten-point change in safe success
  needs about 200 episodes per condition. Episode seeds reproduce the layout exactly, so
  common random numbers across conditions are free.

## Where in the loop we can act

1. **What the policy sees.** Image perturbation (crop, shift, brightness, camera jitter);
   a kick to the observation only when the policy has frozen, to break a stall.
2. **Inside the policy.** The flow-matching sampler: noise scale, initial noise vector,
   number of denoising steps; best-of-N noise per chunk; the prompt text; LoRA weights
   (the only item that touches parameters).
3. **How the chunk is executed.** Actions executed per 16-step chunk; delta scale;
   low-pass filtering between chunks; speed cap near the table; gripper closing rate; a
   downward-velocity clip when a finger is low. Each can be constant, time-gated or
   state-gated.
4. **A supervisor above the policy.** Patience and retry (nothing has moved for N steps:
   lift to a safe pose, hand back); retreat on a danger cue; best-of-N chunk selection
   with a scorer; restart from home when stuck.

## The evolutionary toolbox, matched to the knobs

- **Random search and grid.** The baseline every claim must beat; for three knobs a
  serious competitor.
- **(1+lambda)-ES with resampling.** Simplest noisy-fitness optimiser; children evaluated
  on the parent's episode seeds, so comparisons are paired.
- **CMA-ES.** Default for 3 to 20 continuous knobs; the population mean as the robust
  answer.
- **Plain GA with tournament selection** for mixed knobs (horizon in {4, 8, 16}, scales,
  a patience threshold in steps).
- **NSGA-II, two objectives.** Safe success against completion time, or violations
  against time-outs. The output is a Pareto front, and a front is a safety dial: the
  operating point is chosen at deployment.
- **MAP-Elites (quality diversity).** Descriptors such as descent depth, time to closure,
  approach speed; a shelf of distinct robust strategies rather than one.
- **Neuroevolution of a tiny controller.** A few-weight MLP from cheap, non-privileged
  features (gripper opening, whether the hand has stopped descending, time since then,
  joint velocities) to the execution knobs: the state-dependent dial, the compact
  evolved controller of the original goal.
- **ES over LoRA deltas.** Black-box fine-tuning of the policy; thousands of episodes;
  invites the question "why not gradients".
- **Noisy-fitness machinery for any of these.** Racing or successive halving; early stop
  at the first violation; short screening horizon with full-horizon confirmation;
  rank-based selection; common random numbers.

## Adapters and noise steering

- **Residual action adapter.** Small network adding a delta to each chunk, trained by ES
  with safe success as fitness; compact, no gradients through the VLA.
- **Gating adapter.** Outputs a scalar speed or depth factor; smaller search space, same
  mechanism as the dial, learns where to apply it.
- **Behaviour-cloned LoRA on our own successes.** 24 safe successes is too few; the
  recorder makes collecting more possible; a training project.
- **Prompt evolution.** GA over the instruction text; nearly free; a fine-tuned
  checkpoint probably does not care, one racing screen finds out.
- **Sampling-noise scale sweep.** The plan's original stage 2; todo 2 predicts it flat;
  as a control it lets the paper say the lever is in execution, not in the sampler.
- **Noise-vector evolution.** The archived latent-evolution pilot retargeted at the
  post-contact window; expensive; its earlier failure is partly explained by aiming at
  the approach.
- **Guided sampling with a safety critic** inside the flow; needs the policy's internals
  and a critic that reads rim-versus-body from a 256-pixel image.

## First ranking: expected gain per dollar (superseded)

1. Patience and retry (36 idle episodes; a wrapper and one scripted motion).
2. Post-contact execution dial (horizon and delta scale, gated by a non-privileged cue).
3. Noise-scale sweep as the control, sharing the native arm.
4. NSGA-II over the combined knobs against random search at matched budget.
5. Neuroevolved controller and residual adapter, only if a fixed dial shows a lever.

Superseded because it ranks engineering above contribution: a retry rule on its own is
good engineering and zero evolutionary content.

## Ranking that matters: low-hanging fruit that is a contribution

1. **An evolved supervisor around the frozen VLA, with the Pareto front as the result.**
   Four to six knobs in one wrapper (execution horizon, delta scale after contact, a
   descent cap near the table, a patience threshold for retry, a retreat height), evolved
   with NSGA-II on two objectives. The front is the Safety Dial. Baselines: native,
   hand-tuned, random search at the same episode budget. Claim: multi-objective evolution
   exposes a safety-speed trade-off a frozen policy hides, more cheaply than random
   search. Nothing is trained; the wrapper is a few hundred lines; todo 2 says which
   phase to aim at. About 1,000 to 1,500 episodes, $35 to $50 on the current host. The
   retry idea lives inside it as a knob.
2. **The noisy-evaluation methodology, mostly already in hand.** The noise floor, the
   funnel, same-seed disagreement, per-task rates, common random numbers, racing, early
   stop. Not a paper alone; the method section that makes item 1 credible and explains
   why the earlier pilots found nothing. Zero new episodes.
3. **A neuroevolved state-dependent dial.** If item 1 shows a lever, replace the fixed
   knobs with a tiny network from non-privileged features to the same knobs (CMA-ES or
   NEAT); claim: a compact evolved controller dominates the fixed-knob front. The
   original goal; only meaningful after item 1.
4. **Quality diversity over regrasp strategies.** MAP-Elites with recorder descriptors;
   very GECCO; weaker safety claim; stretch goal.
5. **Prompt evolution.** A cheap racing screen on the side, not the plan.
6. **Noise and latent steering.** A control arm, not a contribution: one sampling-noise
   sweep sharing the native arm. The archived direction stays archived.
7. **LoRA by evolution strategies.** Not for this venue or budget.

## The next experiment this implies

The first step is the same whichever item leads: build the action hook and the
supervisor wrapper with its knobs, prove the native settings reproduce native behaviour
exactly (the identity check), then run the stage 2 gate as a four-arm test on the
development panel (native, slow regrasp, retry on, sampling-noise scale) at about 200
episodes each, about $25. It says whether any knob moves anything, and every episode is
reusable as the native arm and the seed population of the NSGA-II run.

Decide now, because it shapes the framing: the oracle question. The first lever test may
gate on the simulator's closure signal as an upper bound; the deployable version must
gate on something the robot can measure. Running oracle-gated and proxy-gated side by
side makes the gap between them a finding.

Infrastructure facts: the todo 2 recorder is observe-only, so a dial needs an action
hook in the evaluator's execution loop (the archived pilot's noise hook is a starting
point); the recorder itself is still to be validated on one real episode, as the first
hour of the next GPU session. Host: Vast instance H, prepared 23 October 2026.
