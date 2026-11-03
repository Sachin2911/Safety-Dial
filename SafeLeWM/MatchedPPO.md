# Matched-objective PPO comparison

Prospective protocol, 1 November 2026. Sachin requested the RL comparison alongside
rather than instead of the main mechanism study, chose matched-objective PPO, and
asked about sharing the additional RTX 4090 and now-free L4. Use two complete
matched repeats on RunPod pwjry7gqqs278s (4090, $0.89/hour) and one on the existing
RunPod yvrg8ow7i9n9u6 (L4). Both rentals remain on until he instructs otherwise.
No changes to the main eight-GPU study or its limits.

## Question and controls

Does ordinary policy-gradient learning recover safe completion with the same
adapter information and interaction allowance as evolutionary and random search?
Use the Stage A representation winner from AdapterMechanism.md, locked before any
RL comparison training. Reinitialize all methods; do not reuse winning weights.
A complete repeat contains PPO, CMA-ES and parameter random search on the same
physical host. Hardware is blocked within repeat, not assumed interchangeable.
Native is evaluated once per host; each repeat also has an untrained reference.
Report all methods separately per repeat and host before pooling paired contrasts.

The qualified frozen AP/LeWM and memory, 396 actor inputs, GRU32, gain/residual
output, passive hazard monitor, horizon200, eight training layouts and training
planner seed8201 remain unchanged. Final cases are the 24 frozen Stage B layouts
and planner seeds11401/11402/11403. This plan is fixed before their outcomes.
No method or hyperparameter selection may use final outcomes from either study.

Three new repeats R0/R1 on the4090 and R2 on theL4. Seeds use bases12100/12200/12300:
initialization+1, untrained+2, CMA+3, random+4, projection+5, PPO sampling+6 and PPO
minibatch shuffle+7. Share neutral weights, coordinate scales and projections within
a repeat. Orders: PPO/CMA/random, CMA/random/PPO, random/PPO/CMA. Initial sigma0.5.
CMA and random retain the qualified exact-quota implementation and nominee rule.

## PPO objective and implementation

Each completed trajectory receives terminal utility
(81*safe_completion + 9*completion - violation)/90, with zero intermediate utility.
These constants encode the existing eight-case search ranking, not a fitted
penalty. PPO optimizes its expected episodic utility; this is not an unrestricted
lexicographic probability guarantee or a risk-budget constraint. No distance,
progress, clearance, witness guidance or hand-tuned dense reward is supplied.
Original environment rewards and all safety labels remain separately recorded.

PPO's deterministic actor is identical to the selected adapter representation.
Optimize standardized full coordinates, the132 head coordinates, or256 coefficients
of the same row-normalized random projection. Start at neutral coordinates zero.
During collection sample a diagonal Gaussian in the four raw output logits, then
apply the existing tanh gain/residual transformation and final action clipping.
Store raw sampled logits and score their Gaussian log probability, never the
clipped executed action. Evaluate the mean deterministically. PPO learns four
additional global log standard deviations, initially log(0.5), clamped to[-3,1].
They are training-only exploration parameters and are reported as extra parameters.
CMA/random sample parameter vectors while PPO samples actions; this is a declared
algorithmic difference, not identical exploration.

A separate critic receives the same396 observations plus the actor's32-dimensional
hidden state, detached from actor gradients. It is a428-to64-tanh-to1 network;
no privileged state, goal coordinates, cost-to-go or oracle flags. Its parameters
are training-only overhead. PPO hyperparameters are frozen without a sweep:
Adam learning rate3e-4, ratio clip0.2, four epochs, minibatches of four complete
episodes, gradient norm cap0.5, value-loss coefficient0.5, entropy coefficient0,
gamma1, GAE lambda0.95. Normalize advantages over each rollout batch. Recompute
whole-episode recurrent states under current parameters at every update; do not
shuffle individual timesteps or reuse stale recurrent states as episode starts.

Collect up to3200 actual actions per update, cycling through training cases in
fixed order with independent policy-sampling streams. Full episodes end on success
or200 actions, with terminal utility. A quota-cut unfinished trajectory is saved,
charged and excluded from PPO updates, just as incomplete search panels cannot
nominate an ES candidate. No bootstrap or invented terminal reward at a quota cut.
After each update evaluate the current deterministic mean on the eight training
cases, charged to the same40000-action allowance. Only complete eight-case panels
are eligible nominees; maximize the original safety-aware score with earliest tie.
If the remaining allowance is too short, retain and charge the partial panel.
Collection, policy selection and any discarded partial data all count. No extra
unbudgeted validation rollouts. Preserve every update and deterministic checkpoint.

## Limits and qualification

Exactly40000 actual training/selection actions per method per repeat,360000 total.
Final evaluation: native72 on each host, untrained72 for each repeat, and nine
trained nominees72 each,1008 episodes and at most201600 actions. Each host reserves
10000 engineering actions plus200 failed-start actions. Total cap582000 actions:
379800 on the4090 and202200 on theL4. Twenty-hour execution deadline per host stops
the job only, never the rental. Setup, saved-evidence audits and transfers are
separate. Stop on qualification or accounting failure; do not silently retune.

Before training verify immutable assets, preprocessing and environment source,
neutral/native action parity, stochastic log-probability replay, PPO recurrent
replay and clipped-action likelihood handling, gradient flow for all three
representations, terminal/partial return handling, and actual-budget accounting.
Engineering episodes use training cases only, are separately logged and capped.
No held-out environment is used for software qualification. Bind host, packages,
GPU and source hashes before the first episode. Save both original and any failed
attempt, including attempted actions and model calls.

## Evaluation and reporting

Lock all three nominees within a repeat before that repeat's final outcomes.
The selected representation and algorithm settings remain fixed across both hosts.
Report safe completion, completion, violations, unfinished and engagement counts
with denominators, plus actual interactions, model queries, optimizer updates,
complete candidate panels, PPO collection/selection split and wall time. Preserve
all per-episode traces, checkpoints, resource footers and source freezes.

Primary exploratory contrasts: CMA-minus-PPO and CMA-minus-random safe completion,
paired within each host/repeat and case. Use2000 crossed repeat/case bootstrap draws,
seed12501, keeping planner seeds together. Three repeats are not enough for a
precise hardware-adjusted generalization claim. Report intervals and individual
repeat results, including failures. No claim of equivalence from a nonsignificant
contrast. No additional PPO sweep, PPO-Lagrangian, penalty tuning or dial experiment
is authorized by completion of this run.
