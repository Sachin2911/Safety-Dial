# ManiGuard phase three: penalty-free evolution of a safety-dial execution layer

Drafted 18 October 2026 on the 8×RTX 3090 instance, after the frozen phase-two pilot was
paused at a block boundary by Sachin's decision. This note records why the frozen pilot
cannot produce a convincing evolutionary result, what the investigation found, the chosen
direction for a GECCO 2027 full paper (deadline about 26 January 2027, in-person Kraków),
the gates that must pass before the main study, and the budget on this machine only. It is a
plan for approval, not a result. Nothing in it changes the frozen phase-two protocol.

## 1. What the pilot measured

- **Replicate noise dominates one-shot fitness.** Round 0 of each search method evaluates the
  same eight genomes on the same scenes and seeds, so random/s17 and evolution/s17 panel 0 form
  a built-in replicate. Of 22 paired Clutter and Jar episodes completed so far, 9 disagree in
  outcome (safe completion, violation or unfinished). Replays start from an identical robot
  state but the first rendered frame already differs in 40 to 70 percent of pixels by one or
  two levels, and the vision-language-action policy diverges from its first action. A
  candidate therefore has an outcome probability per case, which the frozen protocol estimates
  from one episode per family per panel.
- **Family signal.** Clutter sits near 50 percent safe completion and carries nearly all fitness
  differences. Jar is low but not zero on this host: 2 safe completions in 22 (both evolution
  candidates 3 and 4, whose identical random-search twins timed out). Stack is 0 in 25 here and
  2 in 8 on the old host, almost all timeouts at 4,200 actions.
- **Jar failure anatomy (22 traces).** The lid is closed in 9 of 22 episodes, usually by step
  300 to 350. The dominant losses are tipping the open jar while pushing it before any grasp
  (5 of 8 old-host references) and failing to grasp or lift after closing. The two unsafe
  "successes" on the old host lifted an open jar. These are execution-level failures that an
  execution layer can address; they are not purely plan-order errors.
- **Throughput here.** Median process time under eight-way load: Jar 626 s, Stack 1,161 s,
  Clutter 417 s, about 1,400 Clutter episodes per day on eight GPUs. The simulation is CPU
  bound on this host.

Implication: with one episode per candidate, ranking eight candidates is close to a lottery,
so evolution, random search and PPO tie by construction. The evaluation design, not the
evolutionary operators, is the bottleneck.

## 2. What the literature says (web-verified 18 October 2026)

- No published work evolves a history-dependent, monitor-aware execution layer around a frozen
  VLA policy with random-search and RL controls. The nearest are VLA-Pilot (per-step
  evolutionary refinement of action proposals, about +0.11 over best-of-N, no RL baseline) and
  DSRL (SAC in the noise space of a frozen diffusion policy), which is the RL wrapper baseline
  reviewers will expect. ES at scale (ICML 2026) reports rank-based ES beating PPO on sparse
  outcome rewards with much lower seed variance, which matches our ternary terminal outcome.
- Noise handling in EAs is mature: rank-based selection needs ordering only; prefer larger
  populations over resampling; do not re-evaluate parents; racing or adaptive re-evaluation
  for final comparisons; Extract-QD style re-evaluation of elites (GECCO 2025).
- Penalty-free constraint handling is the supervisor's brief: Deb's feasibility rules,
  stochastic ranking, FI-2Pop and multi-objective fronts (NSGA-II, MOME) replace a tuned
  penalty such as the pilot's U = S − 2V.
- Random search ties when the effective dimension is low and fitness is not heritable under
  the operators. GECCO reviewers expect ten or more seeds, matched budgets, random-search and
  no-adaptation nulls, ablations and corrected effect sizes.

## 3. Chosen direction

**Claim to test:** a penalty-free, noise-aware evolutionary method finds an interpretable
execution layer around a frozen VLA that raises verified safe completion without raising
violations, and does so more reliably than random search at the same episode budget, than the
same ES with the scalar penalty, and than ordinary policy-gradient learning with the same
information.

**Search space: the safety dial.** Ten to sixteen bounded parameters with physical meaning,
driven only by online-legal signals (joints, gripper command history, end-effector position,
tracking error, proposal deltas, time since query): joint-delta caps and smoothing per gripper
state, an extra low-height cap while the gripper is open (anti-tip), a hold window after a
close command with lift attenuation, release delay, query interval, and stall-triggered
re-query thresholds. The identity genotype reproduces the frozen policy exactly. Proposed,
filtered and executed actions stay distinguishable in the trace.

**Optimizers compared at matched budgets:** (a) feasibility-ranked (μ,λ)-ES with no parent
re-evaluation, (b) the same ES ranked by the scalar penalty (ablation), (c) NSGA-II-lite on
(completion, violation) reporting hypervolume and the front, (d) random search with identical
evaluation and final selection, (e) the existing recurrent PPO over the discrete phase-two
choices as the ordinary-learning comparison, plus identity and the best hand rule.

**Evaluation.** Fitness per candidate from R replicate episodes on the training cases; final
comparison on held validation cases with racing-style re-evaluation of finalists; outcomes
reported together (safe completion, violations, unfinished), with stratified bootstrap
intervals and corrected nonparametric tests across ten seeds per method.

**Families.** Clutter for statistical power, Jar for the safety mechanism (anti-tip, lift
hold), Lid as design-exposed transfer. Cabinet and Dusty stay untouched. Stack is dropped
from training because it carries no completion signal at the frozen horizon.

**Second contribution, gated.** Genetic programming of sub-goal re-prompting programs
(close, then grasp, then carry) has the strongest evolutionary mechanism story but depends on
whether π0.5 follows sub-instructions, which a ten-episode probe decides before any build.

## 4. Gates before the main study

1. **Replicate diagnostic (running 18 October).** Eight round-0 genomes × three Clutter
   scenes × eight replicates. Decides the noise level, whether any between-controller
   difference exists, and the replicates needed per candidate.
2. **Heritability probe (next).** Sixteen uniform parents and one Gaussian mutant each in the
   dial space, four replicates on two Clutter scenes and two on one Jar scene, about 250
   episodes. Pass if parent-offspring safe-completion correlation is positive and candidate
   rates are heterogeneous beyond binomial noise. Fail means a different interface or family
   before any seed study.
3. **Sub-instruction probe (optional).** Ten Jar episodes prompted with the lid clause only.

## 5. Budget on this machine only

Per run: population 12, 15 generations, two scenes × two replicates, about 720 episodes.
Four evolutionary or random methods × ten seeds on Clutter plus Jar and PPO at the same
budget and finalist validation come to roughly 30,000 to 35,000 episodes, about 20 days of
wall time at 1,400 Clutter-equivalent episodes per day, about USD 1,100 of rental. The main
study must finish by mid-November to leave writing time for the thesis and the January
paper; the gates are three days. Every attempted episode counts; failures are recorded, never
retried, except zero-interaction setup failures under the standing rule.

## 6. What would make this fail honestly

If gate 2 shows no heritable structure, the EA cannot beat random search in this space and
the paper has no win; the fallback is the re-prompting program space or a family with a
moderate base rate. If violations trade against completion, the result is a front, not an
unqualified safety improvement, and must be reported as such. Zero observed violations in n
episodes bound the rate by about 3/n, not zero.
