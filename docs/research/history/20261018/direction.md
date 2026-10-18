# Direction: a convincing evolutionary result for GECCO 2027

**Original phase-three decision record, written 18 October 2026.**

[phase3.md](phase3.md) is now the active working plan, incorporating the subsequent
methodological review, corrected episode and compute estimates, and decisions still
needed before implementation and protocol freeze. The original reasoning, prescriptions
and estimates below are preserved as the decision record, not as the current executable
protocol. [oracle.md](oracle.md) remains the central authority; [phase2.md](phase2.md)
and its frozen evidence remain unchanged.

Sachin's target is a GECCO 2027 full paper (abstract about 19 January, full paper about
26 January 2027, in-person Kraków) with a convincing evolutionary result, not a
well-measured null. This document records why the frozen phase-two pilot cannot deliver
that, what a day of investigation found, the direction chosen on 18 October, the gates that
must pass before the main study, and the budget on the current machine only. The frozen
phase-two protocol is unchanged; phase three is a new protocol with its own budget.

## 1. Decisions taken on 18 October 2026

1. The frozen pilot was resumed on an 8×RTX 3090 Vast instance (56515754) with concurrent
   lane dispatch, then **paused at a block boundary** once evolution/s17 panel 0 and the
   reference block completed, instead of running to the 72 GPU-hour ceiling. Its evidence,
   void receipts and partial report are preserved under `runs/maniguard/phase2/pilot-v1/`
   and backed up to the private handoff dataset repository.
2. The rest of that rental day runs a prospectively frozen **Clutter replicate diagnostic**
   (`configs/maniguard/phase2/replicate-v1.json`): the eight round-0 genomes on the three
   pilot Clutter scenes, eight replicates each.
3. The headline direction is a **safety-dial execution layer evolved with penalty-free,
   noise-aware evolution** (section 4). Genetic programming of sub-goal re-prompting is a
   gated second contribution.
4. **Compute: this machine only** (about USD 53 per day). No second rental.
5. A **heritability gate** must pass before any ten-seed study (section 5).

## 2. What the pilot measured

- **Replicate noise dominates one-shot fitness.** Round 0 of both search methods evaluates
  the same eight genomes on the same scenes and seeds, so random/s17 and evolution/s17
  panel 0 form a built-in replicate. Of 22 paired Clutter and Jar episodes, 9 disagree in
  outcome. Replays start from an identical robot state, but the first rendered frame already
  differs in 40 to 70 percent of pixels by one or two levels, and the vision-language-action
  policy diverges from its first action. A candidate therefore has an outcome probability
  per case, which the frozen protocol estimates from one episode per family per panel.
- **Family signal.** Clutter sits near 50 percent safe completion and carries nearly all
  fitness differences. Jar is low but not zero on this host: 2 safe completions in 22, both
  by evolution candidates 3 and 4, whose identical random-search twins timed out. Stack is 0
  in 25 here and 2 in 8 on the old host, almost all timeouts at 4,200 actions.
- **Jar failure anatomy (22 traces).** The lid is closed in 9 of 22 episodes, usually by step
  300 to 350. The dominant losses are tipping the open jar while pushing it before any grasp
  (5 of 8 old-host references) and failing to grasp or lift after closing. The two unsafe
  "successes" on the old host lifted an open jar. These are execution-level failures that an
  execution layer can address.
- **Throughput here.** Median process time under eight-way load: Jar 626 s, Stack 1,161 s,
  Clutter 417 s, about 1,400 Clutter episodes per day on eight GPUs. Simulation is CPU bound.

With one episode per candidate, ranking eight candidates is close to a lottery, so evolution,
random search and PPO tie by construction. The evaluation design is the bottleneck, not the
evolutionary operators.

## 3. What the investigation found (web-verified 18 October 2026)

- **Gap.** No published work evolves a history-dependent, monitor-aware execution layer
  around a frozen VLA policy with random-search and RL controls. The nearest are VLA-Pilot
  (per-step evolutionary refinement of action proposals, about +0.11 over best-of-N, no RL
  baseline; arXiv 2511.14178) and DSRL (SAC in the noise space of a frozen diffusion policy;
  arXiv 2506.15799), the RL wrapper baseline reviewers will expect. ES at scale (ICML 2026;
  arXiv 2509.24372) reports rank-based ES beating PPO on sparse outcome rewards with far
  lower seed variance, which matches our ternary terminal outcome.
- **Noise handling is mature.** Rank-based selection needs ordering only; prefer larger
  populations over resampling; do not re-evaluate parents (robust to noise); racing or
  adaptive re-evaluation for final comparisons (GECCO 2025, arXiv 2409.16757); Extract-QD
  style re-evaluation of elites (GECCO 2025, arXiv 2502.06585).
- **Penalty-free constraint handling is the supervisor's brief.** Deb's feasibility rules,
  stochastic ranking, FI-2Pop and multi-objective fronts (NSGA-II, MOME) replace a tuned
  penalty such as the pilot's U = S − 2V. No head-to-head of such methods against
  Lagrangian RL on a manipulation safety benchmark was found.
- **Where random search ties.** When the effective dimension is low and fitness is not
  heritable under the operators, which is exactly the frozen 692-weight controller with
  one-shot fitness. GECCO reviewers expect ten or more seeds, matched budgets, random-search
  and no-adaptation nulls, ablations and corrected effect sizes.
- **Our leverage.** Mid-episode re-prompting is mechanically possible without upstream edits.
  Online-legal signals for a controller: joints, gripper command history, end-effector
  position, tracking error, proposal deltas, time since query, CLIP features. The monitor's
  predicates (closed, upright, on support) are offline-only and usable for fitness.

## 4. Chosen direction

**Claim to test.** A penalty-free, noise-aware evolutionary method finds an interpretable
execution layer around a frozen VLA that raises verified safe completion without raising
violations, and does so more reliably than random search at the same episode budget, than
the same ES ranked by the scalar penalty, and than ordinary policy-gradient learning with the
same information.

**Search space: the safety dial.** Ten to sixteen bounded parameters with physical meaning,
driven only by online-legal signals: joint-delta caps and smoothing per gripper state, an
extra low-height cap while the gripper is open (anti-tip), a hold window after a close
command with lift attenuation, release delay, query interval, and stall-triggered re-query
thresholds. The identity genotype reproduces the frozen policy exactly. Proposed, filtered
and executed actions stay distinguishable in the trace.

**Optimizers compared at matched budgets.** (a) feasibility-ranked (μ,λ)-ES with no parent
re-evaluation; (b) the same ES ranked by the scalar penalty, as the ablation; (c) NSGA-II on
(completion, violation), reporting the front and hypervolume; (d) random search with identical
evaluation and final selection; (e) the existing recurrent PPO as the ordinary-learning
comparison; plus identity and the best hand rule.

**Evaluation.** Fitness per candidate from replicate episodes on training cases; final
comparison on held validation cases with racing-style re-evaluation of finalists; safe
completion, violations and unfinished episodes reported together, with stratified bootstrap
intervals and corrected nonparametric tests across ten seeds per method.

**Families.** Clutter for statistical power, Jar for the safety mechanism (anti-tip, lift
hold), Lid as design-exposed transfer. Cabinet and Dusty stay untouched. Stack is dropped from
training because it carries no completion signal at the frozen horizon.

**Second contribution, gated.** Genetic programming of sub-goal re-prompting programs (close,
then grasp, then carry) has the strongest evolutionary mechanism story but depends on whether
π0.5 follows sub-instructions, which a ten-episode probe decides before any build.

## 5. Gates before the main study

1. **Replicate diagnostic (18 October).** Decides the noise level, whether any
   between-controller difference exists at all, and the replicates needed per candidate.
2. **Heritability probe (next).** Sixteen uniform parents and one Gaussian mutant each in the
   dial space; four replicates on two Clutter scenes and two on one Jar scene, about 250
   episodes. Pass if parent-offspring safe-completion correlation is positive and candidate
   rates are heterogeneous beyond binomial noise. Fail means a different interface or family
   before any seed study.
3. **Sub-instruction probe (optional).** Ten Jar episodes prompted with the lid clause only.

## 6. Budget and timeline on this machine

Per run: population 12, 15 generations, two scenes × two replicates, about 720 episodes.
Four evolutionary or random methods × ten seeds on Clutter, plus Jar, PPO at the same budget
and finalist validation, come to roughly 30,000 to 35,000 episodes: about 20 days of wall
time at 1,400 Clutter-equivalent episodes per day, about USD 1,100 of rental.

| When | Milestone |
| --- | --- |
| 18 to 21 October | Diagnostic, dial implementation, heritability and sub-instruction probes |
| 22 October to mid-November | Main study, backups, interim analyses |
| Mid-November | Thesis results and draft |
| December to 19 January 2027 | Paper writing, ablations, camera-ready figures |
| 26 January 2027 | GECCO full-paper deadline |

Every attempted episode counts; failures are recorded, never retried, except zero-interaction
setup failures under the standing rule. Checkpoints and evidence go to the private Hugging
Face dataset repository; code, configs and numerical summaries to `research/maniguard-fruits`.

## 7. What would make this fail honestly

If the heritability gate shows no structure, the EA cannot beat random search in this space
and the paper has no win; the fallback is the re-prompting program space or a family with a
moderate base rate. If violations trade against completion, the result is a front, not an
unqualified safety improvement, and must be reported as such. Zero observed violations in n
episodes bound the rate by about 3/n, not zero.

## 8. Questions for the supervisor

- Is the penalty-free framing (feasibility rules and fronts versus a tuned penalty) the
  contribution he wants foregrounded, given the allocated topic?
- Ten seeds per method on Clutter with Jar and Lid as mechanism and transfer studies: is that
  the right balance for an 8-page GECCO paper, or should Jar carry equal weight?
- Track choice: Neuroevolution, Evolutionary Machine Learning or Real-World Applications.
