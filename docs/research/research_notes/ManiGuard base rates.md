# ManiGuard base rates for the pi0.5 fine-tuned policies

Recorded 22 October 2026 from the ManiGuard paper (arXiv 2608.17386v1), Tables 13 and 20,
in-distribution condition, three policy-sampling seeds. Counts are over all three seeds.
The policy for each family is that family's own pi0.5-SFT checkpoint, the same
checkpoints used locally. Horizons are from Appendix C.

## Four-outcome counts, pi0.5-SFT, in distribution

| Family | n | Safe success | Unsafe success | Safe unsuccess | Unsafe unsuccess | Horizon |
|---|---|---|---|---|---|---|
| Clutter | 165 | 132 | 0 | 33 | 0 | 1,100 |
| Cabinet | 105 | 0 | 0 | 74 | 31 | 5,700 |
| Lid | 90 | 9 | 1 | 79 | 1 | 1,900 |
| Stack | 84 | 17 | 7 | 46 | 14 | 4,200 |
| Jar | 78 | 21 | 4 | 27 | 26 | 2,000 |
| Dusty | 78 | 0 | 0 | 57 | 21 | 5,100 |

## Derived rates

| Family | Safe success | Task success | Violation | Engaged | Violation given engaged |
|---|---|---|---|---|---|
| Clutter | 80.0% | 80.0% | 0.0% | 87.9% | 0.0% |
| Cabinet | 0.0% | 0.0% | 29.5% | 99.0% | 29.9% |
| Lid | 10.0% | 11.1% | 2.2% | 67.8% | 3.3% |
| Stack | 20.2% | 28.6% | 25.0% | 82.1% | 30.3% |
| Jar | 26.9% | 32.1% | 38.5% | 98.7% | 39.0% |
| Dusty | 0.0% | 0.0% | 26.9% | 59.0% | 45.2% |

For reference, the weaker pi0-SFT checkpoints (Table 12) give Jar 11/8/36/23 (safe
success 14.1%, violation 39.7%) and Stack 13/7/33/31 (15.5%, 45.2%).

## What a safety dial needs from a family

Evolution can only improve safe completion where the base policy already produces
variance on both axes: it must complete sometimes and violate sometimes, and the
violations must come from acting rather than from doing nothing. A family fails as a
testbed if any of these is missing.

- **Clutter** never violates. Any improvement is pure competence; there is no safety to
  trade. Useful only for engineering checks.
- **Cabinet and Dusty** never succeed. There is no completion signal to select on, and
  Dusty engages in only 59% of rollouts, so a controller could look safer simply by
  disengaging.
- **Lid** succeeds 10% of the time but violates in only 2 of 90 rollouts. Its failures
  are almost all safe-unsuccess, so it is a competence problem and a dial has nothing to
  trade. Engagement is 68%, which also leaves the disengagement loophole open.
- **Stack** has the right shape: 29% success, 25% violation, 82% engagement. Its horizon
  is 4,200 steps, 2.1 times Jar's, so every episode costs roughly twice as much. The
  paper's instruction-format study also found no prompt format steered Stack safety.
- **Jar** has the best shape on every axis: 32% success, 38% violation, 99% engagement,
  the shortest horizon among families with violations, and existing local tooling.
  Thirty of 78 rollouts violate and 27 more fail safely, so a large share of rollouts
  are candidates for conversion to safe success. Engagement at 99% closes the
  disengagement loophole. Its specification is an ordering clause (stay on the table
  until closed), which is the kind of temporal requirement a timing or persistence
  knob can plausibly affect.

**Decision: start on Jar. Stack is the declared backup.** If Jar's pi0.5 checkpoint
proves too strong or too weak once local rates are reproduced, the pi0-SFT Jar
checkpoint at 14% safe success with 40% violations is a second option on the same
family.

## Caveats carried from the paper

- All rates are means over 26 base tasks for Jar. Per-task rates are not published.
  Local pilots used mainly task_0019 and task_0009, so local and published rates are not
  directly comparable until per-task outcomes are measured.
- The paper's success predicate does not require jar closure. Its safety clause only
  forbids leaving the table before closure. An open jar slid into the goal is a safe
  success under the benchmark and a failure under the local requested-sequence rule.
  Record both scores on every episode.
- The paper observes 256 by 256 renders from collection-time cameras, executes 8 of
  each 16-step chunk at 20 Hz, and terminates once the goal holds for 10 steps. Local
  runs used the high-fidelity 640 profile. Match the paper's protocol before comparing.
- The renderer is not bit-reproducible: identically seeded runs differed in 39% of
  overview pixels. Seed-level exactness applies to accounting, not physics.

## Measured locally, 23 October 2026

Reproduced under the paper's protocol on one host (machine 54514), seeds 0, 0, 1 and 2,
104 episodes: safe success 23.1% (24), unsafe success 6.7% (7), safe unsuccess 35.6%
(37), unsafe unsuccess 34.6% (36), engaged 99.0%; violations 41.3%. Against the
published 26.9% / 5.1% / 34.6% / 33.3%: p = 0.60 (safe success) and 0.76 (violations).
Per-task rates are now measured over four runs; 13 tasks never gave a safe success, 12
are in the middle, task 12 always succeeded. Full account in
[Jar replication and noise floor](Jar%20replication%20and%20noise%20floor.md). Use the
measured rates, not the published ones, to size every later stage.
