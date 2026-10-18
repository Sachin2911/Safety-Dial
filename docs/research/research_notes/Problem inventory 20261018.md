# Problem inventory, 18 October 2026

**Problems before solutions.** This note collects the problems that our own studies and the
literature have surfaced, with the evidence for each, how open it looks and what it would cost
us to pursue. It selects no method. It draws on our archived and current studies and on four
reviews compiled on 18 October 2026 from repository notes and primary papers. Claims that a
problem is open rest on dated searches (5, 16 and 18 October 2026), not exhaustive reviews.
Numbers marked *via our notes* come from earlier literature sessions and were not re-verified
today.

## The pattern across our studies

In no study did the choice of optimizer decide the result. The signal it optimized, or the
lever it controlled, decided it.

| How the signal or lever failed | Where we saw it | Evidence |
| --- | --- | --- |
| The signal lies | Walker evolution in LeWM; Push-T probe | Growing the search from 16 to 4,096 evaluated candidates, imagined failure went from 0.31% to 0.47% while real failure rose from 19.02% to 29.56%. Imagination caught 0 of 819 first unsafe endpoints. All 36 evolved arms failed more often than the starting controller (16.88%, 54/320). The Push-T probe's 95th-percentile error was 7.5 px on real frames and 35.5 px after five imagined steps, and it was blind to a pusher that had left the arena. |
| The signal mumbles | ManiGuard phase two; Walker stage six | 9 of 22 paired replays of the same controller on the same scene and seed disagree in outcome; first frames already differ in 40–70% of pixels. Of 89 Walker candidates, 13 beat the baseline on their selection episodes, 1 on check episodes and none on both. |
| The signal can be gamed | Push-T dial 60; Walker stage three; VLA inaction | Every Push-T seed at dial 60 left the arena, so its zero violations were vacuous. At k = 4 the Walker search picked a do-nothing controller that failed 64/64 for real. Zero-shot VLAs in ManiGuard are 77–83% safe-unfinished, yet violate in 30–54% of the episodes where they engage. |
| The lever is too easy or too weak | UNISafe; ManiGuard phase one | UNISafe went from 2/64 to 54–56/64 successes with every optimizer and with a fixed rule. Six ManiGuard execution adapters over 196 episodes gave no repeatable safe-completion gain; the failures are choices such as tipping the jar before grasping and failing to lift, not execution speed. |

Our evidence: [Stage 5 main](../../../archive/legacy/docs/evoPlan/results/stage5-main/walker2d-evo-s5-main-20261004-1/README.md),
[stage 3 noise](../../../archive/legacy/docs/evoPlan/results/stage3-noise-diagnostic/walker2d-evo-s3-noise-diagnostic-20261004-1/README.md),
stage 6 under `archive/legacy/docs/evoPlan/results/stage6-*`, [GECCO draft](../../../archive/legacy/docs/geccoPaper/main.tex),
[Safe CEM](../../../archive/legacy/notes/safeCEM.md), [Phase 0](../../../archive/legacy/notes/phase0Report.md),
[LeWM through S4](../../../archive/legacy/docs/mainPlan/reports/through-s4-20260927/report.md),
[UNISafe study](../reports/UNISafe%20grasp%20maintenance%20and%20evolutionary%20adaptation.md),
[ManiGuard six hypotheses](../reports/ManiGuard%20six-hypothesis%20study.md) and [direction.md](../../../direction.md).

## Problems

### 1. Learned safety models crack under search pressure

When candidates are selected by a learned safety model (a world model, readout probe, filter or
critic), more selection pressure finds the model's blind spots. Measured safety gets worse while
predicted safety looks unchanged.

- **Ours (locked protocol, 20 seeds, independent audit).** The Walker numbers above. Real failure
  rose with distance from the baseline actions (rank correlation 0.78). At 256 offspring, sixteen
  generations failed 12.09 points more often than one generation, with no progress gain. Noise plus
  penalty cut the widening gap by 7.22 points, but every arm stayed worse than the starting
  controller.
- **Literature, via our notes.** CMA-ES trained in a deterministic dream scored 2086 imagined but
  193 real, below a random policy's 210 ([World Models](https://worldmodels.github.io/)).
  Best-of-N selection against a learned filter raised hacking from 2.3% to 99.96% as N grew from
  1 to 8192 on a toy task, and that theory excludes adaptive search
  ([Wachi et al.](https://arxiv.org/abs/2608.22915)). Growing a candidate pool from 72 to 288 cut
  minimum-cost feasibility from 0.375 to 0.062 ([ASAR](https://arxiv.org/abs/2607.23602)). World
  models stay optimistic out of distribution ([MiraBench](https://arxiv.org/html/2605.29360),
  [Biased Dreams](https://arxiv.org/abs/2604.25416)).
- **How open.** The 5 October notes found no curve of exploitation against evolutionary pressure,
  and proposed fixes target the model or filter rather than the selection operator.
  [BasinLens](https://arxiv.org/html/2608.22421) reports testing no genetic algorithm or CMA-ES.
  See `exploitation_and_adversarial_evolution.md` and `gecco_and_venue_landscape.md` under
  `archive/legacy/docs/research/research_notes/Evolutionary safe RL open problems/`.
- **Cost for us.** Low. The Walker/LeWM and Push-T pipelines exist and run on one GPU.
- **Scope.** The same risk applies to VLA steering that selects proposals using a learned critic
  or verifier, such as CheckVLA or VLA-Pilot.

### 2. VLA physical safety is too noisy and costly to measure

VLA rollouts in simulation are chaotic, so single-episode outcomes are close to coin flips and
safety comparisons need hundreds of rollouts per arm.

- **Ours.** 9 of 22 paired replays disagree. Under eight-way load, median episodes take 417 s
  (Clutter), 626 s (Jar) and 1,161 s (Stack).
- **ManiGuard.** Same-seed runs from identical states differ in 39% of overview pixels, and the
  paper reports no confidence intervals ([Appendix C](https://arxiv.org/html/2608.17386v1#A3)). A
  two-proportion check that assumes independent rollouts, and is therefore optimistic, puts the
  π0 versus π0.5 safe-success gap at p ≈ 0.28. A ManiGuard rollout costs about 3.8 GPU-minutes,
  derived from the same appendix.
- **Elsewhere.** Re-running BEHAVIOR-1K checkpoints, the simulator family ManiGuard uses, misses
  posted scores by more than 27% on three tasks ([source](https://arxiv.org/html/2604.21192v1)).
  Distinguishing an 8% from a 12.5% violation rate needs about 800 rollouts per arm, and zero
  violations in n episodes bound the rate only near 3/n (`decision_aware_safety_models.md`).
- **How open.** Sequential testing cuts trials by up to 32% ([STEP](https://arxiv.org/abs/2503.10966))
  and 70% ([successor](https://arxiv.org/abs/2603.13616)), but both were validated on success or
  progress, not physical safety.
- **Cost for us.** Medium. The replicate diagnostic already measures the noise.

### 3. Safety metrics reward inaction and exits

Doing nothing, stalling or leaving the workspace scores as safe.

- **Ours.** The Push-T arena exits and the Walker do-nothing controller described above.
- **Literature.** ManiGuard's zero-shot policies above. Filters trade failures for unfinished
  episodes: UNISafe with a diffusion policy moved from 52/44/4 to 57/15/28 percent
  success/failure/incomplete, and LatentCBF reports large stall fractions
  (`task_progress_after_intervention.md`, via our notes).
- **How open.** Partly addressed by engagement-conditioned rates such as ManiGuard's violation
  rate after first contact. Constraint handling that avoids degenerate feasible solutions is less
  studied.
- **Cost for us.** Low. It is a design requirement for any study.

### 4. VLAs succeed unsafely, and stress tests do not look

Tasks complete while safety specifications are violated, and current failure search scores only
task success.

- **Evidence.** In ManiGuard 6–21% of fine-tuned successes are unsafe. In
  [SafeVLA-Bench](https://arxiv.org/abs/2606.00773), policies above 90% success are still 18–28%
  unsafe, and 38–56% of RoboCasa successes are unsafe. In [SafeStage](https://arxiv.org/abs/2609.21223)
  70.8% of successes are unsafe. Displacing objects takes π0 and π0.5 Jar safe success to zero in
  ManiGuard.
- **How open.** Partly. [Q-DEG](https://arxiv.org/abs/2610.14415) (10 October 2026),
  [FATE-VLA](https://arxiv.org/abs/2606.02307) and [DAERT](https://arxiv.org/abs/2604.05595)
  search for task failures only. Q-DEG notes that fitness from four rollouts is sparse and often
  tied. Q-DEG's group is close to this problem.
- **Cost for us.** Medium to high. Unsafe successes are only 2–5% of ManiGuard rollouts for the
  fine-tuned π0 and π0.5, and scene generation needs integration work.

### 5. VLAs skip prerequisites and lack task competence

- **Evidence.** SafeStage prerequisite hazards are violated in 87.7–95.6% of cases, and safety
  prompts lower safe success to 5.9%. In ManiGuard, putting the specification in the training
  prompt raises Jar safe-once-engaged rates by 3–13 points but gives Stack no consistent gain,
  and the LTLf form cuts π0.5 Jar safe success from 26.9% to 14.1%. Our Jar traces close the lid
  in 9 of 22 episodes and often tip the open jar before grasping. In DAERT, π0.5 still succeeds
  54.9% of the time when told "no action", so instructions may not steer it.
- **How open.** The fixes found so far retrain weights, for example with specification-aware data
  or preference optimization on safety labels.
- **Cost for us.** High.

### 6. Simulated violation rates do not predict real ones

ManiGuard's 12 sim-real cells correlate at r = 0.91 for success and 0.88 for safe success, but
0.09 for violation rate. Real Clutter out-of-distribution unsafe success is 13–47%, against zero
in simulation. The evidence is thin and testing it needs hardware.

## Low-hanging assessment

| Problem | Our evidence | Open | Cost for us | Evolution's natural role |
| --- | --- | --- | --- | --- |
| 1. Safety models crack under search | Strong, locked | Yes, per our notes | Low | Selection pressure is the variable under study |
| 2. Safety too noisy to measure | Measured | For safety | Medium | Noisy optimization, racing, rare-event search |
| 3. Inaction scores as safe | Repeated | Partly | Low | Constraint handling |
| 4. Unsafe success unseen | Indirect | Partly | Medium to high | Failure search |
| 5. Prerequisites and competence | Measured | Fixes retrain | High | Weak |
| 6. Sim-real violation gap | None | Yes | Very high | Weak |

## Checks before relying on this

- Our pilot saw Clutter violations in 3 of 8 episodes, seven of them with random adapters, while
  ManiGuard reports zero Clutter violations for fine-tuned π0 and π0.5. ManiGuard's default render
  is now 640×640, whereas the paper used the 256×256 `legacy` setting. Confirm which configuration
  our runs used.
- Numbers marked *via our notes* were compiled in earlier sessions and not re-verified today.
- Openness claims come from dated searches, not exhaustive reviews.
