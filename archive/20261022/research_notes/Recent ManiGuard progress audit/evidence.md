# Recent ManiGuard latent-evolution and continuation evidence

Repository evidence audited for Git range `1432288..07a7e8c`, ending at commit
`07a7e8cc73c53e13163a47fba5cb2b9f964a68d8`. This note separates physical simulator
evidence from decoder/model evidence, preservation evidence and prospective work.

## What did the native-noise latent-evolution pilot and generation-two work find?

### Takeaway

The implementation and bounded native-noise steering mechanism worked, and the complete
frozen Jar comparison ran to its declared boundary. The result was negative for the main
scientific hypothesis: no method achieved safe requested completion in the protected check,
EA did not beat native, and the apparent fixed/random advantage was only a one-violation
difference with identical fixed and random finalist weights.

### Cited Findings

- The implemented shared controller has 176 parameters and adds a bounded four-dimensional
  offset to native flow noise from measured robot state, executed-action history and frozen
  image/text features. The policy, transforms, eight-action prefix, native horizons and passive
  monitor remain fixed. The monitor labels outcomes but never changes commands. [Pilot report,
  implementation and controls](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md#L96-L143)

- The original decoder-authority audit used 320 calls over eight saved observations per family,
  two native noise draws per observation and ten calls per draw. All 32 same-noise identity
  pairs passed. Perturbations changed executed-prefix arm targets in both Jar and Clutter. The
  Jar RMS min/median/max was 0.001131/0.003193/0.006422 radians and the Clutter values were
  0.000666/0.001868/0.006975. This establishes decoder influence, not physical benefit.
  [Pilot report, bounded decoder authority](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md#L341-L361)

- Four full development episodes established integration and timing, but used different
  task/seed cases and therefore did not estimate an adapter effect. They contained one safe
  requested completion, one safe unfinished episode, two unsafe unfinished episodes and two
  violating episodes. The Jar native raw benchmark goal had an open lid and a violation, so it
  was requested unsafe unfinished. [Pilot report, full development episodes](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md#L322-L339)

- All eight shared-initialization episodes ran for 2,000 controls. None completed either the
  requested task or raw benchmark. Native, positive constant, negative constant and random
  conditional initialization had 1/2, 1/2, 0/2 and 2/2 violating episodes. The nonviolating
  negative constant became the fixed reference and initial incumbent, but it never grasped and
  its apparent lead was safe noncompletion, not safe completion. [Paper, shared initialization](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/native-noise-paper.tex#L432-L470)

- Generation one had eight episodes per method. EA and random search each produced four safe
  unfinished and four unsafe unfinished episodes, four violations, no requested or raw
  completion and no grasp, for population utility `-8`. [Paper, completed training](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/native-noise-paper.tex#L528-L550)

- Generation two also used eight episodes per method on one common two-case panel. Each method
  had six safe unfinished and two unsafe unfinished episodes, two violations, one supported
  closure and utility `-4`. EA had zero raw completions or holding episodes. Random search had
  one raw completion and one holding episode, but it occurred after loss of uprightness and
  without closure, so it remained unsafe and requested-unfinished. The earlier EA training
  utility lead did not persist on this common panel. [Generation-two summary](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/latent-evolution/generation-2-summary-20261020.json); [paper interpretation](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/native-noise-paper.tex#L572-L588)

- Across all 40 comparison-training trials, no episode completed the requested sequence, 16
  violated and 24 were safe unfinished. Four safe unfinished episodes achieved supported lid
  closure but not holding. Thirteen episodes issued closing inputs, but none did so after safe
  supported closure. A post hoc progress tie-break would have changed the factual selected
  candidate in two of five observed cohorts, exposing a fitness limitation, but it did not
  reconstruct a new search history or produce physical evidence. [Fitness-information audit](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/latent-evolution/fitness-information-audit-20261020.json); [paper limitations](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/native-noise-paper.tex#L662-L669)

- The protected check had four episodes per condition. Native and EA each had 0 safe
  completions, 1 safe unfinished episode, 3 unsafe unfinished episodes, 3 violations, 1 unsafe
  raw goal and mean utility `-1.5`. Fixed and random each had 0 safe completions, 2 safe
  unfinished episodes, 2 unsafe unfinished episodes, 2 violations, no raw goal and mean utility
  `-1.0`. Fixed and random used the exact same retained genotype, so they are not independent
  discoveries. EA and native were equal in all paired outcome contrasts. [Independent original60
  assessment](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/latent-evolution/independent-original60-assessment-20261020.json); [paper checking table](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/native-noise-paper.tex#L590-L625)

- The experiment has only four checking blocks per condition and one optimizer run. The paired
  effects are descriptive. They do not support statistical superiority, a safety guarantee,
  transfer or an estimated general safety rate. Common seeds also did not create identical
  camera inputs, although initial robot inputs matched and image differences were generally
  small. [Independent original60 limitations](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/latent-evolution/independent-original60-assessment-20261020.json); [paper uncertainty](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/native-noise-paper.tex#L339-L356)

### Inferences

- The pilot falsified the useful form of the current hypothesis for this controller,
  objective, feedback panel and budget. It did not falsify native-noise steering as a whole.
- The dominant evidence is a task-continuation and objective problem. The controller can alter
  decoder outputs and sometimes produce safe closure, but the measured pipeline did not join
  closure to holding, lift and requested completion.
- The fixed/random checking result cannot be credited to random search because both labels point
  to the same negative-constant matrix.

### Gaps

- No optimizer replication, fresh-family transfer, untouched main-study test, competent
  ordinary-learning baseline, fixed-parent local-search ablation or direct-action residual
  baseline was executed.
- The two-task adaptation panel offers at most one centered instruction contrast, so useful
  semantic conditioning was not established.

## What do the original-42, original-60, screen-four, gripper and continuation artifacts establish?

### Takeaway

The repository now contains independently checkable restoration evidence for the complete
60-episode original comparison and the separate four-start continuation screen. The earlier
42-episode proof was a valid intermediate replay but was explicitly insufficient for clearance.
The continuation and gripper evidence confirms what physically happened, but supplies no safe
handoff, causal diagnosis or permission to expand.

### Cited Findings

- The `original-42` proof redownloaded and hash-checked its archives, reverified 42 physical
  receipts and joined 79,370 exact native monitor entries without new physical controls or
  neural/encoder calls. Its own `all_60_original_valid_episodes_covered` field is false, and the
  continuation admission record says it could not open the screen. [Original-42 physical
  recovery proof](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/latent-evolution/original-42-physical-recovery-proof-20261020.json); [continuation admission requirements](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/continuation/README.md#L30-L38)

- The later `original-60` proof covers all 60 valid episodes. It redownloaded and SHA-256 checked
  every archive, reverified all physical receipts and replayed 112,095 exact native monitor
  entries and 13 authoritative source paths, with zero new physics or neural/encoder calls. It
  retains two early development trials whose per-process GPU UUID receipts were unavailable.
  [Original-60 physical recovery proof](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/latent-evolution/original-60-physical-recovery-proof.json); [study README](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/latent-evolution/README.md#L19-L27)

- The separate continuation protocol froze exactly four charged starts on exposed task
  `task_0019/base`, seed `6101920`: native-only, closure-only, clock handoff at control 488 and a
  privileged three-observation supported-closure handoff. The monitor did not veto actions and
  the failed world model was not deployed. This is a capability screen, not fresh generalization,
  EA evidence or learned-filter evidence. [Continuation README](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/continuation/README.md#L1-L28)

- All four screen starts were valid and reached 2,000 controls, for 8,000 controls and 1,000
  physical VLA queries with no replacements. Native-only and closure-only were safe unfinished.
  Clock and privileged handoff were violating unfinished. Every requested completion, raw goal
  and native holding count was zero. No reference qualified and the larger stage was not
  admitted. [Screen summary](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/continuation/screen-summary-20261020.json)

- Closure-only first closed at observation 596 and remained nonviolating for 1,405 closed
  observations, without holding. Clock first violated uprightness at 475 before switching at 488.
  Privileged closure occurred at 445, debounce at 447, switching at 448 and violation at 471,
  23 native controls later while the jar was still closed and supported. Neither handoff produced
  a later ordered holding, lift or transport milestone. These times do not identify the action
  that caused failure, and the clock violation cannot be attributed to its later switch.
  [Paper, completed physical screen](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/model-followup.tex#L41-L77)

- The trace-bound gripper audit reused all 8,000 screen controls and added zero physical starts,
  policy queries or encoder calls. Native and closure-only commanded opening throughout. Clock
  and privileged modes issued 40 and 142 closing inputs, first at controls 1456 and 642, both
  after their first violations. Measured finger-joint positions later approached zero, but native
  holding remained absent. The measurements are query-time joint coordinates, not fingertip
  aperture, continuous motion or a body-link grasp witness. [Screen gripper audit](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/continuation/screen-gripper-audit-20261020.json); [paper gripper figure interpretation](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/model-followup.tex#L79-L102)

- The `screen-four` recovery proof separately restored and reverified all four episodes, 8,004
  exact native monitor entries, service tails, source hashes, RNG chains, camera evidence and
  actions, with no new physics or policy/encoder calls. Its scope explicitly says progress is
  descriptive and grants no authority for a larger stage. [Four-start restored physical replay](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/continuation/screen-four-physical-recovery-20261020.json)

- The preservation catalog embeds 90 byte-verified receipts: 86 receipts across 43 original-run,
  analysis and paper stages, plus four receipts for the two screen waves. All 36 source commits
  are ancestors of the publication revision. The final private round trip restored 639 indexed
  artifacts, including the ten-page paper, its 33 bound inputs, build log, figures and review
  packets. [Latent-evolution README, preservation status](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/latent-evolution/README.md#L29-L48); [final paper and analysis recovery](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/continuation/final-paper-analysis-recovery-20261020.json)

### Inferences

- The restoration chain is evidence-preservation validation, not a second experimental sample.
  It raises confidence that the adverse result was not produced by a report-export mismatch.
- The screen shows that a safe closure prefix can be produced on the selected exposed case, and
  that native continuation really executed after switching. It also shows that this hard
  composition did not safely progress on that case.
- Later closing commands and joint motion rule out a simple story that the handoff always leaves
  the gripper open. They do not isolate perception, controller, dynamics or timing as the cause.

### Gaps

- Full simulator reset-state equality is unproven. Initial camera images differed, and a single
  selected exposed case cannot estimate a continuation success or violation rate.
- Native holding lacks an independently identified jar-body link witness. No claim should turn
  the measured finger coordinates into confirmed grasping.

## What claims were cleared for the paper, and what limitations or non-claims remain?

### Takeaway

The paper is cleared to report a fully executed and preserved negative exploratory result, plus
bounded decoder influence and a failed continuation capability screen. It is not cleared to claim
an evolutionary advantage, a new safety rate, transfer, practical superiority, successful model
assistance or a defensible GECCO contribution.

### Cited Findings

- The supported paper claim is narrow: native-noise steering was verified in two family policies,
  but the completed Jar pilot established no evolutionary safe-completion advantage. Four safe
  closures did not progress to holding, all four methods had 0/4 safe requested completions in
  checking, raw goals failed the safe requested sequence, and the separate four-start screen did
  not qualify for expansion. [Paper conclusion](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/native-noise-paper.tex#L722-L733)

- The stronger gripper-noise diagnostic is decoder-only. At magnitude 0.5 it changed no gripper
  command in either family's four saved frames. At magnitude 2.0 it changed commands in 3/4 Jar
  and 4/4 Clutter frames, with maximum accompanying arm-target differences of 0.05875 and 0.01773
  radians. Three inputs were already on violating factual trajectories and no Jar input showed a
  successful closure transition. It establishes influence in a candidate representation, not
  physical safety repair. [Paper, gripper-coordinate diagnostic](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/native-noise-paper.tex#L472-L517)

- The CPU world-model audit failed every qualification gate. Model-to-persistence RMSE ratios
  were 1.973 for robot/history, 1.538 for overview and 1.164 for wrist, against a maximum allowed
  ratio of 0.90. Actual-feature readout MAE was 0.191 for closure, 0.209 for holding, 0.662 for
  uprightness and 0.210 for support, all above the 0.10 gate. Calibration had one episode against
  a required minimum of 20 and conservative abstention rejected every prefix. The model used
  existing data only and remained excluded from physical action selection. [Paper, CPU model
  audit](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/model-followup.tex#L114-L157)

- The paper itself was built as ten pages, all 33 declared inputs matched published Git objects,
  all ten pages received root and independent visual review, and the final record reports zero
  layout overflows and zero unresolved references. The publication readback explicitly makes no
  novelty or submission-format claim. [Paper publication readback](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/continuation/paper-publication-readback-20261020.json); [pilot closeout](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/latent-evolution/pilot-closeout-20261020.json)

- The final closeout marks `evolutionary_safe_completion_advantage_established` and
  `defensible_gecco_contribution_established` as false. It also records zero remaining work in
  the already authorized physical protocol, no reference qualified for screen expansion and no
  approval or credential blocker. The work ended because the frozen study was complete, not
  because a positive result appeared. [Pilot closeout](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/latent-evolution/pilot-closeout-20261020.json)

- The paper limits its result to Jar, the random four-direction basis, 176 parameters and only
  two-case candidate feedback. It says matched competent learning, optimizer replication and a
  direct-action residual are needed before practical evolutionary or latent-representation
  superiority claims. It also notes that the local 640-pixel rendering profile is not a matched
  reproduction of the paper's described 256-pixel profile, although saved-input source checks
  produced the expected 224-pixel transformed images. [Paper limitations](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/native-noise-paper.tex#L649-L688)

- Response-informed bases, persistence, sampling-distribution control, activation gates and the
  separately budgeted 156-episode adaptation/transfer bridge remain unexecuted. A parameter-
  REINFORCE interface exists and passed CPU checks, but has zero physical evaluations and is not
  the competent state-conditioned latent SAC comparator the paper requires. [Paper, remaining
  evidence](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/native-noise-paper.tex#L682-L695)

- A prospective progress secondary objective now has 20 focused software checks, including 17
  objective checks and three restoration checks. It preserves ordering of observed integer
  panels but does not guarantee expected-risk behavior. It was not connected to a completed
  search or physical actor, so its physical benefit and optimizer advantage remain untested.
  [Pilot report, prospective software](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md#L74-L82)

### Inferences

- A paper can defensibly report the study as a negative, reproducible pilot and use it to motivate
  the next controller/objective design. The closeout record itself says the present result is not
  yet a defensible venue contribution.
- Model readout failure and imagined-dynamics failure were separately measured. Neither result is
  physical safety evidence, and the failed audit is a reason the model was not deployed rather
  than evidence that model-assisted control can never work.
- The best next claim target is explained safe completion under matched total interaction, not an
  optimizer substitution claim.

### Gaps

- No primary-literature novelty recheck after the final negative result is recorded in this Git
  range beyond the preserved precedent notes. Novelty should be reassessed before submission.
- No new protocol or budget has yet tested the revised progress fitness, direct gripper support,
  persistent control, response-informed coordinates or a softer continuation mechanism.

## How were budgets, outcomes, queries and provenance handled?

### Takeaway

The study used frozen caps, charged failures and discarded candidates, protected checking data,
separate physical/decoder/CPU counters and exact restoration receipts. It reported completion,
violations and unfinished episodes jointly. Remaining uncertainty is explicit: some early physics
was uninstrumented, provider billing was unavailable and the reported rental costs are dated upper
bounds rather than invoices.

### Cited Findings

- The frozen physical allocation was 60 valid full episodes: four development, eight shared
  initialization, 16 EA training, 16 random-search training and 16 protected checking episodes.
  At most four genuine infrastructure failures could be replaced, giving a 64-full-attempt cap.
  Partial attempts, invalid attempts, discarded candidates and valid adverse/time-out outcomes
  were all charged. Six separate engineering starts exhausted their 96-control allowance.
  [Paper allocation](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/native-noise-paper.tex#L291-L320)

- The original run completed 60/60 valid episodes with zero invalid full starts and 16/16
  protected checks. Counting all six earlier engineering starts, it recorded 112,131 policy
  controls, 14,019 physical VLA requests, 112,761 returned environment steps and 678,393 known
  physics ticks. The separate 368 decoder-only calls bring server returns to 14,387, with no
  failed or incomplete return. CPU work used 27,412 training rows, 5,395 model-head row queries
  and 74 synthetic sampler calls while reusing 18,666 already acquired controls. [Pilot report,
  final accounting](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md#L21-L33); [paper model accounting](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/model-followup.tex#L126-L156)

- Adding the four continuation starts gives 70 unique physical starts in total: 60 valid original
  episodes, four valid screen episodes and six earlier engineering starts. Combined totals are
  120,131 controls, 15,019 physical VLA requests, 120,801 returned environment steps, 726,749
  known physics ticks, 30,038 encoder images and 67 text calls. All 15,387 physical-plus-decoder
  returns reconcile, with the 368 decoder-only calls kept separate. [Pilot report, combined
  accounting](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md#L58-L72); [pilot closeout](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/latent-evolution/pilot-closeout-20261020.json)

- Original boundary allocation was 41.091 GPU-hours and a $12.586 upper-bound rental estimate.
  The later combined screen snapshot was 43.179 GPU-hours and $13.226. Both count both GPUs,
  setup, idle time and preservation. Provider billing, stopped storage/transfer and final rental
  closure were not verified, and later preservation time has its own scope. [Pilot report,
  original and combined cost scopes](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md#L21-L33); [paper continuation accounting](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/model-followup.tex#L104-L112)

- Shared initialization was acquired once physically but attributed once to each optimizer's
  information budget. Cost curves retain adaptation, failures and discarded candidates.
  Physical controls, policy calls, encoder work, elapsed time and allocated GPU-hours are
  separate counters. Missing pre-callback physics is reported as unknown, not zero. [Paper,
  analysis and uncertainty](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/native-noise-paper.tex#L339-L356); [pilot report, accounting limits](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md#L363-L384)

- Requested completion was stricter than the raw benchmark verdict. Safe completion required the
  requested sequence and no raw native violation. The utility was `1` for safe requested
  completion and `-2` for any violating episode, while nonviolating unfinished episodes scored
  zero. Reports retained all four completion/safety categories, violation counts, denominators,
  contact/grasp context and raw benchmark outcomes. [Paper, outcome definitions](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/docs/research/reports/maniguard-phase3-part1-gecco/native-noise-paper.tex#L127-L145)

- The final preservation system keeps exact private model/dataset revisions, archive sizes and
  hashes, stage order, source/config hashes and canonical episode layout. Recovery runs validate
  evidence without starting a simulator or policy service. Historical PIDs and commands are
  treated as records, not live resume instructions. [Study README, reconstruction contract](https://github.com/Sachin2911/Safety-Dial/blob/07a7e8cc73c53e13163a47fba5cb2b9f964a68d8/configs/maniguard/latent-evolution/README.md#L50-L103)

### Inferences

- The accounting supports a total-interaction interpretation of the negative result. It does not
  hide unsuccessful preparation or decoder/model work, and it does not double-charge reused CPU
  observations as new physical acquisition.
- Physical replay, source-bound verification and exact client/server joins make report corruption
  or missing returned queries unlikely explanations for the result. They cannot remove sampling
  uncertainty from one optimizer run and four checking cases per condition.

### Gaps

- Physics before callback attachment is unknown for some partial starts. The counters are therefore
  lower bounds where explicitly stated.
- Provider runtime and dollar figures are estimates rather than verified billing. Final shutdown,
  stopped-storage and transfer charges are outside the preserved experimental cost snapshot.
