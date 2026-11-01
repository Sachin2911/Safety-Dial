# Safe evolutionary planning recovery

## Which planning failure should the next controller address?

### Takeaway

Fix prediction/execution action semantics first, then test target selection rather
than only increasing residual-action network size. The closest new primary source
reports that frozen LeWM can stall because of the short-horizon final-goal objective,
even when prediction and search are adequate.

### Cited Findings

- **Aim Short to Reach Far**, a September 2026 preprint, studies frozen LeWM on
  Cube, PushT, Reacher and TwoRoom. Its construction shows that exact dynamics and
  globally optimal short-horizon final-goal search can still choose inaction. Its
  Anchored Planning controller retrieves a training segment using current and goal
  encodings, then scores actions toward an early observed successor while retaining
  the original physical success criterion. AP-CEM clips proposals before prediction.
  Target offset and retrieval timing matter, and AP does not beat released LeWM in
  every task/configuration. [Paper, sections 2-4 and appendices A, D, E](https://arxiv.org/html/2609.30036v1)
- The official AP repository separates observation-only AP-CEM from action-labeled
  AP-rank and includes recorded outcomes. Fresh evaluation additionally needs the
  external image datasets and encoded caches; those large assets are not bundled.
  This is an implementation cost, not free privileged memory.
  [Official reproduction guide](https://raw.githubusercontent.com/daybraeklaxry/Aim-Short-to-Reach-Far/main/REPRODUCING.md)
- Hierarchical Planning with Latent World Models reports that coarse-model predicted
  latent states can guide a fine-scale planner and improve non-greedy control. It
  requires learned coarse dynamics and macro-action encoders, so it is evidence for
  temporal decomposition rather than a ready seven-hour frozen-model intervention.
  [HWM, sections 1-2](https://arxiv.org/html/2604.03208v3)
- iCEM demonstrates that correlated action samples and trajectory memory materially
  improve sample efficiency. Consequently, such fixed search refinements are strong
  controls and should not be credited to learned evolutionary weights.
  [Pinneri et al., CoRL](https://proceedings.mlr.press/v155/pinneri21a.html)

### Inferences

- **Rank 1:** establish an action-contract repair control before architecture changes.
  The same legal executed command must be what the frozen world model and learned
  safety model evaluate. Otherwise ES may exploit model/execution discrepancies.
- **Rank 2:** use a small persistent target-selection controller, preferably in
  TwoRoom, instead of evolving 510 unconstrained neural residual parameters. A
  candidate genotype could control current/source versus goal/endpoint retrieval
  weighting, target offset, clearance preference and target-switch hysteresis.
- Select actual recorded target encodings rather than unrestricted latent vectors.
  This keeps the candidate target tied to an observed image, though it does not
  establish reachability from the current state or safety of the intervening route.
- Change only the short-term planning target. Continue checking every proposed action
  with the existing learned safety mechanism and measure completion at the original
  final goal. Temporary detours must be allowed to increase final-goal distance.
- The seven-hour engineering shortcut is a fixed, small, training-only observation
  bank, not training a new high-level world model. Cache encodings once and charge
  all encoder queries. Do not use final-evaluation episodes as memory entries.

### Gaps

- These sources do not establish that target selection learned by ES beats tuned
  fixed retrieval, random search, or ordinary learning under our virtual hazards.
- The AP paper is a preprint and does not test this project's learned safety checker.
- No result here identifies whether the latest PushT failures primarily arise from
  action semantics, target scoring, feasible-route sampling or the safety model.
  Those are prospective diagnostic hypotheses, not established diagnoses.

## What structured waypoint controller is practical and what information does it use?

### Takeaway

Observed subgoals are the most defensible compact representation. Geometry-based
waypoints are a faster fallback but their coordinate information and hand-built
routing structure must be explicit and shared by all optimizer comparators.

### Cited Findings

- Search on the Replay Buffer decomposes distant goals into shorter subgoals,
  using previously seen observations as graph nodes and a learned goal-conditioned
  value function for edge weights. It demonstrates successful long-horizon visual
  navigation; it is not a frozen-LeWM or evolutionary method.
  [SoRB primary paper](https://arxiv.org/abs/1906.05253)
- The AP implementation retains LeWM's pretrained action scaler in its original
  precision, computes live action-scoring encodings separately, and documents
  explicit controller action clipping. Such interface details are part of the
  method rather than incidental preprocessing.
  [Official implementation details](https://raw.githubusercontent.com/daybraeklaxry/Aim-Short-to-Reach-Far/main/IMPLEMENTATION.md)
- Classical attractive/repulsive potential fields can contain unwanted local
  minima even around simple obstacles. Exact navigation functions require additional
  mathematical structure and assumptions.
  [Rimon and Koditschek, 1992](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/r/rimon_koditschek_potential.pdf)

### Inferences

- A compact controller could rank a fixed candidate set of observed nearby targets
  using frozen image similarities, current body proprioception, declared hazard
  geometry and a persistent route-side state. Four to eight scalar weights or a
  small linear scorer would give CMA-ES a much smaller search space.
- Freeze the bank and candidate-generation rule before genotype evaluation. ES
  should learn persistent selection behavior across episodes, not merely optimize
  a fresh waypoint sequence separately for every test episode.
- Avoid a raw attractive-plus-repulsive field as the sole new mechanism. It is easy
  to implement but repeats the local-minimum problem. If used as a proposal source,
  add a declared side commitment and waypoint-passing condition, then retain the
  fixed version as a comparator.
- An exact final-goal XY coordinate is extra information if ordinary LeWM receives
  only a goal image. Either use image-based retrieval, derive coordinates with a
  separately validated frozen readout available to every method, or label the
  controller as coordinate-assisted and give every comparator the same coordinates.
- Counterfactual goal-image rendering from simulator state is simulator preparation
  and must be charged. A recorded-image bank avoids that dependency; no simulator
  previews or hazard rendering are necessary for AP-style observation retrieval.

### Gaps

- Available memory coverage around the new virtual hazards is unknown. Recorded
  trajectories may cross hazards, so a retrieved successor alone is not a safe route.
- If a waypoint family already solves every development task with one hand-chosen
  setting, a subsequent ES improvement is ceiling-limited; recheck task variance
  before an optimizer comparison.

## Which controls make an evolutionary result credible?

### Takeaway

A learned controller can be useful without establishing evolutionary optimizer
superiority. Separate those claims, keep information equal, and count the full
training/selection interaction cost rather than only test episodes.

### Cited Findings

- Robust CEM first forms a feasible trajectory set and then ranks objective value
  within it; if no feasible set exists, it fits toward lower-cost samples. Its
  learned-model Safety Gym results support constrained search as a practical
  mechanism, not a guarantee for this project's calibrated model.
  [Liu et al., algorithm 1](https://arxiv.org/html/2010.07968)
- Mania, Guy and Recht show that compact policy random search is a competitive
  baseline and highlight large variation across random seeds and hyperparameter
  settings. Comparing ES only with a poor native or one random controller would
  therefore be weak evidence.
  [NeurIPS 2018 paper](https://proceedings.neurips.cc/paper_files/paper/2018/hash/7634ea65a4e6d9041cfd3f7de18e334a-Abstract.html)
- The repository requires native, tuned fixed, parameter random search and ordinary
  learning with the same information before an evolutionary-advantage claim.
  [Repository research controls](../../../../AGENTS.md)

### Inferences

- Freeze a small mixed-success development panel, a separate selection panel and
  untouched confirmation cases before evaluating weights. Do not replace difficult
  cases after observing a candidate's result.
- Compare the exact same controller family under CMA-ES, parameter random search
  and a small policy-gradient/ordinary-learning learner. Share initialization,
  observation features, memory, training cases and model/simulator allowances.
- Include zero/native recovery and a tuned fixed subgoal/detour rule. Credit the
  controller representation separately from how its weights were optimized.
- Rank safety feasibility and original task completion before speed or progress.
  Keep unfinished but engaged episodes visible; a violation-free stationary policy
  is not a useful safety result.
- Use several optimizer seeds and paired confirmation episodes if time permits.
  Report real simulator actions, failed starts, cached encoder work, predictor
  work and elapsed time separately. Same candidate caps do not imply equal work.
- If ES beats native and the fixed rule on confirmation but ties random search,
  claim an evolved-controller result with no established optimizer advantage.
  If only a development panel improves, call it a development signal.

### Gaps

- No source guarantees a positive result within seven hours or $10. Null results
  and interface failures must remain in the record rather than being relabelled.
- The cited RCE safety evidence uses its own learned uncertainty model and benchmark;
  it does not certify the current point/disk geometry, full robot safety, or zero
  violations outside the tested tasks.
