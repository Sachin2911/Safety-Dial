# First pilot: targeted experience for safer LeWM

**Status: not started.** Adopted 21 September 2026. This is the execution checklist
for [the research direction](../researchDirection.md), not a second proposal.
Existing experiments are indexed in [experiments/README.md](../../experiments/README.md).
No new replay, acquisition or adaptation runner is implemented yet.

## E0: recover the baseline and validate the experiment

- [ ] Recover the historical Push-T checkpoint, upstream commit, fitted scalers,
  pusher-probe cache if available and a small original-data replay subset.
- [ ] Record artifact hashes, model/package revisions, camera, image processing,
  action normalisation, seeds and hardware in a local run manifest.
- [ ] Load `pusht/lewm` once and reproduce a short baseline trajectory with the
  historical action and observation conventions.
- [ ] Implement seeded reset plus recorded-action-prefix replay. Verify identical
  suffix outcomes at contact, including block position, angle and velocities.
- [ ] Define whole-T forbidden-region intersection from the simulator shape polygons.
  Check geometry overlays on about ten development contact traces.
- [ ] Log every environment step inside each five-action block. Compare dense truth
  with true coarse endpoints and interpolation; inspect substep timing separately.
- [ ] Freeze a development-tested layout generator with clear starting and target
  footprints. Keep representative and contact/rotation stress banks separate.

**Exit evidence:** asset manifest, replay comparison, geometry overlays and declared
sampling/timing. A saved 7-d pose or rendered animation does not pass replay validation.
The first implementation task is this gate, not a new acquisition algorithm.

## E1: diagnose the original model

- [ ] Split complete source trajectories into probe training, development and held-out
  evaluation before drawing roots or sibling branches.
- [ ] Fit linear and small MLP block-pose probes on real encoded observations. Use
  centre plus periodic `(sin theta, cos theta)` and select capacity on development data.
- [ ] Freeze the selected probe. On identical action tapes compare dense truth,
  coarse true poses/interpolation, real-observation latents and imagined latents.
- [ ] Report pose/clearance errors and false-safe decisions by horizon, contact,
  rotation and predicted boundary distance, alongside acceptance counts.
- [ ] Include stationary-block and coordinate-dynamics references; label privileged
  inputs. Keep the existing commanded-pusher arena check as a common control.

**Exit decision:** predictor repair is justified only if the real-latent readout is
usable and imagined predictions add meaningful error. Diagnose poor observability,
readout coverage or temporal omission before expanding the study.

Illustrative screening sizes are 20,000 probe-training frames and 24 roots with eight
five-block action tapes each. These are development starting points, not frozen final
sample sizes or a claim of statistical power. Test states are held out from our study;
unknown pretraining provenance prevents claiming they were never seen by the checkpoint.

## E2: the smallest repairability comparison

- [ ] Compare ordinary versus predicted-boundary-selected contact experience at one
  modest, equal charged simulator-step budget, plus no update and a fixed margin.
- [ ] Freeze encoder, observation projector, running statistics, scalers and the
  physical readout. Cache detached targets from actual new observations.
- [ ] Choose predictor-side trainable modules and one fixed adaptation recipe on
  development data. Use the same original-data replay mix and optimiser steps.
- [ ] Check whether a cheap readout-only residual correction explains the improvement.
- [ ] Evaluate untouched roots and ordinary task motion as well as hazard decisions.

**Pilot deliverables:** an error-decomposition plot, paired before/after results, charged
query/training costs and representative successes/failures. No update to the encoder
means no claim that it learned new safety features. Stop to diagnose if added experience
does not repair the error after a bounded optimisation check.

## E3-E5: conditional main study

- [ ] **E3:** compare random supported experience, predicted-boundary coverage and a
  learned optimistic-error-risk selector. Each starts from the same paid seed data;
  branch selection happens before execution. A true-error selector is an oracle reference.
- [ ] Measure curves at equal charged simulator-step budgets. Count root generation,
  reset prefixes, failed/discarded trials and all queries. Report model compute separately.
  Example branch counts only apply to matched prefix lengths and horizons.
- [ ] Restart each data-budget adaptation from the same released predictor weights,
  with paired seeds, equal optimiser steps and a fixed replay ratio. Use the latest
  available adapted model for the next acquisition round; freeze selection rules on dev.
- [ ] **E4:** evaluate a held-out grid of starts/goals and hazard layouts. Keep all
  descendants of a source trajectory together and test outcomes out of acquisition.
  Hazard relabelling does not create independent trajectories or new physics experience.
- [ ] Report false-safe acceptance with accepted/total counts and acceptance rate;
  undefined when nothing is accepted. Include progress and arena/censoring outcomes.
- [ ] Confirm a useful pilot effect across acquisition seeds and, where claimed,
  independent training seeds. Group uncertainty by root/source trajectory.
- [ ] **E5:** only then freeze weights for matched Safe-CEM and penalty-CEM runs.
  Audit the returned action sequence, declare all-infeasible fallback and report latency.

A learned selector that fails to beat boundary coverage is a legitimate result.
A second task, encoder adaptation and large acquisition sweeps require a separate
scoping decision after this pilot.

## Commands available now

From the repository root:

```bash
# Static checks and a no-download configuration preview.
uv run ruff check .
uv run python scripts/download_data.py --cfg job

# Only if checkpoint/source restoration is needed; this does not provide fitted scalers.
uv run python scripts/download_data.py weights_only=true
```

Recover scalers/replay data from the old run where possible. Downloading the full
Push-T dataset is an explicit asset-recovery alternative, not part of a config preview.
See [checkpoint details](checkpoints.md). Store generated data/checkpoints under ignored
`data/` paths; do not commit them or turn the checklist into claimed experimental results.
Add actual configs under `configs/<group>/` when a runner consumes them, rather than
creating unused configuration scaffolding now.
