# Bounded calibrated-noise diagnostic

No tested noise level passed the declared joint safety-and-progress transfer
screen. At k=1, calibrated noise exposed useful safety ordering in this fixed
population, but progress ordering still failed the screen. Higher noise did not
improve the result. The full fix grid remains on hold, and the original strict
controller gate remains unpassed.

## Frozen protocol

The user accepted this bounded diagnostic after the transfer failure. Protocol
and code were committed as `ab4fbc7` before execution. The visual controller,
world model, action/feature scalers, physical probes, 101 candidate parameter
vectors and their archived real outcomes remained fixed. There was no retraining
or additional simulator interaction.

From the existing fresh baseline bank, a seeded permutation of episodes excluded
all 64 transfer episodes, then chose 64 episodes to fit the noise scale and 64
separate episodes to check errors. All ten recorded transitions per episode were
included, including unsafe frames. One-step predictions consumed three real
latent frames and the three intervening executed action blocks, with the frozen
action scaler applied once. Sigma is the per-dimension sample standard deviation
of predicted-minus-real residuals on the fit group only. No mean-bias correction
was applied. These whole-episode roles are disjoint; all remain development data.

The original planned positive noise levels, k in {0.5, 1, 2, 4}, each used four
samples per root and common random numbers across candidates and levels. Calls
were candidate-local, with P=1, R=64 and n=4. The no-noise baseline reproduced the
archived actions, readouts and initial readout bitwise at its original n=1 shape.
A Deb winner for each level was fixed using the original 32 selection roots,
before this runner loaded real scores. Those scores had already been seen in the
previous pilot, so this is development analysis, not new blind confirmation.

## Results

The declared screen requires both safety and progress Spearman rho at least 0.5,
a positive lower 95% episode-bootstrap bound and at least 90% defined draws.
The primary view excludes the four attenuation controls and contains the baseline
plus 96 residual controllers. Intervals condition on this one fixed candidate
population and one shared noise seed; they do not include search-seed or
Monte Carlo seed variability. Four explored noise levels do not constitute a
confirmatory multiple-comparison test.

| Noise k | Safety rho (95% interval) | Progress rho (95% interval) | Joint screen |
|---|---|---|---|
| 0, preceding pilot | Undefined, all rates zero | 0.373 (-0.056, 0.479) | Fail |
| 0.5 | -0.035 (-0.158, 0.263) | 0.288 (-0.097, 0.455) | Fail |
| 1 | 0.675 (0.243, 0.703) | 0.374 (0.012, 0.574) | Fail |
| 2 | 0.233 (-0.198, 0.453) | 0.166 (-0.127, 0.509) | Fail |
| 4 | -0.204 (-0.441, 0.268) | 0.272 (-0.031, 0.557) | Fail |

At k=1, mean predicted residual-candidate failure probability is only 3.29%,
against 37.89% in the archived real outcomes. Better ordering is not probability
calibration or safe control. The mean predicted rates at k=0.5, 2 and 4 are
0.22%, 28.15% and 65.67%, respectively. Raising the rate alone does not repair
which candidates imagination prefers.

| Noise k | Selected candidate | Real failures / 64 | Real progress, m |
|---|---|---|---|
| Baseline | 0 | 19 | 2.293 |
| 0, preceding pilot | 55 | 27 | 2.317 |
| 0.5 | 7 | 36 | 2.197 |
| 1 | 51 | 18 | 2.266 |
| 2 | 53 | 19 | 2.264 |
| 4 | 97, zero-action control | 64 | 1.249 |

These all-64 counts include selection roots and are descriptive. On the 32
assessment roots, k=1's winner and baseline each have 10 failures. Their paired
real-failure change is 0 percentage points (95% interval -12.5 to +12.5), with
progress change -0.018 m (-0.088 to +0.038). This does not establish a safety
benefit. The most aggressive noise chooses a zero-action control which fails
every real segment; more pessimistic imagination is not necessarily better
selection.

## Calibration and limits

Fit/check one-step latent RMS errors are 0.4219/0.4234. The root-averaged residual
bias contributes about 1.10%/0.88% of squared error. This small global mean does
not rule out state-dependent systematic errors, which can cancel when averaged.
The fitted sigma RMS is 0.4199 across the 192 latent coordinates. In the separate
check group, fixed-real-action open-loop RMS grows from 0.2962 at the first block
to 0.8762 at block ten. The one-step summary averages all ten temporal positions,
so its RMS is not directly the first-block open-loop statistic.

Calibration uses frozen-controller traces. Perturbed candidates may enter a
different error distribution. Independent per-coordinate Gaussian noise omits
cross-coordinate and temporal error structure. These limitations help interpret
the negative joint screen; they are not permission to keep trying settings on
the same assessment roots without a new protocol.

## Costs, validation and next step

The run used 1,036,800 predictor rows: 1,280 for one-step calibration/checks, 640
for check-group open-loop error growth, 640 for zero-control reproduction and
1,034,240 for noisy rollouts. It used zero new simulator steps, renders, encodes
or optimizer updates. Original trajectory collection and real evaluation remain
charged to their source runs. Measured run time was 32.68 s.

All 654 tests passed (49 warnings), including the three new calibration-role and
window-alignment tests. Repository-wide ruff and git whitespace checks passed.
The supervisor run completed normally.

The next route is the implementation plan's real-simulator evolution fallback.
It uses privileged state feedback and direct simulator fitness, and therefore
answers a different question from world-model exploitation or visual-controller
repair. First prepare and validate the state-policy baseline and cost accounting;
then declare the bounded search and comparison budget before collecting outcomes.
Retain k=1's partial safety-ranking improvement as a research lead requiring fresh
confirmation, without treating it as a passed joint gate or launching the fix grid.

The full local bundle contains calibrated scales/residuals, all noisy readouts and
metrics, baseline/selected action tapes, source, configuration and logs. Source
policy weights and real traces are pinned to the already verified private archives.
Archival of this newly completed bundle is pending.
