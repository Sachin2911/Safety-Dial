# What we learned from the Walker divergence diagnostic

The frozen prediction pipeline systematically misses the onset of real failures. Providing the exact executed actions does not resolve this. Supplying fresh real history at every block helps substantially, but one-step predictions still miss most unsafe endpoints. Both local prediction/readout error and accumulation over imagined histories deserve attention before another controller search.

## Evidence on the reused development check

All 89 frozen candidates were evaluated on the same 64 previously exposed source episodes: 5,696 correlated candidate/episode pairs, including 861 failures across 43 source episodes. The first-failure-containing block ended unsafe in 819 pairs; 42 transient dense events were no longer unsafe at that endpoint. The following denominator is those 819 unsafe endpoints.

| Prediction path | Detected endpoints | Detection | Episode-cluster 95% interval |
|---|---:|---:|---:|
| Real history every block, exact real actions | 128 / 819 | 15.6% | 7.0% to 24.5% |
| Real history every two blocks, exact real actions | 101 / 819 | 12.3% | 4.8% to 20.8% |
| Real history every five blocks, exact real actions | 18 / 819 | 2.2% | 1.0% to 4.0% |
| Exact real actions, no history refresh | 1 / 819 | 0.12% | 0% to 0.46% |
| Original closed-loop imagination | 0 / 819 | 0% | 0% to 0% in this observed pool |
| Probe applied to real images | 572 / 819 | 69.8% | 51.9% to 84.6% |

The zero-width observed interval is not a population guarantee. The 32-episode selection bank shows the same ordering: 15.0%, 11.5%, 3.1%, 0.15%, and 0% for the five prediction paths. These are retrospective endpoint measurements, not prospective warning rates: the endpoint can follow the first dense failure by up to 72 ms.

## What the controls establish

**Action feedback is insufficient to explain the misses.** Giving the model the exact real action sequence still detects only one unsafe endpoint. Different actions induced by imagined states can worsen errors, but removing that difference does not restore useful failure detection.

**Accumulation matters, and there is already an error with fresh real history.** More frequent refresh improves endpoint detection and reduces height/pitch error on still-healthy prefixes. Nevertheless, even a one-block prediction from three real history frames detects only 15.6% of unsafe endpoints. Each block is 80 ms. These privileged resets are diagnostic interventions, not a deployable controller repair.

**Predicted height is biased upward around failure.** Across all 861 failed pairs, mean predicted-minus-true height at the first-failure-containing endpoint is +0.156 m with fresh history, +0.375 m with the real action tape, and +0.393 m in closed loop. This supports a concrete failure-rich height-calibration investigation. It does not identify a unique architectural cause: the reported prediction pipeline includes both latent dynamics and a probe applied to predicted latents.

**The real-image probe is not perfect at failure onset.** It detects 799/861 failed trajectories eventually (92.8%), but only 572/819 first unsafe block endpoints (69.8%). The earlier candidate-quality observation of eventual detection for the four highlighted controllers remains true: their eventual counts are 4/4, 12/12, 6/6, and 6/6, while immediate counts are 2/4, 10/12, 4/6, and 4/6. These are different timing questions and different candidate populations. The probe also flags 313/4,835 non-failing pairs somewhere in the segment.

**Advance divergence is not yet a useful failure discriminator.** The predeclared error threshold is crossed before failure in 860/861 fresh-history pairs and all 861 closed-loop pairs, but it is also crossed somewhere in 3,457/4,835 healthy fresh-history trajectories (71.5%) and 4,578/4,835 healthy closed-loop trajectories (94.7%). These healthy and failing observation windows differ, so this is not a matched-time classifier comparison. It establishes that threshold crossing is common outside failures too. Moreover, this residual requires the actual future image readout; it cannot itself be used as a root-time warning signal. The healthy-trajectory and probe-timing checks were added after the primary analysis and are explicitly descriptive in timing_supplement.json.

## Consequence for the next phase

The next experiment should test whether a targeted repair improves local failure prediction on separate development episodes. Freeze episode splits before fitting; compare probe calibration, one-step latent prediction, and their combination using failure-rich examples. Measure first-crossing detection, false alarms at matched observation times, physical height/pitch error, and normal-motion accuracy. Keep a fresh confirmation set untouched until the repair and controller-selection rule are fixed.

Only after prediction quality improves should we rerun the fixed candidate comparison and test whether it selects a safer real controller. This experiment diagnoses an obstacle to controller improvement; it does not demonstrate a controller improvement. Another environment would then test whether the repaired method transfers, rather than obscuring the Walker failure mechanism.

## Provenance and limits

This is exploratory evidence conditional on a fixed, correlated pool and reused episodes. Intervals resample whole source episodes. No final confirmation episodes were queried, no controller was updated, and no model was trained. All 178 archives passed the integrity audit; all 89 original check-bank closed-loop readouts reproduced bitwise. The offline supplement independently reproduces primary event and pre-failure divergence counts. Both figures were visually inspected.

The run used 427,200 new predictor rows and zero new simulator steps, renders, image encodes, or gradient updates. The corrected process took 46.79 seconds including setup and writes. The inherited budget clock recorded 260.65 seconds, including a retained zero-query failed attempt and the repair interval. The first attempt failed on CUDA action normalization before model prediction; its failure record and source snapshot remain preserved. The replacement inherited its deadline. The original frozen report source also remains preserved; the current report adds timing clarification and offline descriptive checks without changing the primary analysis.

See [report](REPORT.md), [primary analysis](analysis.json), [integrity audit](integrity_audit.json), [timing supplement](timing_supplement.json), and [validation](validation.json).
