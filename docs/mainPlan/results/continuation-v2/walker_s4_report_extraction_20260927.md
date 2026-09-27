# Walker S4: saved report extraction

Run `walker2d-s4-recovery-20260927-1` is complete. This is a report-only extraction from the immutable study (SHA256 `30da356a1fc111425b663baf9f4939edc7a39f0a5edff533f78bec471b29c632`), after the passed local checker. Values below are copied, or unit conversions/simple arithmetic explicitly described in the JSON. No new bootstrap or experiments.

## Findings

- Health is the only qualified rule. S3 speed qualification failed; these data do not restore it.
- Each margin is calibrated to development acceptance 0.96875 and frozen for test/stress. Achieved held-out acceptance differs; at_matched is a producer key, not evidence of equal test acceptance.
- Test and stress each use 64 source episodes/roots and 512 branches. Three acquisition seeds share these fixed banks and are not three independent test-bank replications.
- The 12 reported paired intervals are saved 95% source-episode cluster bootstrap intervals with 1000/1000 usable replicates each. They are not adjusted for multiple comparisons, and seeds/banks are not pooled.
- Test: five paired intervals include zero; N512 seed2 indicates increased FSA for boundary. N512 seed0 has an upper endpoint exactly zero and remains inconclusive.
- Stress: all three N128 paired intervals indicate lower boundary FSA, alongside lower boundary acceptance. All three N512 intervals include zero. This is a mixed result, not a uniform acquisition advantage.
- AUC-dial is normalized trapezoid area of FSA across attainable acceptance rates 0.2 to 0.9, using interpolation at range boundaries. Lower is better. All adapted AUC values are below no-update on these fixed banks; this is descriptive, without an AUC uncertainty test or global acquisition ranking.
- Ordinary retention is prediction error on 64 held-out policy tapes over 10 blocks (0.8 seconds), not closed-loop policy return, goal success, or a retention noninferiority test. All final-block state errors worsen; displacement MAE worsens in 11 of 12 adapted points. Stress has zero ordinary-policy rows and no retention estimate.
- Displacement prediction integrates the 10 endpoint speed predictions times 0.08 seconds; truth is saved final minus initial x. Height/pitch/speed errors use metres/radians/metres per second. These are model prediction diagnostics.
- Each point allocates all 300038 root-generation steps plus 12800 common-seed steps and 12800 or 51200 cumulative acquired steps, giving 325638 or 364038 charged steps. These reused allocations must not be summed as lifetime simulation cost.
- Evaluation bank construction is reported separately as 121600 environment steps. Model candidate queries are cumulative acquisition-scoring counts only: random 0; boundary 640 then 1152. They exclude evaluation and training forward passes.
- Each of 12 models is restarted from the same base and receives 1500 optimizer updates with the fixed predictor-side recipe. The sum of measured adaptation intervals is a wall-time accounting sum, not GPU time. S4 reported wall_clock_s excludes unrelated stages and is not total project cost.

## Test: model operating points and dial score

FSA and AR are percentages. AUC is a dimensionless normalized score over AR 0.2 to 0.9. Every row has 512 branches; no censoring is recorded.

| Arm | Seed | N | Accepted | False safe | FSA (%) | AR (%) | AUC-dial |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_update | - | - | 504 | 225 | 44.6429 | 98.4375 | 0.450086 |
| random | 0 | 128 | 496 | 216 | 43.5484 | 96.8750 | 0.354651 |
| random | 0 | 512 | 495 | 218 | 44.0404 | 96.6797 | 0.343998 |
| random | 1 | 128 | 502 | 220 | 43.8247 | 98.0469 | 0.349925 |
| random | 1 | 512 | 492 | 211 | 42.8862 | 96.0938 | 0.345275 |
| random | 2 | 128 | 498 | 221 | 44.3775 | 97.2656 | 0.351358 |
| random | 2 | 512 | 488 | 212 | 43.4426 | 95.3125 | 0.336824 |
| boundary | 0 | 128 | 498 | 219 | 43.9759 | 97.2656 | 0.359462 |
| boundary | 0 | 512 | 488 | 211 | 43.2377 | 95.3125 | 0.343597 |
| boundary | 1 | 128 | 490 | 212 | 43.2653 | 95.7031 | 0.355711 |
| boundary | 1 | 512 | 493 | 214 | 43.4077 | 96.2891 | 0.343073 |
| boundary | 2 | 128 | 497 | 220 | 44.2656 | 97.0703 | 0.360933 |
| boundary | 2 | 512 | 504 | 226 | 44.8413 | 98.4375 | 0.329440 |

### Test: boundary minus random

Point and 95% interval use percentage points. Each cell has 64 source clusters and 1000/1000 usable bootstrap replicates; these are saved unadjusted intervals.

| N | Seed | Difference (pp) | 95% interval (pp) | Saved direction | AR difference (pp) |
| --- | --- | --- | --- | --- | --- |
| 128 | 0 | 0.427517 | [-0.154621, 1.147256] | inconclusive | 0.390625 |
| 128 | 1 | -0.559395 | [-1.582945, 0.326114] | inconclusive | -2.343750 |
| 128 | 2 | -0.111916 | [-0.958661, 0.675885] | inconclusive | -0.195312 |
| 512 | 0 | -0.802699 | [-1.737249, 0.000000] | inconclusive | -1.367188 |
| 512 | 1 | 0.521529 | [-0.090663, 1.197700] | inconclusive | 0.195312 |
| 512 | 2 | 1.398647 | [0.472332, 2.597523] | increase | 3.125000 |

## Stress: model operating points and dial score

FSA and AR are percentages. AUC is a dimensionless normalized score over AR 0.2 to 0.9. Every row has 512 branches; no censoring is recorded.

| Arm | Seed | N | Accepted | False safe | FSA (%) | AR (%) | AUC-dial |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_update | - | - | 494 | 358 | 72.4696 | 96.4844 | 0.714776 |
| random | 0 | 128 | 471 | 337 | 71.5499 | 91.9922 | 0.623814 |
| random | 0 | 512 | 457 | 320 | 70.0219 | 89.2578 | 0.600629 |
| random | 1 | 128 | 470 | 335 | 71.2766 | 91.7969 | 0.610309 |
| random | 1 | 512 | 464 | 329 | 70.9052 | 90.6250 | 0.597840 |
| random | 2 | 128 | 487 | 351 | 72.0739 | 95.1172 | 0.625594 |
| random | 2 | 512 | 480 | 345 | 71.8750 | 93.7500 | 0.590085 |
| boundary | 0 | 128 | 450 | 315 | 70.0000 | 87.8906 | 0.611446 |
| boundary | 0 | 512 | 449 | 316 | 70.3786 | 87.6953 | 0.594410 |
| boundary | 1 | 128 | 447 | 312 | 69.7987 | 87.3047 | 0.616507 |
| boundary | 1 | 512 | 459 | 325 | 70.8061 | 89.6484 | 0.598091 |
| boundary | 2 | 128 | 473 | 336 | 71.0359 | 92.3828 | 0.621909 |
| boundary | 2 | 512 | 488 | 351 | 71.9262 | 95.3125 | 0.586031 |

### Stress: boundary minus random

Point and 95% interval use percentage points. Each cell has 64 source clusters and 1000/1000 usable bootstrap replicates; these are saved unadjusted intervals.

| N | Seed | Difference (pp) | 95% interval (pp) | Saved direction | AR difference (pp) |
| --- | --- | --- | --- | --- | --- |
| 128 | 0 | -1.549894 | [-2.786727, -0.192958] | decrease | -4.101562 |
| 128 | 1 | -1.477938 | [-2.436302, -0.525334] | decrease | -4.492188 |
| 128 | 2 | -1.037981 | [-1.805358, -0.316356] | decrease | -2.734375 |
| 512 | 0 | 0.356737 | [-0.910959, 1.736588] | inconclusive | -1.562500 |
| 512 | 1 | -0.099072 | [-1.077323, 0.865990] | inconclusive | -0.976562 |
| 512 | 2 | 0.051230 | [-0.866183, 0.808129] | inconclusive | 1.562500 |

## Ordinary-policy prediction retention

Test only, 64 fixed ordinary policy tapes, 0.8-second open-loop horizon. These are errors, not goal-success counts. Smaller is better. Stress contains no ordinary-policy rows.

| Arm | Seed | N | Displacement MAE (m) | Change vs base (m) | Last height MAE (m) | Last pitch MAE (rad) | Last speed MAE (m/s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_update | - | - | 0.094034 | 0 | 0.009146 | 0.024810 | 0.196216 |
| random | 0 | 128 | 0.098977 | +0.004943 | 0.029769 | 0.060391 | 0.225404 |
| random | 0 | 512 | 0.115226 | +0.021192 | 0.049287 | 0.094232 | 0.242128 |
| random | 1 | 128 | 0.095548 | +0.001514 | 0.026135 | 0.041014 | 0.233798 |
| random | 1 | 512 | 0.109253 | +0.015219 | 0.033991 | 0.043695 | 0.250857 |
| random | 2 | 128 | 0.091735 | -0.002299 | 0.030287 | 0.078847 | 0.199643 |
| random | 2 | 512 | 0.107084 | +0.013050 | 0.043499 | 0.107552 | 0.276385 |
| boundary | 0 | 128 | 0.097444 | +0.003410 | 0.036966 | 0.077515 | 0.264948 |
| boundary | 0 | 512 | 0.118528 | +0.024494 | 0.046887 | 0.089373 | 0.263048 |
| boundary | 1 | 128 | 0.104044 | +0.010010 | 0.019411 | 0.033158 | 0.255015 |
| boundary | 1 | 512 | 0.111914 | +0.017880 | 0.032246 | 0.041550 | 0.272758 |
| boundary | 2 | 128 | 0.101898 | +0.007864 | 0.033240 | 0.062373 | 0.230188 |
| boundary | 2 | 512 | 0.109289 | +0.015255 | 0.046062 | 0.120176 | 0.279740 |

## Costs and timing

| Arm | Seed | N | Allocated charged steps | Candidate queries | Updates | Adaptation wall seconds |
| --- | --- | --- | --- | --- | --- | --- |
| random | 0 | 128 | 325638 | 0 | 1500 | 202.784302 |
| random | 0 | 512 | 364038 | 0 | 1500 | 201.414104 |
| random | 1 | 128 | 325638 | 0 | 1500 | 198.394193 |
| random | 1 | 512 | 364038 | 0 | 1500 | 200.186812 |
| random | 2 | 128 | 325638 | 0 | 1500 | 207.046732 |
| random | 2 | 512 | 364038 | 0 | 1500 | 202.570279 |
| boundary | 0 | 128 | 325638 | 640 | 1500 | 200.879735 |
| boundary | 0 | 512 | 364038 | 1152 | 1500 | 198.456618 |
| boundary | 1 | 128 | 325638 | 640 | 1500 | 203.703479 |
| boundary | 1 | 512 | 364038 | 1152 | 1500 | 200.747487 |
| boundary | 2 | 128 | 325638 | 640 | 1500 | 198.730840 |
| boundary | 2 | 512 | 364038 | 1152 | 1500 | 200.148868 |

Reported S4 wall interval: 2957.663787 s (49.294 min). Sum of 12 measured adaptation intervals: 2415.063449 s; 18000 optimizer updates across the 12 distinct fits. These are not GPU-hours or total project cost. Separately recorded evaluation bank construction: 121600 steps.

## Evidence limits

- Saved aggregates and paired records are copied; only elementary rates, differences, ranges, counts and time sums are recomputed. No bootstrap, AUC, raw-row error aggregation, selection, model, simulation or network call is performed.
- The passed independent local checker is cited with its exact hash and its original limitations retained. Runtime HF revisions here are copied identities, not a fresh remote durability audit.
- Current source files are hashed to document interpretation of units and formulas; this extraction does not upgrade current source hashes into proof of executed source.
- No single pooled estimate or winner is selected. No task-retention guarantee or population-level safety claim follows from these fixed-bank diagnostics.

The compact report figure retains all 12 saved paired records, charged costs and row-hash references exactly. Source hashes and file identities were unchanged before and after extraction. The JSON retains full numeric precision, full baseline metrics and all ordinary-retention horizons.
