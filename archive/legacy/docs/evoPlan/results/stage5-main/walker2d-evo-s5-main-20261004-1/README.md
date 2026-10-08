# Walker2d main study: completed results

The locked study supports both declared primary comparisons on fresh episodes: stronger search enlarged the real-minus-imagined failure gap in the control, and the combined noise plus ROSARL-style method reduced that enlargement. At the declared high-pressure setting, the combined method also reduced actual simulator failures while retaining 99.04% of control progress. It still failed more often than the frozen baseline controller.

Run: `walker2d-evo-s5-main-20261004-1`. Twenty paired search seeds, four methods, populations 16/64/256, sixteen generations, and 320 independent fresh final episodes. All 720 selected checkpoints were frozen before final evaluation. The 6,400 seed/episode pairs are not 6,400 independent episodes.

| Declared primary comparison | Estimate, percentage points | Adjusted 97.5% interval |
|---|---:|---:|
| Control gap: high pressure minus low pressure | +10.39 | +6.70 to +14.25 |
| Combined method minus control gap amplification | -7.22 | -12.18 to -2.49 |

Both intervals support their declared directions. The two intervals use Bonferroni family coverage of 95% and a paired bootstrap that resamples search seeds and source episodes independently. Low pressure is population 16, generation 1; high pressure is population 256, generation 16. This does not establish monotonic improvement or deterioration with every increase in population size.

| Method at high pressure | Actual failure rate | Own-noise imagined failure rate | Mean forward progress |
|---|---:|---:|---:|
| No noise, zero penalty | 29.56% | 0.47% | 2.167 m |
| No noise, ROSARL-style penalty | 27.47% | 0.47% | 2.162 m |
| Noise, zero penalty | 25.66% | 5.09% | 2.155 m |
| Noise plus ROSARL-style penalty | 22.36% | 4.95% | 2.146 m |

The combined-minus-control actual failure difference is -7.20 percentage points (descriptive paired 95% interval -10.98 to -3.70). Its progress ratio is 0.9904 (0.9824 to 0.9978), exceeding the declared 0.90 retention threshold. Mean progress is approximately 0.96% lower than the control, so there is a small measured progress tradeoff. These practical comparisons are separate from the adjusted primary inference. The other rows provide the full four-arm comparison; their ordering alone does not establish a statistically tested interaction or universal superiority.

The frozen baseline failed on 54/320 episodes (16.88%). The combined high-pressure method increased failures relative to that baseline by 5.48 percentage points (descriptive paired 95% interval +2.56 to +8.44). The result therefore demonstrates partial mitigation of optimization-induced harm under this protocol. It does not establish improvement over the starting controller, calibrated risk probabilities, or safe deployment.

The saved error decomposition is consistent with the earlier diagnosis. At high pressure, missed between-block events contribute only about 0.38 percentage points to the control gap and 0.39 to the combined-method gap. Real-image readouts overstate endpoint violations on average, while imagined predictions remain much lower. These descriptive differences do not isolate one-step model error from accumulated closed-loop divergence.

Historical Gate 0 and the transfer screen remain failed. The model, base controller, probe, projection, noise scale, and analysis choices stayed frozen. These are 0.8-second branches, conditional on the fixed assets, fitness/selection banks, and source distribution. They do not establish full-episode safety or generalization to other environments.

The run completed in 6.819 hours, within its 12-hour cap, charging 444,403,200 predictor rows, 23,567,921 real simulator steps, and zero gradient updates. The scoped ledger records 26,838,972 real steps across 24 preserved evolution runs, plus 3,600,058 shared original data-collection steps. Historical actor-training and some other costs remain unquantified; this is not a lifetime total.

See the [complete report and figures](report/README.md), [all declared cells](report/all_cells.csv), [exact analysis](analysis.json), [error decomposition](decomposition.json), and [cost ledger](report/cost_ledger.json). The [selection freeze check](selection_freeze_integrity.json) records the full declared grid; the [independent final audit](final_integrity.json) passed, verifying 118 source files and 11,322 query archives and reproducing the complete analysis and decomposition without new simulator or model queries.

The [descriptive reporting supplement](supplement/README.md) completes the common-k0 audits, baseline-change plots, factorial table, and all equal-offspring comparisons from saved data. Its figures and source are committed locally and are outside the approved archive payload.

The user-approved main-study bundle is [privately archived at immutable revision `081050f26263`](https://huggingface.co/Sachioster/safetydial-walker2d/tree/081050f26263354f5936b39c29912c301c6722d7/evo/walker2d-evo-s5-main-20261004-1). The [remote verification](remote_verification.json) confirms all 11,743 files and 11,619,385,219 bytes match the approved inventory, the run tag matches this revision, and the repository remains private.
