# Prediction repair worked; controller improvement did not follow

The predictor-only repair passed the predeclared development and audit gates. On the 24 audit source episodes, one-step failure-endpoint detection rose from 11.3% to 67.7%, and height/pitch error on still-healthy endpoints decreased. The same repaired model then selected a controller with 3/24 real failures, compared with 2/24 for the baseline. This supports the prediction repair, but does not support promoting its selected controller.

## What was changed and how it was tested

The image encoder, latent projector, action encoder, scalers, and all running statistics remained frozen. Each of three seeds started from the same original predictor and probe. The predictor and prediction projector received 1,000 updates on real latent transitions, with a latent MSE loss plus a physical prediction term through the original frozen probe. A separate same-capacity probe received 1,000 updates on a different episode group. No controller parameters changed.

The 96 previously exposed source episodes were assigned by a frozen hash rule to four groups of 24: probe fitting, predictor fitting, development, and audit. All 89 controller branches from any episode stayed together. Failure-rich and ordinary fitting examples were sampled equally; states after the first failure-containing endpoint were excluded. This is retrospective, privileged supervision using simulator physical labels and saved real latents. The audit was excluded from fitting and model/controller selection in this run, but it is reused development evidence, not fresh confirmation.

## Audit results

There were 351 failed candidate/episode pairs, with 344 unsafe first-failure-containing endpoints. Seven pairs were no longer unsafe at their containing endpoint. The table averages the three fitting seeds for repairs; baseline has one fixed model. These are correlated branches from 24 source episodes, not hundreds of independent trials.

| Measure | Original predictor, original probe | Repaired predictor, original probe |
|---|---:|---:|
| One-step failure-endpoint detection | 11.3% | 67.7% |
| Healthy-endpoint false-alarm rate | 0.39% | 0.85% |
| Healthy height MAE | 0.02375 m | 0.02250 m |
| Healthy pitch MAE | 0.07903 rad | 0.07571 rad |
| Healthy speed MAE | 0.4462 m/s | 0.4824 m/s |
| Mean height bias at failure-containing endpoint | +0.1488 m | +0.0273 m |
| Real-action-tape failure-endpoint detection | 0% | 31.2% |

The paired episode-cluster 95% interval for the detection improvement is +48.0 to +61.2 percentage points (point +56.4). False alarms increased by 0.46 points [0.22, 0.75], so there is a real tradeoff; they did not stay unchanged. Speed MAE rose by about 8%, within the predeclared 25% retention ceiling. Height and pitch MAE improved. All three repaired predictors showed similar one-step detection: 69.2%, 66.0%, and 68.0%.

The endpoint measurement can occur up to 72 ms after the first dense failure. It measures prediction from real history one block earlier, not a root-time warning across the full 0.8-second horizon. The substantial drop from 67.7% with refreshed real history to 31.2% with the real action tape shows that rollout reliability remains incomplete.

An additional post-run check, unused in the gate or controller nomination, found that healthy-prefix latent RMSE also fell: 0.4540 to 0.3785 for one-step prediction (about 16.6%), and 0.8177 to 0.7910 for the action tape. Thus the observed benefit includes better latent prediction, rather than only a changed physical readout. These latent units are model-specific.

## Why the probe repair was not adopted

The probe-only arm reduced one-step imagined-endpoint detection to 0.9%. The combined arm reached 37.3%, but failed the development speed-error retention gate. On real image latents, the repaired probe reduced height MAE and false alarms while lowering first-event detection from 61.6% to 56.3%. Better average regression error therefore did not imply better boundary detection in this recipe. Keep the original probe for this predictor repair; this result does not rule out other probe-calibration approaches.

## The controller test

The same predeclared risk-and-progress selection rule was applied to original-model and repaired-model development imagination. The original model predicted zero baseline failures, so no candidate could meet the strictly lower-risk requirement and it fell back to candidate 000. The repaired model nominated candidate 076, a fixed random-direction candidate at scale 2.0, with imagined failure rate 12.5% versus 18.1% for the baseline controller. Both nominations were frozen before numerical audit summaries were produced.

On the separate audit group, candidate 076 had 3/24 real failures versus baseline 2/24. Its mean progress was 1.9649 versus 1.9995 (98.3% retention). The paired failure difference was +4.17 percentage points, with descriptive episode-bootstrap interval [0, 12.5]. The small, reused group does not establish a general worsening, but it provides no evidence of controller improvement. No controller should be promoted from this experiment.

The next research question is whether the repaired model can reliably rank controller risk over complete closed-loop rollouts. More one-step accuracy alone is insufficient. A follow-up should freeze a ranking/calibration test on separate development trajectories, retain the original probe as a control, and reserve fresh confirmation for a selection rule that succeeds. Another environment and more controller evolution can wait until that link is demonstrated.

## Verification and archival status

All 534 inference archives, 363 input hashes, 51 frozen source hashes, and three checkpoints passed the offline audit. Numerical summaries, event counts, gate decisions, nominations, and controller outcomes reproduced. The baseline nomination's progress reduction differed by at most 8.9e-16 after NPZ layout conversion; its risk, eligibility, and selected index were identical. The report explicitly allows only 1e-12 progress roundoff, with dedicated tests. All 783 prelaunch tests and two additional report-audit tests passed. The figure was visually inspected.

Study cost: 3,000 predictor and 3,000 probe updates, 768,000 fitting rows for each, 320,400 new inference predictor rows, and 97.17 seconds for the local workflow. Engineering CUDA tests additionally used 16 synthetic predictor fitting rows. There were zero new simulator steps, renders, image encodes, or encoder updates. Historical interaction and training costs remain additional.

The experiment and private archival are complete. Following explicit user approval of the prepared bundle, all three checkpoints and 612 total files were uploaded to the existing private repository at revision `5a4fc59e534aa3cebad64c448625c1bb6a4b1f30`. Every uploaded file hash and the immutable run tag verified. [Archive record](ARCHIVE.md). Initial automatic upload rejections are retained in the approval history; they are resolved.

See [report and figure](REPORT.md), [analysis](analysis.json), [paired changes](paired_changes.json), [latent supplement](latent_supplement.json), [integrity audit](integrity_audit.json), and [prospective protocol](../../../protocols/repair-20261005.md).
