# Walker pre-failure imagination divergence diagnostic

The user requested implementation and results for the recommendation to locate where imagination diverges before real failures. This inference-only development diagnostic uses all 89 frozen candidate interventions and both existing candidate-quality banks (32 selection episodes, 64 check episodes). Every source episode has already been exposed. No controller is selected, optimized or promoted, and no new simulator or final-bank query is authorized or needed by this protocol.

## Five prediction paths

All predictions use the frozen model, probe and controller assets. New predictions use k=0, one candidate per call and the original bank-sized batches. Starting histories, normalization, action blocks and model context windows remain unchanged.

1. Teacher-forced: predict the next block from the preceding three REAL latent frames and executed real action blocks. Restart from observed history every block.
2. Refresh every two blocks: restart the complete three-frame context from observed history at blocks 0, 2, 4, 6 and 8. Recur on predictions between resets.
3. Refresh every five blocks: restart at blocks 0 and 5.
4. Real-action tape: predict all ten blocks from the initial real history, replaying the exact real executed actions.
5. Closed-loop imagination: reproduce the original controller's k0 rollout, including its action cap where applicable; the controller reacts to imagined frames.

The first four modes use the SAME executed real actions. Their comparison isolates accumulated state-prediction error under that tape. The difference between tape and closed-loop predictions also includes the consequences of different controller actions. These differences are diagnostic interventions, not an additive causal attribution theorem. Executed future actions and refreshed real states are privileged hindsight information. Reset modes are not valid root-time segment-risk forecasts or deployable corrections. A favorable diagnostic reset result does not establish a feasible algorithm or improved controller.

Keep the action encoder's full 12-position buffer shape and predictor's three-frame shape unchanged at every block. Teacher forcing at block b may access real frames only through b; it must never read its target frame b+1 as input. All four tape modes must agree exactly at the first predicted block. Reproduce the original k0 check-bank readout exactly for all 89 candidates, and reproduce every real-image readout exactly. Compare the first real and imagined action block within 1e-4 per coordinate, retaining the actual discrepancy; this permits established batch-kernel rounding effects, not a different action interface.

## Measurements and time alignment

Retain predicted latents and physical readouts for every mode and block, RMS latent error against real encoded frames, actual block-end height/pitch/speed, the real-image probe readout, the real dense first failure step, and closed-loop versus executed action divergence.

Separate physical prediction error from readout error by also measuring predicted-minus-real-image probe output. The primary descriptive divergence threshold is absolute height error over 0.05 m OR pitch error over 0.1 rad against the REAL-IMAGE READOUT. Sensitivity thresholds are (0.025 m, 0.05 rad) and (0.1 m, 0.2 rad). These are descriptive scales, not calibrated risk margins.

One physics step is 8 ms and one model block is 80 ms. The stored dense first failure is zero-based. Count divergence as preceding failure only when the offending predicted endpoint occurs strictly before that first dense event. An error at the containing block end, or after the event, is not an advance warning. Keep censored healthy trajectories separate. Report physical/readout MAE only over still-healthy prefixes, alongside full-horizon latent error (which can include post-failure states).

For failed branches, report whether the containing block's true endpoint is unsafe, whether the real-image probe detects that endpoint, and whether each prediction path detects it. Also report any alarm through the failure-containing block, explicitly distinguished from detection at the actual failing endpoint. Refresh modes can observe earlier real history, so their alarm timing is not directly comparable to a forecast issued only at the root. Whole-segment alarms may depend on post-failure real observations and are descriptive only.

Identify the first breached health boundary from dense simulator state (low/high height or negative/positive pitch), resolving simultaneous normalized-margin ties by fixed order. This labels a constraint crossing, not the physical cause of a fall.

## Aggregation and presentation

Report each bank separately and preserve the fixed 89-candidate pool. Candidates share search directions, parents and episodes, so candidate/episode pairs are not independent trials. Descriptive 95% bootstrap intervals resample whole source episodes, including episodes with no failed candidates, conditional on the fixed pool. Use 2,000 replicates and seed 20261503. Do not claim these intervals support a fresh confirmation or model-seed generalization.

Show error against rollout time, detection under increasingly frequent real-history resets, and four illustrative trajectories: candidates 0, 1, 31 and 33, using each candidate's first failed check episode in stored order. The deterministic examples are secondary to the complete-pool results; they cannot establish prevalence.

## Accounting and verification

There are 89 x (32+64) = 8,544 existing branches. Five ten-step model paths require exactly 427,200 new predictor rows. New simulator steps, renders, image encodes, gradient updates and source collection are all zero. Original simulator collection and parent experiment costs remain additional. Real/prospective policy comparisons cannot treat this hindsight analysis as free simulator feedback.

Freeze and commit the protocol and executable sources before inference. Hash the parent archives, banks and source snapshot. Run under supervisor with a 30-minute absolute deadline preserved on resume. Query receipts retain numeric outputs and costs; interrupted queries forbid implicit replay. Tests must cover correct action/history windows, exact predictions under known dynamics, accumulation under known bias, future-frame leakage, dense-event timing and censoring, and episode-clustered denominators. Reconcile all archives and independently reproduce summary statistics before interpreting the results.

```bash
uv run python experiments/scripts/evo_divergence.py
uv run python experiments/scripts/evo_divergence.py --execute --run-id walker2d-evo-s6-divergence-20261005-1
uv run python experiments/scripts/evo_divergence_report.py runs/walker2d-evo-s6-divergence-20261005-1
```
