# Corrected continuation evidence

Status: in progress. Experimental success is not assumed. Negative scientific gates stop downstream claims and experiments.

Original E1, E2 and E3 outputs are preserved. Fourteen original source trajectories crossed development and test roles, so those outputs are development diagnostics. See [the shared audit](../continuation-audit/shared_protocol_audit.md).

Fresh Walker collection contains 2,999,673 training steps, 300,347 probe steps and 300,038 root-source steps. Episode roles and seed ranges are disjoint. The dataset is backed up at private revision `ddd4d51648d2a382725efd20d04cedebedbf8fd2`.

The 1,000-step LeWM smoke run passed nonblack rendering, whole-episode split and both loss checks. Mean prediction loss fell from 0.41495 to 0.22369; SIGReg fell from 16.86563 to 3.80469. The measured rate was 2.214 iterations/s, implying about 27 hours for 214,780 optimizer steps under comparable conditions. Full ten-epoch training is running with the declared upstream architecture, batch and loss recipe. Actual completion and S3 probe outcomes remain pending.

LeWM recovery checkpoints are uploaded every 4,000 optimizer steps. Each contains one atomic recovery bundle with model, optimizer, scheduler, random states and a reproducible training-batch cursor, alongside source snapshots and model exports. The interval was increased from 2,000 to fit both full model histories within verified private storage while retaining a 10 GB reserve. The full-run preflight projects 44.94 GB for both checkpoint histories including overhead, with about 30.72 GB remaining beyond the reserve. A recovered run must use a fresh run ID and identical data, split and recipe.

Push-T physical probes have been refitted with source-aware splits and uploaded at private revision `f03fb888ff15a4e757600a51c038a0988429c05c`. The first corrected bank build exhausted its finite test/familiar source pool at 26 accepted roots, before the requested 64. Its partial banks are preserved and cannot support a completed study. See [the failure record](bank_generation_failure.json). [Fresh continuation-v3 recovery](../continuation-v3/README.md) is running with the same source partitions and geometric filters, bounded extra candidates per source, incremental provenance, and development feasibility checked before final-bank generation.

Fresh CPU replay validation passed on 50 stored roots, including 16 roots with collision contact (the old counter includes walls). Repeated observed poses, block velocities, rendered frames and terminal masks agree bitwise. Ten stopped branches retain explicit censored suffixes. Geometry was cross-checked against an independent implementation; all ten contact overlays passed the separate [visual review](e0-visual-review.json). The old contact counter is not evidence of pusher–block contact specifically, so that mechanistic stratum uses the new body-specific observer. A separate [contact coverage check](e0-contact-coverage-combined.json) adds one recorded-action root: 51 roots from 50 source episodes, with 15 roots showing pusher–T contact during the last prefix block and nine at its final step. All three repeats agree bitwise; the original 50-root qualification is preserved. This validates deterministic replay of the recorded float32 tapes, not equivalence to the original unquantized action arrays.

The original E3 child finished as a development diagnostic. Its automatic queue and the stale Walker queue were then retired, after confirming neither had a live child, because they would launch stale-data or ungated downstream work. See [the retirement record](original_queue_retirement.json). The corrected workflows use finite subprocesses, private upload barriers, exact output hashes and explicit scientific gate stops.

Current machine-readable status is in [continuation_status.json](continuation_status.json). Live stage states and logs are under ignored `runs/workflows/`; scientific results, manifests and pinned receipts are retained in this results tree and private Hugging Face repositories.

The qualified post-E2 continuation is committed in `88f8dea`, with 221 repository tests
passing and Ruff clean at that commit. Its v2 managed service was verified waiting, then
stopped without starting an experiment when bank generation failed and its prerequisites
could no longer arrive. The same qualified downstream design will be used in the fresh
recovery: three paired acquisition seeds, all four additional budgets, final-checkpoint
retention and the geometric transfer report. E5 remains conditional on repeatable gains
and retention for the preselected checkpoint. See [the continuation audit](../continuation-audit/remaining_requirements.md).

The [8,000-step recovery checkpoint](walker_checkpoint_8000.json) passed 23 read-only
checks against the actual saved tensors, optimizer, scheduler, dataset, splits and
local upload receipts. Validation prediction loss is 0.0839862, down from 0.1565982 at
step 4,000. Its pinned private revision is
`691d38fbe2fff4042a7a54861d5523e497dd0c49`. These checks establish the consistency of
this recovery checkpoint; the full training outcome and S3 probes are still pending.

The [24,000-step recovery audit](walker_checkpoint_24000.json) passed all 23 CPU
checks. The atomic model matches its weights export exactly; all Adam states and the
cosine scheduler agree on step 24,000, and the deterministic next-batch cursor is
epoch 1, offset 2,522. The validation prediction loss recorded by training is
0.018410890363156796. Matching upload receipts pin private revision
`25d59af8bc3ab39db11ff3ff71abda486e594695`. This audit did not rerun GPU recovery,
rehash the unchanged dataset or download remote checkpoint bytes; the full
214,780-step training run remains in progress.
