# Corrected continuation evidence

Status: in progress. Experimental success is not assumed. Negative scientific gates stop downstream claims and experiments.

Original E1, E2 and E3 outputs are preserved. Fourteen original source trajectories crossed development and test roles, so those outputs are development diagnostics. See [the shared audit](../continuation-audit/shared_protocol_audit.md).

Fresh Walker collection contains 2,999,673 training steps, 300,347 probe steps and 300,038 root-source steps. Episode roles and seed ranges are disjoint. The dataset is backed up at private revision `ddd4d51648d2a382725efd20d04cedebedbf8fd2`.

The 1,000-step LeWM smoke run passed nonblack rendering, whole-episode split and both loss checks. Mean prediction loss fell from 0.41495 to 0.22369; SIGReg fell from 16.86563 to 3.80469. The measured rate was 2.214 iterations/s, implying about 27 hours for 214,780 optimizer steps under comparable conditions. Full ten-epoch training is running with the declared upstream architecture, batch and loss recipe. Actual completion and S3 probe outcomes remain pending.

LeWM recovery checkpoints are uploaded every 4,000 optimizer steps. Each contains one atomic recovery bundle with model, optimizer, scheduler, random states and a reproducible training-batch cursor, alongside source snapshots and model exports. The interval was increased from 2,000 to fit both full model histories within verified private storage while retaining a 10 GB reserve. The full-run preflight projects 44.94 GB for both checkpoint histories including overhead, with about 30.72 GB remaining beyond the reserve. A recovered run must use a fresh run ID and identical data, split and recipe.

Push-T physical probes have been refitted with source-aware splits and uploaded at private revision `f03fb888ff15a4e757600a51c038a0988429c05c`. Fresh development, representative and stress banks are being generated with geometric start/goal families. Development feasible-route witnesses, the decomposition gate and full paired retention are prerequisites to repairability/acquisition claims.

Fresh CPU replay validation passed on 50 stored roots, including 16 roots with collision contact (the old counter includes walls). Repeated observed poses, block velocities, rendered frames and terminal masks agree bitwise. Ten stopped branches retain explicit censored suffixes. Geometry was cross-checked against an independent implementation; all ten contact overlays passed the separate [visual review](e0-visual-review.json). The old contact counter is not evidence of pusher–block contact specifically, so that mechanistic stratum uses the new body-specific observer. A separate [contact coverage check](e0-contact-coverage-combined.json) adds one recorded-action root: 51 roots from 50 source episodes, with 15 roots showing pusher–T contact during the last prefix block and nine at its final step. All three repeats agree bitwise; the original 50-root qualification is preserved. This validates deterministic replay of the recorded float32 tapes, not equivalence to the original unquantized action arrays.

The original E3 child finished as a development diagnostic. Its automatic queue and the stale Walker queue were then retired, after confirming neither had a live child, because they would launch stale-data or ungated downstream work. See [the retirement record](original_queue_retirement.json). The corrected workflows use finite subprocesses, private upload barriers, exact output hashes and explicit scientific gate stops.

Current machine-readable status is in [continuation_status.json](continuation_status.json). Live stage states and logs are under ignored `runs/workflows/`; scientific results, manifests and pinned receipts are retained in this results tree and private Hugging Face repositories.

The qualified post-E2 continuation is committed in `88f8dea`, with 221 repository tests
passing and Ruff clean. Its managed service, `safetydial_pusht_e3_continuation`, is waiting
for E2 and the exact prerequisite artifacts. It schedules three paired acquisition seeds,
all four additional budgets, final-checkpoint retention and the geometric transfer report.
E5 remains conditional on repeatable gains and retention for the preselected checkpoint.
See [the continuation audit](../continuation-audit/remaining_requirements.md).
