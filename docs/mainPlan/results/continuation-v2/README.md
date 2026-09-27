# Corrected continuation evidence

Status: in progress. Experimental success is not assumed. Negative scientific gates stop downstream claims and experiments.

Original E1, E2 and E3 outputs are preserved. Fourteen original source trajectories crossed development and test roles, so those outputs are development diagnostics. See [the shared audit](../continuation-audit/shared_protocol_audit.md).

Fresh Walker collection contains 2,999,673 training steps, 300,347 probe steps and 300,038 root-source steps. Episode roles and seed ranges are disjoint. The dataset is backed up at private revision `ddd4d51648d2a382725efd20d04cedebedbf8fd2`.

The 1,000-step LeWM smoke run passed nonblack rendering, whole-episode split and both loss checks. Mean prediction loss fell from 0.41495 to 0.22369; SIGReg fell from 16.86563 to 3.80469. The measured rate was 2.214 iterations/s, implying about 27 hours for 214,780 optimizer steps under comparable conditions. Full ten-epoch training resumed from its verified 172,000-step checkpoint after an instance restart on 27 September, using the declared upstream architecture, batch and loss recipe. Actual completion and S3 probe outcomes remain pending.

LeWM recovery checkpoints are uploaded every 4,000 optimizer steps. Each contains one atomic recovery bundle with model, optimizer, scheduler, random states and a reproducible training-batch cursor, alongside source snapshots and model exports. The interval was increased from 2,000 to fit both full model histories within verified private storage while retaining a 10 GB reserve. The full-run preflight projects 44.94 GB for both checkpoint histories including overhead, with about 30.72 GB remaining beyond the reserve. A recovered run must use a fresh run ID and identical data, split and recipe.

Push-T physical probes have been refitted with source-aware splits and uploaded at private revision `f03fb888ff15a4e757600a51c038a0988429c05c`. The first corrected bank build exhausted its finite test/familiar source pool at 26 accepted roots, before the requested 64. Its partial banks are preserved and cannot support a completed study. See [the failure record](bank_generation_failure.json). [Continuation-v3](../continuation-v3/README.md) completed development feasibility and all 128 representative test roots, then its stress proposal search exhausted a finite draw limit. [Fresh stress-only recovery](../continuation-v4/README.md) completed with those exact roots and unchanged proposal primitives and arena guards; the original partial stress bank and its charged costs remain preserved.

Fresh CPU replay validation passed on 50 stored roots, including 16 roots with collision contact (the old counter includes walls). Repeated observed poses, block velocities, rendered frames and terminal masks agree bitwise. Ten stopped branches retain explicit censored suffixes. Geometry was cross-checked against an independent implementation; all ten contact overlays passed the separate [visual review](e0-visual-review.json). The old contact counter is not evidence of pusher-block contact specifically, so that mechanistic stratum uses the new body-specific observer. A separate [contact coverage check](e0-contact-coverage-combined.json) adds one recorded-action root: 51 roots from 50 source episodes, with 15 roots showing pusher-T contact during the last prefix block and nine at its final step. All three repeats agree bitwise; the original 50-root qualification is preserved. This validates deterministic replay of the recorded float32 tapes, not equivalence to the original unquantized action arrays.

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

A [30-sample throughput audit](throughput_audit_20260926.json) on 26 September
measured 99.17% mean GPU activity while Walker training, EGL image rendering and
Push-T bank generation shared the device. Memory use was 15,569 of 32,607 MiB;
spare memory did not indicate idle compute. CPU use averaged 8.32 cores within the
15.36-core quota, with no throttling. Both workers advanced during the observation.
Training settings were left intact. More render workers remain a candidate for a
controlled checkpoint-resume benchmark if sustained idle time or directly measured
loader waits justify it. GPU activity alone does not measure compute efficiency.

A bounded [lossless image-cache feasibility check](walker_cache_feasibility_20260926.json)
then tested 39 codec configurations on fixed training-only samples. The 256-frame
per-image sample matched across fresh renderers and reversed render order; all codec
round trips in both the per-image and temporal samples were exact. The best per-frame scheme projected 54.12 GiB, while the smallest tested
16-frame scheme projected 50.41 GiB before indexing and padding. With about 54 GiB
free, this leaves too little margin for sampling uncertainty and later artifacts.
Warm-memory decoding also does not establish end-to-end training speedup. No full
cache, trainer change or restart was launched; the bounded measurements and scripts
are retained locally with hashes.

A [fresh concurrency sample](gpu_training_concurrency_20260926.json) during E2
adaptation training measured 98.97% mean GPU activity across 30 samples (92-100%),
with 16,333 of 32,607 MiB used. Walker and Push-T remained active together. This
short window still does not measure arithmetic efficiency, but it provides no
evidence of spare compute for a third large workload. Training recipes stayed fixed.

Push-T E2 has now [stopped at its scientific gate](../continuation-v4/README.md):
matched test safety rates were undefined and exact goal retention was incomplete.
Its acquisition and oracle stages did not start. Walker full training continues.
A [30-sample measurement after Push-T stopped](gpu_after_pusht_gate_20260927.json)
found 98.03% mean GPU activity (89-100%), with 14,778 of 32,607 MiB used. No third
heavy workload or change to the frozen training recipe was justified by that window.

The [27 September interruption audit](walker_interruption_20260927.json) found
strong evidence of a kernel/instance restart at 06:29:43 UTC. The infrastructure
trigger remains unknown. The original trainer last logged update 174,400; its
172,000-step atomic checkpoint passed [34 CPU recovery checks](walker_checkpoint_172000_recovery_audit.json).
Model exports, Adam state, scheduler, random states, splits and the deterministic
next batch agree. The [recovery preflight](walker_recovery_preflight_20260927.json)
also verified the original dataset bytes, package versions, passing smoke gate
and sufficient private storage.

A fresh managed run, `walker2d-lewm-a-recovery-20260927-1`,
[restored CUDA state and advanced training](walker_recovery_launch_20260927.json).
It repeats 2,400 logged updates lost after the last atomic checkpoint, then
continues toward the same 214,780-update total with the unchanged 4,000-update
validation/upload cadence. The old workflow and run remain untouched. The fresh
[recovery workflow](../../../../configs/walker2d/recovery-20260927-1.yaml)
retains the original S3 and S4 recipes and scientific gates under fresh output names.

A [30-sample check after recovery](gpu_after_walker_recovery_20260927.json) measured
98.47% mean GPU activity (87-100%) while the trainer advanced from 173,100 to
173,200. The twelve repeated update records from 172,100 through 173,200 matched
the original loss terms at their printed precision. This supports operational
continuity; it does not establish bitwise GPU trajectory equality or arithmetic
efficiency.

Before the resumed workflow reaches S4, its [saved-row export gap was fixed](walker_s4_saved_rows_validation.json).
The queued study now preserves both already-computed baseline decomposition rows
and no-update evaluation rows in its run, results and private bank payload, with
SHA256 references. A failed development gate exports development rows only.
The change adds no model inference, simulation or random draws; 26 focused
export and paired-estimand tests and an independent diff review passed.

The [S3 presentation utility](walker_s3_presentation_validation.json) is prepared
for the eventual probe result. Fourteen focused synthetic tests, independent review
and synthetic figure inspection passed. It preserves the recorded gate and
undefined values, and requires finalized, unchanged workflow outputs before
creating a fresh sibling report. No actual S3 presentation has been generated yet.

After the probe stage is finalized:

```bash
uv run python experiments/scripts/walker_s3_figures.py \
  --run-dir runs/walker2d-probes-recovery-20260927-1 \
  --results-dir docs/mainPlan/results/s3/walker2d-probes-recovery-20260927-1 \
  --output-dir docs/mainPlan/results/s3/walker2d-probes-recovery-20260927-1-presentation \
  --workflow-state runs/workflows/walker2d-recovery-20260927-1/state.json
```

A later [metadata correction](walker_s4_seed_metadata_validation.json) makes each
future adapted checkpoint record its actual acquisition seed. Adaptation already
uses seeds 0, 1 and 2 correctly; only its saved template had retained seed 0.
The saved and executed configuration expressions now agree, with no change to
training, selection, random draws or gates.


## Conditional S5 recovery workflow

**Excluded by the user's revised scope on 27 September:** finish through S4, then
produce the PDF report and preserve results/media in GitHub and private Hugging Face.
The preparation below is retained as an unlaunched record, not remaining work.

The [fresh S5 configuration](../../../../configs/walker2d/recovery_s5-20260927-1.yaml)
is prepared but **not launched**. Its [validation record](walker_s5_recovery_config_validation.json)
checks the original comparison recipe, all four command parsers, a dry run, fresh output
identities, and immediate stops on either negative upstream gate. The recovery names
refer to the new A/S3/S4 runs; the source dataset and all sampling and training controls
are unchanged. The set-B writer uses `--n 92701` only to distinguish its date-derived
manifest/report identity.

Launch remains conditional on completed, audited S3/S4 evidence: full A training,
passing development diagnosis, all planned S4 budgets and three paired acquisition
seeds, equal charged costs, and verified checkpoint/bank provenance. A measured
boundary advantage is not an extra S5 prerequisite. S5 compares exactly 51,200 acquired
transitions in each arm, excludes the common S4 seed from the exact-N adaptation
comparison, and trains B for the same 214,780 updates. This preparation records no
S5 experiment or scientific result.


## S4 report preparation

The saved-evidence [S4 presentation utility](../../../../experiments/scripts/walker_s4_figures.py)
is ready for a terminal S4 stage. It supports a development diagnostic stop or a
completed health-only/health-and-speed comparison, with readable counts and costs,
per-seed paired FSA intervals, and PNG plus vector PDF figures. Undefined outcomes
remain undefined. It verifies frozen output identities and writes only to a fresh
sibling directory. [Preparation validation](walker_s4_presentation_validation.json)
passed 16 focused tests, Ruff, source review and synthetic visual review. No actual
S4 result has been plotted or inferred from these layout tests.


## Post hoc data alignment and original provenance

A [bounded saved-data alignment audit](walker_s1_alignment/README.md) checked fixed
head/tail windows from the first four training and probe episodes, without inspecting
final-test roots. All 384 sampled health and cost labels matched their definitions;
376 successor-based velocity comparisons matched exactly. Final post-action states
were not retained, leaving eight velocity and four benchmark health-termination
comparisons unavailable. Seven focused tests and visual review passed. This is post
hoc evidence, not a claim that the planned review preceded pretraining.

The corrected plan paragraph distinguishes pre-action health from transition cost
and velocity. Probe data retain a copied termination-on render-context metadata flag,
although collection disabled health termination; the audit documents both facts
without changing data. The [S0/S1 evidence index](walker_s0_s1_provenance.json) links
the existing observability results, source data manifests and private upload receipt.
Unknown original S0 seeds, split identities and executed source bytes remain unknown.
