# Phase-two evaluation tooling

These modules support [phase2.md](../../../phase2.md) and have been checked with
synthetic records. No optimizer results are available. The AAC hook now executes the
actual generated benchmark loop with fake clients/environments in tests; parent live
authorization and complete main outcome auditing remain unwired at this checkpoint.
These tools do not establish a performance improvement or safety margin.

## Pilot comparison and model handoff

[`maniguard_pilot_report.py`](../../../experiments/helpers/maniguard_pilot_report.py)
reads saved coordinator state without resuming it. `build_report` reconstructs each
run's validation selection: higher family-macro utility, then lower violation rate,
lower attempted query cost, then earlier nominee. It retains both seeds for evolution,
random search and PPO, reports per-family `S_req`, `C_req`, `V` and `U`, and compares
against identity, fixed and contextual references. Shared reference records count once
in actual costs. Missing work, invalid attempts, unknown costs and regressions remain
visible; episode-time sums are not allocated GPU reservation time.

The [CLI](../../../experiments/scripts/maniguard_pilot_report.py) writes `report.json`,
`report.md` and `sources.json` into a new directory. For a saved pilot state:

```sh
uv run --no-project --python data/maniguard/envs/sim/bin/python python experiments/scripts/maniguard_pilot_report.py \
  --state /path/to/coordinator-state.json \
  --output runs/maniguard/phase2/pilot-report --export-models
```

Optional `--completion-review` and `--mechanism-review` flags accept JSON reviews.
`--export-models` requires complete, consistent selection and exports all six actors
under `selected-actors/`; it never chooses only the best seed. Each model is loaded
again to verify its canonical hash. The export includes the source state, report,
per-run selection records and a manifest with `training_runs` and `selected_actors`
for the main planner. Existing runs and models are preserved.

Screening remains descriptive: two optimizer seeds and six validation cases per actor
cannot establish superiority, unchanged risk or generalization. The helper checks saved
state/source/schedule hashes and selection summaries; it does not independently replay
optimizer archives, audit physical evidence or revalidate the entire freeze chain.
Export and review inputs do not authorize a main study.

## Main evaluation planning

[`maniguard_main_protocol.py`](../../../experiments/helpers/maniguard_main_protocol.py)
builds an explicit common case grid for every declared selected run and fixed comparator.
It binds checkpoints, preprocessing, implementation, deployment, analysis and exposure
metadata, including Lid's design exposure and Cabinet/Dusty's reserved status. There
are no automatic sample sizes, meaningful-effect thresholds, risk margins or GPU caps.
A draft cannot authorize an episode; a freeze requires numerical settings and hashed
review evidence. Hashes establish integrity, not independent scientific review.

Validated controller resource contracts replace a shared query-cost assumption:

| Controller | Normal minimum actions between query groups | Policy RPC proposals per group | Supplementary CLIP |
| --- | ---: | ---: | --- |
| Unchanged benchmark executor | 8 | 1 | None |
| Learned / contextual | 4 | 1 | Every 4 actions |
| Identity | 8 | 1 | Every 4 actions |
| Fixed choices | 4, 8 or 16 | 1 | Every 4 actions |
| AAC-joint | 2 | 20 | None |
| AAC fixed-eight control | 8 | 20 | None |

Normal query bounds require sufficiently long native chunks. Singleton exhaustion can
require one query group per action, including all 20 AAC proposals. Each CLIP execution
is one image batch containing two views; phase-two controllers also budget one text
call and encoder-load attempt per episode. The unchanged benchmark comparator is
separate from an identity adapter: it pays no supplementary CLIP or adapter-decision
cost. Its native eight-action prefix and single policy stream are explicit.
VLA-internal encoding and denoising are
accounted through policy RPCs and timing, separately from supplementary CLIP.

The planner derives envelopes per controller for all 11 work counters, including
encoder attempts and loads. Total caps must additionally fund explicit `prior`,
`startup` and `failed_work` allocations. These allocations cannot add replacement
episode slots or startup/failed policy actions. Measured GPU costs and a durable live
launcher enforcing exact jobs, unique claims and cumulative actual costs are still
required; `authorize_job` alone checks membership, not those live limits.

## Protocol-aware main analysis

[`maniguard_main_analysis.py`](../../../experiments/helpers/maniguard_main_analysis.py)
provides `analyze_protocol(protocol, audited_rows)`. It validates the frozen protocol
and running analysis source, derives the full expected grid and declared bootstrap
settings, and retains the complete job bindings and exposure metadata. Rows must match
every compiled job field, including resource contracts, carry the protocol and pinned
reader hashes, and provide an output path, all 11 cost counters, elapsed time and explicit
cost-audit lists. Missing families or runs, mismatched bindings, invalid outcomes or
unresolved costs withhold performance comparisons while retaining known work.

The estimand weights families equally, base scenes equally within families, declared
cases equally within each scene, and training runs equally. Fixed references are reused
without duplicating physical observations or costs. Descriptive resampling draws base
scenes within each family and independent runs within each method; one actor stays
together across families, and scene draws are shared between methods. Variants and
policy seeds remain in their scene cluster. Matching seed labels does not establish
identical physical starts. Optimization and validation selection are not rerun.

Violation intervals at observed boundaries, including zero events, are withheld. The
report retains the declared risk margin but does not execute safety-margin,
meaningful-effect or multiple-comparison decision rules. It trusts the pinned reader's
physical audit and does not enforce live budgets. The lower-level `analyze` API remains
available for an independently supplied grid; main reporting should use the wrapper.

## AAC-joint proposal sampling

[`maniguard_aac_sampling.py`](../../../experiments/helpers/maniguard_aac_sampling.py)
adds sequential proposal sampling to the
[pure joint-space selector](../../../experiments/helpers/maniguard_phase2_aac.py).
Sample zero retains the original policy stream; extra samples use independent persistent
streams with identical copied observations. Measured joints are recorded before sampling.
The returned prefix preserves sample zero exactly. Gripper entropy and movement use the
benchmark's open condition `command > 0.01`, without rewriting raw commands. The fixed-eight
control generates the same 20 proposals to expose the cost of sampling separately from
adaptive scheduling.

Receipts distinguish durable dispatch intents, entered RPC attempts, returned proposals,
valid prefixes and unknown generation after failures. A partial failure stops the sampler
without an automatic retry; pending dispatch cannot imply zero server work. The new
[`maniguard_aac_runtime.py`](../../../experiments/helpers/maniguard_aac_runtime.py)
persists per-RPC intent/return receipts and accessible proposal arrays, and independently
replays the selected prefix from those artifacts. The
[source hook](../../../experiments/helpers/maniguard_aac_source.py) replaces the actual
query boundary, avoiding an extra ordinary query, and bypasses the native fixed-eight
execution length for adaptive prefixes. One sampler persists throughout an episode.
Actual generated-loop tests cover singleton, two-action and fixed-eight cadence, exactly
20 RPCs per group, no intervening simulator action, native gripper/clipping semantics,
zero supplementary CLIP, failures and tampering.

Sequential requests have no demonstrated batched latency or physical-task result.
The parent claim/lease/budget gate, full outcome/process/physics audit, frozen AAC settings
and costed action-bearing integration remain outstanding. See the
[primary-source review](ManiGuard%20phase2%20related%20work.md) for the scope of this adaptation.


## Durable dispatch accounting

[`maniguard_main_dispatch.py`](../../../experiments/helpers/maniguard_main_dispatch.py)
provides a synthetic-backend dispatcher around exact frozen-protocol jobs. It creates
one canonical study directory, binds protocol/prior/source hashes, persists a unique
claim before calling a backend, reserves all eleven counters, and keeps prior/startup/
failed work separate. Continuous GPU reservation includes idle time and restarts.
Unknown costs and in-flight claims block further dispatch; a new output directory cannot
reset an already claimed study. Fixed comparators retain `actor_sha256=None`.

A failed state write raises `DispatchPersistenceError`, poisons the current handle and
prevents further callbacks or a misleading success report. Best-effort diagnostics retain
uncommitted evidence separately, and unresolved diagnostics also block reopening. Tests
cover claim, backend, terminal, blocked-state, extra-work and synchronization failures.
This module has 55 passing synthetic tests plus an independent reproduction of the
original write-failure bug. It does not yet authorize a real simulator backend; AAC and
the original benchmark profile remain excluded from this synthetic dispatcher's supported
profiles pending their explicit live integration.

## Pilot cost histories and figures

[`maniguard_pilot_history.py`](../../../experiments/helpers/maniguard_pilot_history.py)
builds histories from saved coordinator state. A search point covers one candidate's
three-family evaluation; a PPO point covers one balanced collection batch. Each point's
cost includes all earlier recorded work in that run, including interleaved candidates.
Final selected-validation points pay all 72 training and 12 nominee-validation episodes
per run. Shared reference episodes count once outside the individual run axes.

The four recorded axes are policy actions, attempted policy queries, physics ticks and
episode elapsed seconds. Missing, failed, duplicate or unrecognized work cannot become
free progress: known costs remain lower bounds, and unresolved dependencies suppress
later exact axes. Changing training panels are shown separately; raw observations are
not turned into a monotone best-so-far curve. PPO collection is labeled stochastic and
validation deterministic. Selected validation remains selection-biased.

The CLI captures source-state bytes and writes a hashed history plus three PNG/PDF
figure pairs into a new directory:

```sh
uv run --no-project --python data/maniguard/envs/sim/bin/python python experiments/scripts/maniguard_pilot_figures.py \
  --state /path/to/coordinator-state.json \
  --output runs/maniguard/phase2/pilot-figures
```

Invalid histories are saved without figures. Valid incomplete snapshots are visibly
labeled. Figures show training utility/violations and selected-validation safe completion/
violations against paid action and query costs, retaining every seed. They provide no
confidence intervals or superiority conclusion. Episode-time axes do not estimate GPU
reservation time; preparation, startup, idle time and optimization overhead require their
separate records. Synthetic previews were visually checked and are not empirical results.
