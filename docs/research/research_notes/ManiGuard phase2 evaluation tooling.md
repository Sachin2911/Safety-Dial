# Phase-two evaluation tooling

These four modules support [phase2.md](../../../phase2.md) and have been checked with
synthetic records. No optimizer results are available. Main-study live job gating is
unwired, and AAC is not connected to the evaluator. These tools do not establish a
performance improvement or safety margin.

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
| Learned / contextual | 4 | 1 | Every 4 actions |
| Identity | 8 | 1 | Every 4 actions |
| Fixed choices | 4, 8 or 16 | 1 | Every 4 actions |
| AAC-joint | 2 | 20 | None |
| AAC fixed-eight control | 8 | 20 | None |

Normal query bounds require sufficiently long native chunks. Singleton exhaustion can
require one query group per action, including all 20 AAC proposals. Each CLIP execution
is one image batch containing two views; phase-two controllers also budget one text
call and encoder-load attempt per episode. VLA-internal encoding and denoising are
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
without an automatic retry; pending dispatch cannot imply zero server work. The caller
must persist receipts and reconcile costs. Sequential requests have no demonstrated
batched latency or physical-task result. Live integration, frozen AAC settings and bounded
execution/accounting remain outstanding. See the
[primary-source review](ManiGuard%20phase2%20related%20work.md) for the scope of this adaptation.
