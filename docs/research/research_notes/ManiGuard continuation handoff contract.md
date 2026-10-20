# ManiGuard continuation handoff contract

Prepared on 20 October 2026 for a small prospective engineering screen. The CPU
selector and factual-prefix replay are implemented. Physical integration, RNG
continuity and a launch protocol remain execution-owned. This proposal preserves
the original native-noise comparison and its sixteen independent checking episodes.

## Four reference conditions

Freeze one closure controller from exposed training data and use that same matrix
in all relevant conditions. Record its selection rule, prior information and
acquisition cost. `random-g1-2` is the earlier observed closure example, while
the retained `ea-g1-3` also closes the lid. Neither completes the task. Choosing
between them is development selection, not evidence of optimizer superiority.

| Reference | Controller choice | Actor information |
| --- | --- | --- |
| Native only | Native from the first query | Native causal observations |
| Closure only | Fixed chosen closure matrix throughout | The existing 44 causal features |
| Clock handoff | Chosen matrix, then native at a frozen query step | Causal features and query clock |
| Privileged lid handoff | Chosen matrix, then native after supported closure | Causal features plus actual monitor labels |

All conditions retain the same policy/checkpoint, 640-pixel cameras, encoder
schedule, native Gaussian source, sixteen-action decoded chunk and eight-command
execution prefix. Native also pays the matched feature-extraction cost. Balance
worker assignments and retain initial robot/image hashes; common seeds alone do
not establish identical camera pixels. Freeze cases, fresh seeds, clock step and
execution order before feedback. The privileged reference tests feasibility and
data acquisition, rather than learned-method or safety-filter performance.

## Switching and reset semantics

The proposed debounce is three consecutive post-control observations with closed
lid, support and no preceding raw violation. Initial observation zero does not
count. Qualification cancels if closure/support is lost before the next query.
The selector switches only at the next eight-control query boundary, then remains
latched to native even if closure is later lost or a violation occurs. The monitor
continues logging; neither violation nor stalled progress shortens a valid trial.
Run to the declared 2,000-control horizon or legitimate native termination.

Every new episode resets query history, debounce, violation and branch state.
Partial physical failures stay charged and invalid. This selector has no API for
resuming an interrupted physical episode as a valid trial. A selected native branch
calls the unchanged inference path with no explicit noise argument. It does not
submit zero noise, reset policy RNG or buy a second decoder call at the switch.
The CPU object has no policy RNG handle; actual continuity requires runtime proof.

## Physical receipts and validity

Freeze the compound selector contract and both component matrices by hash, with
all simulator/policy/controller/encoder sources and environment pins. A new
reader must validate the complete query sequence, actual monitor timestamps,
debounce state, at most one branch transition, selected component, noise offset
and policy RNG before/after every request. The first native query must continue
the same stream as the preceding closure query. Log request admissions and returns
so exceptions and interrupted calls remain charged. The existing whole-episode
constant-genotype receipt cannot validate this hybrid by assumption.

Retain proposals, decoded chunks and actual environment-input histories, raw
violations, requested/raw completion and unfinished outcomes. Ordered closed,
supported body grasp, later closed lift and transport are secondary progress
labels, removed from terminal progress fitness after any violation. None replaces
the existing requested-completion reader. Preserve all outcomes, full source/data
hashes and private reconstruction receipts. No learned predicate or risk score
is qualified by the current model audit.

## Small staged allocation proposal

The first screen is capped at four full starts, including partial and failed
starts. If all are valid, these cover the four references on one exposed
development case with a shared fresh seed. Expand by a separately frozen stage for the same references
on a second case only if at least one handoff produces a supported, closed body
grasp in the required order and finishes without a raw violation, or achieves
requested safe completion. A valid timeout is an outcome. An invalid screen must
be diagnosed before further admissions; replacement slots do not permit retuning.

After a passing screen on both cases, a four-episode confirmation may compare
native with the selected nonprivileged reference on two fresh task/seed blocks.
Require the selected nonprivileged reference itself to have reached safe ordered
body grasp or requested safe completion before spending confirmation trials.
If only the privileged reference progresses, retain that feasibility evidence
and diagnose the phase detector rather than confirming inactive fair references.
Freeze selection before screening: maximize summed requested-safe-completion
utility; break ties by safe ordered stage and then a declared identifier. Keep
the privileged reference outside this fair-information selection. This small
confirmation is descriptive and does not establish evolutionary advantage or a
population safety rate.

The proposed staged ceiling is twelve valid episodes plus two infrastructure replacements,
fourteen total starts. It is conditional preparation, not an already launched
allocation or an exception to the first four-start screen. The first four starts use at most 8,000 policy controls
and 1,000 policy queries. Including every possible invalid full-length attempt,
the full ceiling is 28,000 policy controls, 3,500 policy queries, 7,000 encoder
images and fourteen text encodings. Initialization and settling need separately
frozen numeric bounds on all fourteen starts; physical ticks/allocated time are
separate counters. Do not call these policy-control limits total physics limits.
No extra decoder sweep is needed. This is a smaller alternative to the older
16-plus-two proposal, not an additional allocation or automatic dispatch.

Before launch, execution must freeze complete source, task/seed, validation and
work limits, then verify actual resource and time capacity through preservation.
The user supplied an eighteen-hour result window; review relayed a conservative
21 October 03:35 UTC delivery deadline. A provider/shutdown extension is not
established by that instruction. Use at least the nondecreasing latest conservative
wave bound and retain confirmation/preservation time before admitting more work.
Stop at a failed progress stage or insufficient capacity. Do not start a broad
EA sweep to consume remaining time. These trials cannot meet the learned
filter's minimum twenty calibration episodes.

## Completed CPU evidence

Thirteen focused cases check exact original closure codes/offsets, untouched
NumPy RNG, debounce, reopening, raw-violation latch, query boundaries, per-episode
reset, privilege exclusion and rejection of missing histories. Replay stops at
the first changed controller choice and does not parse later original rows as
hybrid evidence. Existing full receipts remain the factual integrity authority.

The saved random-controller case0 prefix qualifies at observation 484 and would
first select native at query 488. The saved EA case0 prefix qualifies at 1189
and would select native at 1192. Neither case1 qualifies. Both clock references
use 488 in this replay; that clock came from the earlier random training example
and is not tuned to each outcome. Closure-only rechecks all 250 recorded codes
per episode; native-only stops at query zero. These are switch timings and
numerical identities, with no hybrid grasp, completion or RNG-continuity result.

Exact reports are `runs/reviewer/handoff-replay-random-20261020-01.json` and
`runs/reviewer/handoff-replay-ea-20261020-01.json`. Their hashes are respectively
`ac8a19458f282998e9d4919f02b21f76d79f4069c909565ad8821251224ae9e4` and
`7fce9f908f2aaaa6482c8ae4ab3dce4ad7fa58582394cf3bcdbcae1be01070b0`.
The exact queue snapshot is `runs/reviewer/handoff-replay-queue-20261020-01.json`,
SHA-256 `a106722aaa9696bdee696230abd7337d5808b77e9cfe0a02508e5d67d013635f`.
All new simulator controls, VLA requests, encoder calls and learned-model queries
are zero. Source hashes and original verified job/receipt/trace hashes are retained
in each report. Existing independent checking outputs are excluded before opening.
