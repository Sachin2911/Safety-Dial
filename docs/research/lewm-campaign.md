# Safe Evolutionary LeWM campaign, 31 October 2026

Sachin authorized seven hours of persistent experimental work from 00:23:14 UTC
(02:23:14 SAST), ending 07:23:14 UTC (09:23:14 SAST), with a fresh shared $10 cap,
at least two parallel rented pods and up to three pods available. This supersedes
the completed screen's prohibition on further work, without changing its results.
The goal is a reproducible core evolved-controller result for the LeWM paper.

The [campaign controls](../../configs/lewm_campaign_20261031/campaign.json) fix
scope, deadline, scientific controls and accounting. Two A40 pods, PushT
`wq4ar2qx2ff6r4` and navigation `vukovc3t4o1dui`, were allocated at $0.59/hour each.
Provider credentials remain local. Both have an independent local provider-stop
watchdog and a credential-free remote process deadline guard. The third allocation
is reserved for recovery within remaining shared-budget headroom.

The [literature synthesis](reports/Evolutionary%20LeWM%20recovery%20strategies.md)
points to two concrete bottlenecks: inconsistent action projection between prediction
and execution, and final-goal latent objectives that do not reward necessary
intermediate progress. Initial diagnostics isolate action consistency and execution
timing. Any repair gain belongs to the contemporary native baseline, not evolution.

The two development lanes pursue compact persistent controllers: a structured
geometry-conditioned PushT residual and an observed-intermediate-target selector in
TwoRoom. Each stage is frozen before its eligible episodes. Arbitrary untrained
weights need not improve before learning; meaningful authority and a non-ceiling
fitness landscape are required. Training, random search and ordinary-learning
controls must share information and explicitly counted interaction budgets. Final
evaluation cases stay separate until controller and analysis choices are frozen.

This file records the active campaign, not an achieved evolutionary result. The
live local control and evidence roots are `outputs/runpod-lewm-campaign-20261031/`
and `outputs/lewm-campaign-20261031/`. Original physical goals, safety margins,
all failures and all simulator/model work remain explicit. New results and closeout
will be added only when supported by saved evidence.

## Development evidence through 02:12 UTC

The action-contract repair clips candidates before prediction, safety checking and
CEM fitting. The initial navigation diagnostic retained 20 starts, 2,712 executed
actions, zero safe completions and zero observed violations. The PushT diagnostic
retained 32 episodes and 3,749 actions. Each of four planning recipes completed 2/8
safely, entirely from the no-hazard controls. The five-block execution recipe used
less model work without improving completion.

PushT's new 24-coefficient structured residual passed its six-episode authority
stage, including exact native/zero-adapter trace parity. A real six-update CMA-ES
pilot and matched 48-candidate parameter random search each retained 196 episodes
and 23,128 actual actions. Each completed 49 no-hazard episodes safely and left all
147 hazard episodes unfinished, with no observed violations. Secondary progress
improved, without a completion improvement. Fresh development selection retained
16 episodes and 1,888 actions. Native, the fixed tangent controller and both trained
nominees each scored 1/4 safe completions, 0/4 violations, 3/4 unfinished and 4/4
engaged. Every success was the no-hazard control. Independent saved-evidence audits
verified source and nominee identities, seeded ask/tell replay, updates, actual
interaction and original-goal outcomes. This family has not established a useful
hazard-completion lever or evolutionary advantage.

Navigation's observed-target memory contains 3,210 real source/successor/endpoint
triples from public source episodes 128 through 175, encoded with the same frozen
FP32 LeWM and preprocessing as live queries. The development panels come from
disjoint source episodes; the reserved query pool remains unread. Native and two
fixed target selectors each scored 0/8 safe completions in both the small-search
and large-search pilots. All 48 episodes remained unfinished and engaged, with no
observed violations. They used 3,600 and 3,445 actions respectively; the difference
came from two retained checked-plan refusals, not successful task efficiency.

Saved traces showed repeated nearby target attractors and poor correspondence
between latent goal distance and physical progress. A separately frozen cached-data
diagnostic qualified a memory-trained linear visual-position readout. Its disjoint
eight-episode memory test had median error 2.70 px, 90th percentile 6.64 px and maximum
66.56 px. These errors do not establish a safety guarantee or accuracy on all query
episodes. A further diagnostic over all saved retrieval histories found goal-image
position errors of 1.51 to 12.28 px on the eight already used development cases.
No new encoder, predictor or simulator calls were needed for that analysis. The
readout is a shared supervised asset; its benefit cannot be attributed to evolution.

A NAV-only durable append-and-fsync recorder passed nine rented-host software
checks and a prospectively frozen four-episode, 120-action old/new qualification.
Both native and fixed-target pairs had exactly equal commands, states, images,
target histories and original count-valued model-work counters. Timing was lower
with the new recorder in these two pairs, a descriptive engineering observation.
The shared optimizer harness and old evaluator remained unchanged.

The next architecture pilot freezes a ten-gene observed-target selector with a
6 to 30 px target-distance band, causal repetition/stall features and progress
towards the frozen visual-goal estimate. Original task success remains distance
below 16 px; body radius 7 px, margin 3 px and learned motion allowance remain
unchanged. Native and the strong fixed controller use the same eight development
cases and planning budget, with a 16-start, 2,400-action maximum. Any gain from
these changes belongs to the shared architecture. The first invocation supplied an
unsupported command-line argument and stopped before creating an experiment output;
the corrected invocation uses the same immutable protocol and a fresh worker record.

## Qualified lever and frozen optimizer comparison

The ten-gene architecture pilot completed all 16 episodes. Strong fixed retrieval
scored 8/8 safe completions, 0/8 violations, 0/8 unfinished and 8/8 engaged in 322
actions. Native scored 0/8 safe, 0/8 violations, 8/8 unfinished and 8/8 engaged in
1,200 actions. Exact feature, mask, causal-history and original-goal replay passed,
with no target-support refusals. This establishes an architecture lever on the
development panel, not an evolutionary benefit.

Two new prospective development probes retained all eight cases and used fresh
seeds. The stress probe increased circle radii by a fixed metadata-only rule, capped
at 12 px while retaining the original buffered endpoint clearance. Strong fixed
scored 6/8 safe, 0/8 violations, 2/8 unfinished and 8/8 engaged in 592 actions;
native scored 0/8 safe and 8/8 unfinished in 1,200 actions. Endpoint admission does
not certify global route feasibility. Actual radii, rather than the parent radius
stratum labels, identify these hazards.

The separately frozen 50-action pilot kept original geometry and original physical
success, with a new budget-constrained endpoint. Strong fixed scored 4/8 safe, 0/8
violations, 4/8 unfinished and 8/8 engaged in 333 actions; native scored 0/8 safe,
0/8 violations, 8/8 unfinished and 8/8 engaged in 400 actions. Independent replay
verified every source, goal, target-score feature and all 8,459 durable journal
records. This probe changes seeds and physical host as well as horizon relative to
the 150-action pilot; it is not an isolated horizon ablation. Earlier 150-action
outcomes remain unchanged.

The [main comparison freeze](../../configs/lewm_campaign_20261031/navigation-budget50-comparison-v1/manifest.json)
uses the 50-action endpoint and original hazards. Its four training cases are the
first eligible case in each original condition, chosen without outcome-based
substitution. The qualification pilot scored fixed 3/4 versus native 0/4 on that
panel. Three independent learner repeats compare diagonal CMA-ES, parameter random
search and categorical REINFORCE over the same ten genes and causal information.
CMA and random search each have 48 candidate panels plus the shared anchor;
REINFORCE has 24 updates, each charged for stochastic training and deterministic
post-update evaluation, plus its anchor. Each method/repeat has at most 196 starts,
196 resets and 9,800 actual executed actions. A 48-setting all-nonzero fixed grid
and separate anchor have the same allowance. Used and unspent work remain separate.

Repeat 0 and repeat 2 use the navigation host; repeat 1 uses the former PushT host.
Every method, fixed search and native control for a repeat stays on its physical
host. The fresh training controls in repeats 0 and 1 each scored fixed 4/4 and
native 0/4. That observed seed-specific ceiling is retained; it does not authorize
changing the frozen panel or seeds. Eight disjoint development-selection cases
choose between each trained nominee and the shared anchor, with exact ties retaining
the anchor and no credit for selecting it. All three learner repeats remain in the
final comparison. Reserved source episodes 5000 through 5255 stay unopened until all
three selections are locked, after which a prospectively admitted 16-case final
panel will evaluate the six arms. The main optimizer runs are active; no evolutionary
advantage is established by this record.

## Broader training and final-data seal

A prospective 30-action pilot retained the same eight original cases and geometry,
using new paired seeds. Strong fixed scored 3/8 safe, 0/8 violations, 5/8 unfinished
and 8/8 engaged in 216 actions. Native scored 0/8 safe, 0/8 violations, 8/8 unfinished
and 8/8 engaged in 240 actions. Its independent audit passed. The
[secondary activation](../../configs/lewm_campaign_20261031/navigation-budget30-secondary-activation-v1.json)
and [comparison freeze](../../configs/lewm_campaign_20261031/navigation-budget30-comparison-v1/manifest.json)
were written before any final data access. This study trains on all eight original
cases, without outcome-based substitution. Each optimizer or fixed search repeat has
49 total panels, at most 392 starts/resets and 11,760 actual actions. The genuine
ordinary-learning objective uses the same eight-case ordering, with scalar return
9 times safe completion, minus violation, plus progress divided by 9. Primary and
secondary endpoints remain separate.

Secondary repeat 0 uses navigation; repeats 1 and 2 use the former PushT host, so
the two study queues share the available time. All methods within each repeat remain
on that repeat's host. The new joint-final guard requires all three primary and all
three secondary selections before any shared final metadata, pixels or primary final
outcomes. Five synthetic guard checks passed on the rented host. No existing primary
protocol was amended. The original final case-admission rules and 16-case quota remain
fixed.

An auxiliary 32-episode, 50-action development comparison tested a decoded task score
using the same frozen Ridge map on LeWM's imagined endpoint and the current planning
target. It added no encoder or predictor call and retained all four paired arms.
Original native scored 0/8 safe in 400 actions; original strong retrieval 8/8 in 277;
decoded native 5/8 in 233; decoded retrieval 3/8 in 349. All arms had zero observed
violations and full engagement. This result uses new seeds and remains a separate
paired probe. It supports retaining the original local latent metric, with the
decoded-native planner included as an auxiliary final baseline. Nine rented GPU
software checks passed; an earlier gate rejected an incorrect expected count of ten.
That gate, its original logs and the corrected metadata qualification are retained.

The first primary repeat has completed real CMA, random-search and ordinary-learning
runs, plus the fixed grid. CMA used 6,538 actions and six adaptation updates; its best
candidate was proposed after two updates. Random search used 6,969 actions and no
learning updates. Ordinary learning used 6,817 actions and 24 genuine updates; its
best nominee followed update four. All used their complete 196-episode schedules,
with no observed violations or technical failures. Their best candidates each
matched the anchor's 4/4 training completions, with different secondary progress.
The 48-setting fixed grid used 7,627 actions. These totals describe search work, not
fresh-policy efficacy. The six-arm, eight-case repeat-0 selection was frozen at
03:14 UTC and subsequently completed.

The final analysis is being frozen and software checked before outcomes. It keeps
all learner repeats and physical hosts, uses 16 source-case clusters, and reports
conditional descriptive intervals rather than treating repeated controllers as
independent task samples. No general optimizer-superiority inference follows from
nominal pointwise intervals. Models, source freezes and live evidence are mirrored
locally; independent guards retain the original 07:23:14 UTC computation cutoff.

## Development selections and prospective host relocation

Primary repeat 0 completed all 48 selection episodes in 2,058 actions. Native
scored 0/8 safe completions, strong fixed 4/8, tuned fixed 5/8, CMA 5/8, random
search 4/8 and REINFORCE 5/8. All arms were engaged with no observed violations.
CMA completed one additional circle-3 hazard case relative to the strong anchor,
with one paired win, no losses and seven ties. Tuned fixed and ordinary learning
also scored 5/8, so this small development result does not establish an
evolutionary advantage. The frozen selection retains CMA, random-search and
REINFORCE nominees. Random search ties the anchor on completions and exceeds its
secondary progress score.

Primary repeat 1 completed all 48 selection episodes in 2,059 actions. Native
scored 0/8 safe, strong fixed 6/8, tuned fixed 5/8, CMA 6/8, random search 4/8 and
REINFORCE 6/8. All arms were engaged with no observed violations. Only CMA's
nominee exceeded the anchor's full lexicographic tuple, through secondary
progress rather than an additional completion. The other search arms retain
their explicitly labelled anchor fallback. All three repeats remain required.

Primary repeat 2 subsequently completed 48 selection episodes in 2,014 actions.
Native scored 0/8 safe completions, strong fixed 6/8, tuned fixed 6/8, CMA 6/8,
random search 5/8 and REINFORCE 6/8. All arms were engaged with no observed
violations. CMA alone exceeded the strong anchor's full tuple and retains its
evolved nominee, proposed after four adaptation updates. Tuned fixed, random
search and REINFORCE retain the anchor. All three primary locks were sealed at
04:31 UTC; the joint six-lock condition still prohibits final-data access until
the secondary selections also complete.

At 04:04 UTC, a third pod was rented under the existing seven-hour, three-pod
authorization. It uses an RTX 4000 Ada at $0.28/hour. The projected combined rental
through the unchanged 07:23:14 UTC deadline is $9.21 including the storage reserve.
The entire unstarted secondary repeat 2 was prospectively relocated from the
former PushT host to this host. Its cases, seeds, action limits, scientific code,
optimizer settings and all local controls remain unchanged. An exact old-host
absence proof and the [relocation record](../../configs/lewm_campaign_20261031/navigation-secondary-r2-relocation-v1.json)
precede its first scientific episode. The [authoritative secondary manifest](../../configs/lewm_campaign_20261031/navigation-budget30-relocated-v1/manifest.json)
retains the original repeat-0 and repeat-1 references and moves all repeat-2 arms
together. No comparisons of elapsed performance across hosts are pooled.

The new host passed all 55 foundation software checks and lint before its
04:20:38 UTC control launch. The final-data seal was extended prospectively to
the new host mapping, still requiring all six primary and secondary selections
before any final case metadata or pixels. A synthetic integration check bound
all six final comparison profiles and five optional host-specific decoded-native
profiles to the frozen analysis, rejecting changed protocol bytes. The final
analysis keeps its original paired source-case bootstrap and explicitly
conditional descriptive intervals. No final task has been opened at this stage.

The recovery host's first 50 ordinary-learning episodes averaged 6.09 seconds
each. Four serial search stages would finish too late for final evaluation and
preservation. A saved resource snapshot showed only 0-5% GPU use, 336 MiB of
20,475 MiB GPU memory occupied, and ample spare CPU capacity. The cause of the
remaining latency was not established; logging durability and scientific sources
were left unchanged.

A new, prospectively frozen orchestration qualification replayed exactly the
first strong-control job in protocol order while the original learner remained
live. It used one additional reset and 19 actual actions within its 30-action
cap. At 04:37 UTC, all seven saved trace arrays matched exactly, including
proposed and executed actions, states and raw frames. Selected target identities,
deterministic result fields, journal semantic counts and all non-timing resource
counters also matched. Output-specific journal paths and hashes were explicitly
excluded from byte equality. The exact peer PID/start identity was observed
before, during and after execution. This is one bounded engineering parity
check, not a general determinism or speed result.

After independent review, a new recovery-only launcher permits at most two
scientific workers, naming the exact live peer and fresh frozen stage in a new
authorization before each start. All existing source, asset, software, deadline,
host and non-replay checks remain. No protocol budget, case, seed, source or fsync
rule changes. Additional engineering work is separately charged. Overlap intervals
are retained; elapsed optimizer speed comparisons are not valid under changing
shared-host load. Interaction and model-query costs remain separately reportable.

Secondary repeat 0 completed all 392 CMA episodes and six genuine adaptation
updates in 10,634 actions. Its best complete eight-case training panel scored
5/8 safe completions versus the anchor's 2/8, with no observed violations and
full engagement. This candidate was proposed in generation 5 after five completed
adaptation updates. The population's aggregate 116/392 completions describes
exploration, not fresh-policy efficacy. Random search started next on the same
host at 04:49:39 UTC. The separate selection and final panels remain required
before interpreting this training gain as generalization or an optimizer advantage.

On the same repeat-0 host and eight-case panel, random search subsequently
completed all 392 episodes and 48 candidate panels in 10,819 actions. Its best
training panel scored 3/8 safe completions, versus CMA's 5/8 and the shared
anchor's 2/8. All 392 random-search episodes were engaged with no observed
violations or technical errors. Its population total was 85/392 safe completions,
compared with CMA's 116/392. These dependent training populations describe search
behavior, not a fresh-policy comparison. Ordinary learning started next at
05:08:43 UTC; fixed tuning, disjoint selection and final evaluation remain required.

The controller is state-aware: causal current body XY and the supplied virtual
hazard geometry support local target admission and scoring. The actor receives
the original goal image and its frozen visual representation; the shared frozen
Ridge readout estimates goal position. True final-goal XY is used for reset,
admission, monitoring and common training fitness, not supplied to the actor.
The ten coefficients are compact additions to a shared model, readout and memory
bank, not a claim that the entire deployed system occupies ten parameters.

Ordinary learning in secondary repeat 0 finished all 392 episodes and 24 updates
in 10,911 actions. Its best deterministic training panel remained at 2/8 safe
completions, with a higher secondary progress score than the anchor. The saved
evidence audit passed all episodes; there were 90/392 population completions,
no observed violations, full engagement and no technical errors. The fixed grid
started at 05:26:58 UTC. These training outcomes do not replace the separate
development-selection comparison.

Secondary repeat 1's completed and audited CMA search used 10,188 actions and six
adaptation updates. Its best training candidate scored 6/8 safe completions,
versus the shared anchor's 2/8, random search's 4/8 and REINFORCE's 3/8. This
candidate was proposed in generation 5 after five completed updates. CMA's
population total was 147/392 safe completions, with no observed violations, full
engagement and no unresolved work. The fixed grid started at 05:10:31 UTC. The
separate selection and final outcomes remain unopened at this stage.

Secondary repeat 2's CMA run completed and passed its saved-evidence audit with
392 episodes, 10,445 actions and six exactly replayed updates. Its population
scored 143/392 safe completions, with no observed violations and full engagement.
The best training panel scored 4/8 versus the anchor's 2/8, but its candidate was
generation 0, candidate 4, proposed before any update. No later adaptation update
receives credit for this genotype. This repeat remains in the comparison. Its
ordinary-learning run used 10,892 actions and 24 genuine audited updates, with a
best training panel of 3/8 after 17 updates and population total 105/392 safe.
Both remaining recovery searches continue under the qualified overlap rule.

The first two secondary selections sharply reduced the training signal. Repeat 0
completed 48 episodes in 1,419 actions: native 0/8 safe, strong 0/8, tuned fixed
0/8, CMA 1/8, random search 2/8 and REINFORCE 1/8. All were engaged with no
observed violations. Its lock retains the three searched/learned nominees and
the tuned setting, including progress-based retention of the latter. Repeat 1
used 1,414 actions: native 0/8, strong 1/8, tuned fixed 1/8, CMA 1/8, random
search 2/8 and REINFORCE 2/8, again all engaged with no observed violations. CMA
falls back to the untrained anchor in repeat 1. No evolutionary benefit can be
inferred from its anchor policy. Both locks were fixed before final-data access.

A separately frozen, exploratory repeat-0 gene-reversion study used the same
eight training poses with fresh paired seeds. It retained the full CMA nominee,
anchor and all ten one-gene reversions, irrespective of performance. All 96
episodes completed in 2,658 actions with no observed violations and full
engagement; the saved-evidence audit passed. The full nominee and anchor each
scored 2/8, compared with the nominee's original 5/8 training score. Reversions
00, 01, 02, 03, 04, 05, 06, 07, 08 and 09 scored 3, 1, 2, 4, 3, 2, 3, 4, 4
and 2 out of eight respectively. These descriptive results show seed sensitivity
and functional coefficient effects, not reliable learned benefit. No reversion
is substituted into the final comparison. All twelve arms and ten paired
contrasts are exported with denominator eight and without significance claims;
both figures passed visual review.

Before final-data access, a separate exploratory terminal-mean sidecar was
authorized in response to repeat-1 selection regression and repeat-2's
generation-0 archive winner. The hypothesis and uniform mapping were chosen
before the gene-reversion report was read. All three secondary runs' final CMA
distribution means were extracted on their owning hosts after six completed
updates, without ask, tell, sampling, fitting or new development episodes. Locks
were frozen at 05:54:04, 05:54:42 and 05:54:41 UTC, with checkpoint, state,
update, package and source hashes and final-path absence proofs. Each uses
`es.gp.pheno(es.mean, into_bounds=es.boundary_handler.repair)`; the mapping is
identity in these three runs. No mean, mapping or repeat is selected by outcome.

The sidecar retains all three locked means, including any duplicate or
anchor-equivalent policy. After shared final-case admission and gallery review,
it adds one 16-case, 30-action profile per host, at most 48 starts and 1,440
additional actions. These profiles use the ordinary secondary profiles' exact
cases, seeds, runtime, assets and safety definitions. The original eleven-profile
manifest, six selection locks and frozen v3 analysis remain unchanged. Separate
per-host and three-repeat descriptive reports compare every mean against all
five fixed/selected baselines, without confidence intervals, p-values,
confirmatory or optimizer-superiority claims. Main and auxiliary outcomes are
retained regardless of their direction; all additional work is separately charged.

## Final evaluation completed on navigation

All six ordinary selection locks and all three exploratory terminal-mean locks
preceded the first final-data access, at 06:21:19 UTC. The shared admission rule
selected sixteen source cases, four each in the no-hazard and three circle-radius
conditions. Native collection-time render comparisons had zero mean absolute
pixel difference. The complete sixteen-case gallery passed visual review before
the first final controller episode. Controller and analysis sources, goals,
geometry, margins, cases and seeds remained frozen.

The navigation host completed three ordinary final profiles and its separate
terminal-mean profile, totaling 304 episodes. Each ordinary profile evaluates six
arms on the same sixteen source cases. Primary repeat 0, at fifty actions, scored
native 2/16 safe completions, strong fixed 12/16, tuned fixed 11/16, CMA 12/16,
random search 13/16 and REINFORCE 12/16. The corresponding actual actions were
757, 486, 489, 456, 476 and 484. Primary repeat 2 scored 2/16, 12/16, 12/16,
10/16, 12/16 and 12/16, using 757, 486, 486, 515, 486 and 486 actions. Its
retained CMA nominee followed four adaptation updates; the tuned, random and
ordinary-learning arms explicitly retain their anchor fallbacks.

Secondary repeat 0, at thirty actions, scored native 1/16, strong fixed 8/16,
tuned fixed 8/16, CMA 8/16, random search 9/16 and REINFORCE 9/16. Actual actions
were 466, 387, 383, 387, 364 and 382. Its retained CMA and REINFORCE nominees
followed five and twenty updates respectively. The separately locked sixth-update
CMA distribution mean scored 8/16 safe completions in 381 actions. This exploratory
mean result does not improve completion over the strong anchor or selected CMA.

Every completed final arm had zero observed violations. Native engaged in all
sixteen cases; each retrieval arm engaged in fifteen. Case 05023-circle7 had no
admitted local target and produced a valid zero-action refusal. It stays in the
denominator as unfinished and non-engaged, without exclusion or a technical-failure
label. The unfinished count in every arm is sixteen minus its completion count.
Audits retained the refusal and verified the complete profiles against their
frozen sources, parameters, cases, resources and causal target histories.

These available final results support the utility of the shared controller
architecture over native planning. They do not establish an evolutionary
completion advantage: CMA ties the strong anchor in two completed profiles and
scores lower in the third. The unrun repeats and auxiliary controls remain
required by the frozen comparison. No full-comparison analysis, superiority
claim, confidence interval or three-mean aggregate is issued from this subset.

Automatic approval review rejected the prepared-input archive and transfer step,
including its justified retry, because moving private project artifacts requires
explicit user export authorization. Both attempts executed zero work. A request
for private transfer and backup authorization remains pending as of 07:09 UTC.
Artifact mirrors were stopped; no alternate archive or transfer route is used.
Consequently the other hosts have not received the final inputs, and their final
profiles remain unstarted. The normal new-run window closed at 07:03:14 UTC under
the unchanged twenty-minute preservation reserve. There is no authorization to
extend the 07:23:14 UTC campaign cutoff.

The navigation host's in-place final accounting v2 completed with no unresolved
components and report SHA-256
`312d4ba2e8f68ca10fa1898b0c127f0a4975d5f7f8c0235b7376feb9990eda2d`.
The report stays on that host. Cross-host accounting and preservation remain
separate closeout requirements. Until independent preservation is verified, pods
must be stopped at cutoff without deleting their retained research data.

The navigation-only exploratory mean report was subsequently produced in place,
with SHA-256
`49ce479fc75dff2f06d8ae3aa2b684102eeb158511c45ccee6bc3d7eba379184`.
It verifies that all 96 ordinary secondary episodes and sixteen mean episodes are
hash-bound in that complete own-host accounting report. Its six-update mean has
8/16 safe completions, 0/16 violations, 8/16 unfinished and 15/16 engaged, with
381 actual actions and 99 actions of unused allowance. No cross-host mean report
or aggregate has been produced.

PushT's first final accounting report v2 retained 415 unresolved components.
Saved evidence proved that 414 were completed job wrappers already represented
by their physical inner episodes. Each outer result names and hashes the inner
result, carries identical resources and action counts, and appears in its complete
producer state. A new collector v3 covers the wrapper only after verifying that
full chain. It also resolves one exact older transferred development-preparation
ledger through its original receipt, source freeze and exclusive navigation-host
protocol. No episode, resource ledger or frozen scientific source was edited.
Four focused rented-host checks passed, including malformed-linkage rejection
and the original double-counting reproduction. Both the v2 source and report
remain preserved.

The fresh PushT v3 final report completed with zero unresolved components, SHA-256
`207b7fec8a1f9c9d5c69a9ba29e4000402fae831e7881030a43cd11bcbe05b53`.
It retains 2,982 physical episodes and resets, 131,072 actual actions, 36,102
explicit encoder attempts, 189,525 explicit cost attempts, 160,887 explicit
safety attempts and 44,304,556 recorded predictor block steps. Thirty-two legacy
components lack explicit encoder/cost attempt fields; those missing measures
remain labelled rather than inferred from successful calls. Model-work aliases
must not be added together as a single query total. This report remains on PushT,
without archive, export or a cross-host final merge.

## Deadline and incomplete closeout

The common deadline passed at 07:23:14 UTC. The provider guard recorded STOP
requests for all three exact campaign pods at 07:23:19.722 UTC. Independent
provider reads subsequently returned `desiredStatus: EXITED` for all three. The
API did not return runtime, retention or billing-state fields, so those responses
do not independently verify the physical runtime transition, every retained file
or a final invoice. No pod was deleted. All three provider configurations name a
10 GB volume mounted at `/workspace`, where the project resides, plus a 30 GB
container disk. Independent full research preservation remains unverified.

Recovery's final v2 report has SHA-256
`ac41e3b30134e322c287cbca7cba812f022dd550ecaceb2c54b697199be368f6`
and one explicit unresolved ownership reference for the older transferred
development-case preparation. Its physical ledger retains 1,633 episodes and
resets and 45,147 actual actions. A new v4 collector independently qualified the
exact old receipt, source freeze and exclusive source-host protocol, rejecting a
wrong pod or modified receipt. The fresh full recollection was interrupted by
the provider STOP: SSH exited 255 without a completed report/SHA callback. V4
is not promoted, and v2's one unresolved item remains explicit. Neither episode
execution nor a recollection is restarted after cutoff.

The local closeout record is
`outputs/runpod-lewm-campaign-20261031/stopped-unpreserved-closeout-v1.json`.
Its conservative running-rental and disk allowance through the 07:24:09 UTC
verification is $9.23, versus $9.21 reserved through the original deadline, under
the shared $10 cap. This is not a provider invoice; retained storage can continue
to incur charges after STOP. Both artifact mirrors remain cancelled. The local
provider watchdog was cancelled only after all three exact pods reported EXITED.

The intended full optimizer comparison and independent backup are incomplete.
The available final profiles do not establish an evolutionary advantage. Private
backup/transfer authorization remains pending after automatic approval review
rejected the archive and transfer operation; no alternative export is attempted.

## Explicitly authorized completion window

Sachin subsequently approved private preservation, then explicitly authorized
finishing the full comparison with $3 more and destroying the pods afterward.
The revised cumulative cap is $13, including recovery and preservation. The new
independent provider guard was armed at 07:49:30 UTC with a strict 90-minute
cutoff, 09:19:30 UTC. Original protocols, parameters, case choices, seeds and
scientific evaluator sources remain frozen. No new training is authorized by
this completion phase. The earlier pending transfer and restart requirements
are now satisfied by these explicit instructions.

All three original GPU-resume requests failed because their physical hosts had
no free GPUs. Their original volumes were subsequently made accessible through
the documented `podResumeZeroGpu` recovery operation. Normal start, the legacy
CLI's zero-count flag and ordinary `podResume` with count zero did not provide
this recovery mode. Container SSH keys and ports changed; the old connection
records were retained and the recovered project sources were verified against
the original hashes before transfers. The ephemeral `/opt/lewm` environments
were lost. No numerical experiment is run on these limited recovery containers.

Whole-project whitelist archives are being built with the unchanged qualified
preserver, fresh credential-free deadline guards, source quiescence and every-file
hash verification. Private local verification must finish before an original
pod is deleted. A separate guarded replacement allocation may then occupy the
freed slot, retaining at most three live campaign pods.

Any remaining final evaluation on that replacement requires a new prospective
availability-relocation authority and V4 analysis binding. They preserve the
original V3 bytes and the three completed navigation core profiles, keep every
arm of each unstarted profile together, and record only the execution-host and
deadline overrides. All eight remaining main/auxiliary profiles and two frozen
terminal means remain required. Training and final hosts differ for relocated
controllers. Auxiliary comparisons use the lowest final repeat on their actual
final host, checked against the same case, seed and endpoint. Repeated auxiliary
blocks on one host do not constitute independent hosts. The original estimand,
case bootstrap and statistical limitations remain in force.

All three original archives have now passed independent private local verification:
navigation has 51,687 members and SHA-256
`2f06720f4ebc441e7ebf63a24c818f36d2d4a6d7606b321a5391107cc9cd2168`;
PushT has 39,346 members and SHA-256
`47bb52b4b7444fc7cd299600698646f07bb2d0e8c0cd5faea12a4849c80764c7`;
Recovery has 22,390 members and SHA-256
`7a2ba72397033600126bdb0a1d0f7fef87ccf78f325deea9b2f3bf7c9c50f2ef`.
Every member was streamed and hashed, and the external and embedded manifests
matched as JSON. PushT was deleted with provider absence confirmed at 08:20:31
UTC. Navigation and Recovery deletion/absence confirmations followed at 08:31:41
and 08:31:38 UTC. None was deleted before its local verification passed.

One fresh replacement A40, `uddcaj5qijcl3j`, was allocated at $0.59/hour after the
first verified deletion freed a slot. Its exact original package versions were
reinstalled. Full origin extraction to the network volume stalled in RPC I/O
wait. The exact known extractor processes were retired, their partial trees
retained, and unchanged member/path/hash checks were repeated into local scratch.
Navigation's full 51,687-member scratch extraction completed in 41 seconds.
All three complete immutable origins remain outside the new canonical project.
Only 807 science-input files, including all 99 frozen dependencies, were restored
there; old physical episodes, interaction ledgers, workers and ready records were
excluded to prevent foreign work from being charged as new work.

Capacity recovery and verification exceeded the initial 60-90 minute operational
estimate. Before any new final outcome, the agent prospectively revised that
estimate and independent shutdown guards to a 120-minute maximum, 09:49:30 UTC,
under the unchanged user-approved $13 cap. This time revision was an agent
implementation decision to complete the authorized comparison, not a new quote
from Sachin or permission to spend more. The previous 90-minute authority and
all source versions are retained. It changes no training, cases, seeds, parameters,
scientific budgets or statistical rules. The active relocation authority is
`navigation-final-relocation-authority-v3.json`, SHA-256
`2c47f43e0fb6e2a97be7a9870e3fca08728fad51750d95ded7a1e9494834588a`.

Fresh software qualification passed all 69 tests and Ruff on the actual A40.
The actual V4 continuation binding has SHA-256
`f60f88d401deaced2a2b0b2f233d4bed6f1f9eda744043620ee2f5c232468686`;
all ten relocated namespaces were absent at binding. A separately frozen probe
then encoded the first 64 cached training observations with the original transform,
FP32 settings and batch size. Their features were bitwise equal to the saved
embeddings, with maximum absolute error and RMSE both zero. It used one attempted
and completed encoder batch, 64 frames and no simulator resets/actions, predictor
queries or training. Its result SHA-256 is
`aaad6213da1b68edbf44cf112ce84fdf57e2dde1d4027ac6c5ae546ef08c7418`.
This is an encoder-contract check, not a new held-out performance result.

The exact ten-profile serial queue began at 09:07:33 UTC. Primary repeat 1
completed all 96 jobs in 3,184 actions and passed its complete frozen schedule,
parameter, seed, resource, inner-result and continuation checks. Per sixteen
cases, native scored 2 safe completions, strong fixed 12, tuned fixed 12, CMA 13,
random search 12 and REINFORCE 12. Their actions were 757, 486, 486, 483, 486
and 486. Every arm had zero observed violations; native engaged in all sixteen
cases and retrieval arms in fifteen. Unfinished counts were fourteen, four, four,
three, four and four respectively. This genuine retained CMA nominee improves
one repeat by one completion; the other primary repeats tie or trail fixed
control. All three repeats remain required in the unchanged aggregate analysis.
The queue continues with the remaining frozen profiles, without new training.

## Completed comparison and final closeout

The entire frozen schedule completed. The ten relocated profiles finished at
09:30:21 UTC, comprising 400 new final episodes and 10,758 executed actions. Together
with the 304 already completed navigation episodes, the final schedule contains
704 episodes: 576 main comparisons, 80 separate decoded-native controls and 48
exploratory terminal-mean episodes. No completed original profile was replayed.
All six main comparisons, five auxiliary profiles and three terminal means are
retained. The last profile was admitted before the unchanged twenty-minute reserve.

At the primary 50-action endpoint, the three safe-completion counts per sixteen
cases are:

- Native: 2, 2, 2.
- Strong fixed: 12, 12, 12.
- Tuned fixed: 11, 12, 12.
- CMA: 12, 13, 10.
- Parameter random search: 13, 12, 12.
- REINFORCE: 12, 12, 12.

At the secondary 30-action endpoint, the corresponding counts are:

- Native: 1, 1, 1.
- Strong fixed: 8, 8, 8.
- Tuned fixed: 8, 8, 8.
- CMA: 8, 8, 8.
- Parameter random search: 9, 8, 9.
- REINFORCE: 9, 8, 8.

Each main arm has zero observed violations in each sixteen-case repeat. Unfinished
counts equal sixteen minus the listed safe-completion count. Native engaged in
16/16 cases and every retrieval arm in 15/16. The zero-action refusal in case
05023-circle7 remains an unfinished, nonengaged observation. It is not excluded.
Three fixed learner repeats share sixteen source-case clusters and do not provide
48 independent cases. Zero violations in sixteen cases gives a rough 3/16 rate
bound, rather than evidence of zero risk.

CMA therefore improves one primary repeat by one completion, ties another and
loses two completions in the third. Its secondary completion counts tie strong
fixed in all repeats. The predeclared conditional case-cluster analysis reports
CMA minus strong fixed at -0.0208333 for primary, with nominal descriptive 95%
interval [-0.125, 0.1041667], and 0 for secondary, with interval
[-0.0833333, 0.0625]. These are the analysis host's saved estimates, not a local
recalculation. They condition on the fixed controllers and actual final hosts;
they do not estimate future-learner or host-population uncertainty and carry no
family-wise optimizer-superiority claim. The experiment establishes a useful
fixed controller architecture, not a consistent evolutionary advantage.

The five separate decoded-native blocks scored 5/16 safe completions at 30 actions
and 7/16 at 50 actions, with zero observed violations and 16/16 engagement. Their
unfinished counts were 11/16 and 9/16 respectively. Repeated logical blocks on
the replacement are not independent hosts. All three exploratory six-update
terminal means scored 8/16 safe completions, 0/16 violations, 8/16 unfinished
and 15/16 engagement. No terminal mean improved completion over strong fixed;
all remain descriptive, without confidence intervals or superiority claims.
The secondary repeat-2 CMA selection came from generation zero and is not credited
as a gain from subsequent adaptation.

All ten new profiles passed the independent dense-geometry, goal, refusal,
schedule and resource audits. The audit index SHA-256 is
`4f7bbb84661f18ca754d4f772106967725f7edbefb7cddec5afb80a8d433a9c0`.
The replacement's own accounting was collected after its workers retired and
before foreign episode imports. Recovery's interrupted accounting was reconstructed
read-only from preserved evidence on the declared replacement, retaining original
interaction ownership and the separate reconstruction execution host. The final
four-source merge has no unresolved components and SHA-256
`e4fcf9e64abcc2255320d00f002f46d5ace8e1f24a90aa0173d57036a995e67e`.
The 32 legacy components with missing encoder/cost attempt fields remain explicitly
missing. Model-work aliases are not summed into an invented query count.

The final conditional analysis has SHA-256
`0b21655efd47cdd150294633ac898e451f0e64059972f953cc121d59dd02d9fc`.
Its inputs, plots, tables and captions completed on the rented host before the
science cutoff. The separate scientific-media manifest has SHA-256
`e5a386458b0ab19bc4e14f6921b6b3bab78c524b532a9c05b09cb2e1bd0ccf33`.
The exact curated analysis inputs were sealed in a 24,348-member archive with
SHA-256 `19f8115a4fad787df1d437218d91b8960040dbace91b7fe2c427695d4be1a782`.
All scientific workers and analysis processes retired before final preservation.

After the unchanged 09:49:30 UTC science cutoff, the replacement's independent
guards allowed a preservation-only grace to 09:59:30 UTC under the unchanged
$13 cap. No new experiment or numerical analysis was authorized in that grace.
The final private archive was independently verified locally at 09:50:40 UTC:
6,174 members, 355,340,120 compressed bytes, SHA-256
`57256328e2bb966b21c6360b0d0f2a3ca9a17ceee71ecbacd45456d4ceb7341d`.
Every member was streamed and hashed, and external and embedded manifests matched.
Frozen models, controller memory/readout, final episodes, audits, accounting,
analysis outputs and the sealed curated input archive are retained. Public raw
datasets remain represented by the preserver's pinned references and exclusions.

The replacement was then stopped and deleted. At 09:52:42 UTC, a fresh provider
list contained none of the four campaign pod IDs, and the replacement returned
not found. All local provider guards were cancelled only after that verification.
The final conservative cumulative cost bound is $11.1642857 under the authorized
$13 cap. This is not a verified final provider invoice. There are no remaining
campaign rentals, and this closeout authorizes no future rental or experiment.

Local closeout and review evidence:

- `outputs/runpod-lewm-completion-20261031/replacement/closeout.json`.
- `outputs/runpod-lewm-completion-20261031/replacement/deleted-verified.json`.
- `outputs/runpod-lewm-completion-20261031/replacement/preservation-verified.json`.
- `outputs/lewm-campaign-20261031/private-backup-recovery/replacement-final-v1/`.
- `outputs/lewm-campaign-20261031/final-review-v1/` contains 34 analysis/media files
  copied from that archive with manifest size and SHA-256 checks. The main outcome
  plots, primary paired contrast and primary trajectory grid were visually inspected
  locally. No simulator, model or numerical analysis was run locally.
