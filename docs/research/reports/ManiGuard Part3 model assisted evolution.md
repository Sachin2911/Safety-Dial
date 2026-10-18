# ManiGuard Part 3: model-assisted evolution with correction-dependent trust

Execution is queued behind the complete Part 2 comparison. The data decision, forecast
criteria, source and prospective 64-episode optimizer pilot are frozen. No model-based
experimental finding is claimed in this progress report.

The hypothesis is that observed prediction errors after command correction can improve
which motion mutants receive physical evaluation. Plain evolution, disagreement-based
assistance and correction-dependent trust share the same four-parameter controller,
physical fitness, population budget and deployed information. The model guides
training-time offspring selection only. The native monitor still labels consequences;
there is no learned online action veto or complete safety filter.

The data decision was sealed before model fitting and before new calibration/audit
outcomes. Part 1's 24 training episodes contain actions, native videos and exact initial
observations, but lack later raw images and numerical margins. Its exposed comparison
is excluded. Part 2 provides exact recordings on only two tasks. Therefore the declared
eight-episode common-recording option is included: B/C on `task_0046/base` and
`task_0006/base`, two fresh seeds each, with the native horizon. It is not a rescue batch
after model failure. Total new physical scope is 80 or, with this option, 88 full episodes
if the model qualifies; a failed/inconclusive scientific gate leaves the 64 optimizer
episodes unstarted.

Whole tasks are separated: all sixteen Part 2 episodes on 0039/0010 fit the model; four
recordings on 0046 calibrate reliability; four on 0006 audit forecasts. No adjacent-frame,
seed or task-variant leakage is allowed. Original acquisition costs are charged once.
The final two validation tasks 0048/0050 and the main-study test partition stay outside
model development.

A pinned frozen DINOv2 small encoder supplies 2x2 spatially pooled patch features from
both cameras. Training-only standardization and randomized PCA retain 32 visual
coordinates, with nine measured arm/finger coordinates kept separately. A one-step
causal observation/action history conditions five episode-bootstrap residual MLPs.
They learn from actual environment inputs, never unexecuted proposals. Separate ridge
readouts predict minimum upright margin and negative target-to-goal distance. These
are partial targets, distinct from spill/drop predicates and native held-object goal
completion. The final raw observation is preserved but omitted from model pairs because
it has no subsequent recorded two-finger proposal frame; no missing state is fabricated.

The audit compares persistence, ridge, action-shuffled and ensemble predictions at one
and four actions; raw, PCA and random-projection readouts; actual versus predicted future
state heads; consequential errors near the constraint; and matched nominal acceptance
for disagreement versus correction-dependent trust. It also measures acquisition latency.
Every numerical criterion was sealed before opening the forecast audit. Insufficient
near-threshold/event support is an inconclusive gate, distinct from a failed accuracy
criterion or an engineering/time-limited incomplete stage. Thresholds will not be relaxed.

If eligible, four shared initial controllers receive eight full physical episodes once.
Each method then evaluates a fresh elite, one unguided timing mutation and two motion
mutations on two training cases, totaling 24 descendant episodes. Twelve confirmation
episodes select three finalists before twenty fresh comparison episodes. Model ranking
sees only the first four already-queried commands, propagates causal predicted robot
state and cannot rank later query timing. All proposals, trust abstentions and uniform
fallbacks remain in lineage. Physical outcomes alone determine fitness and final selection.

The frozen source is commit `31d8902`. See
[data decision and forecast criteria](../../../configs/maniguard/followup/part3-data-decision.json),
[source freeze](../../../configs/maniguard/followup/part3-model-source-freeze.json) and
[model environment](../../../configs/maniguard/followup/model-freeze.txt).
Raw records use `runs/maniguard/phase3-parts2-3/20261018-56582094/part3`; trained artifacts
use `data/maniguard/models/phase3-parts2-3-20261018`. The shared
[GECCO paper](maniguard-phase3-part1-gecco/named.pdf) will carry the audited outcome and
limitations. No surrogate/evolutionary novelty or optimizer superiority is established
by implementing these ingredients.
