# LeWM proposal-adapter integration

30 October 2026. Sachin authorized one fresh RunPod rental to implement the proposed
behavioural adapter and test it on one known PushT engineering case. The frozen
[protocol](../../configs/lewm_proposal_20261030/smoke.json) allows one valid episode,
with up to two failed technical starts, a fresh $1 cap and a two-hour cutoff.
This is an integration study, not an optimizer comparison.

## Implemented stack

The adapter has 29 causal features, eight hidden units and 510 parameters. Inputs
include native pusher position and velocity, planner feedback, supplied hazard
geometry and eight fixed random projections of the frozen latent goal difference.
The projection is not trained. No T-block pose or geometric task-goal coordinates
enter the adapter. Hazards remain virtual and are absent from LeWM's image inputs.

The network proposes three blocks of five two-dimensional actions, bounded in
normalized action space. One proposal replaces the final existing CEM candidate
in each iteration. Ordinary candidates remain, and nominal search-prediction work
is unchanged. The existing learned motion checker, geometry, final plan check and
refusal rule remain in the execution path. Network bounds do not guarantee safety.

The initial controller is seeded and untrained. Its weights remain fixed during
the episode. Separate CMA-ES ask/tell and checkpoint/resume mechanics are implemented
and software-tested with synthetic fitness. No real evolutionary training ran.

## Software and episode results

Eleven focused GPU checks passed on the rental, including proposal serialization
and replay, solver injection, unsafe-plan refusal, query accounting, artifact
directory creation and evolutionary checkpoint/resume. Ruff passed using the
repository's existing settings. No experiment tests or model execution ran locally.

The single valid island episode achieved **1/1 safe completions**, **0/1 violation
episodes**, **0/1 unfinished episodes** and **1/1 engagement**, using 35 actions.
Minimum full-pusher hazard clearance was 6.453640 px against the fixed 6 px margin.
There were seven adapter calls and 105 injected-candidate evaluations. None of the
raw adapter sequences was selected directly. This does not identify its indirect
effect on search, and there is no contemporary planner-only comparison.

Charged valid-episode work was one reset, two reset physics steps, 350 executed
physics steps, 118 world-model batch calls, 26,263 candidate rollouts and 78,789
predictor steps. The independent rented-host audit reproduced the native task goal,
dense safety outcome and every proposal from saved features and frozen weights.

The first launch failed while writing the initial controller's missing parent
directory, before simulator creation or model evaluation. Its log and original
sources remain preserved. The repair added an atomic directory-creating writer
and regression check without changing the case, seeds or controller settings.
Earlier preparation fixes handled SSH environment inheritance and archive ownership.

## Evidence and lifecycle

The launch, result, local verification and pod closeout have separate receipts in
`progress/receipts/`, prefixed `20261030-lewm-proposal-`. Artifacts are under
`outputs/lewm-proposal-smoke-20261030/`, including source versions, frozen inputs,
the initial controller, traces, audits and an annotated saved-frame replay.
Overlays are for human review and never entered model observations.

Automatic approval review rejected transferring the RunPod API credential to the
pod. It remained local. A local provider-stop guard and credential-free pod process
guard were used. Destruction follows independent local artifact verification; the
closeout receipt records provider absence and the conservative rental estimate.

The result establishes functioning integration on one reused engineering case.
Completion benefit, generalization and evolutionary advantage remain unestablished.
Any real training requires a new prospective development protocol and budget.
