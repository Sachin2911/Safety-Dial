# Main plan: which experience makes LeWM safer to use?

**Status: adopted 26 September 2026.** This folder turns the
[21 September research direction](../researchDirection.md) into an executable plan. The
supervisor, Geraud Nangue Tasse, has seen the 21 September direction and is happy with it.
The Walker2d track and the pretraining-versus-adaptation stretch were added on
26 September. **No experiment described in this folder has run yet.**

Student: Sachin Mohan (2699183), BSc Honours Computer Science, University of the
Witwatersrand. Thesis due in late November 2026.

## Decisions recorded on 26 September 2026

1. The thesis is due in late November 2026. Core results freeze on 8 November.
2. The Safety-Gymnasium model is trained from scratch on `SafetyWalker2dVelocity-v1`.
   PointGoal1 and HalfCheetah are documented alternatives, not scheduled work.
3. The pretraining-versus-adaptation experiment (S5) is included as a stretch goal.
4. The supervisor has approved the 21 September direction that this plan elaborates.
5. The historical Vast volume no longer exists. Everything is rebuilt on a fresh RTX 5090
   instance, and every model checkpoint is stored in private Hugging Face repositories.

## What is in this folder

| File | Contents |
|---|---|
| [protocol.md](protocol.md) | Shared definitions: decisions, metrics, splits, adaptation, acquisition, budgets, statistics |
| [pushT.md](pushT.md) | The main study on the released Push-T LeWM, E0 to E5 |
| [walker2d.md](walker2d.md) | Training a LeWM on Safety-Gymnasium Walker2d, S0 to S5 |
| [infrastructure.md](infrastructure.md) | Fresh 5090 setup, Hugging Face checkpoints, file layout, rules that protect earlier work |

Committed results of the new study go under `results/` in this folder, created when the
first experiment runs.

## How this relates to the other documents

- [researchDirection.md](../researchDirection.md) still defines the question and the
  interpretation rules: what a frozen encoder can and cannot show, charged budgets,
  false-safe accounting. This plan does not change them.
- [research/checkpoints.md](../research/checkpoints.md) and
  [research/relatedWork.md](../research/relatedWork.md) remain valid references.
- [research/pilot.md](../research/pilot.md) was the first-pilot checklist. This folder
  replaces it for execution; where they differ, this folder controls. Its asset-recovery
  step no longer applies because the old assets are gone.
- Completed experiments ([Safe CEM](../safeDial/) and [Phase 0](../phase0/)) are preserved
  evidence. Nothing in this plan edits them or reruns them in place; see
  [protecting earlier work](infrastructure.md#protecting-earlier-work).

## The question

> At the same interaction budget, which additional experience most improves a LeWM's
> predictions of unsafe outcomes, and does the improvement transfer to new hazards,
> starting states and goals?

The safety rule is supplied, for example "no part of the T may enter this region". This is
not label-free safety discovery. The object of study is whether the model preserves enough
physical information to answer the rule, first in real observations and then in imagined
futures, and which extra experience repairs it when it does not.

## Hypotheses

Each hypothesis has a useful negative outcome. None is assumed.

| | Hypothesis | Tested in | If it fails |
|---|---|---|---|
| H1 | False-safe decisions come mostly from imagined dynamics, not from reading real observations or from checking only at prediction steps | E1, S4 | Report the readout or temporal-sampling bottleneck instead |
| H2 | Predictor-side adaptation on modest new experience reduces false-safe decisions without harming ordinary prediction or goal reaching | E2, S4 | Report the bounded optimisation check and pause acquisition work |
| H3 | At equal charged simulator steps, boundary-focused experience repairs more than random experience | E3, S4 | Random matching boundary is a legitimate result |
| H4 | Boundary experience is concentrated, so its advantage shrinks on held-out hazards and starts while random experience transfers more evenly | E4 | Boundary data transferring well is equally informative |
| H5 | Offline gains carry into closed-loop Safe CEM | E5 | Report where offline and closed-loop results disagree |
| H6 | The same near-failure experience has a different effect in pretraining, where it also shapes the encoder, than when added later to the predictor | S5 | Equal effects suggest the encoder was not the bottleneck |

## Two tracks

**Main track: Push-T with the released checkpoint.** Diagnose the error, test whether new
experience repairs it, compare acquisition strategies, then test transfer. See
[pushT.md](pushT.md).

**Parallel track: our own LeWM on Walker2d.** Train a LeWM from scratch on
`SafetyWalker2dVelocity-v1` with deliberately expert-like data, then rerun the minimum
study on it. Because we control its training data, its test states are genuinely unseen
and H6 becomes testable. MuJoCo state restore is exact, so branching needs no replayed
prefix. The track has a hard go/no-go on **18 October 2026** and is not a dependency of
the Push-T study. See [walker2d.md](walker2d.md).

## Timeline

| Week of | Push-T | Walker2d | Milestone |
|---|---|---|---|
| 28 Sep | Fresh 5090 setup and assets to Hugging Face; E0 | S0: rules, timing, observability check; start policy training | E0 gate |
| 5 Oct | E1: probes and error decomposition | S1: data set A and probe set; S2 smoke run | E1 gate |
| 12 Oct | E2: repairability | S2: full LeWM-A training; S3 probes | S3 go/no-go on 18 Oct; E2 gate |
| 19 Oct | E3: random versus boundary, first budgets | S4: decomposition | |
| 26 Oct | E3: remaining budgets and seeds; E4 transfer | S4: random versus boundary; S5: collect set B, train LeWM-B | |
| 2 Nov | E5 if E3 holds | S5: comparison | Core results frozen 8 Nov |
| 9 Nov | Writing: method and results | S5 only if writing is on track | |
| 16 Nov | Writing and figures | | Full draft to supervisor |
| 23 Nov | Revisions and submission | | Thesis due (late November) |

The Walker2d track needs roughly 6 to 8 days of hands-on work in weeks 1 to 3, which is
exactly when E0 and E1 matter most. The 18 October gate exists so that it cannot eat the
main study. When the two compete for time, Push-T wins.

## Priority order

If time runs short, cut from the bottom:

1. E0: assets, replay and geometry.
2. E1: the error decomposition.
3. E2: repairability.
4. E3 minimum: random versus boundary at one budget with one seed.
5. S0 to S3: a trained and validated Walker2d LeWM.
6. E3 full: all budgets and three acquisition seeds.
7. E4: transfer.
8. S4: decomposition and random versus boundary on Walker2d.
9. Either the learned error-risk selector or S5. If boundary clearly beats random, the
   selector is the natural next step; if it does not, S5 is more informative.
10. E5: closed-loop Safe CEM.

**Minimum thesis:** items 1 to 4, the Walker2d training outcome (whatever S3 finds) and
the two completed studies. **Strong thesis:** items 1 to 8. **Stretch:** items 9 and 10.

## Thesis outline

1. Introduction.
2. Background: JEPA world models and LeWM; safe planning with world models; data
   selection and model adaptation (FARM, AdaJEPA, ReDRAW, FOSP, WMPO).
3. Preliminary studies: Safe CEM and the dial on Push-T, and Phase 0 (termination is not
   irreversibility). Both motivate the question: a probe cannot police its own validity
   domain, and benchmark failure flags are not physical facts.
4. Method: constraint decisions from imagined futures, the four-source error
   decomposition, predictor-side adaptation, acquisition with budget accounting, and dial
   curves.
5. Results on Push-T: E0 to E4, and E5 if run.
6. A LeWM trained on Safety-Gymnasium Walker2d: training, replication and, if run,
   pretraining versus adaptation.
7. Discussion, limitations and future work: visible hazards in PointGoal1, encoder
   adaptation, and a workshop version.

An ICLR 2027 workshop remains a possible target after submission. Check the actual call
before citing a deadline; no acceptance is assumed.
