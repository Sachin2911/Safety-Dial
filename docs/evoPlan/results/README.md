# Evolution study results (docs/evoPlan)

**Current status, 4 October 2026:** strict Gate 0 remains unpassed. The fresh
measurement-readiness audit passed; the bounded transfer and noise diagnostics
are complete, and every tested noise level failed the declared joint transfer
screen. The main Stage 5 experiment has not run. A bounded development search/power
pilot is now running under the existing readiness scope; see its protocol below.
All 657 tests now pass, including the three new residual/selection/variance checks.

The user's latest decision is to stop further Gate 0 controller repairs and
prepare Stage 5 under an explicit extension of the readiness route, with k=1 as
the proposed main imagination setting. The
[Stage 5 declaration](../protocols/stage5-readiness-draft-20261004.md) is a draft
for supervisor review, not an approved launch or a retrospective gate pass.
The earlier [direct-simulator fallback preparation](fallback-preparation-20261004.md)
is retained as history and deferred by this decision. Earlier reports' next-step
recommendations reflect their dates; this index and the new draft record the
current requested direction.

| Completed work | Current finding |
|---|---|
| [Controller repair sequence](stage2-status-20261004.md), [clean targets](stage2-clean-targets/walker2d-evo-s2-clean-targets-20261004-1/README.md), [ensemble](stage2-ensemble/walker2d-evo-s2-ensemble-20261004-1/README.md) | Repeated development failures remain. These reused 24-root diagnostics are not independent confirmation. Further controller repairs are stopped. |
| [Fresh readiness baseline](stage2-readiness-baseline/walker2d-evo-s2-readiness-baseline-20261004-1/README.md) | 51/256 real versus 2/256 imagined violations, 100.3% of recorded progress. Separate pilot-readiness criteria pass; strict controller criterion fails. |
| [Transfer pilot](stage4-transfer-pilot/walker2d-evo-s4-transfer-pilot-20261004-1/README.md) | Baseline plus 96 residual candidates: zero imagined violations, 14 to 49 real failures per 64 roots. Safety ranking undefined. Declared screen fails; official Gate 1 not run. |
| [Noise diagnostic](stage3-noise-diagnostic/walker2d-evo-s3-noise-diagnostic-20261004-1/README.md) | At k=1, residual-only safety/progress rho are 0.675/0.374; all-candidate values are 0.713/0.446. Every level fails the joint screen. Predicted risk remains severely underestimated. Zero new real steps; 1,036,800 predictor rows. |
| [Gate/environment review](gate-and-environment-review-20261004.md) | Walker2d retained, historical gates preserved. Readiness initially authorized bounded pilots only; a main experiment needs the proposed explicit extension. |

The user subsequently asked to move the decision and engineering forward. The
[bounded search/power pilot](../protocols/stage5-power-pilot-20261004.md) was
committed as `9c043c1` before launch and is running as
`walker2d-evo-s5-power-pilot-20261004-1`. It compares k=0 and k=1 with four paired
search seeds, using existing development episodes and a zero extra unsafe penalty.
Its budget is 30,980,480 predictor rows and 470,400 real steps. The exact archived
zero-correction imagined control passed. No ROSARL comparison or new final-bank
collection is included, and the main declaration still awaits scientific review.
This is an in-progress status, not a result or a claim of successful selection.

The noise write-up and its metrics were committed in `55a7be5`. The current
readiness-route proposal preserves that negative joint-screen result, even though
it supersedes the report's recommendation to move next to direct-simulator evolution.
At k=0, zero predicted violations do not preclude selection-induced increases in
real failures through progress optimisation. At k=1, improved ordering does not
establish calibrated probabilities or safe controllers.

The readiness and transfer bundles are privately archived and remotely hash-verified
at revisions `b706cfae23f30382a05ffeb58c418584d6f6b3d6` and
`24f4881fc2b3a10992d2151db3085b21ce0e1b9d`. The complete new noise bundle is
prepared locally (17.85 MiB); its separate upload approval is still pending.
No supervisor message or GitHub push has been sent.

## Historical baseline record, 3 October 2026

Committed results of the [implementation plan](../README.md). Run bundles are in the private
model repository `Sachioster/safetydial-walker2d` under `evo/<run_id>`, each pinned by the
revision in its `hf_upload.json`. Every run folder holds its resolved `config.yaml` and a
`manifest.json` (git commit, packages, hardware, seeds, input revisions, costs).

**Historical status at the close of 3 October 2026:** stopped at a failed Gate 0. The chronology below records that earlier state; the current 4 October status is above.

| Stage | Run | Outcome |
|---|---|---|
| 0 Assets | [walker2d-evo-s0-20261003-1](stage0/walker2d-evo-s0-20261003-1/check.json) | Pass. All 704 saved S4 no-update rows (dev 192, test 512) reproduce the imagined health clearance bitwise (max difference 0.0, decisions identical). |
| 1 Harness | commit 6ebc21f | Done. Closed-loop imagination is bitwise equal to `WalkerImaginer.rollout` on fixed tapes; the real executor is bitwise equal to `execute_branch` on all 1,216 stored S4 branches; real outcomes do not depend on batching, chunking or worker count; 603 tests pass. |
| 2 Starting policy | [walker2d-evo-s2-20261003-1](stage2/walker2d-evo-s2-20261003-1/gate0.json) | **Gate 0 fail.** theta_BC violates the health rule in 171 of 256 evaluation segments (67%, Wilson 61% to 72%); at most 3 were allowed. Its forward progress is 1.57 m per 0.8 s against 1.91 m for the recorded policies (ratio 0.82, which alone would pass). |

## Gate 0 in detail

- **Thresholds (provisional, declared in [stage2.yaml](../../../configs/evo/stage2.yaml) before
  the run):** violations of theta_BC at most max(2 x recorded, 3) of 256, and mean real progress
  at least 0.5 x recorded. The recorded policies' violation rate on representative roots is zero
  by construction (a root needs 100 further recorded steps and the recording environment
  terminates on the same health rule), so the user set the 3-of-256 resolution floor on 3 October.
- **Behaviour cloning fitted well offline.** 59,100 frames from 300 late PPO-Lagrangian episodes;
  ridge strength 0.1 chosen on held-out episodes (validation MSE 0.0074); R2 of 0.92 to 0.98 per
  motor. The failure is closed-loop.
- **When it fails.** First unsafe steps spread over the whole segment (median step 55 of 100);
  23 of the 64 test episodes violate from all four of their start states, 5 from none.
- **Imagination does not see it.** The LeWM imagines theta_BC (k = 0) violating in only 4.7% of
  the same 256 segments, against 67% in reality. The probe on real frames reports 75% and the true
  state at block ends 66%, so the probe reads real frames correctly: the gap lies in imagined
  dynamics, before any selection pressure.

## Post-hoc diagnostic: the held action alone (not part of the gate)

[diagnostic_hold.json](stage2/walker2d-evo-s2-20261003-1/diagnostic_hold.json), from
[evo_gate0_hold_diagnostic.py](../../../experiments/scripts/evo_gate0_hold_diagnostic.py):
the recorded policies' own actions replayed open loop from the same 256 states, with no
cloning error at all.

| Action interface | Violations of 256 | Progress (m per 0.8 s) |
|---|---|---|
| Exact recorded actions, 125 Hz | 0 (0%) | 1.91 |
| Each block's mean action held for the block, 12.5 Hz | 126 (49%) | 1.83 |
| Each block's first action held (sample and hold), 12.5 Hz | 211 (82%) | 1.73 |

The recorded-policy mean-action replay fails in about half of segments. This diagnoses harm
from that replay interface; it does not prove that every possible policy using held actions
must fail. The earlier impossibility interpretation was too strong. Later controllers emit
ten distinct actions per block; their failures remain separate evidence.

## What the plan prescribes, and the open decisions

The plan's response to a Gate 0 failure: change the policy class before continuing, either
output the full 60-d block instead of a held action, or use a small MLP, and evolve a
low-dimensional perturbation around the behaviour-cloned weights. The diagnostic favours the
60-d block (the exact recorded blocks never violate open loop). The 49% rate was measured for
mean-action replay, not a proven lower bound for a separately learned held-action MLP. Any new policy class needs its own pre-registered
Gate 0 run with the same thresholds. Those were the open choices on 3 October; the later diagnostics and current decision above supersede that action list.

Provisional, to be confirmed with the supervisor: Gate 0 thresholds (including the 3-of-256
floor), Gate 1 and Gate 2 thresholds, and the ROSARL adaptation (V_MIN - V_MAX from running
unpenalised segment returns; a violated segment keeps its pre-violation return).
