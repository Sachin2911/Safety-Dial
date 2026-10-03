# Evolution study results (docs/evoPlan)

Committed results of the [implementation plan](../README.md). Run bundles are in the private
model repository `Sachioster/safetydial-walker2d` under `evo/<run_id>`, each pinned by the
revision in its `hf_upload.json`. Every run folder holds its resolved `config.yaml` and a
`manifest.json` (git commit, packages, hardware, seeds, input revisions, costs).

**Status, 3 October 2026: stopped at a failed Gate 0 (stage 2).** Stages 3 to 7 have not run.

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

Holding one action for 0.08 s already makes about half of the segments violate, so no policy in
the planned class (one held action per block) can meet the Gate 0 threshold.

## What the plan prescribes, and the open decisions

The plan's response to a Gate 0 failure: change the policy class before continuing, either
output the full 60-d block instead of a held action, or use a small MLP, and evolve a
low-dimensional perturbation around the behaviour-cloned weights. The diagnostic favours the
60-d block (the exact recorded blocks never violate open loop); a small MLP that still holds one
action per block would keep the 49% floor. Any new policy class needs its own pre-registered
Gate 0 run with the same thresholds. These choices await the user (and supervisor).

Provisional, to be confirmed with the supervisor: Gate 0 thresholds (including the 3-of-256
floor), Gate 1 and Gate 2 thresholds, and the ROSARL adaptation (V_MIN - V_MAX from running
unpenalised segment returns; a violated segment keeps its pre-violation return).
