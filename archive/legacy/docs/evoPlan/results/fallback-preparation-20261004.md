# Preparing the direct-simulator fallback

The user accepted a bounded noise diagnostic following the transfer failure.
Its prospective rule was to prepare the real-simulator fallback if no level passed
the joint safety-and-progress screen. None did. This document specifies the next
implementation task; it is not a report of a completed fallback experiment.

## Research question and retained limits

Use the same Walker2d health constraint, but ask how evolutionary constraint
handling compares when fitness comes directly from the simulator. The original
[implementation plan](../README.md#fallback-track-if-gate-1-fails) proposes a
state-based linear policy with 17 inputs and PPO-Lagrangian as the reference.
The physical state is privileged information. Success on this track would not
pass the visual-controller gate or prove that evolution exploits imagination.
Retain all failed gates and the calibrated k=1 partial safety-ranking result.

## Next concrete work

1. Implement a 17-input, six-action linear policy with clipped actions and
   feedback at every 125 Hz simulator step. Keep normalisation fitted only on
   whole-episode training data. Validate state extraction, action scaling,
   deterministic reset and sequential solver state before any search.
2. Prepare a behaviour-cloned initialisation from existing training episodes,
   with whole-episode validation and a small, predeclared ridge-strength grid.
   Report offline fit and real closed-loop competence separately. The near-perfect
   teacher-specific replay in earlier diagnostics does not imply that a single
   deployable policy can reproduce every source gait.
3. Benchmark the candidate executor on development roots, with the final PPO and
   PPO-Lagrangian references receiving the same 125 Hz feedback. Count all real
   steps, including initialization checks. Keep short-branch outcomes distinct
   from full-episode locomotion. A working fallback baseline needs both.
4. Freeze the initial policy, normalisation, simulator budget and seed schedule
   before a small CMA-ES pilot. Begin with the already implemented Deb ranking.
   Compare learned candidates and PPO-Lagrangian on disjoint new episodes after
   selection. Avoid selecting a winner and then calling its fitness roots test data.
5. Expand to fixed penalties and ROSARL only after basic competence and cost
   accounting work. The documented supervisor check of the ROSARL adaptation
   still applies before its comparison; this preparation does not settle it.

Do not automatically repeat the previous visual-controller architecture search
in a new representation. If the predeclared linear initialisation fails, report
that outcome and make the next policy-class decision explicitly. A nonlinear
teacher residual or state MLP would be a different policy class and must be named
and budgeted before its experiment.

## Current assets and costs

The full readiness and transfer-pilot bundles are already hash-verified in the
private Hugging Face repository. The latest noise test adds no real interaction;
its calibration and real fitness references are pinned to those archives.
The fallback can use the existing simulator, exported actors, state/action
training data and deterministic snapshot machinery. A new world model or new
environment is not a prerequisite. The full penalty/noise grid remains on hold.
