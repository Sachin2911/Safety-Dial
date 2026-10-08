# Functional averaging improved validation but left four development failures

The combined six-member ensemble was selected on validation, with 3/64 failures
and mean progress 2.342 m. The three recorded-target members also had 3/64, with
2.312 m; the three clean-target members had 5/64 and 2.332 m. Progress broke the
failure-count tie according to the declared rule. Only the frozen validation winner
was evaluated on development, where it failed 4/24 segments with progress 2.102 m.
Three failures were PPO roots and one was PPO-Lagrangian. Imagination reported zero
failures, while the probe on real images reported four.

Each member is a 444-512-256-60 visual action-history network. One checkpoint per
training seed was selected using existing 64-root validation results from the
recorded-target and clean-target experiments. The six-member policy has 2,247,528
parameters. It averages the clipped action proposals; all members receive the
same previous combined action actually executed. Reality and imagination use the
same policy interface. Member identities and every source checkpoint were frozen
by file hash before this run. No true state or source-teacher identity enters the
controller. No new fitting was performed.

The ensemble reduces validation failures relative to the earlier selected single
controllers, but its development count does not improve on their 4/24. Reusing
validation for both member and ensemble selection makes validation performance
optimistic. The repeatedly inspected development set is also not untouched final
evidence. This run does not establish a viable starting controller or justify
advancing through Gate 0. The original mixed controller remains the reference.

The run used 21,600 simulator steps, 240 imagined rows, 2,488 renders, 2,424 encodes
and zero new optimiser updates in 13.24 seconds. Historical source training costs
remain charged to their source runs. All 643 tests passed with 49 warnings; Ruff
passed. Protocol commit: 8835657. Gate 0 was not rerun and stages 3 to 7 have not run.

All weights, member choices, validation outcomes, the frozen selection, development
trajectory, configuration, source and logs are retained. Private archival is pending.
