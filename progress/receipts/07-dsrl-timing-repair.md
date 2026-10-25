# Task 7 final-batch timing reservation

25 October 2026, before live engineering or development episodes. The original
projection counted twenty measured collection rounds, 90,000 learner updates,
3,000 CPU actor calls and a 1,500-second allowance. The queue separately requires
2,700 seconds remaining when admitting any cached episode: its 1,500-second wall
timeout plus the 1,200-second preservation reserve. If the measured episode is
shorter than twenty minutes, a projection that only just fits could therefore be
rejected when the final evaluation block starts.

For measured episode duration d, the corrected projection uses
`20*d + 90000*update_seconds + 3000*cpu_actor_seconds + max(1500, 2700-d)`.
This retains the post-work allowance and ensures that the predicted final block
starts with at least the queue's required 2,700 seconds remaining. The source
records the components individually. Finite, positive episode and update times
are required. The full 24-training/48-evaluation schedule, seed panel, source SAC
settings, episode timeouts, preservation reserve and cumulative $15 cap are fixed.

Five focused tests cover short, intermediate and full-timeout episode durations,
zero durations and nonfinite update measurements. The complete local DSRL suite
passes 195 tests with zero skips. Ruff and formatting pass. All tests use synthetic
data or mocked services and add no neural or simulator calls.

The previously completed numerical diagnostics remain valid and unchanged. This
repair changes only the later admission calculation and its receipt fields. Original
source freezes, process identities, query records and twenty engineering update
calls are preserved. No live episode has been admitted under the old projection.
