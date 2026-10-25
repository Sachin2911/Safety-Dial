# Task 7 frozen-schedule validation repair

25 October 2026, before any task-7 simulator episode or VLA query. Source inspection
found that the reused recording validator selected task 6's schedule internally.
The task-7 coordinator already launched the freshly frozen task-7 seeds, but the
validator would have rejected their otherwise valid recordings as seed mismatches.
This is an integration bug, not a reason to change the new seeds or repeat a rollout.

The validator now takes an optional explicit job schedule. Both coordinator paths
that validate an episode pass the schedule from their already verified source freeze.
Historical callers that omit it retain the original task-6 schedule. Native result
seeds, scene identifiers, exploration seeds, query metadata and job manifests are
still checked against the supplied frozen schedule; no test is weakened. Actor and
critic inputs, source SAC equations, rewards, terminals, numerical tolerances,
episode schedules, budgets and accounting remain unchanged.

Three end-to-end recording fixtures prove that a task-7 episode is rejected under
the historical schedule, accepted under its own schedule through the live queue's
validation method, and that an old task-6 episode cannot enter the new schedule.
The complete local DSRL suite passes 186 tests with zero skips. Ruff and formatting
pass. These regression tests use synthetic recordings and mocked services, with no
new model, gradient or simulator calls. The JUnit receipt is preserved in the new
source amendment.

The independent fixed-batch learner checks ran under the original stage-A source
while the simulator installed. Default kernels failed repeatability and direct
author-versus-bridge parity; the declared deterministic setting passed all five
pairs with exact equality. All twenty update calls remain in the original ledger.
This validation repair changes neither of those diagnostic operations or their
saved results. Their original process and source launch receipts are retained.
