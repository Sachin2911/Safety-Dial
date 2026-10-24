# Task 5 pre-simulation accounting amendment

Frozen 2026-10-24T17:22:35.157726+00:00, with no task 5 simulator starts.
The original plan is preserved unchanged. Execution uses `05-adapter-plan-v2.json`,
which cites the original plan SHA-256 and updates source hashes only. The
controller, tasks, seeds, order, information and interaction ceilings are unchanged.

Code review found that the recorder counts returned action chunks, which omits a
policy call interrupted before its response. The evaluator hook now journals each
query attempt before the call and its return immediately afterward. The task 5
copy of the existing synthetic probe does the same, with flushed output, including
failed warm-up calls. Its observations, sampling seed and retry behavior are
unchanged. Analysis reports attempts and returns separately.

The live-hook fixture exercises both query boundaries. All 17 tests pass, Ruff
passes, and the launch shell passes `bash -n`. Synthetic aggregation and pairing
checks also passed without simulator interactions. Revised patched evaluator
SHA-256: `9d7f5e2f68a5e9c50d1049720ace479b291ec80104c32973c95d8ff5a1906c63`.
V2 plan SHA-256: `ff3a688c8acd818faf5408909f6273afbdf759cdfb0bac7a7dffdae6f218f19e`.

Launch command: `bash experiments/scripts/launch_adapter05.sh engineering`, then
`pilot` only after the evidence-based admission file exists. The launch script
selects the v2 plan explicitly; the old plan remains an auditable earlier freeze.
