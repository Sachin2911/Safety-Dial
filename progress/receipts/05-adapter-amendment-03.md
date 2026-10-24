# Task 5 runtime metadata correction before simulation

Frozen 2026-10-24T18:13:08.383058+00:00, before any task 5 simulator start. The waiting
engineering launcher was stopped while asset installation continued, after
verifying that no task 5 episode directory existed. V1, v2 and v3 plans remain
unchanged; execution now explicitly selects `05-adapter-plan-v4.json`.

A read-only comparison with task 4's first validated recording and the pinned
`eval_config.py` revealed a mismatch in the new validator:
`resolve_source_config` converts the YAML value `legacy` to Python `None`.
The recorder saves `vars(cfg)`, so its runtime metadata contains an explicit null,
while the evaluator's exported configuration spelling is `legacy`.

The validator now requires the source-profile key to exist and equal None, along
with 256-pixel cameras and all existing contract checks. It rejects both a
high-fidelity value and a missing key. The synthetic fixture matches the resolved
runtime configuration. **19 tests pass**, including the two new negative contract
tests. Ruff and shell syntax checks pass. This fixes a checker that would otherwise
reject a valid recording; it does not change the rendering configuration.

V4 also pins the shared plotting-only correction from commit `f7b2a72`: lid and
gripper plots use physical scales so negligible numerical jitter is not visually
magnified. Numeric metrics and scoring are unchanged.

The controller, queries, task/seed panel, deadlines and interaction ceilings are
unchanged. No task 4 performance outcome was used to change those choices. The
evaluator hook itself is unchanged from v2.

V4 plan SHA-256: `631880a5f83e745fc11a392f7f06f1054e6ce7b9b4154f6d9ab358f07e5ae9cb`.
