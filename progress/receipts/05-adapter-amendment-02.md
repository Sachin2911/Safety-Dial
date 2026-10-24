# Task 5 replacement-host and pre-contact accounting amendment

Frozen 2026-10-24T17:38:21.378300+00:00, before any task 5 simulator start. The v1 and v2
plans remain unchanged. Execution now selects `05-adapter-plan-v3.json` explicitly.
Its parent hash identifies v2. Controller settings, task/seed panel, arm order,
information, horizon and interaction ceilings remain unchanged.

The first container never reached SSH. Its stop intent was acknowledged, then it
was destroyed and provider absence verified. The failed-provisioning receipt
records zero simulator starts and queries. Replacement instance 57544241 is on
baseline machine 54514. It inherits the original allocation deadlines: admissions
close at 21:26:49 UTC and the rental stops at 22:56:49 UTC. The queue now honors the
absolute admission time in the rental receipt; the replacement's watchdog duration
is shortened to the same original shutdown deadline. There is no extra runtime
allocation for the replacement.

The shared tracking analysis was corrected independently in commit `5ed3366`:
the transition into first contact is excluded from the pre-contact reference
window, so both endpoint states precede contact. V3 pins that corrected source.
No result has been inspected to select or change the controller or its panel.

All 17 tests pass; Ruff and shell syntax checks pass. Plan SHA-256:
`6495821a0c18f17d4dbbcd16bbe959d84848481ead4655158e7e2722880d07c4`.
