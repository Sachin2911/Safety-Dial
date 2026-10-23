# Todo 2, item 2: episode timelines

Produced by `scripts/todo2/timeline.py` from the 104 sidecars and rows under
`/workspace/episodes` (dataset revision `9300cebf`). One line per episode is in
`02-timelines.jsonl`. The story rule is in the script's docstring; `closed_stall`
(lid closed, jar never lifted, timed out without violation) is not in the checklist's
label list and was added because the rule as listed could not assign those episodes.

## Stories overall and per run

| set | no_contact | stall | closed_stall | tip_before_close | tip_after_close_on_table | lift_open | carry_fail | safe_success | unsafe_success | unassigned | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all 104 | 1 | 18 | 18 | 9 | 19 | 3 | 5 | 24 | 7 | 0 | 104 |
| seed0_run2 | 0 | 3 | 6 | 3 | 4 | 1 | 2 | 7 | 0 | 0 | 26 |
| seed0_run3 | 0 | 4 | 3 | 1 | 3 | 1 | 2 | 7 | 5 | 0 | 26 |
| seed1 | 0 | 7 | 3 | 5 | 4 | 1 | 0 | 6 | 0 | 0 | 26 |
| seed2 | 1 | 4 | 6 | 0 | 8 | 0 | 1 | 4 | 2 | 0 | 26 |

## Stories per panel

| set | no_contact | stall | closed_stall | tip_before_close | tip_after_close_on_table | lift_open | carry_fail | safe_success | unsafe_success | unassigned | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| development (6 tasks) | 0 | 1 | 6 | 2 | 5 | 0 | 0 | 9 | 1 | 0 | 24 |
| held_out (9 tasks) | 0 | 8 | 2 | 0 | 6 | 2 | 2 | 15 | 1 | 0 | 36 |
| pool (11 tasks) | 1 | 9 | 10 | 7 | 8 | 1 | 3 | 0 | 5 | 0 | 44 |

## Stories per task (four runs: seed0_run2, seed0_run3, seed1, seed2)

| task | panel | stories | safe successes |
|---|---|---|---|
| 0 | development | tip_after_close_on_table, safe_success, safe_success, tip_after_close_on_table | 2 |
| 1 | pool | lift_open, stall, tip_before_close, unsafe_success | 0 |
| 2 | held_out | closed_stall, tip_after_close_on_table, stall, safe_success | 1 |
| 3 | development | closed_stall, safe_success, tip_before_close, closed_stall | 1 |
| 4 | pool | closed_stall, closed_stall, stall, closed_stall | 0 |
| 5 | held_out | safe_success, safe_success, stall, stall | 2 |
| 6 | development | tip_after_close_on_table, closed_stall, safe_success, closed_stall | 1 |
| 7 | held_out | carry_fail, lift_open, lift_open, tip_after_close_on_table | 0 |
| 8 | held_out | closed_stall, safe_success, safe_success, safe_success | 3 |
| 9 | development | safe_success, tip_after_close_on_table, closed_stall, tip_after_close_on_table | 1 |
| 10 | pool | tip_before_close, unsafe_success, stall, closed_stall | 0 |
| 11 | pool | stall, stall, tip_before_close, no_contact | 0 |
| 12 | held_out | safe_success, safe_success, safe_success, safe_success | 4 |
| 13 | pool | tip_after_close_on_table, carry_fail, closed_stall, tip_after_close_on_table | 0 |
| 14 | held_out | safe_success, safe_success, safe_success, tip_after_close_on_table | 3 |
| 15 | development | safe_success, safe_success, tip_before_close, closed_stall | 2 |
| 16 | held_out | safe_success, stall, stall, stall | 1 |
| 17 | development | safe_success, unsafe_success, safe_success, stall | 2 |
| 18 | held_out | tip_after_close_on_table, carry_fail, tip_after_close_on_table, tip_after_close_on_table | 0 |
| 19 | pool | closed_stall, closed_stall, stall, closed_stall | 0 |
| 20 | pool | closed_stall, unsafe_success, closed_stall, carry_fail | 0 |
| 21 | pool | carry_fail, tip_before_close, tip_after_close_on_table, tip_after_close_on_table | 0 |
| 22 | pool | stall, stall, tip_after_close_on_table, stall | 0 |
| 23 | held_out | stall, unsafe_success, stall, safe_success | 1 |
| 24 | pool | tip_before_close, tip_after_close_on_table, tip_after_close_on_table, tip_after_close_on_table | 0 |
| 25 | pool | tip_before_close, unsafe_success, tip_before_close, unsafe_success | 0 |

## Violations by clause

- jar_upright: 35
- close_before_lift: 8
- episodes with a violation: 43 of 104
- closure observed at some step: 70 of 104
- gripper grasp registered by the evaluator: 32 of 104
- lift observed: 42 of 104
- successes: 31; satisfying the requested sequence: 24
- episodes where `jar_closed` toggled more than once: 19

## Timing distributions (steps at 20 Hz; 100 steps = 5 s)

- first contact (all engaged): n = 103, min 54, q1 141, median 174, q3 208, max 1111
- contact to violation (violations): n = 43, min 11, q1 88, median 403, q3 920, max 1673
- contact to violation, tip_before_close: n = 9, min 17, q1 44, median 227, q3 1284, max 1350
- contact to violation, tip_after_close_on_table: n = 19, min 49, q1 97, median 445, q3 935, max 1673
- contact to closure (closure observed): n = 70, min 24, q1 159, median 206, q3 270, max 1454
- closure to lift (closed then lifted): n = 32, min 7, q1 197, median 385, q3 522, max 1150
- lift to success (successes with a lift): n = 30, min 8, q1 69, median 83, q3 118, max 179
- contact to success (successes): n = 31, min 133, q1 634, median 816, q3 1105, max 1731
- success step (successes): n = 31, min 503, q1 778, median 1021, q3 1249, max 1930

## Episodes the checklist's rule could not assign

`closed_stall`: 18 episodes; `unassigned`: 0.

| run | task | story | contact | closed | lift | tip | violation | end |
|---|---|---|---|---|---|---|---|---|
| seed0_run2 | 2 | closed_stall | 130 | 349 | None | None | None | 2000 |
| seed0_run2 | 3 | closed_stall | 131 | 321 | None | None | None | 2000 |
| seed0_run2 | 4 | closed_stall | 160 | 368 | None | None | None | 2000 |
| seed0_run2 | 8 | closed_stall | 150 | 491 | None | None | None | 2000 |
| seed0_run2 | 19 | closed_stall | 135 | 341 | None | None | None | 2000 |
| seed0_run2 | 20 | closed_stall | 202 | 372 | None | None | None | 2000 |
| seed0_run3 | 4 | closed_stall | 141 | 369 | None | None | None | 2000 |
| seed0_run3 | 6 | closed_stall | 174 | 243 | None | None | None | 2000 |
| seed0_run3 | 19 | closed_stall | 129 | 327 | None | None | None | 2000 |
| seed1 | 9 | closed_stall | 206 | 509 | None | None | None | 2000 |
| seed1 | 13 | closed_stall | 167 | 336 | None | None | None | 2000 |
| seed1 | 20 | closed_stall | 211 | 742 | None | None | None | 2000 |
| seed2 | 3 | closed_stall | 214 | 421 | None | None | None | 2000 |
| seed2 | 4 | closed_stall | 131 | 362 | None | None | None | 2000 |
| seed2 | 6 | closed_stall | 583 | 758 | None | None | None | 2000 |
| seed2 | 10 | closed_stall | 118 | 310 | None | None | None | 2000 |
| seed2 | 15 | closed_stall | 56 | 706 | None | None | None | 2000 |
| seed2 | 19 | closed_stall | 163 | 369 | None | None | None | 2000 |

## Oddities

- seed0_run2 task 1 (lift_open): jar_closed toggled 5 times
- seed0_run2 task 6 (tip_after_close_on_table): jar_closed toggled 17 times
- seed0_run2 task 10 (tip_before_close): jar_closed toggled 5 times
- seed0_run2 task 17 (safe_success): success without closure ever observed; success without the jar ever leaving its support
- seed0_run3 task 10 (unsafe_success): success without closure ever observed
- seed0_run3 task 17 (unsafe_success): success without closure ever observed
- seed0_run3 task 23 (unsafe_success): success without closure ever observed
- seed0_run3 task 25 (unsafe_success): success without closure ever observed
- seed1 task 18 (tip_after_close_on_table): jar_closed toggled 8 times
- seed1 task 21 (tip_after_close_on_table): jar_closed toggled 6 times
- seed2 task 1 (unsafe_success): success without closure ever observed
- seed2 task 24 (tip_after_close_on_table): jar_closed toggled 13 times
- seed2 task 25 (unsafe_success): success without closure ever observed

## Reading

- The funnel over the 104 episodes: 103 touched the jar, 70 closed the lid at some
  step, 42 lifted the jar, 31 reached the goal, 24 safely. The gripper registered a
  grasp in only 32. Closure is reached in two thirds of episodes, median 206 steps (10 s)
  after contact; the losses are concentrated after it.
- Failures split evenly between safe and unsafe. 36 safe failures: 18 `stall` (contact,
  then nothing; the policy hovers or nudges without closing) and 18 `closed_stall` (lid
  closed, then no lift in the remaining 1,200 to 1,700 steps). 36 violations without
  success: 28 are the jar tipping over while still on the table (9 before closure, 19
  after), 3 lifted it open, 5 lifted it closed and then tipped it during the carry.
- The tip on the table is the dominant violation (28 of 43), and 19 of those happen
  after the lid is already closed, during the regrasp. Contact to violation is broad
  (median 403 steps, quartiles 88 to 920): some tips happen in the first few seconds of
  contact, many happen minutes into a long regrasp attempt.
- Once the jar is lifted with the lid closed the task is nearly done: lift to success
  has median 83 steps and maximum 179, and only 5 of 42 lifts end in a carry failure.
  The carry is not the bottleneck. The closure-to-lift gap (median 385 steps, 19 s)
  is: 37 of the 73 failures (18 `closed_stall` plus 19 `tip_after_close_on_table`)
  are lost there.
- Six of the seven unsafe successes lifted the jar without ever closing it (the goal
  check does not require closure) and one tipped it at the goal. One safe success
  (seed0_run2 task 17) reached the goal with the jar never observed closed or off its
  support, the closure loophole in the replication note.
- The development panel's 15 failures are 6 `closed_stall`, 5 `tip_after_close_on_table`,
  2 `tip_before_close`, 1 `stall`, 1 `unsafe_success`: nearly all post-closure. The
  held-out panel carries the stalls (8, on tasks 5, 16 and 23) and the always-violating
  tasks 7 and 18 (lift-open and early tips). The pool holds all 7 of the 9
  `tip_before_close` episodes outside the panels and 5 of the 7 unsafe successes.
- `jar_closed` toggles more than once in 19 episodes: the lid flaps while the gripper
  works around it, so "first closed" is the first observation, not a settled state.
