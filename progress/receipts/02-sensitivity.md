# Todo 2, item 4: sensitivity map

Produced by `scripts/todo2/sensitivity.py` from `02-timelines.jsonl` and the sidecars and
main videos of the same-seed pair (`seed0_run2`, `seed0_run3`). Class rule in the script
docstring. `AP div` is the first step at which the pair's proposition vectors differ; `video
div` is the first step at which the mean absolute pixel difference between the pair's main
videos exceeds three times its mean over steps 0 to 20 (`base`, in 8-bit units).

| task | panel | stories (run2, run3, seed1, seed2) | pair agree | all agree | coarse agree | key events | key spread | violation steps | AP div | video div | video div 1.5x | base | pre-contact ratio | contact (pair) | class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | development | tip_after_close_on_table, safe_success, safe_success, tip_after_close_on_table | no | no | no | 665, 1529, 1908, 453 | 1455 | 665, 453 | 361 | 742 | 20 | 1.267 | 0.66 | 208, 193 | knife_edge |
| 1 | pool | lift_open, stall, tip_before_close, unsafe_success | no | no | no | 682, 2000, 692, 433 | 1567 | 682, 692, 433 | 682 | 980 | 640 | 1.046 | 1.06 | 254, 229 | knife_edge |
| 2 | held_out | closed_stall, tip_after_close_on_table, stall, safe_success | no | no | no | 2000, 554, 2000, 703 | 1446 | 554 | 345 | 533 | 505 | 1.059 | 0.64 | 130, 109 | knife_edge |
| 3 | development | closed_stall, safe_success, tip_before_close, closed_stall | no | no | no | 2000, 814, 428, 2000 | 1572 | 428 | 321 | 505 | 29 | 1.008 | 0.88 | 131, 126 | knife_edge |
| 4 | pool | closed_stall, closed_stall, stall, closed_stall | yes | no | yes | 2000, 2000, 2000, 2000 | 0 | - | 368 | 564 | 34 | 1.001 | 0.85 | 160, 141 | mixed |
| 5 | held_out | safe_success, safe_success, stall, stall | yes | no | no | 1153, 1309, 2000, 2000 | 847 | - | 778 | 964 | 19 | 1.454 | 0.58 | 162, 171 | mixed |
| 6 | development | tip_after_close_on_table, closed_stall, safe_success, closed_stall | no | no | no | 269, 2000, 1145, 2000 | 1731 | 269 | 243 | 306 | 36 | 1.07 | 0.78 | 172, 174 | knife_edge |
| 7 | held_out | carry_fail, lift_open, lift_open, tip_after_close_on_table | no | no | yes | 243, 250, 255, 225 | 30 | 243, 250, 255, 225 | 204 | 225 | 28 | 1.153 | 0.68 | 163, 158 | knife_edge |
| 8 | held_out | closed_stall, safe_success, safe_success, safe_success | no | no | no | 2000, 759, 695, 729 | 1305 | - | 356 | 540 | 407 | 0.935 | 0.73 | 150, 149 | knife_edge |
| 9 | development | safe_success, tip_after_close_on_table, closed_stall, tip_after_close_on_table | no | no | no | 906, 997, 2000, 1374 | 1094 | 997, 1374 | 406 | 690 | 37 | 0.814 | 0.73 | 169, 165 | knife_edge |
| 10 | pool | tip_before_close, unsafe_success, stall, closed_stall | no | no | no | 1572, 1117, 2000, 2000 | 883 | 1572, 1117 | 1117 | 864 | 12 | 1.123 | 0.78 | 222, 197 | knife_edge |
| 11 | pool | stall, stall, tip_before_close, no_contact | yes | no | no | 2000, 2000, 646, 2000 | 1354 | 646 | None | 393 | 22 | 1.185 | 2.85 | 1111, 390 | mixed |
| 12 | held_out | safe_success, safe_success, safe_success, safe_success | yes | yes | yes | 632, 626, 658, 1010 | 384 | - | 307 | 456 | 22 | 0.924 | 1.09 | 126, 151 | mixed |
| 13 | pool | tip_after_close_on_table, carry_fail, closed_stall, tip_after_close_on_table | no | no | no | 754, 1620, 2000, 580 | 1420 | 754, 1620, 580 | 423 | 665 | 21 | 1.064 | 0.86 | 169, 169 | knife_edge |
| 14 | held_out | safe_success, safe_success, safe_success, tip_after_close_on_table | yes | no | no | 1529, 1720, 1161, 1887 | 726 | 1887 | 572 | 1596 | 25 | 1.301 | 0.95 | 312, 290 | mixed |
| 15 | development | safe_success, safe_success, tip_before_close, closed_stall | yes | no | no | 505, 601, 670, 2000 | 1495 | 670 | 367 | 443 | 40 | 1.108 | 1.0 | 237, 56 | mixed |
| 16 | held_out | safe_success, stall, stall, stall | no | no | no | 978, 2000, 2000, 2000 | 1022 | - | 371 | 612 | 27 | 0.997 | 0.75 | 182, 189 | knife_edge |
| 17 | development | safe_success, unsafe_success, safe_success, stall | no | no | no | 1178, 824, 972, 2000 | 1176 | 824 | 824 | 660 | 21 | 1.129 | 0.73 | 201, 208 | knife_edge |
| 18 | held_out | tip_after_close_on_table, carry_fail, tip_after_close_on_table, tip_after_close_on_table | no | no | yes | 367, 270, 262, 332 | 105 | 367, 270, 262, 332 | 268 | 241 | 212 | 0.934 | 0.59 | 199, 177 | knife_edge |
| 19 | pool | closed_stall, closed_stall, stall, closed_stall | yes | no | yes | 2000, 2000, 2000, 2000 | 0 | - | 327 | 546 | 30 | 1.244 | 0.67 | 135, 129 | mixed |
| 20 | pool | closed_stall, unsafe_success, closed_stall, carry_fail | no | no | no | 2000, 704, 2000, 540 | 1460 | 704, 540 | 270 | 285 | 24 | 0.92 | 0.65 | 202, 201 | knife_edge |
| 21 | pool | carry_fail, tip_before_close, tip_after_close_on_table, tip_after_close_on_table | no | no | yes | 252, 248, 257, 241 | 16 | 252, 248, 257, 241 | 244 | 259 | 225 | 0.866 | 0.66 | 202, 177 | knife_edge |
| 22 | pool | stall, stall, tip_after_close_on_table, stall | yes | no | no | 2000, 2000, 1071, 2000 | 929 | 1071 | None | 972 | 23 | 0.801 | 1.41 | 182, 153 | mixed |
| 23 | held_out | stall, unsafe_success, stall, safe_success | no | no | no | 2000, 1826, 2000, 766 | 1234 | 1826 | 1826 | 814 | 35 | 0.997 | 1.88 | 239, 199 | knife_edge |
| 24 | pool | tip_before_close, tip_after_close_on_table, tip_after_close_on_table, tip_after_close_on_table | no | no | yes | 1328, 1180, 1013, 1375 | 362 | 1328, 1180, 1013, 1375 | 310 | 270 | 30 | 1.035 | 1.53 | 95, 94 | knife_edge |
| 25 | pool | tip_before_close, unsafe_success, tip_before_close, unsafe_success | no | no | yes | 77, 75, 1412, 577 | 1337 | 77, 75, 1412, 577 | 75 | 168 | 21 | 1.015 | 1.21 | 60, 64 | knife_edge |

## Counts

- knife_edge: 18 tasks: 0, 1, 2, 3, 6, 7, 8, 9, 10, 13, 16, 17, 18, 20, 21, 23, 24, 25
- systematic: 0 tasks: none
- mixed: 8 tasks: 4, 5, 11, 12, 14, 15, 19, 22
- development: knife_edge 5, systematic 0, mixed 1
- held_out: knife_edge 6, systematic 0, mixed 3
- pool: knife_edge 7, systematic 0, mixed 4
- all four runs share a coarse outcome (safe success / safe failure / violation) in 8 of 26 tasks: 4 (safe_failure), 7 (violation), 12 (safe_success), 18 (violation), 19 (safe_failure), 21 (violation), 24 (violation), 25 (violation)
- same-seed pair agrees on the story in 8 of 26 tasks; all four runs agree in 1
- pair video divergence: 26 of 26 tasks diverge; steps min 168, median 543, max 1596
- pair AP divergence: 24 of 26 tasks differ; steps min 75, median 358, max 1826
- pair contact steps differ by: min 0, median 12, max 721

## Reading

- No task is `systematic` under the checklist's rule, because the rule needs all four
  story labels equal and the key events within 50 steps. The story labels are fine
  grained (a crash two steps after a momentary lift is `carry_fail`, two steps before it
  is `tip_after_close_on_table`), so the rule is strict. Under the coarse outcome (safe
  success, safe failure, violation) eight tasks are systematic: 7, 18, 21, 24 and 25
  violate in every run, 4 and 19 close the lid and retreat in every run, and 12 succeeds
  in every run. Tasks 7, 18 and 21 also violate within 100 steps of contact in every run
  (key spreads 30, 105 and 16), so they are systematic crashes in substance.
- 18 tasks are `knife_edge`: the same policy seed, the same episode seed and the same
  machine gave a different story on the second run. Among panel tasks: development 0, 3,
  6, 9, 17 (five of six; 15 is `mixed` because both repeats succeeded while the other
  seeds did not); held-out 2, 8, 16, 23 and the always-violating 7 and 18.
- Where the repeats diverge: first contact differs by a median of 12 steps (0.6 s), and
  the pre-contact ratio is at or below 1.0 in 20 of 26 tasks, so the two main videos are
  as alike during the approach as they are at the start. The propositions first differ
  at a median of 358 steps, which in most tasks is the step at which one repeat closes
  the lid and the other has not yet (closure is the first proposition to change). The 3x
  pixel divergence comes later still, median 543 steps, and after contact in every task:
  the arm's path is the same through the approach and often through the closure, and
  the images only separate when the jar itself ends up in a different state. The 1.5x
  column mostly fires inside the renderer's start transient (steps 12 to 40) and is not
  informative; it is left in the table for the record.
- Implication. The variation between repeats is not in the approach, so an observation
  side or sampling side push that only changes the approach has little to act on; the
  split happens at and after contact, in how the fingers meet the lid and the jar during
  closure and regrasp. A small push there is a coin toss across 18 tasks and would need
  the 200-episode budgets of the replication note to be measured. A systematic
  correction has clear targets: the eight coarse-systematic tasks fail or succeed the
  same way every time, and the three fast crashes (7, 18, 21) fail within five seconds of
  contact in the same posture, which is where an execution-time rule (slower descent,
  stop on contact force, shorter chunk) would act identically on every repeat.
