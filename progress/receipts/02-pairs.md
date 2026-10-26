# Todo 2, item 5: success against failure on the same task

The PNG files were removed from git on 26 October 2026. They remain in the private dataset `Sachioster/maniguard-jar-replication-20261022` under `analysis/02/02-pairs/` at revision `11b5a0f9b91fa51660e09474bfab0f8460c25f51` ([02-preserve.md](02-preserve.md)).

Strips: `02-pairs/task_<nn>.png` (rows: success main, failure main, success wrist, failure
wrist; columns contact - 50 to contact + 250 every 25 steps, each episode aligned on its own
first contact) and `02-pairs/task_<nn>_late.png` (the same rows over absolute steps from
anchor - 250 to anchor + 50, anchor = the earliest of the success lift, the failure violation and
the failure lift), made by `scripts/todo2/pairs.py`. Pair choice is in `02-pairs/pairs.json`:
the same-seed pair wherever it splits (tasks 0, 3, 8, 9, 10, 16, 17, 20, 23, 25), else the
success and the failure with the closest first contact. Every strip was looked at; the notes
are what the frames show.

## Summary

- Before contact the pairs are indistinguishable in 14 of 16. The two exceptions are
  different-seed pairs on pool tasks (1 and 15) where the approach itself differs. The
  same-seed pairs are identical through contact and, in 8 of 10, through the lid push as
  well: the propositions first differ at the closure step, and the pictures first differ
  afterwards.
- The visible split is almost always in the regrasp, the second phase that starts after the
  lid is down: whether the gripper leaves the lid for a top-down view of the jar (tasks 0,
  3, 8, 20), whether it then descends (3, 8, 20) and whether the descent lands on the body
  or on the rim (0, 2, 9). Where the lid push itself is the difference (5, 6, 14, 15, 16),
  the failing episode keeps pushing at a lid that does not go over, for hundreds of steps.
- Three pairs split on a policy choice after contact rather than on physics: in tasks 10,
  17 and 23 the successful repeat gives up on the lid and grasps the body (open), which
  is an unsafe success twice and a loophole success once. Task 25's repeats are identical
  including the crash.
- Verdict per pair, where the difference first becomes visible:

| task | panel | pair | first visible difference |
|---|---|---|---|
| 0 | development | seed0_run3 safe_success vs seed0_run2 tip_after_close_on_table (same-seed pair) | contact (same seed; identical through closure, split at the regrasp descent) |
| 1 | pool | seed2 unsafe_success vs seed0_run2 lift_open (closest contact) | before contact (different seeds; body grasp versus lid push) |
| 2 | held_out | seed2 safe_success vs seed0_run3 tip_after_close_on_table (closest contact) | contact (different seeds, identical until the fingers meet the rim) |
| 3 | development | seed0_run3 safe_success vs seed0_run2 closed_stall (same-seed pair) | after closure (same seed; identical through closure, one retreats from the top grasp) |
| 5 | held_out | seed0_run2 safe_success vs seed2 stall (closest contact) | after closure (different seeds; the push finishes in one and not the other) |
| 6 | development | seed1 safe_success vs seed0_run3 closed_stall (closest contact) | after contact, during the push (different seeds) |
| 8 | held_out | seed0_run3 safe_success vs seed0_run2 closed_stall (same-seed pair) | after closure (same seed; one drifts off the jar instead of descending) |
| 9 | development | seed0_run2 safe_success vs seed0_run3 tip_after_close_on_table (same-seed pair) | after closure (same seed; a lower sideways sweep tips the jar) |
| 10 | pool | seed0_run3 unsafe_success vs seed0_run2 tip_before_close (same-seed pair) | after contact (same seed; one abandons the lid push and grasps the body) |
| 14 | held_out | seed0_run2 safe_success vs seed2 tip_after_close_on_table (closest contact) | after contact, during the push (different seeds) |
| 15 | development | seed0_run2 safe_success vs seed2 closed_stall (closest contact) | before contact in timing, after contact in substance (different seeds; the push) |
| 16 | held_out | seed0_run2 safe_success vs seed0_run3 stall (same-seed pair) | after contact, during the push (same seed; nothing visible differs before) |
| 17 | development | seed0_run2 safe_success vs seed0_run3 unsafe_success (same-seed pair) | after closure (same seed; one abandons the lid and grasps the open jar) |
| 20 | pool | seed0_run3 unsafe_success vs seed0_run2 closed_stall (same-seed pair) | after closure (same seed; descent happens or not) |
| 23 | held_out | seed0_run3 unsafe_success vs seed0_run2 stall (same-seed pair) | after contact, at the lid (same seed; one grasps the body around the open lid) |
| 25 | pool | seed0_run3 unsafe_success vs seed0_run2 tip_before_close (same-seed pair) | nowhere: the repeats are identical including the crash |

- Counted: before contact 2 (both different seeds), at contact or during the push 6, after
  closure in the regrasp 7, nowhere 1. For the same-seed pairs: at the push 2 (10, 16), in
  the regrasp 6 (0, 3, 8, 9, 17, 20), at the lid 1 (23), nowhere 1 (25).
- Reading for the lever. The policy does not have a good and a bad approach mode on these
  tasks; it has one approach, and the outcome is decided by small differences in how the
  push and the regrasp descent go. That is the regime of execution and recovery: a slower or
  shallower descent during the regrasp, a stop when a finger meets the rim, a retreat and
  retry when the lid has not gone over after N steps. Steering between modes before contact
  has nothing to act on in the same-seed pairs.

## Per pair

**Task 0** (development; seed0_run3 safe_success against seed0_run2 tip_after_close_on_table, same-seed pair).
Early strip: the two seed 0 repeats are indistinguishable in both cameras through contact + 250: same overhead descent, same finger placement on the body, and in the wrist view the lid swings over at the same frames and is lying on the jar by c + 175 in both. The monitor calls run2 closed at 361 and run3 only at 1134, although the lid looks shut in both; the hinge must rest just outside the 5 percent closed band in run3.
Late strip: (window 415 to 715, anchor run2's violation at 665): run3 keeps the fingers at the closed lid in one unchanged posture for the whole window (it hovers there until 1134 and lifts at 1529). run2 leaves the lid at about 490: the wrist view rotates, by 515 it looks straight down at the closed jar between the open fingers, the fingers descend over the body from 540 to 640, the jar starts to tilt at 640 and is over at 665. The split is the moment run2 moves on to the top grasp, about 280 steps after contact, 175 steps before the crash; the crash itself is the descent landing on the rim.

**Task 1** (pool; seed2 unsafe_success against seed0_run2 lift_open, closest contact).
Early strip: different seeds and a visible difference before contact. seed2 (unsafe success) comes in from the side with the fingers around the jar body, the lid out of the wrist view, closes the gripper and lifts the open jar at c + 75; run2 (lift open) comes from above onto the lid, pushes it, and the lid disc fills the wrist view for the whole window. Two modes: body grasp and lid push.
Late strip: (183 to 483, anchor seed2's lift at 433): seed2 holds the jar body from above between the fingers through the window and lifts it; run2 has the oversized red lid at the fingers at an angle for the whole window and never gets past it. Two different plans from before contact (different seeds).

**Task 2** (held_out; seed2 safe_success against seed0_run3 tip_after_close_on_table, closest contact).
Early strip: seed2 (success) and run3 (tip after close) touch at 105 and 109 and are indistinguishable through c + 250 in both cameras; both close the lid at about c + 200.
Late strip: (304 to 604, anchor run3's violation at 554): both episodes show the same top-down view of the closed jar between the open fingers from 304 to 479 and the same descent; at 504 run3's jar is tilting under the fingers and at 554 it is over, while seed2's descent at the same frames leaves the jar standing (seed2 lifts it later at 703). Nothing differs until the fingers meet the rim, about 50 steps before the violation.

**Task 3** (development; seed0_run3 safe_success against seed0_run2 closed_stall, same-seed pair).
Early strip: the seed 0 repeats are indistinguishable through c + 250 in both cameras, with the lid pushed shut at the same frame (321 and 323).
Late strip: (564 to 864, anchor run3's lift at 814): at 564 both look straight down at the closed jar between the fingers. run3 descends, closes on the body around 714 and lifts at 814. run2 at 589 has already swung away (the wrist sees the sphere, then the table edge, then sky) and the main view shows the arm folded at the right side of the table for the rest of the window. The retreat starts from the same top-grasp posture, about 440 steps after contact.

**Task 5** (held_out; seed0_run2 safe_success against seed2 stall, closest contact).
Early strip: run2 (success) and seed2 (stall) differ from the first frame of the window: seed2's wrist camera at c - 50 is already at the jar from a lower angle and the contact (115) comes 47 steps earlier; from c + 0 on, both show the fingers at the lid and the lid swinging over at the same relative frames, and by c + 250 both lids are down. Different seeds, same closure, the split comes later.
Late strip: (903 to 1203, anchor run2's lift at 1153): run2 still has the lid at the fingers at 903, rotates away at 978, looks down at the closed jar at 1028, descends and lifts at 1153. seed2 (different seed) has the open lid at the fingers for the entire window, pushing at it without it going over (closure never observed). The difference is that run2's push finished and seed2's did not.

**Task 6** (development; seed1 safe_success against seed0_run3 closed_stall, closest contact).
Early strip: seed1 (success) and run3 (closed stall) are alike through c + 50; at c + 75 run3's lid is observed closed and the wrist camera is already looking at the table from a tilted pose while seed1 is still pushing the lid; from c + 100 run3's main view shows the arm folded low beside the jar and its wrist view shows the lid, jar and finger at odd angles, the beginning of the parked posture. seed1 closes the lid at c + 200 (417) and continues. The difference is visible about 60 steps after contact, during the push.
Late strip: (895 to 1195, anchor seed1's lift at 1145): seed1 is in the top grasp, descends and lifts; run3 (different seed) looks at the sky from a parked arm for the whole window, having retreated at about 250 (seen in the early strip).

**Task 8** (held_out; seed0_run3 safe_success against seed0_run2 closed_stall, same-seed pair).
Early strip: the seed 0 repeats are indistinguishable in both cameras through c + 250; run3 is observed closed at c + 207 (356) and run2 at 491, though the wrist frames at c + 225 and c + 250 show the lid lying on the jar in both.
Late strip: (509 to 809, anchor run3's lift at 759): both start in the top-down posture over the closed jar (run2 at 509 to 559 shows the same view as run3). From 584 run2's view slides off the jar, at 659 the jar is leaving the frame, and from 684 the wrist sees the sphere and then sky as the arm withdraws to the right. run3 descends from the same posture and lifts at 759. The split is at about 580, 430 steps after contact, when run2 drifts off the jar instead of descending.

**Task 9** (development; seed0_run2 safe_success against seed0_run3 tip_after_close_on_table, same-seed pair).
Early strip: the metal jar; the seed 0 repeats are indistinguishable through c + 250 in both cameras, the lid pushed over at the same frames and closed at c + 237 and c + 246.
Late strip: (656 to 956, anchor run2's lift at 906): the metal jar. run2 at 656 has the closed jar in a slanted view, moves around it (the wrist sees sky at 706 to 756), looks down at it from 806, descends between 831 and 881 and lifts at 906. run3 has the same slanted view at 656, but by 681 the jar is already tilted under the fingers, at 706 the wrist sees only sky and the main view shows the arm high with the jar rolling, and from 781 the jar lies on its side (violation recorded at 997 when it passes the tilt limit). The split is at about 680, 515 steps after contact, during a sideways sweep that run3 makes with the fingers lower.

**Task 10** (pool; seed0_run3 unsafe_success against seed0_run2 tip_before_close, same-seed pair).
Early strip: the seed 0 repeats are indistinguishable through c + 250; run3 (unsafe success) has the lid over the jar by c + 150 and run2 (tip before close) the same, but only run2's monitor later records closure (423); run3 lifts the jar open at 1117 without the hinge ever reaching the closed band.
Late strip: (867 to 1167, anchor run3's lift at 1117): run3 looks down at the jar from 917, descends from 942, closes on the body and lifts at 1117 with the lid still open (the open lid rim is visible beside the body in the wrist view). run2 has the open lid at the fingers from 867 to 1167, pushing at it for the entire window without the hinge closing (closure observed only at 1599 after the jar has tipped). Same seed, the split is whether the policy gives up on the lid and grasps the body.

**Task 14** (held_out; seed0_run2 safe_success against seed2 tip_after_close_on_table, closest contact).
Early strip: run2 (success) and seed2 (tip after close) are different seeds with contact 312 and 214; aligned on contact, both show the same overhead descent and the same lid push in the wrist view through c + 250, the lid standing open at the right finger in both at c + 250.
Late strip: (1279 to 1579, anchor run2's lift at 1529): run2 has the closed jar between the fingers from above for the whole window, descends slowly and lifts at 1529. seed2 (different seed) has the lid at the fingers through the window: it is still pushing the lid at 1579 (the lid is observed closed only at 1668) and tips the jar at 1887. The difference is the lid push taking 1,450 steps instead of 260.

**Task 15** (development; seed0_run2 safe_success against seed2 closed_stall, closest contact).
Early strip: run2 (success, contact 237) and seed2 (closed stall, contact 56) are different seeds and differ before contact: seed2 touches the jar on the way down with the wrist camera still inside the sphere mesh (black frames), run2 arrives later from the right. From c + 25 both show the fingers at the lid and the lid being pushed over, and by c + 150 the lid lies on the jar in both; seed2's gripper then stays above the jar.
Late strip: (255 to 555, anchor run2's lift at 505): run2 has the lid pushed by 380, looks down at the jar at 405, descends and lifts at 505. seed2 (different seed) has the lid rim at the fingers from 255 to 505 and the wrist camera goes black at 530 (the gripper is inside the sphere mesh above the jar); the lid is observed closed only at 706 and the arm then stays above the jar. The push is the difference again.

**Task 16** (held_out; seed0_run2 safe_success against seed0_run3 stall, same-seed pair).
Early strip: the seed 0 repeats are indistinguishable through c + 250 in both cameras, the fingers around the open lid at the right finger in both at the end of the window; the monitor records closure for run2 at 371 and never for run3, a difference that is not visible at this resolution.
Late strip: (728 to 1028, anchor run2's lift at 978): run2 looks down at the closed jar between the fingers from 728, descends over the body and lifts at 978. run3 has the open lid at the fingers in the identical hover posture for the whole window, and for the rest of the episode. Same seed; one push went over and the other did not, and nothing in the images before that differs.

**Task 17** (development; seed0_run2 safe_success against seed0_run3 unsafe_success, same-seed pair).
Early strip: the seed 0 repeats are indistinguishable through c + 250, the lid pushed over at the same frames; run2 is observed closed at 513 and run3 never.
Late strip: (574 to 874, anchor run3's lift at 824): at 574 both have the lid at the fingers. run2 keeps pushing the lid through the window (it is observed closed at 513 and the gripper stays at the rim, later dragging the jar into the sphere). run3 at 649 swings away from the lid (the wrist sees the sphere, then the arm link, then sky), comes back looking down at the open jar at 724, descends on the body and lifts it open at 824. Same seed; run3 abandons the lid push and grasps the open jar.

**Task 20** (pool; seed0_run3 unsafe_success against seed0_run2 closed_stall, same-seed pair).
Early strip: the seed 0 repeats are alike through c + 50; at c + 75 run3's lid is already observed closed and its gripper is rising off the jar, while run2 is still pushing the lid, which is observed closed only at c + 175 (377). From c + 100 run3's wrist view is a top-down view of the closed jar and sphere, the start of a grasp from above; run2's is still the lid at the fingers. The split is visible about 60 steps after contact, in how fast the lid went over.
Late strip: (498 to 798, anchor run3's lift at 698): both have the closed jar below the fingers from 498. run3 descends from 598, closes on the body at 673 and lifts at 698; the jar tilts in the gripper at 704 while still over the sphere, which still counts as the goal. run2 holds the same view of the closed jar from above for the whole window without descending and keeps that posture to the horizon. Same seed, the split is whether the descent happens.

**Task 23** (held_out; seed0_run3 unsafe_success against seed0_run2 stall, same-seed pair).
Early strip: the two-jar scene; the seed 0 repeats are indistinguishable through c + 250 in both cameras, the lid of the target pushed over at the same frames; run3 later lifts the open jar (1826) and run2 holds its pose.
Late strip: (1626 to 1926, anchor run3's lift at 1826): the two-jar scene. run3 has the open target jar body between the fingers from 1626, closes and lifts at 1826, the lid hanging open. run2 has the open lid rim at the fingers in one posture for the whole window. Same seed; one holds the lid for 1,800 steps, the other eventually grasps the body around it.

**Task 25** (pool; seed0_run3 unsafe_success against seed0_run2 tip_before_close, same-seed pair).
Early strip: the seed 0 repeats are indistinguishable through c + 250, including the violation itself: the arm sweeps across the jar inside the sphere and tips it at 75 and 77, and the fallen jar rolls the same way in both wrist views. run3's later lift of the fallen jar into the goal (1121) is what makes it an unsafe success.
Late strip: (0 to 127, anchor the pair's violation at 75 and 77): identical frames in both cameras from step 0 to the violation and 50 steps beyond: the arm comes down into the sphere, the fingers sweep the jar and it rolls over at the same frame. This pair is a repeat, not a coin flip; the crash is deterministic on this scene.
