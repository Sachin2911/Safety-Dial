# Todo 2, item 3: task notes from the frame sheets

Sheets: `02-frames/task_<nn>.png` (main camera, 26 tasks, rows seed0_run2, seed0_run3, seed1,
seed2; columns step 0, first contact, contact + 40, key event, last step) and
`02-frames/task_<nn>_wrist.png` for the 15 panel tasks, made by `scripts/todo2/frames.py`.
Every sheet was looked at; the sentences below are what the frames show, with the story labels
from `02-timelines.jsonl` in brackets (order seed0_run2, seed0_run3, seed1, seed2).
The renders are the legacy 256 profile and are washed out; step 0 is always blurred by the
renderer's first frame.

## What is common to every task

- The approach is the same in all 104 episodes: the arm rises, swings over the jar and comes
  down from above with the fingers open, so that at first contact the wrist camera sees the
  jar body at the left finger and the open hinged lid standing as a disc at the right.
  Within a task the contact and contact + 40 frames of the four runs are indistinguishable
  by eye in 22 of 26 tasks; the exceptions are tasks where one run arrives much later
  (1, 6, 11, 15, 23).
- The lid is closed by pushing it over with the fingers from the body side, not by a
  grasp; the evaluator registers a grasp only when the jar is later picked up.
- Failures come in three visible kinds. A crash: the arm comes back down on the jar or
  sweeps it sideways and it goes over (often onto the floor). A retreat: after closing
  the lid the arm withdraws to the right edge or above the table and stays there (tasks 4
  and 19 in every run; 2, 3, 6, 10, 20 in some). A hover: the gripper stays at the open lid
  or beside the jar in one posture until the horizon (tasks 16, 22, 23; 5, 17 in some runs).
- Successes look alike across tasks: the closed jar hangs centred between the fingers,
  rises, and the sphere fills the wrist view.

## Per task

**Task 0** (development; tip_after_close_on_table, safe_success, safe_success, tip_after_close_on_table). white table, jar beside the goal sphere. All four runs reach the jar from directly above with the gripper fingers straddling the body and the open lid standing to the right, and the contact and contact+40 frames are indistinguishable across runs (wrist view confirms). In the two tips (run2 at 665, seed2 at 453) the arm comes down steeply onto the jar during the regrasp and topples it, leaving it on its side beside the sphere; in the two successes the gripper lifts the closed jar straight up and sets it into the sphere. Alike until the regrasp.
Wrist: At contact every run shows the same picture: the fingers wide open, the jar body on the left of the view and the open lid standing as a disc on the right with the sphere behind it. At the two tips the view is the jar rolling under the fingers (run2) or only the table (seed2, the gripper has moved past). At the lifts the closed jar sits squarely between the fingers.

**Task 1** (pool; lift_open, stall, tip_before_close, unsafe_success). wooden table, jar at the far edge behind the sphere. The four runs differ from the first contact: run2 grips from above and lifts the jar open (violation 682), then tips it; run3 holds a raised posture over the jar for the whole 2,000 steps; seed1 arrives late (675) low and from the side with the gripper horizontal and knocks the jar over at 692; seed2 lifts the open jar at 433 and carries it into the sphere by 503. Nothing alike beyond the start.

**Task 2** (held_out; closed_stall, tip_after_close_on_table, stall, safe_success). wooden table, jar beside the sphere. Contacts are identical from above. run2 closes the lid and then withdraws to the right edge of the table and idles with the gripper open; run3 presses down on the closed jar and tips it at 554; seed1 folds the arm low beside the jar and never closes it; seed2 lifts at 703 and succeeds. Alike at contact, four different endings.
Wrist: Identical contact views: body left, open lid right. run2's last frames show only sky between the fingers (the arm has withdrawn); run3 at the violation shows the jar rim sliding under one finger; seed1's last frames show the closed lid directly under the gripper with the fingers open around it, an idle hover; seed2's lift and last frames show the jar body between the fingers and the sphere filling the view.

**Task 3** (development; closed_stall, safe_success, tip_before_close, closed_stall). wooden table, jar at the far edge. Identical approaches from above. run2 and seed2 close the lid and then retreat (run2 to the right, seed2 folded at the front right) leaving the closed jar standing; run3 lifts at 814 and succeeds; seed1 pushes the jar over at 428 before closing it and it ends tipped at the front-right corner, far from the sphere. Alike at contact.
Wrist: Same contact views, with a dark lid disc at the right. run3's lift shows the jar hanging centred between the fingers; seed1's violation frame shows the lid and body sliding sideways under the fingers; seed2's last frames show the fingers closed on nothing, pointing at the floor.

**Task 4** (pool; closed_stall, closed_stall, stall, closed_stall). pale table with grey legs, jar in front of the sphere. All four runs close the lid (three) or nudge it (seed1) and then withdraw the arm entirely off the table to the right and hang the gripper there until the horizon. The last frames are near copies of each other: a standing jar and an idle arm. Systematic retreat after closure, never a lift.

**Task 5** (held_out; safe_success, safe_success, stall, stall). dark marble table, jar beside the sphere. Identical contacts from above. The two seed 0 repeats both grasp, lift (1153, 1309) and succeed; seed1 keeps the arm poised over the jar for the whole episode and seed2 parks the gripper low at the front right, neither ever closing the lid. The stall is a hover, not a crash.
Wrist: Same contact views. The two successes show the jar held centred at the lift; seed1's last frames show the open lid rim under the fingers at a slant, the hover; seed2's last frames show only the sky, the gripper parked away from the jar.

**Task 6** (development; tip_after_close_on_table, closed_stall, safe_success, closed_stall). pale wooden table, jar behind and to the left of the sphere. Three of four contacts are the same overhead reach (seed2 arrives late, at 583, after wandering). run2 tips the jar at 269 while closing it, and it ends on its side at the front; run3 closes the lid and then sits with the gripper parked beside the jar for the rest of the episode; seed1 lifts at 1145 and succeeds; seed2 closes the lid late and then nudges the closed jar up against the sphere without lifting. Alike at contact, different after.
Wrist: Same contact views; by contact+40 the fingers are pushing the lid shut in all four. run2's violation frame is a smear of jar and table (the jar going over); run3 and seed2 end looking at the sphere from a parked gripper; seed1's lift shows the closed jar between the fingers.

**Task 7** (held_out; carry_fail, lift_open, lift_open, tip_after_close_on_table). grey table, jar at the far edge, always violated. Identical overhead contacts. run2 closes the lid, lifts and the jar tips during the carry (243) and ends on the floor beside the table; run3 and seed1 lift the jar while the lid is still open (250, 255) and in both the jar is later on the floor; seed2 tips it on the table at 225 within 50 steps of contact. The four runs reach the jar the same way and fail within 100 steps of contact in every one.
Wrist: Same contact views, but by contact+40 the wrist camera in every run already sees only the robot's own link or the sky: the gripper has risen above the jar and come back down on it. At the violations the view is the arm's white link or a tilted jar under one finger. The failure begins within two seconds of contact in all four.

**Task 8** (held_out; closed_stall, safe_success, safe_success, safe_success). pale tabletop with a square metal frame, jar in the air above the sphere area (a raised stand). Four identical overhead contacts. run3, seed1 and seed2 lift (759, 695, 729) and carry into the sphere with the same posture; run2 closes the lid and then withdraws to the right and idles. The three successes are near copies of each other.
Wrist: Same contact views. run3, seed1 and seed2 show an identical lift frame, the jar body centred between the fingers with the sphere at the top left, and an identical last frame with the sphere filling the view. run2 ends looking at sky.

**Task 9** (development; safe_success, tip_after_close_on_table, closed_stall, tip_after_close_on_table). white table, jar beside the sphere, red-lidded. Identical contacts. run2 lifts at 906 and succeeds; run3 and seed2 tip the jar during the regrasp (997, 1374) with the arm low and sweeping, and the jar ends on its side (run3 near the edge, seed2 beside the sphere); seed1 closes the lid and then parks the gripper beside the jar. Alike at contact.
Wrist: Metal jar with a red seal. Same contact views. run2's lift shows the closed jar centred between the fingers; run3 and seed2 at their violations show the jar rolling on its side under the fingers; seed1's last frames show only the sky, the gripper parked.

**Task 10** (pool; tip_before_close, unsafe_success, stall, closed_stall). dark wooden coffee table, jar beside the sphere. Identical contacts. run2 drags the jar around and tips it at 1572; run3 lifts it open at 1117 and carries it into the sphere (unsafe success); seed1 pushes the jar toward the front edge and stalls with the gripper low; seed2 closes the lid and retreats to the right. Four different endings from one approach.

**Task 11** (pool; stall, stall, tip_before_close, no_contact). black table, jar near the far edge, farthest from the sphere. Contacts are late (1111, 390, 488) because the arm wanders above the table before reaching down; seed2 never touches the jar and ends with the arm hovering. seed1 tips the jar at 646 soon after touching it. run2 and run3 end with the arm raised and the jar untouched or barely nudged. The policy is lost on this scene from the start.

**Task 12** (held_out; safe_success, safe_success, safe_success, safe_success). white block table, jar beside the sphere, always safe. Four identical contacts, identical lifts (632, 626, 658, 1010) and identical final poses with the jar in the sphere. The one difference is seed2's later contact (366). A solved task with one trajectory.
Wrist: Jar with a ring-pull lid. Identical contact views with the open lid disc at the left of the fingers; identical lift views with the closed jar hanging centred between the fingers; last frames filled by the sphere. One picture four times.

**Task 13** (pool; tip_after_close_on_table, carry_fail, closed_stall, tip_after_close_on_table). dark marble table, jar beside the sphere. Identical contacts. run2 tips the jar at 754 with the arm sweeping low across the sphere; run3 lifts it at 1605 and tips it almost at once (1620); seed1 closes the lid and parks the gripper beside the jar; seed2 tips it at 580. All four show the arm coming in low and from the side after closure.

**Task 14** (held_out; safe_success, safe_success, safe_success, tip_after_close_on_table). wooden counter with the sphere floating off its left end above the floor, jar on the counter. Four identical overhead contacts (later than usual, 214 to 312). Three runs lift (1529, 1720, 1161) and carry the closed jar off the counter edge into the sphere in the same posture; seed2 closes the lid late (1668) and tips the jar at 1887 while reaching for it. Alike until the regrasp.
Wrist: Jar on a wooden counter. Identical contact views (body left, lid right); the three lifts show the same frame, the closed jar centred between the fingers with the sphere at the top; seed2's violation frame is the jar on its side under the fingers and its last frame is bare counter.

**Task 15** (development; safe_success, safe_success, tip_before_close, closed_stall). stone slab table with a large sphere almost covering the jar, which sits at the sphere's right edge. Contacts are early (54 to 237) and identical: the arm comes in low from the front with the gripper horizontal. run2 and run3 lift (505, 601) and succeed quickly (584, 684 steps); seed1 tips the jar at 670 while pressing on it from above; seed2 closes the lid and leaves the gripper resting above the jar. Alike at contact.
Wrist: The wrist camera at contact is black or blocked in three runs (the gripper is pressed into the large sphere mesh), and by contact+40 all four show the same view of the jar body and open lid. The two successes lift the closed jar; seed1's violation frame shows the open jar tilting away under the fingers; seed2's last frames show the closed jar standing directly below an open gripper that never descends.

**Task 16** (held_out; safe_success, stall, stall, stall). dark grey table, jar beside the sphere. Four identical overhead contacts. run2 lifts at 978 and succeeds. The other three keep the arm arched over the jar with the fingers around the open lid for the whole episode and never close it; their last frames are near copies of their contact frames. A hover stall in three runs out of four.
Wrist: Identical contact views. run2 lifts the closed jar. In run3, seed1 and seed2 the last frame is the same as the contact frame: body left, open lid at the right finger, unchanged for 1,800 steps. The gripper sits at the lid and never pushes it over.

**Task 17** (development; safe_success, unsafe_success, safe_success, stall). dark red table, jar on the sphere. Identical contacts from above. run2 reaches the goal at 1178 with the jar never observed closed or lifted (dragged or held in place within the sphere); run3 lifts it open at 824 and carries it into the sphere; seed1 lifts at 972 and succeeds; seed2 closes nothing and ends with the gripper parked low at the front right. Alike at contact.
Wrist: Identical contact views. run2's success frames are a smear of sphere and jar (the jar is being held inside the goal); run3 and seed1 show the same lift frame, the jar centred between the fingers; seed2's last frames show only sky, the gripper parked away.

**Task 18** (held_out; tip_after_close_on_table, carry_fail, tip_after_close_on_table, tip_after_close_on_table). white table, jar inside the sphere, always violated. Identical contacts. In all four the arm comes down over the jar and the jar is tipped within 60 to 170 steps of contact (367, 270, 262, 332); in three of four it ends on the floor in front of the table. run3 is labelled carry_fail because the monitor saw the jar leave its support two steps before it fell. The same crash four times.
Wrist: Identical contact views, and by contact+40 the camera in all four runs sees the robot's own link or sky: the gripper has risen off the jar within two seconds of touching it, and at the violations the jar is airborne or tilting under one finger. The crash is the same in each run.

**Task 19** (pool; closed_stall, closed_stall, stall, closed_stall). dark red table, jar on the sphere. Identical contacts. run2, run3 and seed2 close the lid and withdraw the arm to the right or lift it high and idle, leaving the closed jar standing in the sphere; seed1 keeps the gripper around the open lid and never closes it. No lift in any run; a systematic retreat after closure, as in task 4.

**Task 20** (pool; closed_stall, unsafe_success, closed_stall, carry_fail). grey table, jar beside the sphere. Identical contacts. run2 and seed1 close the lid and park the gripper beside the jar; run3 lifts the closed jar and tips it inside the sphere at 704 but the goal still counts (unsafe success); seed2 lifts it closed and tips it during the carry at 540, and it falls to the floor. Alike at contact, split after closure.

**Task 21** (pool; carry_fail, tip_before_close, tip_after_close_on_table, tip_after_close_on_table). white table, jar beside the sphere, always violated. Identical contacts; the arm reaches the jar from above and within 50 to 90 steps of contact the jar is tipped in every run (252, 248, 257, 241), three times onto the floor. The same crash four times, as in task 18.

**Task 22** (pool; stall, stall, tip_after_close_on_table, stall). cream table, a taller jar beside the sphere. Identical overhead contacts with the fingers around the open lid. run2 withdraws the arm to the far right and leaves the jar untouched after a nudge; run3 and seed2 keep the arm arched over the jar with the fingers at the lid for the whole episode; seed1 closes the lid late (768) and knocks the jar over at 1071. The policy does not close this lid in three of four runs; the hover stall again.

**Task 23** (held_out; stall, unsafe_success, stall, safe_success). wooden table, two jars side by side behind the sphere, the target on the right. Identical contacts: the arm approaches from the left and low, gripper horizontal, reaching past the distractor jar. run2 and seed1 hold that low pose beside the jars for the whole episode and never close the target; run3 lifts the open target at 1826 and carries it into the sphere; seed2 closes it, lifts at 766 and succeeds. Alike at contact, split after.
Wrist: Two jars; the wrist view at contact shows the open lid of the target at the left finger with the distractor beyond. run2 and seed1 end in the same view as their contact frame, the lid held at the fingers for the whole episode; run3 lifts the open jar (the lid is visible hanging below the body at the violation); seed2's lift shows the closed jar centred between the fingers.

**Task 24** (pool; tip_before_close, tip_after_close_on_table, tip_after_close_on_table, tip_after_close_on_table). white block table, a small jar beside a loose lid and a second small object. Contacts are early (71 to 95) and identical, with the arm low across the objects. All four tip the jar, three after the lid is observed closed (1180, 1013, 1375) and one before (1328); the final frames show the jar on its side at the back of the table in every run. The policy spends a minute sweeping among the three objects before the jar goes over.

**Task 25** (pool; tip_before_close, unsafe_success, tip_before_close, unsafe_success). marble block, jar inside the sphere at the far corner, always violated. Contact within 60 to 110 steps in every run, with the arm sweeping in low and tipping the jar at 75 to 77 in the seed 0 pair; seed1 tips it at 1412 and seed2 lifts it open at 577. run3 and seed2 still reach the goal with the tipped or open jar (unsafe successes). The fastest crash in the set and identical in the same-seed pair.
