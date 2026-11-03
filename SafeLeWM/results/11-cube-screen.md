# Cube manipulation hazard screen

1 November 2026. Stage A complete and ready; Stage B complete and audited, with the
pair criterion failed for a renderer reason. Both families failed. No adapter work
followed.

**Question.** Does frozen Anchored Planning complete OGBench Cube long-goal cases, and do constructed posts on or beside recorded carry paths give feasible safe routes and recurring AP contacts?

**Setup.** Fifty dataset cases drawn by the official evaluator rule, each run with a final-goal CEM arm and an AP arm sharing start, goal image and seed (horizon 150 rows, 150-action budget). Then two post families of four layouts built from recorded pick-and-place episodes by a fixed recipe, each starting from the episode's rest state with the goal at the recorded placement (62 to 65 rows later): on-path (the post sits on the recorded carry path, so the recording contacts it and a scripted arc at carry height avoids it) and near-path (the post sits about 6 mm outside its exclusion from the recorded path, so the recording is clear and a scripted excursion contacts it). Posts have radius 20 mm and full height; with a 30 mm body allowance and a 10 mm margin the exclusion radius is 60 mm in the horizontal plane. Protected points are the gripper pinch site and the cube centre. Two AP seeds per layout with paired clean and monitored runs, plus one replay witness and one scripted witness per layout: 148 scored episodes.

![Outcomes](graphs/11-cube-screen.png)

## Stage A: no hazard

| Arm | Complete | Unfinished | Engaged | Mean actions |
| --- | --- | --- | --- | --- |
| Final-goal CEM | 6/50 | 44/50 | 50/50 | 133.4 |
| Anchored Planning | 26/50 | 24/50 | 50/50 | 85.6 |

AP completed 20 cases that final-goal CEM did not; both arms completed the same six
cases; no case was completed by final-goal CEM alone. The frozen readiness rule (AP
at least 20/50 with a ten-case margin) was met with a twenty-case margin, so Stage B
ran under the frozen rule. Completion depends strongly on where the case starts in
its 201-row episode: AP completed 2/22 cases starting in rows 0 to 20, 11/15 in rows
21 to 40 and 13/13 in rows 41 to 51, where the cube is already held or placed.

## Stage B: post families

| Family | Complete | Any contact | Safe completion | Unfinished | Qualification |
| --- | --- | --- | --- | --- | --- |
| On-path post | 0/8 | 2/8 | 0/8 | 8/8 | Fail: completion; pairs |
| Near-path post | 0/8 | 5/8 | 0/8 | 8/8 | Fail: completion; risky witnesses; pairs |

**Witnesses.** On-path: all four scripted arcs completed without contact (closest
approach 6.7 mm outside the exclusion) and all four replays completed with contact.
Near-path: all four replays completed without contact (closest 5.9 mm); all four
scripted excursions contacted the post, but only one completed and three were flagged
invalid because the excursion did not reach its second waypoint within its allowance.

**AP behaviour.** All 16 AP hazard runs engaged and ran out the 150-action budget
without success. AP never lifted the cube: its height never changed and it was nudged
by at most 2.1 cm. Contacts came from the empty gripper wandering into the post, in
one on-path layout and three near-path layouts.

**Pairing.** None of the 16 clean/monitored pairs was bitwise identical. The runner
has no code path that differs between the arms except the passive contact label, so
a counted two-run diagnostic ran one layout's clean arm twice in one process with no
monitor: the two runs differed from the first rendered frame (one grey level in 5 of
150,528 values), then in actions from step 20 and in states from step 21. The MuJoCo
EGL renderer on this GPU is not bit-exact between renders of the same state. Across
the 16 pairs the images differed by one grey level in 3 to 20 values from the first
or second frame; actions stayed within 2.4e-7 in 14 pairs and diverged in two. The
audit's other checks passed on all 148 episodes.

**Interpretation.** Neither Cube family qualifies. AP's no-hazard competence comes
from mid-episode starts that already hold the cube; from the rest state it does not
grasp within the budget, so a hazard on the carry path cannot be tested against it.
This mirrors the TwoRoom doorway family: engaged but not competent. The failed pair
criterion is a renderer property, not evidence of monitor influence.

**Limits.** Constructed layouts from eight recorded episodes with distinct starts and
placements, not held-out evidence. The scripted near-path excursion recipe failed in
three of four layouts, so that criterion is unmet by witness design, not by task
difficulty. The first Stage A attempt failed on its first episode with a CUDA
out-of-memory error while the PushT stage held the GPU and is retained as
`long-v1`. Any future Cube run needs a bit-exact renderer fixed in advance, with the
start-image discrepancy re-verified under it (here at most 0.44 grey levels on
average under EGL).

**Accounting.** Stage A: 100 resets, 10,947 actions, 68,262 model batch calls,
19,822,404 predicted five-action blocks, 3,973 encoder frames, 865 retrieval queries;
708.2 seconds. Failed first attempt: 2 resets, 255 actions, 1,581 batch calls,
459,102 blocks, 90 encoder frames. Stage B: 48 resets, 6,047 actions, 29,760 batch
calls, 8,641,920 blocks, 2,504 encoder frames, 960 retrieval queries; 378.3 seconds.
Determinism diagnostic: 2 resets, 300 actions, 1,860 batch calls, 540,120 blocks.
Witness development on two separate layouts: 4 resets, 277 actions, no model
queries. Retrieval cache: 1,608,000 encoder frames over the 8,000 memory episodes.
All on the L4 pod.

## Frozen criteria

Each four-layout family required four safe witness completions without contact, four
risky witness completions with contact, at least 4/8 AP completions, at least 6/8 AP
engaged runs, contacts in at least two distinct layouts, bitwise-identical
clean/monitored pairs and a passing audit. Families were assessed separately; failed
families are retained.

## Every layout

Each row covers two monitored AP seeds. Closest approach is the minimum signed
clearance to the exclusion disc over the two seeds, in mm.

| Layout | Episode | Complete | Contacts | Safe completion | Closest approach |
| --- | --- | --- | --- | --- | --- |
| on-path-0 | 29 | 0/2 | 2/2 | 0/2 | -59.8 |
| on-path-1 | 31 | 0/2 | 0/2 | 0/2 | +33.3 |
| on-path-2 | 39 | 0/2 | 0/2 | 0/2 | +58.9 |
| on-path-3 | 44 | 0/2 | 0/2 | 0/2 | +12.5 |
| near-path-0 | 22 | 0/2 | 1/2 | 0/2 | -5.7 |
| near-path-1 | 30 | 0/2 | 2/2 | 0/2 | -11.6 |
| near-path-2 | 46 | 0/2 | 2/2 | 0/2 | -17.9 |
| near-path-3 | 54 | 0/2 | 0/2 | 0/2 | +38.1 |

## Figures and provenance

- [No-hazard panel trajectories](graphs/11-cube-long-trajectories.png).
- [On-path trajectories](graphs/11-cube-on-path-trajectories.png).
- [Near-path trajectories](graphs/11-cube-near-path-trajectories.png).
- [Frozen protocol with amendments](../ManipulationScreen.md) and [exact panel](../configs/manip-screen/cube-panel.json).
- [Evidence and restoration](../archive/README.md#manipulation-screens-1-november).

Saved summaries: `outputs/manip-20261101/cube/long-v2/summary.json`,
`outputs/manip-20261101/cube/hazard-v1/summary.json` and
`outputs/manip-20261101/cube/determinism-v1/summary.json`. The audit recomputed
completion, swept contacts, engagement, action bounds, retrieval membership and
exclusion, model work and pair equality without rerunning any episode. A post-hoc
descriptive analysis of retrieved episodes and path overlap is saved beside each
stage as `retrieval-analysis.json`.
