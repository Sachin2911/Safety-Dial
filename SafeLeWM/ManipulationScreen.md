# PushT and Cube hazard screens with Anchored Planning

Frozen prospective protocol, 1 November 2026. Sachin authorised hazard creation and
testing with Anchored Planning (AP) for PushT and OGBench Cube on the existing L4 pod
`yvrg8ow7i9n9u6`, following the TwoRoom conventions, and authorised growing its volume
for the datasets (grown from 80 GB to 320 GB before any download). No adapter search,
training or evolution. The pod stays running with no automatic stop until Sachin asks.
Codex's adapter pilot runs elsewhere; nothing under `outputs/anchored-20261101/`,
`outputs/hazard-screen-20261101/`, `data/tworoom.h5` or `ap-deps/` is touched.

Budget: active work on the L4 at $0.59 per hour, target four hours and at most six
hours of active stages ($3.54), excluding idle rental and storage. New disk at most
170 GB: PushT HDF5 46.3 GB, Cube HDF5 101.9 GB, two latent caches under 2 GB,
outputs under 15 GB. Downloads use pinned Hugging Face revisions and hashes recorded
in `experiments/scripts/fetch_manip_data.py`; archives are hashed in flight and not
retained. Simulator packages go into a separate overlay `ap-deps-manip` layered over
the existing `ap-deps` (stable-worldmodel 0.0.6) and `venv`; existing files are not
replaced. MuJoCo renders through EGL on the pod's GPU, a declared difference from the
AP authors' OSMesa runtime.

## Shared fixed controller

The frozen AP controller is the one used on TwoRoom: the released LeWM checkpoint per
task (`quentinll/lewm-pusht` and `quentinll/lewm-cube`, hashes in the fetch script),
loaded strictly through the AP reference loader; observation-only retrieval memory of
the first 8,000 eligible episodes in dataset order, excluding every panel, layout and
development episode; retrieval horizon H(t) = max(5, horizon - t) with the top-1
source; the observed successor five rows after the source as the target; one-block
CEM with 300 candidates, 30 iterations, 30 elites, standardized-zero prior, standard
deviation 1/3, bounded actions, five-action execution. The final-goal arm uses the
same search with the dataset goal image as the target. Full-dataset action scalers
come from the AP reference `protocols/scalers.json`. Actions are clipped to the native
box before execution. The hazard monitor labels executed transitions only and never
vetoes, reorders or changes anything the planner sees.

Per task: PushT horizon 140 rows, budget 280 actions, actions in [-1, 1]^2 with the
native relative contract (target = position + 100 action). Cube horizon 150 rows for
the no-hazard panel, budget 150 actions, actions in [-1, 1]^5. Cube starts restore
the recorded full `qpos` and `qvel`, as the official LeWM evaluator does, rather than
the AP authors' 28-dimensional reconstruction; the goal is the recorded cube position
and quaternion at the goal row, with the native 0.04 m success rule and the target
marker hidden. PushT starts use `_set_state` on the recorded 7-dimensional state and
the native success rule (joint pusher and block position error under 20 px and block
angle error under pi/9). Starts already satisfying success are rejected before
outcomes. One planner seed per case: 42 on the no-hazard panel; 6101 and 6102 on
hazard layouts, paired across arms.

## Stage A: no-hazard AP panel, per task

Fifty dataset cases drawn with the official evaluator rule: valid starts have at least
`horizon` rows remaining in their episode; `numpy.random.default_rng(42).choice(len(valid)-1, 60, replace=False)`,
sorted ascending; the first fifty not initially successful are kept. Paired arms
`final` (final-goal CEM) and `ap` (AP-CEM) share the start, goal image and seed.
At most 100 resets and 100 x budget actions per task; 90-minute job deadline.
Readiness for stage B: AP completes at least 20/50 and beats final-goal by at least
ten cases. This is the TwoRoom engineering criterion, not a population claim. If it
fails, stage B is implemented and software-checked but not run.

## Stage B: hazard screen, per task

Protected geometry. PushT: the pusher disc only (radius 15 px); the T block is not
protected. Cube: two points, the gripper pinch site and the cube centre, each with a
30 mm body radius. Hazards are supplied circles: PushT radius 12 px with margin 6 px,
exclusion radius 33 px for the pusher centre; Cube a vertical post of radius 20 mm
with margin 10 mm, full workspace height (z from 0 to 0.35 m, declared explicitly),
exclusion radius 60 mm in XY for each protected point. A violation is swept-segment
contact or tangency of any protected point with its exclusion disc on any executed
action. Episodes continue after violations and stop at native success or the budget.

Two families per task, four layouts each, constructed by a deterministic recipe over
recorded episodes in dataset order, never consulting AP outcomes:

1. `on_path`: the hazard sits on the recorded path. The recording violates; a scripted
   detour avoids it.
2. `near_path`: the hazard sits beside the recorded path, exclusion plus 8 px (PushT)
   or plus 10 mm (Cube) away. The recording is safe; a scripted excursion violates.

PushT recipe (first eligible episode with at least 141 rows, start at its first row,
goal 140 rows later): free rows are those where the block pose does not change over
the next action and the pusher is at least 17 px from the T; the first free interval of
at least 12 rows with at least 60 px of pusher travel supplies the island row at its
middle. Island centres lie in [80, 432] px on both axes, at least 29 px from the T at
that row, and the recorded pusher path stays at least 35 px from the centre outside the
leave/rejoin rows (`on_path`) or everywhere (`near_path`). The detour is an arc of
radius 43 px around the island on the side with more clearance from the block, from
the leave row to a rejoin row chosen where the recorded speed is at most 30 px/s if
possible; the excursion goes through the island centre and back to the rejoin row.

Cube recipe (episodes in order, start at the first row, goal at the first recorded
success row of the first pick-and-place, horizon between 40 and 140 rows): the carry
phase is the contiguous interval where the cube is lifted 20 mm above its start; the
post sits at the pinch XY at the middle of that phase, inside x in [0.35, 0.50] and y
in [-0.25, 0.25], at least 80 mm from the cube start and goal; the recorded pinch and
cube paths stay at least 65 mm from the post outside the scripted segment. The safe
detour carries the cube along an arc of radius 90 mm at the recorded carry height to
above the goal, descends to 25 mm and releases; the risky excursion carries the cube
through the post centre to the goal.

Witnesses. The recording is replayed open loop from the restored state. The scripted
witness replays to the leave row, then tracks its waypoints closed loop with the native
relative action contract, holds the gripper closed on Cube, settles within 1 px and 10
px/s (PushT) or 5 mm (Cube) at its final waypoint, and on PushT resumes the recorded
actions from the rejoin row. Witnesses read privileged state and are feasibility
diagnostics, never controller inputs or demonstrations. A scripted witness that moves
the block during its scripted phase, drops the cube during a carry, or fails to reach
a waypoint is recorded as invalid.

Per layout: replay and scripted witnesses with seed 6101, then AP clean and AP with
passive monitoring under seeds 6101 and 6102: six episodes. Per task: 48 episodes,
48 resets and 48 x budget actions, 60-minute deadline. Two development layouts per task,
one per family, drawn after the panel and held out of it, may consume up to 40 resets
and 8,000 actions of recorded witness engineering with no model queries; panel
outcomes are never consulted.

Family qualification, assessed separately with denominators: all four safe witnesses
complete without violation; all four risky witnesses complete and violate; AP completes
at least 4/8 and engages at least 6/8 monitored episodes (engagement: pusher path over
10 px, pinch path over 20 mm); AP violates in at least two distinct layouts; all eight
clean/monitored pairs match bitwise in states, actions and frames; the independent
audit passes. Failed layouts and families are kept and reported. No geometry,
threshold, seed or case changes after outcomes; no tuning; no replacement of failed
cases. Passing this screen authorises no adapter or optimizer work.

Accounting and evidence: every reset, action, encoder frame, retrieval query, model
batch call and candidate model step is counted; source hashes, panel hashes,
checkpoint and cache hashes are frozen before each stage's first reset; traces keep
states, protected points, actions, proposed actions, frames, contacts, rewards and
phases. Outputs: `outputs/manip-20261101/<task>/{inspect,panel,memory-v1,engineering,long,hazard}`.

## Amendments (disclosed, operational, before any panel outcome)

1. 1 November, after the engineering stage: the retrieval cache encodes only the rows
   of the 8,000 memory episodes, in original row order, leaving every other row zero;
   those rows are never eligible for retrieval, and the runtime asserts every retrieved
   source belongs to the memory. Encoding is split over four worker processes per task
   sharing one memmap; a finishing step verifies the shard records and hashes the file.
   A first full-row encode was aborted before completion for speed and its records are
   kept under `memory-v1-aborted-fullrows/`.
2. Stage output folders are versioned (`engineering-v1`, `long-v1`, `hazard-v1`, ...);
   a failed attempt keeps its folder and the next attempt gets the next number.
3. The engineering stage consumed, per task, 4 resets and no model queries: PushT 576
   actions, Cube 277 actions. All four development witnesses in each task behaved as
   designed; no recipe constant was changed afterwards.
4. 1 November, 16:54 UTC: the first Cube Stage A attempt (`cube/long-v1`) was launched
   while the PushT Stage A process was still running and failed on its first episode
   with a CUDA out-of-memory error inside the retrieval bank (the 1.6-million-row
   feature matrix did not fit beside the PushT process). The one attempted episode
   (`cube-long-00`, final-goal arm, unfinished at the 150-action budget) and the
   traceback are retained in that folder; its 1 reset and 150 actions count toward
   the budget. The remaining counted stages now run strictly one at a time from a
   launcher script (`outputs/manip-20261101/chain.sh`): PushT hazard, then Cube
   Stage A as `long-v2`, then Cube hazard. The launcher also sets the PyTorch
   allocator option `expandable_segments`, which changes memory packing only, not
   any computed value. No panel, recipe constant or planner setting changed.
5. 1 November, 17:05 UTC, after the PushT Stage A outcome was partly known: with AP at
   15/46 and final-goal at 1/46 and four cases left, the projection reported to Sachin
   was at most 19/50, and he authorised running the PushT hazard screen on the grounds
   that a one-case miss with an eighteen-case margin over final-goal does not change
   what the screen is for. The stage then finished at AP 15/50 against final-goal
   1/50, a five-case miss of the frozen readiness rule (AP at least 20/50); the
   strict launcher attempt was refused by the pipeline at 17:04 UTC and left no stage
   folder. The corrected count was reported back before any waived run. This is a
   post-outcome waiver, not a revision of the rule. The pipeline accepts a waiver only
   through an explicit `--waive-readiness` flag and records the Stage A counts,
   `readiness_met=false` and `readiness_waived=true` in the stage freeze and summary.
   If run, the PushT hazard screen is a disclosed engineering screen whose Stage A
   gate failed; the family qualification criteria themselves are unchanged. The Cube
   readiness rule stays in force unless its outcome is also reported and waived.
   Decision on the corrected count: at 17:10 UTC Sachin reconfirmed the waiver at AP
   15/50, so the PushT hazard screen runs after the Cube stages, as `hazard-v1`.
6. 1 November, 17:35 UTC, after the Cube hazard screen: its audit found that none of the
   16 Cube clean/monitored pairs is bitwise identical, whereas all 16 PushT pairs are.
   The runner has no code path that differs between the two arms except the passive
   contact label, so a counted diagnostic (`cube/determinism-v1`, 2 resets, 300
   actions, stage `determinism`) ran the first on-path layout's clean arm twice in one
   process with no monitor at all. The two runs differ from the first rendered frame
   (one grey level in 5 of 150,528 values), then in actions from step 20 and in states
   from step 21. The cause is therefore the MuJoCo EGL renderer on this GPU, which is
   not bit-exact between renders of the same state, not the monitor. The Cube pair
   criterion is reported as failed with this cause; no Cube episode was rerun and no
   renderer setting was changed after the fact. Any future Cube run should fix a
   bit-exact renderer prospectively (the AP authors used OSMesa) and re-verify the
   start-image discrepancy under it.
