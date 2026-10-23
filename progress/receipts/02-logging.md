# Todo 2, item 6: what the evaluator can record

ManiGuard cloned at `6bf4c61a1f168cc46e02eb22c41bcb51815e9726` under
`third_party/ManiGuard` (gitignored), code read before any install. Files read:
`maniguard/eval/benchmark.py` (the per-task evaluator, 1,156 lines),
`maniguard/eval/recording.py`, `maniguard/eval/eval_config.py`,
`maniguard/utils/safety_monitor.py`, `maniguard/eval/goal_checker.py`,
`maniguard/utils/goal_region.py`, `maniguard/data/recording/observer.py`, and the Jar
task's `diagnostics.jsonl` from the benchmark at `af09438c`.

## What is written per episode today

| artefact | written by | per step? | content |
|---|---|---|---|
| `results.jsonl` row (`row.json` in the dataset) | `benchmark.py` after the rollout | no | scene, seed, `episode_seed`, `steps`, `status`, `success`, `success_step`, `success_first_step`, `goal_detail` (held flag at the last check, goal centre and radius), `total_reward`, `ltl_violated`, `ltl_violation_step`, `ltl_violation_count`, `ltl_formula`, `ever_contacted`, `first_contact_step`, `ever_grasped`, `grasp_steps`, `target2spawn_max_dist`, `eef2target_min_dist`, `outcome`, `safety_evaluated`, `counted_violation` |
| `rollout_diagnostics` (inside the row) | `benchmark.py` | no, counters | `phase`, `action_attempts`, `env_steps_returned`, `observations_returned`, `last_finite_observation_step`, `last_action_finite` at the three boundaries (`raw`, `controller`, `clipped`), `first_nonfinite_action`, `termination_reason`, `exception` |
| `<task>_ltl.json` sidecar | `TaskLTLMonitor.summary()` | yes | per step: the four propositions, automaton `state`, `accepting`, `doomed`; plus formula, constraints, violation step and count |
| `<task>_main.mp4`, `<task>_wrist.mp4` | `StreamingVideoWriter` at `action_frequency` fps | yes | the 256 x 256 overview and wrist frames the policy was sent, frame i = step i |
| `summary.json`, `eval_config.json` | `benchmark.py` | no | run totals and the resolved config |
| `metrics` | config field | no | the list `["success", "safety"]`; it names which checkers run, it is not a per-step record |

The propositions are evaluated by OmniGibson object states on the single `hinged_jar_*`
object: `jar_on_support` = `OnTop` of `desk.n.01_*`; `jar_closed` = not `Open`, read
from the lid hinge joint position (open once past 5 percent of its range, the
`_is_open_via_joints` fallback because the jar has no `openable` ability); `jar_dropped`
= `Dropped` with `floor_z` 0 and `z_margin` 0.05; `jar_upright` = `Upright` with
`max_tilt_deg` 30. The goal check (`GoalRegionChecker`) requires `robot.is_grasping(jar)`
to be `TRUE` (the assisted-grasp state) and the jar inside the sphere, for ten
consecutive steps. First contact is any robot link in `ContactBodies` of any task
object.

Two debug hooks exist but are off and write JSON text, not arrays: `EVAL_DEBUG_IO`
(state and first action of each chunk) and `EVAL_DEBUG_STEP` (raw and clipped action,
eef position before and after, per executed step). `EVAL_DEBUG_IMG` dumps the step 0
images.

## What is not written

- The 16-step action chunk the server returns (only `chunk[0]` under `EVAL_DEBUG_IO`).
- The 8 executed actions per chunk at the three boundaries (raw policy output,
  controller command, clipped command); `EVAL_DEBUG_STEP` has two of them as text.
- Robot joint positions and velocities (the row has nothing; the policy state has the
  7 arm joints and the mean gripper opening, and it goes to the server, not to disk).
- Gripper finger positions (two joints), end-effector pose, grasp state per step
  (`grasp_steps` is a count).
- Jar and lid poses, the lid hinge angle, jar velocities; the propositions are the only
  trace of them, thresholded.
- Contact flags per step (first contact is the only one kept), finger contact, contact
  forces.
- Positions of the other task objects (tasks 23 and 24 have distractors).
- Timing per step.

## The hook

`benchmark.py` already carries a passive observer interface. `make_observer(cfg, scene)`
(`maniguard/eval/recording.py`) imports `cfg.recording_factory` as `module:function`
and the per-task loop calls the returned object at seven points: `start(env, robot,
obs, monitor, episode_seed, scene)` after warm-up and monitor reset; `proposal(step,
chunk, chunk_len, action_space)` after every policy query; `before_action(step, ci,
raw, ctrl, clipped)` and `applied(step)` around every `env.step`; `transition(step,
obs)` after the new observation; `oracle(step, monitor)` after `monitor.step`; and
`finish(result)` with the row. Both `--recording-factory` and `--recording-output-dir`
are CLI flags. Upstream's own observer (`maniguard.data.recording.observer`) needs the
high-fidelity source capture (`env._maniguard_source_capture`) and snapshots the whole
scene every step, so it cannot run under the legacy 256 profile and would be far too
heavy; under the legacy profile `resolve_source_config` returns early and sets no
default factory, so the factory must be named explicitly.

This is the cleanest hook: no upstream file changes, scoring untouched (the observer
only reads state after each executed action and never steps or renders), and the
callbacks already distinguish the proposed chunk from the executed actions at the three
boundaries the evaluator itself distinguishes.

`scripts/todo2/record_rollout.py` implements it as `RolloutRecorder` with
`create_observer(cfg, scene)`. Per episode it writes one
`<output dir>/<scene name>.npz` (compressed) holding, for every step 0..N:

| field | shape | source |
|---|---|---|
| `step`, `wall_s`, `recorder_s` | (N+1,) | step index, monotonic clock, seconds inside the recorder |
| `q`, `dq` | (N+1, 9) | `robot.get_joint_positions/velocities()` (7 arm + 2 finger) |
| `eef_pos`, `eef_quat` | (N+1, 3), (N+1, 4) | `robot.get_eef_position/orientation()` world |
| `gripper_q` | (N+1, 2) | finger joints |
| `policy_state` | (N+1, 8) | the state vector sent to the server |
| `jar_pos`, `jar_quat` | (N+1, 3), (N+1, 4) | target jar base pose |
| `jar_link_pose` | (N+1, links, 7) | every jar link (body and lid) |
| `jar_joint_q`, `jar_joint_dq` | (N+1, joints) | the lid hinge |
| `jar_lin_vel`, `jar_ang_vel` | (N+1, 3) | jar velocities |
| `jar_robot_contacts`, `jar_finger_contact` | (N+1,) | robot links touching the jar; any finger link touching it |
| `grasping` | (N+1,) | `robot.is_grasping(jar)` |
| `other_obj_pos` | (N+1, objects, 3) | the other movable task objects |
| `ap`, `ap_valid`, `monitor_doomed` | (N+1, 4), (N+1,), (N+1,) | the monitor's record for that step |
| `proposal_step`, `proposal_planned`, `proposal_chunk` | (P,), (P,), (P, 16, 8) | every chunk the server returned |
| `exec_step`, `exec_proposal`, `exec_offset`, `exec_raw`, `exec_ctrl`, `exec_clipped` | (N,), (N,), (N,), (N, 8) x 3 | the executed actions at the three boundaries |
| `meta` | JSON string | joint, link and object names, hinge limits, config, scene, seed, the result row, recorder overhead |

Invocation (behavior env, from the ManiGuard checkout):

```
PYTHONPATH=/workspace/Safety-Dial/scripts/todo2 python -m maniguard.eval.benchmark \
    --config configs/eval/jar_transport_joint_legacy256.yaml ... --tag smoke \
    --recording-factory record_rollout:create_observer \
    --recording-output-dir /workspace/recordings/<run>
```

No upstream file is changed, so there is no `02-logging.diff`. The eval config copied
into `configs/eval/` is a new file (byte-identical to `01-jar-eval.yaml`), as in todo 1.

## Item 7: validation of the recorder

Not validated on a real episode. The host check passed (OptiX, EGL, the NVIDIA Vulkan
ICD all present), the pinned install was replayed (98 minutes, two recorded deviations
repaired) and the policy server was warmed, but the evaluator segfaulted four times inside
Isaac Sim 4.5's RTX renderer at startup under this pod's driver 595.91.07, before any
scene loaded; the account, backtrace and attempts are in `02-host.md`. No episode was
started, no `.npz` exists, the scoring comparison and the size and overhead measurements
are deferred to the next Vast rental with a driver 580.x host.

What was checked instead, so the next host only has to run the episode:

- `record_rollout.py` was driven through the evaluator's exact callback sequence
  (`start`, then per chunk `proposal` and per executed action `before_action`, `applied`,
  `transition`, `oracle`, then `finish`) with mock robot, jar, distractor and monitor
  objects for 19 executed steps in 3 chunks. `scripts/todo2/check_recording.py` on the
  result: every state field has 20 rows (steps + 1), 19 executed actions at the three
  boundaries, 3 proposals of shape (16, 8), `step` is 0..19, the monitor record is valid at
  every step, 0.1 ms per step, 0.01 MB. Exit 0.
- `check_recording.py` also compares the result row's keys and scoring fields
  (`success`, `success_step`, `ltl_violated`, `ltl_violation_step`, `first_contact_step`,
  `ever_grasped`, `grasp_steps`, `outcome`, `counted_violation`, ...) with the instance C
  smoke row in `01b-smoke.jsonl`, and the recorded propositions with the sidecar, so the
  "scoring unchanged" check is scripted:
  `python3 -I scripts/todo2/check_recording.py <npz> <results.jsonl> --sidecar <task>_ltl.json`.
- The invocation on a working host is `scripts/todo2/run_smoke_pod.sh` with
  `PY` pointed at that host's behavior env; it is the todo 1 smoke command plus
  `--recording-factory record_rollout:create_observer --recording-output-dir <dir>` and
  `PYTHONPATH=scripts/todo2`.

Expected size: per step about 9 + 9 + 3 + 4 + 2 + 8 + 3 + 4 + 14 + 2 + 2 + 3 + 3 floats
plus the executed actions (24 floats) and one chunk per 8 steps (16 floats), roughly 100
float32 per step, under 1 MB per 2,000-step episode after compression. Expected overhead:
a dozen simulator reads per step, well under the 50 ms of a 20 Hz step; to be measured.
