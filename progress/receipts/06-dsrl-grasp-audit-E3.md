# Task 6 native grasp enum validity repair

24 October 2026 UTC, 25 October local time. This is a technical recording and derived
validation correction. It does not change the reward, native goal predicate, controls,
noise bound, seeds, task panel, budgets, model or simulator state. Raw recordings,
journals and query artifacts remain byte-for-byte unchanged.

During engineering on instance 57575920, E3's first native policy query was global
query 4, returned at `2026-10-24T23:37:57.013547+00:00`. The scorer reported invalid
states despite valid native AP observations. Its validity rule assumed a nonnegative
raw grasp enum was known. The pinned OmniGibson enum instead defines FALSE as -1,
UNKNOWN as 0 and TRUE as 1. Thus ordinary not-grasping states were rejected, while
UNKNOWN was incorrectly permitted. The initial inference that -1 necessarily meant
a caught recorder exception was wrong; it is the native FALSE value too.

Read-only inspection of installed pinned sources established:

- BEHAVIOR/OmniGibson revision `88454bd04f75dc57c00ab1f1a00bcde1ff505950`,
  `OmniGibson/omnigibson/controllers/controller_base.py`, lines 40 to 43:
  `IsGraspingState(IntEnum)` has TRUE=1, UNKNOWN=0, FALSE=-1. SHA256
  `855a338190d3794f4f078b25ba6588e39708729221c85af0f8b0b12d5538c12b`.
- The same revision's `OmniGibson/omnigibson/robots/manipulation_robot.py`, lines
  456 to 492, returns that enum. Its assisted mode returns TRUE or FALSE from the
  object-in-hand and release state. SHA256
  `7a565f80c06a11f2eaa9d7514edbebf5772ee7b3aa73ad765cae4406c169c4ed`.
- ManiGuard revision `6bf4c61a1f168cc46e02eb22c41bcb51815e9726`,
  `maniguard/utils/goal_region.py`, lines 705 to 711, defines `robot_holds_target`
  by the same `robot.is_grasping(candidate_obj=target_obj)` call and compares its
  result exactly with `IsGraspingState.TRUE`. SHA256
  `e7e4bef5fee97000ed9df0b5d360e48ffb6eb32e1460d7707c451ae465748f82`.
- The same ManiGuard revision's `maniguard/eval/goal_checker.py`, lines 71 to 104,
  supplies that predicate as boolean `detail['held']`. Jar has no Lid-family
  assembly alternative. SHA256
  `ace47ce46cc0a540c759b0f5d0c89f050bdebbe83877e6052e770bf09acc74b0`.

The native per-control goal result is already journalled after the existing goal
check, and its held value is saved in the `goal_held` NPZ column. The explicit pure
initial goal read supplies the same field at state zero without advancing the
native ten-control goal counter. No additional simulator or goal check is needed.
At the time of this diagnosis, all 1,087 synced E3 scorer states, zero through 1,086,
had known raw enums and exact agreement between `(raw == 1)` and native held. This
partial observation is not a substitute for the completed episode's full proof.

After E3 completed its full 2,000 controls, a separate read-only check verified all
2,001 states in the final raw NPZ and journal. AP validity and goal-checked flags
were true throughout; raw grasp was -1 throughout; canonical native held was false
throughout. Every strict boolean journal held value equalled the NPZ held column and
`(raw == 1)`. Every old false validity flag exactly reproduced the old rule. Input
hashes were checked before and after that read and did not change:

- `task_0011_base.npz`: SHA256
  `48b652a522abe79b1a6bbc86baef213017d2aed4ede351caafc71b66e695e186`.
- `task_0011_base.journal.jsonl`: SHA256
  `49f4a6e45354320535716bd75eef1e22302d92b0654c75d6f4c52831808717db`.

The forward recorder now preserves the raw `grasping` enum and accepts it only when
it is -1 or 1, the AP and goal checks are available, and `(raw == 1)` exactly equals
the native held boolean. UNKNOWN=0 and unexpected enum values remain invalid. Missing
native held fails closed. Additional diagnostic columns and journal fields record
enum knownness and native predicate agreement. Metadata identifies the repaired rule
as `dsrl_grasp_validity: native-enum-pm1-held-parity-v1`.

Retrospective correction belongs only in newly derived validation. It requires all
N+1 states, raw enums exclusively in {-1,1}, valid APs, checked native goals, strict
boolean held values, agreement with the untouched NPZ `goal_held` column, and exact
agreement with `(raw == 1)`. The previous per-state and final false validity flags may
be corrected only when they exactly reproduce the old `raw >= 0` rule. The derived
receipt must identify corrected steps and preserve raw source hashes. It may not
substitute UNKNOWN, missing predicates, disagreement, incomplete data or a real
failure. The protocol/validation implementation owns these retrospective checks.

Forward-recorder validation: 21 server/recorder CPU tests passed in 1.14 seconds,
including FALSE, TRUE, UNKNOWN, unexpected values, disagreement and missing native
held. Ruff check and formatting passed. The repair itself made no simulator/model
query, learner update, raw-artifact rewrite or remote mutation. Source deployment,
the amended freeze and coordinator reconciliation remain the root agent's actions.
