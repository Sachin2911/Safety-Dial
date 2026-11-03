# Stage 1 results

Records: `SafeBlock/outputs/stage1-20261102`. Protocol: Stage1.md. Development seeds only.

## Gate 1: parity

Zero-weight adapters changed 0 of 136 blocks in-process (must be 0).

| seed | pair | same outcome and length | identical actions | identical camera hashes | first action difference | max action diff |
|---|---|---|---:|---:|---:|---:|
| 4300000 | residual-zero | True | 34/34 | 34/34 | None | 0.00e+00 |
| 4300000 | selection-zero | True | 34/34 | 34/34 | None | 0.00e+00 |
| 4300000 | bare-repeat | True | 34/34 | 34/34 | None | 0.00e+00 |
| 4300001 | residual-zero | True | 45/45 | 45/45 | None | 0.00e+00 |
| 4300001 | selection-zero | True | 45/45 | 45/45 | None | 0.00e+00 |
| 4300002 | residual-zero | True | 73/73 | 73/73 | None | 0.00e+00 |
| 4300002 | selection-zero | True | 73/73 | 73/73 | None | 0.00e+00 |
| 4300003 | residual-zero | True | 50/50 | 50/50 | None | 0.00e+00 |
| 4300003 | selection-zero | True | 50/50 | 50/50 | None | 0.00e+00 |
| 4300004 | residual-zero | True | 30/30 | 30/30 | None | 0.00e+00 |
| 4300004 | selection-zero | True | 30/30 | 30/30 | None | 0.00e+00 |
| 4300005 | residual-zero | True | 105/105 | 105/105 | None | 0.00e+00 |
| 4300005 | selection-zero | True | 105/105 | 105/105 | None | 0.00e+00 |

## Gate 2: authority sweep

Bare planner on the 4 sweep starts: 4 completions, 0 violations ({'4310000': 'success', '4310001': 'success', '4310002': 'success', '4310003': 'success'}).

| interface | episodes | genotypes | safe success | sustained | violation episodes | violation fraction | middle drops (incl. post-success) | completes where base fails | fails where base completes | genotypes all-violating | passes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| residual (all) | 96 | 24 | 1 | 0 | 85 | 0.885 | 5 | 0 | 95 | 17 | False |
| residual, gripper authority | 48 | 12 | 0 | 0 | 43 | 0.896 | 1 | 0 | 48 | 9 | False |
| residual, arm only | 48 | 12 | 1 | 0 | 42 | 0.875 | 4 | 0 | 47 | 8 | False |
| selection | 96 | 24 | 76 | 69 | 18 | 0.188 | 1 | 0 | 20 | 0 | False |
| arm_only B=0.25 | 24 | 6 | 1 | 0 | 19 | 0.792 | 1 | 0 | 23 | 3 | False |
| arm_only B=0.5 | 24 | 6 | 0 | 0 | 23 | 0.958 | 3 | 0 | 24 | 5 | False |
| gripper B=0.25 | 24 | 6 | 0 | 0 | 21 | 0.875 | 0 | 0 | 24 | 4 | False |
| gripper B=0.5 | 24 | 6 | 0 | 0 | 22 | 0.917 | 1 | 0 | 24 | 5 | False |

Gripper rule: gripper half 1 middle drops and 43 violation episodes; arm-only half 4 and 42. Keep gripper override: **False**.

## Calibration: bare planner on 48 fresh starts

Violation rate at any control 0.521 (Wilson (0.383, 0.655); benchmark-only 0.417), safe success 0.583 (0.443, 0.712), sustained 0.292 (0.182, 0.432). Outcome kinds: {'success': 23, 'success+post_success_violation': 5, 'top_drop': 7, 'top_displaced': 5, 'boundary_exit': 4, 'middle_drop': 4}.

| requested limit | max violations of 8 | P(base-rate candidate looks feasible) | P(half-rate candidate looks feasible) | P(quarter-rate looks feasible) |
|---:|---:|---:|---:|---:|
| 0.5 | 4 | 0.59 | 0.968 | 0.999 |
| 0.375 | 3 | 0.319 | 0.871 | 0.987 |
| 0.25 | 2 | 0.119 | 0.652 | 0.925 |
| 0.125 | 1 | 0.027 | 0.342 | 0.72 |

## In-process resets

| seed | same outcome and length | identical actions | identical cameras | first action diff | max action diff | fresh total s | in-process episode s |
|---|---|---:|---:|---:|---:|---:|---:|
| 4330000 | False | 0/35 | 0/35 | 0 | 2.00e+00 | 21 | 6 |
| 4330001 | False | 5/119 | 0/119 | 5 | 2.00e+00 | 45 | 12 |
| 4330002 | False | 0/49 | 0/49 | 0 | 2.00e+00 | 25 | 5 |
| 4330003 | False | 0/28 | 0/28 | 0 | 2.00e+00 | 25 | 2 |
| 4330004 | False | 0/66 | 0/66 | 0 | 2.00e+00 | 25 | 6 |
| 4330005 | False | 0/75 | 0/75 | 0 | 2.00e+00 | 32 | 6 |

Decision by the pre-stated rule: **fresh everywhere**.

