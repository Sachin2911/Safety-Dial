# PushT authority screen: replay action-bound correction

2 November 2026, after the first attempt stopped. This is a documented engineering
correction within the original two-reset/560-action reserve, not a gate waiver or
additional parameter search. The fixed protocol already clips commands to the native
[-1,1] box. A recorded replay command [1.0535855, -0.15184173] in scene 5 was
rejected by a pre-clipping assertion in the inherited episode runner. Original
attempt run-v1, source freeze, trace and logs remain unchanged.

The separate v2 episode runner requires finite commands, records the raw proposed
command, then clips to the native box before execution. It records the clipped
base command for witness provenance. Learned-policy action interpretation and all
other code paths are unchanged. This may change replay outcomes; no witness failure
will be waived, filtered or repaired using outcomes.

Continue in run-v2. Copy already-completed episode records byte-for-byte with an
origin/hash inventory, plus the originally locked parameter vectors, and skip those
job IDs. Retry only the failed scene-5 witness, then execute the previously scheduled
remaining jobs. Do not repeat completed cross-GPU, witness or planner episodes.
Keep all original cases, source groups, memory, seeds, parameters, gates, assignment,
280-action horizons and the original preparation-based four-hour deadline.

Account unique completed episodes once, and add all failed-attempt execution
footers separately. Assert combined work remains within 330 resets and 92,400
attempted actions. Tests cover finite-command clipping, raw-command preservation,
exact in-bounds recovery and rejection of nonfinite inputs. No optimizer training,
new case selection, additional genotype or rental shutdown is authorized.
