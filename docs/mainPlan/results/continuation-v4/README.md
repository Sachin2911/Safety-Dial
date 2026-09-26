# Push-T stress-bank recovery with fixed completed roots

Status: stress-only recovery completed and uploaded after 372 passing tests. The development diagnosis passed; the managed workflow is now running repairability and goal retention. The completed v3 development
and representative test banks are retained byte for byte. Main-study outcomes are pending.

The [v3 test audit](../continuation-v3/test_bank_completion_audit.json) verifies 128 roots
from 128 reserved source episodes, 64 per family, and 2,048 distinct stored tapes. The
[stress failure record](../continuation-v3/stress_generation_failure.json) preserves the
17-root, 272-branch partial stress bank and its 13,040 charged simulator steps. Generation
stopped before querying branches for root `test-familiar-c00318`: all 20 draws for its
first toward-hazard slot failed the existing arena guard. No evaluated-method outcome
caused the stop or informed this recovery.

The fresh stress protocol keeps every completed test root and layout, eight contact-target
slots plus eight toward-hazard slots, the original parameter distributions, float32 action
rounding, duplicate rejection and arena guard. It increases the fixed proposal-search cap
from 20 to 4,096 draws per slot for every root. All 2,048 controls and their proposal
provenance must be frozen before any new stress branch is queried. The earlier stress
run is preserved separately and its cost remains part of the study accounting.

The [pure proposal preflight](stress_preflight_audit.json) filled all 128 fixed-root
pools with 2,048 tapes from 2,668 draws. The hardest slot required 61 draws. The original
17 completed roots' full proposal audits and the failed root's first 35 trial records
match exactly. This check used saved initial root geometry only, with no model scores,
future outcome inspection or simulator queries.

The new runner copies the completed development and test bank bytes into a fresh data
folder. A separately identified copy of the passing development witness relocates only
its sampling-plan path, records the original report identity, and reruns the existing
feasibility checks against identical bank, plan and generator bytes. Its changed report
receives a new private upload receipt. Source snapshots remain local, outside uploaded
bank bundles.

The [managed continuation](../../../../configs/pusht/continuation-v4.yaml) retains the
full development diagnosis, repairability and 20-case retention gates, three paired
acquisition seeds and four budgets, checkpoint-specific retention, geometric transfer,
and conditional closed-loop stages. The [oracle continuation](../../../../configs/pusht/continuation-oracle-v4.yaml)
retains the same retrospective full-pool accounting and waits for qualified prerequisites.
Scientific gate failures remain valid negative outcomes.

The [actual-input integration check](recovery_integration_audit.json) verifies exact bank
copies, the relocated witness gate and frozen-control round trips in a temporary folder.
It made no simulator queries or remote uploads; production execution remains separate.

[Prelaunch validation](recovery_validation.json) passed all 372 repository tests and
Ruff, both workflow dry runs, the pure proposal tests and independent implementation
and provenance review. Scientific outcomes remain pending.

The [managed launch record](continuation_status.json) records the active workflow and
waiting oracle. The original failed stress data was privately archived before new
replay. Both completed final banks and the relocated witness are now privately uploaded.
The recovered stress bank contains 128 roots and 2,048 branches, with 101,373 new
simulator steps. Accounted v3/v4 bank construction and development-witness cost is
574,008 steps, including the failed stress run. Scientific outcomes remain pending.

The [completed-bank audit](recovery_completion_audit.json) independently verifies every
frozen control, exact root/layout identity, charged step and all four new private
uploads. The development diagnosis attributes 21 of 28 known, uncensored attributable
false-safe errors to imagined dynamics (75%), passing the predeclared development
gate. One accepted development future remains unresolved; its false-safe rate is
undefined, with reported bounds rather than an assumed safe label. Repairability
and goal retention are running; successful repair is not yet established.
