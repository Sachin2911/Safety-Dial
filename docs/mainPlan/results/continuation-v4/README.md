# Push-T stress-bank recovery with fixed completed roots

Status: the development diagnosis passed, but the completed E2 repair and retention
gate did not pass on 27 September. Both the managed Push-T workflow and the waiting
oracle stopped at that gate. Acquisition, transfer and conditional closed-loop work
were not launched. Walker LeWM training continues in the separate
[Walker continuation](../continuation-v2/README.md). The completed v3 development
and representative test banks remain byte-identical.

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
and provenance review. This validation establishes implementation readiness, not scientific success.

The [managed launch record](continuation_status.json) records the active workflow and
oracle state. The original failed stress data was privately archived before new
replay. Both completed final banks and the relocated witness are now privately uploaded.
The recovered stress bank contains 128 roots and 2,048 branches, with 101,373 new
simulator steps. Accounted v3/v4 bank construction and development-witness cost is
574,008 steps, including the failed stress run. This scope excludes E1 evaluation and E2 repair costs.

The [completed-bank audit](recovery_completion_audit.json) independently verifies every
frozen control, exact root/layout identity, charged step and all four new private
uploads. The development diagnosis attributes 21 of 28 known, uncensored attributable
false-safe errors to imagined dynamics (75%), passing the predeclared development
gate. One accepted development future remains unresolved; its false-safe rate is
undefined, with reported bounds rather than an assumed safe label. The later E2 gate
did not establish qualified repair and goal retention.

The [independent diagnosis audit](development_diagnosis_audit.json) reproduces the gate
from the saved development rows and verifies input identities, 6,880 charged root-prefix
steps and all 41,920 horizon rows. The [corrected figures](decomposition-figures/README.md)
show explicitly labeled lower bounds where false-safe acceptance is undefined and use
consistent pose-series styles. These are presentation corrections from the immutable
report; all original outputs and the gate decision remain unchanged. The numeric
plotting policy passed 12 focused tests and Ruff.

The [data-only results archive](decomposition_archive.json), including both numerical
NPZ files and the original and corrected figures, is now privately stored. After the
user explicitly requested GitHub and Hugging Face storage on 27 September, the
[publication audit](decomposition_archive_publication.json) verified all 18 files
(11,400,309 bytes) and the immutable tag at revision
`e792c41aa22e95c2f6a229dc3b44cd64128c8000`. The earlier rejected attempt remains
recorded; the later authorized upload did not change the original scientific outputs.

The [E2 adaptation-bank audit](e2_adaptation_bank_audit.json) verifies 64 source-disjoint
roots and 446 distinct guarded branches, of which 422 have complete usable training
horizons. All 446 traces remain stored and charged. The recorded collection cost is
38,827 simulator steps; the rejected-root aggregate is metered but cannot be independently
reconstructed per attempt because no candidate journal was saved. The existing proposal
allocation and arena rejection produce an approximate bank size, not exactly 512 branches.
The [input durability audit](e2_input_durability_audit.json) verifies all 11 bank and clip
payload files against pinned private remote metadata. Completed repair and goal-retention
evidence is summarized below.

The [censoring and recipe-selection note](e2_gate_censoring_note.json) separates
development eligibility, the deterministic fallback and the final held-out repair gate.
The [adopted protocol](../../protocol.md#predictor-side-adaptation) requires development-only
choices fixed before acquisition; it does not state an additional hard eligible-recipe
gate. A default fixed without held-out feedback can therefore be reported as a fallback
after inconclusive development ranking, provided the unchanged final repair and retention
gates pass. It must not be described as a development-validated optimum. The [development-selection audit](e2_development_selection_audit.json) confirms all 16
ordered grid recipes were evaluated and all had undefined development false-safe rates,
leaving zero eligible recipes. The existing first-record fallback selected predictor-side
updates, teacher-forced loss, learning rate `2e-5` and 1,000 steps. The selected recipe
and thresholds remained unchanged. The final gate did not pass. An undefined point rate
does not by itself prove that adaptation cannot improve the model.

The [completed E2 evidence audit](e2_completion_audit.json) independently reproduces
all saved final rows, paired truth and censor masks, margins, retention trajectories
and gate arithmetic. At matched development-selected margins, the test evidence is:

| Arm | Accepted rows | Known false-safe rows | Accepted unresolved futures | False-safe identification bounds |
|---|---:|---:|---:|---:|
| No update | 3,139 | 756 | 49 | 24.08% to 25.65% |
| Adapted | 2,950 | 620 | 50 | 21.02% to 22.71% |
| Readout correction | 2,905 | 795 | 34 | 27.37% to 28.54% |

All three point rates remain undefined. These are bounds over unresolved labels in
the saved sample, not sampling confidence intervals. A supplemental conservative
bound places the relative reduction between 5.70% and 18.05%; even its upper bound
is below the 25% target for these fixed accepted sets under any consistent completion
of unresolved labels. This is not a population-effect estimate or evidence of no gain. The frozen gate requires a
defined relative reduction of at least 25%, superiority to readout correction, clip
retention within 15% and complete passing goal retention. It did not pass.

Both goal-retention arms used the same 20 source-distinct cases and a 50-block maximum.
All 40 case executions finished, but two baseline and four adapted trajectories
terminated early without verified 95% whole-T goal coverage. Thus the retention
comparison is incomplete under the existing criterion. Clip prediction MSE improved
from 0.008682 to 0.006402 and passed its separate retention threshold; it does not
substitute for the missing goal-retention qualification.

E2 charged 58,475 simulator steps: 38,827 collection, 3,140 adaptation/readout
contexts, 6,880 evaluation contexts and 9,628 goal-retention steps. The independent
audit reconciles these components without rerunning models or simulations.

The [model durability audit](e2_model_durability_audit.json) matches all seven uploaded
payload files to private revision `cd896c7bcbb2377140fcda988fe2ef59673b1454`. That
upload includes the model and retention evidence; it does not include `repair.json`,
the saved evaluation rows or paired reporting figures. Completed output directories
remain unchanged, and the downstream gates were neither relaxed nor retried.

The [presentation supplement](repair-presentation/README.md) fixes the original
paired-report table formatting and annotates the same 18 saved examples with units,
complete hazard extents and explicit censoring. It explains both the first-recipe
fallback and the 80 px fixed-margin fallback after an undefined development target.
All saved metrics, examples, original outputs and gates remain unchanged. Sixteen
focused CPU tests, independent provenance/code review and visual review passed.


The [summary charts](repair-summary-charts/README.md) provide PNG and vector PDF
figures for test censoring bounds and acceptance, ordinary prediction and goal-retention
completeness, and charged E2 costs. They retain undefined point rates, distinguish
identification bounds from confidence intervals, and label full-horizon counts separately
from goal success. [Validation](e2_summary_charts_validation.json) includes seven
focused tests and independent source/visual review. The original E2 outputs are unchanged.
