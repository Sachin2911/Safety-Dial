# Push-T continuation after bounded bank-source exhaustion

Status: the representative test bank completed, then stress-bank construction failed after 17 roots. The completed banks and partial stress bank are preserved. The [failure record](stress_generation_failure.json) and [test completion audit](test_bank_completion_audit.json) distinguish the completed evidence from the failed stage. [Fresh stress-only recovery](../continuation-v4/README.md) completed with the same fixed roots and proposal primitives; development diagnosis passed and the managed study has advanced to repairability and retention. Bank generation remains frozen at commit `8601568`; the downstream reporting and oracle work at `d56abc2` passes all 345 repository tests and Ruff. Main-study outcomes remain pending.

The [v2 failure record](../continuation-v2/bank_generation_failure.json) preserves the
partial banks: 24 development roots with 166 tapes and 26 test roots with 416 tapes.
The single-candidate builder exhausted its 605-source test/familiar allocation before
reaching 64 roots. Its complete simulator-cost ledger was not saved, so that failed
run's exact cost is unavailable. No completed-study claim uses those partial banks.

The new recovery retains the exact 605/605/605 development, familiar-test and held-out-test
source reservations and the same 185 historical source exclusions. It freezes two
seeded temporal pairs per source, crossed with controller depths 0, 2, 4 and 6, before
simulation. Initial depths rotate across sources. There are at most eight candidates
and one accepted root per source episode. Geometry, hazard sizes, source-family rules,
controller compute and arena checks remain fixed. Rejected generation candidates never
move a source between roles, and final-test performance does not select candidates.

The declared bank sizes remain 24 development roots with eight distinct tapes each and
64 roots per source family with 16 tapes each in both final banks. The exact nominal
tape is required in representative banks; the remaining tapes use the declared Gaussian
scales. Stress banks reuse the identical final roots and retain their declared stress
primitives. Proposal checks use the float32 controls that are actually executed. No
post-query outcome removes a bank branch. Journals, roots, layouts and charged steps
are saved incrementally, and incomplete banks are rejected by downstream readers.

[The managed workflow](../../../../configs/pusht/continuation-v3.yaml) builds the development
bank first, then requires independently replayed feasible routes before generating the
final test and stress banks. Its ten stages continue through decomposition, repair with
20 fixed retention episodes, three paired acquisition seeds and four additional budgets,
checkpoint-specific retention, geometric transfer reporting and conditional closed-loop
evaluation. Negative scientific gates remain valid stops; no favorable outcome is assumed.

The completed physical probes from continuation-v2 are reused. Bank bundles include the
frozen candidate plan and source hashes. Exact code snapshots remain local, outside
uploaded bank directories, while the optional source-backup approval is pending.

The [first completed-root audit](first_root_audit.json) passed: eight unique guarded tapes,
matching saved progress and journal costs, and body-specific contact metadata. The
[frozen plan verification](frozen_plan_verification.json) confirms that the source
reservations, exclusions and requested bank sizes match the declared protocol.

The [additional candidate-pass audit](additional_candidate_pass_audit.json) confirms that
the final-bank builder continued beyond its first 605 candidates under the frozen plan.
At the audited snapshot, the next pass had added one root from a new source, giving
29 distinct roots and 464 tapes while retaining every prior root. Candidate identities,
source skips and charged-step accounting agree. This is an incremental metadata audit;
the full final banks and their performance evaluation remain incomplete.

The [familiar-family completion audit](familiar_family_completion_audit.json) verifies
64 unique familiar-test source roots and 1,024 completed metadata branches, with
16 distinct prequery float32 tape hashes per root. Frozen candidate identities,
source reservations and 176,353 charged steps agree at family completion. Held-out
collection then started and had reached four roots at the audit snapshot. This is
construction metadata evidence; the full test/stress banks and outcome evaluation
remain incomplete.

The later analysis stages now emit cumulative-horizon attribution and pose/clearance
errors by regime, an E2 paired table with deterministic observed examples, and charged
history-replay ledgers. E3 records exact model/input and candidate-pool identities,
root-cluster intervals, counts and compute, plus a standalone result manifest.
Acquisition controls are rounded to their stored float32 representation before the
unchanged arena guard and execution.

The prospective acquisition root builder freezes all 5,000 source episodes already
reserved for acquisition, with one candidate and at most one accepted root per source.
It retains the same geometry, nominal-depth schedule and requested 96 roots. Every
construction attempt is journaled; finite exhaustion remains a failure with preserved
partial provenance. This change is prospective and does not inspect final-test outcomes.

A [separate oracle workflow](../../../../configs/pusht/continuation-oracle-v3.yaml) is
queued under supervisor to follow the completed random/boundary acquisition and E4 report. It uses the
identical source-root cache, candidate tapes, common-seed keys, root schedule, three
acquisition seeds and four additional budgets. Its entire queried pool is charged and
persisted incrementally. Any censored tape makes the full-future optimistic-error
reference undefined; all paid outcomes remain stored and the workflow records that
scientific stop. Oracle cost does not support an interaction-efficiency comparison.

The [development completion audit](development_completion_audit.json) verifies exact
root/tape counts, the saved journal and 117,182 combined development-generation and
feasibility steps, pinned private upload bytes, and the required development-to-witness-
to-final chronology. The [parent manifest](manifest.json) records the completed-stage
provenance while the overall study remains running.
