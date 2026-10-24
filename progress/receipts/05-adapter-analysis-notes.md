# Task 5 offline analysis and preservation notes

24 October 2026, after the engineering checks passed and the unchanged v4 pilot
launcher was admitted. Workstation-only reporting changes were made. None
is copied into the running instance, alters any action or changes any outcome,
panel, seed, budget, runtime validator or admission rule.

The analysis now labels the last applied coefficients separately from the state
after the last update. In fixed mode, the applied coefficients are [0.2, 0, 0],
while the unused adaptive state remains zero. The previous `final_coefficients`
label reported that internal state and could be mistaken for the fixed gain.
The new fields are `last_applied_coefficients` and `state_after_last_update`.
This is a reporting correction, not a change to controller mathematics.

Accounting also explicitly totals returned startup and reseeding queries from
the already recorded `adapter05_query_returned` markers. Attempt counts remain
unchanged. A startup request can be charged even when its response is lost during
initial compilation; returned counts therefore cannot replace attempted counts.

Preservation additionally includes the local task 4 gate-evidence snapshot in
the authorized private dataset. Automatic approval review rejected sending that
private evidence to the rented host. The safer launch succeeded with only a
Boolean admission flag, the decision receipt's SHA-256 and verification of the
two engineering recordings already on that host. No task 4 evidence was sent to
the rental. The admission receipt remains on the workstation.

Executed engineering and pilot source archives retain the frozen v4 files.
The final analysis source and its hashes are preserved separately with the
results so both the actual execution and offline reporting are reproducible.

A final workstation-only reporting command, `summarize_adapter05.py`, checks
completion against the unchanged plan, validates every numerical recording field
and video header, reconciles attempt and query ledgers, checks visual-review
coverage and aggregates tracking statistics by recorded steps. It performs no
simulation, tuning, exclusion or rescoring. Its source hash is included in the
analysis-source receipt. Full video decoding remains in `analyze_adapter05.py`.
