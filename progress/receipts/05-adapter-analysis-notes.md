# Task 5 offline analysis and preservation notes

24 October 2026, after the engineering checks passed and the unchanged v4 pilot
launcher was admitted. Two workstation-only reporting changes were made. Neither
is copied into the running instance, alters any action or changes any outcome,
panel, seed, budget, runtime validator or admission rule.

The analysis now labels the last applied coefficients separately from the state
after the last update. In fixed mode, the applied coefficients are [0.2, 0, 0],
while the unused adaptive state remains zero. The previous `final_coefficients`
label reported that internal state and could be mistaken for the fixed gain.
The new fields are `last_applied_coefficients` and `state_after_last_update`.
This is a reporting correction, not a change to controller mathematics.

Preservation additionally includes the local task 4 gate-evidence snapshot in
the authorized private dataset. Automatic approval review rejected sending that
private evidence to the rented host. The safer launch succeeded with only a
Boolean admission flag, the decision receipt's SHA-256 and verification of the
two engineering recordings already on that host. No task 4 evidence was sent to
the rental. The admission receipt remains on the workstation.

Executed engineering and pilot source archives retain the frozen v4 files.
The final analysis source and its hashes are preserved separately with the
results so both the actual execution and offline reporting are reproducible.
