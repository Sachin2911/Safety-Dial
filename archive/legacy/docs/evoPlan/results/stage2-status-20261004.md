# Stage 2 status, 4 October 2026

The real and imagined rollout harness works and all 637 tests pass. The visual
starting controller is not reliable enough to advance. Gate 0 remains unpassed,
and no evolutionary search or stages 3 through 7 have run.

## What the recent diagnostics show

All development counts below use the same 24 repeatedly inspected S4 roots.
They are diagnostic outcomes, not independent final-test estimates.

| Comparison | Development failures | Interpretation |
|---|---:|---|
| Earlier mixed-data visual controller | 4/24 | Retained reference |
| One-round student-state aggregation | 6/24, same as teacher-state control | No demonstrated repair |
| Previous action history | 4/24 versus 6/24 masked control | Better point estimate, uncertain paired difference |
| Larger visual controller with real-validation selection | 4/24 | Still fails |
| Privileged state reference with the same selection | 2/24 | Diagnostic only, not a visual controller |
| Full temporal context | 5/24, same as matched short-context control | Extra context did not repair development failures |

The state reference had zero failures on its 64 validation roots; the matched
visual controller had seven. Full context had six validation failures against
seven for short context, but both had five development failures. The development
outcomes were excluded from checkpoint selection, although repeated inspection
has informed experiment design. The 256 Gate 0 test-role roots were inspected in
earlier linear-policy runs; a future rerun must acknowledge that reuse.

The exact source teachers acting every simulator step had zero failures on these
24 development roots. Holding a teacher action for a full block failed 22/24.
Using a single final PPO teacher instead failed 3/24, and a single final
PPO-Lagrangian teacher failed 7/24. Teacher identity and feedback rate therefore
matter; substituting one final actor is not an established solution.

The latest context experiment still imagined zero failures for both controllers,
while reality produced five each. That gap is evidence of model error before
selection pressure; it is not yet the proposed evolutionary cheating result.

## Next repair decision

Keep the original mixed controller as the reference. The next useful direction is
a controlled comparison of imitation targets: sufficiently broad, noiseless
source-teacher action blocks versus the recorded noisy blocks, using fit episodes
only for training and the established validation roots for selection. Specify the
sampling and simulator budget before execution. The earlier small aggregation
comparison does not establish that broader teacher relabelling will help.

Do not loosen Gate 0, substitute privileged state inputs, or proceed to evolution
on the strength of imagined safety. Freeze a viable visual candidate before a
separate Gate 0 run with the existing thresholds.

## Preservation and approval

The linear-block, MLP, coverage, teacher and aggregation bundles are archived at
pinned private Hugging Face revisions recorded in their upload receipts.
Five newer bundles, including the failed information setup and successful
continuation, were uploaded after explicit user approval. The 202.21 MiB payload
is pinned by the five new upload receipts. Remote file sizes and LFS SHA256 or Git
blob hashes were verified against the audited local bytes. These receipts supersede
the earlier pending-archive text retained in the immutable run reports.
No GitHub push has been performed.

See the individual reports for exact configurations, accounting and uncertainty:
[action history](stage2-action-history/walker2d-evo-s2-action-history-20261004-1/README.md),
[information comparison](stage2-information-resume/walker2d-evo-s2-information-resume-20261004-1/README.md),
[context comparison](stage2-context/walker2d-evo-s2-context-20261004-1/README.md).
