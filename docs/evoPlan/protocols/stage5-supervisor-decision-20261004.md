# Supervisor decision draft: readiness-route Stage 5

**Draft for the user to review and share. Not sent.**

Can we proceed to a preregistered Walker2d selection-pressure study under an
explicit extension of the readiness protocol, keeping strict Gate 0 and the
transfer screen as reported failures and ending further controller repairs?

The frozen visual controller fails 51/256 fresh short segments but retains 100.3%
of recorded progress. Imagination predicts only 2/256 failures. Among nearby
controller variants, calibrated k=1 noise improves safety ordering (primary
residual-only rho 0.675), but progress ordering is 0.374, below the declared 0.5
threshold. All noise levels failed the joint screen. Predicted risk also remains
far below actual risk. These findings motivate a study of selection-induced error;
they do not establish safe control or calibrated safety probabilities.

The proposal compares k=0 and k=1, each with and without the ROSARL-style unsafe
penalty, across population sizes and generations. Both penalty arms use identical
termination so the penalty is the intended difference. Ten paired search seeds
and 320 fresh evaluation episodes are provisional until power and throughput
review; variation between search seeds was absent from the earlier 272-root
estimate. Fitness, policy-selection and final evaluation episodes are separate.
The observed-return adaptation of ROSARL needs your agreement before execution.

If the selection curve is flat, we will report its interval and detection limits,
retain the baseline prediction failure, and test the narrower risk-ordering claim
without changing endpoints after seeing results. No publication outcome is assumed.

The [full review draft](stage5-readiness-draft-20261004.md) contains the estimands,
scoring, episode roles, budget and null-result interpretation. The
[results index](../results/README.md) links the completed evidence. No Stage 5
search or new-bank collection has begun.
