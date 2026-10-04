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

The subsequent bounded development search pilot found k=0 gap amplification of
10.94 pp (exploratory 95% interval 1.30 to 20.31 pp) across four search seeds and
96 previously inspected assessment episodes. Real failures rose from 19.79% to
30.73% while imagined failures remained 0.52%. Noise alone did not clearly reduce
amplification. This supports fresh-episode confirmation but is not that confirmation.

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
[results index](../results/README.md) links the completed evidence. The main Stage 5
experiment and new-bank collection have not begun. A separate bounded development
search/power pilot completed under the previously adopted readiness scope, with
k=0 and k=1, four paired seeds and no extra unsafe penalty. It does not execute
the ROSARL comparison or replace this scientific review.
