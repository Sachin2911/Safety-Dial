# What the candidate-quality diagnostic changes

The bounded diagnostic completed under protocol commit `a74a274`. It evaluated 89 fixed interventions on 32 simulator-selection episodes and 64 separate development-check episodes. All episodes were previously exposed in Stage 6. No new final bank was collected. The source model, probe, baseline controller and residual projection remained frozen.

## Candidate quality and selection

Thirteen candidates beat the baseline's observed selection failure rate while retaining at least 90% progress. None of the four original imagined winners did. The fixed simulator rule selected a half-strength version of the k1 ROSARL-style winner from seed 20261301: selection failure fell from 6/32 to 4/32.

That apparent gain did not carry over. On the separate development check, this candidate failed on 6/64 episodes versus 4/64 for the baseline. The paired difference was +3.125 percentage points, with an exploratory unadjusted 95% interval [0, +7.8125]. Mean progress retention was 99.04%, with interval [97.83%, 100.02%]. It rescued none of the four baseline failures and introduced two new failures. The simulator-selected capped winner also failed on 6/64, retaining 99.32% mean progress. Smaller or capped corrections did not produce a demonstrated baseline improvement.

Only one of the 88 nonbaseline candidates beat the baseline on the check criterion: an outcome-blind population sample from the first generation of k0-zero, seed 20261301, candidate index 107. It failed on 3/64, retaining 99.09% progress, rescuing two baseline failures and introducing one new failure. It had tied the baseline at 6/32 on selection, so the declared strict improvement rule did not accept it. This is a retrospective candidate-quality observation, not an independently selected success. No candidate met the improvement criterion on both banks.

Both common imagined selectors chose candidate-001, which failed on 12/64 versus the baseline's 4/64, retaining 100.60% progress. Every candidate had zero k0 imagined failures. Across the fixed correlated pool, the descriptive k1 imagined-versus-real failure rank correlation was only 0.041. This is not a population correlation estimate and does not replace the earlier transfer diagnostics.

## What the safety readings suggest

For the baseline, the retrospective 3/64 candidate, the frozen overall choice, and the frozen capped choice, every dense failure was also visible at a true block end. The frozen probe applied to real images detected every one of those endpoint failures, with respectively 4, 4, 5 and 2 false-positive episodes. In contrast, k0 imagination predicted no failures for any of these controllers. For these specific traces, neither missed events between block ends nor probe false negatives explain the imagined false safety. This directs attention toward the predicted trajectories. It does not prove that probe errors are irrelevant across the complete pool or that correcting dynamics will improve control.

## Recommended next research question

The present evidence does not justify a larger confirmation of controller improvement, nor a claim that action caps solve the problem. It also does not establish that the policy class is incapable of improvement: only a bounded, highly correlated pool and small reused banks were checked.

Before another optimization sweep, use the saved real/imagination traces to localize where predicted height and pitch diverge before failure. Separate teacher-forced prediction from closed-loop error growth, and assess the safety readout on the same real frames. If this identifies a repairable dynamics defect, declare one targeted correction and test it on a new development split, with equal adaptation budgets and an unchanged baseline. Any subsequent controller update needs independent simulator selection and a fresh final check; useful short-branch gains must then survive longer rollouts. Hopper and independent world-model replication remain later robustness tests once the method is stable.

The immediate requested diagnostic is complete. These next experiments are recommendations, not work claimed as executed.

## Verification and costs

The completed run used 768,960 predictor rows, 854,400 simulator steps, 85,440 encodes, 85,504 renders including the startup fingerprint, and 386.06 seconds. There were zero gradient updates and zero new source-collection steps. Historical source collection, model training and parent search are additional costs. No trained controller was promoted, no historical experiment was changed, and no external upload was made.

All 534 query archives, 112 frozen source files and 19 parent input files were verified. The full analysis and frozen choices reproduced exactly; both baseline banks reproduced the earlier run bitwise. The offline report's original timestamp check referred to a non-retained start-time field; the report-only correction checks retained completion timestamps and frozen barrier identities. No experiment query was repeated for that correction. The action replay audit is recorded separately in `action_audit.json`, including its inference-only cost.
