# Bounded readiness search and power pilot

The user asked to move the agreed readiness direction forward. This is a bounded
engineering/power pilot on already inspected development episodes, within the
adopted readiness route. It does not assert supervisor approval of the main
Stage 5 declaration, run its ROSARL arm, generate its final bank, or pass any gate.
The strict-controller repair sequence remains stopped.

The scientific recommendation is to retain Walker2d, the frozen visual controller
and k=1 as the main proposed imagination setting, with k=0 as its control. The
main study's scientific scope and ROSARL adaptation remain for supervisor review.
Useful implementation and power preparation proceeds in parallel.

## Prospective design

Four paired search seeds compare k=0 and k=1 at population sizes 16 and 256, each
for 16 generations. Both use the shared first-violation task-return truncation
with zero extra unsafe penalty. Checkpoints are generations 1, 4 and 16. This is
16 searches and 48 selected-policy slots, plus the exact frozen real baseline.
CMA-ES uses diagonal covariance and 1,020 residual coefficients around the frozen
ensemble, with base gain fixed to one. Initial expected raw action RMS is 0.05.
Projected-feature RMS scaling uses the prior noise-fit episodes only.

The existing 256-episode baseline bank is partitioned into 96 fitness, 64 selection
and 96 assessment episodes. The 64 prior noise-fit episodes are all in the fitness
role. The remaining indices are assigned by a fixed permutation, independent of
new outcomes. All roles are previously inspected development evidence. Nothing
from this pilot will be called an untouched final test.

Each generation uses 32 fitness roots and common random noise across its candidates.
The noisy arm uses four fitness samples, 16 selection samples and 32 assessment
samples per root, with separate role-specific streams. Candidate-local model calls
preserve fixed kernel shapes within each phase. Zero noise uses one sample.

The generation's fitness winner is nominated to the separate fixed selection bank.
The checkpoint winner is the best baseline or nominee available by that checkpoint,
using only that bank. No cross-generation fitness score is used as a checkpoint
selection score. Every pick and its parameters are frozen before any new real
assessment. Both k=0 and k=1 audit predictions accompany every selected policy's
real trajectory. Actual executed actions are fed back into visual-policy history.

## Costs and failure accounting

| Component | Charged amount |
|---|---:|
| Fitness predictor rows | 27,852,800 |
| Selection predictor rows | 1,479,680 |
| Assessment predictor rows | 1,647,360 |
| Archived zero-control reproduction | 640 |
| Total predictor rows | 30,980,480 |
| Real steps | 470,400 |
| CMA-ES generations | 256 |
| Candidate fitness evaluations | 34,816 |
| New episode collection / gradient updates | 0 / 0 |

A two-hour wall-time cap is checked between calls. Model predictions and physics
steps are counted at their call sites, including partial failures. Logs and partial
bundles are retained on error. This pilot deliberately refuses implicit resume:
the existing CMA checkpoint alone does not include the new nominee-selector state,
so a failure cannot silently resume with missing selection history. A future main
runner needs complete resume-state tests before launch.

## Interpretation

Paired high-minus-low contrasts use population 256/generation 16 versus population
16/generation 1, separately for each noise level. The analysis reports real-failure
changes, gap changes and the difference between those gap changes. Bootstrap draws
resample search seeds and assessment episodes on crossed axes with all arms paired.
Exploratory variance components illustrate power over proposed seed/root counts.

Four seeds are insufficient to establish stable variance estimates or guarantee
main-study power. The ROSARL arm's variance is not measured at all. The pilot is
therefore evidence for planning and an executable search check, not a replacement
for the main protocol review or a positive-result gate. Main final-root and seed
counts remain unlocked until that review is complete. No controller repair, noise
retuning or automatic expansion follows an adverse outcome.
