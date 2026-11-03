# Recurrent-adapter pilot: optimizer correction

Frozen before the corrected run, 1 November 2026. Sachin explicitly instructed
the agent to fix the optimizer after the first attempt was stopped, and to keep
the rental on until told otherwise. The replacement host and $3.70/hour ceiling
are unchanged. This authorizes one corrected implementation run, not tuning after
an unfavorable result. The original protocol and every original output remain intact.

## Failure and correction

At the actual 19,172-parameter dimension, pycma 4.4.4 selected two-point step-size
adaptation (TPA). Its first two samples have special roles. The original runner
permuted the population before `tell`, conflicting with that requirement. A warning
at the fourth update exposed the error. The original 19-dimensional software check
did not cover this dimension-dependent default or shuffled updates.

Use explicit cumulative step-size adaptation (`CMAAdaptSigmaCSA`), diagonal covariance,
and `CMA_mirrors=0`. Preserve `ask` order through `tell`; equal fitness stays equal,
without score jitter or presentation shuffling. The former tie seed is unused.
A regression exercises all four updates at the full network dimension, including
an entirely tied generation, no warnings, unchanged candidate order and checkpoint
restoration. The runner and regression share `helpers/pilot_optimizer.py`.

## Fresh run and accounting

`adapter_pilot_v2.py` writes to `outputs/adapter-pilot-20261101/run-v2`. It freezes
this correction, the original design and the new implementation before episodes.
Repeat the original parity checks, witnesses, 256-episode CMA search, 256-episode
random search and 48-episode final evaluation. Keep all original panels, seeds,
network, information, scalar rank encoding and selection rules. No final AP/adapter
evaluation episodes ran in the interrupted attempt, so that panel remains unused
for selection. Training outcomes do not motivate any architecture or fitness change.

The stopped attempt charged 459 resets and 58,179 attempted actions: 451 complete
episodes and eight interrupted episodes. Its coordinator had collected 415 results;
another 36 completed episode files are retained and must be included in accounting.
Do not report the flawed CMA run as a valid optimizer comparison. Its observation
that candidates produced different trajectories is only engineering evidence.

The corrected run has the same 592-reset and 118,400-action absolute caps and two-hour
job deadline. Combined with the stopped attempt, the new cumulative bounds are
1,051 resets and 176,579 attempted actions. Planned total: 1,050 resets. Report
both attempts, software-check failures, actual model work and elapsed time. Interrupted
calls are charged conservatively before execution; saved traces may end before the
last charged call completes. Rental shutdown remains manual.

Run the unchanged frozen audit assertions with cached NPZ reads, then independently
validate native completion from saved states. These postprocessing checks add no
simulator or world-model calls. Preserve both attempts in the verified private archive.
