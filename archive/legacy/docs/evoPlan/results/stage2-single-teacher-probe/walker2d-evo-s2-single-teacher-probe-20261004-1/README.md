# Fixed final teachers do not cover every development start

The two fixed final actors failed on some of the existing 24 development starts,
despite the earlier source-matched teachers having zero failures on these starts.

| Fixed teacher | Health violations | Mean progress in metres |
|---|---:|---:|
| PPO-e200 | 3/24 | 2.472 |
| PPOLag-e200 | 7/24 | 1.794 |

This diagnostic was declared in commit 750a968 before outcomes were inspected.
Both actors used their exported observation normalisation and acted at every
environment step without added noise. The evaluation used privileged simulator
observations; these are teacher references, not visual controllers.

The result rules out treating either tested final actor as a demonstrated safe
universal teacher on this development set. It does not establish why actors fail,
or that no other fixed teacher could work. No Gate 0, policy fitting, or model
selection based on these results ran.

The run charged 4,800 new simulator steps and 4,800 teacher queries, with no
rendering, encoding, imagination or optimisation. Physics and teacher queries
reuse the tested teacher_branch implementation. The run manifest pins the actor
weights, code and root identity. All individual outcomes are in diagnostic.json;
dense trajectories remain in the local run bundle pending archival.
