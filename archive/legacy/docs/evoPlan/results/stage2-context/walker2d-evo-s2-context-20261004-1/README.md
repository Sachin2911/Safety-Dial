# Full temporal context did not reduce development failures

Both selected visual controllers failed 5 of 24 development segments: one of eight
PPO roots and four of sixteen PPO-Lagrangian roots. Short context achieved mean
progress 2.033 m; full context achieved 2.088 m. Both imagined zero violations.
The existing mixed controller's 4/24 remains the earlier reference. Neither new
controller qualifies as a viable starting policy, and Gate 0 was not rerun.

The matched 696-512-256-60 networks each have 503,612 parameters. Full
context uses three latent frames and two preceding executed action blocks. The
control masks the oldest latent difference and action block, retaining two frames
and one action block. Both use frozen feature and action scalers. Early cached rows
without complete context were excluded from both arms, leaving 45,506
fit examples balanced across policy families. Validation contains
11,700 PPO-Lagrangian and 11,700 PPO examples,
with equal total family weights.

Each arm trained three seeds for 100 epochs and 13,500 updates. Candidates were
epochs 20, 50, 100 and each seed's best offline checkpoint. Selection used the same
64 setA validation roots as the information experiment, never the 24 development
outcomes. The recorded progress reference was reused from its pinned result.
Candidates needed at least half that progress, then were ordered by violations,
progress, offline MSE, seed and epoch.

Short context selected seed 20261005, epoch
50, with 7/64 validation failures.
Full context selected seed 20261005, epoch
50, with 6/64 validation failures.
The paired development failure-rate difference is zero, with a source-episode
bootstrap interval [-0.208,
0.208]. Mean progress difference is
0.055 m, interval
[-0.096, 0.225].
These small, repeatedly inspected development results do not establish equivalence
or impossibility. They do not support advancing either controller to evolutionary
search. More temporal context alone did not repair the observed control gap.

The run consumed 158,400 simulator steps, 480 imagined rows, 16,168 renders,
16,104 encoded frames and 27,000 optimiser updates in 74.31 seconds. The original
cached-data collection and reused reference costs belong to their source runs;
no new recorded-reference steps were spent here. All 637 tests passed with
49 warnings. Ruff passed before execution. Protocol commit: 9aeeafd.

All checkpoints, candidate validation trajectories, final trajectories, selection,
resolved configuration, source hashes and logs are retained. Private archival is
pending. Stages 3 through 7 have not run; the active controller goal is incomplete.
