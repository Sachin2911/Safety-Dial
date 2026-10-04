# State information and closed loop validation leave a visual control gap

With the same 444-512-256-60 network (374,588 parameters), the selected visual
controller had 4/24 development failures and mean progress 2.120 m. The privileged
state reference had 2/24 and 2.103 m. The reference uses true clipped environment
observations and their block differences; it is not deployable inside the visual
world model. Both also receive the preceding executed action block.

Selection used 64 existing setA validation episodes, 32 per family, with no
development outcomes. Candidates were epochs 20, 50, 100 and each seed's best
offline checkpoint. The selected state reference had zero validation failures;
the visual reference had 7/64. Progress had to
reach half the recorded reference before candidates were ranked by failure count,
then progress, offline MSE, seed and epoch. This real-validation selection and the
larger network were both changes from the earlier history experiment; their
individual effects are not isolated by a comparison to that earlier result.

Training completed under protocol bb73e80. The original job then stopped because
the descriptive progress-floor string could not be converted to a number.
Continuation e4fa8f7 preserved all checkpoints and corrected only that intended
ratio to numeric 0.5. No training was repeated. The same predeclared candidate
pool and validation roots were hash-verified before continuation.

The results support further diagnosis of temporal information and imitation, not
a claim that a visual controller is impossible. Neither controller passed a new
Gate 0, and the state reference must not be substituted for a visual policy.

All 634 tests passed with 49 warnings and no skips or failures. Ruff passed.
The continuation used 158,400 simulator steps; the failed setup had already spent
6,400, so the combined count is 164,800. Training used 27,000 updates once.
The continuation used 240 imagined rows, 7,608 renders and 7,544 encoded frames,
with another 256 renders and 192 encodes in the failed setup. Its wall time was
31.88 seconds, excluding the earlier training run.

Selection, individual validation results, development rows, source hashes,
weights and trajectories are retained. The original failed run is preserved too.
Archival is pending. The active controller goal remains incomplete.
