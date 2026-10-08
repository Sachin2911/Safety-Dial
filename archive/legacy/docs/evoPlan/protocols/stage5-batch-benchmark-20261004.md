# Main batch throughput validation declaration

This bounded engineering check measures the frozen model at the six batch shapes
proposed for the readiness main study. It does not enable that study or execute
search, the ROSARL comparison, physics, collection or training. The main protocol
remains a disabled review candidate.

The benchmark cycles the existing 64 noise-fit history rows to reach the requested
sizes. A 320-row timing batch therefore contains only 64 distinct existing roots,
not 320 independent episodes. Its readouts are retained for query integrity, not
used to claim new scientific evidence or change noise calibration.

| Role | k=0 roots times samples | k=1 roots times samples |
|---|---:|---:|
| Fitness | 32 times 1 | 32 times 4 |
| Selection | 64 times 1 | 64 times 16 |
| Final audit | 320 times 1 | 320 times 32 |

Each shape runs one zero-residual candidate for ten model blocks, with two charged
warmups and three charged measurements. Total predictor rows are exactly 590,400.
The check separately measures initial clipped action deviations for the zero
residual and 16 fixed-seed Gaussian vectors at the declared initial scale. These
18 policy calls include the baseline and spend no physics or predictor rows.

Timing includes CUDA synchronization and copying rollout output to CPU; query
archive compression happens afterwards. Peak allocated and reserved CUDA memory
are recorded. Timing projections cover model queries only and exclude physics,
collection, encoding, optimizer work, archival, analysis and interruptions. A timing
projection is not a wall-clock guarantee or a power calculation.

The configuration and executable source must be committed before execution.
Failure retains its cost counters and query journal; there is no automatic retry.
Use a new run ID for a separately declared revised check. Final root counts and
search-seed counts still need review, including variation in the unmeasured ROSARL
arm. Changed batch shapes require a new timing check before the main launch.
