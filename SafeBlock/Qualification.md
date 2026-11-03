# Block Plucking qualification protocol

Written 1 November 2026 for authorisation. **Not authorised, not run.** It rebuilds
the migrated UNISafe stack on a fresh rented host and answers the five questions the
[design](BlockPlucking.md) needs before anything is frozen. No optimizer update,
no training and no reserved start is touched.

## Host

As task 9's proposal: one GPU with at least 24 GB (RTX 3090 or 4090 class), at least
64 GB RAM, 12 effective cores, 200 GB disk, driver 580.x, the Vast template image
`cuda-13.0.3-cudnn-devel-ubuntu24.04-py312` recorded in
`archive/20261022/configs/vast/unisafe-platform.json`. Speed-test downloads before
installing (task 11 lesson). Rate ceiling $0.65 per hour. Compute cap $6, preservation
reserve $2, total cap $8, maximum 7 hours including setup. Preservation to the
private dataset before deletion, with every archive member hash-verified locally.

## Steps, in order

| Step | What runs | Episodes | Gate |
|---|---|---|---|
| Q1 Platform | `setup_unisafe09.sh`, checkpoint download, pins check; native actor on start 9,100,000 with and without the filter | 2 | Outcomes and control counts match receipts `09-unisafe-A-v7.json` for engineering-00 and -01 |
| Q2 Parity | Adapter with zero residual, filter in shadow, on 3 fresh starts, against native with matched seeds | 6 | Bitwise identical executed commands and physical arrays |
| Q3 Authority | 24 generic genotypes (hidden 8, history 3, weights N(0, 0.1)) on 4 shared starts, filter in shadow | 96 | At least 3 distinct completion counts across genotypes; at least one genotype beats native by 2 or more on those starts; violation rate across genotypes between 5 and 60 percent |
| Q4 Throughput | In-process resets on 3 starts twice (parity against fresh-process runs); one process with 4 environments on 16 starts, then 8 if 4 passes | 6 + 16 (+16) | Outcome parity on the 16 starts against single-environment runs; measured episodes per lane-hour recorded |
| Q5 Anchors | Native, one untrained genotype, release rule d15_g075, native with filter live, each on the 16-start selection panel | 64 | None; these are the reference rows for the freeze |

Starts for Q2 to Q4 come from 3,000,000 upward and are development starts thereafter.
Q5 uses the selection panel 3,200,000 to 3,200,015, which is permitted because no
candidate is selected here.

Totals: at most 206 episodes, about 2.6 lane-hours at 44 s per episode, plus up to
1 hour of setup and warm-up and 30 minutes of preservation. Hard caps: 230 attempts,
75,000 controls, 7 hours wall time.

## What each answer decides

- Q1 failing stops everything; the stack has drifted and must be repaired first.
- Q2 failing means the adapter path is not a clean wrapper; fix before Q3.
- Q3 failing on spread means the interface has no authority at these bounds; the
  declared fallback is the gain form, under a new protocol. Failing on violation rate
  means the needle does not move for generic adapters and the design's safety claims
  must be rethought.
- Q4 sets the main comparison's budget: 1, 2 or 3 optimizer repeats and whether PPO
  stays in the first freeze.
- Q5 gives the anchor rows that the frozen analysis will compare against.

## Rules

Freeze source hashes, seeds and this file before the first episode. Record every
attempt, including failed starts and invalid episodes, with actual controls, physics
steps, model queries and elapsed time. No outcome-driven reruns. The frozen task
predicates, filter settings, checkpoint revision and action contract from
`configs/unisafe09/` are reused unchanged. This protocol authorises no training.
