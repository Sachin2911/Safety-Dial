# Task 7: repair DSRL integration and obtain a learning result

25 October 2026. Sachin authorized the next step, a four-GPU rental and persistence
through recoverable failures. The original $15 cumulative DSRL cap remains in force.
Task 6's conservative $3.5396544875 is retained; its stopped queue is not resumed.
This is a new experiment with new output paths, receipts, seeds and source freezes.

The objective is a working noise-policy integration and an interpretable result.
First diagnose the numerical failures on existing diagnostic-pool observations,
then verify live controller execution and run the development learning comparison
when correctness and the measured remaining allowance permit it. A failed diagnostic
starts diagnosis and a recorded repair, rather than automatically ending the task.
No guarantee of improved completion is implied.

## Numerical diagnosis before new simulator outcomes

Reuse the eight task-6 saved observations and their recorded Gaussian draws, plus
the recorded engineering image/state batch. Hash them before rental. They contain
only diagnostic-pool tasks 1, 4, 10 and 11, and never enter training replay.

1. On each of four policy servers, replay each of eight inputs three times using
   the original RNG state. Compare repeats from the same live process at the original
   1e-5 absolute/relative tolerance and require identical converted gripper commands.
   Compare servers separately, retaining every numerical difference as a reproducibility
   result. Cross-server agreement is not silently substituted for same-process identity.
2. On server 0, provide the exact full native draw through the explicit-noise path
   and compare with that process's native replay. This checks that injection preserves
   the decoder when the noise is held fixed. It must pass the same 1e-5 criterion.
3. On server 0, repeat the original fourteen magnitude-1 tiled vectors on all eight
   inputs. Preserve the old two-input/two-task gripper criterion as a reported diagnostic,
   not a launch gate. It samples a small set of vectors, and useful arm steering need
   not flip a gripper command. Integration admission instead requires finite, correctly
   attributed decoded actions and repeatable nonzero arm or gripper influence. This
   distinction is declared before new outcomes. It does not establish a useful lever
   until a completion comparison succeeds.
4. On a recorded fixed batch, compare repeated author updates and wrapper updates,
   reporting parameters, optimizer state, RNG and nonfinite values separately. Run
   five paired checks with default GPU settings to diagnose the previous discrepancy,
   then five with learner-only deterministic XLA operations enabled before JAX starts.
   Production uses the deterministic setting if it passes the original 1e-6 full-state
   criterion. If needed, test deterministic operations with autotuning disabled, followed
   by source-level diagnosis. Never increase tolerance to pass an observed difference.
   Every update attempt is reserved durably before dispatch, including failures.

The first model-only battery uses 216 queries: 96 native repeats, eight native-draw
passthroughs and 112 tiled candidates, plus separately counted startup queries.
At most 1,000 diagnostic VLA queries and 2,000 engineering updates are permitted.
Repairs and extra diagnoses use fresh operation IDs and an explicit source amendment;
failed attempts and earlier results stay in the record. No 90-minute engineering
deadline is inherited. The money deadline and preservation reserve remain binding.

## Live validation and learning

Preserve the task-6 benchmark, actor information, pinned author SAC configuration,
reward, gamma, terminal handling, recording and stochastic evaluation semantics.
Use learner seed 6060 for comparability, but fresh episode seeds and actor streams.
Native Gaussian noise remains distinct from fixed zero and tiled Gaussian warm-up.
The Jar checkpoint, native normalization, cameras, 20 Hz, eight executed controls
per query and 2,000-control horizon stay fixed. No monitor veto or action residual.

Freeze the complete episode schedule before any new episode. Reuse the six engineering
job roles on pool tasks, with root seeds 70000 through 70005. New training roots are
71000 through 71003 across development tasks 0, 3, 6, 9, 15 and 17. Evaluation roots
are 72000 and 72001 on those six tasks, paired across native, fixed-zero, bounded-random
and learned-noise arms. Actor RNG streams use a task-7 namespace. Held-out tasks
receive no new episodes.

After numerical checks pass, four native engineering episodes and two live noisy
episodes verify the recorder and actor path. Profile 128 learner updates, memory,
latency and CPU/GPU actor agreement. Check source parity again on the live recorded
batch. Preserve an initialized actor separately from any diagnostic optimizer state.

Freeze stage B with the resolved numerical configuration and measured resource
projection before development collection. Its target is 24 training episodes and
48 evaluations, with the same six-episode warm-up, 20 updates per new transition,
three-episode publication rounds and final episode-24 checkpoint as task 6's unexecuted
design. No engineering data enter replay. Do not cherry-pick checkpoints, discard
unsuccessful attempts or tune reward from evaluation outcomes. Report all four arms
with denominators, safe completion, violations, unfinished episodes and engagement.
Retain the original descriptive screen: two additional safe completions over native,
no increase in violations, with uncertainty and no evolutionary claim.

If the complete stage B cannot fit, use the remaining allowance to finish diagnosis
and live integration evidence. Any smaller learning allocation requires its own
explicit pre-outcome freeze and must be described as an incomplete pilot, not as the
full comparison. Work does not stop merely because the first parity attempt fails.

## Accounting, rental and preservation

Reuse the tested management primitives with a new `dsrl07` namespace. Import task 6's
verified historical active charges, transfer and conservative stopped-disk allowance
into the cumulative ledger. Rent four RTX 4090s at no more than $2.20/hour including
200 GB disk, with driver 580.x, at least 250 GB RAM and 48 allocated CPU cores.
Prefer the same physical host and driver for diagnosis if available at that rate.
The $15 cap includes setup, retries, transfer and retained disks across both tasks.
The independent local and remote stop guards reserve 20 minutes for preservation
and begin provider stop requests five minutes before the financial deadline.
The old six-hour allocation limit is replaced by an eight-hour cumulative backstop;
the unchanged dollar cap is expected to bind first. No automatic budget increase.

The new interaction ceiling is 80 starts including two infrastructure replacements,
160,000 policy controls, 800 explicit warm-up controls, 21,000 VLA queries and 120,000
learner calls, at most 2,000 engineering and 90,000 learning. Report historical task-6
usage separately, never erase it. Count actor queries and scene/reset physics separately;
unexposed internal physics remains unknown. All paid time and failed work count.

Code goes under `experiments/`, local evidence under `outputs/dsrl07/`, and receipts
under `progress/receipts/07-dsrl-*`. Preserve evidence in the existing private dataset
`Sachioster/maniguard-jar-replication-20261022` under fresh `receipts/dsrl07/`,
`recordings/dsrl07/` and `jar/dsrl07_*` paths. Store checkpoints in private
`Sachioster/safety-dial-dsrl07-20261025`. Verify every uploaded file at pinned revisions
before destroying the stopped disk. Retain final provider absence and billing proofs.
Append the finished task to the Odyssey. Preserve task 6's evidence and source archives.
