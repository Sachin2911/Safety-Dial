# Shared-controller phase-two study

The research scope and decision gates are in [phase2.md](../../../docs/research/history/20261018/phase2.md).
Implementation and evidence are tracked in
[the development log](../../../docs/research/research_notes/ManiGuard%20phase2%20development.md)
and [requirements ledger](../../../docs/research/research_notes/ManiGuard%20phase2%20requirements.json).
Implementation does not establish improved behavior, transfer or an evolutionary advantage.

The primary controller uses 64 shared observable features, eight recurrent units and
692 weights. Four categorical heads select query timing, joint-target caps, smoothing
and bounded opening delay. The VLA and common CLIP encoder remain frozen. Evolution,
random search and recurrent PPO use the same actor and episodic utility. Privileged
goal and safety predicates are confined to evaluation.

## Versioned inputs

- `families.json`, `sources.json` and `splits.json` pin all six families and preserve
  the original split assignments. Cabinet and Dusty remain excluded from development.
- `encoder.json` pins the public CLIP archive, loader, preprocessing and projection.
- `completion.json` defines supplemental requested-sequence completion. Original
  benchmark categories and raw violation flags are retained separately.
- `preparation.json` freezes 12 short checks and 24 full training-only reference
  episodes. Short checks never supply optimizer fitness.
- `preparation_amendment.json` binds one prospective recovery after Clutter identity
  initialization failed. It retains that invalid paid attempt and replaces the unclaimed
  Clutter composition slot with an identity recovery. The twelve-attempt, action and
  GPU-hour ceilings stay unchanged; the omitted composition check is disclosed.
- `pilot.json` describes 522 full training/selection episodes. Its initial status is
  `freeze_incomplete`; the coordinator refuses execution until the required evidence
  and measured resource ceiling are supplied.

The main study needs its own costed protocol after the pilot decision. Reserved data,
additional seeds, ablations and the literature baseline are not silently added to the
pilot allocation.

## Environments and execution

Use the existing separate simulator and policy environments. Do not sync the root
project environment into either one. A focused pure-code check is:

```bash
uv run --no-project --python data/maniguard/envs/sim/bin/python python -m pytest experiments/tests/test_maniguard_phase2_model.py experiments/tests/test_maniguard_phase2_adapter.py experiments/tests/test_maniguard_phase2_optim.py -q
```

Prepare the pinned public encoder with `scripts/maniguard/phase2_encoder.py` and new
family assets with `scripts/maniguard/phase2_download.py`, using their explicit intended
Python environments. Downloads and model files stay under ignored `data/` paths.

The policy service runs `scripts/maniguard/serve.sh --family FAMILY --phase2
--parallel-clients` on localhost. Each episode verifies the actual checkpoint metadata.
Never switch its checkpoint while a simulator episode is active. Managed phase-two
launchers and policy switches hold the same process-owned execution lease. Legacy/manual
entry points do not honor this lease and must not run concurrently. Each service startup
keeps a local transcript as well as streaming output to the instance's managed logs.

Run `experiments/scripts/maniguard_phase2_prepare.py` under supervisor, selecting exact
job IDs from `preparation.json`. Each attempt is claimed before launch, uses a fresh
bounded child process, and stores its source snapshot, actor, trace, benchmark result,
physics/encoder/query ledger and attempt receipt. The process stops on invalid evidence.
An existing claim cannot be launched again automatically. If present, the separately
hashed preparation amendment is applied before any new claim. The original manifest stays
unchanged, retired jobs are refused, and the failed receipt remains immutable. Interrupted or timed-out
children are terminated as a process group; a surviving descendant makes the attempt
invalid even if its parent returned success.

`--reaudit` reprocesses matching terminal saved attempts without physics. It retains the
original audit verdict, records the new reader hashes, and preserves physical counts
and elapsed execution time. Repairing a reader is not permission to rerun an episode.

The single-episode entry point is `experiments/scripts/maniguard_phase2_episode.py`.
It enforces split purpose, deterministic evaluation and reserved-evaluation freeze.
The original-executor comparison flag is restricted to engineering identity checks.

`PilotCoordinator` is the resumable dependency scheduler. A launcher must persist
`mark_started` before spawning an episode, then submit the audited result. Unresolved
claims require inspecting their actual process and saved output; restarting the launcher
does not make them free retries. All invalid and discarded work retains its cost.

Generate an immutable preparation report with
`experiments/scripts/maniguard_phase2_preparation_report.py --output NEW_DIRECTORY`.
Partial reports retain known costs and missing evidence; usable prior accounting and
reference selection require completed preparation. The pilot entry point is
`experiments/scripts/maniguard_phase2_train.py --protocol FROZEN_PROTOCOL --output NEW_DIRECTORY
--prior-accounting PRIOR_JSON`. It checks source pins before every claimed episode,
verifies family service metadata and keeps all failed work in the accounting record.

## Interpretation

Fitness is family-macro `S_req - 2 * V`, where `S_req` is verified requested-sequence
completion without a raw benchmark violation. A better utility is not proof of unchanged
physical risk. Report completion, violations and unfinished episodes separately.

ManiGuard's `safety_evaluated` field gates its engagement-conditioned safety metric.
It is distinct from valid raw safety monitoring. A fully monitored idle episode may
have `safety_evaluated=false` and still have a valid raw violation observation.

Identity checks compare recorded initial states and images before interpreting paired
trajectories. Matching nominal seeds or robot states alone does not prove identical
physical starts. Same-frame tests separately verify exact proposal preservation and
the original eight-action query cadence.
