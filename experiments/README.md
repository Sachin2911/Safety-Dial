# Experiments

The [adopted direction](../docs/researchDirection.md) studies targeted experience for
safer LeWM predictions. Begin new work with the [pilot checklist](../docs/research/pilot.md).
The studies below are preserved at their existing paths so imports, outputs and
reproduction commands remain stable.

## Completed Push-T work

- [PushTDataExploration.ipynb](PushTDataExploration.ipynb): expert-data survey.
- [LinearProbeAndAvoidance.ipynb](LinearProbeAndAvoidance.ipynb): initial probes and hazards.
- [LinearProbeAndAvoidance_4.ipynb](LinearProbeAndAvoidance_4.ipynb): later probes,
  imagination checks, historical penalty sweep and demonstrations.
- [safe_dial_pusht.py](scripts/safe_dial_pusht.py) and
  [validate_arena_fix.py](scripts/validate_arena_fix.py): corrected planning pilot.
- [probes.py](helpers/probes.py), [linProbeHelpers.py](helpers/linProbeHelpers.py),
  [hazardSweep.py](helpers/hazardSweep.py), [safeCEM.py](helpers/safeCEM.py): reusable helpers.
- [Findings](../notes/safeCEM.md), [report/results](../docs/safeDial/),
  [videos and rendering commands](../animations/README.md).

The historical pusher probe is not a block-pose probe. The corrected penalty results
supersede the original infeasible-start comparison. Check the saved variant provenance
before attempting a numerical reproduction of a previous arena constraint.

## Completed Phase 0 work

- [run_triage.py](scripts/run_triage.py): locomotion recovery diagnostics.
- [oracle.py](helpers/oracle.py): finite recovery search, not an impossibility proof.
- [locoEnv.py](helpers/locoEnv.py), [locoCollect.py](helpers/locoCollect.py),
  [locoData.py](helpers/locoData.py), [locoPolicies.py](helpers/locoPolicies.py),
  [locoMetrics.py](helpers/locoMetrics.py): environment, storage and analysis infrastructure.
- [verify_replay.py](scripts/verify_replay.py),
  [verify_env_equivalence.py](scripts/verify_env_equivalence.py): locomotion checks;
  these do not establish Push-T arbitrary-root replay.
- [Findings](../notes/phase0Report.md), [report/data](../docs/phase0/),
  [videos](../animations/README.md).

## New study: implementation pending

The first additions should be verified Push-T branching, whole-block geometry and
block-pose diagnostics. Predictor-side adaptation and acquisition follow the gates
in the pilot checklist. None of those new runners or trained artifacts exists yet.

Keep reusable Python in `helpers/`, command-line entry points in `scripts/`, consumed
Hydra configs in `../configs/`, and generated artifacts under ignored `../data/` paths.
Do not move or overwrite historical experiment files to make room for the new study.
Use `uv run` and launch Jupyter with this directory as its working directory.
