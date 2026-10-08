# Experimental notes

These files preserve evidence, interpretation and debugging knowledge from completed
experiments. They are not competing research plans. New work follows the
[adopted direction](../docs/researchDirection.md) and [pilot checklist](../docs/research/pilot.md).

## Push-T

- [safeCEM.md](safeCEM.md): corrected penalty/Safe-CEM results and arena-repair provenance.
- [pushTDataExp.md](pushTDataExp.md): expert HDF5 schema and exploration.
- [LinearProbe&HazardAvoidance.md](LinearProbe%26HazardAvoidance.md): early probe/hazard notes;
  read the later Safe-CEM corrections before citing the original failure explanation.

## Phase 0

- [phase0Report.md](phase0Report.md): the later consolidated recovery-triage report.
- [terminationIsNotIrreversibility.md](terminationIsNotIrreversibility.md): earlier recovery
  findings; prefer the later report and recorded data if numbers or interpretations differ.
- [safetyGymLocomotion.md](safetyGymLocomotion.md): measured environment and tooling facts.

Historical prose reflects the assumptions at the time. Failed finite search does not
prove irreversibility, and zero observed violations does not guarantee safe deployment.
Replay/equivalence claims need their recorded scope; the new Push-T replay gate is pending.

Notebook/script locations are in the [experiment index](../experiments/README.md).
Recorded results and compiled reports remain under [docs/safeDial](../docs/safeDial/) and
[docs/phase0](../docs/phase0/); videos remain under [animations](../animations/README.md).
