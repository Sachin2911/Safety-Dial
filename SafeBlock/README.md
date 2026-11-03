# SafeBlock

Working area for the **Block Plucking main study** of *Safe AI via Evolutionary
Algorithms*. It is the manipulation counterpart of [SafeLeWM](../SafeLeWM/README.md)
and uses the same shared question, adapter family and learning recipe
([Rule42](../SafeLeWM/Rule42.pdf), [FirstExperiment.md](../SafeLeWM/FirstExperiment.md)).

- [Design](BlockPlucking.md): the frozen UNISafe system, the adapter interface, the
  objective, comparisons, panels, budgets and open decisions. Prospective.
- [Qualification](Qualification.md): the first bounded run, written for authorisation.
  **Not authorised and not run** as of 1 November 2026.
- [Pilot](Pilot.md): the TwoRoom adapter recipe on Block Plucking in two hours. Authorised
  and run 1 November 2026, three repeats, 48 fresh starts: no signal. Per-repeat results in
  [PilotResults.md](PilotResults.md), [PilotResults-r1.md](PilotResults-r1.md),
  [PilotResults-r2.md](PilotResults-r2.md); pooled in [PilotPooled.md](PilotPooled.md).
- [Search 2](Search2.md): four fixes to the search after the pilot (eight starts per candidate,
  selection panel, elite re-evaluation, frozen-system candidate scores). Run 1 to 2 November on
  three hosts, stopped early: pooled verdict mixed, untrained draws as good as the search. Results
  in [Search2Results-r0/1/2.md](Search2Results-r1.md) and [Search2Pooled.md](Search2Pooled.md).

Created 1 November 2026 at Sachin's request so that Block Plucking can proceed in
parallel with the LeWM opening study. This folder is part of the parent repository.

## Runtime dependencies

The required UNISafe runtime was moved into this folder on 3 November:

- [Runtime helpers](experiments/helpers/README.md): model loading, source compatibility,
  controller, environment, outcome scoring and renderer checks.
- `configs/unisafe09/`: original settings, requirements and checkpoint provenance.
- `data/unisafe09/`: ignored local weights and compatibility runtime.
- `third_party/unisafe09/UNISafe/`: ignored pinned upstream checkout.
- [Historical receipts](../docs/records/README.md) and
  [Principia](../docs/research/principia.pdf): earlier UNISafe evidence and summaries.

Frozen configurations retain their original bytes. Working entry points translate
their old asset/output prefixes using `experiments/project_paths.py`. Historical
protocol commands may need the same path translation. The earlier local output
deletion remains recorded in [its receipt](../docs/records/progress/receipts/local-deletion-20261103.md).

## Layout

- `BlockPlucking.md`, `Qualification.md`: the active design and first protocol.
- `experiments/{helpers,scripts,tests}/`: new implementation, created when work begins.
- `configs/`: frozen protocol files for each stage.
- `outputs/`: local evidence path; its existing contents were deleted on 3 November.

Datasets, checkpoints, credentials and generated runs stay outside Git. No rental,
installation or experiment is authorised by this folder.
