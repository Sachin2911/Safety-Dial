# Part 4: counterexample acquisition

The controlling plan is [phase3Part4.md](../../../phase3Part4.md). This directory is a
new namespace. Earlier protocols and raw evidence remain unchanged.

The [bounded premise check](../../../docs/research/reports/ManiGuard%20Part4%20counterexample%20premise.md)
failed. Physical scene engineering, shared initial episodes, acquisition, confirmation,
repair and fresh audit remain unstarted under that scientific gate. Acquisition and
matched-update CPU helpers exist for future use; they are not physical validation.

`offline-protocol.json` freezes the causal event target, historical task groups,
bounded direct-predictor experiment and premise criteria before reading new Part 4
event labels. `allocation.json` uses the current provider's original start time.
It does not create another rental. No comparative acquisition is authorized by a
passing unit test: it requires measured predictor, scene/reset and timing receipts.

Use an explicit uv environment, leaving the base and historical environments alone:

```bash
uv venv --python 3.12 data/maniguard/envs/part4-cpu
uv pip install --python data/maniguard/envs/part4-cpu/bin/python \
  -r configs/maniguard/part4/requirements-cpu.txt
uv run --no-project --python data/maniguard/envs/part4-cpu/bin/python python \
  -m pytest experiments/tests/test_maniguard_part4.py -q
```

The offline entry point is `experiments/scripts/maniguard_part4_offline.py`. It requires
the verified donor restore and pinned benchmark metadata in its run namespace.
`scripts/maniguard/part4_restore.py --run NEW_DIRECTORY` restores the exact inputs and
native proposition metadata from the frozen private revision, verifying archive and
member hashes. It refuses to overwrite a restored run. Pass the same `--run` to the
offline entry point for an explicitly requested reproduction, rather than rerunning
the completed experiment in place.
`--resume-baseline` resumes scoring from identical saved weights after an engineering
failure; it never repeats training or changes a finished audit. The initial reload
failure and all charged time remain in the accounting.

`maniguard_part4_acquisition.py` implements R/U/E/D parent/variation/archive bookkeeping,
task balance, charged invalid slots, deterministic ties and confirmation selection.
`maniguard_part4_gate.py` refuses physical acquisition unless the predictor, scene/reset,
capacity and shared-initial receipts pass. `repair_predictor` keeps preprocessing fixed,
starts each arm from the same trained baseline, matches updates and rejects calibration,
development, confirmation, audit and other-arm donors. These conditional stages were
not invoked for empirical results.

The maximum physical design has eight shared initial, 48 acquisition, eight
confirmation and sixteen common audit episodes. Invalid attempts and failed launches
consume slots. Three separately charged engineering attempts allow at most 32 actions
each. A failed premise ends the study with its diagnostic; unexecuted stages have no
invented outcomes or replacement deliverables.
