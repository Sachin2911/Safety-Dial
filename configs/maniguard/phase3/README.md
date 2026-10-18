# ManiGuard evolutionary safety dial pilot

`part1.json` is the checked Part 1 protocol. `allocation.json` preserves this allocation's
actual start and absolute deadlines. Reusing these deadlines does not authorize a later
allocation. The native Clutter horizon remains 1,100 actions.

The implementation uses one frozen Clutter policy server and fresh serial simulator
processes. It searches a four-coordinate categorical controller at dial one, physically
re-evaluates elites, and selects from the final population. The same frozen finalist
is compared at zero, one-half and one. Selecting the neutral genome stops the redundant
comparison, as prescribed by `phase3Part1.md`.

CPU verification in the explicit simulator interpreter:

```bash
uv run --no-project --python data/maniguard/envs/sim/bin/python \
  -m pytest experiments/tests/test_maniguard_phase3.py -q
```

The one-episode entry point is `experiments/scripts/maniguard_phase3_episode.py`.
It requires the exact prospective job and, outside engineering, the frozen protocol.
The coordinator is `experiments/scripts/maniguard_phase3_pilot.py`; its actions are
`draft`, `engineering`, `freeze` and `run`. An engineering retry requires an explicit
new `--engineering-attempt` number. Existing attempts and costs remain intact.
An incomplete search population stops without automatic retries or fitness imputation.

The independent `maniguard_phase3_budget.py` supervisor guard must be running before
physical execution. It reads the persisted allocation, tracks owned process groups,
samples GPU memory, and terminates episodes and the policy inside their deadlines.
The coordinator checks stage estimates and reserves each job's timeout plus cleanup.
Restarting a process never resets the allocation clock. A partially started population
is preserved; invoking `run` again refuses to overwrite its plan or repeat its jobs.

`maniguard_phase3_report.py` re-audits each raw attempt once, including failed work,
and emits outcome figures, lineage, costs and matched-video indices. PhysX counters
cover callbacks after app launch, including scene initialization. Physical advances
before the subscription are unknown. Interrupted counters are lower bounds.
Environment inputs are recorded before the native low-level controller's clipping.

Run files, source snapshots, native videos and checkpoints stay outside Git. The
credential-scanning `scripts/maniguard/backup_part1.py` preserves run evidence in a
private Hugging Face dataset, excluding assets, environments and keys. The concise
report and selected audit summaries belong under `docs/research/`.

For a saved completed run, regenerate the descriptive audit in a new directory, then
verify the matched comparison and compose every available validation video block:

```bash
PILOT_RUN=runs/maniguard/phase3-part1/20261018-56578764
PILOT_AUDIT="$PILOT_RUN/audit-reproduction"
uv run --no-project --python data/maniguard/envs/sim/bin/python \
  experiments/scripts/maniguard_phase3_report.py "$PILOT_RUN" "$PILOT_AUDIT"
uv run --no-project --python data/maniguard/envs/sim/bin/python \
  scripts/maniguard/part1_report_tables.py "$PILOT_AUDIT"
uv run --no-project --python data/maniguard/envs/cpu/bin/python \
  scripts/maniguard/part1_matched_audit.py "$PILOT_RUN" "$PILOT_AUDIT"
uv run --no-project --python data/maniguard/envs/cpu/bin/python \
  scripts/maniguard/match_part1_videos.py "$PILOT_RUN" "$PILOT_AUDIT/videos.json" \
  "$PILOT_RUN/comparison-matched-videos-reproduction"
```

These commands read recorded physics; composing videos advances no simulator state.
The matched audit checks this pilot's selected horizon-16 controller, all sealed jobs,
native base/derived sampling seeds, execution order, exact zero identity and initial
observation differences. `export_part1_audit.py` additionally exports the small public
CSV ledger, controller, figures and audit JSON from a complete audit with its retained
shutdown and postprocessing receipts. It never exports raw run directories or weights.
