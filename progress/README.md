# Progress

Working todo lists for the current plan in [firstPrinciples.md](../firstPrinciples.md).

1. [Replicate the published Jar baseline](01-replication.md): **closed 23 October 2026.**
   Seed 0 on instance B, then reassigned.
2. [Seeds 1 and 2 on two GPU lanes, instance C](01b-lanes.md): done.
3. [Seed 0 twice on instance D](01c-seed0-rerun.md): done.
4. [After replication: failures, variance, panels, harness](02-after-replication.md): next.

Story and results: [Jar replication and noise floor](../docs/research/research_notes/Jar%20replication%20and%20noise%20floor.md).
Results of record are the four runs on one host (seeds 0, 0, 1, 2 on machine 54514,
104 episodes); instance B's seed 0 is excluded and kept under
[receipts/excluded-instance-B/](receipts/excluded-instance-B/README.md).

Tick items as they are done and add a one-line note with the date and evidence path.

## Evidence

- `receipts/`: install, smoke, run, comparison and preservation receipts from each
  instance (`01-*` protocol and B's install, `01b-*` instance C, `01c-*` instance D),
  the reviewer's notes in `receipts/01-review.md`, and run inputs under `receipts/01-runs/`.
- Dataset `Sachioster/maniguard-jar-replication-20261022` (private), revision `9300cebf`:
  every rollout video, LTL sidecar and results row, task folders named `<n>_<outcome>`,
  one `index.json` per run.

## Rentals

- [rental-20261022.json](rental-20261022.json): four instances for todo 1 (A never
  booted; B Norway, seed 0, excluded; C and D New Zealand machine 54514, the runs of
  record). All destroyed; nothing is renting. Check `vastai show instances` before
  assuming otherwise.
