# Excluded: instance B's seed 0

Instance B (57261620, Norway, one RTX 3090, driver 580.178.04) ran seed 0 on 22 October
2026: 2 safe success, 1 unsafe success, 8 safe unsuccess, 15 unsafe unsuccess over 26
tasks. On 23 October 2026, after the same seed was run twice on the machine that ran
seeds 1 and 2 (7 / 0 / 9 / 10 and 7 / 5 / 7 / 7), the user decided to exclude this run
from the results entirely, so that every result of record comes from one host. The
decision and its effect are written up in
[Jar replication and noise floor](../../../docs/research/research_notes/Jar%20replication%20and%20noise%20floor.md)
and in [01-review.md](../01-review.md).

These files are B's run outputs and receipts, moved here unchanged on 23 October 2026.
B's install recipe, pinned sources, smoke receipts and the paper-protocol config stay
under `progress/receipts/01-*`, because instances C and D replayed them. B's videos,
sidecars and receipts were removed from the dataset
`Sachioster/maniguard-jar-replication-20261022` in revision `44bdf1e7`; revisions before
it still hold them under `jar/seed0/`, `smoke/seed0/` and `receipts/01/`.

- `01-seed-one.md`, `01-results-seed0.md`, `01-compare-seed0.json`: the run and its
  interim comparison.
- `01-runs/seed0/`: `results.jsonl`, `summary.json`, `eval_config.json`,
  `expected_rollouts.json`.
- `01-runs/smoke/`: B's smoke episode outputs.
