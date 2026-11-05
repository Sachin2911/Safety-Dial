# AP reproduction

Completed on 5 November 2026. AP-CEM succeeded in **306/512 cases (59.8%)**,
compared with **47/512 (9.2%)** for matched final-goal CEM, a paired gain of
50.6 percentage points. All 1,024 scientific cases passed the independent evidence
audit, with no missing or repeated cases.

Each row contains **128 original queries per arm**. Counts are successful cases:

| Environment | AP-CEM | Final-goal CEM | Paper AP-CEM |
| --- | ---: | ---: | ---: |
| TwoRoom | 58/128 | 2/128 | 60/128 |
| PushT | 88/128 | 4/128 | 89/128 |
| Reacher | 100/128 | 32/128 | 97/128 |
| OGBench Cube | 60/128 | 9/128 | 60/128 |

All four AP rates meet the declared practical tolerance of no more than
10 percentage points below publication. Agreement on this fixed panel does not
establish statistical equivalence or independent training-seed replication.

Read the [private dataset](https://huggingface.co/datasets/Sachioster/LeWMExp1_AP_Reproduction),
its [environment galleries](https://huggingface.co/datasets/Sachioster/LeWMExp1_AP_Reproduction/blob/main/reports/ap-reproduction-20261105/README.md),
or the local [Odyssey PDF](../../theOdessy/reproduction.pdf).

Both arms use the same starting state, supplied goal, optimizer seed, frozen
weights, CEM settings and action budget. AP selects intermediate targets from the
authors' memory; the control plans toward the final goal. The supplied goals are
**100-150 recorded actions ahead**. Interpretation is limited to these original
fixed queries. Full canonical-task completion and safety require separate tests.
The [nearby-goal baseline](../baseline_reproduction/README.md) remains a separate
historical experiment, so its success rates are not a matched AP comparison.

Eight ghost comparisons show the first AP success and first AP failure by original
query order in each completed environment. Each pairs AP with its control from the
same query. The videos blend saved observations 50/50 and show recorded paths and
executed actions. Labels distinguish the action timeline, observation time and
stopping point; frames are held after an arm stops. The final saved image can
precede the scored final state. No simulation or model evaluation was rerun to
create this media.

The evidence is available separately from the presentation:

- [Scientific audit](evidence/final-audit.json): outcomes, uncertainty, paired
  comparisons, initial successes and engagement for all 1,024 cases.
- [Complete accounting](evidence/final-accounting.json): 183,765 scientific actions,
  engineering and cache work, and preserved failed-run history.
- [Source and gate verification](evidence/source-gate-verification.json), including
  [official source hashes](evidence/official-source-verification.json): pinned
  code, assets, engineering gates and verified backups for all 35 runs.
- [Paired-media provenance](evidence/paired-media-provenance.json): identical
  starting conditions and seeds, source hashes and frame indices for all 512 pairs.

The [frozen protocol](protocol.md) and [asset pins](assets.json) document the method.
The [recording correction](evidence/recording-correction-validation.json) preserved
14 completed PushT cases and continued only untouched query indices; its interrupted
batch remains in the accounting. This completed protocol authorizes no further runs.

AP supplies a matched long-range reference for future adapters. Testing safe
completion still needs explicit hazards, engagement measures and comparisons with
equal budgets on held-out cases. These results make no claim about evolutionary
optimizer superiority.
