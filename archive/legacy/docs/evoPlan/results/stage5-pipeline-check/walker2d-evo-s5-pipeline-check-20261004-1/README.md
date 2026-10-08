# Readiness study pipeline validation

The integrated study scheduler passed a bounded frozen-model and real-simulator
check under declaration commit `c38ef5c`. It completed four small searches (two
seeds, k=0/k=1, zero extra penalty), froze eight checkpoint selections, and executed
all 30 planned paired evaluation queries. This validates the runner; it is not a
main-study result, a ROSARL comparison or an additional transfer screen.

## Verified behavior

- Search paused after one completed generation, then resumed through all four arms.
- No evaluation ran before every selected policy and its parameters were frozen.
- Evaluation paused after three completed queries, then resumed without repeating
  them. Invoking completed search and evaluation again used zero extra queries.
- Study identity pins model/controller provenance, banks, search configuration,
  evaluation noise/sample settings and initial parameters. Completed query records
  additionally pin their exact inputs and numeric output hashes.
- The real baseline matched archived actions, qpos, qvel, progress, dense failures
  and readouts bitwise on the four reused development roots. The recorded-tape
  reference matched its archived actions, qpos, qvel, progress and dense failures.
- All measured predictor and simulator counts matched the declaration.

Five focused scheduler/query tests and four existing checkpoint-core tests passed.
They also verify that episode overlap is rejected, an interrupted query preserves
its measured partial physics cost and refuses replay, cached output corruption is
rejected, and evaluation summaries agree with dense simulator trajectories.

## Incremental costs

| Item | Count |
|---|---:|
| Search predictor rows | 4,400 |
| Audit predictor rows | 2,000 |
| Real simulator steps | 4,000 |
| Renders including fingerprint | 424 |
| Encodes | 360 |
| New episode collection / gradient updates | 0 / 0 |
| Wall time including loading | 10.68 s |

All data came from existing inspected development episodes. The fitness, selection
and evaluation roles in this validation were disjoint by source episode. These
reused roles are not an untouched main-study evaluation bank.

An interrupted query records its requested shape before executing. A caught failure
also records measured partial costs. After abrupt process death, an uncommitted
query's cost may be unknown; its journal prevents automatically treating that work
as free and retrying it. A complete atomic query archive can be reused safely.
The CMA core still requires reconciliation of an interrupted generation before
replay, even if individual query records survived.

The integrated scheduling/evaluation components are ready. The executable main
launch and fresh-bank collection still need integration with a reviewed and locked
protocol. Main-study analysis and the ROSARL variance check also remain. Neither
supervisor approval nor archival upload is claimed here.

See [check.json](check.json), [declaration.json](declaration.json) and the pinned
[configuration](config.yaml) for machine-readable evidence.

Full regression verification: 666 tests passed, 49 warnings, 76.55 seconds. Ruff passed.
