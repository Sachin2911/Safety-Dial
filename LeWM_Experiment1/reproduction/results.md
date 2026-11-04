# LeWM frozen-checkpoint reproduction

Audit status: complete

Main comparison uses the paper's CEM iterations and a true 25-action goal offset.
The frozen public checkpoints are evaluated without training or safety adaptations.

| Environment | Paper | Seed 42 | 95% Wilson interval | Seeds 43 / 44 | Within 10 pp |
| --- | ---: | ---: | --- | --- | --- |
| tworoom | 87% | 45/50 (90%) | 78.6% to 95.7% | 37/50 / 45/50 | yes |
| reacher | 86% | 32/50 (64%) | 50.1% to 75.9% | 34/50 / 43/50 | no |
| pusht | 96% | 48/50 (96%) | 86.5% to 98.9% | 44/50 / 44/50 | yes |
| cube | 74% | 36/50 (72%) | 58.3% to 82.5% | 42/50 / 32/50 | yes |

The 10 percentage-point tolerance is an engineering criterion, not an
equivalence test. Intervals are descriptive Bernoulli intervals; repeated
trajectories and a shared checkpoint limit independence. These results concern
nearby-goal competence and do not establish safe or long-horizon completion.

## Pooled main-condition description

- tworoom: 127/150 (84.7%); descriptive Wilson interval 78.0% to 89.6%.
- reacher: 109/150 (72.7%); descriptive Wilson interval 65.0% to 79.2%.
- pusht: 136/150 (90.7%); descriptive Wilson interval 84.9% to 94.4%.
- cube: 110/150 (73.3%); descriptive Wilson interval 65.7% to 79.8%.

Pooling summarizes these panels; it is not an independent training-seed
replication or a formal equivalence test.

## Released shared-config sensitivity

- cube, seed 42, CEM 30: 34/50 (68%).
- reacher, seed 42, CEM 30: 39/50 (78%).
- tworoom, seed 42, CEM 30: 42/50 (84%).

## Evidence

Audited scientific controller actions: 37,500.
Audited completed engineering actions: 700.
Total attempted controller actions including failures: 38,210.
Scientific model work: 27,000 cost calls, 8,100,000 candidate rollouts and 40,500,000 predicted latent steps.
Summed scientific panel time: 4693.4 seconds, including panel setup, planning and control. Downloads, transfer retries and time between panels are additional session work.
Per-panel evidence and model-query counts are in audit.json. Failed and
engineering runs remain separate. See compatibility.md for the API, renderer,
checkpoint-history and exact-goal-index decisions.

Low-level physics counters were instrumented for PushT and Cube; TwoRoom
and Reacher have exact controller-action counts only. For scientific panels:

- pusht: 450 reset/restoration physics frames and 75,000 evaluation physics frames attempted.
- cube: 10,000 reset/restoration physics frames and 250,000 evaluation physics frames attempted.

The first Reacher engineering attempt lost its ten action trace rows when
JSON serialization rejected unused NaN telemetry. Its ten actions are
retained in the durable ledger. Its model-query counts are reconstructed
from the completed CEM solve, separately from measured counters.

Reacher's initial multi-panel job was stopped after its first completed panel
because the next panel stalled during construction. That next panel had
zero actions and retained its original reservation when continued in a
separate process. The first result was preserved and never rerun.
