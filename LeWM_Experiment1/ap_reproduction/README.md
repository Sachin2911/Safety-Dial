# AP reproduction

In progress. This study reproduces Anchored Planning on the authors' original
long-range cases in TwoRoom, PushT, Reacher and Cube: 128 cases per environment,
each evaluated with AP-CEM and matched final-goal CEM.

The [frozen protocol](protocol.md) defines the cases, engineering gates, budgets
and analysis. [assets.json](assets.json) pins the released inputs and runtime
configuration. Results and videos will be published in the private dataset
[LeWMExp1_AP_Reproduction](https://huggingface.co/datasets/Sachioster/LeWMExp1_AP_Reproduction).

All simulation and model computation runs on the rented host through the
[compute pipeline](../computeConfigs/README.md). Run the stages in order for
each task: `prepare.py`, `encode_cache.py`, `engineering.py`, then `evaluate.py`.
Each entrypoint takes the task name; configuration and output locations come
from the pipeline environment. `audit.py` inspects saved evidence independently.

These cases use recorded goals 100-150 actions ahead. The
[completed baseline](../baseline_reproduction/README.md) used nearby goals and
is a separate experiment. No results are claimed here until execution and the
evidence audit are complete.
