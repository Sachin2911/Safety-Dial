# Frozen-model resume validation

The new readiness CMA core reproduces uninterrupted searches exactly when resumed
from the completed first generation. Both k=0 and k=1 passed using the frozen
Walker LeWM and residual controller, with four candidates and two generations.
The configuration and source were committed as `dba1117` before execution.

Candidate hashes, nominee parameters and outcomes, checkpoint picks, fitness-only
penalty bounds and query counters match exactly. The k=1 check uses four noise
samples per root; each fitness and selection call uses four roots. Only the zero
extra penalty objective was exercised on the world model. This does not validate
a ROSARL scientific effect or safe locomotion.

All duplicated verification queries are included: 4,400 predictor rows, zero real
simulator steps and zero gradient updates, in 6.73 seconds including model loading.
The inputs are existing inspected development episodes. No final episodes were
collected. See [check.json](check.json) for exact comparisons and
[declaration.json](declaration.json) for source hashes and input provenance.

The validation ran concurrently with the final seed of the bounded development
pilot. Its recorded execution window must be considered when interpreting pilot
throughput. The run remains local; no upload is claimed.

The complete project regression suite passed: 661 tests, 49 warnings, 79.29 seconds. Ruff passed.
