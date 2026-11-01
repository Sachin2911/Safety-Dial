# Evolving a recurrent recovery adapter

29 October 2026. This is the implementation Sachin requested after the offline-adapter
discussion. Work proceeds through an iterative implementation and training pipeline,
without separate small-study approval gates. Each launched version still saves its
inputs before execution, keeps held-out tasks untouched and accounts for all work.
The existing Vast shutdown deadline and $10 session cap remain in force.

## Controller

Frozen pi0.5 supplies its original sixteen-action chunks, with eight actions executed
before querying again. A recurrent adapter operates at every 20 Hz control. It has
35 normalized features, four recurrent hidden units and nine output logits, for
205 evolvable parameters. The outputs specify seven joint-target residuals, a
discrete gripper preference and a continuous residual mixing strength.

The input features use robot joint state, the native command, relative jar/end-effector
pose, jar orientation, the original safety propositions, grasp state, contact and
causal history. Jar pose, propositions, contact and grasp state are declared privileged
simulator inputs. No image-only or latent-only claim is made.

Joint corrections are bounded at 0.08 radians and smoothed across controls. Gripper
preference has a fixed deadband and produces the original -1 close/+1 open convention.
These are initial implementation choices, not validated optimal values or safety
guarantees. The initialized output layer produces exactly native commands. Native
reference episodes run the same observer and feature computations with modification
disabled. The benchmark monitor remains passive and unchanged.

## Training and development evaluation

The first implementation uses diagonal CMA-ES over all controller parameters.
Random parameter search is also implemented using the same representation, initial
distribution and episode panels. A fixed controller is exported as JSON after each
completed generation. A revised run can initialize from that artifact through
`search.initial_controller`; its source and configuration must be preserved separately.

The default objective ranks strict safe completions above every possible secondary
progress difference. Ordered partial progress is closure while supported, a
ten-control grasp after closure, then departure while grasped. A raw violation or
incorrect closure-before-departure sequence removes all partial-progress credit.
Violations break remaining progress ties. This is an auxiliary training signal;
partial progress is never reported as task completion. `completion_only` is available
as a separate, prospectively selected objective.

The development configuration declares four generations of six candidates,
two development tasks per candidate and one policy seed per generation. All candidates
in a generation receive the same task/seed panel; native references use that panel too.
Evaluation order is randomized, with native placement alternating by generation.
Matched policy seeds do not make contact dynamics deterministic.

Selection is the final CMA-ES distribution mean. Subsequent validation uses different
development tasks and three fresh policy seeds, interleaving the selected adapter
with native execution. Validation does not update or select the controller. The planned
maximum is 68 episode starts, subject to interaction and wall-clock budgets. This is
development evidence. A final optimizer comparison, ordinary-learning baseline,
optimizer repeats and held-out evaluation remain separate work.

## Running on the rented host

Use the behavior environment explicitly. Do not sync the root environment into it.
The package must contain the implementation, configuration, frozen panel file and the
existing passive recorder. Install `cma==4.4.4` and the software-check tools in that
environment if needed. Tests, training, preparation and replay run on the rented host.

```bash
PYTHONPATH=experiments/helpers:experiments/scripts:scripts/todo2 /workspace/miniforge3/envs/behavior/bin/python -m pytest -q experiments/tests/test_maniguard_evolution.py
/workspace/miniforge3/envs/behavior/bin/python experiments/scripts/run_maniguard_evolution.py train --config configs/maniguard_evolution/development-v2.json --out /workspace/maniguard-evolution-20261029/run-v2
/workspace/miniforge3/envs/behavior/bin/python experiments/scripts/run_maniguard_evolution.py status /workspace/maniguard-evolution-20261029/run-v2
```

Use the `evaluate` subcommand with `--config`, `--controller` and a fresh `--out` to
evaluate an exported controller against native on the declared validation panel.
The `replay JOB_DIRECTORY` subcommand rechecks saved controller decisions and outcomes
without simulator interaction.

Resume with the original command plus `--resume`. The atomic checkpoint includes
optimizer state, RNG states, candidate population and completed jobs. A reserved job
is recovered from saved outputs, never silently rerun. An incomplete or invalid job
is retained and reported as an error. Repairs use fresh run directories; completed
controller artifacts can supply the next version's initialization.

The run writes source snapshots, a manifest, evaluator instrumentation diff, candidate
parameters, decisions, videos, original monitor logs and separate native/modified/executed
action boundaries. The generated evaluator is separate from the public checkout.
Accounting includes failed starts, synthetic sampler-reset queries, warmups, resets,
model calls, instrumented physics and elapsed time. Initial bootstrap physics remains
unknown. Saved controller decisions are replayed against each episode recording.

Stop by sending SIGTERM to the trainer. It terminates its active simulator process
group and saves state. The rental's independent guards remain armed.

## Engineering record

Version 1 failed at simulator startup because the SSH launcher omitted the existing
simulator environment, including its previously accepted non-interactive EULA setting.
It returned zero environment steps and completed one synthetic model query; bootstrap
physics remains unknown. Its source snapshot, reservation and logs are preserved.
Version 2 carries over the public setup's cache, renderer and consent environment
and exports the CMA-ES mean in phenotype coordinates. Its accounting reports the
predecessor alongside new work. No controller settings, panels or seeds were retuned
from episode outcomes during that repair.

## Supervision and preservation

Sachin subsequently requested supervision through completion, automatic repair of
technical failures, verified saving and uploading of all artifacts, then destruction
of this Vast instance. The existing deadline and session cap remain in force.

Completed episodes and frozen source are preserved in immutable interim snapshots
under the private dataset's `21_recurrent_adapter_development/20261029T204801Z/`
prefix. Every indexed file is downloaded at its commit revision and checked by SHA-256.
Open episode files are excluded from interim snapshots. The final snapshot requires
all declared training and validation episodes and retains the failed predecessor.

The final audit checks the schedule, source snapshots, loadable optimizer checkpoint,
selected controller, generation scores and every saved episode's action/outcome replay.
It uses saved evidence without simulator interactions or model queries. Instance
destruction follows final local and pinned private-upload verification.

## Review findings and separate next version

Review during the frozen version 2 run found an auxiliary-progress ordering bug.
Carry-stage credit could be granted when departure occurred before the required
ten-control grasp, followed by continued grasp away from the support. The strict
safe-completion endpoint is unaffected. The original sources and outcomes are retained;
the corrected function is separately versioned as `ordered_progress_v3`.

The initial candidates also applied arm corrections almost continuously, with observed
joint residual RMS around 0.011 to 0.025 radians. Many lost engagement. This is
consistent with disruptive exploration, but the simulator's nondeterminism prevents
attributing the outcome difference to corrections alone. Safe inactivity also tied
with safe target contact before closure, and the native controller was absent from
the sampled candidate population.

The unlaunched version 3 draft adds smaller initial arm and recurrent perturbations,
a native anchor for both implemented optimizers, and safe-target-contact tie credit
below every strict completion. It also corrects the auxiliary ordering bug. A separate
copy passed 33 software tests on the rented host, including the original pipeline
checks. These changes have received no robot rollouts and establish no performance
gain. Their source, findings and test receipt are retained with the run under
`review-v3/` and in the [review receipt](../../progress/receipts/20261029-maniguard-evolution-code-review.json).

## Completed development run

Version 2 completed all 68 planned episodes on 30 October 2026, with four optimizer
updates and no invalid version 2 episodes. Across the 48 sampled-candidate training
episodes, strict safe completion was 0/48, benchmark safe completion 1/48, raw
violations 7/48, unfinished episodes 47/48 and engagement 19/48. The eight native
training references had 5/8 strict and benchmark safe completions, 2/8 raw violations,
3/8 unfinished episodes and 8/8 engagement. The sampled-candidate pool contains
different controllers and is not an evaluation of the selected final controller.

On the fresh development validation panel, the selected final CMA-ES mean achieved
0/6 strict and benchmark safe completions, with 2/6 raw violations, 6/6 unfinished
episodes and 4/6 engagement. Native execution achieved 1/6 strict and benchmark safe
completions, with 3/6 raw violations, 4/6 unfinished episodes and 6/6 engagement.
Requested-sequence completion, before applying the safety requirement, was 0/6 for
the selected adapter and 2/6 for native. Counted and raw violation episode counts
coincided in both validation arms.

This run establishes the implementation and preservation workflow. It does not
establish a completion lever or an evolutionary advantage. The small development
panel, one optimizer seed, privileged inputs and absent matched optimizer controls
remain limitations. No held-out tasks were used. The separately tested version 3
draft has not been evaluated on robot episodes.

The final archive contains 1,429 files, all downloaded and SHA-256 verified at private
revision `981d947a2ad3e7682ff87e686a91fedc5cd5a75c`. The complete local copy also
matches the remote inventory. After this verification, Vast instance 58267560 was
destroyed at about 02:34 SAST on 30 October; two provider queries confirmed its
absence. The local rental guard and mirror are retired. The
[final preservation receipt](../../progress/receipts/20261029-maniguard-evolution-final-preservation.json)
and [lifecycle closeout](../../progress/receipts/20261030-maniguard-evolution-closeout.json)
retain the exact artifact revisions, accounting and destruction evidence.

Sachin subsequently requested the [next experiment](maniguard-evolution-v3-experiment.md)
and [future goal prompt](maniguard-evolution-v3-goal.md), plus a fresh ten-hour Vast
rental. That allocation prepares sources and dependencies only; the experiment has
not been started. Its separate protocol adds conservative evolution, measured
native/elite/mean selection and fresh development validation, with a native fallback.
