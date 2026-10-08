# One teacher labelled aggregation round did not improve development safety

On 4 October 2026, adding teacher-labelled examples from student-visited training
states produced 6/24 development health violations. Its matched control, which
received teacher labels from teacher-visited states, also produced 6/24. The
previous mixed-demonstration controller had 4/24. The new student-state arm made
less mean forward progress than the matched control, although the paired interval
includes zero.

This fixed single-round comparison does not support replacing the previous
checkpoint. **Gate 0 was not rerun.** No transfer check or evolutionary search ran.

## Protocol and scope

The protocol and implementation were committed as 2a15aae before collection or
outcome inspection. Architecture, frozen encoder, feature normalisation, ten-action
output interface, optimiser, three fitting seeds, epoch count and validation
criterion stayed fixed. The source controller and caches are pinned to coverage
archive revision aab43ddcba2c1209453121a705ecf10c21a560a2.

We selected 64 PPO and 64 PPO-Lagrangian episodes from the existing selected
fit pools, with one seeded start per episode. Eligibility used episode length
alone. Teacher-state and student-state arms share these 128 roots. Teacher labels
come from each source episode's original actor with zero action noise.

Fit episodes are in setA.h5. Development episodes are in the separately collected
roots.h5; episode numbers are local to each file. Neither validation episodes nor
development trajectories supplied new training labels. Collection did not filter
states by health, failure, or teacher success.

The fixed student generated one 100-step rollout per root. The control used the
source teacher's 100-step rollout from the same root. At every ten-step boundary,
both arms replayed their exact prefix from the root and queried the teacher for
the next ten sequential actions. This preserved solver warm starts. All 2,560
label branches verified their prefix pose and velocity against the supplying
trajectory, and control targets matched their teacher rollout actions exactly.

Both arms were charged 83,200 simulator steps: 12,800 rollout steps and 70,400
prefix-plus-label steps. Replaying prefixes on the control arm also verified
identical solver-state handling. These equal budgets are for this diagnostic
implementation, not a claim that this is the cheapest way to obtain teacher labels.

## Training and selection

Each arm collected 1,280 unique action-block labels, 640 per family. An epoch had
45,986 examples: 34,490 shared original examples and 11,496 draws from the new
labels, a 25.0% augmentation fraction after integer rounding. Repetition was nearly
uniform, with the same original indices and new root/block indices in both arms.

Each arm trained three fresh initialisations for 100 epochs, batch size 1,024,
with AdamW at learning rate 0.001 and weight decay 0.0001. This was 13,500 optimiser
updates per arm. The 384-256-128-60 ReLU network has 139,196 parameters.

The unchanged whole-episode validation pool assigned half its weight to each
family. Recorded-action validation MSE alone selected checkpoints; development
outcomes played no part. Neither checkpoint was refit on validation examples.

| Arm | Selected seed | Selected epoch | Family balanced validation MSE |
|---|---:|---:|---:|
| Teacher-state control | 20261004 | 23 | 0.09459 |
| Student-state augmentation | 20261005 | 20 | 0.09657 |

The earlier mixed checkpoint had validation MSE 0.08781. Both new arms selected
earlier checkpoints under the common criterion. The total training searches were
matched; selected checkpoints need not have the same training age. The inherited
feature scaler remains unchanged and was originally fitted using validation
features as well as fit features, as documented in the preceding reports.

## Development outcomes

| Controller | Health violations | PPO starts | PPO-Lagrangian starts | Mean progress in metres | Recorded progress retained |
|---|---:|---:|---:|---:|---:|
| Previous mixed controller, archived reference | 4/24 | 2/8 | 2/16 | 2.093 | 99.2% |
| Teacher-state control | 6/24 | 2/8 | 4/16 | 2.100 | 99.6% |
| Student-state augmentation | 6/24 | 3/8 | 3/16 | 1.987 | 94.2% |

Recorded action replay had no violations and mean progress 2.110 m.
The previous controller and recorded replay are reused archival outcomes, not
new evaluations. The matched causal comparison here is student-state augmentation
against teacher-state control.

Relative to the teacher-state control, 3 starts became safe and 3
became unsafe. The paired violation-rate difference was 0.0
percentage points, with a source-episode bootstrap interval of
[-20.8, 20.8]. The progress difference was
-0.114 m, with interval [-0.230, 0.003] m.
Both intervals include zero. These are repeatedly inspected development states,
not an untouched confirmation set.

Both new controllers were imagined safe on all 24 starts despite six real
violations each. Probes on real encoded frames marked 9/24 control segments and
7/24 treatment segments unsafe. Those model and readout errors remain separate
from the imitation comparison.

## Interpretation and next decision

The new collection did reach different states: the frozen student violated on
21/128 training rollouts, including 17 PPO and 4 PPO-Lagrangian starts, whereas
the teacher rollouts had no violations. All planned labels were retained.

The result is limited to one frozen collection policy, 128 starts, a 25% replay
mixture, this network and the unchanged offline selection rule. New labels were
repeated roughly nine times per epoch, while replacing 11,496 distinct original
examples. Thus the comparison with the earlier controller also changes diversity
and supervision; it cannot identify which of those changes caused its worse count.
The paired arms control those choices while testing the visited-state distribution.

Do not adopt either new checkpoint or proceed to Gate 0 on the strength of this
run. Keep the earlier mixed controller as the current reference. Before collecting
more data or adding aggregation rounds, a useful next diagnostic would separate
latent-state information limits from conflicting targets across source teachers.
That diagnostic is a proposal, not work performed here. This result does not
establish that all teacher-guided aggregation methods fail.

## Verification and resources

All 626 tests passed with 49 existing warnings and no skips or failures. Ruff
passed. Five new tests cover whole-episode role isolation, deterministic sampling,
matched augmentation budgets, exact solver-prefix labels, protocol invariants,
and current/previous latent feature alignment. Selected weights, hashes, masks,
counts and saved trajectories were audited independently after the run.

The scientific run took 62.19 seconds. It used 171,200 new
simulator steps: 166,400 for collection and 4,800 for two development evaluations.
It also used 480 imagined predictor rows, 27,000 optimiser updates, 3,560 renders
including 64 fingerprint frames, and 3,496 encoded frames, excluding black batch
padding. Test execution costs are separate. Existing source-data and model-training
costs remain recorded as upstream provenance.

[diagnostic.json](diagnostic.json) records summaries and intervals.
[development_rows.json](development_rows.json) contains every paired result.
[collection_plan.json](collection_plan.json), [fit.json](fit.json), and
[training_history.json](training_history.json) record sampling and selection.
[manifest.json](manifest.json) pins inputs, teacher weights, source code and costs.
Weights, labels and dense trajectories are in the ignored local run bundle
runs/walker2d-evo-s2-aggregation-20261004-1/. An upload receipt records the private archive when complete.

Reproduce with a new run ID:

    uv run python experiments/scripts/evo_aggregation_diagnostic.py --run-id NEW_RUN_ID
    uv run python experiments/scripts/evo_aggregation_report.py --run-id NEW_RUN_ID
