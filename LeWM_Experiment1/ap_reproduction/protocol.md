# AP checkpoint reproduction

Frozen before data preparation or model/simulator execution on 5 November 2026.
Sachin authorized this new reproduction, its GPU rental, private Hugging Face
dataset and a second page in the Odyssey overview. Earlier AP experiments remain
completed; none of their unused allowances transfers here.

## Question and comparison

Does Anchored Planning recover the published long-range recorded-goal results
with frozen LeWM checkpoints? Compare AP-CEM (`gaussian_observed`) with the
authors' matched final-goal CEM (`gaussian_final`) on all four environments.
Both use the same model and CEM; the scoring target changes from the final goal
to a retrieved observation five actions ahead. This is the paper's clean,
standard-start comparison. Perturbation conditions, AP-rank, learned targets,
ablations and training are excluded. Published native LeWM scores may be shown
as external references, never as runs performed here. The earlier nearby-goal
reproduction uses different cases and cannot serve as the paired control.

Use the [official source](https://github.com/daybraeklaxry/Aim-Short-to-Reach-Far)
at `984327424ca7a2e52c75a8279197750d9d0b7c8e`, and its native evaluator.
`protocols/main.json` SHA-256 is
`3243f4ec7c55d349a3f7a73b093664ca7ae695d369f6f718824b54ea47b894b6`;
`protocols/scalers.json` SHA-256 is
`623d70c2b5e1b4f61ea5f647e0d2c639daa3fbe77f229547e1c52ae6c8e3b02a`.
The only protocol changes are conditions `[clean]` and arms
`[gaussian_final, gaussian_observed]`, in that order. Preserve original and
derived protocols in each evaluation run.

## Fixed cases and method

| Environment | Original queries | Recorded goal offset | Policy-action cap per case | Memory episodes | Optimizer seed base |
| --- | ---: | ---: | ---: | ---: | ---: |
| TwoRoom | 128 | 100 | 200 | 8,000 | 28091351 |
| PushT | 128 | 140 | 280 | 16,688 | 26091331 |
| Reacher | 128 | 150 | 300 | 8,000 | 27091351 |
| Cube | 128 | 150 | 150 | 8,000 | 26091351 |

Keep every original query, row index, environment seed, goal and memory episode.
Use optimizer seed base plus 1,000 times original query ordinal. Deterministic
sharding may divide original ordinals without renumbering or changing seeds.
The two arms share each case. Do not select cases or settings from outcomes.
Count initially successful cases and disclose them separately; do not replace them.

CEM uses 300 candidates, 30 elites, 30 iterations, zero initial mean, initial
normalized standard deviation 1/3, standard-deviation floor 1e-5 and the native
action clipping. Compare initial and final means using the native implementation.
Plan five actions, execute at most five before replanning, and stop on native
success after each primitive action. Retrieval uses the native shrinking temporal
horizon, bounded below by five. Preserve the pinned full-data action scaler,
including its original finite-action treatment. Do not refit on selected memory.

Public checkpoint and dataset revisions and compressed hashes are in
[assets.json](assets.json). They are the verified released assets used for our
baseline; the AP authors did not record their historical HF download revisions.
Strict model loading must consume every expected checkpoint tensor. Preserve
native retrieval encoding: original full row order, batch 128, bf16 autocast and
float16 latent storage. Online retrieval uses the same bf16 encoder; native CEM
scoring remains fp32 without autocast.

Use the existing image digest
`sha256:7a63a5d1d11c429a82b3bb0cfef6d1e48e61ee8c28680ef0e5dd83950428bace`,
OSMesa/llvmpipe rendering and the unmodified AP checkout. Reacher uses an isolated
MuJoCo 3.10.0 / DM-Control 1.0.43 overlay; other tasks use the image's 3.12.0 /
1.0.44 versions. Record package versions, overlay wheels and source hashes.
Archive and video utilities may be installed without changing scientific packages.

## Gates, accounting and limits

All runtime computation takes place on the rented host. Engineering uses the
first two sorted eligible memory episodes per task, outside the query episodes,
with ten recorded actions each. Compare native and passively recorded transitions,
success labels and existing renders exactly. This plans 16 resets and 160 actions;
the shared ceiling is 32 resets and 800 actions including failed attempts.
At most four five-action CEM optimization calls per task may check numerical
finiteness and throughput on these memory cases, without executing their actions:
496 predictor batches and 144,032 predicted blocks total. These are engineering
checks, never extra scientific cases or performance gates. No simulator version
or tracing change may bypass a failed parity gate.

Scientific work is exactly 1,024 attempted arm/case combinations and at most
238,080 policy actions. There are at most 47,616 decisions, 1,476,096 predictor
batches and 428,639,232 predicted five-action blocks. Hard conservative guards
are 1.5 million batches and 450 million blocks. Reserve each identity before its
first reset; never repeat a completed scientific case. Preserve failed attempts,
partial evidence and reasons for missing cases. Report an incomplete reproduction
if infrastructure prevents completion within these limits; do not replace cases.

Full cache construction encodes 7,277,545 frames. The separate 16 million-frame
ceiling includes failed or partial cache attempts and permits at most a bounded
retry identified explicitly. Record actual encoder images/calls separately from
policy prediction batches, candidates, retrievals, wall time, controller actions,
reset-internal controller calls and simulator physics steps. Reservations are
upper bounds, never reported as measured interactions. No optimizer training.

The initial rental has one RTX 4090, a six-hour work limit, maximum $1/hour,
200 GB additional input and 5 GB output allowances, and a $10 estimated budget.
The deadline stops work, not billing. Shutdown requires Sachin's direction.
Process public datasets sequentially to fit the 200 GB disk. Verified public
input caches may be retired after their task's evidence and derived cache have
verified backups; retain all unique scientific outputs. No second rental or
budget extension is implicit in an incomplete run.

## Evidence, analysis and presentation

Publish immutable per-case native traces and passive physical observations,
provenance, full derived retrieval caches, failures and accounting through
`run_io` to private `Sachioster/LeWMExp1_AP_Reproduction`. Verify fresh downloads
against pinned manifests. Audit query coverage, paired starts, source-memory
membership, temporal boundaries, seeds, native stopping and physical success.
Success, unfinished cases, initial successes and engagement retain denominators.
These environments have no added safety monitor, so report no safety rate.

Report each arm's successes/128 with Wilson 95% intervals and the paired AP-CEM
minus final-goal difference. Use 10,000 paired bootstrap samples with seed
20260913; the aggregate weights tasks equally and resamples within each task.
Published AP-CEM counts are TwoRoom 60, PushT 89, Reacher 97 and Cube 60 of 128;
matched final-goal CEM counts are 2, 3, 33 and 9. A score no more than ten
percentage points below its published reference meets a descriptive practical
tolerance. This does not establish equivalence. Do not rerun poor outcomes.

Capture only renders already requested by the evaluator. Videos show planner
observations at five-action intervals, with the fixed goal alongside and visible
action indices. The last image may precede the stopping action; label this clearly.
No replay or additional simulator work is authorized for media. Select the first
AP-CEM success and first failure per environment by original query order. If an
outcome is absent, report that instead of generating another case. Organize the
dataset by environment with inline players, settings and results. Add one simple,
results-first AP page to `theOdessy/reproduction.tex` and rebuild its PDF.

The interpretation is long-range recorded-goal completion with a supplied temporal
offset. It is not unrestricted canonical benchmark completion. Query and memory
episodes are disjoint; related PushT scenes can still occur across that split.
This reproduction establishes a planning reference for subsequent testing, not
an evolutionary advantage or learned safety.
