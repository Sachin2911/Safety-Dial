# Implementation plan: evolution cheats imagination

**Status: plan, 3 October 2026.** This turns the experiment in
[paperIdea.md](../paperIdea.md) into stages, code, gates and dates. Nothing in it has run
yet. Scope is Walker2d only, using the assets of the completed
[LeWM experience study](../mainPlan/README.md); no new model training. Committed results
go under `docs/evoPlan/results/<stage>/`; run outputs stay in ignored `runs/` and small
run bundles go to the private Hugging Face model repository, as before.

The question each stage serves: when evolution searches inside the Walker LeWM, how far
does the imagined safety of the policies it selects drift from their real safety, and do
noise plus ROSARL's penalty close that gap?

## Stages at a glance

| Stage | What | Where | Gate or output | Target date |
|---|---|---|---|---|
| 0 | Fetch pinned assets; reproduce saved S4 numbers | Laptop | Assets load correctly | 5 Oct |
| 1 | Write the harness and its tests | Laptop | Equivalence tests pass | 9 Oct |
| 2 | Behaviour-cloned starting policy | Laptop | **Gate 0:** the policy walks | 10 Oct |
| 3 | Throughput benchmark and noise calibration | One 5090 | Run sizes fixed | 11 Oct |
| 4 | Transfer check | One 5090 | **Gate 1:** imagined and real rankings agree | 17 Oct |
| 5 | The cheating curve | One 5090 | **Gate 2:** measurable cheating | 24 Oct |
| 6 | The fix grid | One 5090 | Main result | 5 Nov |
| 7 | Figures, tables, freeze | Laptop | Results frozen | 8 Nov |

The laptop is the local RTX 4070 Laptop GPU (8 GB), enough for model inference, EGL
rendering and small-batch tests. Rent a 5090 only when a stage is ready to run, and shut
it down between stages.

## Fixed inputs

All pinned by revision; never load a moving branch.

| Input | Location | Revision |
|---|---|---|
| Walker LeWM (214,780 updates) | `Sachioster/safetydial-walker2d`, `lewm-a/walker2d-lewm-a-recovery-20260927-1` | `9e05478658484d0f26ea740fead9e0ab7625367e` |
| Physical probes (height, pitch, speed; MLP) | `Sachioster/safetydial-walker2d`, `probes/walker2d-probes-recovery-20260927-1` | `cb1560d7b02b4fbe6585b3c0c325a78d56b337e6` |
| State-only data (`roots.h5`, `probe.h5`, `setA.h5`, `splits.json`) | `Sachioster/safetydial-walker2d-data`, `data/walker2d-data-20260926-3` | `ddd4d51648d2a382725efd20d04cedebedbf8fd2` |
| S4 root banks (development, test, acquisition) | `Sachioster/safetydial-walker2d-data`, `banks/walker2d-s4-recovery-20260927-1` | `09ac490696cecb951e83644db48e646878905f87` |
| PPO and PPO-Lagrangian actor ladder (S0, 42 actors) | `Sachioster/safetydial-walker2d`, `policies/walker2d-policies-complete-20260926-1` | `b42e3f3e7dd80bc4d95cc2ef43c753cee96b9b55` |

Load with `HFStore().download_run(key, path, revision=...)` into `data/hf/`.

## Definitions

**Time.** One environment step is 0.008 s. A block is 10 steps (0.08 s), the model's step.
A segment is 10 blocks (0.8 s), the validated horizon. Each start state carries three
history frames (t − 20, t − 10, t) and the two action blocks between them, as in
`WalkerRoot`.

**Policy.** `LinearLatentPolicy`: features `[z_t, z_t − z_(t−1)]` (384 numbers) mapped
linearly to Walker's six motor commands, clipped to [−1, 1] and held for the 10 steps of
the block. That is 384 × 6 + 6 = 2,310 parameters. The difference term supplies velocity,
which a single latent may not carry.

**Reward.** Forward progress over a segment. Imagined: the sum over blocks of the probe's
speed estimate × 0.08 s. Real: the change in the torso's x position. No healthy bonus, so
reward does not encode safety.

**Safety.** The health rule from `walkerRules.py` (torso height inside (0.8, 2.0) m and
pitch inside (−1, 1) rad). Three readings of every segment:

| Reading | How | Separates |
|---|---|---|
| Imagined | Probe on imagined latents, block ends | What evolution sees |
| Real readout | Probe on real encoded frames, block ends | Probe error |
| Dense truth | Simulator state, every environment step | Ground truth |

Imagined minus dense truth is the gap the paper is about; the real readout splits it into
model-dynamics error and probe error, as in the earlier E1 decomposition.

**Start-state roles.** Reuse the S4 roles, which separate whole source episodes:

| Role | Source | Use |
|---|---|---|
| Fitness | S4 acquisition roots (96 reserved) | Scoring candidates during evolution |
| Tuning | S4 development roots | Noise calibration, step sizes, penalty scale |
| Evaluation | S4 test roots (64 source episodes), plus 3 more per test episode drawn once with a fixed seed: 256 | Every reported number |

Intervals cluster by source episode.

**Noise.** Each imagined step becomes `z_(t+1) = f(history, actions) + k × σ ⊙ ε`, with
`ε` standard normal and `σ` the per-dimension standard deviation of one-step
teacher-forced residuals (predicted minus real encoded latent) on tuning episodes.
`k ∈ {0, 0.5, 1, 2, 4}`. With `k > 0`, each candidate gets `n = 4` noisy rollouts per start
state.

**Ranking rules.** All three score the same imagined segments.

| Rule | Per-candidate score |
|---|---|
| Deb (feasibility first) | Lexicographic: lower violation rate p first, then higher mean return |
| Fixed penalty | Mean over segments of (return − λ × 1[violation]); λ ∈ {0.1, 1, 10} × the BC policy's median safe-segment return, so sensitivity is visible |
| ROSARL | A segment ends at its first imagined violation and adds `R_unsafe = V_MIN − V_MAX`, where V_MIN and V_MAX are the running minimum and maximum segment returns seen so far; score is the mean over segments and noise samples |

The ROSARL row is our adaptation of its value-based penalty to evolution, using observed
segment returns in place of a value function. It must be checked against the RLC 2026
paper and with the supervisor before stage 6.

**Fitness aggregation.** Each generation scores every candidate on the same K fitness
roots, resampled each generation from the 96 (common random numbers within a generation),
times `n` noise samples.

## Code to write

New files only; existing experiment files stay untouched.

| File | Contents |
|---|---|
| `experiments/helpers/evoPolicy.py` | `LinearLatentPolicy`: features, parameter vector to weights, batched action for a whole population (einsum over population, roots and noise samples), zero-order hold to 60-d blocks with the model's action scaler |
| `experiments/helpers/evoImagine.py` | Closed-loop imagination with `LeWM.predict` and `action_encoder` over a three-step history window; optional noise; probe readout per block; noise calibration from residuals |
| `experiments/helpers/evoReal.py` | Closed-loop real executor: `restore` the snapshot, render and encode, act, run 10 steps, repeat; dense truth plus real readout; batched across start states in one process, with encoding batched on the GPU |
| `experiments/helpers/evoRanking.py` | Deb, fixed-penalty and ROSARL scoring; conversion to the ranked fitness values `pycma` minimises |
| `experiments/helpers/evoRoots.py` | Build fitness, tuning and evaluation banks from the S4 roots and `roots.h5`; cache encoded history latents to `npz` |
| `experiments/helpers/evoRun.py` | CMA-ES loop (`pycma` 4.4.4, already installed), checkpoints, per-generation logs, accounting of imagined and real steps |
| `experiments/scripts/evo_setup_check.py` | Stage 0 |
| `experiments/scripts/evo_bc_init.py` | Stage 2 |
| `experiments/scripts/evo_bench.py`, `evo_calibrate_noise.py` | Stage 3 |
| `experiments/scripts/evo_transfer.py` | Stage 4 |
| `experiments/scripts/evo_best_of_n.py`, `evo_cmaes.py` | Stages 5 and 6 |
| `experiments/scripts/evo_report.py` | Stage 7 figures and tables |
| `configs/evo/*.yaml` | One Hydra config per stage; every run records its config, seeds and input revisions |
| `experiments/tests/test_evo_*.py` | Tests listed under stage 1 |

## Stage 0: assets (laptop, by 5 October)

1. Download the pinned model, probes, data and S4 banks.
2. Load the model and probes, render a root's history frames, encode, and run
   `WalkerImaginer.rollout` on a few saved S4 test tapes.
3. **Check:** reproduce the saved imagined health clearances in
   `docs/mainPlan/results/s4/walker2d-s4-recovery-20260927-1/no_update_evaluation_rows.json`
   for those rows to floating-point tolerance. This proves the assets and preprocessing
   are the ones the earlier study used.

## Stage 1: harness (laptop, by 9 October)

Write the modules above with these tests, all CPU or small-GPU and fast:

- **Equivalence:** closed-loop imagination fed a fixed action tape, with `k = 0`,
  reproduces `WalkerImaginer.rollout` exactly. This catches errors in the action-window
  encoding.
- **Real executor:** with a fixed action tape it reproduces `execute_branch` bitwise, and
  repeated runs from one snapshot are identical.
- **Policy:** parameter vector round trip; population batching matches a loop over
  individual policies; actions are clipped and held for exactly 10 steps.
- **Ranking:** Deb puts every lower-violation candidate first; the fixed penalty is
  monotone in λ; ROSARL terminates at the first violation and updates V_MIN and V_MAX
  correctly; ties are broken deterministically.
- **Accounting:** imagined steps, real steps, renders and encodes are counted per run.
- Ruff clean; the full suite still passes (see `AGENTS.md` for the local pytest command).

## Stage 2: starting policy and Gate 0 (laptop, by 10 October)

1. Render and encode block-end frames from `setA.h5` (PPO and PPO-Lagrangian episodes;
   a subset is enough), and fit the linear policy by ridge regression onto the mean
   action of the following block. This is θ_BC.
2. Run θ_BC on the 256 evaluation start states in the real executor, and compare with the
   recorded policies' own outcomes from the same states (`policy_tape`).
3. **Gate 0 (proposed thresholds):** θ_BC's real health violation rate is at most twice the
   recorded policies', and its forward progress is at least half theirs. If it fails,
   change the policy class before continuing: output the full 60-d block instead of a held
   action, or a small MLP, and evolve a low-dimensional perturbation around the
   behaviour-cloned weights.

## Stage 3: benchmark and noise (one 5090, about 11 October, 1 to 2 hours)

1. Check the real CPU quota (`/sys/fs/cgroup/cpu.max`) before anything else.
2. Measure imagined rows per second for population × roots × noise batches, and real
   segments per second including rendering and encoding.
3. Fit `σ` from one-step residuals on tuning episodes; record the multi-step error growth
   for reference.
4. **Output:** fixed run sizes (K, `n`, generations) chosen so stages 4 to 6 fit about 12
   GPU-hours. The imagined cost of one CMA-ES run is λ × K × n × 10 predictor rows per
   generation, so sizes follow the measured rate. Priority if time is short: stage 5a,
   then stage 6 at `k ∈ {0, 1, 4}`, then the rest.

## Stage 4: transfer check and Gate 1 (one 5090, by 17 October)

1. Build about 100 policies spanning good to bad: θ_BC plus Gaussian perturbations at four
   scales, and interpolations toward zero.
2. Score each in imagination (`k = 0`) and for real, on the 256 evaluation start states.
3. **Gate 1 (proposed):** Spearman correlation between imagined and real scores is at least
   0.5 for both mean return and violation rate, with 95% bootstrap intervals (clustered by
   source episode) above zero.
4. **If it fails:** the centre of the work becomes the fallback track below; the
   imagination results become a negative finding.

## Stage 5: the cheating curve and Gate 2 (one 5090, by 24 October)

No noise (`k = 0`), Deb's rule, 10 seeds per setting.

- **5a, pure selection (best of N).** Per seed, draw 4,096 policies around θ_BC, score all
  in imagination once, and take the imagined best of the first N for
  N ∈ {1, 4, 16, 64, 256, 1024, 4096}. Evaluate each pick for real. This isolates
  selection pressure from the optimiser's dynamics.
- **5b, CMA-ES.** Start at θ_BC; population λ ∈ {16, 64, 256}; 200 generations. At
  generations {0, 5, 10, 25, 50, 100, 200}, evaluate the best-so-far and the distribution
  mean in imagination and for real.

**Measured:** imagined violation rate, real violation rate (dense truth), real readout,
real return, and their gap, against N or generations.

**Gate 2 (proposed):** the real-minus-imagined violation gap at N = 4,096 exceeds the gap
at N = 1, with a paired bootstrap interval excluding zero. If no cheating appears, report
that the model was not exploited at this scale, and stage 6 becomes a comparison of rules
for safe evolution in imagination.

## Stage 6: the fix grid (one 5090, by 5 November)

CMA-ES from θ_BC with λ = 64 and the generation count at which stage 5b showed clear
cheating (200 if unclear). Grid: `k ∈ {0, 0.5, 1, 2, 4}` × rules {Deb, ROSARL, fixed
penalty at three λ} × 10 seeds = 250 runs. Evaluate every final best on the 256
evaluation start states, in imagination and for real.

**Main comparison:** at each noise level, ROSARL against Deb and the fixed penalties on
real violation rate and real return; and how each rule's imagined-real gap changes with
`k`.

**Success (from the paper idea):** noise plus ROSARL shrinks the gap more than the other
rules without wrecking return. Too much noise is expected to over-correct; the sweep shows
where.

## Stage 7: analysis and freeze (by 8 November)

- **Figures:** the cheating curve (gap against N and generations); the noise × rule grid;
  imagined against real scatter for stage 4; the gap decomposition into dynamics and
  probe error.
- **Tables:** real violation rate (Wilson intervals) and return per configuration.
- Commit small results and manifests under `docs/evoPlan/results/`; upload run bundles.

## Statistics

- 10 seeds per configuration before the freeze; 20 for GECCO where cheap.
- Across seeds: interquartile mean with bootstrap intervals. Rule comparisons:
  Mann-Whitney U with Holm correction.
- Violation rates: Wilson intervals, clustered by source episode where pooled. Zero
  violations in n segments bounds the rate only at about 3/n.
- Declare these thresholds and run sizes in the stage configs before looking at stage 4
  to 6 outcomes.

## Accounting

- **Real steps:** in this design they are spent only on evaluation, never on training, so
  evolution itself causes no real violations. Report evaluation steps separately, and
  count the roughly 3M steps of LeWM training data once in every comparison with
  real-environment methods.
- **Imagined rows, renders, encodes and wall-clock time** per run, from the harness
  counters.

## Fallback track (if Gate 1 fails)

Run the same three ranking rules with CMA-ES directly in MuJoCo, on a state-based linear
policy (17 inputs), against the recorded PPO-Lagrangian policy. This needs CPU workers
rather than a GPU and answers the allocated topic without a world model. It also stays
useful as the ground-truth chapter if Gate 1 passes and time allows.

## Risks

| Risk | Mitigation |
|---|---|
| The latent policy cannot walk at 12.5 Hz control | Gate 0 and its fallback policy classes |
| The model's errors are bias, which noise cannot fix | Report it; the real readout separates probe error from dynamics error; ensembles are a December option |
| Too few violations to measure differences | 256 evaluation start states; add more per test episode if intervals are too wide |
| Imagined throughput too low for the grid | Stage 3 sets sizes; priority order above |
| ROSARL adaptation does not match the paper's intent | Confirm before stage 6 |

## After the freeze (December to January)

- `SafetyHopperVelocity` as a second environment (new policies, data, LeWM and probes).
- A LeWM ensemble for noise that scales with model disagreement.
- Top-k real re-checking of elites as an extra fix.
- The GECCO paper, targeting the Neuroevolution track.

## Decisions to confirm with the supervisor

- The ROSARL adaptation (V_MIN and V_MAX from observed segment returns).
- The proposed thresholds for Gates 0, 1 and 2.
- Whether the health rule alone is enough, or the benchmark's speed cost should also be
  reported.
