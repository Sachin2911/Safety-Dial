# Agent notes

## TLDR

This Honours project asks whether the **failure set** a safe-RL agent needs can be **derived from dynamics instead of labelled by a human**. The criterion is **irreversibility**: a transition is unsafe when the agent cannot get back from it. That is a property of the transition structure, needs no annotation, and is only computable with a world model (it depends on rollouts of actions that were never taken). The work builds **return-reachability estimators in the latent space of a JEPA world model (LeWM / SIGReg)**, a substrate where reachability analysis has not been attempted before, and then exposes the resulting conservatism as the **Safety Dial**: a threshold an operator sets at deployment rather than an engineer fixing it during training.

**Project name:** SafetyDial. The dial is now the **reachability threshold `d`**, not a Pareto front.

> **Direction change (August 2026).** The project pivoted from *evolutionary multi-objective planning* (NSGA-II Pareto fronts over imagined plans) to *label-free latent reachability*. Most LaTeX documents in `docs/` still carry the old title and framing. The authority on the current plan is `docs/revisedProp/`. See [Doc authority](#doc-authority) and [Legacy framing](#legacy-framing-what-changed).

## Project snapshot

- **Name:** SafetyDial
- **Current title:** Label-Free Reachability in Joint-Embedding Predictive Architectures for Safe Reinforcement Learning. Subtitle: The Safety Dial: Irreversibility as a Deployment-Time Control
- **Former working titles:** Evolutionary Multi-Objective Planning in JEPA (EMOP in JEPA); SafetyDial: Selection Pressure for Safe Latent Planning in JEPA. Both are legacy
- **Student:** Sachin Mohan (2699183), BSc Honours CS, University of the Witwatersrand
- **Supervisor:** Geraud Nangue Tasse
- **Current proposal:** `docs/revisedProp/` (20 September 2026). LaTeX source plus built `RevisedProposal.pdf`
- **Guiding move:** stop asking a human to say what "unsafe" looks like; read it off the dynamics as irreversibility, then make conservatism a runtime knob

### The gap being attacked

Latent safety filters (Nakamura et al. 2025, UNISafe, AnySafe) removed the penalty weight from safe RL, which is real progress. They did not remove **supervision**: the failure set that seeds the reachability computation is a classifier over human-annotated failure observations, so the filter is blind to any failure mode nobody labelled. Irreversibility is proposed as the label-free replacement for that classifier.

### Research question

Can the failure set required by a latent safety filter be derived from irreversibility in the dynamics, without failure labels, and does such a filter, computed in a JEPA latent, detect failure modes that a supervised latent safety filter trained on *different* labelled failures cannot?

### Hypotheses

| ID | Claim |
|----|-------|
| H1 | **Metric validity.** Distance in the SIGReg-regularised JEPA latent tracks action-steps between states, not just observational similarity |
| H2 | **Detectability.** A return-reachability score by counterfactual rollout separates irrecoverable from recoverable states well above chance, and better than forward-dispersion measures (which are confounded by leftover actuator freedom) |
| H3 | **Transfer.** On a failure mode excluded from the supervised baseline's labels, the label-free method detects materially better, while staying competitive on the labelled mode |
| H4 | **Dial monotonicity.** Varying the reachability threshold at deployment gives a monotone failure-rate against task-success trade-off, from one trained model rather than one training run per operating point |

### Method (five gated stages)

0. **Substrate and environment.** Use a *released* LeWM checkpoint (world-model training is infrastructure, not contribution). Environment must contain genuinely irreversible failures, which rules out Safety-Gymnasium (its hazards are recoverable). **OGBench-Cube is the primary candidate** (cube off the table is unrecoverable). Build a data-collection harness recording (observation, action, latent, t, trajectory id, simulator state), with simulator state reserved for evaluation only.
1. **Latent metric validity (gating).** Regress latent distance against true simulator-state distance *and* against action-steps separating the pair. A latent that only tracks visual similarity cannot support reachability. Run a **probe-space variant in parallel** (linear probe for object position, reachability computed in the probe subspace) as both a fallback and a cross-check. Probe supervision is of a generic physical quantity, not of the safety concept; say so explicitly in all reporting.
2. **Irreversibility estimation.** Two estimators, developed in parallel:
   - *Rollout*: sample K action sequences from a **fixed** distribution from `z_{t+k}`, roll forward through the JEPA predictor, score `rho = min_j d(z_hat, z_t)`. Fixing the action distribution is what makes differences reflect dynamics rather than control authority.
   - *Precedence* (Grinsztajn et al. 2021 style): classifier on unlabelled trajectories predicting which of `(z_i, z_j)` came first; confident asymmetry marks an irreversible transition. No rollout at inference, so it also answers the compute risk.
   - Forward dispersion is implemented as the comparison that is expected to *lose*. Where a covariance statistic is needed use **effective rank**, not log-determinant (undefined when rank-deficient).
3. **Zero-shot failure transfer** (the principal empirical claim). Define two irreversible failure modes. Train a supervised latent safety filter with labels for mode A only; train the proposed method on the identical dataset with no labels; evaluate both on mode B. **Critical control:** the shared unlabelled dataset must actually contain mode B instances. Verify and report dataset composition *before* running the comparison, or the result is void.
4. **Safety Dial and evaluation.** Threshold the reachability margin at `d` chosen at deployment; calibrate on held-out offline data (conformal prediction supplies the coverage statement). Report irreversible-failure rate against task success as `d` varies, and compare compute to recover the whole curve against penalty-swept baselines (one training run per operating point). Stretch: a `d`-conditioned value function over frozen latents as terminal cost.

### Validation harness (Stage 2)

About 100 states annotated recoverable / irrecoverable, **for evaluation only**, generated **programmatically** (scripted push-off-edge for irrecoverable, matched mid-manipulation states for recoverable) rather than picked by eye, to avoid selection bias toward visually obvious cases. Compare estimators by AUROC. Two mandatory controls: score states at joint limits (control authority reduced, nothing broken) to prove the measure does not track actuator freedom; sweep rollout horizon H to prove it does not just track predictor drift.

### Success criteria (decreasing necessity)

1. A characterisation of whether JEPA latent distance carries dynamical meaning, with the probe-space comparison. Reportable either way.
2. A label-free irreversibility estimator at **AUROC >= 0.8** against held-out annotations, with both confounds controlled.
3. Zero-shot detection of an unlabelled failure mode beating a supervised baseline on the same data.
4. A calibrated Safety Dial giving a monotone trade-off from a single trained model.

### Timeline (from the revised proposal)

| Stage | Period |
|-------|--------|
| 0 Substrate, environment, data harness | late August 2026 |
| 1 Latent metric validity, gating decision | early September |
| 2 Estimators, annotation harness, AUROC | September |
| 3 Supervised baseline, zero-shot transfer | October |
| 4 Dial calibration, trade-off curves, ablations | late October |
| Write-up | November |

### Target venue: ICLR 2027 workshop

Decided 20 September 2026. **The ICLR 2027 main track is not the target.** Its abstract deadline (18 September 2026, abstract registration mandatory) passed without a submission, deliberately: the idea was judged not ready, and a rushed abstract was not worth the slot. Do not treat that as a missed deadline to recover from.

| Date | What |
|------|------|
| 29 November 2026 | ICLR 2027 accepted workshop list announced. No specific CFP exists before this |
| ~1 February 2027 | Suggested workshop paper deadline. Each workshop sets its own on OpenReview, clustering late January to early February |
| 26 February 2027 | Mandatory accepted-paper notification |
| 29 to 30 April 2027 | Workshops, San Francisco (main conference 26 to 28 April) |

Every ICLR workshop must accept short papers of 3 to 5 pages in ICLR format; full workshop tracks are usually 4 to 9 pages. Workshop papers are non-archival and ICLR's dual-submission policy explicitly permits them, so a workshop paper does **not** block a later full submission to ICLR 2028, NeurIPS 2027 or ICML 2027.

**Scoping consequence, which governs what work is in scope.** Results must be frozen by mid-January 2027. Stage 1 plus Stage 2 is the workshop paper. Stages 3 and 4 are the conference paper that follows. The thesis write-up lands November 2026, leaving December and January to cut it down.

Candidate workshops, in fit order, from the ICLR 2026 list as a predictor: World Models: Understanding, Modelling and Scaling (was on its 2nd edition); VerifAI (AI verification); Principled Design for Trustworthy AI; Agents in the Wild; and ICBINB ("I Can't Believe It's Not Better") as the negative-results fallback.

The proposal's Section 9 carries the same schedule. These dates and the proposal are the only record; the earlier root-level planning notes were removed on 20 September 2026.

## Doc authority

When sources conflict, follow this order:

1. **`docs/revisedProp/latex/main.tex`** (20 Sep 2026). **The current plan.** Label-free irreversibility, evaluated on the Safety-Gymnasium locomotion suite, three hypotheses, five gated stages. Supersedes everything below on research content, environment and scope
2. **`notes/`** and the `experiments/` notebooks. What was actually run and what it showed
3. **`docs/*/submitted/*.pdf`.** All still describe the older frameworks (NSGA-II Pareto planning, and older still, EA for LLM alignment). Historical only. (`readme.md` was rewritten to the reachability framing on 20 September 2026 and is current)

Do not let the NSGA-II / Pareto-front story, or the older CoEvoRL / LLM-alignment story, override the current plan. If asked to write new project prose, write the reachability story.

### Legacy framing (what changed)

| Was | Now |
|-----|-----|
| Safety as a second objective on a Pareto front | Safety as a reachability threshold, a filter override |
| NSGA-II replacing CEM as the planner | Planner is not the contribution. CEM stays as-is; the filter sits on top |
| The dial is the Pareto front | The dial is the threshold `d`, calibrated conformally |
| Structural signals: surprise, latent energy, jerk, probes | Return-reachability (rollout) and precedence classification. Probes survive, as the interpretable probe-space variant |
| Metrics: hypervolume, IGD | AUROC against recoverability annotations; failure-rate against success curves |
| Primary envs: Reacher, Push-T | OGBench-Cube primary (needs genuinely irreversible failures). Push-T is where the pilot work happened |
| Baselines: penalty-tuned CEM, CPO, Lagrangian PPO, ROSARL | Supervised latent safety filter (UNISafe style) trained on labels for one failure mode |

`pymoo` was removed from the dependencies on 20 September 2026; nothing used it. If NSGA-II vocabulary appears in old docs, it is legacy.

### Ideas held in reserve

`thesis_trajectory.md` (31 Aug 2026) proposed a **quasimetric** route: measure irreversibility as
the asymmetry of a learned temporal distance, `r = d(z'->z) - d(z->z')`, fit by an Interval
Quasimetric Embedding on `(z_t, z_{t+k}, k)` triplets. That file was deleted on 20 September 2026
and the route was **not** adopted: the current proposal uses rollout return-reachability plus a
precedence classifier. Three ideas from it are worth keeping and are recorded here so they are not
lost with the file:

- **A false-safe bound.** Bound the filter's false-safe rate in terms of estimator approximation
  error and predictor Lipschitz constant, and check it is non-vacuous empirically. The current
  proposal has conformal calibration but no bound.
- **The substrate comparison as a first-class result.** End-to-end versus pretrained encoders,
  originally LeWM versus V-JEPA 2. This survives in the proposal as H1, but framed as frozen
  general-purpose features versus an end-to-end JEPA.
- **The directional confound control.** Expert demonstration data is directionally biased, so any
  asymmetry measure must be refit on reversed trajectories and on a random-policy subset. This
  matters for the precedence estimator too, and the proposal's Stage 2 should inherit it.

The quasimetric route is cheaper at inference (one forward pass, no rollout) and remains the
obvious fallback if rollout-based estimation proves too expensive or too noisy.

## Where things stand

Work so far is Push-T pilot work in `experiments/`, on the released `pusht/lewm` checkpoint (192-d latents). It predates the pivot but the plumbing and the probe carry over into Stage 1.

**Probes from LeWM latents to pusher (x, y)**, 100k frames, 80/20 split:

| Probe | R^2 | RMSE |
|-------|-----|------|
| Ridge (alpha=1) | 0.965 | 18.9 px |
| MLP 192-256-128-2, Adam, 80 epochs | 0.999 | 3.2 px |

**Probe on imagined latents** (horizon 5, frameskip 5, 256 clips): the probe still works on predicted `z_hat`, R^2 0.98 at t=5. Radial error at t=5: median 10.2 px, p95 35.5 px, max 112 px. The p95 is what `PROBE_MARGIN` uses. Do not use the per-coordinate holdout RMSE for a planning margin: radial error is about sqrt(2) larger, and the tail is what breaches a cushion.

**Penalty-CEM hazard sweep** (7 lambdas x 5 seeds, hazard box calibrated to sit on the lambda=0 path):

| lambda | violating steps | final block err | out of bounds |
|--------|-----------------|-----------------|---------------|
| 0 | 0.27 | 16.4 px | 0.4 |
| 0.03 | 0.06 | 60.6 px | 1.0 |
| 0.1 to 10 | 0.02 to 0.04 | 70.7 px | 1.0 |

70.7 px is exactly the start-to-goal block distance, meaning **the block never moved**. Every non-zero lambda bought safety by abandoning the task, and every non-zero lambda drove the pusher out of the arena. Task `success` was False in all runs. This is a clean, first-hand demonstration of the penalty-fragility argument the proposal opens with, and it is worth writing up as motivation. It is also part of why the project stopped trying to tune a scalar trade-off.

Open threads from that work: Push-T `success` never fires even when the block lands close, so the task axis needed replacing with coverage and final block error; out-of-bounds states break the observation space and any episode with them is not a valid data point.

## Repo layout

There is **no `src/` package**. The `src/safetydial/...` tree described in earlier versions of this file was scaffolding and has been removed. Current work lives in notebooks plus a helpers module.

| Path | Role |
|------|------|
| `experiments/*.ipynb` | Where the work happens. `PushTDataExploration.ipynb` (dataset survey), `LinearProbeAndAvoidance.ipynb` (first probe plus hazard run), `LinearProbeAndAvoidance_4.ipynb` (**current**: probes, imagination check, calibrated hazard box, lambda sweep, demo) |
| `experiments/helpers/linProbeHelpers.py` | Push-T state construction, rendering and animation, trajectory plots, box penetration and path interpolation, `HazardAugmentedCostModel`, violation stats, CEM history summaries |
| `experiments/helpers/hazardSweep.py` | The corrected sweep layer: episode metrics, hazard-box calibration from the baseline path, margin from imagination error, detour feasibility, `run_episode`, `sweep_lambda`, `plot_front`. Its module docstring lists what it fixed and why |
| `experiments/helpers/locoEnv.py` | **Vendored Safety-Gymnasium velocity task.** Threshold table, env registration under `safetydial/*`, `env_manifest()` fingerprint |
| `experiments/helpers/locoCollect.py` | State-only HDF5 writer and rollout loop. Stores `(qpos, qvel, action, ...)`, never pixels |
| `experiments/helpers/locoData.py` | Lazy-render dataset: subclasses `swm.data.Dataset`, renders pixels on demand from stored state. Worker-safe EGL, render fingerprint |
| `experiments/helpers/locoPolicies.py` | Uniform `act(obs)` adapters: random, scripted-forward, mixed, trained-actor |
| `experiments/helpers/locoMetrics.py` | Gate 0 statistics, disjointness lift, matched pairs, AUROC, trivial baselines |
| `experiments/helpers/oracle.py` | **Recoverability oracle.** The project's headline contribution. Evaluation only |
| `experiments/helpers/equivCheck.py` | Vendored-env equivalence gates G1..G9. Deliberately numpy-only and torch-free |
| `notes/` | Short markdown findings. `pushTDataExp.md` is the Push-T HDF5 layout reference, `safetyGymLocomotion.md` the measured Safety-Gymnasium facts, `terminationIsNotIrreversibility.md` the headline Phase 0 result |
| `scripts/` | `setup.sh` (local or Vast), `download_data.py` (LeWM clone plus Hub weights and data), `helpers/_common.sh` |
| `configs/download/` | Hydra configs for `download_data.py`: `all` (default), `pusht`, `cube` |
| `docs/` | Deliverables and papers, see [Docs map](#docs-map) |
| `data/`, `third_party/` | Local artifacts, contents gitignored, `.gitkeep` tracked |
| `docs/revisedProp/` | **The current proposal.** LaTeX source, bib, and built PDF |

New Python that outgrows a notebook goes in `experiments/helpers/` unless we deliberately re-introduce a package. `pyproject.toml` no longer points at a `src/` package or a `tests/` directory; if either comes back, re-add `[tool.pytest.ini_options]` and update ruff's `src`.

### Known warts

- Notebooks are committed with outputs, so they run 2 to 5 MB each.
- `docs/papers/myPapers/unisafe.pdf` and `UncertaintyAwareLatentSafety.pdf` are the **same paper** (Seo et al., UNISafe) committed under two names; `unisafe.pdf` (5.4 MB) was also never Ghostscript-compressed, unlike the 1.4 MB copy. `CEM-EVO.pdf` is actually CEM-RL.

## Data and checkpoint layout

`scripts/download_data.py` clones LeWM and pulls weights and expert data from the Hub, then writes `STABLEWM_HOME` into `.env`.

```
data/raw/                                  # downloaded .zst archives
data/processed/pusht_expert_train.h5       # decompressed Push-T expert data (~2.3M steps)
data/processed/cube_single_expert/         # decompressed Cube expert data
data/stablewm/                             # STABLEWM_HOME
  hf_pusht/, hf_cube/                      # raw HF snapshots (weights.pt + config.json)
  checkpoints/pusht/lewm_object.ckpt       # converted torch object checkpoints
  checkpoints/cube/lewm_object.ckpt
third_party/le-wm/                         # upstream clone, gitignored
```

Sizes: weights are about 72 MB each; Push-T expert data about 13 GB compressed, Cube about 46 GB. `weights_only=true` skips the datasets.

**Disk is the live blocker for Stage 0.** Push-T decompresses at about 3.5x (13 GB to 46 GB). At that ratio Cube needs well over 100 GB, against roughly 63 GB free on the current Vast box. OGBench-Cube is the proposal's primary environment, so this has to be solved before Stage 0 can run there. The obvious first reclaim is `data/raw/pusht_expert_train.h5.zst` (13 GB), which `download_data.py` keeps after decompressing and which is re-downloadable from the Hub. Note the default `--config-name all` pulls Cube too; use `--config-name pusht` or `weights_only=true` to avoid filling the disk.

`decompress_dataset` writes to a `.part` path and renames only on success, so an interrupted or out-of-disk decompress no longer leaves a truncated file that `processed_ready()` would treat as complete and skip.

Model ids are `pusht/lewm` and `cube/lewm`, loaded as `swm.policy.AutoCostModel("pusht/lewm")`.

Push-T dataset shape lives in `notes/pushTDataExp.md`: flat timestep arrays, 2,336,736 steps over 18,685 episodes, `ep_offset` plus `ep_len` slicing, 7-d `state` (pusher xy, block xy, block theta, pusher vx vy), 224x224x3 `pixels`. Arena coordinates are roughly 0 to 512 but can leave that range.

## Working with LeWM and Push-T (gotchas)

Collected from the notebooks and helpers. These cost real debugging time; do not rediscover them.

- Notebooks find the repo by walking up for `scripts/download_data.py`, then put `third_party/le-wm` on `sys.path` to import `utils.get_img_preprocessor` and `get_column_normalizer`, then set `STABLEWM_HOME` to `data/stablewm` (assign, do not `setdefault`).
- `import hdf5plugin` is required before reading the compressed HDF5, even though nothing references it directly.
- Feed LeWM through **its own** preprocessor (`get_img_preprocessor(source, target, img_size=224)`). Raw env frames will not do. `HDF5Dataset` order is `(1, H, W, C)` permuted to `(1, C, H, W)` before the preprocessor.
- The planner needs a `process` dict of fitted `StandardScaler`s for `action`, `proprio`, `state`, plus `goal_` duplicates for the non-action columns. Fit on `dataset.get_col_data(col)` with NaN rows dropped.
- `swm.PlanConfig(horizon=5, receding_horizon=1, action_block=5)` matches LeWM. **`action_block` must stay 5** (the LeWM frameskip); setting it to 1 is wrong.
- `CEMSolver(seed=...)` does not control every RNG. Seed `torch` and `numpy` globally as well, or two runs at the same nominal seed diverge.
- Load `AutoCostModel` **once** outside any episode loop. Reloading LeWM per episode dominated runtime.
- `EVAL_BUDGET = 50` matches the le-wm Push-T eval; `World(..., max_episode_steps=2*EVAL_BUDGET)`.
- Random fancy-indexing into the uncompressed HDF5 is very slow. Read contiguous chunks at spread-out offsets instead (the notebooks take 100 chunks of 1000).
- Arena is 512 px with image-style coordinates (y down, origin top-left). Rendered frames are 224 px, so the arena-to-image scale is `224/512`.

## Docs map

| Path | Role |
|------|------|
| `docs/revisedProp/` | **Current proposal** (20 Sep 2026). `latex/main.tex` plus `references.bib`; `./compile.sh` builds `RevisedProposal.pdf`. Also published as an artifact for the supervisor |
| `docs/ideation/` | Frozen `submitted/ID.pdf` plus `whiteBoard.md`. Legacy framing. LaTeX source removed 20 Sep 2026 |
| `docs/researchProp/` | Original proposal, frozen `submitted/RP.pdf`. Superseded by `docs/revisedProp/`. LaTeX source removed 20 Sep 2026 |
| `docs/AB/` | Annotated bibliography, frozen `submitted/AB.pdf`; `whiteBoard.md` holds the 7-paper list. LaTeX source removed 20 Sep 2026 |
| `docs/litReview/` | Literature review. Frozen `submitted/LR.pdf`. Oldest framing (EA / LLM alignment) |
| `docs/papers/myPapers/` | Student-chosen papers (PDF). 15 files |
| `docs/projectPresentation/`, `docs/projectReport/` | Placeholders for later deliverables |
| `readme.md` | Front door: the reachability pitch, setup quickstart, deliverable checklist |

Only `docs/revisedProp/` still carries LaTeX sources. The shape is `latex/main.tex` plus `references.bib`, `compile.sh` to build (`./compile.sh` builds, `./compile.sh clean` removes artifacts), intermediates in `latex/build/` (gitignored), final PDF at the folder root. The older deliverables were reduced to `submitted/` (the frozen handed-in PDF) plus `guides/` and `whiteBoard.md` on 20 September 2026; their sources all carried the superseded title.

Building needs a LaTeX toolchain, which is not in the base image. On a fresh box:

```bash
apt-get install -y --no-install-recommends texlive-latex-recommended texlive-latex-extra \
  texlive-fonts-recommended lmodern latexmk biber texlive-bibtex-extra
```

## Reading list

Core to the current direction:

| Paper | Where | Role |
|-------|-------|------|
| Generalizing Safety Beyond Collision-Avoidance (Nakamura, Peters, Bajcsy 2025, arXiv 2502.00935) | `myPapers/LatentSafetyFilters.pdf` | The framework being modified: HJ reachability in a learned latent, seeded by labelled failures |
| UNISafe: Uncertainty-aware Latent Safety Filters (Seo, Nakamura, Bajcsy 2025, arXiv 2505.00779) | `myPapers/unisafe.pdf` **and** `myPapers/UncertaintyAwareLatentSafety.pdf` (same paper, committed twice) | The supervised baseline for Stage 3 |
| AnySafe (arXiv 2509.19555, ICRA 2026) | not yet in repo | Runtime-adjustable constraints via conformal similarity. Closest prior work on the "dial" idea |
| No Turning Back (Grinsztajn et al. 2021) | not yet in repo | Precedence-classifier reversibility estimation. Basis of the second estimator |
| LeWorldModel (LeWM, Maes et al. 2026) | **Notion** (removed from repo, commit e631dc0) | The frozen JEPA world model. Also the source of "physical quantities are linearly probeable" |
| LeJEPA / SIGReg (Balestriero and LeCun 2025, arXiv 2511.08544) | `myPapers/LeJEPA.pdf` | Why the latent is isotropic, which is what makes latent distance well posed |
| Hamilton-Jacobi reachability (Bansal et al. 2017) | not yet in repo | The reachability machinery being ported |
| OGBench (Park et al. 2025) | not yet in repo | Source of the Cube environment, the Stage 0 primary candidate |

Supporting and comparison:

| Paper | Where | Role |
|-------|-------|------|
| Hierarchical Planning with Latent World Models | `myPapers/HierarchicalPlanningWithLatentWorldModels.pdf` | Latent MPC, single scalar objective |
| DINO-WM | `myPapers/DINO-WM.pdf` | Alternative substrate: world model on frozen pre-trained features |
| SafeDreamer | `myPapers/SafeDreamer.pdf` | Safe RL inside a generative world model |
| URWM | `myPapers/URWM.pdf` | Uncertainty-aware robotic world model |
| World Models (Ha and Schmidhuber) | `myPapers/worldModels.pdf` | Origin point |
| SKY-JEPA, FF-JEPA | `myPapers/SKYJepa.pdf`, `FF-Jepa.pdf` | Further JEPA variants |
| Constrained MBRL with Robust Cross-Entropy Method | `myPapers/MPC-RCE.pdf` | Safety-aware CEM planning |
| CEM-RL (Pourchot and Sigaud) | `myPapers/CEM-EVO.pdf` (filename is misleading, the paper is CEM-RL) | Evolutionary plus gradient policy search. Legacy from the EA framing |
| ROSARL | **Notion** (removed from repo, commit 8280a44) | Supervisor work. Derives a sufficient penalty from intrinsic quantities, still one scalar fixed before deployment |
| Safety-Gymnasium | **Notion** (removed from repo, commit 8280a44) | Standard benchmark. **Explicitly rejected** as the evaluation environment: its hazards are recoverable |
| OmniSafe | `myPapers/omniSafe.pdf` | Safe-RL implementations |
| DQN | `myPapers/DQN.pdf` | Background |
| MAP-Elites, DQD-RL | **Notion** (removed from repo, commit 8280a44) | Legacy QD track from the evolutionary framing |

## Environment and tooling

- **Package manager:** `uv`. Python pinned to **3.11** via `.python-version`.
- **Sync:** `uv sync --frozen --extra dev` (what `setup.sh` runs). Commit `uv.lock` when dependencies change.
- **Run anything:** `uv run python ...`, `uv run pytest`, `uv run ruff check .`
- **Direct deps:** `stable-worldmodel[env,train]` (the LeWM stack: `swm.World`, `swm.policy.AutoCostModel`, `CEMSolver`, `HDF5Dataset`), `torch`, `gymnasium`, `numpy`, `hydra-core` plus `omegaconf`, `huggingface_hub`, `h5py`, `zstandard`, `hdf5plugin`, `wandb`, `tqdm`. Dev extras: `pytest`, `ruff`, `ipykernel`.
- **Transitive but used directly in notebooks:** `scikit-learn`, `matplotlib`, `scipy`, `stable_pretraining`, `gym-pusht`, `ogbench`. They arrive via `stable-worldmodel`; if a notebook starts leaning on one, promote it to an explicit dependency.
- **Lint:** ruff, `line-length = 100`.
- **Not installed as a package** (`[tool.uv] package = false`).
- Notebooks import helpers as `import helpers.linProbeHelpers as lph`, so run JupyterLab with `experiments/` as the working directory. `setup.sh` registers a kernel named **"Python (safetydial .venv)"** (`safetydial-venv`); select it, and refresh the page if it does not appear.

## Setup scripts and remote GPU

| Script | Use |
|--------|-----|
| `scripts/setup.sh` | Local or Vast setup. Auto-detects Vast. `--pull` updates the clone first. Piped or pasted (Vast On-start) it clones or pulls, then re-runs from the repo |
| `scripts/helpers/_common.sh` | Shared helpers (do not run directly): dotenv loading, uv install, system deps (`swig` for box2d, `btop`), git identity, HTTPS push token, venv sync, Jupyter kernel, HF login, sanity check |
| `scripts/download_data.py` | Clone `third_party/le-wm`, then Hydra Hub download. Default `--config-name all` (Push-T plus Cube weights and expert data). `pusht` / `cube` for one task. `weights_only=true` skips datasets; `clone_source=false` skips the git clone |

Idempotent: `git pull && bash scripts/setup.sh` (or `bash scripts/setup.sh --pull`) is the normal refresh. Vast hosts should have `cuda_max_good>=13.0` so the torch CUDA build works. Real runs happen on Vast at `/workspace/Safety-Dial`; the local `.venv` is usually only partially synced.

## Secrets

Secrets live in a gitignored `.env` at the repo root, loaded by `_common.sh` and by `download_data.py`; on Vast they also come from account Environment Variables. Recognised keys:

- `WANDB_API_KEY` (logging)
- `GITHUB_TOKEN` (optional; enables push over HTTPS)
- `HF_TOKEN` (optional; Hugging Face downloads)
- `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL` (repo-local git identity, not global)
- `STABLEWM_HOME` (written automatically by `download_data.py`)
- `LEWM_REPO_URL`, `LEWM_GIT_REF` (optional overrides for the LeWM clone)

Rules: never commit `.env`, never print token values into logs or terminal output, and do not put them in configs or code. There is no tracked `.env.example`; document new keys here instead.

## Glossary

- **SafetyDial:** Project name. A runtime-adjustable conservatism threshold on a label-free reachability margin, so the safety-performance operating point is chosen at deployment
- **Irreversibility:** The failure criterion. A transition is unsafe when no available action sequence returns the system to where it was. Dynamical, not annotated
- **Return-reachability:** The measure. `rho(z_t -> z_{t+k}) = min_j d(z_hat_{t+k+H}^{(j)}, z_t)` over K action sequences from a fixed distribution. Large `rho` means one-way
- **Precedence classifier:** Label-free reversibility estimator (Grinsztajn et al.). Predicts which of two latents came first; confident asymmetry marks irreversibility. Free labels, no rollout at inference
- **Forward dispersion:** Empowerment-style spread of terminal latents. Implemented as the comparison expected to lose, since actuator freedom survives a failure
- **Effective rank:** `(sum lambda_i)^2 / sum lambda_i^2`. Used instead of log-determinant, which is undefined for rank-deficient sample covariance
- **Latent safety filter:** Labelled failures -> latent failure classifier -> reachability value function -> binary override. The proposed work replaces step one and keeps the rest
- **HJ reachability:** Hamilton-Jacobi backward reachable tube of a failure set: states from which failure is inevitable
- **Viability kernel:** Control-theory analogue (Aubin). Largest set from which the system can stay inside a constraint set forever, defined as a fixed point rather than seeded by labels
- **JEPA:** Joint-Embedding Predictive Architecture. Predicts future latents, not pixels
- **LeWM:** LeWorldModel. The JEPA world model used, via `stable_worldmodel`. Model ids `pusht/lewm`, `cube/lewm`
- **SIGReg / LeJEPA:** Sketched Isotropic Gaussian Regulariser. Prevents collapse and pushes the latent toward isotropy, which is what makes latent distance comparable across directions
- **Probe:** Small regressor from latent to a physical quantity (here pusher xy). Task-agnostic supervision, not safety supervision. Basis of the interpretable probe-space fallback
- **UNISafe / AnySafe:** Uncertainty-aware and runtime-adjustable latent safety filters. UNISafe is the supervised baseline; AnySafe is the nearest prior "dial"
- **OGBench-Cube:** Target environment. A cube off the table is genuinely irrecoverable
- **Conformal calibration:** Gives the deployment threshold `d` a distributional coverage interpretation instead of being an arbitrary constant
- **Zero-shot failure transfer:** The headline experiment. Baseline gets labels for failure mode A; both methods are evaluated on mode B
- **CEM:** Cross-Entropy Method. LeWM's planner, used as-is under the filter
- **ROSARL:** Reward-Only Safe RL (supervisor work). Derives a sufficient Minmax penalty from intrinsic quantities; still one scalar fixed in advance
- **swm:** `stable_worldmodel`, the package providing `World`, `AutoCostModel`, `CEMSolver`, `HDF5Dataset`, and the `swm/PushT-v1` environment
- **uv:** Python package and environment manager. Everything runs under `uv run`
- **Vast.ai:** Rented GPU provider used for runs; `scripts/setup.sh` auto-detects it
- **EMOP / NSGA-II / hypervolume / IGD / MAP-Elites:** Legacy vocabulary from the evolutionary multi-objective framing. Recognise it in old docs, do not use it in new writing

## Agent must-know

### Research

- The project is **label-free latent reachability**, not multi-objective evolutionary planning. Prefer the reachability story in all new writing.
- The failure criterion is **irreversibility**; the contribution is removing failure labels, not removing the penalty weight (latent safety filters already did that).
- The dial is the **reachability threshold `d`**, calibrated conformally, not a Pareto front.
- **Stage 1 gates everything.** If JEPA latent distance turns out to track appearance rather than dynamics, that negative result is itself the deliverable, and the probe-space formulation is the fallback.
- The world model stays **frozen and released**. Training LeWM is infrastructure, not contribution.
- **Safety-Gymnasium locomotion is now the primary environment** (`SafetyWalker2dVelocity-v1` and siblings), **vendored** into `experiments/helpers/locoEnv.py` rather than installed. See `notes/safetyGymLocomotion.md` for the measured facts and `notes/envEquivalence.md` for the equivalence report. The earlier blanket rejection applied to the *navigation* suite, whose hazards are extrinsic painted regions; the locomotion suite carries an intrinsic irreversible failure (falling) that is absent from the `cost` channel. That split is the whole experiment.
- **Gate 0 gates the data regime, not the environment.** Published OmniSafe numbers back out to a converged PPO-Lagrangian that runs episodes to truncation and almost never falls, while unconstrained PPO violates on ~90% of steps. Both fail a naive "falls while cost stays rare" test, from opposite directions. That is a property of converged experts, not of the benchmark, and nobody deploys a runtime safety filter on a policy that never fails. The central statistic is **disjointness lift**, not "does cost ever fire".
- **`terminated` is not ground truth for irreversibility, and the gap is large.** Measured: at the step the benchmark declares failure, Walker2d is typically still *standing* at z ~ 1.09, merely leaning past 55 degrees, and does not reach the ground for another ~60 steps. CEM on the true simulator recovers from **96%** of states at the flag. `terminated` is a threshold on torso height and pitch, both of which decode from a raw 32x32 grayscale frame at R^2 = 0.99, so scoring against it is also nearly circular. Use the recoverability oracle in `experiments/helpers/oracle.py`. Full write-up in `notes/terminationIsNotIrreversibility.md`; this is the workshop paper's headline.
- **Oracle labels are evaluation only.** Nothing in the method path may import `oracle.py` or read its label files, and no method hyperparameter may be selected on oracle AUROC.
- **Collect with termination DISABLED.** `rollout_episode` records state *before* each action, so `terminated[t]` means "action t made it unhealthy" while `qpos[t]` is the last HEALTHY state; under the benchmark's own termination the episode then ends, so **every row in such a dataset is healthy and it contains no failures at all**. Build the env with `terminate_when_unhealthy=False` and pass `stop_after_unhealthy`. Use the `healthy` column, never `terminated`, to identify failure states. `steps_to_failure` is signed: positive before the first unhealthy step, zero at it, negative after.
- **Report matched-pair AUROC, not population AUROC.** Pose alone scores ~0.98 on the raw population. A matched pair holds pose almost fixed and varies only recoverability, so it is the honest measurement. Report it as a margin over the strongest trivial baseline.
- Scope claims to **irreversible failures**, and state that limitation explicitly rather than defending it. Irreversibility is not the same thing as danger.
- Do not skip the two Stage 2 controls (joint limits, horizon sweep) or the Stage 3 dataset-composition check. Without them the results are confounded or void.
- Every probe use must be labelled as supervision of a **generic physical quantity**, never of the safety concept.
- This is an active area with a high preprint rate. Check arXiv and OpenReview before claiming novelty.
- The target venue is an **ICLR 2027 workshop**, not the main track. Scope to what can be frozen by mid-January 2027: Stage 1 plus Stage 2. See [Target venue](#target-venue-iclr-2027-workshop).

### Engineering

- Run everything through **`uv run`**; do not `pip install` into the venv or create a second
  project environment. **One narrow exception:** out-of-band *tooling* that cannot share the
  project's Python pin, currently `safety-gymnasium` and `omnisafe`. Both are capped below 3.11
  by hard `==` pins on gymnasium 0.28 and mujoco 2.3, so adding them makes `uv lock` fail
  outright. They run as ephemeral invocations that never touch `.venv` or `uv.lock`:

  ```bash
  uv run --isolated --no-project --python 3.10 \
    --with "safety-gymnasium==1.0.0" --with "numpy<2" \
    python experiments/scripts/emit_reference_traj.py
  ```

  They may produce only small artefacts (reference trajectories, policy weights), never datasets
  and never a reported number. Everything that appears in the thesis or the paper is produced
  inside `.venv`.
- New code goes in `experiments/helpers/` (or a notebook). There is no `src/` package; do not resurrect one casually.
- Ruff is clean. Notebook-idiom rules are silenced per-file in `pyproject.toml`; if a new error appears in a `.py` file, fix it rather than widening the ignore list.
- Configs go in `configs/<group>/` as Hydra groups.
- Never commit secrets, checkpoints, datasets, run outputs, or `wandb/`.
- Compress any new PDF under `docs/papers/` before committing (see below).
- Follow the writing rules below (no em dashes).

## Locomotion gotchas

Collected from Phase 0. These cost real debugging time.

- `MUJOCO_GL=egl` must be set **before** `import mujoco`. `locoEnv.py` does it at module import, so import it first.
- Render size must be passed at `make()` time. MuJoCo sizes its offscreen framebuffer once when the GL context is created; setting `width`/`height` afterwards silently does nothing.
- **An EGL context inherited across `fork()` does not raise.** It renders black or stale frames, which surfaces days later as "the world model will not learn". `locoData` defends with a lazy context, an owner-pid assertion, a render fingerprint checked in every worker, and `spawn` as the default.
- **`set_state` does not fully reset the solver.** It leaves `data.qacc_warmstart` and `data.ctrl` holding values from whatever ran before, so the same rollout replayed after different predecessors differs in the last two digits. `RecoverySimulator._reset_to` zeroes them. Labels are only bit-reproducible because of this.
- HalfCheetah and Swimmer never terminate upstream and have their own `step()` bodies; they are deliberately not registered.
- Oracle throughput on the 5090: ~8,200 sim steps/s per process. T1+T2 is ~8 s per hard state, full CEM ~39 s. Multiprocess physics saturates around 24 workers; 48 is slower.

## No em dashes

Never use em dashes in project writing (LaTeX `---`, Unicode, or pasted en/em dash characters used as clause breaks). Prefer commas, parentheses, colons, or a full stop. For compound modifiers such as safety-performance, use a plain hyphen (`-`), not `--`. When editing existing docs, remove any em dashes you find rather than leaving them.

## Compress paper PDFs before committing

Papers under `docs/papers/` often ship with high-resolution embedded figures and can be tens of megabytes each. After adding or replacing any PDF in that tree, compress it with Ghostscript `/ebook` (downsamples images to about 150 DPI; does **not** remove images or text).

```bash
# Compress one paper in place (only replace if smaller)
in="docs/papers/.../Paper.pdf"
tmp="${in}.tmp.pdf"
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
  -dNOPAUSE -dQUIET -dBATCH \
  -sOutputFile="$tmp" "$in"
# if tmp is smaller than in: mv "$tmp" "$in"; else rm "$tmp"
```

Batch all papers:

```bash
find docs/papers -name '*.pdf' -print0 | while IFS= read -r -d '' f; do
  tmp="${f}.tmp.pdf"
  gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook \
    -dNOPAUSE -dQUIET -dBATCH -sOutputFile="$tmp" "$f"
  if [ -s "$tmp" ] && [ "$(stat -c%s "$tmp")" -lt "$(stat -c%s "$f")" ]; then
    mv "$tmp" "$f"
  else
    rm -f "$tmp"
  fi
done
```

Quality presets if `/ebook` figures look too soft: `/printer` (about 300 DPI, larger) or `/screen` (about 72 DPI, smaller). Prefer `/ebook` by default.
