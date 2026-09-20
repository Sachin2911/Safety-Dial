# Agent notes

## TLDR

This Honours project asks whether the **failure set** a safe-RL agent needs can be **derived from dynamics instead of labelled by a human**. The criterion is **irreversibility**: a transition is unsafe when the agent cannot get back from it. That is a property of the transition structure, needs no annotation, and is only computable with a world model (it depends on rollouts of actions that were never taken). The work builds **return-reachability estimators in the latent space of a JEPA world model (LeWM / SIGReg)**, a substrate where reachability analysis has not been attempted before, and then exposes the resulting conservatism as the **Safety Dial**: a threshold an operator sets at deployment rather than an engineer fixing it during training.

**Project name:** SafetyDial. The dial is now the **reachability threshold `d`**, not a Pareto front.

> **Direction change (August 2026).** The project pivoted from *evolutionary multi-objective planning* (NSGA-II Pareto fronts over imagined plans) to *label-free latent reachability*. Most LaTeX documents in `docs/` still carry the old title and framing. The authority on the current plan is `SafetyDial_Proposal.pdf` at the repo root. See [Doc authority](#doc-authority) and [Legacy framing](#legacy-framing-what-changed).

## Project snapshot

- **Name:** SafetyDial
- **Current title:** Label-Free Reachability in Joint-Embedding Predictive Architectures for Safe Reinforcement Learning. Subtitle: The Safety Dial: Irreversibility as a Deployment-Time Control
- **Former working titles:** Evolutionary Multi-Objective Planning in JEPA (EMOP in JEPA); SafetyDial: Selection Pressure for Safe Latent Planning in JEPA. Both are legacy
- **Student:** Sachin Mohan (2699183), BSc Honours CS, University of the Witwatersrand
- **Supervisor:** Geraud Nangue Tasse
- **Revised proposal:** `SafetyDial_Proposal.pdf` (root), dated 16 August 2026. PDF only, no LaTeX source is in the repo
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

`newInfo.md` holds a shorter-horizon todo list written just before the proposal (prior-art week, then a Push-T go/no-go on forward-rollout dispersion, an ICLR-vs-thesis decision point, abstract by 11 Sep for an 18 Sep deadline, OpenReview admin). Where it disagrees with the proposal, the proposal wins: the proposal moves the primary environment to OGBench-Cube and demotes forward dispersion from the method to a comparison baseline.

## Doc authority

When sources conflict, follow this order:

1. **`SafetyDial_Proposal.pdf`** (root, 16 Aug 2026). The current plan: label-free reachability, irreversibility, Safety Dial as threshold
2. **`newInfo.md`** (root). Same direction, shorter horizon, written slightly earlier
3. **`notes/`** and the `experiments/` notebooks. What was actually run and what it showed
4. **`docs/ideation/`, `docs/researchProp/submitted/RP.pdf`, `docs/AB/`, `docs/litReview/`, `readme.md`.** All still describe the older frameworks (NSGA-II Pareto planning, and older still, EA for LLM alignment). Historical only

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

`pymoo` is still a declared dependency from the NSGA-II era. Nothing currently uses it.

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
| `notes/` | Short markdown findings. `pushTDataExp.md` is the Push-T HDF5 layout reference |
| `scripts/` | `setup.sh` (local or Vast), `download_data.py` (LeWM clone plus Hub weights and data), `helpers/_common.sh` |
| `configs/download/` | Hydra configs for `download_data.py`: `all` (default), `pusht`, `cube` |
| `docs/` | Deliverables and papers, see [Docs map](#docs-map) |
| `data/`, `third_party/` | Local artifacts, contents gitignored, `.gitkeep` tracked |
| `SafetyDial_Proposal.pdf` | The current proposal |
| `newInfo.md` | Direction statement plus near-term todo |

New Python that outgrows a notebook goes in `experiments/helpers/` unless we deliberately re-introduce a package. If a package comes back, `pyproject.toml` already points `pythonpath = ["src"]` and `testpaths = ["tests"]` at directories that do not currently exist.

### Known warts

- `experiments/.ipynb_checkpoints/` is **tracked in git** (three stale notebook copies). It should be gitignored and removed from the index.
- Notebooks are committed with outputs, so they run 2 to 5 MB each.
- `readme.md` still pitches "Safe AI via Evolutionary Algorithms" and titles the project EMOP.
- `docs/researchProp/latex/main.tex` is an empty section-heading shell; `docs/AB/latex/main.tex` and `docs/litReview/latex/main.tex` are titled "Evolutionary Multi-Objective Planning in JEPA".
- The revised proposal exists only as a PDF. There is no LaTeX source for it in the repo.
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
| `SafetyDial_Proposal.pdf` (root) | **Current** revised proposal. Highest authority |
| `newInfo.md` (root) | Current direction plus near-term todo |
| `docs/ideation/` | Ideation doc (`ID.pdf`). Legacy: selection-pressure / Pareto framing |
| `docs/researchProp/` | Original proposal. Frozen `submitted/RP.pdf`; living `latex/main.tex` is an empty shell. Superseded by the root PDF |
| `docs/AB/` | Annotated bibliography. Frozen `submitted/AB.pdf`; `whiteBoard.md` holds the 7-paper list |
| `docs/litReview/` | Literature review. Frozen `submitted/LR.pdf`. Oldest framing (EA / LLM alignment) |
| `docs/papers/geraudsPapers/` | Supervisor-suggested papers, each with a `.md` text dump beside the PDF |
| `docs/papers/myPapers/` | Student-chosen papers (PDF). Only `LeWorldModel.md` has a dump, in `docs/papers/myPapersMd/` |
| `docs/projectPresentation/`, `docs/projectReport/` | Placeholders for later deliverables |
| `readme.md` | Setup quickstart plus a stale pitch |

Each `docs/<deliverable>/` follows the same shape: `latex/main.tex` plus `references.bib`, `compile.sh` to build (`./compile.sh` builds, `./compile.sh clean` removes artifacts), intermediates in `latex/build/` (gitignored), final PDF at the folder root, `submitted/` holding the frozen handed-in version.

## Reading list

Core to the current direction:

| Paper | Where | Role |
|-------|-------|------|
| Generalizing Safety Beyond Collision-Avoidance (Nakamura, Peters, Bajcsy 2025, arXiv 2502.00935) | `myPapers/LatentSafetyFilters.pdf` | The framework being modified: HJ reachability in a learned latent, seeded by labelled failures |
| UNISafe: Uncertainty-aware Latent Safety Filters (Seo, Nakamura, Bajcsy 2025, arXiv 2505.00779) | `myPapers/unisafe.pdf` **and** `myPapers/UncertaintyAwareLatentSafety.pdf` (same paper, committed twice) | The supervised baseline for Stage 3 |
| AnySafe (arXiv 2509.19555, ICRA 2026) | not yet in repo | Runtime-adjustable constraints via conformal similarity. Closest prior work on the "dial" idea |
| No Turning Back (Grinsztajn et al. 2021) | not yet in repo | Precedence-classifier reversibility estimation. Basis of the second estimator |
| LeWorldModel (LeWM, Maes et al. 2026) | `myPapers/LeWorldModel.pdf` (+ `myPapersMd/LeWorldModel.md`) | The frozen JEPA world model. Also the source of "physical quantities are linearly probeable" |
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
| ROSARL | `geraudsPapers/ROSARL.pdf` (+ `.md`) | Supervisor work. Derives a sufficient penalty from intrinsic quantities, still one scalar fixed before deployment |
| Safety-Gymnasium | `geraudsPapers/safetyGym/SafetyGym.pdf` (+ `.md`) | Standard benchmark. **Explicitly rejected** as the evaluation environment: its hazards are recoverable |
| OmniSafe | `myPapers/omniSafe.pdf` | Safe-RL implementations |
| DQN | `myPapers/DQN.pdf` | Background |
| MAP-Elites, DQD-RL | `geraudsPapers/IlluminatingSearchSpacesByMappingElites.pdf`, `ApprxGradsForDiffQDinRL.pdf` (+ `.md`) | Legacy QD track from the evolutionary framing |

## Environment and tooling

- **Package manager:** `uv`. Python pinned to **3.11** via `.python-version`.
- **Sync:** `uv sync --frozen --extra dev` (what `setup.sh` runs). Commit `uv.lock` when dependencies change.
- **Run anything:** `uv run python ...`, `uv run pytest`, `uv run ruff check .`
- **Direct deps:** `stable-worldmodel[env,train]` (the LeWM stack: `swm.World`, `swm.policy.AutoCostModel`, `CEMSolver`, `HDF5Dataset`), `torch`, `gymnasium`, `numpy`, `hydra-core` plus `omegaconf`, `huggingface_hub`, `h5py`, `zstandard`, `hdf5plugin`, `wandb`, `tqdm`, `pymoo` (legacy). Dev extras: `pytest`, `ruff`, `ipykernel`.
- **Transitive but used directly in notebooks:** `scikit-learn`, `matplotlib`, `scipy`, `stable_pretraining`, `gym-pusht`, `ogbench`. They arrive via `stable-worldmodel`; if a notebook starts leaning on one, promote it to an explicit dependency.
- **Lint:** ruff, `line-length = 100`.
- **Not installed as a package** (`[tool.uv] package = false`).
- Notebooks import helpers as `import helpers.linProbeHelpers as lph`, so run JupyterLab with `experiments/` as the working directory. `setup.sh` registers a kernel named **"Python (safetydial .venv)"** (`safetydial-venv`); select it, and refresh the page if it does not appear.

## Setup scripts and remote GPU

| Script | Use |
|--------|-----|
| `scripts/setup.sh` | Local or Vast setup. Auto-detects Vast. `--pull` updates the clone first. Piped or pasted (Vast On-start) it clones or pulls, then re-runs from the repo |
| `scripts/helpers/_common.sh` | Shared helpers (do not run directly): dotenv loading, uv install, system deps (`swig` for box2d, `btop`), Vast Codex install/login, git identity, HTTPS push token, venv sync, Jupyter kernel, HF login, sanity check |
| `scripts/download_data.py` | Clone `third_party/le-wm`, then Hydra Hub download. Default `--config-name all` (Push-T plus Cube weights and expert data). `pusht` / `cube` for one task. `weights_only=true` skips datasets; `clone_source=false` skips the git clone |

Idempotent: `git pull && bash scripts/setup.sh` (or `bash scripts/setup.sh --pull`) is the normal refresh. Vast hosts should have `cuda_max_good>=13.0` so the torch CUDA build works. Real runs happen on Vast at `/workspace/Safety-Dial`; the local `.venv` is usually only partially synced.

On Vast, setup installs Codex CLI if missing using the official standalone installer, without interactive prompts. It reuses an existing installation. With `OPENAI_API_KEY` injected, it saves API-key authentication through stdin, replacing any cached login. Without the key, it preserves existing authentication or prints `codex login --device-auth` for manual ChatGPT login. API-key usage is billed separately from ChatGPT subscription usage. Local setup does not install or change Codex authentication.

## Secrets

Secrets live in a gitignored `.env` at the repo root, loaded by `_common.sh` and by `download_data.py`; on Vast they also come from account Environment Variables. Recognised keys:

- `WANDB_API_KEY` (logging)
- `GITHUB_TOKEN` (optional; enables push over HTTPS)
- `HF_TOKEN` (optional; Hugging Face downloads)
- `OPENAI_API_KEY` (optional; automatic Codex API-key login during Vast setup, with separate API billing)
- `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL` (repo-local git identity, not global)
- `STABLEWM_HOME` (written automatically by `download_data.py`)
- `LEWM_REPO_URL`, `LEWM_GIT_REF` (optional overrides for the LeWM clone)

Rules: never commit `.env`, never print token values into logs or terminal output, and do not put them in configs or code. There is no tracked `.env.example`; document new keys here instead.

Codex stores authentication in its private credential cache (normally `~/.codex/auth.json` on headless Linux). Setup excludes `OPENAI_API_KEY` from its copies into `/etc/environment`; subsequent SSH sessions use the cached login. Never commit or print the Codex authentication cache.

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
- **Safety-Gymnasium is rejected** for evaluation (recoverable hazards). OGBench-Cube is the primary candidate.
- Scope claims to **irreversible failures**, and state that limitation explicitly rather than defending it. Irreversibility is not the same thing as danger.
- Do not skip the two Stage 2 controls (joint limits, horizon sweep) or the Stage 3 dataset-composition check. Without them the results are confounded or void.
- Every probe use must be labelled as supervision of a **generic physical quantity**, never of the safety concept.
- This is an active area with a high preprint rate. Check arXiv and OpenReview before claiming novelty.

### Engineering

- Run everything through **`uv run`**; do not `pip install` into the venv or create a second environment.
- New code goes in `experiments/helpers/` (or a notebook). There is no `src/` package; do not resurrect one casually.
- Configs go in `configs/<group>/` as Hydra groups.
- Never commit secrets, checkpoints, datasets, run outputs, or `wandb/`.
- Compress any new PDF under `docs/papers/` before committing (see below).
- Follow the writing rules below (no em dashes).

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
