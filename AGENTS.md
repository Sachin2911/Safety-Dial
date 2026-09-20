# Agent notes

## TLDR

**Current direction, adopted 20 September 2026:** SafetyDial asks whether planner
optimisation amplifies optimistic constraint-prediction errors in JEPA-style world
models, and whether a simple correction reduces those errors while preserving task
performance. Accept Safety-Gymnasium's supplied costs. The core is an offline audit,
followed by one small repair; closed-loop Safe-CEM is an extension.

**Read [the adopted research plan](docs/researchDirection.md) first.** It supersedes the
irreversibility proposal in `docs/revisedProp/` and all earlier planning notes.
The project is now **supervised constraint prediction and safe planning**, with
programmatic benchmark labels. It no longer requires discovering safety from
irreversibility. The Safety Dial is an empirical deployment conservatism margin.

## Project snapshot

- **Name:** SafetyDial
- **Working title:** When Are JEPA Predictions Reliable Enough for Safe Planning?
- **Student:** Sachin Mohan (2699183), BSc Honours CS, University of the Witwatersrand
- **Supervisor:** Geraud Nangue Tasse
- **Authority:** `docs/researchDirection.md`, adopted 20 September 2026
- **Environment:** Safety-Gymnasium locomotion, initially Walker2d if a competent
  actor, usable data and observable velocity are available
- **Minimum contribution:** paired actual-versus-imagined constraint decisions,
  comparing ordinary, adapted and planner-selected action sequences
- **First extension:** one horizon-dependent margin or a diagnosed auxiliary
  future-velocity loss, with matched data and useful-progress comparisons
- **Later extensions:** closed-loop Safe-CEM and replication on a second robot
- **Initial model:** frozen visual features plus a small history-conditioned
  action predictor. New locomotion predictor training is required; released
  Push-T/Cube checkpoints are not assumed transferable
- **Scope:** thesis in November 2026; a possible ICLR 2027 workshop remains the
  target. Check the actual CFP before citing dates. No main-track submission is
  planned. Aim to freeze workshop results by mid-January

### Research question and gates

Does planner optimisation amplify optimistic safety errors in predictive latents,
and can a simple correction reduce them without sacrificing task performance?

1. Verify a competent nominal policy, simulator replay, transition-aligned labels,
   cost-boundary coverage and constraint observability.
2. Audit real encoded futures versus imagined futures with the same readout.
3. Compare initial proposals, adapted CEM candidates and the returned plan on paired
   simulator roots, controlling horizon, predicted margin and search budget.
4. Test one correction against persistence, a reactive guard, a state-dynamics
   predictor and fixed-margin references.
5. Attempt closed-loop intervention only if the offline evidence supports it.

False-safe acceptance is `unsafe accepted / all accepted`; report counts and
acceptance rate, and mark it undefined when none are accepted. Group uncertainty
estimates by trajectory or root. Report useful progress and falls alongside costs.

### What changed

The August plan used label-free return-reachability and precedence estimation.
The superseded September locomotion proposal assumed falling was irreversible.
Phase 0 found recoveries and strong predicate/horizon sensitivity, so that premise
cannot support the original experiment. The latest direction accepts the benchmark's
safety specification and studies prediction reliability under planning.

The older NSGA-II/Pareto and LLM-alignment directions are also historical.
Irreversibility, Cube failure engineering, exact Sokoban diagnostics, quasimetric
learning, failure-mode label holdout, full HJ filtering and new conformal guarantees
are not core requirements. Do not silently reintroduce them.

## Doc authority

1. **`docs/researchDirection.md`**: adopted question, hypotheses, method, gates and scope.
2. **`notes/`, experiment code and recorded results**: evidence of what actually ran.
   Prefer the latest correction and recorded provenance over an older narrative.
3. **`readme.md` and this file**: concise current guidance and engineering conventions.
4. **`docs/revisedProp/` and `docs/*/submitted/`**: superseded proposals and graded
   deliverables. Preserve the historical files, but do not use their research
   requirements to override the adopted plan.

## Where things stand

Completed work includes the Push-T probe/Safe-CEM pilot and Phase 0 locomotion triage. The new constraint-prediction audit has not yet run. The pilot uses the released `pusht/lewm` checkpoint (192-d latents); its infrastructure carries over, not its weights or action semantics.

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

70.7 px is the start-to-goal block distance, so the block did not move in that original sweep. **This is a confounded historical result, not clean evidence of penalty fragility.** The start lay inside the inflated hazard; corrected geometry made penalty-CEM substantially more effective. The later Safe-CEM pilot found zero true-box violations in nine usable-dial episodes, identified an off-screen arena failure, and validated an action-space arena correction. Read `notes/safeCEM.md` for the corrected results and variant provenance. Zero observed violations is not a guarantee.

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
| `experiments/helpers/oracle.py` | Phase 0 recovery-search evaluator. Historical infrastructure; not a required training target for the adopted plan |
| `experiments/helpers/equivCheck.py` | Vendored-env equivalence gates G1..G9. Deliberately numpy-only and torch-free |
| `notes/` | Short markdown findings. `pushTDataExp.md` is the Push-T HDF5 layout reference, `safetyGymLocomotion.md` the measured Safety-Gymnasium facts, `terminationIsNotIrreversibility.md` the headline Phase 0 result |
| `scripts/` | `setup.sh` (local or Vast), `download_data.py` (LeWM clone plus Hub weights and data), `helpers/_common.sh` |
| `configs/download/` | Hydra configs for `download_data.py`: `all` (default), `pusht`, `cube` |
| `docs/` | Deliverables and papers, see [Docs map](#docs-map) |
| `data/`, `third_party/` | Local artifacts, contents gitignored, `.gitkeep` tracked |
| `docs/researchDirection.md` | **Adopted research plan.** Constraint prediction, planner selection and one repair |
| `docs/revisedProp/` | Superseded irreversibility proposal, retained as LaTeX and PDF |

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

**Historical Cube storage issue:** the earlier irreversibility plan required a large Cube dataset. That download is not a prerequisite of the adopted locomotion audit. The default download config still pulls Cube; use task-specific or weights-only options deliberately and check free disk space.

`decompress_dataset` writes to a `.part` path and renames only on success, so an interrupted or out-of-disk decompress no longer leaves a truncated file that `processed_ready()` would treat as complete and skip.

Model ids are `pusht/lewm` and `cube/lewm`, loaded as `swm.policy.AutoCostModel("pusht/lewm")`.

Push-T dataset shape lives in `notes/pushTDataExp.md`: flat timestep arrays, 2,336,736 steps over 18,685 episodes, `ep_offset` plus `ep_len` slicing, 7-d `state` (pusher xy, block xy, block theta, pusher vx vy), 224x224x3 `pixels`. Arena coordinates are roughly 0 to 512 but can leave that range.

## Working with LeWM and Push-T (gotchas)

Collected from the notebooks and helpers. These cost real debugging time; do not rediscover them.

- Notebooks find the repo by walking up for `scripts/download_data.py`, then put `third_party/le-wm` on `sys.path` to import `utils.get_img_preprocessor` and `get_column_normalizer`, then set `STABLEWM_HOME` to `data/stablewm` (assign, do not `setdefault`).
- `import hdf5plugin` is required before reading the compressed HDF5, even though nothing references it directly.
- Feed LeWM through **its own** preprocessor (`get_img_preprocessor(source, target, img_size=224)`). Raw env frames will not do. `HDF5Dataset` order is `(1, H, W, C)` permuted to `(1, C, H, W)` before the preprocessor.
- The planner needs a `process` dict of fitted `StandardScaler`s for `action`, `proprio`, `state`, plus `goal_` duplicates for the non-action columns. Fit on `dataset.get_col_data(col)` with NaN rows dropped.
- For the released Push-T checkpoint, `swm.PlanConfig(horizon=5, receding_horizon=1, action_block=5)` matches its frameskip. **`action_block` must stay 5 for that checkpoint**; newly trained locomotion models need their own recorded time/action-block settings.
- `CEMSolver(seed=...)` does not control every RNG. Seed `torch` and `numpy` globally as well, or two runs at the same nominal seed diverge.
- Load `AutoCostModel` **once** outside any episode loop. Reloading LeWM per episode dominated runtime.
- `EVAL_BUDGET = 50` matches the le-wm Push-T eval; `World(..., max_episode_steps=2*EVAL_BUDGET)`.
- Random fancy-indexing into the uncompressed HDF5 is very slow. Read contiguous chunks at spread-out offsets instead (the notebooks take 100 chunks of 1000).
- Arena is 512 px with image-style coordinates (y down, origin top-left). Rendered frames are 224 px, so the arena-to-image scale is `224/512`.

## Docs map

| Path | Role |
|------|------|
| `docs/researchDirection.md` | **Current plan**, adopted after Phase 0 and Safe-CEM on 20 Sep 2026 |
| `docs/revisedProp/` | Superseded irreversibility proposal. `./compile.sh` rebuilds the historical PDF |
| `docs/ideation/` | Frozen `submitted/ID.pdf` plus `whiteBoard.md`. Legacy framing. LaTeX source removed 20 Sep 2026 |
| `docs/researchProp/` | Original proposal, frozen `submitted/RP.pdf`. Superseded by `docs/revisedProp/`. LaTeX source removed 20 Sep 2026 |
| `docs/AB/` | Annotated bibliography, frozen `submitted/AB.pdf`; `whiteBoard.md` holds the 7-paper list. LaTeX source removed 20 Sep 2026 |
| `docs/litReview/` | Literature review. Frozen `submitted/LR.pdf`. Oldest framing (EA / LLM alignment) |
| `docs/papers/myPapers/` | Student-chosen papers (PDF). 15 files |
| `docs/projectPresentation/`, `docs/projectReport/` | Placeholders for later deliverables |
| `readme.md` | Front door: the reachability pitch, setup quickstart, deliverable checklist |

`docs/revisedProp/`, `docs/phase0/` and `docs/safeDial/` carry LaTeX sources. The shape is `latex/main.tex` plus `references.bib`, `compile.sh` to build (`./compile.sh` builds, `./compile.sh clean` removes artifacts), intermediates in `latex/build/` (gitignored), final PDF at the folder root. The older deliverables were reduced to `submitted/` (the frozen handed-in PDF) plus `guides/` and `whiteBoard.md` on 20 September 2026; their sources all carried the superseded title.

Building needs a LaTeX toolchain, which is not in the base image. On a fresh box:

```bash
apt-get install -y --no-install-recommends texlive-latex-recommended texlive-latex-extra \
  texlive-fonts-recommended lmodern latexmk biber texlive-bibtex-extra
```

## Reading list

Prior-work and substrate references (see the adopted plan for current positioning):

| Paper | Where | Role |
|-------|-------|------|
| Generalizing Safety Beyond Collision-Avoidance (Nakamura, Peters, Bajcsy 2025, arXiv 2502.00935) | `myPapers/LatentSafetyFilters.pdf` | Background on learned latent safety filters; full HJ filtering is outside the core plan |
| UNISafe: Uncertainty-aware Latent Safety Filters (Seo, Nakamura, Bajcsy 2025, arXiv 2505.00779) | `myPapers/unisafe.pdf` **and** `myPapers/UncertaintyAwareLatentSafety.pdf` (same paper, committed twice) | Prior uncertainty-aware filtering, including SafeOnly; do not describe every variant as requiring failure labels |
| AnySafe (arXiv 2509.19555, ICRA 2026) | not yet in repo | Runtime-adjustable constraints via conformal similarity. Closest prior work on the "dial" idea |
| No Turning Back (Grinsztajn et al. 2021) | not yet in repo | Historical precedence/reversibility reference from the superseded plan |
| LeWorldModel (LeWM, Maes et al. 2026) | **Notion** (removed from repo, commit e631dc0) | Pilot substrate and model implementation reference; physical-state probes do not establish imagined constraint accuracy |
| LeJEPA / SIGReg (Balestriero and LeCun 2025, arXiv 2511.08544) | `myPapers/LeJEPA.pdf` | Why the latent is isotropic, which is what makes latent distance well posed |
| Hamilton-Jacobi reachability (Bansal et al. 2017) | not yet in repo | Historical reachability background, not a required implementation |
| OGBench (Park et al. 2025) | not yet in repo | Source of the Cube environment considered in the superseded irreversibility plan |
| PSG-JEPA (arXiv 2608.06799) | linked in `docs/researchDirection.md` | Physical grounding prior work relevant to auxiliary velocity supervision |
| The Intervention Gap in Latent World Models (arXiv 2608.29998) | linked in `docs/researchDirection.md` | Close prior work separating decodability from imagined action effects |

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
| Safety-Gymnasium | **Notion** (removed from repo, commit 8280a44) | Current benchmark family. The adopted study accepts its supplied cost constraints |
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

- **SafetyDial:** Deployment conservatism for predicted constraint decisions. The initial
  proposal uses `predicted_velocity_h + d * m_h <= v_max`, with dimensionless dial `d`
  and a held-out empirical velocity margin `m_h`.
- **False-safe acceptance:** Actually unsafe accepted plans divided by all accepted
  plans. Report the denominator; undefined if none are accepted.
- **Selection amplification:** Additional optimistic error associated with adapted or
  selected plans, evaluated against ordinary candidates under declared controls.
- **JEPA:** Joint-Embedding Predictive Architecture. Predicts embeddings rather than pixels.
- **LeWM:** LeWorldModel, the existing Push-T/Cube substrate. A new frozen-feature
  locomotion predictor is not automatically an end-to-end LeWM experiment.
- **Probe:** Small readout of a generic physical quantity. A velocity label is physical
  supervision; a cost label is direct safety supervision. Declare which is used.
- **Safe-CEM:** Feasibility-first candidate ranking. It does not certify model accuracy,
  feasibility of an averaged solver output, or safe fallback behaviour.
- **UNISafe / AnySafe / SafeDreamer:** Relevant uncertainty, latent filtering and
  world-model safe-control prior work. See the current plan's source links.
- **Irreversibility / return-reachability / precedence:** Concepts from the superseded
  label-free direction, retained for interpreting Phase 0 and earlier proposals.
- **HJ reachability / conformal calibration:** Potential later machinery, not required
  for the core audit. Offline calibration does not establish closed-loop guarantees.
- **CEM:** Cross-Entropy Method; used for candidate optimisation.
- **swm:** `stable_worldmodel`, providing environment, model and planning interfaces.
- **uv / Vast.ai:** Project environment manager and remote GPU provider.
- **EMOP / NSGA-II / hypervolume / IGD / MAP-Elites:** Historical evolutionary framing.

## Agent must-know

### Research

- Follow `docs/researchDirection.md`: accepted benchmark costs, offline predictive
  audit, one small repair, optional intervention.
- The contribution sought is planner-selected false-safe error and a useful
  correction. A cost head, auxiliary physical loss or runtime dial alone is not
  automatically novel. Check arXiv and OpenReview before novelty claims.
- Supervision is explicit. Velocity probes use generic physical-quantity labels;
  cost heads use safety labels. Neither is label-free safety discovery.
- Pin the cost implementation, timestep, observation history and camera. Walker2d
  currently uses signed forward velocity; do not silently change it to absolute speed.
- The collector records state before action. Align returned cost/reward with that
  transition and next observation. Split complete trajectories before sampling roots.
- Termination-disabled collection belongs to explicit continuation diagnostics.
  Standard benchmark evaluation retains declared termination/time-limit semantics;
  report early falls and censored horizons, never treat missing steps as safe.
- Freeze models for final evaluation and dial sweeps. Locomotion predictor training
  is permitted and required unless a compatible released checkpoint is verified.
- Use a competent nominal actor. Do not make CEM learn walking from scratch.
- Validate replay and velocity observability before large data collection/training.
- Compare actual encoded versus imagined futures and initial versus adapted versus
  selected candidates. Audit the actual returned action sequence, not only an elite.
- False-safe acceptance requires its denominator and acceptance rate. No accepted
  plans means undefined FSA, not zero. Include all-infeasible/fallback outcomes.
- Compare with persistence, a concrete reactive guard and a state-model reference.
  Declare privileged information and preserve useful progress in comparisons.
- Horizon margins are empirical. Neither offline quantiles, feasibility-first ranking,
  nor zero observed violations establishes a closed-loop safety guarantee.
- Phase 0 recovery search remains historical evidence/evaluation infrastructure.
  `terminated` is not irreversibility, and failed finite search is not an impossibility
  certificate. Recoverability labels are not the new method's training target.
- The original Push-T penalty collapse was confounded; use `notes/safeCEM.md`.
  Penalty-CEM can change lambda without retraining. Do not conflate it with
  separately trained penalty-RL policies.

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
