# Agent notes

## Current direction

**Adopted 21 September 2026:** SafetyDial studies which additional experience makes
an existing LeWM's predictions of unsafe outcomes more reliable at the same interaction
budget, and whether the improvement transfers to new hazards, starts and goals.

Read [docs/researchDirection.md](docs/researchDirection.md) first. It is the research
authority. Use [the pilot checklist](docs/research/pilot.md) for the next work and
[the checkpoint reference](docs/research/checkpoints.md) for verified assets and interfaces.
The literature boundary is in [related work](docs/research/relatedWork.md).

- **Working title:** Which Experience Makes LeWM Safer to Use?
- **Student:** Sachin Mohan (2699183), BSc Honours CS, University of the Witwatersrand.
- **Supervisor:** Geraud Nangue Tasse.
- **Initial task/model:** Push-T with the released `pusht/lewm` checkpoint, 192-d latents.
- **Main constraint:** the whole T-shaped block footprint avoids a specified virtual region.
- **Preferred repair:** freeze the encoder, observation projector and physical readout;
  adapt predictor-side modules using new transitions and a fixed replay mixture.
- **Core comparison:** ordinary supported experience versus predicted-boundary coverage;
  add learned optimistic-error acquisition only after demonstrating repairability.
- **Extensions:** transfer, then optional closed-loop Safe-CEM/penalty-CEM and a second task.
- **Scope:** thesis in November 2026; possible ICLR 2027 workshop. Check the actual call
  before citing deadlines. No publication outcome or GPU runtime is assumed.

## Research gates and controls

1. Recover the baseline assets, validate reset-and-prefix replay and whole-T geometry.
2. Separate dense-event sampling, real-image readout and imagined-dynamics errors.
3. Show that extra experience repairs the diagnosed error with one fixed mechanism.
4. Compare acquisition strategies at equal charged simulator-step and training budgets.
5. Evaluate untouched trajectories, starts/goals and hazard layouts; retain useful progress.
6. Attempt closed-loop control only after the offline effect survives.

- The question concerns **additional adaptation experience**. The released checkpoint
  cannot establish what caused its original representation to form.
- A frozen encoder does not learn new safety features. Better readouts, better imagined
  dynamics and representation changes support different claims.
- Use the same frozen physical readout before/after predictor-side adaptation. If an
  encoder is adapted later, fit equal-capacity probes with the same independent protocol;
  old-probe failure can reflect feature-coordinate changes rather than lost information.
- Freeze encoder/projector running statistics as well as gradients. Cache detached target
  features, keep preprocessing/scalers fixed and declare which predictor-side modules train.
- Fix architecture, replay ratio, training steps and evaluation across acquisition arms.
  Restart from the same pretrained predictor at each data-budget comparison.
- Prospective selection cannot inspect the unqueried actual future. Retrospective
  false-safe selection is an oracle data-usefulness reference, not interaction efficiency.
- Count root generation, replay prefixes, discarded trials and every queried branch.
  Equal branch counts only mean equal interaction cost when prefix/horizon lengths match.
  Report model-query and training cost separately.
- Split whole source trajectories before roots/branches; siblings stay in one split.
  Keep probe, adaptation, development, calibration and final-test roles distinct.
- Reusing a trajectory under new virtual hazards creates more labels, not more independent
  experience. Keep that relabelling policy equal across methods.
- False-safe acceptance is `unsafe accepted / all accepted`. Report counts and acceptance
  rate; zero accepted plans means undefined. Group uncertainty by root/source trajectory.
- Report task progress, hazard exposure, arena exits and censored outcomes. Never hide
  an all-infeasible fallback or score an unseen future as safe.
- A simulator pose used for labels is privileged supervision. A cost/failure label is
  direct safety supervision. Neither is label-free discovery of safety.
- Include simple physical/action checks, persistence, a coordinate-dynamics reference,
  fixed margins and a readout-only correction where appropriate. Label information advantages.
- Literature already covers failure readouts, data selection and model adaptation. A new
  cost head, margin or fine-tuning run is not automatically novel; recheck primary literature.

The previous locomotion-first selection-amplification audit and irreversibility proposals
are superseded. New locomotion training, recovery labels, failure engineering, NSGA-II,
HJ filtering, conformal guarantees and encoder pretraining are not core dependencies.
Do not silently restore those requirements.

## Preserved experiments and evidence

See [experiments/README.md](experiments/README.md) and [notes/README.md](notes/README.md).
The Push-T probe/penalty/Safe-CEM pilot and Phase 0 locomotion triage are completed work.
The newly adopted acquisition study has not run; no block-pose readout, verified Push-T
branch replay, acquisition runner or predictor-adaptation runner exists yet.

The original penalty collapse was confounded by the starting position inside the inflated
hazard. Corrected penalty-CEM worked substantially better. The usable Safe-CEM dials had
zero observed true-box violations in nine episodes, not a guarantee. A high-dial arena
failure was repaired using commanded-action geometry. Use [notes/safeCEM.md](notes/safeCEM.md)
for variant provenance; penalty lambda can change at inference without retraining.

The historical pusher probe used 100k frames with a random frame split. Its low readout
error is not a trajectory-independent validation for the new study. Imagined pusher radial
error at horizon five had median 10.2 px and p95 35.5 px; those old geometry-specific tails
are not a calibrated margin for block-footprint predictions.

Phase 0 remains evidence that `terminated` is not irreversibility. Failed finite recovery
search is not an impossibility certificate. Its data, code, reports and videos remain intact.
Historical notes may contain superseded interpretations; the current plan controls new work.

## Repository map and authority

1. `docs/researchDirection.md`: adopted scope and protocol.
2. `docs/research/`: current execution checklist, checkpoint audit and related work.
3. Experiment code, notebooks and recorded results: evidence of what actually ran.
4. `readme.md` and `AGENTS.md`: concise navigation and engineering guidance.
5. Submitted academic deliverables: historical records, not current research requirements.

- `experiments/*.ipynb`: completed exploration and Push-T pilots, retain outputs.
- `experiments/helpers/`: reusable implementation; there is **no `src/` package**.
- `experiments/scripts/`: existing runnable studies and future small entry points.
- `notes/`: experimental findings and data references, not competing project plans.
- `docs/safeDial/`, `docs/phase0/`: reports, LaTeX, figures and recorded results.
- `animations/`: preserved videos/GIFs and reproduction instructions.
- `docs/papers/`: reference library; filenames `CEM-EVO.pdf` and `MPC-RCE.pdf` refer
  to CEM-RL and robust constrained CEM respectively. UNISafe has two historical copies.
- `docs/*/submitted/`, `docs/*/guides/`: submitted work and assignment guidance.
- `configs/download/`: `pusht` default, optional `cube` and `all`.
- `data/`, `third_party/`: ignored local assets. Presence must be checked on the run machine.

Superseded proposals, brainstorming whiteboards and duplicated research drafts were
removed during the 21 September cleanup. Tracked history is available in git. Do not
recreate competing authorities under `reports/` or `research_notes/`.

## Checkpoints, data and Push-T gotchas

The official release lists Push-T, Cube, Reacher and TwoRooms. Only Push-T has pilot
validation here; the downloader handles Push-T/Cube. Other tasks require their own model,
normalisation and action-interface validation. No locomotion release was verified.

Expected local layout:

```text
data/raw/pusht_expert_train.h5.zst
data/processed/pusht_expert_train.h5
data/stablewm/hf_pusht/                    # public config and weights
data/stablewm/checkpoints/pusht/lewm_object.ckpt
data/probes/                             # historical pusher cache
third_party/le-wm/
```

The 21 September audit found those assets absent locally. Recover historical weights,
source revision, fitted scalers and a small replay subset if available. Public weights
alone do not include the fitted normalisers; existing code fits them on expert data.
Push-T weights are about 72 MB and expert data about 13 GB compressed. Cube is optional,
not a prerequisite. The downloader stages decompression in `.part` and renames on success.

- Load `swm.policy.AutoCostModel("pusht/lewm")` once outside episode loops.
- Assign `STABLEWM_HOME`, do not `setdefault`; use the project's expected model directory.
- Use LeWM's own image preprocessor, 224x224, with correct channel ordering. Import
  `hdf5plugin` before compressed HDF5 reads.
- Preserve fitted action/proprio/state scalers and goal duplicates. Changing normalisation
  during acquisition would confound the comparison.
- `PlanConfig(horizon=5, receding_horizon=1, action_block=5)` matches this checkpoint.
  Each block packs **five sequential 2-d actions**, not necessarily a repeated action.
  The inspected 10 Hz setup gives 0.5 seconds/block and 2.5 seconds/five-block horizon.
- The 7-d dataset state is pusher xy, block xy/theta and pusher vx/vy. It omits block
  velocities. `_set_state` also advances a physics substep. Neither saved pose nor rendered
  animation is proof of dynamic replay. Start with verified reset-and-action-prefix replay.
- Whole-T safety needs actual shape polygons plus predicted centre and periodic angle.
  A safe centre is not a safe footprint. Check every environment step inside action blocks;
  do not confuse this with continuous-time collision certification.
- Seed numpy and torch as well as `CEMSolver`. Audit the solver's actual returned action
  sequence: an elite mean can be infeasible even when individual elites were feasible.
- The arena is about 0..512 with image-style y down; rendering is 224 px, scale 224/512.
  Pusher exits invalidate observation-based prediction. Preserve them as bad outcomes.
- The historical Push-T `success` flag was uninformative, so report block coverage and
  pose error. `EVAL_BUDGET=50` in the pilot is historical, not a chosen new-study budget.
- Read contiguous HDF5 chunks instead of scattered fancy indexing. Dataset schema:
  [notes/pushTDataExp.md](notes/pushTDataExp.md).

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
| `scripts/download_data.py` | Clone `third_party/le-wm`, then Hydra Hub download. Default `--config-name pusht`. `cube` and `all` (Push-T plus Cube) are explicit alternatives. `weights_only=true` skips datasets; `clone_source=false` skips the git clone |

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

## Engineering

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
