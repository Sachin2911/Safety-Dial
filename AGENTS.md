# Agent notes

## TLDR

This Honours project asks whether **selection pressure can substitute for reward engineering** in safe RL. A frozen JEPA world model (LeWM) imagines plans in latent space; a multi-objective evolutionary planner (NSGA-II) trades off task performance against structural safety signals from the model itself, returning a Pareto front so the safety-performance operating point is chosen at deployment rather than fixed before training.

**Project name:** SafetyDial. The dial is the Pareto front: you choose the safety-performance operating point at deployment instead of locking in a λ penalty before training.

## Project snapshot

- **Name:** SafetyDial
- **Title:** SafetyDial: Selection Pressure for Safe Latent Planning in JEPA
- **Former working title:** Evolutionary Multi-Objective Planning in JEPA (EMOP in JEPA); prefer SafetyDial in new writing
- **Slogan:** Selection pressure as an alternative to hand-crafted reward penalties
- **Student:** Sachin Mohan (2699183), BSc Honours CS, University of the Witwatersrand
- **Supervisor:** Geraud Nangue Tasse
- **Guiding move:** swap *tuning* (penalty weights, hand-crafted costs) for *search* (Pareto front over imagined plans); the front is the dial

### Method (four phases)

1. Reproduce LeWM on Reacher and Push-T; freeze the world model.
2. Replace CEM with NSGA-II (single-objective first, then multi-objective Pareto plans).
3. Add structural safety objectives: prediction surprise, latent energy/jerk, small learned probes (not a hand-crafted cost).
4. Evaluate against baselines; optional later: value bootstrapping, MAP-Elites on trajectory embeddings.

### Evaluation

- **Primary envs:** Reacher, Push-T. OGBench-Cube only if schedule allows.
- **Baselines:** vanilla CEM, penalty-tuned CEM, CPO / Lagrangian PPO, ROSARL.
- **Metrics:** hypervolume (HV), inverted generational distance (IGD), task success; violation-of-expectation study for surprise as a safety signal.

## Docs map

| Path | Role |
|------|------|
| `docs/ideation/` | Current distilled direction (living LaTeX / `ID.pdf`). Prefer this for ongoing writing. |
| `docs/researchProp/` | Full research proposal. Frozen content: `submitted/RP.pdf`. Living `latex/main.tex` may be a shell. |
| `docs/AB/` | Annotated bibliography. Frozen: `submitted/AB.pdf`. See `whiteBoard.md` for the 7-paper list. |
| `docs/litReview/` | Literature review. Frozen: `submitted/LR.pdf`. Older framing (EA / LLM alignment); do not treat as current method. |
| `docs/papers/geraudsPapers/` | Supervisor-suggested papers, each with a `.md` text dump alongside the PDF. |
| `docs/papers/myPapers/` | Student-chosen papers (PDF only, no `.md` dumps). |
| `docs/projectPresentation/`, `docs/projectReport/` | Placeholders for later deliverables. |
| `readme.md` | Project pitch plus Vast setup quickstart. Pitch text still reflects the earlier “Safe AI via Evolutionary Algorithms” framing. |

Each `docs/<deliverable>/` folder follows the same shape: `latex/main.tex` + `references.bib` sources, `compile.sh` to build (`./compile.sh` builds, `./compile.sh clean` removes artifacts), intermediates in `latex/build/` (gitignored), final PDF copied to the folder root, and `submitted/` holding the frozen handed-in version.

### Reading list

| Paper | Where | Role |
|-------|-------|------|
| LeWorldModel (LeWM) | `myPapers/LeWorldModel.pdf` | The frozen JEPA world model this project plans in |
| Hierarchical Planning with Latent World Models (HWM) | `myPapers/HierarchicalPlanningWithLatentWorldModels.pdf` | Closest prior work: latent MPC, single scalar objective |
| DINO-WM | `myPapers/DINO-WM.pdf` | World model on pre-trained visual features, zero-shot planning |
| SafeDreamer | `myPapers/SafeDreamer.pdf` | Safe RL inside a world model; comparison point |
| Uncertainty-aware Latent Safety Filters | `myPapers/UncertaintyAwareLatentSafety.pdf` | OOD failure avoidance in latent space; motivates surprise as a safety signal |
| URWM | `myPapers/URWM.pdf` | Uncertainty-aware robotic world model, offline model-based RL |
| ROSARL | `geraudsPapers/ROSARL.pdf` | Supervisor work; scalar Minmax penalty baseline |
| Safety-Gymnasium | `geraudsPapers/safetyGym/SafetyGym.pdf` | Safe RL benchmark suite |
| Illuminating Search Spaces by Mapping Elites | `geraudsPapers/IlluminatingSearchSpacesByMappingElites.pdf` | MAP-Elites; optional QD extension |
| Approximating Gradients for Differentiable QD in RL | `geraudsPapers/ApprxGradsForDiffQDinRL.pdf` | DQD-RL; optional QD extension |

## Repo layout

Code scaffolding exists but is currently empty (`.gitkeep` placeholders only). Put new code in the matching slot rather than inventing a new top-level folder.

| Path | Role |
|------|------|
| `src/safetydial/world_model/` | LeWM loading, freezing, latent rollout |
| `src/safetydial/planners/` | CEM baseline, NSGA-II planner, MPC loop |
| `src/safetydial/objectives/` | Structural safety signals (surprise, latent energy/jerk, probes) |
| `src/safetydial/envs/` | Reacher, Push-T, optional OGBench-Cube wrappers |
| `src/safetydial/baselines/` | Penalty-tuned CEM, CPO / Lagrangian PPO, ROSARL |
| `src/safetydial/metrics/` | Hypervolume, IGD, task success, violation counting |
| `src/safetydial/train/` | Training and evaluation entry points |
| `configs/` | Hydra configs: `download/` (Hub presets), plus planned `env/`, `world_model/`, `planner/`, `experiment/` |
| `scripts/` | Setup and launch shell scripts |
| `tests/` | Pytest suite |
| `third_party/` | Vendored upstream code (for example a LeWM checkout) |
| `data/`, `checkpoints/`, `runs/`, `experiments/` | Local artifacts; contents gitignored, `.gitkeep` tracked |

## Environment and tooling

- **Package manager:** `uv`. Python pinned to **3.11** via `.python-version`.
- **Sync:** `uv sync --frozen --extra dev` (what the setup scripts run). Commit `uv.lock` when dependencies change.
- **Run anything:** `uv run python ...`, `uv run pytest`, `uv run ruff check .`
- **Core deps:** `torch`, `gymnasium`, `numpy`, `hydra-core` + `omegaconf` (config), `pymoo` (NSGA-II and Pareto metrics), `wandb` (logging), `tqdm`. Dev extras: `pytest`, `ruff`.
- **Lint:** ruff, `line-length = 100`, sources `src` and `scripts`.
- **Tests:** pytest, `testpaths = ["tests"]`, `pythonpath = ["src"]`.
- **Not installed as a package** (`[tool.uv] package = false`). Pytest picks up `src` via `pythonpath`, but plain `uv run python` does not, so scripts need `PYTHONPATH=src` or an equivalent.
- Prefer `pymoo` for NSGA-II, hypervolume, and IGD rather than hand-rolling them.

## Setup scripts and remote GPU

| Script | Use |
|--------|-----|
| `scripts/setup.sh` | Local or Vast setup. Auto-detects Vast. `--pull` updates the clone first. Piped/pasted (On-start) clones or pulls then re-runs from the repo. |
| `scripts/helpers/_common.sh` | Shared helpers (do not run directly) |
| `scripts/download_data.py` | Clone `third_party/le-wm`, then Hydra Hub download. Default `--config-name all` (Push-T + Cube weights and expert data). `pusht` / `cube` for one task. `weights_only=true` skips datasets; `clone_source=false` skips the git clone. Configs: `configs/download/` |

Idempotent: `git pull && bash scripts/setup.sh` (or `bash scripts/setup.sh --pull`) is the normal refresh. Vast hosts should have `cuda_max_good>=13.0` so the torch CUDA build works.

## Secrets

Secrets live in a gitignored `.env` at the repo root, loaded by `_common.sh`; on Vast they also come from account Environment Variables. Recognised keys:

- `WANDB_API_KEY` (logging)
- `GITHUB_TOKEN` (optional; enables push over HTTPS)
- `HF_TOKEN` (optional; Hugging Face downloads)
- `GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL` (set repo-local git identity, not global)

Rules: never commit `.env`, never print token values into logs or terminal output, and do not add them to configs or code. There is no tracked `.env.example` (it was removed); document new keys here instead.

## Doc authority

When sources conflict, follow this order:

1. `docs/ideation/` (living direction)
2. `docs/researchProp/submitted/RP.pdf` (submitted JEPA + multi-objective latent planning plan)
3. Everything else (`docs/litReview/`, early `readme.md`, AB framing)

Do not let the older LLM-alignment / CoEvoRL story override the SafetyDial plan (JEPA latent planning + NSGA-II Pareto safety). Older docs may say EMOP; treat that as the same project under the former working title.

## Glossary

- **SafetyDial:** Project name. Pareto-front planning on a frozen JEPA world model so the safety-performance operating point is dialled at deployment, not baked in as a penalty weight
- **EMOP:** Former working title (Evolutionary Multi-Objective Planning). Legacy only; prefer SafetyDial
- **JEPA:** Joint-Embedding Predictive Architecture (predict in embedding space, not pixels)
- **LeWM:** Le World Model; JEPA substrate used as the frozen planner world model
- **CEM:** Cross-Entropy Method; LeWM’s default single-objective planner (baseline to replace)
- **NSGA-II:** Multi-objective evolutionary algorithm used for planning over action sequences
- **Structural safety signals:** Surprise (prediction error / OOD), latent energy and jerk (smoothness), small probes; not a hand-crafted cost
- **MPC:** Model predictive control (execute first actions of a plan, then replan)
- **ROSARL:** Reward-Only Safe RL (supervisor work; Minmax penalty; scalar baseline, not Pareto)
- **MAP-Elites / QD:** Quality-diversity illumination of behaviour space; optional extension
- **HV / IGD:** Hypervolume and inverted generational distance (Pareto-front metrics)
- **pymoo:** Python multi-objective optimisation library; source of NSGA-II, HV, and IGD implementations
- **uv:** Python package and environment manager used for this repo; all commands run under `uv run`
- **Vast.ai:** Rented GPU instance provider used for training runs; see `scripts/vast_*.sh`

## Agent must-know

### Research

- Prefer the name **SafetyDial** in new writing, code, and discussion; EMOP is legacy.
- Current thesis is **JEPA latent planning + NSGA-II Pareto safety**, not the lit review’s LLM/CoEvoRL story.
- Prefer **ideation + submitted RP** over AB, LR, or `readme.md` when they conflict.
- World model is **frozen LeWM**; novelty is the planner and structural safety objectives.
- Safety is a **separate objective on the Pareto front** (the dial), not a tuned λ penalty in a scalar reward.
- Primary envs are **Reacher** and **Push-T**; match CEM planning success before multi-objective work.
- Must compare to **penalty-tuned CEM, constrained RL, and ROSARL**.

### Engineering

- Run everything through **`uv run`**; do not `pip install` into the venv or create a second environment.
- Code goes under **`src/safetydial/<subpackage>/`**, configs under **`configs/<group>/`** as Hydra groups.
- Never commit secrets, checkpoints, datasets, run outputs, or `wandb/`; `.gitignore` already covers them.
- Compress any new PDF under `docs/papers/` before committing (see below).
- Follow the writing rules below (no em dashes).

## No em dashes

Never use em dashes in project writing (LaTeX `---`, Unicode `—`, or pasted en/em dash characters used as clause breaks). Prefer commas, parentheses, colons, or a full stop. For compound modifiers such as safety-performance, use a plain hyphen (`-`), not `--` or `–`. When editing existing docs, remove any em dashes you find rather than leaving them.

## Compress paper PDFs before committing

Papers under `docs/papers/` often ship with high-resolution embedded figures and can be tens of megabytes each. After adding or replacing any PDF in that tree, compress it with Ghostscript `/ebook` (downsamples images ~150 DPI; does **not** remove images or text).

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

Quality presets if `/ebook` figures look too soft: `/printer` (~300 DPI, larger) or `/screen` (~72 DPI, smaller). Prefer `/ebook` by default.
