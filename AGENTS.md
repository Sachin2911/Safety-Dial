# Agent notes

## Current direction and authority

Read [oracle.md](oracle.md) first. On 8 October 2026, Sachin chose **task continuation
under learned safety filtering** as the focus of the research rethink. The question is
whether evolutionary adaptation can help a pretrained manipulation policy work more
effectively with its safety filter and improve safe task completion.

The choice is conditional on meaningful, avoidable stalls in an accessible manipulation
task. The environment, world model, behavioural search space, evolutionary algorithm,
ROSARL integration and experiment budget remain open. UNISafe is a candidate foundation.
Do not turn those open choices into assumed requirements or inherit the old Walker
method, timelines or run gates as the new implementation plan.

- **Student:** Sachin Mohan (2699183), BSc Honours CS, University of the Witwatersrand.
- **Supervisor:** Geraud Nangue Tasse.
- **Research context:** latent world models, safe RL, evolutionary algorithms and ROSARL.
- **Target venue/context:** [GECCO notes](docs/gecco2027.md) and
  [allocated topic](docs/allocatedTopic.md). The exploration is not restricted to the old
  November thesis schedule.
- **Evidence:** [comparative review](docs/research/reports/Safe%20manipulation%20progress%20and%20robustness.md)
  and [UNISafe compute audit](docs/research/reports/UNISafe%20inference%20and%20training%20compute.md).

## Repository map

1. `oracle.md`: central authority for the current research focus and decisions.
2. `docs/research/`: current literature reviews, supporting notes and asset audits.
3. `docs/papers/`: shared reference library.
4. `experiments/`: workspace for new implementation; add reusable Python under `helpers/`
   and entry points under `scripts/`. There is no `src/` package.
5. `scripts/`, `configs/download/`, `pyproject.toml`, `uv.lock`: shared setup and downloads.
6. `archive/README.md`: navigation for preserved Push-T, Hopper, Walker and earlier studies.
7. `archive/legacy/`: original relative layout of historical code, configs, tests, notes,
   reports, figures, videos and deliverables. Its plans are historical context.

The user explicitly authorized this archive reorganisation on 8 October 2026. Archived
source and evidence were moved without changing their contents. `archive/manifest.json`
records their original paths and hashes; `uv run python archive/verify.py` checks them.
Caches were moved but are excluded from the manifest. Shared symlinks retain access to
root data, upstream checkouts, downloads and current reference material.
Ignored local build artifacts can be checked with `archive/verify.py --include-local`.

Original root guidance is preserved as
[AGENTS.md.txt](archive/legacy/provenance/AGENTS.md.txt). Read it for historical interfaces
and study-specific interpretation rules when reusing old work, not as a current plan.

## Research and preservation controls

- Keep archived experiment code, notebook outputs and recorded results unchanged. New
  experiments use new files and output paths. Do not regenerate results as part of tidying.
- Do not restart old S5 stages, gated studies or managed jobs from archived plans. Legacy
  supervisor scripts include original absolute paths and are historical run records.
- Distinguish poor task continuation from a restrictive filter, prediction error, missing
  information or an insufficient horizon. A timeout alone does not diagnose the cause.
- Ordinary task-policy learning with the same filter present is an essential comparison.
  A standard EA can support a contribution, but an optimizer substitution is not novelty
  by itself. Recheck primary literature before making novelty claims.
- Report safe completion, physical violations and unfinished episodes together. Fewer
  interventions can reflect inactivity. Never report imagined safety as measured safety.
- Count all simulator interactions, including adaptation, discarded candidates and model
  training when relevant. Report model queries and elapsed time separately.
- Keep proposed, filtered and executed actions distinguishable. Physical dynamics need
  executed-action histories; a task critic can intentionally model proposals to a filtered
  environment. Match information advantages and budgets across comparisons.
- With zero violations in n independent episodes, the approximate 95% upper rate bound is
  3/n, not zero. Preserve counts, denominators and the scope of every empirical claim.
- Preserve historical corrections: the first Push-T penalty collapse was confounded by
  the starting position; use the corrected Safe-CEM provenance. Benchmark termination
  and unsuccessful finite recovery search do not establish irreversibility.
- When reusing LeWM results, retain their trajectory splits, frozen scalers/readouts,
  interaction accounting and distinction between readout and imagined-dynamics errors.
  Detailed controls remain with the archived studies.

## Environment and engineering

- Package manager: `uv`; Python is pinned to 3.11 in `.python-version`.
- Setup: `bash scripts/setup.sh`, or `uv sync --frozen --extra dev`.
- Use `uv run python ...`, `uv run python -m pytest`, and `uv run ruff check .`.
  Using `python -m pytest` keeps tests in the project interpreter.
- Run legacy entry points/tests with `bash archive/run.sh python ...`. It changes to
  `archive/legacy/` and explicitly uses the root environment with `--no-sync`. Run setup
  at the repository root first if needed. Do not create a second project environment.
- Archived `pyproject.toml` and `uv.lock` preserve the original environment specification.
  The launcher deliberately selects the maintained root environment instead.
- Preserve the narrow historical exception for `safety-gymnasium` and `omnisafe`: their
  incompatible pins require ephemeral `uv run --isolated --no-project --python 3.10`
  tooling. It may emit small reference artefacts, not datasets or reported measurements.
  The complete original command and restrictions are in the archived guidance.
- New reusable code goes in `experiments/helpers/`; Hydra configs go in `configs/<group>/`.
  Keep notebooks alongside their study and launch kernels with the correct working directory.
- Ruff line length is 100. Fix new Python lint errors instead of broadening ignores.
- Commit `uv.lock` when dependencies change. Do not install with `pip` or bypass `uv`.
- Do not commit datasets, checkpoints, credentials, `wandb/` or generated run directories.
  Model checkpoints belong in private Hugging Face repositories pinned by revision.
- Existing assets must be checked on the run machine. The old Vast volume is gone;
  [historical infrastructure notes](archive/legacy/docs/mainPlan/infrastructure.md) and
  [checkpoint audit](archive/legacy/docs/research/checkpoints.md) describe reconstruction.
- Read relevant archived environment gotchas before reusing MuJoCo code, especially EGL
  ownership, render dimensions, replay solver state and action normalization.

## Secrets

Secrets stay in the ignored root `.env` or the remote account environment. Never print
values, commit them, or place them in configs, manifests or command arguments. The existing
setup helpers and downloader load the root `.env`.

Recognized keys remain `WANDB_API_KEY`, `GITHUB_TOKEN`, `HF_TOKEN`, `HF_NAMESPACE`,
`GIT_AUTHOR_NAME`, `GIT_AUTHOR_EMAIL`, `STABLEWM_HOME`, `LEWM_REPO_URL` and `LEWM_GIT_REF`.
There is no tracked `.env.example`; document new keys here if needed. Never repurpose
`HOME` or `CODEX_HOME` for task-specific paths.

## Writing and reference files

Use plain prose and no em or en dashes as clause breaks. For compound modifiers use a
plain hyphen. Current deep-research output belongs under `docs/research/`; earlier
research remains in the archive. Do not create competing top-level research-plan folders.

Compress newly added paper PDFs under `docs/papers/` with Ghostscript `/ebook` before
committing, replacing only when smaller. Preserve archived PDFs and notebook outputs as
recorded; moving them is not a reason to recompress or rebuild them.
