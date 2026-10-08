# Research archive

Archived 8 October 2026 while narrowing the current research to
[task continuation under learned safety filtering](../oracle.md).

Earlier work is preserved in `legacy/` with its original relative directory layout.
The shared helpers and configurations make a single historical workspace more reliable
than splitting code between separate task folders. The index below provides task-level
navigation. Code, notebook outputs, reports, figures, videos and result data retain their
original contents.

## Push-T Safe CEM and the safety dial

- [Corrected findings and variant provenance](legacy/notes/safeCEM.md).
- [Report](legacy/docs/safeDial/SafeDialReport.pdf),
  [recorded results](legacy/docs/safeDial/results/) and [LaTeX source](legacy/docs/safeDial/latex/).
- [Safe CEM helper](legacy/experiments/helpers/safeCEM.py),
  [pilot runner](legacy/experiments/scripts/safe_dial_pusht.py) and
  [arena-fix validation](legacy/experiments/scripts/validate_arena_fix.py).
- [Initial probe notebook](legacy/experiments/LinearProbeAndAvoidance.ipynb),
  [later exploration](legacy/experiments/LinearProbeAndAvoidance_4.ipynb) and
  [dataset exploration](legacy/experiments/PushTDataExploration.ipynb).
- [Dial-sweep video](legacy/animations/dial_sweep.mp4),
  [penalty comparison](legacy/animations/penalty_vs_safe_cem.mp4) and
  [arena-repair demonstration](legacy/animations/dial60_arena_escape.mp4).

Use the corrected results and recorded variant provenance. The original infeasible-start
comparison and the later arena-geometry repair describe different conditions; the notes
preserve those distinctions.

## Hopper and Walker locomotion

- [Phase 0 findings](legacy/notes/phase0Report.md),
  [report](legacy/docs/phase0/Phase0Report.pdf) and [data](legacy/docs/phase0/data/).
- [Hopper get-up video](legacy/animations/hopper_getup.mp4) and
  [Walker fall-and-get-up video](legacy/animations/walker2d_fall_and_getup.mp4).
- [Animation provenance and caveats](legacy/animations/README.md).
- [Locomotion environment](legacy/experiments/helpers/locoEnv.py),
  [recovery search](legacy/experiments/helpers/oracle.py) and
  [triage runner](legacy/experiments/scripts/run_triage.py).

These results distinguish benchmark termination from irreversibility. A failed finite
recovery search is not an impossibility certificate, including for the historical clip
named `hopper_irrecoverable`.

## LeWM experience and Walker evolutionary studies

- [Experience-study report through S4](legacy/docs/mainPlan/reports/through-s4-20260927/README.md)
  and [plan with recorded results](legacy/docs/mainPlan/README.md).
- [Walker model implementation](legacy/experiments/helpers/walkerLewm.py) and
  [checkpoint/interface audit](legacy/docs/research/checkpoints.md).
- [Evolution-study results index](legacy/docs/evoPlan/results/README.md),
  [original implementation plan](legacy/docs/evoPlan/README.md) and
  [original paper idea](legacy/docs/paperIdea.md).
- [Walker paper draft](legacy/docs/geccoPaper/main.pdf),
  [LaTeX source](legacy/docs/geccoPaper/main.tex) and
  [figure provenance](legacy/docs/geccoPaper/figure_provenance.json).

The historical stage names, gates and dates remain attached to their original studies.
The current research does not inherit their outstanding execution steps.

## Earlier research and academic records

- [Earlier research reports](legacy/docs/research/reports/) and
  [brainstorming/source notes](legacy/docs/research/research_notes/).
- Submitted [ideation](legacy/docs/ideation/submitted/),
  [annotated bibliography](legacy/docs/AB/submitted/),
  [literature review](legacy/docs/litReview/submitted/) and
  [research proposal](legacy/docs/researchProp/submitted/).
- Original root [README](legacy/provenance/readme.md.txt) and
  [agent guidance](legacy/provenance/AGENTS.md.txt), preserved as historical text.

The paper library and current manipulation reviews remain under the root `docs/`.
Relative symlinks make those shared references available from the historical layout.
Local datasets and upstream checkouts also remain in root `data/` and `third_party/`.
The archive's `.venv` is a relative link to the root environment, preserving old local
source links without copying an environment. That target exists after root setup.

## Running preserved code

Set up the shared environment at the repository root using `bash scripts/setup.sh`.
The launcher changes to `archive/legacy/` and explicitly uses that root environment:

```bash
bash archive/run.sh python experiments/scripts/safe_dial_pusht.py --help
bash archive/run.sh python -m pytest -q
```

Use the Python-command portion of an old `uv run python ...` example after the launcher.
For notebooks, use the root project's kernel and open the notebook in
`archive/legacy/experiments/` so its `helpers` imports resolve.

Archived `pyproject.toml` and `uv.lock` preserve the earlier environment specification.
Use the launcher rather than creating another environment in the archive. Checkpoint and
normalizer availability still governs numerical reproduction; moving the source does not
rebuild missing assets. The [infrastructure record](legacy/docs/mainPlan/infrastructure.md)
contains the original asset-recovery procedure.

The shell/config files in `legacy/scripts/managed/` preserve original remote jobs, including
absolute `/workspace/Safety-Dial` paths. Prepare fresh job configuration before deliberately
resuming any such study; these files are not current launch instructions.

## Preservation and verification

[manifest.json](manifest.json) records original paths, destination paths, file sizes,
permissions and SHA-256 hashes. It covers historical source/evidence and copied environment
metadata. Ignored local build artifacts are listed separately and remain local; use
`--include-local` to verify those too. Disposable caches were moved but excluded from the
manifest. Shared references are listed separately. New archive navigation and tooling are
outside that frozen inventory.
Full original permission modes are recorded; the verifier compares executable status
because Git does not preserve group-write permissions across checkouts.

```bash
uv run python archive/verify.py
```

See [verification.md](verification.md) for relocation checks and the test-suite results.
