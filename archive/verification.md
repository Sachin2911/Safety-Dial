# Archive reorganisation verification

Verified 8 October 2026. Source repository revision before reorganisation:
`af727a7bc780d9067ef3624c8d12e4051ed64ca1`.

## Preservation

- 1378 versioned or previously nonignored files match their pre-move SHA-256
  hashes, sizes and permissions.
- 37 ignored local build artifacts were also moved unchanged.
  They remain local and are checked with `--include-local`.
- Notebook outputs, figures, PDFs, videos, configuration files and result data were
  preserved; no historical experiment source or recorded result was rewritten.
- 15 relative symlinks provide shared references and local asset
  paths. The `.venv` link targets the root environment and may be dangling before setup.
- The archive launcher was checked to use the repository root's Python interpreter,
  `archive/legacy/` as its working directory and the preserved configuration paths.

Verify preservation from the repository root:

```bash
uv run python archive/verify.py
uv run python archive/verify.py --include-local
```

The default check works with committed content. The second command also requires the
ignored build artifacts present on the machine where the reorganisation occurred.

## Regression checks

The same bounded unit-test selection was run before and after the move:

| Check | Before | After |
| --- | --- | --- |
| Passing tests | 773 | 773 |
| Existing skips | 11 | 11 |
| Explicitly deselected test | 1 | 1 |
| Test warnings | 65 | 65 |

The deselected test is
`tests/test_walker_training_recovery.py::test_interrupted_optimizer_and_randomness_match_uninterrupted`.
Its multiprocessing DataLoader stalled in the local environment before relocation,
including with one CPU thread. No test source was changed or permanently marked to skip.
The complete suite therefore has not been shown to finish in this environment.

The post-move command was:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  bash archive/run.sh python -m pytest -q \
  -k 'not test_interrupted_optimizer_and_randomness_match_uninterrupted'
```

Tests were invoked through `python -m pytest` to select the project interpreter. The
bare `pytest` executable in this local environment selected a different interpreter.

Ruff and shell syntax checks passed. Root pytest discovery points to the preserved tests.
The relocation link check covers the 518 previously valid relative Markdown links in
moved documents. Links to root references and the local environment are preserved through
relative symlinks; existing historical dead links are not represented as repaired.

## Scope

The reorganisation updates navigation and current research guidance around `oracle.md`.
It preserves older plans and job files as records, without restarting experiments,
downloading data or retraining models. Original remote absolute paths in managed jobs
and provenance remain historical and need deliberate configuration before reuse.
