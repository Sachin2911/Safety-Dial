# Research plan history

[phase3.md](../../../phase3.md) is the active working plan and direction.
[AGENTS.md](../../../AGENTS.md) contains engineering and preservation guidance.

## Documents archived on 18 October 2026

These documents retain their complete contents as they stood immediately before this
reorganization. They record earlier decisions, hypotheses, results and initial estimates;
their prospective steps are historical context, not current execution instructions.

- [fruits.md](20261018/fruits.md): the six ManiGuard candidate mechanisms and the
  completed phase-one screen and follow-up evidence. Its SHA-256 still matches the
  completed-study requirements audit.
- [ideation.md](20261018/ideation.md): the earlier execution-adaptation and task-stage
  coordination discussion following UNISafe.
- [direction.md](20261018/direction.md): the original phase-three decision record and
  initial study estimates, before the methodological refinements in phase3.md.
- [oracle.md](20261018/oracle.md): the full earlier research authority, including the
  learned-filter motivation, UNISafe findings, phase-two transition and adjacent ideas.
- [phase2.md](20261018/phase2.md): the paused pilot's protocol, decisions and handoff.

The phase-two plan has been moved here unchanged. Its configs and experiment evidence
retain their existing locations and contents. The archived handoff is historical context,
not an instruction to resume the pilot. References to original root paths in frozen
JSON and source snapshots remain as recorded; use the manifest's path mapping to find
relocated plans.
The [older experiment archive](../../../archive/README.md) remains separate and retains
its existing manifest and original layout.

## Preservation and compatibility

[manifest.json](20261018/manifest.json) records each document's original path, archived
path, size, permissions and SHA-256. It also records shared links, retired compatibility
links and navigation changes.

The root `fruits.md`, `ideation.md`, `direction.md`, `oracle.md` and `phase2.md` paths have
been removed. Markdown links now point directly to the archived plans. No compatibility
aliases or index files remain at the root; the only root Markdown files are `phase3.md`,
`AGENTS.md` and `readme.md`.

Two navigation documents were hash-pinned by the completed ManiGuard audit. Their exact
original bytes are preserved under `20261018/preserved-navigation/`, with a `.txt`
suffix. Their current copies differ only in Markdown link destinations. The
`navigation_originals` entries map the audit's original paths and hashes to those preserved
copies and record the updated navigation hashes. Frozen audit JSON and source snapshots
retain their original paths and checksums.

The dated directory contains relative links named `docs`, `archive`, `configs` and
`phase3.md`. They preserve the archived plans' original root-relative Markdown links without
editing the archived text or copying the referenced material. They are shared references,
not additional archived datasets or experiment workspaces.

No experiment source, numerical result, frozen config, checkpoint or dataset was changed
or regenerated. Report and README navigation links were updated. The existing
problem-inventory note remains in `docs/research/research_notes/`.
