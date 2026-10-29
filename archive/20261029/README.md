# Consolidated document archive

On 29 October 2026, [Principia](../../principia.pdf) replaced the loose experiment
write-ups and progress logs as the reading copy. Superseded plans, prompts, literature
working notes, reports, the Odyssey and the four-page framing guide were removed from
working directories after the 39-page PDF was built and visually checked.

[documents.tar.gz](documents.tar.gz) preserves the original document bytes at their
original repository-relative paths. [manifest.json](manifest.json) records each member's
size and SHA-256, plus the compressed bundle's hash. Every member was read back and
verified before its loose original was removed. Existing `legacy/` and `20261022/`
archives, code, frozen configs and machine-readable evidence were left intact.

## Read or restore historical prose

From the repository root, list members or read one without restoring the directory:

```bash
tar -tzf archive/20261029/documents.tar.gz
tar -xOf archive/20261029/documents.tar.gz progress/receipts/19-evo-adapter-results.md
```

Before reproducing a historical packaging or source-freeze workflow, restore only
the documents it expects. For example:

```bash
tar -xzf archive/20261029/documents.tar.gz progress/06-dsrl-plan.md progress/07-dsrl-repair.md
```

Full extraction into the repository also replaces current navigation files with their
historical snapshots. Extract into a separate directory when inspecting the whole bundle.
Historical instructions and rental records are evidence, never authorization to resume.

Verify the bundle with `.venv/bin/python archive/verify.py --manifest 20261029/manifest.json`.
Principia's source hashes can be checked against members without extraction.
