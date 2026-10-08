# Historical remote coverage audit sources

These are inert, byte-identical `.py.txt` copies of the helpers actually used for the completed Walker S0/S1 and seven historical-model remote coverage audits. They were not executed again during preservation. The exact original preparation/inventory JSON inputs are also preserved; those remain proposals or inventory snapshots, not final completion/publication certificates.

The canonical preserved results are under `docs/mainPlan/results/continuation-audit/`:

- `walker_s0_s1_remote_coverage_20260927.{json,md}`: exact private pinned S0 complete policy (45 files), earlier policy (44 files) and S1 data (seven files) coverage. The 96 occurrences total 808,115,786 bytes, with 41 policy files duplicated across the two ladders. S1's live `s4/` descendants were excluded.
- `historical_model_remote_coverage_20260927.{json,md}`: four Push-T historical bundles verified at their run-ID tags (21 files); three Walker historical bundles remain unresolved (12 files, 94,390,134 bytes). An exact tag/pinned-head miss is not a search of all repository history.

The four report files are exact copies of the original `runs/continuation-audit/` results. Consequently their embedded original paths remain unchanged; the source manifest maps the preserved paths. `review.json` checks stored content identity comparisons, summary arithmetic, exact helper hashes and the distinction between verified and unresolved results. No scientific/model payload was copied or rehashed, no new remote query occurred, and no original result/model directory was changed.

The audited helpers use existing authorization with a fixed Hugging Face endpoint, read-only metadata methods and bounded deadlines. Copies are documentation, not entrypoints to rerun. The S0/S1 helper hardcodes fresh outputs and its original seven-file S1 allowlist. The historical helper derives exact model prefixes from local manifests, uses run-ID tags or one pinned repository head, and sanitizes failures. It is a limited coverage check, not a whole-history remote search.

No coverage claim repairs historical scientific contamination or unknown S0/S1 executed source. These records do not publish external reports, later media, current S4 results or final project/source archives. No upload, remote mutation or Git commit/push occurred here. The separate exact E2 archive approval hold remains in force.
