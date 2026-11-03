# Retained records

This directory holds the historical receipts and supporting records moved out of
the repository root on 3 November 2026. Their contents and original dates are
preserved. Historical paths inside them describe the layout at the time of recording.

| Former location | Current location |
| --- | --- |
| `progress/` | [progress/](progress/README.md) |
| `results/` | [results/](results/) |
| `environment/` | [environment/](environment/) |
| `configs/`, except UNISafe runtime settings | [configs/](configs/) |
| `configs/unisafe09/` | [SafeBlock/configs/unisafe09/](../../SafeBlock/configs/unisafe09/) |
| `provider-status.json` | [admin/provider-status.json](admin/provider-status.json) |

The [consolidation receipt](consolidation-20261103.json) records every moved file's
original path and hash, its new path, and the removed legacy files. The
[validation report](consolidation-validation-20261103.json) records the final checks.
These records do not authorize resuming historical experiments or rentals.

The [earlier deletion receipt](progress/receipts/local-deletion-20261103.md) describes
the explicitly deleted root archive and output trees. Its original relative links
and inventory paths predate this move; use the mapping above. Deletion did not
establish backup coverage for every removed file.

The independently verified SafeLeWM archives have their own retained
[cloud recovery guide](../../SafeLeWM/archive/README.md). Keep its pinned cloud
revisions and original manifests unchanged.
