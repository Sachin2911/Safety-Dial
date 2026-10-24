# Task 4 preservation and rental closure

Completed 24 October 2026. All twelve episodes and every recording were retained.
Task 4 instance **57534766** was stopped, then destroyed after both local and private
dataset verification. No other session's rental was changed.

## Full local copy

After the queue finished and the resource sampler stopped, the host generated a
SHA-256 manifest over all 192 evidence files, totaling 194,148,002 bytes. A final
SSH/rsync copy returned those files and the manifest to
`outputs/tracking04/remote/`. At 18:27:20 UTC every local file matched its host hash
and size, with zero mismatches. The manifest itself is retained alongside the data.

This includes all twelve full NPZs and crash-surviving journals, all 24 camera
videos, monitor sidecars, evaluator rows/configs, per-attempt commands and validation
reports, the attempt ledger, logs, resource samples, pinned source archive, source
diffs and environment/download provenance. No failed starts exist and no successful
or unsuccessful episode was discarded. All 24 videos decoded through N+1 frames.

## Private dataset verification

Dataset: `Sachioster/maniguard-jar-replication-20261022`, confirmed private.
The evidence revision is **`a40b2baa5c96d3c49809cb71bfa09fa21f75269d`**.

The staged bundle contains 330 files in its SHA-256 manifest plus the manifest
itself. **All 331 files, 298,303,851 bytes, were downloaded from that pinned revision
and matched the local SHA-256 values.** The larger bundle includes canonical video
folders as well as the untouched raw copy, all analysis figures/metrics and the
frozen protocol, code and receipts. It does not replace recordings with summaries.

- [Canonical outcomes and videos](https://huggingface.co/datasets/Sachioster/maniguard-jar-replication-20261022/tree/a40b2baa5c96d3c49809cb71bfa09fa21f75269d/jar/tracking04_native)
- [Complete raw evidence](https://huggingface.co/datasets/Sachioster/maniguard-jar-replication-20261022/tree/a40b2baa5c96d3c49809cb71bfa09fa21f75269d/recordings/tracking04/raw)
- [Protocol, code, results, figures and manifest](https://huggingface.co/datasets/Sachioster/maniguard-jar-replication-20261022/tree/a40b2baa5c96d3c49809cb71bfa09fa21f75269d/receipts/tracking04)

Each seed has an `index.json` and task folders named `<task>_<outcome>`. The analysis
includes all twelve reviewed timelines, joint plots and event sheets, nine-lag
statistics, window statistics, review notes, the explicitly post-hoc joint-limit
audit and environment comparison. The executed-source archive was also inspected
for sensitive filenames and known credential values; none were present.

Final closure metadata, this receipt, the terminal rental record, the compiled
Odyssey and a copy of the interactive results canvas are added in a separate
metadata commit after destruction. The original evidence manifest is not rewritten.
Its upload/verification record and the final metadata revision are in the repository
receipt [04-tracking-completion.json](04-tracking-completion.json). The rental
snapshot in the first bundle correctly records the earlier stopped state; the
final metadata contains the terminal state.

## Provider state and estimated charges

| Event | UTC, 24 October 2026 |
|---|---|
| Rental created | 16:06:08.450818 |
| Queue finished | 18:23:22.369556 |
| Full local hash verification | 18:27:20.930502 |
| Stop request | 18:27:34.094188 |
| Provider observed exited/stopped | 18:28:52.959866 |
| Destroy request confirmed by provider | 18:37:54.048547 |
| Provider list verified without this instance | 18:38:35.785756 |

An earlier destroy CLI invocation encountered its interactive confirmation and
aborted before sending a provider request. Exit code alone was insufficient; the
instance listing exposed this. The subsequent invocation used the confirmation
flag, received the provider success message and was independently checked against
the instance list. Both records are retained. The six-hour watchdog deadline was
22:06:08 UTC; closure completed well before it.

The posted compute rate was $0.52/hour and selected 120 GB storage was approximately
$0.044444/hour, total $0.564444/hour while running. Charging compute conservatively
through the observed stopped timestamp gives $1.2371; charging storage through the
observed removal gives $0.1129. The time-based compute/storage estimate is therefore
**about $1.35**, not a final invoice.

Transfer is additional: listed download $0.004/GB and upload $0.0053333/GB. The
provider's raw billed counters at stop are retained in the rental record, but their
units were not independently established, so no conversion or total transfer charge
is claimed. Workstation-to-private-dataset preservation is separate from Vast's
host transfer counters.

## Completion checks

All planned episodes, all nine lags and all case reviews are complete. The source
checks and meaningful recorder/alignment tests passed; the analysis and audit ran
on all twelve real recordings. The results canvas passed strict TypeScript checking.
The Odyssey compiled to four pages with `pdflatex`; the new page was rendered and
visually inspected. The frozen panels and earlier receipts remain unchanged.

Workstation credentials were never copied to the GPU host. A failed provider CLI
request earlier in the session exposed its API key in a diagnostic URL; the key was
not included in any repository or dataset artifact. Rotation remains a separate
account-security action for the user.
