# Task 5 preservation and teardown

24 October 2026. The full two-engineering and 36-scored experiment is complete,
locally verified, preserved in the established private dataset and torn down.
No Task 5 rental remains. No optimizer or new intervention was launched.

## Verified private evidence

Dataset: `Sachioster/maniguard-jar-replication-20261022`, verified private before
upload and after download verification. The immutable experiment revision is
[`3ba307c58164b493ae48dae6950f2766c139567c`](https://huggingface.co/datasets/Sachioster/maniguard-jar-replication-20261022/tree/3ba307c58164b493ae48dae6950f2766c139567c).
All **1,095 files, 1,163,799,093 bytes**, including the
SHA-256 manifest itself, were downloaded at that exact revision and matched the
local staged copies by SHA-256. The verification receipt is
[05-adapter-upload-verification.json](05-adapter-upload-verification.json).

Canonical scored runs are `jar/adapter05_native/seed400`, `seed401`, and the
corresponding `adapter05_fixed` and `adapter05_adaptive` paths. Engineering uses
separate `jar/adapter05_engineering_<arm>/seed<seed>` paths. Every run has its
own index and every task has its outcome folder. Engineering never enters the
scored denominators.

The complete remote output tree is under `recordings/adapter05/raw`: all 38
attempts, 76 videos, recordings, monitor sidecars, result rows, journals, server
and evaluator logs, frozen source archives, plans and environment provenance.
Analysis, all 38 visual reviews, the protocol and receipts are under
`receipts/adapter05`. The task 4 admission-evidence snapshot was copied directly
from the workstation into this authorized private dataset; it was never sent to
the rental. No dataset or checkpoint was committed to the source repository.

Before stopping the instance, all 625 files in its remote manifest,
744,023,524 bytes, matched the workstation copy. The manifest's own SHA-256 was
also checked against the value computed on the rental. Additional earlier
preparation-log copies were retained. See
[05-adapter-local-copy.json](05-adapter-local-copy.json). The upload is larger
because it also contains analysis, source, receipts and canonical copies of
videos and sidecars.

## Provider state and charges

Instance 57544241 completed the queue at 20:09:03 UTC. Its launcher and both
policy servers had exited with no GPU compute processes by 20:09:47 UTC.
The full local copy was verified at 20:10:31 UTC. The provider subsequently
confirmed `cur_state=stopped`, `intended_status=stopped` and
`actual_status=exited` at 2026-10-24T20:12:11.120780+00:00. The disk was retained
until the private evidence verification succeeded.

Destruction used the CLI's explicit `--yes` option. Absence from the provider's
instance list was verified at 2026-10-24T20:26:03.191021+00:00. The failed
provisioning instance, 57539828, was also absent. Its failure consumed no
simulator controls or model queries, and its cost is included below.

| Provider charge | USD |
| --- | ---: |
| Successful rental GPU | 1.316 |
| Successful rental storage | 0.155 |
| Successful rental download | 0.144 |
| Successful rental upload | 0.007 |
| Failed provisioning rental storage | 0.096 |
| Total reported | **1.718** |

These are provider-reported charges queried at 2026-10-24T20:26:09.454509+00:00, after
both instances were destroyed. They are a billing snapshot, not an assertion
that later adjustments are impossible. Earlier conservative estimates remain
labeled separately. Exact provider rows are in
[05-adapter-billing.json](05-adapter-billing.json); terminal state is in the
[rental record](../rental-adapter05-20261024.json).

## Completion metadata and local record

The immutable experiment snapshot predates teardown. Final billing, terminal
rental state, this receipt and the updated Odyssey source and PDF are preserved
separately under `receipts/adapter05/completion/`. The pinned revision and full
download verification for that metadata are recorded in
[05-adapter-completion-verification.json](05-adapter-completion-verification.json).
That verification record is kept locally with the source history so the receipt
does not need to contain its own future commit identifier.

The final recording audit reconciles 38 first attempts, 73,409 policy controls,
380 warm-up controls, 9,227 model-query attempts and 9,223 confirmed returns.
There are no invalid attempts, replacements or held-out tasks. Every numerical
recording field is finite, all 76 videos have the expected size and frame count,
and analysis decoded every frame. All episode plots and event frame sheets were
reviewed. Source and analysis hashes are recorded separately; no reporting
change was copied into a running experiment.

The Odyssey has one new Task 5 page, compiled twice with `pdflatex` and visually
inspected. Its earlier four pages have identical extracted text. The PDF and
LaTeX hashes are in
[05-adapter-odyssey-verification.json](05-adapter-odyssey-verification.json).
The result and its limits are in [05-adapter-results.md](05-adapter-results.md).
