# ManiGuard latent evolution: agent execution prompt

Prepared 19 October 2026 for an implementation agent working in this repository.
The research authority is the root
[Evolutionary improvement on ManiGuard](../../../Evolutionary%20improvement%20on%20ManiGuard.md),
including its execution contract. This handoff supplies execution instructions and
storage destinations. Earlier Phase 3 prompts remain historical.

In Codex, start the following goal. The agent should then read the instructions below;
an already active goal does not need to be created again.

```text
/goal Implement and execute the ManiGuard latent-evolution pilot described in docs/research/research_notes/ManiGuard latent evolution goal prompt.md and Evolutionary improvement on ManiGuard.md. Read AGENTS.md first. Deliver a reproducible physical comparison, tested implementation and verified GitHub/Hugging Face preservation. Diagnose, repair and resume through recoverable failures. Keep the comparison fair, preserve negative outcomes and obey the current rental deadline.
```

## Mission and completion standard

Implement the shared, observation- and instruction-conditioned controller around
ManiGuard's frozen family policies, starting with bounded native-noise steering.
Establish useful action authority, then execute the prospectively frozen physical
pilot with native, fixed, evolutionary and random-search conditions. Seek improved
safe requested completion and investigate why a method succeeds or fails. A positive
result for evolution is not required for an honest completed comparison.

Read the root plan, its past-evidence review and the linked preservation records before
choosing implementation details. Reuse verified interfaces and assets while keeping
archived code, evidence and frozen protocols unchanged. Put new reusable Python in
`experiments/helpers/`, entry points in `experiments/scripts/`, configs under a new
`configs/maniguard/latent-evolution/` study directory, and raw outputs under ignored
`runs/`. Do not restart archived coordinators or inherit their task splits blindly.

A written plan, green unit tests, an inactive controller, a saved model or decoder
probes alone do not complete this goal. Completion requires the selected valid physical
comparison, analysis, accounting and verified preservation. If a hard resource limit
or external blocker prevents that, report the objective as unmet and preserve a
precise, resumable partial result.

## Current rental and binding limits

Read [the dated rental record](../../../configs/vast/maniguard-rental-20261019.json)
and verify the actual machine and current UTC time before doing expensive work.
The authorized instance is **56777435**, a verified Community Cloud host with two
24 GB RTX 3090s, driver **580.173.02**, approximately 128 GB allocated system RAM and
20 effective CPU cores. Its observed rate is **$0.6125925926/hour including 200 GB
storage**, below the user's approved **$0.65/hour** ceiling. The original quoted
Oregon offer became unavailable; this replacement was acquired after approval.

The provider start is recorded in the rental JSON. The independently scheduled stop
is **20 October 2026 at 03:00 UTC, 05:00 SAST**, job **4986**, retaining the disk.
This is earlier than eight hours from creation. Time already spent on startup and
preparation is charged. This deadline is absolute; starting Codex does not restart
the allowance. Check that the stop remains configured. Preserve incrementally and
reserve the final 30 minutes for audit completion, uploads and handoff. Do not cancel
or postpone the stop, restart an expired/stopped instance, or create another rental
without further authorization. If this dated allocation has expired, continue useful
local preparation and report the missing allocation rather than silently renting.

The two-GPU rental supersedes the plan's illustrative single-GPU hardware assumption
for this session. It does not double the episode allowance. Count both allocated GPUs:
one wall-clock hour is two allocated GPU-hours, including idle time. Keep simulator
interactions, model calls and elapsed time as separate counters. The 156-episode
transfer bridge and paper-level replication remain separately budgeted future work.

Read `/etc/vast-agents-guide.md` on the host. The base image supplies Ubuntu 22.04,
Python 3.10.21 in `/venv/main`, CUDA toolkit 12.4 and uv; it does not supply an installed
or validated ManiGuard research stack. Recheck GL, OptiX and Vulkan libraries and
perform an actual camera smoke test. Manage persistent services with supervisor.
Use one GPU for simulation and the other for policy serving initially; validate the
device mapping. Additional concurrent rollouts require a declared evaluation design.

## Preparation, protocol and physical work

Prepare the simulator and policy environments separately, using uv and explicit Python
paths. Choose their versions from the pinned upstream requirements. Do not root-sync
the legacy `pyproject.toml`/`uv.lock` into either environment or run the UNISafe installer
as a ManiGuard installer. Verify checkpoint revisions, asset hashes, benchmark scenes,
the extended-finger Franka asset, renderer, camera mapping and native horizons on this
host. Confirm the long-finger asset is loaded instead of assuming a successful import
proves it. Repair matching graphics userspace libraries if needed; leave the
host-injected NVIDIA driver unchanged.

Freeze a machine-readable protocol, including task/family partitions, features,
controller parameterization, objective, validity rules, optimizer settings, bounded
repair/development allowance, shared initialization and independent audit. Reserve a
whole family before inspecting behavioral responses. Preserve the research programme's
exposure history. A small robot-only integration prototype is insufficient for the
shared semantic-controller claim.

Follow the plan's bounded authority audit, up to 320 decoder calls across designated
development families, and four full development episodes for identity, integration
and timing. Verify that zero steering preserves native noise sampling and that
perturbations influence the actually executed action prefix. Use direct physical
outcomes for search. The monitor supplies labels and does not veto actions; privileged
monitor or goal state is not an actor input.

Select Jar and the complete pilot size from engineering feasibility and conservative
timing before seeing comparative scores. Clutter is the declared capacity fallback.
The illustrative pilot has 60 planned valid episodes plus at most four infrastructure
replacements, with a protected independent final comparison. Charge partial attempts
and valid timeouts. Do not shorten benchmark horizons to fit more trials. If the full
comparison cannot fit the remaining allocation, make that explicit before launching
it and preserve a frozen, runnable implementation and justified partial study.

Use the same parameter domain, observations, initialization data, tuning allowance,
candidate accounting and finalist protection for EA and random search. Freeze finalists
before independent evaluation. Report safe/unsafe completion and safe/unsafe unfinished
outcomes, violation counts, denominators and requested-sequence semantics. Implement
ordinary-learning interfaces where feasible, but do not claim evolutionary superiority
until competent matched learning and independent replication have been evaluated.
Bias/scale, persistence and response-informed coordinates are bounded development
options; activation gating is a distinct follow-up. They cannot consume the final audit
or silently expand this session's allocation.

## Diagnose, improve and resume

Follow the root plan's execution contract after every failure. Capture evidence,
reproduce the smallest informative case, distinguish technical defects from scientific
outcomes, test a specific repair, validate it and resume the interrupted stage.
When a repair fails, update the hypothesis and try the next informative fix within
the reserved allowance. Continue independent work while a genuinely necessary
credential or clarification is pending. A first exception, low completion rate,
inactive trigger, weak feature/predictor or disappointing learning curve does not
end the assignment.

Trace inactive steering through the server and sampler. Diagnose reset/render/noise
differences before blaming the optimizer. Inspect candidate diversity and real
behavioral authority before expanding search. Use the declared direct-residual fallback
only with its own prospective protocol and available allowance. Keep an already valid
comparison complete even if evolution loses. Version later development changes and
their costs; do not tune against audit outcomes or present exposed tasks as untouched.

Maintain a durable status file, attempt ledger, optimizer/RNG checkpoints, candidate
ancestry and pending/completed evaluation queue. Reconcile receipts after interruption.
Worker exceptions return control to the recovery loop. Do not attach provider shutdown
to an exception handler or worker failure. The hard deadline still applies.

## GitHub: source, protocols and small evidence receipts

The GitHub repository is **https://github.com/Sachin2911/Safety-Dial.git**. Start from
the pushed `main` handoff, record its exact commit and inspect existing changes.
Use a study branch such as `maniguard/latent-evolution-20261019`; resume it if it
already contains this run. Commit and push tested implementation, prospective configs,
small numerical summaries, research reports, manifests and preservation receipts
incrementally. Record the pushed commit SHA beside each exported controller/run.
Preserve the `main` launch revision and existing history. Do not force-push or erase
unrelated work. A source upload or pull request does not substitute for execution.

GitHub stores the reproducible recipe and compact evidence. Keep datasets, base-policy
weights, controller checkpoints, raw recordings, videos, caches, `wandb/`, generated
run directories and credentials out of Git. Use the existing Git/gh credential setup
or the ignored `GITHUB_TOKEN` environment credential. Never put tokens into a remote
URL, command argument, config, report or commit.

## Hugging Face: private artifacts and reconstruction

Use the configured `HF_TOKEN` and `HF_NAMESPACE`; the historical namespace is
**Sachioster**. Verify authenticated identity, permissions and repository privacy
without printing token values. Keep credentials in the ignored root `.env` or remote
account environment. GitHub and HF credentials may need providing on this fresh host;
their existence on a previous machine does not establish access here.

Create or resume separate **private** repositories for this study, under the verified
namespace:

- **Dataset:** `Sachioster/maniguard-latent-evolution-runs` for attempt ledgers,
  observations and action traces, evaluation receipts, videos, raw outcomes and backups.
- **Model:** `Sachioster/maniguard-latent-evolution-controllers` for controller weights,
  optimizer checkpoints, fitted bases/features/scalers and their frozen configs.

These are proposed new destinations, not claims that they already exist. If
`HF_NAMESPACE` identifies another authorized namespace, use the same repository names
there and record the resolved IDs and types. Check an existing destination before
writing and append immutable run/stage prefixes instead of overwriting earlier runs.
Do not upload licensed asset bundles or encryption keys as study artifacts; retain
their authorized download references.

Existing private historical storage is a reconstruction source, not a new output
destination:

- `Sachioster/safety-dial-phase2-handoff-20261018` is a **dataset**. Its revisions,
  archive hashes and exclusions are in
  [handoff-20261018.json](../../../configs/maniguard/phase2/handoff-20261018.json).
  It excludes BEHAVIOR assets/encryption keys, pretrained policies and the external
  policy interpreter. Its runtime is historical and needs path/version validation.
- `Sachioster/safety-dial-phase3-part1-20261018` is a **dataset** of preserved evidence.
  The donor revision and hashes used by later studies are recorded in
  [frozen-baseline-export.json](../../../configs/maniguard/part4/frozen-baseline-export.json).
- `Sachioster/safety-dial-phase3-part4-20261019` is a **dataset**;
  `Sachioster/maniguard-phase3-part4-20261019` is a **model** repository containing the
  earlier warning predictor, not the pretrained VLA. Pinned revisions, files and
  verification receipts are in
  [closeout.json](../../../configs/maniguard/part4/closeout.json).
- `Sachioster/unisafe-pretrained` is a **model** repository for the separate historical
  UNISafe study, pinned by [unisafe.json](../../../configs/download/unisafe.json).
  It is not a ManiGuard checkpoint source.

Download historical files at their recorded immutable revision, verify SHA-256 before
reuse, inspect archive contents and extract into staging rather than over the checkout.
Do not revive archived services or frozen study protocols from runtime bundles.
Acquire the required family VLA checkpoints and licensed simulator assets from their
correct upstream sources. Download only what the authorized stage needs; monitor the
200 GB disk rather than restoring every historical archive and family checkpoint.

Back up incrementally after development milestones and search generations. For every
upload, record repository ID/type, private status, exact remote commit revision,
file paths, sizes and SHA-256 hashes. Read back small receipts and verify remote hashes
or redownload-and-hash large artifacts. A successful upload call alone is insufficient.
Publish only nonsecret preservation receipts in GitHub. Verify that both code and
artifacts are recoverable without this rental before declaring preservation complete.

## Final handoff

Deliver the implemented controller/runtime and meaningful checks; the frozen protocol
and source/checkpoint pins; development and comparative outcomes; lineage and complete
interaction/query/time/GPU-hour accounting; a concise contribution assessment; pushed
GitHub commits; and verified private HF revisions with restoration instructions.
Distinguish completed physical work from untested transfer, latent extensions and
ordinary-learning comparisons. Keep negative and adverse evidence visible.

Before the deadline, finish feasible analysis and preservation, leave a restart command
and exact remaining work, and respect the independently scheduled provider stop.
Do not destroy or recycle the instance. Its disk is ordinary container storage and
will survive stop/start but not destruction/recycling; stopped storage remains billed.
If a genuine blocker remains, give its evidence, attempted repairs, consumed/remaining
budget and smallest external change needed. Keep working toward a defensible result
through all useful authorized recovery paths.
