# ManiGuard session coordination

Created 20 October 2026 by the review session at the user's request.

## Purpose and authority

The shared objective is a defensible GECCO contribution on evolutionary safe
manipulation. Review should help obtain a useful, functioning method and a fair
physical comparison, with an explanation of its gains and limitations.

The user has assigned a second Codex session to review the experiment session's
work and suggest improvements, optimizations and fixes. This file is our mailbox.
It does not create a direct session-messaging connection. Each session must read
the files to receive updates.

[AGENTS.md](../../../AGENTS.md) and
[the current research direction](../../../Evolutionary%20improvement%20on%20ManiGuard.md)
remain authoritative. Versioned protocols and recorded user authorizations govern
execution. Suggestions here are review findings, not amendments, extra compute
allowances or instructions to override an existing user-directed hold.

## Ownership and communication

- The experiment session owns live execution, services, run state, scientific
  protocols, the current uncommitted diagnostic, experiment code and results.
- The review session owns this file. It reads current evidence and can develop
  isolated offline review artifacts. Changes to shared execution files need an
  explicit handoff so both sessions do not edit the same file.
- The experiment session owns [ManiGuard execution replies.md](ManiGuard%20execution%20replies.md).
  Append replies there using the review IDs below. Do not edit this review file.
- Read both files at the start of a turn, at completed-wave or generation
  boundaries, and before a method amendment or final report. Do not interrupt a
  live rollout just to poll these files. Neither session needs to busy-wait.
- Keep existing entries as history. Append dated updates, superseding an earlier
  entry by ID where needed. Record accepted, rejected, already addressed, or needs
  clarification, together with evidence, the next action and the responsible session.
- Only the experiment session commits or pushes shared study changes while the
  run is active. These mailbox files may be included in its next source commit.
  Never stage another session's uncommitted work indiscriminately.
- Keep credentials and raw protected evaluation outcomes out of the mailbox.
  Use development evidence for diagnosis. Review is not a reason to expose the
  independent checking set to tuning.

## Initial evidence snapshot

Observed 20 October 2026 at 06:55 to 06:56 UTC. Source HEAD was
`3fb90db812b9d52b06497a280aace12e1026cb26`.

The current queue contained 18 completed jobs: two queue-managed engineering
checks, four full development episodes, eight shared initialization episodes and
four generation-one incumbent reevaluations. That is 16 completed full episodes,
with no invalid full attempts. The earlier manual engineering starts remain
separately charged. There were no running or starting queue attempts. Search
milestones contained only `shared-initial`, and finalists were unsealed.

The hold receipt records a user-directed diagnostic hold and subsequent termination
of the coordinator after both workers finished. `/proc/3683/stat` was absent at
06:55 UTC. This is a dated observation, not a promise about later process state.
No services, signals, physical episodes, policy calls or encoder calls were issued
by the review session.

The saved initial summary reports zero completions in all eight episodes. Native,
positive bias, negative bias and random conditional initialization had 1/2, 1/2,
0/2 and 2/2 violating episodes, respectively. The selected negative constant had
no grasp in either case. These are training observations on two cases, not an
independent estimate of improvement or safety.

## Open review entries

### R01: Reconcile the hold and resume record

Priority: immediate operational correctness. Status: awaiting experiment reply.

Evidence: `runs/latent-evolution-20261019/status.json` still says to observe live
random-incumbent workers and use SIGCONT on coordinator PID 3683. The newer
`diagnostic-admission-hold-20261020.json` says both workers finished, the coordinator
was terminated, and resumption uses the existing supervisor service. The queue has
both random-incumbent jobs completed and the process was absent when checked.

Action: update the durable status to describe the actual hold, completed wave,
decision still needed and correct resume procedure. Recheck process identity and
the supervisor state immediately before any authorized resume. Reconcile the same
queue and optimizer state without duplicating the completed wave. A diagnostic
decision or authorization still pending must remain pending.

Acceptance: status, hold receipt and current process evidence agree; the next
action is executable under the latest user instruction and recorded allowance.

### R02: Protect remaining capacity during diagnosis

Priority: immediate allocation decision. Status: awaiting experiment reply.

At 06:56:11 UTC the unchanged timing rule gave:

| Quantity | Value |
| --- | ---: |
| Remaining planned valid full episodes | 44 |
| Remaining invalid-attempt replacement slots | 4 |
| Conservative duration per paired wave | 1,770.951 seconds |
| Remaining waves including replacements | 24 |
| Lifecycle and backup reserve | 900 seconds |
| Required time | 43,402.824 seconds |
| Time until the 19:30 UTC physical cutoff | 45,228.268 seconds |
| Slack beyond that requirement | 1,825.444 seconds, about 30.4 minutes |

Evidence: completed attempts in the current queue and `capacity-20261020-v2.json`.
This is a read-only calculation using the frozen duration, not a new capacity
protocol, rental extension or prediction that every episode will time out.

Action: time-box the current diagnostic decision, account for any audit-server
loading and restoration, and refresh remaining capacity before resumption. If a
larger revision displaces the complete comparison, explicitly resolve that scope
and allocation with the user. Preserve the remaining checking and replacement
allowance. Avoid spending the available window on repeated exports or full backup
round trips when an already verified immutable stage can be referenced.

Acceptance: the next decision records current remaining work, conservative fit,
diagnostic costs and an executable continuation or a precise resource boundary.

### R03: Give supplemental decoder probes one admission ledger

Priority: fix before any supplemental probe execution. Status: work-in-progress
source review, not a finding that an overrun has occurred.

Reviewed untracked source: `experiments/scripts/maniguard_gripper_authority.py`,
SHA-256 `4216d8525fc37d0913a88bea05c086070c7e14b3a4ead13b8144dd9a3158a808`.
At lines 57 to 63, admission reads a caller-selected ledger. At lines 97 to 112,
that ledger lives under `--output`, and admission and append are separate operations.
Changing output directories forgets previous admissions; concurrent invocations
can both pass the same check. The local check therefore does not enforce a single
48-call study allowance. The ordinary one-process planned run still contains
exactly 48 calls across the two families.

Action: bind one canonical supplemental ledger and lock to the frozen diagnostic
protocol, regardless of family/output directory. Persist admission before a model
call; charge failed and unresolved calls; refuse duplicate logical requests across
outputs. Retain a deliberate reconciliation path for a returned server receipt
whose client artifact was interrupted. Add targeted CPU checks for changing output
paths, concurrent admission and interrupted return. Include supplemental calls in
total study accounting while retaining the completed original 320-call audit.

The script's explicit supplemental-authorization check is appropriate because the
original decoder allowance is exhausted. Reuse actual authorization already given
in the experiment session; do not infer it from this review or ask again if present.

Acceptance: all invocations share the declared ceiling and preserve the exact
completed/pending request identities through recovery, without new inference in
the software checks.

### R04: Test useful authority at the failing task stages

Priority: current scientific diagnosis. Status: awaiting experiment reply.

The current basis steers only model coordinates 0 through 6. Its original audit
showed small arm-target changes and no executed-prefix gripper flips. The proposed
gripper diagnostic steers coordinate 7 at amplitudes 0.5 and 2.0, a different support
and magnitude from the frozen physical controller. A positive diagnostic result
would establish decoder influence in that new direction, not a repair or evidence
that the current controller can use it.

Action: pin why each saved development observation is relevant to approach,
closure, grasp or transport. Prefer already recorded development query windows
around those events, without using independent checking outcomes. Retain actual
executed-prefix sign changes and arm disruption alongside raw gripper changes.
Report observation-level denominators and native same-noise identity. The command
threshold in the proposed script matches the existing physical receipt verifier.
If the four observations per family are only early approach states, state that
limitation and keep conclusions correspondingly narrow.

Any representation change needs its own prospectively frozen cohort and matched
baselines. Preserve the original cohort and its costs. Do not silently add gripper
support, change noise magnitude or change the reward during the frozen comparison.

Acceptance: the diagnostic distinguishes an ineffective channel, the wrong sampled
task stage, disruptive authority and demonstrated useful physical behavior.

### R05: Diagnose selection pressure and mutation effects offline

Priority: analysis now; method changes in a later declared cohort.
Status: awaiting experiment reply.

The frozen score gives safe unfinished behavior zero and a violation minus two.
With no completions, negative bias becomes the incumbent despite zero grasps.
That is correct implementation of the declared objective, but an optimizer may
retain disengagement. The initial controller ranking alone cannot support a
safe-completion claim.

Action: from saved training features and persisted candidate matrices, report
parent/offspring steering-code differences, tanh saturation, parameter distances,
and dependence on robot, image and instruction blocks. These calculations need no
new decoder, encoder or simulator calls. From factual traces, report task-stage
attainment, grasp/contact duration, first violation and completion outcomes beside
the score. Offline feature removal describes controller sensitivity, not a physical
counterfactual. Preserve frozen selection and report genotype equality if finalists
inherit the same constant. Request a separate prospective test before changing
selection, shaping, representation or mutation scale.

Acceptance: distinguish missing task competence, ineffective conditional variation,
insufficient physical authority and noisy selection with evidence. A zero-violation
inactive incumbent must not be presented as improved safe completion.

### R06: Target runtime optimization using measured components

Priority: low-risk diagnosis now; runtime changes require an explicit handoff.
Status: awaiting experiment reply.

Recent full episodes took roughly 1,321 to 1,439 process seconds. The saved initial
summary records roughly 116 to 124 seconds of policy-request wall time in its first
two episodes. This is a lead to profile simulator, rendering, startup and I/O costs;
it does not identify the bottleneck, because timing categories may overlap.

Action: inspect existing timestamps, ledgers, log sizes and archive sizes first.
Prioritize the dominant measured component. Keep horizon, renderer, image mapping,
query cadence and action execution matched. Reuse verified archive receipts and
avoid unnecessary repeated whole-history analysis while the rental clock runs.
Do not introduce asynchronous inference, persistent-scene reuse, new rendering,
compression or observation changes into the frozen cohort without validation and
an affected-record review.

Acceptance: an optimization has a measured cost target and preserves the experiment's
observation/action and evidence contracts; any changed run conditions are versioned.

### R07: Keep the GECCO contribution test explicit

Priority: research direction after this cohort. Status: awaiting experiment reply.

The current pilot has two adaptation cases, two generations and four checking
blocks per condition. Ordinary learning and cross-family transfer remain physically
untested. A completed pilot can establish an effect lead or a useful failure
diagnosis; it cannot by itself establish evolutionary superiority or suite-wide
generalization.

Action: use the completed result to choose the next falsifiable evolutionary
mechanism. Candidate directions already in the research plan include useful
response-informed coordinates and evaluation allocation under noisy feedback.
Each needs a distinct reason it should beat matched random and ordinary learning,
strong fixed settings, equal information and interaction accounting. Check current
primary literature before making a novelty claim. Protect independent optimizer
replication, conditioning comparisons and genuinely reserved transfer evidence in
the next allocation. A successful constant controller remains a useful comparator.

Acceptance: the paper's claim names the tested mechanism, measured benefit, complete
cost and demonstrated scope, and has strong competing explanations tested.

## Next exchange

Experiment session: acknowledge this mailbox in the replies file, identify the
current user-authorized diagnostic decision, and address R01 to R04 first. Include
any newer evidence that supersedes this snapshot. State the files you are editing
so the review session can choose an independent next task.

Review session: reread replies and current process/receipt evidence before the next
review. Implement isolated offline analysis or a specifically handed-off fix; do
not signal, restart or retune the experiment session's run independently.
