# Task 4: native command and motion check

Authorized by Sachin on 24 October 2026 after reviewing task 2 and the literature.
This is the current run instruction. The twelve-arm proposal in `04-screen.md` has
not run and is not part of this rental.

## Question and frozen design

Determine whether the lid-push and regrasp failures identified in task 2 are
associated with a discrepancy between commanded and realized motion, an unhelpful
command sequence, or a continuation problem. This is descriptive mechanism evidence;
it cannot by itself establish that a controller would improve completion.

Run the native Jar policy on development tasks 0, 3, 6, 9, 15 and 17, once at each
policy seed 300 and 301. These seeds are chosen before looking at any new outcomes.
Twelve planned episodes, all on one host. No pool or held-out episodes. The first
planned episode also validates the recorder and counts in the twelve.

Use ManiGuard `6bf4c61`, the pinned Jar checkpoint and the task 1 legacy 256 config:
20 Hz, eight executed actions from each sixteen-action chunk, horizon 2,000, goal
held ten steps. No action intervention, proxy, early violation stop or park stop.
One evaluator and one policy server per GPU. A two-GPU host has two independent lanes;
a one-GPU host runs the same frozen jobs sequentially. Do not infer deterministic
replay from matching seeds. Consecutive episodes on a server must have distinct
episode seeds; reset sampling explicitly before retrying the same episode.

## Recording and validation

Use `scripts/todo2/record_rollout.py` with the official evaluator's passive observer
hook. Preserve original chunks, all executed-command boundaries, joint positions and
velocities, gripper state, end-effector and jar poses, contact flags, monitor traces,
and both camera videos. Contact and monitor information is for offline diagnosis.

Validate each completed recording against its row and monitor sidecar: N actions,
N+1 state rows, ceil(N/8) proposals, continuous step indices, finite action and robot
state arrays, and identical recorded scoring fields and propositions. Check the first
recording before starting the rest. An instrumentation defect stops the queue; retain
the failed attempt and document any repair before resuming.

## Accounting and stop rules

The twelve planned episodes receive no outcome-driven repeats. Allow at most two
additional starts in total for crashed or technically invalid attempts, at most one
replacement per job; preserve and count every attempt. Maximum 14 simulator starts
and 28,000 executed steps. No extra simulator smoke episodes. Model-only probes are
counted separately. Each simulator process has a 25-minute wall timeout, and new jobs
are not admitted after four rental hours. Stop or destroy this task's instance by six
rental hours, preserving partial work first. Preparation failure is not a reason to
leave an idle rental running.

Select an on-demand RTX 3090 host reporting a verified 580.x driver, at least 120 GB
disk and the established template `62414e8fca6befba085051de7ed36190`. Prefer the known
hosts if available. Target rental rate at most $0.75/hour including selected disk;
record the actual offer and bandwidth charges before purchase. Expected total spend
is a few dollars; the six-hour ceiling bounds compute/storage at $4.50 at that rate,
with transfer charges recorded separately.

## Analysis fixed before outcomes

Score all four benchmark outcome classes and safe requested-sequence completion
(closure no later than first departure from support), plus engagement and unfinished
episodes, with denominators. Keep diagnostic and historical rates separate by host.

Align the seven arm targets with the corresponding measured joints using recorded
joint indices and executed step identifiers. Report command-to-state discrepancies
before and after each action, target variation, actual joint motion, gripper commands,
and timing relative to contact, closure, grasp, lift, and first violation. Compare
tracking at lags 0 through 8 steps, reporting all lags; use pre-contact behavior as a
reference rather than declaring all instantaneous error abnormal. Contact can cause
error, so distinguish discrepancies preceding contact from those following it.

Inspect every episode's aligned plot and event frames, including successes and safe
stalls. Native targets that track the parked state suggest little commanded progress;
large requested moves with little realized motion suggest a tracking or contact
constraint, not automatically inadequate controller gain. Commands followed into an
unhelpful motion motivate policy steering. Mixed or ambiguous evidence stays mixed or
ambiguous. No post-hoc threshold is a confirmatory decision rule.

## Preservation and completion

Use fresh output prefixes `jar/tracking04_native/`, `recordings/tracking04/` and
`receipts/tracking04/` in the private dataset. Preserve every NPZ, video, sidecar,
results row, attempt log and the exact protocol and source files. Write SHA-256 hashes
and verify uploaded files against local copies at a pinned dataset revision. Also
download the full recordings and necessary sidecars to ignored local storage before
destroying the instance. No recording may be discarded after being summarized.

Write `04-tracking-host.md`, `04-tracking-run.md`, `04-tracking-results.md` and
`04-tracking-preserve.md` under `progress/receipts/`. Add a short completed-task entry
to the Odyssey and recompile it. Record the final provider stop/destroy state and
estimated charges. Do not change earlier receipts or the frozen task panels.
