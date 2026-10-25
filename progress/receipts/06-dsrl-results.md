# Task 6 result: engineering gates did not admit learning

25 October 2026. The DSRL bridge, recorder, learner, scheduler, budgets and
preservation tools were implemented and exercised on four RTX 4090 GPUs on
Vast machine 139769. The final local suite passed 166 tests. Engineering stopped
at 00:19:36 UTC, before its 00:25:22 deadline. Stage B was not admitted: no
development training episodes, learned-policy evaluations or optimizer comparisons
ran. Safe-completion efficacy remains untested.

## Failed gates

The frozen 128-query decoder gate completed every request but failed native replay
identity. Four of sixteen factual comparisons passed, all from lane 0. Twelve
comparisons with factual outputs collected on lanes 1 to 3 differed by up to
0.0031306744 when replayed on lane 0. All sixteen retained the same converted
gripper commands. The two repeats on each saved input were exactly equal. This
establishes a lane-associated numerical pattern without proving its hardware or
kernel cause. The original tolerance was not relaxed.

Descriptively, one of eight inputs had a bounded-candidate gripper-command change
meeting the original margins, on task 10 at control 800. The minimum was two inputs
across two tasks. The invalid identity gate prevents treating this as a valid
negative authority screen. The [decoder audit](06-dsrl-decoder-audit.md) retains
all eight inputs and fourteen original vectors.

GPU source-versus-bridge SAC checks also refused differences above the unchanged
1e-6 threshold. Retained errors were 3.183726e-6 and, after matching static reference
metadata, 2.134591e-6 in an actor optimizer moment. GPU reduction nondeterminism
is plausible but unproven. A learner-only deterministic-kernel setting was identified
from pinned source but was not tried after the gate stopped engineering. The 128
timed updates and remote CPU/GPU actor comparison did not run. See the
[learner runtime receipt](06-dsrl-learner-runtime.md).

## Retained evidence and accounting

Four diagnostic-pool native episodes completed their full horizons: strict safe
completion 0/4, raw violations 2/4, counted violations 2/4, unfinished 4/4 and contact
engagement 4/4. Benchmark classes were two safe failures and two unsafe failures.
None registered a grasp. These are engineering observations, separate from baseline
estimates and the untouched development/held-out panels.

A grasp enum validity bug was corrected against the already saved native held
predicate. Every state was audited; raw recordings and unsuccessful outcomes were
unchanged. No episode was repeated or replaced. Discounted reward identities
reconcile to numerical zero, maximum residual 4.86e-17. The
[analysis](06-dsrl-analysis.md) and [independent audit](06-dsrl-independent-audit.json)
give denominators and source hashes.

Accounting is four starts, 8,000 policy controls, 40 warm-up controls, 1,140 returned
and client-confirmed VLA queries, and eight engineering SAC calls (two workstation,
six GPU). VLA calls comprise 1,000 task queries, 12 startup queries and 128 probes.
There were 134 workstation actor forwards and no remote controller inference
requests. Internal scene/reset physics is unmeasured, not asserted zero. The
[run record](06-dsrl-run.md) preserves source amendments, failed HTTP operations
and the explicit engineering measurement deviation.

Compute was confirmed stopped at 00:27:37 UTC after the immutable raw inventory
and query backup were verified locally. Quoted active compute/disk is about $2.93;
metered transfer is about $0.54, before final retained-disk charges. The hard cap
remained $15. Private revisions, disk destruction and provider billing are in the
[completion receipts](06-dsrl-completion/README.md).

Further experiments require a new frozen protocol. This run leaves DSRL's effect
on safe completion untested and provides no trained ordinary-learning comparator
or evolutionary result.
