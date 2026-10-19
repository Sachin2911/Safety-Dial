# ManiGuard evolutionary safety dial pilot

18 October 2026. Sachin Mohan, University of the Witwatersrand.

[GECCO-template paper, PDF](maniguard-phase3-part1-gecco/named.pdf) ·
[Every matched video and picture](maniguard-phase3-part1-gecco/videos.md).

The planned 48 full episodes were implemented, executed and independently audited.
The evolved controller did **not** improve fresh safe completion: full strength scored
**4/8**, compared with identity's **5/8**, and recorded **2/8** violating episodes
against identity's **0/8**. The midpoint scored **3/8** with **2/8** violations.
All outcomes are measured in ManiGuard's physical simulator.

| Dial | Safe complete | Unsafe complete | Safe unfinished | Unsafe unfinished | Raw violations |
| --- | --- | --- | --- | --- | --- |
| 0, identity | 5/8 | 0/8 | 3/8 | 0/8 | 0/8 |
| 0.5 | 3/8 | 1/8 | 3/8 | 1/8 | 2/8 |
| 1, full anchor | 4/8 | 2/8 | 2/8 | 0/8 | 2/8 |

The primary full-minus-identity contrast is one fewer safe completion, or -12.5
percentage points, accompanied by two additional violating episodes. Full strength
completed six tasks against identity's five, but two were unsafe. Benchmark success
and requested completion agreed on all 24 comparison episodes; their separate labels
remain in the [audit](../research_notes/ManiGuard%20phase3%20Part1%20audit.json).
The midpoint's poorer result and the endpoints' violation counts provide no calibrated
or monotonic safety-dial interpretation.

![Per-scene outcomes, queries and process time](../figures/ManiGuard/phase3-part1/per-scene-outcomes-and-costs.png)

On `task_0029/base`, safe completions were 1/4, 0/4 and 1/4 at dial 0, 0.5 and 1.
On `task_0032/base`, they were 4/4, 3/4 and 3/4. Each adapted setting recorded one
violation on each scene. The first scene's first failing predicate was
`container_upright=false`; the second scene's was `obstacles_upright=false`.
The full-strength episodes still reached the goal after these violations. The native
monitor records these physical states but does not modify actions. The
[three measured dial points](../figures/ManiGuard/phase3-part1/dial-outcomes.png)
show counts and denominators without a fitted risk curve.

**Frozen controller and search.** The [controller export](../../../configs/maniguard/phase3/controller.json)
contains genome `[3,3,0,2]`, source/code/checkpoint hashes and the original finalist
file hash. It decodes to a 0.04 rad free-motion cap, a 0.02 rad grasp-conditioned cap,
smoothing 1.0 and a 16-action query horizon. The genome SHA-256 is
`e5077796728af4cbf29a159d72004085d32e488614c4b6ea33c1d4641ddc3fda`.
Dial settings 0, 0.5 and 1 use horizons 8, 12 and 16. Zero copies the native proposal
exactly; the midpoint blends half the anchor's arm-command deviation. The native
binarized gripper passes through unchanged.

Caps operate relative to measured joints. Smoothing uses measured joints initially,
then the previous actual environment input after action-space clipping. Only proposals,
arm/finger measurements and that causal history enter the controller. The fixed finger
heuristic chooses the grasp cap. No object identity, object states, goal flags or monitor
labels enter it. Joint-target caps do not bound contact force or guarantee object
uprightness. Logged environment inputs precede the native low-level controller's clipping.

The [Part 1 plan](../history/20261019/phase3Part1.md), from planning commit `8df7ac9`, took
precedence. Its [machine-readable protocol](../../../configs/maniguard/phase3/part1.json)
was frozen before search. One PCG64 seed, `20261018`, evaluated three populations of
four candidates on two training scenes. Utility was `int(S_req) - 2*int(V)`, using
raw `ltl_violated` and the pinned requested-completion checklist. Each transition
physically re-evaluated the copied elite and generated three mutations from the best
two parents, with fixed tie-breaking. Infrastructure failures supplied no fitness.

| Population | Candidate mean utilities, in birth order | Safe completions | Violating episodes |
| --- | --- | --- | --- |
| 0 | 0.5, 1, 0.5, 1 | 6/8 | 0/8 |
| 1 | 1, 1, 0, 0.5 | 5/8 | 0/8 |
| 2 | 0.5, 1, 0.5, -0.5 | 5/8 | 1/8 |

The finalist was a new offspring of population one's second eligible parent and
safely completed both final training episodes. The copied elite scored only 0.5 on
its last re-evaluation. Selection used the final population, not a historical maximum.
The discarded violating candidate and all costs are retained. The
[lineage](../research_notes/ManiGuard%20phase3%20Part1%20lineage.json) and
[search plot](../figures/ManiGuard/phase3-part1/search-lineage.png) show all twelve
candidate measurements, including elite re-evaluations.

The timing-only selector chose 48 episodes after population zero and before selection
inspected scores. Its conservative planning duration was 417 s; the remaining estimate
was 16,980 s against 21,149.97 s available. The shared initial eight episodes count
once. The 28-episode fallback was unused. No outcome-dependent expansion or retry was
added. Every full episode retained the native 1,100-action horizon and early-success rule.

Engineering used `task_0040/base`; search used `task_0034/base` and `task_0002/base`.
Whole-task partitions were verified and test scenes remained untouched. Training base
seeds were `1000+10*g+i`; comparison seeds were `2000+10*j+r`, with four repetitions
per scene. The unchanged evaluator derives policy sampling seeds using
`crc32(f"{base_seed}:{scene}") & 0x7fffffff`; the wrapper resets a stream per episode ID.
Independent schedule seed `20261019` sealed adjacent three-setting blocks before
comparison, cycling through all six permutations. The same finalist was frozen before
any comparison outcome. All 24 jobs, seeds, order, query schedules and zero identity
passed the [matched audit](../research_notes/ManiGuard%20phase3%20Part1%20matched-audit.json).

**Behavior and cost.** Lower query cost at full strength accompanied worse measured
safety and safe completion. This experiment changes both commands and query horizon;
it does not isolate their contributions.

| Dial | Policy actions | Queries | Query time | Episode process time | Changed commands |
| --- | --- | --- | --- | --- | --- |
| 0 | 6,160 | 774 | 76.8 s | 1,014.0 s | 0/6,160 |
| 0.5 | 6,656 | 558 | 55.2 s | 1,043.6 s | 300/6,656, 4.51% |
| 1 | 4,928 | 312 | 31.3 s | 873.5 s | 412/4,928, 8.36% |

Contact/grasp episode counts, each out of eight, were (5,5), (5,4) and (7,6) respectively.
The grasp heuristic activated on 955 midpoint actions and 1,325 full-strength actions;
anchor caps activated on 235 and 350. By-scene counts are in the audit. Changed-command
counts include exact numerical differences; cap activation is reported separately.
Identity's diagnostic anchor counters never change its commands. Non-engagement in
several unfinished episodes limits interpretation of zero violations alone.

**Verification and execution.** CPU checks passed: 33 Part 1 tests and 109 relevant
regression tests, with four skips. They cover identity, clipping/history, disabled caps,
finger switching, interpolation, scheduling, seeded evolution, evidence corruption and
process-group cleanup. Three successful 32-action engineering checks used 4, 6 and 8
queries for an engineering anchor of horizon four. Returned chunks supported horizon
sixteen. Both cameras were inspected.

An earlier check failed after one completed query and zero policy actions because
strict JSON rejected infinite native action bounds. Logging was repaired without
changing clipping, then checked before search. Its costs remain. A coordinator naming
bug produced nested engineering paths; it was fixed before search and those paths were
preserved. All 48 full episodes passed physical trace replay and provenance audits.

The confirmed replacement instance was `56582094`, with one RTX 4090 and driver
`575.57.08`, verified by `nvidia-smi`, the kernel and the provider. The requested
`56578764`/`615.78.08` identity remains historical provenance. The user explicitly
confirmed the replacement with the original clock. No host driver was changed; the
working driver falls within ManiGuard's [documented tested 5xx range](https://nu-ideas-lab.github.io/ManiGuard/docs/getting-started/installation/).
Five-second samples reached **17,714 MiB**, approximately **17.3 GiB**; this is a
sampled maximum, not a bound on unobserved peaks. CPU/memory quotas were 30.6 cores
and approximately 299 GiB.

Separate uv-managed Python 3.10 simulation and Python 3.11 policy environments were
prepared, without running the UNISafe installers or syncing the root environment.
The checked legacy renderer used 256 x 256 RGB overview/wrist inputs and 20/20/120 Hz
action/render/physics rates, raw joint control and assisted grasping. Source revisions,
environment freezes and download receipts are retained. The Clutter checkpoint was
revision `f572e33bdb498a8d9992a72312ee849be868a2b6`, step `7100`, with native inference
settings and no additional encoder or policy learning.

**Accounting and preservation.** The [52-row ledger](../research_notes/ManiGuard%20phase3%20Part1%20attempts.csv)
counts every raw attempt once, including the failed check, three successful checks,
elite re-evaluations and discarded candidates.

| Work | Attempts | Policy actions | Policy queries | Recorded physics ticks |
| --- | --- | --- | --- | --- |
| Engineering, including failure | 4 | 96 | 19 | 920 |
| Search | 24 | 17,318 | 1,738 | 105,972 |
| Fresh comparison | 24 | 17,744 | 1,644 | 108,528 |
| Total | 52 | 35,158 | 3,401 | 215,420 |

All 3,401 queries completed; the failed check was a logging failure. There were
35,678 environment-step attempts and returns, 1,795.17 recorded simulated seconds,
360.03 s of query latency and 5,982.17 s summed outer episode-process time.
Physics counters include scene initialization after application launch. Advances before
callback subscription are unknown for all 52 attempts, not zero; failed-attempt counters
remain lower bounds. Process durations overlap the persistent server and are not added
as extra allocated GPU-hours.

Allocation began at **16:00:37 UTC**, including setup, implementation, failure and idle
time before and after replacement. The independent supervised guard preserved the
**23:00:37 UTC** physical target and **00:00:37 UTC on 19 October** hard end across
restarts. The policy stopped at **18:25:29 UTC**, and process/GPU queries confirmed
complete exit by **18:25:57 UTC**, at **2.422 allocated GPU-hours**. Reporting, backup
and idle rental after that point continue to count. The public audit's closeout receipt
records the later accounting cutoff and rental-stop scope.

There are 104 native video files, including the failed check's recordings, and
[16 matched videos](../research_notes/ManiGuard%20phase3%20Part1%20matched-videos.json)
covering every comparison scene/seed block with both cameras. CPU composition retains
native frame order and marks ended episodes with black frames. No physics was replayed.
Credential-scanned raw evidence, source snapshots, videos and the report are backed up
in the approved private dataset
[Sachioster/safety-dial-phase3-part1-20261018](https://huggingface.co/datasets/Sachioster/safety-dial-phase3-part1-20261018),
excluding asset bundles, checkpoints and keys. Code, protocol and selected derived
artifacts are committed; generated run directories remain outside Git.

The conclusion is a working execution interface and an adverse fresh pilot result.
Scenes are conservatively treated as exposed. All sixteen matched initial-state pairs
were identical numerically, but none of their camera-image pairs were identical.
Eight repetitions across two scenes and one optimizer seed support a descriptive
comparison. Even identity's zero violations in eight independent homogeneous episodes
would only give the approximate 95% upper bound `3/8`, or 37.5%; repeated scenes further
limit interpretation. This is no safety guarantee or evolutionary-superiority result.
A new allocation should isolate query scheduling from caps and compare strong fixed
settings and matched random search before enlarging evolution. Ordinary policy learning
remains necessary for the broader phase-three claim.
