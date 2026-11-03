# Recorded SafeLeWM results

These seventeen accounts consolidate the completed 1-2 November 2026 development work.
[Rule43](../Rule43.pdf) gives one page per account; [Rule42](../Rule42.pdf) remains
the prospective research plan. Counts are descriptive, not independent held-out
replications or evidence of an evolutionary advantage.

| Account | Main result | What it established |
| --- | --- | --- |
| [1. Initial qualification](01-initial-qualification.md) | Planner 0/3 clean; targets 0/6 | Feasible geometry, inadequate completion signal |
| [2. Planner settings](02-planner-settings.md) | random-05 9/9 diagnostic; paper profile 5/50 dataset | Selected-case recovery depends on recipe |
| [3. Fixed baseline](03-fixed-baseline.md) | random-05 9/50 | Diagnostic successes did not generalize |
| [4. Short-goal replication](04-short-goal-replication.md) | 44/50; authors' same-seed record 45/50 | Near reproduction of the short-goal setting |
| [5. Long-goal AP comparison](05-anchored-planning.md) | AP 20/50 versus matched final-goal 0/50 | A useful fixed target-selection lever |
| [6. Hazard qualification](06-hazard-qualification.md) | AP 3/6 safe; 1/6 violations; 2/6 unfinished | Family failed its frozen admission threshold |
| [7. Broader hazard screen](07-hazard-screen.md) | Both same-room families pass; doorway completion 0/8 | Feasible safe routes and recurring AP violations |
| [8. Recurrent adapter pilot](08-recurrent-adapter-pilot.md) | New-layout safe completion: evolved 5/12 versus native 2/12 | An exploratory safe-completion lever, with a completion tradeoff |
| [9. Matched-budget replication](09-adapter-replication.md) | CMA safe completion 8/36, 6/36, 11/36; native 3/36 | Repeated gains on this panel; broad superiority remains unestablished |
| [10. PushT hazard screen](10-pusht-screen.md) | AP 15/50 no hazard (gate waived); on-path family 8/8 contacts | One memorised-path scene where AP always contacts the island |
| [11. Cube hazard screen](11-cube-screen.md) | AP 26/50 no hazard; 0/16 with posts | No task competence from rest; renderer not bit-exact |
| [12. Representation and mechanisms](12-adapter-mechanism.md) | CMA safe completion 45/216; random 47/216; native 7/72 | Head coordinates selected; no aggregate CMA advantage over random |
| [13. PushT distinct-scene screen](13-pusht-authority.md) | Native 4/16 complete, 2/16 safe; safe witnesses 6/8 | Prerequisite gate failed; no random adapter probes or training |
| [14. Matched PPO comparison](14-matched-ppo.md) | CMA 44/216 safe; random 37/216; PPO 20/216 | CMA ranked highest here; primary uncertainty intervals include zero |
| [15. PushT AP reproduction](15-pusht-ap-reproduction.md) | Authors 88/128; our wrapper 89/128; published 89/128 | Matched baseline competence; no safety or learning claim |
| [16. PushT validated qualification](16-pusht-validated.md) | Native 12/16 complete, 0/16 safe; probes only 0-1/10 admitted safe | Baseline/hazard gate passed; adapter authority failed; no training |
| [17. Safety-dial library](17-safety-dial-library.md) | 17/45 targets missing; 5/28 available settings exceeded risk limits | Some policy tradeoffs persist; calibrated risk remains unestablished |

The first seven accounts contain 389 scored episodes, plus separately counted
preflights. The corrected adapter pilot adds 560 training/evaluation episodes and
31 engineering/witness episodes. Its interrupted first attempt adds 459 charged
resets and remains separately accounted, excluded from the optimizer comparison. The earlier paper-profile outcome shown in account 3 is the same run
reported in account 2, not another 50 episodes. The authors' saved result in account
4 is an external reference, not a run performed here. Shared inputs and repeated
scenes limit independence.

## Figures

`graphs/01-*.pdf` through `graphs/16-*.pdf` are outcome charts used in Rule43;
matching PNGs appear in the Markdown accounts. Bars report completion including
unsafe completion. The hazard charts separately show any-violation frequency;
these are overlapping outcomes, not portions to add together.

Original trajectory figures are retained unchanged:

- [Initial qualification](graphs/01-qualification-trajectories.png).
- [Planner diagnostic panel](graphs/02-diagnostic-trajectories.png) and
  [dataset panel](graphs/02-dataset-trajectories.png).
- [Fixed baseline](graphs/03-fixed-baseline-trajectories.png).
- [AP paired long-goal panel](graphs/05-ap-trajectories.png).
- [AP hazard panel](graphs/06-hazard-trajectories.png).
- [PushT no-hazard panel](graphs/10-pusht-long-trajectories.png), [on-path](graphs/10-pusht-on-path-trajectories.png) and [near-path](graphs/10-pusht-near-path-trajectories.png) islands.
- [Cube no-hazard panel](graphs/11-cube-long-trajectories.png), [on-path](graphs/11-cube-on-path-trajectories.png) and [near-path](graphs/11-cube-near-path-trajectories.png) posts.

`record.json` stores the curated counts and prose. `render_graphs.py` redraws only
the summary charts from those recorded counts, without experiment computation.
Rule43's LaTeX and the Markdown accounts remain directly editable; keep all three
representations consistent when adding results.

## Evidence and accounting

Each account identifies its original saved summary and frozen protocol. The
[archive manifest and restoration tool](../archive/README.md) retain exact original
file bytes, sources, traces, audits and accounting. The full earlier README is
recoverable from the archive for detailed implementation history and source links.

The three-stage AP run totals 174 scored episodes, 175 resets including preflight,
21,965 actions, 122,507 model batch calls, 39,202,726 predicted blocks and 933,980
encoder frames including cache construction. The two preflight cost calls are
included in the corrected batch-call aggregate. Its original preservation receipt
verified 590 evidence files and 85 source bindings before this consolidation.
No training, additional evaluation or pod operation was performed during cleanup.

The [broader screen](07-hazard-screen.md) adds three all-layout trajectory figures,
72 independently audited episodes and its own [verified archive](../archive/README.md#broader-hazard-screen-1-november).
Step 1 is complete; no adapter work followed.

The corrected adapter pilot and its interrupted attempt are privately preserved in
`adapter-pilot-20261101.tar.gz`. All 6,557 files passed independent local hash
verification; the corrected run passed trajectory, native-state, optimizer and
checkpoint-replay audits. See account 8 for the full accounting and limitations.

The replication adds 2,254 audited episodes, including budget-truncated work, and
294,707 actions. Exactly 240,000 are training actions. Its crossed-cluster analysis
and figures were generated on the rented host; `render_graphs.py` preserves those
figures rather than replacing them with the single-panel template. See account 9.

The two manipulation screens add 296 scored episodes plus 8 witness-development
episodes, one failed Cube start and one two-episode determinism diagnostic, all on
the L4 pod and all independently audited; the Cube pair-equality check failed for the
renderer reason recorded in account 11. Their traces, freezes, audits, caches and
panels are preserved in `manip-screen-20261101.tar.gz` and
`manip-memory-20261101.tar.gz` with their own manifest; see the archive README.

The mechanism study adds 9,510 audited episodes and 1,216,506 actions, including
960,000 training actions. The audit amendment preserves the original timestamp
failure and changes no outcomes. Its graphs and crossed-bootstrap analysis were
computed on the rented host; see account 12. The matched PPO study remains separate.

The distinct-scene PushT screen adds 72 audited completed episodes and one failed
replay attempt: 73 resets and 16,365 actions in total. All parity checks passed;
baseline/witness qualification failed. No random parameter probe was executed.
The native-box clipping correction and byte-identical reuse are recorded in account 13.

Accounts 14 and 15 close the matched PPO comparison and PushT reproduction.
Both passed their saved-evidence audits. PPO adds 4,169 episode records and
507,387 actions; AP adds 260 resets and 46,417 actions. Their frozen protocols,
original failures and verified preservation receipts are linked from each account.

Account 16 adds 329 audited episodes and 81,502 actions. All 16 parameter probes
were completed, but the useful-authority gate failed. The twelve reserved cases
remain untouched and no optimizer training followed.
