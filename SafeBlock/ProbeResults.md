# Probe results, 1 November 2026

Development evidence for the two qualification probes in [Probes.md](Probes.md).
Development seeds only; nothing here is a paper claim. Raw records (per-control
traces, summaries, implementation hashes) are on the host under
`outputs/probes-20261101/` and mirrored without camera frames under
`SafeBlock/outputs/probes-20261101/` (gitignored). Tables come from
`experiments/summarise_probes.py`.

## Platform notes from the smoke runs

- The pinned October Vulkan manifest named that host's driver file. On this host the
  simulator started but every camera returned a constant frame. The worker now
  writes a manifest for the installed driver (EGL path) and requires `vulkaninfo`
  to enumerate the NVIDIA device before launching; the record is saved with each
  episode (`renderer.json`).
- First simulator start compiled shaders for about 3 minutes; afterwards start-up is
  16 s per episode. An anchored-planner episode averages 26 s wall time including
  start-up, so about 140 episodes per lane-hour on this host when episodes end early.
- The Hard variant's restored terminations agree with the released `subtask_terms`
  predicates on every recorded control (0 disagreements), so the restoration matches
  the authors' own Hard observation thresholds.
- The library encoder found that 597 of 1,369 success-only demonstrations do not
  start with `is_first` and a zero action (they look like cut continuations). Their
  first frame is encoded from a zero state like every other episode; the anomaly is
  recorded per episode in the library manifest.

## Probe A: planners on the frozen world model, regular task, 8 starts

| condition | safe success | sustained | violation (physical, boundary) | timeout | K1 overhead episodes | mean plan time |
|---|---:|---:|---:|---:|---:|---:|
| native_unfiltered | 1 | 0 | 1 (1, 0) | 6 | 8 | - |
| native_filtered | 0 | 0 | 0 | 8 | 8 | - |
| cem_goal | 0 | 0 | 5 (2, 3) | 3 | 5 | 0.075 s |
| ap_cem | 7 | 3 | 1 (1, 0) | 0 | 6 | 0.040 s |
| ap_rank | 7 | 4 | 1 (1, 0) | 0 | 3 | 0.006 s |

Seeds 4,100,000 to 4,100,007. Sustained means the success predicate still held 40
controls after first success with no violation in between.

**Reading.** Both anchored planners pass the reading rule (at least 3/8, not only on
the native actor's start). The rank variant is carried into Probe B as the rule
specified for a tie. The goal-only CEM reproduces the TwoRoom finding: the terminal
latent distance to a far goal is flat (about 151 at the start and 153 at the end of
one failed episode), so the search has no gradient and wanders into drops and
boundary exits. Aiming at a near anchor from a similar demonstration fixes this on
Block Plucking as it did on TwoRoom.

**What the anchored planners do wrong.** One physical failure each: `ap_cem` dropped
the top block at control 26 on seed 4,100,000; `ap_rank` dropped the middle block at
control 180 on seed 4,100,003. Three to four successes per planner are not
sustained because the planner keeps following anchors after the block is out and
disturbs the stack again (the predicate is lost, typically the top block leaves the
base or the middle block is pushed back inside 7 cm). The planner cannot see the
success predicate, so "stop when done" is behaviour it does not have. That is
precisely the kind of thing a small adapter with a risk budget would be asked to
supply, and sustained safe success is the natural task return.

**Filter and keep-out signals.** The released filter would have intervened on 25 to
50% of the native actor's controls but on only 2 to 5% of the anchored planners'
controls, consistent with planner trajectories staying near the demonstrations. The
overhead keep-out K1 is entered on 31 to 61% of native controls and 3% of `ap_rank`
controls; K2 and K3 are rare for everything except `cem_goal`.

**Retrieval.** `ap_rank` used 9 to 19 distinct demonstrations per episode across 13
to 45 plans, mean retrieval distance about 50 to 67 in squared feature units.

## Probe B: Hard variant with keep-out monitors, 12 starts

| condition | safe success | sustained | violation (physical, boundary) | timeout | K1 overhead episodes | K3 lateral episodes |
|---|---:|---:|---:|---:|---:|---:|
| native_unfiltered | 0 | 0 | 11 (11, 0) | 1 | 8 | 0 |
| native_filtered | 3 | 0 | 7 (6, 1) | 2 | 11 | 1 |
| ap_rank | 5 | 3 | 6 (5, 1) | 1 | 8 | 3 |

Seeds 4,200,000 to 4,200,011. Hard: base block 7 cm, top block 0.25 kg, success
stacking tolerance 0.03 m, failure displacement 0.07 m, terminations restored.

**Failure types.** Native unfiltered: 4 top-block displacements, 5 middle-block drops,
2 top-block drops, 1 timeout, failing between control 30 and 184. Native filtered:
4 middle drops, 2 top drops, 1 boundary exit, 2 timeouts; its 3 successes all lost
the block during the 40-control continuation (none sustained). `ap_rank`: 4 top drops,
1 middle drop, 1 boundary exit, 1 timeout; 3 of its 5 successes were sustained.

**Filter effect on Hard.** Unlike the regular task, the live filter changes outcomes
here: 3 completions against 0 and 7 violations against 11 on the same 12 starts, with
interventions on 7 to 68% of controls. The released filter's own signals would fire
on 3 to 79% of `ap_rank` controls on Hard versus 2 to 5% on the regular task, so the
ensemble and reachability heads do register the shifted dynamics.

**Admission (rule in Probes.md).** Hard alone, on native physical violations, admits
a risk-budget comparison for two conditions: `ap_rank` (5 completions, 6 violation
episodes on distinct starts) and the filtered native actor (3 and 7). No keep-out
family is needed; K1 remains available as a second violation type if wanted.

## What the probes establish, and what they do not

- Move 1 works mechanically: the released Dreamer world model supports anchored
  planning from the authors' own demonstrations, and the anchored rank planner is a
  competent frozen base on the regular task (7/8) with a weakness that is physical
  (one drop in 8) and a behaviour that costs sustained success (it keeps moving after
  the block is out).
- Move 2 works: the Hard variant turns the same frozen base into a system that
  completes 5/12 and violates 6/12. Completion and risk are now in tension, and the
  released filter and the task actor exist as hand-designed references on the same
  starts.
- Block Plucking now has the TwoRoom shape: frozen encoder, frozen dynamics, Anchored
  Planning as base, a real failure predicate, and an adapter slot on the planner's
  command or on anchor selection.
- Not established: anything about adapters, evolution, ranking rules or the dial. The
  seeds are development seeds; the planner settings were fixed before running and were
  not tuned, but neither were they validated on held-out starts.

## Proposed next step (needs a decision)

Write the first risk-budget comparison on Hard with `ap_rank` as the frozen base,
sustained safe success as the task return, native physical violation as the
violation, requested limits around the observed 0.5 rate, and the filtered native
actor plus bare `ap_rank` as references. Adapter interface candidates: a bounded
residual on the planner's command, or anchor selection among the k retrieved
segments. Throughput on this host is about 140 episodes per lane-hour.
