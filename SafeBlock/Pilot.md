# Pilot: the TwoRoom adapter recipe on Block Plucking, in two hours

Proposal written 1 November 2026, after Stage 1 ([Stage1Results.md](Stage1Results.md))
and the LeWM adapter replication ([../SafeLeWM/AdapterReplication.md](../SafeLeWM/AdapterReplication.md)).
It replaces the proposed Gate 2b. Authorised 18:12 UTC; the protocol was frozen as
written before the first episode.

## Purpose

One question, answered on the frozen Block Plucking system in about two hours of
simulator time: does the recipe that moved safe completion on TwoRoom (neutral-start
CMA-ES on a small adapter around a frozen Anchored Planning base, safe-completion-first
fitness, random-search and untrained references) also move it on a physical
manipulation task?

For the paper this gives the second environment of the "one recipe, two scales"
paragraph: a four-row table (native, untrained, random search, CMA-ES) on sixteen
fresh starts, plus the learned weights, which are interpretable. It does not test
the risk-budget rule; that remains Stage 2.

## What is fixed

Everything in `configs/stage1-freeze.json`, unchanged: Hard variant with restored
terminations, `ap_rank` (64 neighbours, lookahead 5, block 5), model revision and
checkpoint hash, library hash, violation = any control including the continuation,
task return = sustained safe success, fresh simulator process per episode. Stage 1
proved zero-weight parity bitwise on this system.

## Adapter

The 5-weight linear anchor selector from Stage 1: at each replan the planner's cost
for each of the 64 retrieved segments is reduced by `w . z`, where `z` is the
within-set standardised vector (retrieval distance, imagined cost, mean arm
magnitude, closing fraction, demo progress). `w = 0` is the bare planner.

Chosen over the residual (destructive at every authority tested in Stage 1) and over
an embedding-input selector (the better science, but not implementable and qualified
inside two hours; it is the Stage 2 adapter if this pilot shows the search finds
anything). The Stage 1 sweep showed this selector changes about 3 of 15 choices per
episode at weight scale 0.5 and produced genotypes with four sustained successes and
no violation, so it has authority and signal.

## Recipe pieces taken from the LeWM track

| Piece | LeWM | Here |
| --- | --- | --- |
| Start | neutral heads = native AP | `w = 0` = bare `ap_rank` |
| Coordinates | standardised, fan-in scaled | `w = z`, scale 1 (features already standardised) |
| Optimiser | pycma 4.4.4, diagonal, explicit CSA, pop 8, sigma 0.5 | identical, seed 5,300,000 |
| Fitness | lexicographic: safe completion, completion, violations | `81 * sustained + 9 * success - violated` per episode, summed |
| References | native, untrained draw, random search at matched budget | identical |
| Budget | 40,000 actions | 128 training episodes per search (controls reported) |
| Memory | GRU | none: 5 weights, no state (deliberate; see Stage 2) |

`success` = the success predicate met before any violation; `violated` = violation at
any control including the continuation; `sustained` = success held through the
continuation with no violation. Common random numbers: every candidate in a
generation runs the same two starts, so within-generation ranking is exact on those
starts (the simulator is deterministic per seed in fresh processes); generalisation
is measured on the held-out panel only.

## Arms

1. **CMA-ES.** 8 generations x 8 candidates x 2 starts = 128 episodes. Nominee =
   the final distribution mean after generation 8.
2. **Random search.** 16 draws `z ~ N(0, 0.5^2 I)`, generator seeds 5,200,001 to
   5,200,016, each on the same 8 starts = 128 episodes. Nominee = highest summed
   fitness; ties to the lower seed.
3. **Untrained.** One draw from the same distribution, seed 5,200,000.
4. **Native.** `w = 0`.

## Starts

Training starts come from the Stage 1 calibration panel (development seeds, base
outcome known), by a fixed rule: the first eight base-violated and the first eight
base-safe seeds in ascending order.

| Generation | Base-violated start | Base-safe start |
| ---: | --- | --- |
| 1 | 4320001 | 4320000 |
| 2 | 4320002 | 4320003 |
| 3 | 4320006 | 4320004 |
| 4 | 4320009 | 4320005 |
| 5 | 4320010 | 4320007 |
| 6 | 4320012 | 4320008 |
| 7 | 4320013 | 4320011 |
| 8 | 4320014 | 4320015 |

Random search uses the generation 1 to 4 starts (8 starts). The four base-failure
starts proposed for Gate 2b are generations 1 to 4's violated column, so Gate 2b's
question ("does any adapter complete where the base fails") is answered inside this
pilot.

Evaluation panel: 16 fresh seeds, 4,400,000 to 4,400,015, never used before. All
four arms run on it, native included (16 episodes), so the comparison is paired by
start.

## Budget and timeline

Two lanes on the running host (instance 58666468, RTX 3090); about 40 s per episode
per lane when both run, about 180 episodes per hour.

| Phase | Episodes | Cumulative time from go |
| --- | ---: | --- |
| Code, sync, `pip install cma==4.4.4`, parity smoke (2 episodes) | 2 | 0:25 |
| CMA-ES, 8 generations | 128 | 1:10 |
| Evaluation A: native, untrained, CMA nominee | 48 | 1:27 |
| Random search | 128 | 2:10 |
| Evaluation B: random-search nominee | 16 | 2:16 |
| Analysis, `PilotResults.md` | 0 | 2:25 |

322 episodes, about two hours of simulator time, about $0.90. The order puts a usable
three-row table at about 1 h 30; random search completes the table. No third lane
(the host is CPU-bound at two).

## Pre-registered reading of the result

On the 16-start panel, paired by start, counts with Wilson 95% intervals and the
discordant pairs. No significance claim at n = 16.

- **Primary:** sustained safe success, CMA nominee versus native.
- **Secondary:** violated episodes, CMA versus native; CMA versus random-search
  nominee; untrained versus native.
- **Verdict rules.** *Transfers:* CMA sustained >= native + 3 and CMA violated <=
  native. *No signal:* both differences within +-1. *Mixed:* anything else, reported
  without a claim. *Random-search tie:* |CMA - random| <= 1 on sustained, stated as a
  tie, as the TwoRoom report does.
- Also reported: training trace per generation (best and mean fitness, sigma, mean
  weights), the nominee weights with a one-line reading, controls consumed per arm.

## Implementation (about 25 minutes)

- `probe_worker.py`: add `--genotype-json <file>` to load explicit weights (hash
  recorded in the summary). No other worker change.
- `pilot_search.py`: driver with a two-slot lane pool; phases in the order above;
  writes one attempt directory per episode as Stage 1 did; resumable (skips episodes
  with a `summary.json`); ledger per phase; pycma state saved after every generation.
- `pilot_analyse.py`: writes `PilotResults.md` and `outputs/pilot-20261101/pilot-results.json`.
- `configs/pilot-20261101.json`: this protocol as data, with the freeze, library and
  code hashes.
- Parity smoke before anything else: `w = 0` through `--genotype-json` on starts
  4320005 and 4320002 must reproduce the calibration outcomes and action hashes
  bitwise. Failure stops the pilot.

Records mirrored to `SafeBlock/outputs/pilot-20261101/` (gitignored), host copy in
`outputs/pilot-20261101/`. No credential is sent to the host.

## What this cannot claim

One repeat, sixteen starts, a five-parameter adapter, a fitness without a requested
limit. It can show that the recipe moves safe completion on a second environment, or
that it does not. The budget comparison (feasibility-first versus penalty-weighted
versus Lagrangian, three repeats, 64-start final panel) stays Stage 2.

## Authorisation

Authorised by Sachin on 1 November 2026 at 18:12 UTC ("yes go ahead start the clock and
implement"). Clock start 18:12 UTC. Frozen settings: `configs/pilot-20261101.json`.

## Repeats 1 and 2 (authorised 19:33 UTC, queued behind repeat 0)

Sachin: "Yes please queue those as well." Two further repeats of the protocol above,
run one after the other on the same host once repeat 0 completes. Written before either
starts. Settings: `configs/pilot-20261101-r1.json`, `configs/pilot-20261101-r2.json`;
records `outputs/pilot-20261101-r1/`, `-r2/`.

Identical to repeat 0 except:

| | Repeat 1 | Repeat 2 |
| --- | --- | --- |
| CMA-ES seed | 5,300,001 | 5,300,002 |
| Random-search generator seeds | 5,201,001 to 5,201,016 | 5,202,001 to 5,202,016 |
| Untrained draw seed | 5,201,000 | 5,202,000 |
| Training starts | next 8 base-violated + next 8 base-safe calibration seeds | the following 8 + 7; generation 8 reuses 4320000 (no eighth base-safe seed remained) |
| Evaluation panel | 4,400,016 to 4,400,031 | 4,400,032 to 4,400,047 |

Native runs on every panel. Each repeat is read first by the repeat-0 rules on its own
16 starts. The pooled reading, pre-registered here: pool the three panels (48 starts,
paired by start). *Transfers* if pooled CMA sustained safe success is at least native
plus 9 and pooled violations are at most native's. *No signal* if both pooled
differences are within plus or minus 3. *Mixed* otherwise. Also reported: whether the
sustained difference has the same sign in all three repeats, McNemar's exact p on the
discordant pairs as a descriptive statistic, and the three nominees' weights side by side.
Random search and the untrained draw are read against native the same way.

Cost: about 80 minutes and $0.50 per repeat (repeat 0 took 82 minutes). Expected
completion about 22:30 UTC.

## Outcome (2 November 2026, 00:20 SAST; all three repeats complete 22:17 UTC)

Results: [PilotResults.md](PilotResults.md) (repeat 0), [PilotResults-r1.md](PilotResults-r1.md),
[PilotResults-r2.md](PilotResults-r2.md), pooled in [PilotPooled.md](PilotPooled.md).
966 episodes, 7.8 lane-hours, no failed episode, about $1.40 on the one host.

Pooled over 48 fresh starts, paired by start:

| controller | sustained safe success | violated |
| --- | ---: | ---: |
| Native `ap_rank` | 10/48 | 28/48 |
| Untrained selector | 12/48 | 25/48 |
| Random-search nominee | 7/48 | 30/48 |
| CMA-ES nominee | 10/48 | 26/48 |

Pre-registered pooled verdict: **no signal** (CMA minus native: sustained 0, violated -2;
random-search tie; the sustained difference was +2, +2, -4 across the repeats). Training
succeeded every time (a candidate with sustained safe success on both of a generation's
starts appeared in every repeat), but the three nominees are three different weight
vectors, and none of them carried its training gain to fresh starts. The untrained draws
did as well as the searched controllers.

Reading: with five linear weights over planner-derived segment features, two training
starts per candidate and 128 training episodes, the search overfits its training starts
and the nominee is not better than a random point. This is the Block Plucking counterpart
of the TwoRoom caution (one untrained reference beat its paired CMA run). It says nothing
yet about the risk-budget rule, which was not tested here.
