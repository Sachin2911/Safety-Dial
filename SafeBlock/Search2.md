# Search 2: four fixes to the evolutionary search on Block Plucking

Written 2 November 2026 (00:50 SAST) after the pilot ([Pilot.md](Pilot.md), pooled verdict
"no signal"). Sachin chose ideas 1 to 4 from the post-pilot discussion for now, 5 and 6
(embedding inputs, planner-knob interface) afterwards. Frozen as written before the first
training episode. Settings: `configs/search2-20261102-r{0,1,2}.json`; driver
`experiments/search2.py`; records `outputs/search2-20261102-r{0,1,2}/`.

## What the pilot showed and what changes

| Pilot weakness | Fix | Where |
| --- | --- | --- |
| Two starts per candidate; the search fit its starts | **1.** Eight common starts per candidate, rotating per generation; lexicographic key with continuous tie-breakers (top-block excursion, progress, completion time) so generations are not flat | `fitness_key` |
| Nominee = blended final mean, never validated | **2.** Two nominee candidates (final mean, last elite) scored on a 16-start selection panel; the winner goes to the final panel. Random search gets the same treatment (its top two) | `phase_select` |
| Candidates discarded each generation; training gains did not carry | **3.** The generation's best is force-injected into the next generation and re-scored on new starts (a (mu+lambda) step inside CMA-ES) | `es.inject(force=True)` |
| Five planner statistics as the only inputs | **4.** Three frozen-system scores per candidate: min critic value at the imagined terminal state, world-model failure probability there, ensemble disagreement along the block. Computed on the imagined features the planner already produces; RNG state restored so recorded shadow signals are unchanged | `selection2`, `frozen_candidate_scores` |

Unchanged: the frozen system (Hard, `ap_rank`, library, violation and task-return
definitions), fresh process per episode, neutral start, standardised coordinates,
diagonal CMA-ES (pop 8, sigma 0.5, CSA), random-search and untrained references,
pre-registered reading, three repeats pooled over 48 paired final starts.

## Fitness key (higher is better, compared lexicographically)

1. sustained safe successes, 2. successes at any time, 3. fewer violated episodes,
4. smaller mean of the top block's maximum xy excursion, 5. larger mean progress
(middle-block displacement over the 0.07 m success distance, capped at 1), 6. faster mean
completion among successes. CMA-ES receives within-generation ranks. This is not a
penalty weight: no amount of progress outranks one more sustained success.

## Arms and panels (per repeat)

- **CMA-ES**: 8 generations x 8 candidates x 8 starts = 512 training episodes; nominee
  candidates final mean and last elite; selection panel 16 starts (32 episodes).
- **Random search**: 64 draws (sigma 0.5), eight per block on the matching generation's
  starts = 512 episodes; top two by key to the selection panel (32 episodes).
- **Untrained**: one draw. **Native**: `w = 0`.
- **Final panel**: 16 fresh starts, all four arms (64 episodes). Training, selection and
  final seeds are fresh and disjoint (4,50r,000+, 4,51r,000+, 4,52r,000+); no labelling
  by base outcome, so a generation's eight starts carry about four base violations on
  average.
- Smoke before training: zero-weight `selection2` reproduces two Stage 1 bare episodes
  bitwise, including the recorded shadow-filter signals.

## Budget

| Per repeat | Episodes |
| --- | ---: |
| Smoke | 2 |
| CMA-ES training | 512 |
| Selection A (final mean, elite) | 32 |
| Final A (native, untrained, CMA nominee) | 48 |
| Random search | 512 |
| Selection B (top two draws) | 32 |
| Final B (random nominee) | 16 |
| Total | 1,154 |

About 5 hours per repeat on the present host (two lanes, about 31 s per episode); three
repeats about 15 hours and $5.50 on one host, or about 5 hours on three hosts (plus setup).
CMA-ES with its selection and final panels (about 2.6 hours) lands first in every repeat.

## Reading (pre-registered, as the pilot)

Per repeat on its 16 final starts: *transfers* if CMA sustained >= native + 3 and CMA
violated <= native; *no signal* if both differences within +-1; *mixed* otherwise;
random-search tie if |CMA - random| <= 1. Pooled over 48 starts: +9, +-3, 3. Also
reported: which nominee candidate won selection in each repeat (mean or elite), the
selection-panel keys, the final-panel kinds of failure, and the per-candidate frozen
scores of chosen versus base choices.

## Authorisation

Design and implementation authorised 22:40 UTC 1 November ("let's go with 1-4 now"). The
smoke and one feature-sanity episode run on the present host. The full run awaits
Sachin's confirmation of the budget above.

Full run authorised 23:10 UTC 1 November ("I'll go with your recommendation", "Go for it!"):
three fresh single RTX 3090 hosts with 28 to 48 cores, one repeat each, 4 to 5 lanes, about
$3 in total; the smoke became host-local (bare versus zero-weight on the same host).
Provisioning records: `outputs/vast-setup-20261102/`.

A fourth repeat on the pilot host was proposed at 23:00 UTC and withdrawn minutes later on
Sachin's decision ("stick with the 3"); it ran three smoke episodes and no training episode,
and its records (`outputs/search2-20261102-r3/` on the pilot host) are discarded. The pilot
host is to be destroyed once the data copies to the three new hosts are verified.

**Execution note (23:25 UTC).** The first launch used five lanes per host. Each simulator
process takes about 6 GB of GPU memory, so five overflowed the 24 GB card: two smoke episodes
on repeat 0's host failed with CUDA out-of-memory and a retry that ran under that pressure did
not match its pair bitwise, so the smoke correctly stopped that driver. All three hosts were
stopped and restarted from scratch at three lanes (about 19 GB) at 23:25 UTC; the discarded
five-lane records (a few smoke episodes and, on repeat 1, part of generation 1) are kept on the
hosts under `outputs/discarded-5lanes-*` and are not used. Expected run time per repeat at
three lanes: about 3 hours.

**Cross-host determinism (23:25 UTC).** The host-local smokes passed bitwise on all three
hosts, but the same seed gives a different episode on each host (seed 4320005: 70, 177 and 96
controls; seed 4320002: 23, 26 and 26), so renders and hence trajectories depend on the driver
(615.71, 595.99, 580.173). Every comparison in this study is within one host, so this does not
affect the paired readings; it does mean a seed is not an episode across machines. The pilot
host 58666468 was destroyed at 23:38 UTC after the copies were verified.

**Host speed (23:36 UTC).** Isaac Sim start-up is single-thread bound: median episode 24 s on
the Ryzen 9 7950X host (repeat 1), 57 s on the Xeon E5-2680 v4 (repeat 2) and 71 s on the Xeon
E5-2697 v2 (repeat 0), all at three lanes. Projected repeat times 2.6 h, 6.1 h and 7.6 h.

## Outcome (2 November 2026, 04:25 UTC; stopped early on Sachin's instruction)

Sachin asked at 04:15 UTC to stop within five minutes and destroy the rentals. State at the
stop: repeat 1 complete (1,156 episodes); repeats 0 and 2 complete through the CMA arm,
selection panel A and final panel A (596 episodes each), with their random-search arms about
a third done and lost with the hosts. No episode failed anywhere. Results:
[Search2Results-r0.md](Search2Results-r0.md), [-r1](Search2Results-r1.md),
[-r2](Search2Results-r2.md), pooled in [Search2Pooled.md](Search2Pooled.md).

Pooled over 48 fresh final starts, paired by start (random search: repeat 1's 16 only):

| controller | sustained safe success | success (any time) | violated |
| --- | ---: | ---: | ---: |
| Native `ap_rank` | 9/48 | 27/48 | 28/48 |
| Untrained selector | 16/48 | 29/48 | 24/48 |
| CMA-ES nominee | 14/48 | 21/48 | 30/48 |
| Random-search nominee (repeat 1) | 6/16 | 9/16 | 6/16 |

Pre-registered pooled verdict: **mixed** (CMA minus native: sustained +5, violated +2;
per repeat +1, +3, +1 sustained, verdicts no signal / mixed / no signal). The random-search
tie cannot be read with one repeat. Paired against native: the untrained selector gained
sustained success on 10 starts and lost it on 3 (McNemar exact p = 0.09) and removed more
violations than it added (11 versus 7); the CMA nominee gained 9, lost 4, and added more
violations than it removed (8 versus 6).

**Reading.** The four fixes did what they were built for inside the search: generations were
no longer flat, the selection panel chose between two agreeing candidates, and the injected
elite's re-evaluation rank (0, 7, 6, 3, 4 in repeat 1) showed the generalisation gap directly.
They did not produce a nominee that beats a random draw of the same adapter on fresh starts.
The consistent effect across the study is that *any* re-ranking of the planner's 64
candidates, untrained included, raises sustained safe success over the planner's own argmin
(16/48 against 9/48), and that episodes in which the adapter overrode more choices did
better. The nominee of repeat 2 (the elite) learned the safety-shaped rule one would write by
hand, strongly avoiding candidates the world model expects to fail, and still gained nothing
on fresh starts. The search is therefore not the bottleneck at this adapter and budget; the
information and interface are. Next: a no-search reference that re-ranks at random among the
top candidates (to measure the diversification effect on its own), then ideas 5 (embedding
inputs) and 6 (planner-knob interface).

Cost: about 2,350 episodes across three hosts, roughly $4 for the Search 2 hosts plus the
pilot host's last hours.
