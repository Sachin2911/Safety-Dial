# Roadmap after the 1 November probes

Working plan for the Block Plucking track, written 1 November 2026 after Probes A and
B ([ProbeResults.md](ProbeResults.md)). Companion to [Move32.pdf](Move32.pdf) and
[Rule42](../SafeLeWM/Rule42.pdf). Nothing below is authorised until Sachin says so;
each stage names what it needs.

## Where we stand

- Shared contribution unchanged: risk is a budget, not a penalty. Feasibility-first
  ranking at a requested violation limit, against penalty-weighted evolution and
  Lagrangian RL at matched simulator budgets, then the dial.
- Block Plucking now has the TwoRoom shape. Frozen encoder and dynamics (released
  UNISafe Dreamer), Anchored Planning as the base planner (library = authors' public
  success-only demonstrations), a real failure predicate, an adapter slot.
- On the Hard variant the bare planner completes 5/12 and violates 6/12. Completion and
  risk are in tension. The filtered native actor (3/12, 7/12) and the task actor are
  hand-designed references on the same starts.
- Throughput about 140 episodes per lane-hour with one simulator process per episode.
- LeWM track (Codex): Anchored Planning is the fixed baseline on TwoRoom (20/50
  long-goal); the first hazard family did not qualify. The recipe will be debugged
  wherever violations exist first, which may now be Block Plucking.

## Stage 1: freeze the system and qualify the adapter slot (1 to 2 days)

Goal: a frozen base that later stages never touch, plus evidence that an adapter can
move outcomes both ways and that the optimiser will have signal.

1. **System freeze.** Hard variant with restored terminations; `ap_rank` with the probe
   settings (64 neighbours, lookahead 5, block 5, library rev as encoded); violation =
   native physical failure or boundary exit; task return = sustained safe success
   (success held 40 controls, no violation); horizon 300 + 40. Write `configs/` and
   hashes. No tuning of planner settings after this point.
2. **Adapter interface qualification** (fresh development seeds). Two candidates,
   same information: (a) bounded residual on the planner's seven-coordinate command,
   `a = clip(a_plan + B tanh(u))`, decision every block; (b) anchor selection: a small
   network scores the k retrieved segments and picks which anchor to aim at. For each,
   zero-adapter parity (bitwise identical to bare `ap_rank` on 6 starts) and an
   authority sweep (24 random genotypes x 4 starts): outcomes must change in both
   directions and not collapse to all-violation. Select one family before Stage 2;
   the other becomes an ablation.
3. **Noise calibration.** Bare `ap_rank` on 48 fresh starts to estimate the violation
   rate and its episode-to-episode variance, so the episodes-per-candidate (8 proposed)
   and the requested limits are set from data, not guessed.
4. **Throughput lever.** In-process resets: keep one simulator process across episodes
   (saves the 16 s start) and check the physics-counter and camera-hash identity against
   fresh-process episodes on 6 starts. If it passes, episodes drop to about 10 s
   (about 350 per lane-hour). Vectorised envs are the second lever, only if needed.

Needs: authorisation, this host or an equivalent, about 250 episodes, under $3.

## Stage 2: the first risk-budget comparison (2 to 3 days)

Goal: the matched comparison that Move32 page 2 describes, on the frozen system.

- **Objective.** Feasibility-first: candidates with estimated violation rate at or
  below the requested limit outrank the rest; among them higher sustained success wins;
  among the rest smaller shortfall wins. Two requested limits in the first freeze,
  chosen from Stage 1's calibration (likely about half and a quarter of the base rate).
- **Arms at matched simulator budget.** CMA-ES with feasibility-first ranking; random
  search with the same ranking (cheap control, since random has tied CMA-ES before);
  penalty-weighted CMA-ES at three fixed weights (the method we are arguing against);
  PPO-Lagrangian on the same adapter and information.
- **References.** Bare `ap_rank`; filtered native actor; untrained adapter x3; the task
  9 release rule is no longer comparable (different base) and is cited, not rerun.
- **Panels.** Training seeds in blocks of 8 per generation; a 16-start selection panel
  for nominees; a 64-start final panel opened once, after all selections are locked;
  3 repeats per arm. Fresh seed ranges, all probe and October seeds excluded.
- **Budget.** Population 16 x 6 generations x 8 episodes = 768 training episodes per
  run; 4 arms x 2 limits x 3 repeats = about 18,000 training episodes plus about
  3,000 selection and final episodes. At 350 episodes per lane-hour that is about
  60 lane-hours, about $21 on RTX 3090 hosts; 3 hosts make it one day. Without
  in-process resets, 150 lane-hours.
- **Output.** For each limit: measured violation rate and sustained success on the
  final panel per arm, with intervals, and whether feasibility-first lands near the
  requested limit without a weight sweep. This is the GECCO core result.

Needs: Stage 1 complete, protocol frozen, authorisation for about $25.

## Stage 3: the dial and the ablations (3 to 5 days)

- **Dial.** Four requested limits with the winning ranking rule, one controller per
  limit, the figure: requested risk on one axis, measured risk and completion on the
  other, one curve per method. No monotonicity assumed, limits fixed before running.
- **Ablations**, each a focused run: adapter interface (the family not chosen in Stage
  1); filter shadow versus live under the adapter; episodes per candidate (4, 8, 16) as
  the noise-handling question; regular task instead of Hard (does the recipe transfer
  within the environment); K1 overhead keep-out as a second violation type.
- Budget: about the same as Stage 2.

## Stage 4: two scales, one figure

- Same ranking rule, arms and figure on TwoRoom with Anchored Planning once a hazard
  family qualifies there (Codex track). If TwoRoom never supplies enough violations,
  it stays the development environment for the recipe and Block Plucking carries the
  claim; Move32's "one experiment at two scales" is then stated honestly as one scale.
- Cross-environment table: base rate, requested limits, measured risk, completion,
  per method.

## Stage 5: writing

- **GECCO 2027** (deadline typically late January; confirm the exact date). One focused
  paper: constraint handling under noisy probability estimates. Feasibility-first versus
  penalty-weighted evolution versus Lagrangian RL, on Block Plucking (and TwoRoom if
  Stage 4 lands). Random search ties are reported, not hidden. Draft skeleton after
  Stage 2, full draft after Stage 3.
- **Honours thesis.** Broad story: SafeCEM opening, the October UNISafe study (what a
  hand rule fixes), the probes (why a planner base), the risk-budget comparison, the
  dial. Reuses the same figures.

## Review after Stage 1 (1 November, evening)

What Stage 1 changed, and what it did not.

**Changed.**

1. *Adapter interface.* The residual with generic authority (bounds 0.25 to 0.5,
   N(0, 0.5) initialisation) is destructive: 1 completion in 96, 85 violations, mostly
   boundary exits and top-block displacements. Anchor selection is healthy: 76 of 96
   completions, 69 sustained, 18 violations, five random genotypes with four sustained
   successes and no violation. Unless Gate 2b reverses this, Stage 2's main adapter is a
   selector over the retrieved demonstration segments (Rule42's "target selection"
   candidate), and the residual becomes an ablation at low authority or is dropped.
   A state-conditioned selector (the 26 shared inputs plus the 5 segment features,
   hidden 8, about 265 parameters) should be qualified alongside the 5-parameter linear
   one, because only a state-conditioned scorer can behave differently before and
   after the block is out.
2. *Base rate and limits.* Violation rate 0.52 under the frozen any-control definition.
   Candidate Stage 2 limits are 0.25 and 0.125 (about half and a quarter), with 0.375
   as the fallback upper limit if Gate 2b shows nothing near 0.125. Chosen after Gate 2b.
3. *Episode cost.* In-process resets diverge from control 0 because the renderer keeps
   history; physics does not. Fresh processes everywhere at about 25 s per episode.
   Stage 2 as planned (4 arms x 2 limits x 3 repeats, 768 training episodes per run,
   plus selection and final panels) is then about 21,000 episodes, 146 lane-hours,
   about $51 on 3090s, one day on four hosts with two lanes each. Two levers: a
   render-flush fix for in-process resets (one hour of engineering, 4x if frames can be
   made identical) or trimming to 2 repeats and 5 generations (about $25).
4. *Panel design.* Four starts can all be easy for the base (it happened). Qualification
   panels must include starts where the base fails, chosen by a fixed rule.

**Unchanged.** The shared contribution, the feasibility-first rule, the arms (CMA-ES,
random search, penalty-weighted CMA-ES, PPO-Lagrangian at matched budgets), the
references, the frozen base, the panels, the dial figure, and the writing plan. With a
selector adapter, PPO-Lagrangian maps naturally onto a categorical policy over the 64
retrieved segments, which makes that arm cleaner than before.

**Newly visible risk.** The planner is sensitive to pixel noise below visual threshold;
reproducibility rests on fresh-process determinism, which holds. A 5-parameter selector
may not reach 0.125; the dial could be flat at the bottom. Gate 2b and the limit choice
address this before Stage 2 spends anything.

**Decisions now.** (a) Run Gate 2b with three families (linear selection,
state-conditioned selection, residual at bound 0.1) on four base-failure starts, about
288 episodes, 1.6 hours. (b) Spend one hour on the render-flush test before Stage 2.
(c) Stage 2 shape: 3 repeats at $51 or 2 repeats at $25. (d) Keep the host overnight
(about $8) or destroy and rebuild in 30 minutes when Stage 2 starts.

## Decisions for Sachin

1. Adapter interface to qualify first: residual on the planner command (my
   recommendation, matches LeWM Candidate B) or anchor selection.
2. Requested limits for Stage 2: set from the Stage 1 calibration, or fixed now.
3. PPO-Lagrangian in the first freeze (recommended, as the third arm) or deferred.
4. Host: keep instance 58666468 running (about $8.40 per day, shader cache warm,
   library and datasets in place) or destroy it and rebuild when Stage 1 starts. Either
   way, push the encoded library (0.5 GB) to the private HF dataset first.
5. Commit the SafeBlock work now (nothing is committed yet).

## Risks

- Hard's violation rate for the planner is about 0.5, high enough that a requested
  limit of 0.125 may be infeasible for small adapters; Stage 1 calibration decides.
- In-process resets may change renderer or physics history; the identity check is the
  gate, and fresh processes remain the fallback at 2.5x the cost.
- The world model is out of distribution on Hard (the filter's signals fire on up to
  79% of planner controls). That is part of the setting, not a bug, but anchors from
  regular-task demonstrations may cap what any adapter can reach.

## Review after the pilot (2 November, 00:20 SAST)

The two-hour pilot ([Pilot.md](Pilot.md), three repeats, 48 fresh starts) found no signal:
CMA-ES on the 5-weight selector equals native on sustained safe success (10/48 each) and
is within two violations of it; random search and untrained draws do the same. Training
gains did not carry to fresh starts and the three nominees disagree.

**Changed.**

1. *Adapter capacity.* The 5-weight selector is too small to carry a generalising policy.
   Stage 2 should qualify the embedding-input selector (layer-normed world-model feature
   and each candidate's imagined terminal feature, small hidden layer) before any budget
   comparison, with the pilot design reused as the qualification: three repeats, 48
   paired starts, the same pre-registered rules.
2. *Candidate evaluation.* Two starts per candidate let the search fit its starts. Stage
   2 keeps eight episodes per candidate as planned; the pilot's budget (128 training
   episodes) is far below Stage 2's 768 and should not be read as the recipe's ceiling.
3. *References.* The untrained draw must stay in every table; it matched or beat the
   searched controllers here.

**Unchanged.** The shared contribution, the frozen system, the feasibility-first rule, the
arms, the panels, the writing plan. The pilot is reportable as the second-environment
check that did not reproduce the TwoRoom gain at a matched small budget.

**Decisions now.** (a) Host 58666468 is idle and still billing ($0.351/h, no auto-stop):
keep or destroy. (b) Qualify the embedding selector with the pilot design (about 4 hours,
$1.50) before Stage 2, or go straight to Stage 2 with the 5-weight selector and accept a
flat result. (c) Commit the SafeBlock work (still nothing committed).

## Review after Search 2 (2 November, 06:25 SAST)

Search 2 ([Search2.md](Search2.md)) fixed the search machinery (eight common starts per
candidate, continuous tie-breakers, a selection panel, elite re-evaluation, frozen-system
candidate scores) and still produced a CMA nominee no better than an untrained draw of the same
adapter on 48 fresh starts (14/48 versus 16/48 sustained; native 9/48). Pooled verdict mixed.

**Changed.** (1) The adapter interface and its inputs are the bottleneck, not the optimiser; stop
tuning the search. (2) Any re-ranking of the planner's candidates helps a little; a no-search
random re-ranking reference must be in every future table before any learning claim. (3) Ideas 5
and 6 (embedding inputs via a frozen PCA of the world-model feature; a planner-knob interface
choosing lookahead and neighbour count) are the next qualification, with the Search 2 driver as
is. (4) Hosts: filter Vast offers by single-core speed (cpu_ghz >= 4.5, recent CPU); three lanes
per 24 GB card; renders are driver-specific, so comparisons stay within a host.

**Unchanged.** The frozen system, the feasibility-first rule and the Stage 2 budget comparison,
which still waits for an adapter that qualifies.

**Open.** Nothing is committed. All rentals are destroyed (04:21 UTC 2 November).
