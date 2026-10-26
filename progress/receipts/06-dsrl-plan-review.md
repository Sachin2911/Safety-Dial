# DSRL final planning review

25 October 2026. Review of [06-dsrl-plan.md](../06-dsrl-plan.md). Sachin selected
potential-shaped safe completion and a hard $15 cap. The final plan replaces the
earlier $45 allocation and flat unsafe penalty. The independent feedback has been
incorporated; rental and execution await a launch instruction.

## Scope and scientific interpretation

The allocation is one learner with 24 training episodes, six engineering episodes
and 48 evaluations, twelve per native, fixed-zero, bounded-random and DSRL arm.
Two additional starts are reserved for infrastructure failures. A second learner
with sparse reward is deferred. Recording sparse returns on the trained learner's
trajectories does not constitute a reward ablation.

This is an integration and learning pilot with a small descriptive outcome panel.
A positive result can nominate a development replication; a negative result cannot
separate limited training, representation, steering authority and reward design.
One learner seed and twelve evaluations per arm do not support a generalization,
reward-shaping or evolutionary-advantage claim.

## Review findings addressed

- **Objective:** terminal base reward is 1 for strict safe requested completion and
  0 otherwise. Raw monitor violations and requested-order failures remain visible
  even when the benchmark's contact gate does not count them.
- **Shaping:** a fixed bounded potential rewards changes in latched first closure,
  subsequent grasp, legal carry and the instantaneous native goal predicate. The potential
  is zero after an irreversible safety/order failure and at every valid terminal.
  It is scorer information, not a privileged actor or critic observation.
- **Discount:** gamma is 0.9998 per executed control, with each transition using
  gamma raised to its actual duration. This preserves approximately the source's
  end-of-horizon discount over the longer Jar episode and still favors faster
  completion. The shaping scale is fixed at 0.25.
- **Return check:** adding 0.25 times the initial potential to the discounted shaped
  return must recover the discounted base return. Include success, timeout, partial
  chunks, intermediate violations and nonzero initial potential. The identity
  preserves the base environmental return, not SAC convergence or safety guarantees.
- **Inactivity:** repeated closure cannot earn positive additional episode return.
  Advancement requires completion gains on at least two tasks, engagement and no
  observed increase in raw or counted violations.
- **Source fidelity:** reuse the official learner and actual launcher defaults,
  including 10 critics, mean reduction and stochastic actor evaluation. The reward,
  terminal masks, horizon discount and three-episode collection cadence are adaptations.
- **Comparison:** fresh controls run on the same host and a frozen development panel,
  with three episodes per arm per GPU. Zero noise is not native; random noise per
  query is not parameter random search; fixed zero is not a tuned best setting.
- **Accounting:** ceilings are 80 starts, 160,000 policy controls, 800 explicit
  warm-up controls, 21,000 VLA query attempts and 120,000 learner updates. Failed
  attempts remain charged and do not reset the ledger.
- **Money:** at most $13 active compute/disk plus $2 transfer/other storage, within
  a hard $15 total. Rate is capped at $2.20/hour including disk, aggregate rental
  at six hours or the money deadline, whichever is earlier. The computed deadline
  stops admissions and compute; no automatic top-up is allowed.
- **Feasibility:** prepare integration locally before renting. Engineering must finish in
  90 minutes and demonstrate that the entire fixed pilot fits the remaining
  allowance, including updates, startup and preservation. Otherwise finish with
  engineering evidence; do not silently shrink the comparison or exceed $15.
- **Claims and held-out data:** retain the prior baseline/diagnostic exposure of
  held-out tasks in the record and run no new held-out intervention episodes.
  Replication, stronger baselines and equal information/budgets precede evolution.

## Checks required during future implementation

Validate the expected 16 by 32 latent hook, native equivalence on identical inputs
and RNG state, state/image normalization, action attribution and isolated streams.
Check the reused learner against the pinned source on a fixed batch. Audit existing
predicate definitions and causality, per-control safety history, terminal masks and
the shaping identity. Count model and controller parameters. Verify checkpoint,
replay and budget resume. Measure three-worker/one-learner resource use and all
preservation charges. Freeze exact jobs, sources, assets, full inherited defaults
and analysis before the eligible stage. These are future implementation checks;
no simulator or learner tests have been run for this planning task.

## Independent review and response

The independent reviewer found no blocking defect in the shaping identity, discount
conversion, control design or accounting. The following pre-launch design issues
have now been addressed:

1. **Workstation shape audit completed.** The exact named serving configuration
   at ManiGuard 6bf4c61 explicitly sets action_horizon = 16, action_dim = 32 and
   pi05 = true. Three small checkpoint metadata files match the earlier pinned
   provenance by Git blob SHA-1. The [audit receipt](06-dsrl-shape-audit.json) records
   source hashes and the static method; no JAX import, weight load, neural query,
   physics or rental was required. Runtime shape assertions remain a backstop.
2. **Authority probe frozen.** The [probe specification](06-dsrl-decoder-probe.json)
   fixes fourteen magnitude-1 tiled vectors, eight rule-selected native pool
   snapshots and 128 total decoder queries. The minimum is robust opposite gripper
   commands on at least two snapshots from two tasks, plus all native identity
   checks. The vectors are generated and hashed on the workstation. Absent authority
   stops engineering before E4/E5 or any development learning; a larger bound or
   additional directions would require a new plan. The archived additive-noise
   results are cited as prior evidence, with their different mechanism and scope.
3. **Cold starts bounded explicitly.** The first attempt per GPU has 40 wall minutes;
   later attempts have 25. Startup remains charged. Cold starts are staggered and
   learner compilation follows the initial renderer/policy compilation. The 90-minute
   engineering clock belongs to the host running engineering; failed hosts still
   count toward six aggregate rental hours and the same $15 cap. Cold/warm admissions
   reserve 60/45 minutes respectively before the global deadline.
4. **Authority wording aligned.** AGENTS.md now requires native, tuned fixed,
   parameter random search and ordinary learning before any evolutionary claim,
   while permitting this earlier development-only ordinary-learning pilot.

Closure is now latched once observed, matching requested-order scoring; the raw
current predicate remains recorded. The plan uses raw violation for the strict
objective and reports the benchmark's contact-gated counted label separately.
The author's -1-per-unsuccessful-transition, 0-on-success step cost is described
explicitly. We retained zero bootstrap and zero potential at timeout as a declared
finite-episode departure from the source's continuing bootstrap. No privileged
clock was added to the actor or critic; partial observability remains a limitation.
Engineering reference/parity/profiling updates now have a separate 2,000-update
ledger inside the unchanged global cap; ordinary post-warm-up training is at most
90,000 updates.

One interpretation in the review needs correction: closure improvements alone cannot
raise the completed, corrected discounted shaped return. The potential terms cancel,
so that return equals the discounted base return. Intermediate reward, potential
occupancy and critic estimates can improve while completion does not. The plan now
states this distinction. Unsafe failure and a safe stall both have zero base reward;
the separate violation screen is not an optimized constraint or a guarantee against
unsafe exploration.

The reviewer projected approximately 4.1 hours and $8.80 in active rental cost, before
transfer/storage, using assumed learner throughput. This is a feasibility estimate,
not a measurement; stage A must replace it with actual timings. The reviewer's
probability estimates are subjective and are not used as statistical inputs or gates.
The likely durable output is an interpretable integration/learning diagnostic or
ordinary-learning comparator; a small positive screen still requires replication.

## Documentation checks

Completed with the local venv's Python and existing LaTeX tools:

- All 85 checked local links resolve, and `git diff --check` passes.
- Budget arithmetic confirms 24 training episodes, 48 evaluations, 80 maximum starts,
  160,000 policy controls, 800 warm-up controls and 120,000 maximum learner updates.
  Training, engineering and evaluation seed ranges are disjoint. Each evaluation
  arm has three episodes per GPU.
- The reference six-hour compute/disk cost is $12.813336; allocations sum to $15.
  The money watchdog includes a $0.10 shutdown margin inside the $13 allowance.
- Five numerical traces covering success, timeout, violation, nonzero initial
  potential, partial chunks and repeated closure satisfy the shaping-return identity
  to below 1e-12 in float64. These are formula checks, not learner or simulator tests.
- The selected horizon discounts are 0.9998^2000 = 0.670293 and
  0.999^400 = 0.670186, as stated in the plan.
- SHA-256 comparisons against the pre-finalization snapshot found no unexpected
  changes to completed evidence, frozen panels, rental records, superseded originals
  or the other session's research drafts.
- The Odyssey compiled twice without final warnings. Its previous five pages have
  identical extracted text. The updated sixth page was rendered and visually checked,
  and shows the selected $15 scope.

Post-feedback checks also verified the static source hashes and metadata matches,
all fourteen distinct magnitude-1 vectors and both vector-file hashes, the 128-query
allocation, and the 92,000-update upper bound for engineering plus training within
the global cap. A latched closure-flapping trace satisfies the return identity with
zero numerical error. Cold/warm admission reserves cover their 40/25-minute episode
limits plus 20 minutes for preservation. All 299 snapshotted progress and unrelated
research files either match exactly or are the explicitly revised planning/log files.
The updated six-page Odyssey compiled twice without warnings, its first five pages
retain identical extracted text, and its sixth page was rendered and visually checked.
No decoder inference, simulator execution or rental was performed.

The independent prompt is [06-review-prompt.md](https://github.com/Sachin2911/Safety-Dial/blob/1291199e665ed574050de42a7708c19965ecda35/progress/06-review-prompt.md).
