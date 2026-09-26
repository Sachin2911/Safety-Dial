# Continuation protocol audit: feasible routes, retention and matched experience

This is an implementation audit, not a new experimental result. Historical outcome files
and the active bank-generation job were preserved.

## Development feasibility

`pushT.md`, problem definition, requires: "Tune the generator on development cases that
have a demonstrated feasible route, then freeze it. Never drop a test case because a
method fails on it." Clear start/goal footprints and nominal-path crossing alone do not
establish a feasible route.

`pusht_development_witness.py` now searches a declared bounded list of recorded development
branches and source-expert continuations. A passing artifact contains at least three
independent development source episodes. Each witness uses its actual frozen bank hazard,
checks the nominal swept path and widest-margin start/goal clearance, remains clear of the
whole-T hazard at every observed environment step, reaches at least 0.90 geometric goal
coverage, and repeats exactly from reset and prefix. Actions, states, actual simulator
polygons, failed attempts, charged steps, bank hashes and generator hashes are retained.
No test root is selected or filtered using these outcomes.

E2 and E5 require the passing artifact for the identical development bank and generator.
The separate `configs/pusht/continuation-after-e1.yaml` schedules this check before E2.
The check validates the already frozen generator. If it fails, it is not an impossibility
certificate. Any subsequent generator tuning requires fresh frozen banks and cannot
retroactively validate old test outcomes.

## Retention

`pushT.md` E2 requires held-out expert prediction error, "20 fixed goal-reaching episodes
without hazards", and ordinary motion; E4 explicitly includes retention. The prior
helper accepted incomplete or differently configured episode horizons and paired root IDs
without verifying distinct source episodes.

New comparisons require at least 20 distinct source episodes and root IDs, exact case
identities, and the same declared maximum horizon of at least 50 five-step blocks. Raw
`censored` and `censored_future` mark an unexecuted suffix. An early episode can nevertheless
complete the no-hazard task only through `completed_on_verified_goal`: actual environment
termination, no truncation or arena exit, the installed simulator's combined pusher/block
xy error below 20 and periodic angle error below pi/9, and independently computed whole-T
goal coverage of at least 0.95. Environment termination alone does not establish coverage;
the installed environment does not use its `success_threshold` attribute in `eval_state`.
Timeouts, unverified terminal outcomes and unresolved censored episodes cannot pass.

E2 saves and hashes the exact case file. `pusht_retention_eval.py --cases` reuses those
bytes for later repaired checkpoints. E5 binds the selected weight hash, E2 case-file
hash, per-row case identities and E2 horizon and recomputes the paired comparison.

`pusht_transfer_retention.py` covers the largest acquired budget of every reported arm and
seed and, when distinct, the budget selected by the E5 qualification gate. It evaluates
one common released baseline and every actual selected checkpoint with the frozen cases.
Failures remain reported; intermediate checkpoints do not inherit a goal-retention claim.
Neither these runs nor passing scientific outcomes are asserted by this implementation audit.

## Typed contact and interpretation

The upstream `n_contacts` includes wall collisions. New observer-backed banks separately
store `pusher_block_contacts` and `block_wall_contacts` plus availability. Legacy counts
remain unchanged and are labeled `any_collision`; their pusher-contact value is unknown,
not false. Unknown rows enter neither the pusher-contact nor free-motion stratum and
cannot support a contact-specific dynamics claim. E2 root/branch metadata uses the new
observer. The already running continuation-v2 E1 process retains its original imported code and
aggregate counters, so its contact-specific mechanism remains unavailable. Future fresh E1
collections now select the passive body-specific observer; nullable unknown counters are
handled explicitly without changing active artifacts.

## Walker S5

`walker2d.md` S5 asks whether "the same experience" helps more during pretraining and
compares A adaptation "with N steps" against B's N replaced ordinary steps. The initial
implementation used B's 512 additional branches while A's imported S4 checkpoints also
used 128 common seed branches: 51,200 versus 64,000 transitions. Disclosure alone does not
control that difference. The S5 comparison now fits separate A variants on exactly the N
selected acquired transitions, excluding the common seed, and checks exact experience
identity and row counts. The original S4 seed-plus-budget study is preserved. Matching
regressions are covered by the Walker comparison tests.
