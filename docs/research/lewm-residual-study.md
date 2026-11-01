# LeWM residual mixture: implementation and development screen

Sachin authorized implementing all four proposed next steps on 30 October 2026:
measure proposal influence, strengthen its search role, compare candidates on a
common panel, and test for a completion benefit before expansion. The prospective
[protocol](../../configs/lewm_residual_20261030/protocol.json) and
[launch receipt](../../progress/receipts/20261030-lewm-residual-launch.json) were
written before any episode. This is a fresh allocation with a $1 cap and one-hour
cutoff. The provider rejected the 3090 and RTX 4000 requests without allocating
either pod. Pod `qwm6ejjm54ar0e` uses an A40 at $0.59 per hour, with a 00:28:25 SAST
cutoff on 31 October. Provider credentials stay local.

The same 510-parameter network now emits a bounded correction to the current CEM
centre, beginning with the shifted previous plan. In every iteration it shifts half
of the newly sampled candidates, rounded down. The native mean slot and retained
archive are preserved. The candidate count, search iterations, frozen LeWM,
learned native-proprioception motion model, 15 px pusher and 6 px margin remain
fixed. Both candidate branches pass the same learned feasibility check and final
execution-time check. The passive monitor records actual dense violations.

The zero adapter has seeded hidden features and exactly zero output weights and
bias. It must reproduce native actions, states, dense traces and model-query work.
A separate seeded untrained control has random readout weights. Diagnostic records
count guided samples, feasible guided samples, their elite membership, their role
in the best sampled plan and the magnitude of the correction. Retained candidate
provenance carries across iterations and replans. A final mean may combine both
branches; these membership counts do not establish a causal contribution by
themselves.

Nine initial diagnostic episodes compare native, zero and untrained control on
three known geometries. Training opens only if zero exactly recovers native and the
nonzero controller changes at least one action trace and enters an elite set.
Technical failures stop for diagnosis; safe refusal and non-completion remain valid
outcomes. Earlier evidence is preserved unchanged, including the original planner
used for software equivalence checks.

All four CMA-ES generations use the same four development cases. Population six and
four updates evaluate 24 genotypes in 96 episodes. Parameter random search evaluates
24 independent genotypes from the same initial Gaussian distribution on the same
panel, with the same feedback and per-episode caps. Native, zero and untrained
development controls add twelve episodes. The training elite is chosen using
comparable panel scores, with earliest generation and index breaking exact ties.
No final distribution mean is deployed automatically.

Selection compares native, zero, untrained, CMA elite and random-search elite on a
different common four-case panel, adding twenty episodes. Exact fitness ties prefer
the listed arm order. Total maximum valid episodes are 233, with at most two failed
technical starts additionally charged to the limits. Actual actions, model queries,
discarded candidates, reset steps and elapsed time remain separate counters;
matched caps do not imply equal actual work.

Expansion requires zero selection violations, strictly more CMA safe completions
than native, zero and untrained, and at least native engagement. Progress alone
cannot open it. A passing screen would require a newly frozen next-stage protocol
for fresh scenes, optimizer repeats, tuned fixed settings and ordinary learning
before any evolutionary advantage claim. A failing screen closes this iteration
without opening fresh-scene validation or authorizing another rental. This pilot
uses one known source scene; fresh planner seeds do not establish generalisation.

The implementation and development screen are complete. Nineteen rented-GPU
software checks passed, including exact recovery of the preserved earlier planner
over multiple replans, controlled guided-elite selection, all-unsafe refusal,
controller serialization and common-panel checkpoint recovery without duplicate
jobs. The first software suite passed 18/19: one test incorrectly assumed a random
candidate population must select a guided elite. Its repaired version uses a
controlled quadratic objective and candidate population with a known better guided
direction. Both source versions and test logs are retained; no simulator episode
was affected.

The real diagnostic passed. Native and zero had identical trajectories and model
work on all three cases. The nonzero untrained controller changed actions in all
three and contributed 2,379 fitted elite slots. This establishes influence, not an
efficacy claim. Later local verification found exact native/zero actions, states,
dense physics, images and goal images in all eleven paired diagnostic, development
and selection cases. A final mean can combine branches, and provenance labels also
exist for zero-correction allocations; branch membership is not a causal benefit.

All 233 scheduled episodes completed without technical failed starts. CMA-ES made
four real updates and evaluated 24 genotypes; random search evaluated 24 independent
genotypes. Each used the same four-case panel in 96 training episodes. The native
and zero development controls both scored 4/4 safe completions, 0/4 violations,
0/4 unfinished and 4/4 engaged, using 130 actions. This panel had a native completion
ceiling. The CMA elite, generation 1 candidate 5, retained 4/4 safe training
completions in 111 actions. The random-search elite was candidate 23. Training
efficiency did not establish transferred completion benefit.

On the separate four-case selection panel, native and zero each scored 2/4 safe
completions, 0/4 violations, 2/4 unfinished and 4/4 engaged, using 350 actions.
The CMA elite also scored 2/4 safe completions, 0/4 violations, 2/4 unfinished and
4/4 engaged, using 355 actions. The seeded untrained controller scored 3/4 safe
completions, 0/4 violations, 1/4 unfinished and 4/4 engaged, using 244 actions.
Random search scored 3/4 safe completions, 0/4 violations, 1/4 unfinished and 4/4
engaged, using 268 actions. Untrained won the declared fitness selection. This is
a small development observation on four planner-seed cases from one source scene,
not a replicated generalisation result or proof that untrained guidance is superior.

The evolved controller did not improve completion, so the predeclared expansion
gate failed. Fresh-scene validation, additional optimizer seeds, a new fixed-setting
search and ordinary-learning comparisons stayed closed. These conditional later
studies were not run or claimed complete. No evolutionary advantage is established.
The
[results receipt](../../progress/receipts/20261030-lewm-residual-results.json) retains
all outcome denominators, common-panel nomination, selection and limitations.

Across the heterogeneous diagnostic, control, candidate-search and selection
episodes, totals are 151/233 safe completions, 0/233 observed violations, 82/233
unfinished and 233/233 engaged. These correlated episodes are not one efficacy
test. Actual work was 17,342 environment actions, 173,420 execution physics steps,
466 reset physics steps, 59,328 model batch calls, 13,154,238 model candidate rollouts
and 39,462,714 predictor steps. Episode runtime summed to 1,058.1 seconds, excluding
setup, software checks, orchestration and preservation. CMA training used 7,702
actions and 17,468,931 predictor steps; random search used 7,347 actions and
16,771,113 predictor steps. Candidate/panel counts and caps matched; actual work
did not. Sampled training policies are not independent final-performance trials.

Independent audits verified frozen source and assets, every saved residual against
its weights and features, dense safety, native goals, actual interaction counts,
all candidate genotypes and panel fitness, common-panel elite nomination and the
selection rule. A separate audit recomputed the completion gate directly from
episode outcomes. All
[2,510 archive members](../../outputs/lewm-residual-20261030/release/lewm-residual-manifest.json)
were downloaded and hash-verified, with zero configured-credential matches. All
sixteen
[paired development videos](../../outputs/lewm-residual-20261030/remote/results/residual-v1/media/provenance.json)
decoded successfully. The
[CMA gallery](../../outputs/lewm-residual-20261030/remote/results/residual-v1/media/gallery.png)
and the
[untrained extra completion](../../outputs/lewm-residual-20261030/remote/results/residual-v1/media/selection-untrained-bar-s304121.mp4)
were inspected. Original frames, all source versions, environment records,
optimizer checkpoints and the
[CMA controller](../../outputs/lewm-residual-20261030/remote/results/residual-v1/controllers/cma.json)
are preserved locally. No external upload was performed.

Pod `qwm6ejjm54ar0e` was deleted after verification. Provider get and list confirmed
absence at 23:58:11 SAST on 30 October, and both shutdown guards retired. The
[closeout receipt](../../progress/receipts/20261030-lewm-residual-pod-closeout.json)
records a conservative rental estimate of $0.296 under the fresh $1 cap, not an
invoice. No other rental changed. This completed protocol authorizes no additional
experiment or lease extension.
