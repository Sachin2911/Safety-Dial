# First principles

Established 22 October 2026; status reconciled 29 October with evidence through
27 October. The goal remains a compact evolved controller that improves safe task
completion of frozen ManiGuard policies under matched interaction budgets. Jar is
the chosen family; Stack is the backup. [AGENTS.md](AGENTS.md) carries engineering
and research controls. [Principia](principia.pdf) is the consolidated experiment record.

## Current evidence

The baseline established useful Jar variance: 24 benchmark safe successes and
43 counted violations in 104 episodes, with engagement in 103/104. Repeated scenes
and seeds can yield different trajectories.

The Jar retry, execution-feedback, DSRL and execution-knob studies did not establish
a replicated completion lever. Task 6 failed numerical gates; task 7 ran learning
and failed its completion screen; task 11 stopped at simulator startup. Task 15's
screen was promising, but final recovery-cohort confirmation remains unverified.

Tasks 16 and 18 established useful fixed planning configurations at unchanged
margins in TwoRoom, Reacher, PushT and Cube. Feedback added no established benefit.
Task 17 completed an optimizer comparison without establishing an evolutionary
advantage; eight of ten evolution runs selected shared preset anchors. These
LeWM studies are separate from the original ManiGuard objective.

Task 19's Jar shield had 32/107 strict safe completions in each arm and reduced raw
violation episodes from 37/107 to 21/107. More episodes stalled after closure.
The secondary endpoint, design-time inspection, host differences and absent optimizer
controls limit inference. CMA-ES stayed near its hand-chosen start. The UNISafe
grasp-sensing ablation confirmed useful information in a separate Dreamer system.

## Four preconditions, in order

1. **Variance:** native execution sometimes completes safely and sometimes does not.
2. **A lever:** a controller improves safe completion, beyond merely changing actions.
3. **Stable evaluation:** sufficient task and learner repeats distinguish gain from noise.
4. **Optimizer advantage:** evolution beats tuned fixed, parameter random search and
   ordinary learning with the same information and matched interaction budgets.

Jar stage 1 is established. A replicated ManiGuard completion lever remains missing.
Fewer violations alone does not complete stage 2. Freeze each new stage before its
eligible episodes, including panels, seeds, information, controller domain, selection,
budget, optimizer repeats, analysis and stopping rules. An unspent allocation or
negative result does not authorize extensions.

## Controls that remain in force

- Reproduce the paper protocol: 20 Hz, eight executed actions per sixteen-step chunk,
  256-pixel collection-time cameras, Jar horizon 2,000, ten-step goal hold and three
  policy seeds. Record deviations.
- Score benchmark and requested-sequence outcomes, raw and counted violations,
  unfinished episodes and engagement with denominators. Jar closure precedes carry.
  Under independence, zero events in n trials gives an approximate 3/n upper bound.
- The benchmark monitor labels and never vetoes commands. An intervention shield is
  separate. Keep proposed, filtered and executed actions distinguishable.
- Charge simulator interaction for engineering, failed starts, sweeps, discarded
  candidates, tuning and selection. Report actual work, model queries, learner updates
  and elapsed time separately. Equal episode caps do not ensure equal work.
- Keep comparisons by physical host. Use uncertainty over task and optimizer repeats;
  correlated rollouts are not independent evidence. Movement need not be useful progress.
- Keep [panels.json](progress/panels.json) frozen. Held-out tasks have baseline and
  descriptive evidence, but receive no new intervention episodes before the final
  comparison and are unavailable for method selection.
- Preserve original evidence, and use fresh files and output paths for new studies.

## Current work

Repository consolidation and manuscript polishing are the current work. Task 7's
closeout and task 15's missing final confirmation remain unresolved records, not
launch instructions. No current rental status is inferred here.

Completed protocols and all pre-22 October studies are historical. This cleanup
authorizes no new experiment or rental. The prior plan, old task documents and
original status wording are preserved in the [recovery bundle](archive/20261029/README.md).
