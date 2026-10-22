# Safe manipulation research opportunities

Research date: 8 October 2026.

**The strongest practical opportunities are helping a robot finish after safety intervention,
and maintaining safe task performance when objects or physical conditions change.** Both
start from documented weaknesses in existing systems. Neither requires inventing a new
planning architecture before establishing the problem. Standard evolutionary algorithms
are plausible tools for adapting behaviour and examining trade-offs, although the evidence
does not establish that they are the best tools. A useful contribution could be a clear
empirical result about an unresolved failure, supported by a simple improvement and strong
comparisons. The priorities below are our assessment; proposed evolutionary roles are
inferences, not results reported by the cited papers.

| Practical problem | Evidence and remaining uncertainty | Plausible evolutionary role | Entry burden and priority |
| --- | --- | --- | --- |
| Task stalls after safety intervention | Authors identify a task-policy continuation problem; its presence in the released block-plucking task remains unverified. | Adapt existing task behaviour while evaluating the complete filtered controller. | Medium; first shortlist. |
| Changed objects or physics undermine safety | UNISafe reports difficult physical variations and limits on unseen configurations; aggregate failures do not identify the cause. | Search behaviour that works across bounded variations. | Medium; second shortlist. |
| Harmless novelty causes unnecessary warnings | A manipulation failure detector reports benign-background false alarms. | Tune interacting settings or compare safety/progress trade-offs, if simple calibration is insufficient. | Low to medium offline; secondary. |
| Available computation limits safe control | Model-based safety evaluation can be expensive; the bottleneck is implementation-specific. | Compare ordinary search families at equal budgets. | Conditional on a runnable planner; secondary. |
| Observations omit safety-relevant information | MultiSafe demonstrates a cost of missing information. | No clear EA advantage until the information problem is specified. | Higher; scientifically interesting, less immediate. |

## Task continuation offers a concrete agent-side opening

A safety mechanism can prevent failure while leaving the robot unable to complete its
job. In the September 2026 preprint *When World Models Lie*, the authors attribute some
remaining incompletions to a nominal task policy that was not trained to resume from states
produced by safety intervention. Their online adaptation addresses world-model mismatch,
but the task policy remains a separate limitation. **Avoiding failure and finishing the
task are distinct outcomes.** ([Paper, section V-B](https://arxiv.org/html/2609.34300))

This is a stronger research starting point than assuming that a more sophisticated planner
is needed. The open practical issue is whether useful task behaviour remains available
after intervention, and why the existing controller fails to use it. However, the cited
explanation comes from another manipulation setup. An incomplete UNISafe block-plucking
episode could instead reflect inaccurate predictions, an impossible configuration, a poor
task policy or an insufficient horizon. A timeout alone does not diagnose bad continuation.

Some nearby solutions already exist. LatentCBF smooths safety interventions and trains its
safety value using mixed nominal-policy and safety-policy experience. It nevertheless
acknowledges that filtering can induce unfamiliar states and reduce task performance.
**Training that safety value is not automatically training the nominal task policy to
resume.** This distinction leaves a focused question without pretending that intervention
smoothness or mixed-policy data are untouched ideas.
([LatentCBF, sections 5 and 7](https://arxiv.org/html/2511.18606))

Evolution has a natural, modest role here: search task-policy or motion parameters using
the outcomes of the complete filtered controller. CMA-ES is a candidate when that search
space is compact. Ordinary policy fine-tuning with the filter present, improved filtering
alone, and simple task-specific continuation rules remain credible alternatives. The
contribution would be evidence about task-policy compatibility with safety intervention,
not the mere replacement of gradient learning with evolution. This is medium effort after
the baseline runs; useful post-intervention training data have not been verified in the
release.

## Bounded physical robustness is the second strong candidate

UNISafe's harder block-plucking evaluation varies block size, weight and friction. For its
diffusion-policy combination, Table 15 reports **38% safe success, 31% failure and 31%
incompletion**. These figures describe that setting and policy, not every UNISafe variant.
The authors also acknowledge that entirely different tower configurations cannot be
reliably filtered. This establishes concrete headroom, without locating the fault in the
world model, uncertainty estimate, filter or task controller.
([UNISafe, appendices D.2 and E.2](https://arxiv.org/html/2505.00779))

The accessible question is robustness within a declared task family and meaningful range
of physical changes. Evolution could search existing behaviour parameters across those
conditions, using CMA-ES or a multiobjective method if several operating trade-offs matter.
A useful result would identify a reproducible weakness, improve safe completion, and show
where the improvement stops working. Standard domain-randomized policy tuning and robust
planning would be important comparisons.

Robustness itself is established territory. Robust cross-entropy planning already combines
constraints with model uncertainty. A manipulation safety-filtering paper already handles
uncertain mass and friction, evaluates critical transitions and uses safe physical probing.
Therefore, adding uncertainty, a worst-case score or a probing action is not sufficient
novelty. The opportunity is a bounded, informative result in learned-model manipulation,
with medium extra burden for controlled variations and independent evaluation.
([Robust planning](https://learn-to-race.org/workshop-sl4ad-icml2022/assets/papers/paper_16.pdf);
[manipulation uncertainty](https://arxiv.org/html/2509.12674))

## Warnings and computation are secondary unless they explain lost progress

**Unnecessary warnings are a documented problem.** A world-model failure detector for
bimanual manipulation acknowledges false alarms caused by benign background changes. Its
authors also identify autonomous-policy evaluation as future work. This suggests an
approachable diagnostic question: which unfamiliar changes warrant intervention, and
which merely interrupt useful work? Its limited hardware evaluation does not establish
that the same false-alarm mechanism dominates every manipulation filter.
([Bimanual failure detector, sections V-B and VI](https://arxiv.org/html/2603.06987))

An EA could examine trade-offs among missed dangers, unnecessary interventions and task
completion when several settings interact. For one or two thresholds, simple sweeps are
more appropriate baselines. Threshold optimization alone is a weak contribution; a stronger
result would explain a meaningful warning failure and demonstrate improved robot outcomes.
UNISafe already uses ensemble disagreement, and its adaptive successor already responds
to observed prediction error. New uncertainty scoring should be assessed against that
existing work. Offline diagnosis is relatively accessible; prevention claims require
closed-loop evidence.
([UNISafe, section 4](https://arxiv.org/html/2505.00779);
[adaptive filter, section IV](https://arxiv.org/html/2609.34300))

**Computation becomes a research problem when it changes safety or progress.** LatentCBF
reports a substantial gap between repeated model-based action evaluation and its learned
safety critic. Those measurements concern its DINO-WM implementation, so they do not imply
that all latent models are too expensive. They also show that a cheaper learned evaluator
is already a serious alternative to more elaborate search.
([LatentCBF, Table 3](https://arxiv.org/html/2511.18606))

A fair comparison of constrained CEM, CMA-ES and quality-diversity search at limited
budgets could be useful if it reveals a consistent control failure. GuSS already combines
learned dynamics with MAP-Elites-based safe action planning, and ELVIS uses mixture-based
latent MPC to retain alternative futures. M-QD and DA-QD already use models to support
repertoire search. Consequently, neither diverse imagined candidates nor selective
simulator validation is an untouched contribution. The open question must be more specific
than which optimizer obtains the largest average reward.
([GuSS](https://arxiv.org/abs/2206.09743);
[ELVIS](https://arxiv.org/html/2605.04709);
[M-QD](https://arxiv.org/html/2008.04589);
[DA-QD](https://arxiv.org/html/2109.08522))

**Missing information is a deeper fifth problem.** MultiSafe's empty-bottle example permits
safe tilts in 18% of trials with calibrated RGB alone, versus 60% with multimodal
information. More optimization cannot generally distinguish physically different situations
that provide identical observations. This direction remains attractive, but new sensing,
history, task design or model training could dominate its cost. There is no reason to
force an evolutionary mechanism onto it before identifying what information is missing.
([MultiSafe, sections 5-6](https://arxiv.org/html/2510.06492))

## Released assets determine what is actually low effort

Here, low effort means avoiding unnecessary new components after a working baseline is
established. It does not mean that a checkpoint download guarantees reproduction. The
current asset comparison favours UNISafe for immediate safety/task evidence, while other
models offer different advantages.

| Starting point | Available foundation | Material remaining work |
| --- | --- | --- |
| UNISafe block plucking | World-model/filter assets, task policy, trajectories and native success/failure fields. | Original Isaac Sim 4.2 stack needs migration for a 5090; checkpoint behaviour and post-intervention data need validation. ([Release](https://github.com/CMU-IntentLab/UNISafe), [wrapper](https://github.com/CMU-IntentLab/UNISafe/blob/isaaclab/dreamer_wrapper.py), [NVIDIA compatibility guidance](https://forums.developer.nvidia.com/t/isaac-sim-4-5-not-working-on-windows-11-via-wsl2/366463)) |
| TD-MPC2 / Meta-World | Many task-matched 5M-parameter checkpoints and a latent-control implementation. | Released Meta-World interface uses state observations; a dedicated safety cost was not verified. ([Models](https://www.tdmpc2.com/models), [wrapper](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/envs/metaworld.py)) |
| LeWM Cube | Visual weights, goal-planning configuration and about 46 GB compressed expert data. | Safety specification, readout and validation remain necessary. ([Weights](https://huggingface.co/quentinll/lewm-cube/tree/main), [data](https://huggingface.co/datasets/quentinll/lewm-cube/tree/main)) |
| MultiSafe and LatentSafe successors | Relevant methods and demonstrations. | MultiSafe lists code as forthcoming; the current latent-CBF release documents a ready Dubins-car pipeline, not a verified ready arm package. ([MultiSafe](https://cmu-intentlab.github.io/multisafe/), [latent CBF](https://github.com/CMU-IntentLab/latent_cbf)) |

Keeping selected components frozen can bound work and clarify comparisons. It is an option,
not a scientific requirement or novelty claim. A new task or training stage remains
reasonable if it addresses the chosen problem. This assessment is not restricted to the
previous Walker experiments or thesis deadline.

## Practical judgment

**A standard EA can support a worthwhile contribution without a new planning theory.**
What matters is identifying an actual limitation, explaining why the intervention helps,
and assessing safety alongside useful progress at comparable interaction and computation
cost. Credited feasibility-first CEM and SafeDreamer are existing baselines, not new
constraint-handling ideas. Simple comparisons can be valuable when they answer an
unresolved question; optimizer swaps and threshold tuning alone usually do not.
([Constrained CEM](https://papers.neurips.cc/paper_files/paper/2018/hash/34ffeb359a192eb8174b6854643cc046-Abstract.html);
[SafeDreamer](https://arxiv.org/html/2307.07176v3))

The practical shortlist is therefore task continuation and bounded robustness, while
keeping the other problems available if diagnosis makes them more compelling. This is
problem selection, not a commitment to a method or proof of novelty. Several important
sources, including *When World Models Lie*, LatentCBF and ELVIS, are preprints; no GPU
runs or implementation checks were performed here. Supporting detail is preserved in
the [documented problems](../research_notes/Safe%20manipulation%20research%20opportunities/documented_open_problems.md),
[evolutionary fit and prior work](../research_notes/Safe%20manipulation%20research%20opportunities/evolutionary_fit_and_prior_work.md)
and [assets and entry cost](../research_notes/Safe%20manipulation%20research%20opportunities/assets_and_entry_cost.md).
