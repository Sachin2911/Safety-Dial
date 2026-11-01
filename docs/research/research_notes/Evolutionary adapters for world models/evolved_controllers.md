# Evolved controllers with learned world models

Research checked 30 October 2026. These notes concern persistent controller parameters, not online action-sequence evolution or planner-parameter tuning. No experiments were run.

## What did World Models actually demonstrate?

### Takeaway

There is a strong successful precedent for evolving a small persistent controller after learning a world model. The CarRacing controller was trained in the actual simulator; the separate VizDoom controller was trained inside a learned simulator and transferred back. [Primary paper/project](https://worldmodels.github.io/)

### Cited Findings

- Ha and Schmidhuber, *Recurrent World Models Facilitate Policy Evolution*, NeurIPS 2018, separates a learned visual encoder V, recurrent predictive model M and linear controller C. CMA-ES evolves C's weights; C maps the current latent z and M's hidden state h to actions. It does not consume predicted future z directly. [Project, Controller Model](https://worldmodels.github.io/)
- CarRacing used an 867-parameter controller after representation learning, with evolutionary fitness from actual-environment rollouts. Reported evaluation reward was 906 +/- 21 over 100 tracks. This is task reward, not a separately constrained safety metric. [Project, Car Racing](https://worldmodels.github.io/)
- VizDoom used 1,088 controller parameters and evolved in a learned latent environment that also predicted death. At dream temperature 1.15, actual-game survival averaged 1,092 +/- 556 steps over 100 trials; random achieved 210 +/- 108. [Project, VizDoom and temperature results](https://worldmodels.github.io/)
- Dream exploitation was substantial: temperature 0.10 produced virtual score 2,086 +/- 140 but actual score 193 +/- 58. Increasing model stochasticity reduced exploitation. [Project, model exploitation](https://worldmodels.github.io/)
- Official author code is available in `carracing` and `doomrnn`; its README identifies historical dependency requirements. A current recreation would need deliberate environment porting. [Author repository](https://github.com/hardmaru/WorldModelsExperiments)

### Inferences

- This supports the basic architecture of the proposed adapter: pretrained predictive representations plus an evolution-trained compact controller. It does not establish a hazard-conditioned proposal adapter, a safety-constrained planner initializer, or effectiveness on LeWM.
- Survival in VizDoom is relevant to avoiding danger, but differs from completing an independent manipulation goal while never breaching an explicit clearance margin.

### Gaps

- The work has no matched ordinary-learning comparison for our exact controller, safety predicate and model-query budget.
- It does not show that new arbitrary hazard geometries can be supplied after training without retraining the controller.

## Which subsequent or complementary results are useful?

### Takeaway

Useful precedents include model-based CMA-ES policy search in robotics, CMA-ES on improved learned representations, and model-assisted population learning. These substantiate the components but change important assumptions relative to a frozen LeWM proposal adapter. [Black-DROPS](https://arxiv.org/html/1703.07261v2), [ASR](https://proceedings.mlr.press/v162/huang22f/huang22f.pdf), [GDA-QD](https://arxiv.org/html/2211.12610v1)

### Cited Findings

#### Black-DROPS: the strongest robotics policy-search complement

- Chatzilygeroudis et al., *Black-Box Data-efficient Policy Search for Robotics*, IROS 2017. It learns Gaussian-process dynamics/reward models and searches parameterized policies with uncertainty-handling BIPOP-CMA-ES. Models are updated between physical episodes, unlike a permanently frozen LeWM. Its neural policies have one hidden layer with ten units. [Full primary paper, Sections IV-V](https://arxiv.org/html/1703.07261v2)
- Across 120 cart-pole replicates, its neural-policy version reported 93.33% success, versus PILCO's 92.5%. The paper describes roughly five four-second episodes to find a working policy in the median trend. It additionally evaluates a physical four-DOF reaching arm with ten replicates. These are task-performance results, not explicit hazard-constrained safe-completion rates. [Primary results, Table I and Section VI](https://arxiv.org/html/1703.07261v2)
- The official repository explicitly warns that some runs fail completely and that low physical-interaction time does not imply low computation time. [Author code and limitations](https://github.com/resibots/blackdrops)

#### Action-Sufficient State Representations: the quality of adapter inputs matters

- Huang et al., *Action-Sufficient State Representation Learning for Control with Structural Constraints*, ICML 2022, learns compact representations and includes a World Models-style CMA-ES controller comparison. The controller population is 64, fitness averages sixteen rollouts, and checkpoints every 25 generations are tested over 1,024 random rollouts. Its CarRacing comparison reports a best average score 65 reward units above its World Models comparator. [Published primary paper, Section 5.1 and Figure 8](https://proceedings.mlr.press/v162/huang22f/huang22f.pdf)
- Its title's structural constraints concern representation structure, not obstacle-clearance safety. The paper also uses ordinary RL algorithms elsewhere; do not attribute all reported ASR results to CMA-ES. [Primary paper, Sections 2 and 5](https://proceedings.mlr.press/v162/huang22f/huang22f.pdf)

#### Deep Neuroevolution of World Models: related, but not a frozen-model adapter

- Risi and Stanley, *Deep Neuroevolution of Recurrent and Discrete World Models*, GECCO 2019, evolves the visual encoder, recurrent model and controller jointly using a GA. It reports CarRacing reward 903 +/- 72, with generalization evaluation on 100 random tracks. Three independent evolutionary runs were used for each principal treatment. [Primary paper, Sections 4-5](https://arxiv.org/html/1906.08857v1)
- The final 903 result followed a further 200 generations after the original 1,000-generation treatment, with changed mutation and elite-evaluation settings. This is useful evolutionary precedent, not a frozen-stage analogue of our prospective experiment. Evolving only the controller over randomly initialized encoder/recurrent features failed in their treatment. [Primary results](https://arxiv.org/html/1906.08857v1)
- Official code: [sebastianrisi/ga-world-models](https://github.com/sebastianrisi/ga-world-models).

#### GDA-QD: model-assisted persistent populations, with gradients and model updates

- Lim, Flageat and Cully, *Efficient Exploration using Model-Based Quality-Diversity with Gradients*, arXiv preprint first submitted 22 November 2022, evolves populations of closed-loop policies in learned dynamics, verifies promising policies in the environment, and combines parameter perturbations with critic-based policy-gradient updates. Dynamics and critic models are updated from collected transitions. [Primary paper, Sections 3-4](https://arxiv.org/html/2211.12610v1)
- It tests PointMaze, AntTrap and AntMaze with fifteen optimizer repeats. AntTrap median final QD-score was 76.28% versus MAP-Elites 42.44%; these are quality-diversity scores, not safe-completion percentages. [Primary paper, Table 1](https://arxiv.org/html/2211.12610v1)
- The algorithm is neither pure ES nor a small adapter around a fixed latent world model. It is useful evidence that model-based evaluation can support reusable evolved policies, not direct evidence for hazard-safe LeWM adaptation. Current author-group code cites a 2023 ALIFE publication; do not call it ICLR 2023 solely from aggregator metadata. [Primary manuscript](https://arxiv.org/html/2211.12610v1), [author-group code](https://github.com/adaptive-intelligent-robotics/imagined-qd)

### Inferences

- The strongest direct conceptual reading is World Models, followed by Black-DROPS for uncertainty-aware model-based evolutionary policy optimization.
- ASR suggests that choosing decision-relevant features may matter more than selecting a larger controller or a different optimizer.
- GDA-QD motivates retaining multiple controllers if a single policy collapses to one behaviour, but adds substantial model/critic machinery beyond our proposed small adapter.
- Risi and Stanley is valuable GECCO context, but using it as evidence for a frozen-model safety adapter would conflate different methods.

### Gaps

- I did not verify a reliable public implementation for the 2022 Huang et al. ASR paper. A similarly named ICML 2026 GitHub project is a different paper and must not be substituted.
- This search found no verified direct successor that simultaneously uses frozen LeWM, evolutionary proposal-adapter training, explicit new hazards, a learned clearance checker and matched ordinary-learning controls. This is a bounded search result, not proof of novelty.

## Do these papers establish that our hazard adapter will work?

### Takeaway

They establish that evolutionary training of persistent controllers with learned predictive models can work. They do not establish the complete proposed safety-adaptation method. [World Models](https://worldmodels.github.io/), [Black-DROPS](https://arxiv.org/html/1703.07261v2)

### Cited Findings

- World Models supplies successful persistent controller evolution but directly outputs actions; it does not train a proposal network for a subsequent safety-constrained planner. Its low-temperature failure illustrates why imagined fitness alone is insufficient evidence. [Primary project](https://worldmodels.github.io/)
- Black-DROPS is closer to robotic control and handles model uncertainty, but trains models iteratively and optimizes reward without our fixed external-hazard predicate. [Primary paper](https://arxiv.org/html/1703.07261v2)
- GDA-QD evaluates reusable policies across learned and actual dynamics but uses known benchmark reward/descriptor machinery and no separate strict hazard-completion endpoint. [Primary paper](https://arxiv.org/html/2211.12610v1)

### Inferences

- For our proposal, the persistent genotype should be adapter weights, and the phenotype should produce action proposals or corrections. Online action-sequence evolution is a separate optional component.
- Supplying hazard geometry explicitly is necessary in the current project setup, because the hazards are virtual and not painted into LeWM observations. A representation-only controller would otherwise lack the changed constraint information.
- If the adapter is trained only in imagination, our motion and latent predictors must jointly support its closed-loop inputs, goal signals and termination semantics. Existing open-loop candidate scoring does not automatically supply that training environment.
- A model-trained adapter can exploit predictive errors even when a separate checker rejects obviously unsafe commands. Actual executed completion and violations on reserved cases remain necessary measurements.
- A fair ablation compares the same planner with and without the evolved adapter, then compares evolution with other ways of training the same adapter. Report training costs, online prediction work and physical interactions separately.

### Gaps

- No found result supplies a universal safety guarantee or demonstrates arbitrary newly supplied hazard transfer for this adapter.
- Small controller size alone does not determine training difficulty, optimizer advantage or completion under our available time budget.
