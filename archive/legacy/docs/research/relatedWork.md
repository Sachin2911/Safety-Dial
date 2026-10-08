# Related work for targeted safety experience

Literature checked 21 September 2026. This supports the
[adopted research direction](../researchDirection.md); it is not a second experiment plan.
Recheck the closest papers before making novelty claims.

The candidate contribution is a controlled comparison of which additional experience
improves transferable safety decisions, holding the adaptation mechanism and evaluation
fixed. Failure probes, fine-tuning, active data selection and collecting unsafe examples
already have precedents.

## Closest overlaps

1. **FARM, 10 September 2026.** It freezes a VLA-JEPA backbone and extracts failure information with a small supervised readout. It compares fixed-readout transfer, expanded source supervision and readout-only adaptation, using trajectory-disjoint evaluation. **Exact overlap:** asking whether pretrained predictive states already contain accessible failure information, and whether more readout data improves transfer. **Difference to test here:** action-conditioned future constraint decisions on counterfactual branches, plus a controlled acquisition intervention that may repair prediction itself. FARM studies execution monitoring; changing the word “failure” to “safety” would not establish novelty. [FARM](https://arxiv.org/html/2609.11445v1)

2. **AdaJEPA, 30 June 2026.** It adapts selected encoder and predictor layers during MPC from recent transitions. Its ablations already compare predictor-only and joint adaptation. It also varies training-data amount and shape diversity while fixing optimisation steps; diversity helps unseen-shape planning. **Exact overlap:** pretrained JEPA adaptation, deciding which component to update, and data quantity versus diversity. **Difference to test here:** selecting experience to improve one-sided constraint-decision errors at a matched interaction budget, with safety and retained task progress as outcomes. Its evidence does not make a fresh generic adaptation or diversity claim available. [AdaJEPA, Sections 4.3-4.4 and Appendix B](https://arxiv.org/html/2606.32026v1)

3. **ReDRAW, 3 April 2025.** It learns a latent dynamics residual while freezing the pretrained model. It explicitly studies collection policies and dataset size: diverse source experience helps transfer, target expert data usually helps more reliably than random actions, and too little target data can overfit. **Exact overlap:** a small residual repair and studying which experience supports it. **Difference:** the proposed safety study concerns misleading accepted predictions under existing dynamics, rather than transfer between changed source and target dynamics. Its fully observable setup can include otherwise hidden velocities, so its sample-efficiency results are not directly transferable to a pixel-only partially observable setting. [ReDRAW, Sections 3 and 5.1.3](https://arxiv.org/html/2504.02252v1)

4. **FOSP, first posted 6 July 2024; revised 2 March 2025, ICLR 2025.** It studies safe offline-to-online world-model RL, including visual navigation and robotic reaching. Section 5.3 already varies safe/unsafe data balance and dataset size, measuring resulting policy behaviour. **Exact overlap:** safety outcomes depend on the composition of world-model training data. **Difference:** a component-isolated, fixed-controller audit of counterfactual acquisition can identify why decisions change, rather than attributing the combined behaviour of a changing model, critic and policy to the model alone. “More unsafe data helps safety” is too broad and could even be wrong if it induces excessive conservatism. [FOSP](https://arxiv.org/html/2407.04942v2)

5. **WMPO, first posted 12 November 2025, ICLR 2026.** It adapts a pretrained pixel world model using the base policy's rollouts because expert demonstrations underrepresent its failures. It compares budgets of 128 and 1,280 real trajectories in its manipulation experiments. **Exact overlap:** collecting nonexpert outcomes to improve a pretrained model's failure predictions. **Difference:** the proposed work compares which branches to acquire and measures constraint prediction separately from policy improvement. Its generative VLA system is much larger than the proposed LeWM experiment; its reported policy improvements do not identify an optimal safety-data acquisition rule. [WMPO, Sections 3.2 and 4](https://arxiv.org/html/2511.09515v1)


## Interpretation rules

- Physical-state probes use supervised labels. Applying a supplied hazard rule does
  not demonstrate that the model discovered safety without labels.
- Predictor-only gains with a frozen encoder concern imagined dynamics. Encoder
  adaptation is a different experiment and needs a common fresh-probe protocol.
- A weak probe is not proof that information is absent from the representation.
- A released checkpoint supports a study of additional experience, not a causal claim
  about the data that originally produced its representation.
- A retrospectively selected set of known failures is an oracle data-usefulness
  comparison. Prospective acquisition must choose before seeing each actual future
  and pay for every simulator query, including root generation and replay prefixes.
- Keep the same relabelling policy, repair architecture, training budget and retained
  task-performance checks across acquisition arms. New hazards alone do not establish
  a new constraint-parameterisation method; AnySafe already studies that question.

Useful positive or negative findings include simple boundary coverage matching a learned
selector, contact diversity beating repeated severe failures, or prediction improving
when current-state information was already readable. No finding is assumed in advance.
The reviewed sources do not certify that the precise comparison is absent everywhere.

Additional context: [AnySafe](https://arxiv.org/html/2509.19555v1),
[UNISafe](https://arxiv.org/html/2505.00779v1), and
[The Intervention Gap](https://arxiv.org/html/2608.29998v1).
