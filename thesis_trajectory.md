# Thesis and Trajectory: Irreversibility from Asymmetric Geometry in JEPA Latents

Sachin Mohan · Supervised by Geraud Nangue Tasse · University of the Witwatersrand

---

## 1. The question underneath everything

A JEPA is trained to keep what is predictable and discard the rest. Safety needs to see a bad
outcome before it happens. So the foundational question of *Safe JEPA* is:

> **A JEPA throws information away. Is the information safety needs still in there?**

Every latent safety filter to date runs on a reconstructive world model, where the answer is
trivially yes. Nobody has asked it of a non-reconstructive latent.

## 2. The thesis

Irreversibility is the safety concept most naturally grounded in dynamics rather than in
human labels: a transition is unsafe when no action sequence returns from it. Irreversibility
is exactly the **asymmetry** of dynamical distance, d(z → z′) ≠ d(z′ → z). This asymmetry is a
*quasimetric*, and a JEPA latent regularised toward an isotropic Gaussian is symmetric by
construction. The thesis claim:

> **Asymmetric temporal structure is recoverable from JEPA latents by a quasimetric head
> trained only on timestamps, it survives isotropic regularisation because SIGReg constrains
> the marginal and not the temporal geometry, and it yields a label-free failure set for a
> latent safety filter with a provable false-safe bound in terms of quasimetric error.**

The Safety Dial follows: conservatism becomes a threshold on this quantity, set at deployment
by an operator, not at training time by an engineer.

## 3. Research question and hypotheses

**RQ.** Can the failure set required by a latent safety filter be derived from irreversibility
read off the asymmetric geometry of a JEPA latent, without failure labels, and does the result
retain the filter's safety guarantee?

- **H1 (Recoverability).** A quasimetric embedding fit on frozen JEPA latents, supervised only
  by inter-frame step counts, produces an asymmetry score whose AUROC against ground-truth
  irreversibility exceeds a symmetric-distance baseline and a forward-dispersion baseline.
- **H2 (Compatibility).** The quasimetric head can be trained jointly with the JEPA objective
  without degrading SIGReg's isotropy or downstream planning success.
- **H3 (Substrate dependence).** A web-pretrained encoder (V-JEPA 2) retains asymmetry for
  failure modes absent from robot data; a single-environment encoder (LeWM) does not.
- **H4 (Guarantee).** Filter false-safe rate is bounded by a function of quasimetric
  approximation error and predictor Lipschitz constant; the bound is non-vacuous empirically.

## 4. The one claim the paper makes

> Asymmetric temporal structure survives isotropic regularisation and is recoverable from a
> frozen JEPA latent; it degrades with predictor horizon at a characterisable rate; and
> web-pretrained encoders retain it for failures never seen in robot data where
> single-environment encoders do not.

Everything in the project either serves this claim or is cut.

## 5. Trajectory

Ordered by dependency, not by calendar. Each stage terminates in a reportable result.

### Stage A — Recoverability on a frozen latent
Fit an Interval Quasimetric Embedding on LeWM Push-T latents using (z_t, z_{t+k}, k) triplets.
Compute asymmetry r = d(z′→z) − d(z→z′) along held-out trajectories.
- **Kill test:** if r is flat across known one-way transitions, the thesis is dead cheaply.
- **Confound control (mandatory, designed first):** expert data is directionally biased. Fit
  on reversed trajectories and on a random-policy subset; asymmetry that flips or vanishes is
  policy artefact, not physics.

### Stage B — Ground-truth irreversibility and AUROC
Define irreversibility from the simulator: a state is irreversible within horizon H if no
sampled action sequence from a fixed distribution returns within ε of the prior state.
Report AUROC of r, of symmetric latent distance, and of forward dispersion. Report the
horizon-degradation curve on predicted latents ẑ_{t+1…t+H}.
- **Control:** joint-limit / arena-boundary states (control authority reduced, nothing broken)
  must not score as irreversible.

### Stage C — Substrate comparison
Same quasimetric head, same environment, two encoders: LeWM (end-to-end, single-env) and
V-JEPA 2 with a small action-conditioned predictor trained on sim data. Environment must
contain genuine irreversibility: OGBench-Cube (cube off table) is primary; LIBERO/ManiSkill
Franka tasks are alternatives if V-JEPA 2 needs a richer scene.
- **Control:** verify by inspection which failure modes each training set actually contains;
  otherwise the comparison measures data coverage, not representation.

### Stage D — Joint training
Add the quasimetric loss to the LeWM objective and backprop into the encoder. Measure SIGReg
statistic, probing accuracy, planning success, and asymmetry AUROC against the frozen variant.
Only run if Stage A shows asymmetry is weak on frozen latents; if frozen works, this stage
becomes an ablation and LeWM's one-hyperparameter property is preserved.

### Stage E — Filter and bound
Threshold r to define the failure set; run a latent safety filter (UNISafe-style, HJ value
in latent) on top. Derive and empirically check the false-safe bound. Compare against the
supervised filter given failure labels for mode A, both evaluated on mode B.

### Stage F — Safety Dial
Sweep the threshold; report irreversible-failure rate against task success from one trained
model. Conformal calibration of the threshold on held-out data, with the exchangeability
caveat stated (offline calibration covers the data distribution, not the deployed policy's).

## 6. Positioning

| Line of work | Relation |
|---|---|
| Latent safety filters (Nakamura, Seo, Bajcsy) | Framework being modified; supervised baseline |
| Quasimetric RL, Hilbert representations, contrastive RL | Temporal-distance learning; none on a JEPA, none for safety |
| Reversibility-aware RL (Grinsztajn et al.), relative reachability | Same criterion, arrived at from the alignment side |
| LeWM, V-JEPA 2, DINO-WM | Substrates; the end-to-end vs pretrained axis is the experiment |
| "Model uncertainty fails as a risk signal" (2026) | Live debate; asymmetry is a non-uncertainty alternative |
| ROSARL (Nangue Tasse et al.) | Nearest local work; removes the weight, this removes the concept |

## 7. Known limits, stated up front

- Irreversibility ≠ harm. Task progress can be irreversible. Scope: environments whose
  harmful failures are the irreversible ones.
- Horizon-bounded: "cannot return within H" is what is measured, not "cannot return ever."
- Rotation is weakly encoded in current JEPA latents; toppling-type failures are a blind spot.
- Rollouts near the failure boundary are where the predictor is least trustworthy; the
  filter's uncertainty axis may need to gate the irreversibility estimate.
- Simulation only. The irreversibility argument is strongest as a physical-systems argument.

## 8. Success criteria (decreasing necessity)

1. A characterisation of whether asymmetric structure exists in JEPA latents, with the
   directional confound controlled. Reportable either way.
2. AUROC ≥ 0.8 for label-free irreversibility on at least one environment with genuine
   one-way failures.
3. A non-vacuous false-safe bound with matching empirical curve.
4. Substrate comparison showing a representation-level difference between end-to-end and
   web-pretrained encoders.
5. Zero-shot failure-mode transfer beating a supervised filter on the same data.

## 9. One-sentence self-description

> I work on safe model-based RL, specifically on reading the definition of "unsafe" off the
> asymmetric geometry of JEPA world-model latents instead of off human labels, and on making
> the resulting conservatism a deployment-time control.
