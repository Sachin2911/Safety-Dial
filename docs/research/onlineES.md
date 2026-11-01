# Safe Online-ES with a fixed safety filter

29 October 2026. Exploratory idea from discussion with Sachin.

This note records a candidate research question. It does not change
[firstPrinciples.md](../../firstPrinciples.md), freeze an experimental protocol or
authorize implementation, simulator interactions or rentals.

## Research question

Can Online-ES help a VLA learn productive behavior under a fixed safety filter,
increasing safe task completion beyond the original policy with the same filter?
Would evolutionary exploration add value over ordinary learning with the same
information and matched interaction budgets?

The motivation is the gap between avoiding harm and finishing. Task 19's Jar shield
reduced observed violations without increasing safe completion, with more stalls
after closure. The [consolidated record](../../principia.pdf) retains the evidence
and its limitations. A learner might discover useful alternative movements where
a fixed filter alone repeatedly blocks progress; this remains a hypothesis.

## Candidate approach

1. Start from an existing task-fine-tuned VLA checkpoint and reproduce its native
   behavior under the benchmark protocol.
2. Freeze the vision-language backbone initially, while allowing the action expert
   to learn. This is adjacent to the current fully frozen-policy controller question.
3. Generate action chunks and explore small perturbations, then pass candidate
   actions through a fixed safety filter before execution.
4. Keep ManiGuard's original monitor separate and unchanged. Record task completion,
   benchmark and requested-sequence safety, violations, unfinished episodes and
   engagement, with denominators.
5. Use execution feedback for action-expert updates. A simple proposed baseline gives
   positive feedback only to safe completions and negative feedback otherwise.
   This objective does not independently constrain the violation rate or guarantee
   safe exploration. Filter thresholds must not be relaxed to obtain more success.

[Online-ES](https://arxiv.org/html/2609.38855v1) explores action trajectories and
transfers feedback into model parameters through a weighted regression objective.
It is a hybrid of evolutionary exploration and gradient-based learning. The paper
uses GR00T N1.5 with GeoVLA-style depth features and updates only its action expert.
Porting it to pi0.5 or ManiGuard's GR00T N1.6 would require separate validation.

## Why examine pi0.5 and AEGIS first

[AEGIS](https://arxiv.org/html/2512.11891v2) already filters pi0.5 actions for
end-effector collision avoidance using control barrier functions. Its authors
identify safety-induced distribution shift: filtering can take the robot into
unfamiliar states where it fails to recover toward the goal. This gives a concrete
learning problem to investigate. Switching to GR00T is not required for this idea.

[AnySafe](https://arxiv.org/html/2509.19555v1) is an alternative for image-specified
constraints. It requires a learned world model, a failure representation, calibration
and a trained safety policy. Its hardware study uses planar sweeping with
teleoperation. Adapting those components to Jar would introduce additional learning
and validation problems; its published setup does not directly enforce Jar's LTLf
requirements.

## Open technical questions

- **Constraint coverage:** Jar requires remaining upright throughout and staying on
  its support until closure. End-effector collision avoidance alone does not establish these
  properties. Specify which constraints the filter can enforce, using what state,
  history and dynamics assumptions. The benchmark monitor labels violations and
  never vetoes actions. See [ManiGuard](https://arxiv.org/html/2608.17386v1).
- **Action compatibility:** ManiGuard uses joint-position targets; AEGIS's published
  formulation modifies end-effector motion. Verify the conversion and preserve
  native behavior when the filter is inactive.
- **Learning attribution:** retain the native proposal, exploration perturbation,
  filtered command and executed action separately. Filtering changes execution;
  using corrected actions as targets changes the original Online-ES update. Define
  and justify the estimator for the filtered system before implementing it.
- **Feasibility:** verify a differentiable action-generation path, output parity and
  gradients before learning. Handle the executed eight actions per sixteen-step
  chunk, action normalization, clipping and smoothing explicitly. Determine whether
  safe positive examples are available and measure actual training memory and cost.

## Evidence needed

The central comparison is native policy plus filter against learned policy plus the
same filter. Success-only Online-ES versus safety-aware Online-ES would isolate the
feedback change. Any evolutionary claim also requires native, tuned fixed,
parameter-random-search and ordinary-learning controls with the same allowed
information and matched interaction budgets. Account for filter development and
training, failed starts, discarded candidates, model queries and elapsed time.

[FailBank](https://arxiv.org/html/2609.39820v1) is close prior work: it learns from
counterfactual filter corrections using guarded LoRA updates, with an observe-only
teacher during collection. It does not use Online-ES, but overlaps the motivation.
Combining existing methods is not yet an established novelty claim.

The next intellectual step is to map each Jar constraint to what a candidate filter
can actually control, and identify how learning could preserve progress. Any later
study needs its own prospective protocol, development panel, seeds, budget and
stopping rules, with reserved tasks untouched and comparisons blocked by host.
