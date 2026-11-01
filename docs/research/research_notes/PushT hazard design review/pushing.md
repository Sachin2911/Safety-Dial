# Pushing benchmarks and hazard design

## Does original PushT define a conventional hazard layout?

### Takeaway
Original PushT is a manipulation task with arena boundaries, not an internal obstacle-avoidance benchmark. Its task score should not be confused with the goal-state rule used in our LeWM experiments. [Original environment code](https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/diffusion_policy/env/pusht/pusht_env.py)

### Cited Findings
- The original Diffusion Policy environment constructs four boundary walls, a circular pusher of radius 15, a T-shaped block and a target pose. Its setup contains no added internal hazard region. [Environment setup](https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/diffusion_policy/env/pusht/pusht_env.py)
- Task reward is clipped target coverage divided by 0.95; termination requires coverage above 0.95. This is an area-overlap objective, not our joint pusher/block position and angle rule. Supporting code excerpt: `done = coverage > self.success_threshold`. [Reward and termination](https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/diffusion_policy/env/pusht/pusht_env.py)

### Inferences
- We should describe added hazards as our experimental extension. There is no standard internal PushT hazard shape established by this original implementation.
- A comparison to a safety extension using coverage requires explicit reporting of the different completion rules.

### Gaps
- This inspection establishes the original environment's construction, not a comprehensive absence claim about all subsequent PushT variants.

## Is there a directly relevant frozen-policy PushT safety precedent?

### Takeaway
Fiedler, Vahl and Zhang provide a close precedent for adding rectangular restricted regions to pretrained PushT Diffusion Policy without retraining. Their restricted-action checks are weaker than executed-trajectory safety certification. [Paper](https://tams.informatik.uni-hamburg.de/people/fiedler/publications/constrained_diffusion.pdf)

### Cited Findings
- They test left workspace strips occupying 10% or 50% of workspace width, and central square restrictions with sides 10% or 20% of workspace size. The action workspace is 512 by 512. [Section III](https://tams.informatik.uni-hamburg.de/people/fiedler/publications/constrained_diffusion.pdf)
- They use pretrained original weights, matched 25 scenes and 200 action steps. Metrics are a coverage-derived task score and counts of action samples inside the restricted area. They do not report our joint dense-path safe-completion endpoint. [Section III](https://tams.informatik.uni-hamburg.de/people/fiedler/publications/constrained_diffusion.pdf)
- The paper explicitly acknowledges the approach's “inability to plan around constraints”. The author's publication list places this work at German Robotics Conference 2026. [Paper](https://tams.informatik.uni-hamburg.de/people/fiedler/publications/constrained_diffusion.pdf), [author publication list](https://tams.informatik.uni-hamburg.de/people/fiedler/)

### Inferences
- A finite central rectangle is a well-motivated simple diagnostic for our setup. The paper is evidence of precedent, not proof that a particular placement has a feasible pushing solution.

### Gaps
- The reported restriction is on pusher action points. This does not establish full pusher-disc or T-block-footprint avoidance, nor validate the complete path between sampled actions.

## What does contact-aware pushing research add?

### Takeaway
Neve et al., ECC 2024, treat collision avoidance and feasible pushing maneuvers together, using circular or ellipsoidal obstacles. This supports geometric simplicity while showing why a free point-agent route is insufficient evidence of a feasible manipulation route. [Paper](https://paperhost.org/proceedings/controls/ECC24/files/0245.pdf)

### Cited Findings
- Their optimizer jointly determines a pushing trajectory and a sequence of changes in pushing direction through clutter. These switches address cases where maintaining one pushing contact direction is insufficient. [Abstract and Section III](https://paperhost.org/proceedings/controls/ECC24/files/0245.pdf)
- Section III.D.2 models obstacles as circles or ellipsoids; ellipse dimensions account for both obstacle and slider geometry. This is more informative than testing only the slider centre against the unexpanded obstacle. Supporting excerpt: “account for both the geometry of the obstacle and the slider”. [Section III.D.2](https://paperhost.org/proceedings/controls/ECC24/files/0245.pdf)
- Their nonlinear program enforces obstacle constraints at trajectory collocation points. It does not supply a certification method for our nonconvex T-block in Pymunk. [Section III.D](https://paperhost.org/proceedings/controls/ECC24/files/0245.pdf)

### Inferences
- Our first proposed family should use one compact circle, ellipse or rectangle with generous room around it. A second family can use a broad offset gate; narrow corridors and concave traps should come later.
- Before freezing evaluation cases, obtain an independent successful simulator replay for each proposed hazard placement, using the exact monitored bodies, margin and action budget. Keep these engineering witnesses separate from candidate-method outcomes and charge their interactions.
- Choose between a pusher workspace restriction and an object protection zone explicitly. For a pusher disc, account for its physical radius before adding a requested clearance. For the T-block, use transformed component geometry and orientation, not only its centre. These are design proposals, not findings that the reviewed papers establish for our runtime.

### Gaps
- None of these sources determines appropriate hazard dimensions or guarantees success for the frozen LeWM checkpoint. Those require prospective engineering under a new frozen protocol.
