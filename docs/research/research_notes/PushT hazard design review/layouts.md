# PushT geometry and feasible hazard layouts

## What geometry should the safety predicate protect?

### Takeaway

The protected body must be explicit. Pusher-centre avoidance, full-pusher clearance,
and object-safe manipulation are different tasks. For a clear safe-manipulation
demonstration, I recommend monitoring both the pusher disk and the T-block footprint,
while retaining separate labels for which body violated the constraint.

### Cited Findings

- Classical configuration-space construction accounts for body shape by combining
  the obstacle with the reflected body shape. This supports obstacle inflation for
  a translating disk; it does not justify treating an arbitrarily rotating T-block
  as a point with the same fixed margin. LaValle, *Planning Algorithms*, sections
  [4.3.2](https://msl.cs.illinois.edu/~lavalle/planning/node161.html) and
  [4.3.3](https://msl.cs.illinois.edu/~lavalle/planning/node165.html).
- The installed pinned PushT environment gives the circular pusher default scale
  40 and radius `0.375 * scale`, hence 15 px. Default variation randomizes start
  positions and the block angle, not pusher scale. Task 18 constructs this class
  without geometry overrides. Local sources:
  [environment](../../../../.venv/lib/python3.11/site-packages/stable_worldmodel/envs/pusht/env.py)
  and [runtime](../../../../experiments/helpers/lewm18_runtime.py).
- `add_tee` fixes its scale to 30. Its two polygons are a 120 by 30 px crossbar and
  30 by 90 px stem, forming an unrotated 120 by 120 px bounding box. This is a
  geometry fact, not a proof of the minimum passable doorway width. Same
  [local environment source](../../../../.venv/lib/python3.11/site-packages/stable_worldmodel/envs/pusht/env.py),
  `add_tee`.

### Inferences

- **Derived geometry:** retaining a 60 px straight-sided opening while protecting
  a radius-15 pusher and requesting 12 px extra clearance leaves a centre interval
  only `60 - 2 * (15 + 12) = 6` px wide at the opening. The existing centre-only
  margin does not already include that radius.
- For the T-block, use transformed polygon distance/intersection at its actual
  orientation. A bounding circle is a conservative alternative, with potential
  false infeasibility. A bounding-box width is not a necessary doorway condition:
  the existing band is only 16 px thick and a rotation/threading maneuver may
  matter.
- State whether clearance is Euclidean distance. Uniformly expanding rectangle
  coordinates produces square corners; a disk Minkowski expansion has rounded
  corners. Both can be defined consistently, but they are different predicates.
- Adding the object to the executed monitor does not automatically upgrade the
  current analytic pusher-path checker. Candidate screening would need a specified
  object-pose prediction method, uncertainty treatment, or counted simulator
  prefix previews. The information and compute must match across compared methods.

### Gaps

- We have not certified full-body passage through any proposed opening or chosen
  robust dimensions. Those require simulator engineering, charged to the budget.

## What establishes a meaningful safe alternative?

### Takeaway

A pusher waypoint path is insufficient. Preserve the manipulation goal and obtain
an executed safe pushing trajectory, including approach, contact changes and final
orientation, before treating a layout as evidence that safety can be improved.

### Cited Findings

- Lynch and Mason's 1996 *Stable Pushing: Mechanics, Controllability, and Planning*
  treats pushing as motion under velocity and contact constraints, and presents
  planning among obstacles. Its point/line-contact results explain why object
  reachability and pusher motion must be considered together. This is mechanics
  evidence, not validation of our learned model or proposed task sizes.
  [Publisher abstract](https://journals.sagepub.com/doi/10.1177/027836499601500602),
  [author institution record](https://publications.ri.cmu.edu/stable-pushing-mechanics-controllability-and-planning).
- Collision checking must validate motion between states. Sampling too sparsely
  can miss a thin obstacle; LaValle describes clearance and displacement bounds
  for segment validation, including rotation.
  [Section 5.3.4](https://msl.cs.illinois.edu/~lavalle/planning/node217.html).

### Inferences

- **Proposed construction:** choose a scene, obtain a successful reference push,
  place a keep-out region on a task-relevant part of that trajectory, then verify
  a different successful pushing route with the same start and goal. Call the
  first trajectory a reference shortcut unless optimality has actually been
  established.
- Monitor the swept pusher disk and moving T polygons, including rotational
  sweep. Physics-substep snapshots improve resolution but are not by themselves
  a continuous-time certificate; specify segment checking or a justified bound.
- A recorded safe witness establishes existence for that simulator state and
  budget. It does not establish reliability over every planner seed. Repeatability
  belongs in evaluation.
- Use a prespecified engineering case-construction procedure, record rejected
  scenes and all interactions, then freeze case IDs, transformations and criteria
  before tested-method outcomes. Do not select final cases because the proposed
  controller succeeds or a comparator fails. Report any reference-based selection
  as an intentionally constructed challenge distribution, not natural prevalence.

### Gaps

- No safe witness was generated in this review. A 2D pathfinder through free
  centre space cannot establish executable contact-rich manipulation.

## Which shapes should we try first?

### Takeaway

Start with one compact convex hazard that creates an understandable detour, then
add a broad edge and a generous doorway. These are proposed designs informed by
the geometry above, not a standard PushT safety protocol.

### Cited Findings

- The distinction between body geometry and free configuration space is established
  by [LaValle section 4.3.2](https://msl.cs.illinois.edu/~lavalle/planning/node161.html).
  The need for executable pushing rather than arbitrary planar translation is
  supported by [Lynch and Mason](https://journals.sagepub.com/doi/10.1177/027836499601500602).

### Inferences

1. **Round island, first diagnostic.** Put a disk in an interior part of the
   reference object's swept route, clear of start and goal footprints. Leave ample
   room on both sides and enough room for pusher recontact. A successful detour is
   visually clear; radial clearance has no corner ambiguity. Protecting both bodies
   makes the object visibly go around the region instead of crossing it while the
   pusher happens to avoid it.
2. **Short rounded rectangle, directional extension.** Replace the disk with a
   finite bar crossing the reference transport route. Its finite ends offer explicit
   alternatives. Vary aspect ratio, position and orientation after setting robust
   sizes on development scenes. A long capsule tests sustained clearance without
   immediately creating a narrow passage or concave trap.
3. **Wide offset gate, later stress test.** Retain the two-pillar idea but choose
   opening width from full-body swept clearance and demonstrated contact access,
   rather than reuse 60 px. This tests transport and alignment through a passage.
   Separate its evidence from the simple-detour family.

- Avoid L/U-shaped traps and near-goal pockets in the first diagnostic: failures
  would combine long-horizon escape, object rotation, recontact and safety. They
  can become a later difficulty family after a lever exists.
- Keep these as virtual semantic keep-out zones for the first frozen-LeWM study.
  Physical obstacles change the dynamics; painting hazards into model observations
  changes its visual inputs. Both are separate interventions requiring their own
  validation. Human replay overlays can show the geometry without changing model
  observations.

### Gaps

- Disk and rounded-bar dimensions, witness action counts and tolerable planner
  horizons remain design parameters, not results. Do not claim that choosing a
  round shape alone makes the problem feasible or fair.
