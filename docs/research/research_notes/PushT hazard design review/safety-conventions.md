# Safety benchmark conventions for PushT hazards

## Which shapes and safety predicates are conventional?

### Takeaway

Simple geometric forbidden regions are well established, but physical obstacles and virtual keep-out zones encode different problems. A centre-distance safety predicate is a benchmark convention, not a guarantee that a manipulated object's footprint stays safe.

### Cited Findings

- Safety Gym, introduced by Ray, Achiam and Amodei (2019), separates task reward from auxiliary safety costs and includes Goal, Button and Push tasks. Its intended comparison covers final task performance, final constraint satisfaction and safety cost throughout training. Source locations: introduction, pp. 2-3. [Paper](https://cdn.openai.com/safexp-short.pdf)
- Safety-Gymnasium, Ji et al. (NeurIPS 2023), supports vector and vision inputs across single-agent and multi-agent safety tasks. [Paper](https://arxiv.org/abs/2310.12567)
- In Safety-Gymnasium, hazards are circular ground regions implemented as shallow cylinders with collision disabled. Their cost uses distance from the agent's XY position to the hazard centre. The code does not inspect the pushed box's footprint in this hazard predicate. Source locations: `Hazards.get_config`, lines 43-59; `Hazards.cal_cost`, lines 73-83. [Official hazard implementation](https://raw.githubusercontent.com/PKU-Alignment/safety-gymnasium/main/safety_gymnasium/assets/geoms/hazards.py)
- The documentation explicitly distinguishes noncolliding hazards from solid cylindrical pillars, where contact incurs cost. It also provides noncolliding boundary walls. Source locations: Pillars, Hazards and Sigwalls sections. [Official geometry documentation](https://safety-gymnasium.readthedocs.io/en/stable/components_of_environments/objects/geom.html)
- Push1 includes two hazards and one pillar, with the pillar unconstrained; Push2 includes four hazards and four contact-constrained pillars. The object must be pushed to its goal. Source locations: level table and Costs sections. [Official Push documentation](https://safety-gymnasium.readthedocs.io/en/stable/environments/safe_navigation/push.html)

### Inferences

- A single disk or short rounded rectangle is a defensible first virtual hazard for our setup. A finite rectangle would retain compatibility with our existing region checker. These are proposed choices, not published PushT standards.
- Keeping hazards virtual preserves existing contact dynamics. Adding solid pillars would change both dynamics and rendered inputs, creating an additional transfer problem for frozen LeWM.
- Name the scope explicitly: pusher-centre avoidance, pusher-body avoidance, object-footprint avoidance, or both bodies. Our current centre-only check can support a pusher-safety claim, but cannot establish object safety.

### Gaps

- These sources do not establish one universal hazard geometry or protected-body convention for PushT. Safety Gym Push is a mobile-robot box-pushing task, not the T-shaped tabletop task.

## How is safe manipulation separated from doing nothing?

### Takeaway

Report task completion and safety independently, then intersect them. Engagement should expose inactivity rather than become a substitute for completion.

### Cited Findings

- ManiGuard evaluates frozen scenes with separate success and safety checkers; unsafe success is a distinct outcome. [Evaluation overview](https://nu-ideas-lab.github.io/ManiGuard/docs/evaluation/)
- Its monitor evaluates physics-grounded propositions each step and records the first violation. Supported checks include touching, dropped, upright and an overhead-forbidden AABB. Source locations: proposition evaluation and custom checks. [LTL safety system](https://nu-ideas-lab.github.io/ManiGuard/docs/foundations/ltl_safety/)
- Its clutter-pickup task asks the robot to move a target while avoiding dropping or toppling other objects. This illustrates manipulation-specific safety involving affected objects rather than only end-effector position. [Clutter pickup](https://nu-ideas-lab.github.io/ManiGuard/docs/pipelines/clutter_pickup/)
- ManiGuard logs contact, target displacement and proximity. Its engagement ladder separates idle, reached, manipulated and success; raw success and violation rates remain available beside contact-gated diagnostics. Source locations: Per-rollout engagement signals, outcome ladder and Reporting. [Engagement documentation](https://nu-ideas-lab.github.io/ManiGuard/docs/evaluation/engagement_metric/)

### Inferences

- For PushT, log pusher-T contact, T displacement and pose progress. Mere pusher travel is too weak an engagement signal to establish useful manipulation.
- Retain raw region incursions throughout every rollout. Do not copy ManiGuard's contact gate into our primary planar hazard rule: a pusher crossing a forbidden zone before contacting the T can still be unsafe.

### Gaps

- ManiGuard's specific predicates and numerical engagement thresholds belong to its 3D household tasks. They do not supply suitable PushT pixel thresholds.

## What should placement and feasibility checks establish?

### Takeaway

Randomized layouts help avoid memorized paths. They do not by themselves show that a particular safe pushing solution exists within our interaction budget.

### Cited Findings

- Safety Gym randomizes layouts at episode starts and deliberately uses unconstrained baselines to test whether task and safety objectives have a meaningful conflict. Source locations: introduction, pp. 2-3. [Paper](https://cdn.openai.com/safexp-short.pdf)
- ManiGuard releases monitor-verified safe-success demonstrations, using the same monitor for data annotation and evaluation. [Author project page](https://nu-ideas-lab.github.io/ManiGuard/)

### Inferences

- Our stronger admission rule should require a successful, dense-monitor-checked safe pushing witness at the largest intended margin. Freeze it before comparing methods. Count witness-search interactions and failed starts.
- Place a finite hazard across a development reference pushing route, with clear initial and final body footprints and enough surrounding room to go around. Freeze placement rules before test outcomes. Calling a route direct or unconstrained does not establish optimality.
- Include a no-hazard control, mirrored or rotated layouts, and a small prespecified range of hazard sizes. Do not keep only layouts where our preferred method wins.

### Gaps

- No source inspected here establishes that centre-clear endpoints or random nonoverlapping placement certifies a dynamically feasible, budget-feasible nonprehensile pushing route. Witness validation is our proposed design control, not a universal benchmark requirement.
