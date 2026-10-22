# First principles plan

Sachin Mohan, 22 October 2026. Supersedes nothing; this is a staging order for the
existing goal, previously set out in
[Evolutionary improvement on ManiGuard](archive/20261022/Evolutionary%20improvement%20on%20ManiGuard.md).
The research controls in [AGENTS.md](AGENTS.md) still apply at every stage.

## Why this plan exists

Every study so far started at the optimizer. Push-T, Walker, UNISafe, ManiGuard Parts 1
to 4, the six-hypothesis study and the latent-evolution pilot each built a full
comparison around an untested lever on a family whose base rate was never measured.
The latest pilot ended 0/4 in every condition and could not say whether the policy, the
lever, the objective or the budget was at fault
([progress audit](archive/20261022/reports/Recent%20ManiGuard%20progress%20audit.md)). The fix is to
measure the preconditions before buying another comparison.

## What evolution needs, in order

1. **Variance.** The base policy must sometimes complete safely and sometimes not.
2. **A lever.** A knob whose setting changes the safe-completion rate, not merely the
   decoded actions.
3. **Cheap, stable evaluation.** Enough episodes per candidate that fitness is signal.
4. **An optimizer**, compared first against random search and the best hand setting.

Each stage below establishes one requirement and is a measurement, not a bet. A stage
can only inform; the first point at which the plan can fail is stage 4.

## Stage 1: base-rate survey

**Update, 22 October 2026.** The ManiGuard paper already supplies this survey; see
[ManiGuard base rates](docs/research/research_notes/ManiGuard%20base%20rates.md). Jar is
chosen and Stack is the backup. Stage 1 is therefore a reproduction of the published Jar
in-distribution row under the paper's protocol, which also yields the per-task variance
split and the development and held-out task panels. The frozen protocol for that
reproduction is still to be written. The original survey text is kept below for the
record.

Run every ManiGuard family (Jar, Lid, Stack, Clutter, Cabinet, Dusty) with its native
policy and no controller. Twenty episodes per family, fresh seeds, native horizons,
matched benchmark contract. Record safe requested completion, raw goal, violations and
unfinished together, with denominators.

- Budget: 120 episodes. At the historical five minutes per episode that is ten serial
  hours, about five on two GPUs.
- Output: a table of per-family rates with 95% bounds (zero successes in n gives an
  upper bound of about 3/n).
- Gate: choose the family whose safe-completion rate is closest to the middle, roughly
  15% to 70%. Historical hints are thin: Jar identity gave 2/12 in the six-hypothesis
  study and 0/4 in the pilot; Stack gave 1/6. Those are consistent with single-digit
  rates, which is why the survey is needed. If no family exceeds about 10%, stop and
  report that the benchmark offers no selection signal to this approach; the diagnosis
  paper becomes the contribution.

Jar is not privileged. It was chosen before because it was hard. That is the wrong
reason for a first result.

## Stage 2: find a dial

On the chosen family, sweep one scalar lever by hand. The primary candidate is the
native-noise scale, because the explicit-noise hook already exists, scale 1.0 with zero
bias reproduces the native RNG path exactly, and the decoder audits showed influence
grows with magnitude ([pilot report](archive/20261022/docs/research/reports/ManiGuard%20latent%20evolution%20pilot.md)).
Fallback candidates are a gripper-coordinate-only scale and a motion-speed scale.

- Design: five settings, including native, twenty episodes each on a fixed seed panel
  shared across settings. One hundred episodes, about eight serial hours.
- Output: safe completion, violations and unfinished per setting on common seeds.
- Gate: at least one setting changes the safe-completion rate by more than its
  standard error with violations not rising to match. If no setting moves the rate,
  try one fallback lever once, then stop. A lever that cannot move the rate by hand
  cannot be improved by evolution, and that finding is reported, not hidden.

This stage already answers the literal Safety Dial question: does a single knob trade
safety against completion in a frozen manipulation policy?

## Stage 3: size the evaluation

Use the stage 1 and 2 numbers to fix the per-candidate panel. With base rate p and n
episodes the standard error is sqrt(p(1-p)/n); n = 20 at p = 0.3 gives 0.10. Fitness is
the count of safe requested completions on a fixed common-seed panel, with violations
reported alongside, never imagined safety. Freeze the panel size, the seeds, the
parameter domain and the total episode ceiling before any search runs.

No new physical work in this stage. It is arithmetic plus a protocol freeze.

## Stage 4: the evolutionary question

Only now compare optimizers. The genotype is the stage 2 lever plus at most a handful
of conditioning parameters; the shared recurrent controller returns later, not here.

- Arms: evolutionary search, random search, best fixed setting from stage 2, native.
  All search arms receive the same episode budget and see the same seeds.
- Illustrative size: eight candidates per generation, three generations, twenty
  episodes each, about 480 episodes per search arm. Resize from stage 3 numbers.
- Reserve fresh seeds for independent checking before looking at search outcomes.
- Gate for a claim: the evolved setting beats random search and the best fixed setting
  on reserved seeds, with safe completion up and violations not up. One optimizer seed
  supports a pilot claim only; replicate before calling it an advantage.

Transfer across families, ordinary-learning comparators and the observation-conditioned
controller are stage 5 and later. They are earned by a stage 4 positive, not scheduled
ahead of it.

## Rules carried forward

- Count every simulator interaction including sweeps, discarded candidates and failed
  starts. Report elapsed time and model queries separately.
- Report safe completion, physical violations and unfinished episodes together.
- Keep proposed, filtered and executed actions distinguishable. The monitor labels and
  never vetoes.
- Freeze each stage's protocol before running it. Do not read the next stage's
  outcomes while choosing the current stage's design.
- Do not resume the completed pilot or the failed continuation screen.

## Immediate next action

Write the stage 1 protocol: family list, seed panel, horizon contract, render profile,
outcome definitions per family, episode ceiling and stop rule. Then reconstruct the
ManiGuard host and run the survey. Nothing else is authorized until the survey table
exists.
