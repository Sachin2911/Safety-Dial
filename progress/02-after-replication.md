# Todo 2: after replication

Goal: find where the failures are, show that the variance is reachable by a noise dial,
fix the task panels, and validate the harness. Starts only after todo 1 passes.

**Status, 23 October 2026.** Todo 1 passed. The episodes of record are the 104 on machine
54514 (dataset runs `jar/seed0_run2`, `jar/seed0_run3`, `jar/seed1`, `jar/seed2`), not 78.
Already answered by todo 1 and the review: the same-seed repeat (13 of 26 tasks agree),
the closure-loophole count among safe successes (1 of 24), and a first task sort (13 tasks
never safe, 12 middle, task 12 always safe). See
[Jar replication and noise floor](../docs/research/research_notes/Jar%20replication%20and%20noise%20floor.md).

- [ ] **Inspect every episode.** For each of the 78: four-class outcome, whether the
      jar was actually closed at goal time, and which safety clause failed
      (left table before closed, dropped, tipped). The closure predicate and clause
      identity live in the LTL sidecar, not `results.jsonl`, so this needs a small reader.
- [ ] **Failure-mode histogram.** Count violations by clause. Mostly left-before-closed
      points at a timing or hold dial; mostly drops points at the gripper.
- [ ] **Closure rule versus benchmark rule.** Count how many published-style safe
      successes fail the closure rule. That is the size of the loophole.
- [ ] **Sort the tasks.** Group all 26 by safe successes over three seeds: 0/3, 1/3,
      2/3, 3/3. The mixed ones (1/3 and 2/3) are where a dial can work.
- [ ] **Renderer-only variance.** Pick one mixed task, repeat it five times with the
      same seed (`REPEAT=5`). Variation here is the renderer, not the policy's sampling
      noise. The remainder of the within-task variance is what a noise dial can reach.
- [ ] **Development panel.** Choose five or six mixed tasks. Run ten seeds each to get
      real per-task rates. Target tasks near 30 to 50% safe success.
- [ ] **Held-out panel.** Choose a disjoint set of tasks now, write them down, and do
      not run them again until the final comparison.
- [ ] **Validate the noise harness.** Feed saved observations through the official
      policy server and through the noise hook at zero bias and unit scale. Action
      chunks must be identical. Only then may the harness run physical episodes.
- [ ] **Write the stage 2 protocol.** Dial, settings, panel, seeds, episode ceiling and
      gate, frozen before the sweep runs.

Budget: about 70 episodes, roughly 6 serial hours, plus the offline harness check.
