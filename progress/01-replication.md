# Todo 1: replicate the published Jar baseline

Goal: show that the local stack reproduces the ManiGuard paper's Jar in-distribution
result for the pi0.5 fine-tuned checkpoint before anything is built on it. Published
target over 26 tasks and 3 seeds (78 episodes): 21 safe success, 4 unsafe success,
27 safe unsuccess, 26 unsafe unsuccess, engagement about 99%. See
[ManiGuard base rates](../docs/research/research_notes/ManiGuard%20base%20rates.md).

- [ ] **Pin the sources.** ManiGuard commit from around 20 August 2026 (when the
      checkpoints were last updated), the Jar checkpoint
      `IDEAS-Lab-Northwestern/pi05-base-datagen-v1-jar-joint-2cam-lora` by revision
      hash, and the OpenPI revision it needs. Record all three.
- [ ] **Match the paper's rendering.** Set `source_profile: legacy` and
      `camera_resolution: 256` in the Jar eval YAML. Confirm the policy server receives
      256 by 256 left-overview and wrist images on the first frame.
- [ ] **Confirm the rest of the protocol.** Horizon 2,000, 8 executed actions per
      16-step chunk, goal held for 10 steps, left overview plus wrist cameras. Record the
      GPU and driver version.
- [ ] **Smoke test.** One episode through the official runner. Check `results.jsonl`
      has `success`, `ltl_violated`, `counted_violation`, `safety_evaluated`,
      `first_contact_step` and a non-null `status`.
- [ ] **Seed one: all 26 Jar tasks.** `LEVELS=base bash scripts/eval_family.sh
      jar_transport` with one `SEED`. Expect about 7 safe successes and 10 violations.
      Stop rule: 0 or 1 safe successes means stop and debug the setup.
- [ ] **Seeds two and three.** Two more distinct `SEED` values, 78 episodes total.
- [ ] **Compare to the paper.** Fisher exact test on safe success and on violations
      against the published counts, two-sided, 0.05. Report engagement alongside.
- [ ] **Decide.** Matches: the stack is trusted, move to todo 2. Does not match: find
      the cause before any other work.

Budget: about 80 episodes, roughly 7 serial hours at 5 minutes per episode.
