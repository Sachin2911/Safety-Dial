# Task 6 offline analysis and financial recovery checks

25 October 2026. `experiments/scripts/analyze_dsrl06.py` is frozen before any
development outcome is read. It uses only preserved files and performs no
provider requests, model inference, simulator work or gradient updates.

```bash
.venv/bin/python experiments/scripts/analyze_dsrl06.py \
  --run-dir outputs/dsrl06/remote \
  --stage-a progress/receipts/06-dsrl-stage-a.json \
  --output progress/receipts/06-dsrl-analysis.json
```

The script writes a complete JSON report, a concise Markdown account and, when
training exists, a CSV of returns and outcomes for the fixed six-task cycles.
It checks the frozen task/seed/arm identities, source hashes, native result labels
and discounted-return identity. Invalid recordings remain outside outcome
denominators, with their attempts and resource use retained. Missing accounting
or source evidence cannot produce a passing efficacy screen.

A valid negative decoder-authority result is reported as
`engineering_negative_authority` after the four native engineering episodes have
valid receipts. It is distinct from invalid integration or an incomplete run.
The full screen requires all 24 training jobs, the hashed complete `episode24`
checkpoint, its final actor version and all 48 evaluation task/seed/arm cells.
Each of the eight frozen criteria is reported independently. It never evaluates
an earlier checkpoint as the scheduled final result.

The report includes strict safe completion, the unchanged counted-safe requested
completion, native benchmark classes, raw and counted violations, unfinished
episodes, contact-based engagement, closure, grasp, support departure and lift
while grasped. Counts retain their denominators, with nominal Wilson intervals
and explicit unscored safety. Per-task counts and paired task/seed differences
are retained. Training output separates base rewards, shaping rewards, potential,
cost and corrected discounted returns; it also retains latent distribution and
saturation summaries, learner diagnostics, memory samples and episode timing.

All recorder journals, query SQLite rows and learner update reservations are
reconciled with the protocol ledger, including failed attempts. The two prior
CPU engineering gradients are counted once. Controller sampling calls remain
separate from VLA queries: workstation preflight incurred 64 reference plus 66
CPU worker calls, for 130, and no VLA query. Remote actor request ledgers add to
that count. The raw 66-request worker journal is copied to top-level
`outputs/dsrl06-preflight/actor/cpu-actor-requests.jsonl` for source freezing and
private preservation. No additional calls were made for these analysis checks.

Eight analysis tests pass. They cover every screen gate, missing panel cells,
actual versus expected denominators, unscored safety, a complete negative
authority report from synthetic recording fixtures, changed source hashes,
invalid return identities, imported gradient accounting, final-checkpoint
identity, the full conditional report and fixed-cycle CSV. Twenty-three infra
and manager tests pass, including duplicate-label recovery from both uncertain
and acknowledged creation, charging both hosts, immediate watchdog stop before
the deadline, and one create followed by stop requests for both IDs. The destroy
test fixture now supplies the preservation manifest and local-verification
boundary required by the manager. Ruff and formatting checks pass.

The duplicate-label cost hazard found in the independent review is resolved by
the root implementation and covered by regression tests. The learner and CPU
worker checks are recorded separately in `06-dsrl-learner-build.md`. No rental
was created by this subtask.
