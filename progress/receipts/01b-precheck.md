# Todo 1b, item 4: check of instance B's seed 0 before launching the lanes

Checked 2026-10-22 22:10 UTC after `git pull --rebase` (HEAD `ade5919`).

- Instance B's last pushes are the item 3 and 4 review (`66c9f77`), the todo 1b
  checklist (`6a97497`, `77dc98e`) and the seed reassignment (`ed2c8bf`, 21:00 UTC). Item 5
  of `01-replication.md` is unticked; no `receipts/01-runs/seed0/`, no `01-seed1-stop.md`
  and no seed 0 `results.jsonl` exist on `origin/main`. B's checklist pushes seed 0
  only when item 5 completes, so no interim rows are expected.
- Stop rule (six or more completed episodes with zero safe successes): **not met on the
  available evidence**, because no seed 0 episode has been pushed. Nothing indicates a
  failure either. The user was not available to ask (autonomous run), so the lanes go
  ahead as the checklist's "otherwise go on" branch.
- Local evidence that the stack produces safe successes: the lane 1 smoke episode
  (task_0000, seed 2) was a safe success at step 982 ([01b-smoke.md](01b-smoke.md)).
  The lane 0 smoke (seed 1) was an unsafe unsuccess, as was B's seed 0 smoke.
- During the lane runs this instance pulls `origin/main` between checks and re-reads any
  seed 0 rows B pushes; if they show six or more completed episodes with zero safe
  successes the finding is recorded in `01b-results.md` for review, since the per-episode
  evaluator runs are left to finish.
